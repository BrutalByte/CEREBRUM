"""
Beam-search attention traversal — the forward pass of CEREBRUM (Section 5.1).

BeamTraversal walks the graph hop-by-hop from seed entities, scoring each
candidate next-hop using CSA attention weights and pruning to beam_width at
each step. Returns all explored paths for downstream ranking.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import heapq
import math

import numpy as np

from core.graph_adapter import GraphAdapter
from core.attention_engine import CSAEngine
from core.bridge_engine import BridgeTwinEngine, BRIDGE_RELATION
from core.resource_governor import ResourceGovernor
from reasoning.path_scorer import community_coherence


# ---------------------------------------------------------------------------
# CVT passthrough helpers
# ---------------------------------------------------------------------------

#: Multiplicative penalty applied when traversing through a CVT mediator hop.
#: 0.85 discounts the path score slightly to reflect the extra indirection,
#: while still allowing high-quality paths through CVT nodes to rank well.
CVT_HOP_PENALTY: float = 0.85


def _is_cvt_node(node_id: str) -> bool:
    """
    Return True if node_id is a Freebase CVT / MID mediator node.

    CVT (Compound Value Type) nodes have opaque MID identifiers of the form
    ``/m/xxxxx`` or ``/g/xxxxx``.  They carry no semantic label and therefore
    produce near-zero cosine-similarity scores when used as attention targets.
    Identifying them lets BeamTraversal collapse A→CVT→B into a single hop
    scored on the A↔B semantic similarity instead.
    """
    return isinstance(node_id, str) and (
        node_id.startswith("/m/") or node_id.startswith("/g/")
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_valid_at(edge, query_time: Optional[float]) -> bool:
    """
    Return True if the edge is temporally active at query_time.

    Rules:
      - query_time is None  → no temporal filter, always valid
      - valid_from is None  → edge has no start constraint
      - valid_to   is None  → edge has no end constraint

    An edge is active when query_time falls within [valid_from, valid_to].
    """
    if query_time is None:
        return True
    vf = getattr(edge, "valid_from", None)
    vt = getattr(edge, "valid_to", None)
    if vf is not None and query_time < vf:
        return False
    if vt is not None and query_time > vt:
        return False
    return True


@dataclass
class TraversalPath:
    """A single reasoning path through the knowledge graph."""

    nodes: List[str] = field(default_factory=list)
    """
    Alternating sequence: [entity, relation, entity, relation, entity, ...].
    Entity nodes are at even indices; relation labels at odd indices.
    """

    seen_entities: Set[str] = field(default_factory=set)
    """Set of all entity IDs in the path (for cycle detection)."""

    embedding: Optional[np.ndarray] = None
    """Current aggregated embedding (updated at each hop)."""

    score: float = 1.0
    """Running product-of-weights path score."""

    attention_weights: List[float] = field(default_factory=list)
    """Attention weight assigned at each hop."""

    community_sequence: List[int] = field(default_factory=list)
    """Community ID of each entity node visited."""

    edge_confidences: List[float] = field(default_factory=list)
    """Per-edge confidence values along the path (from Edge.confidence)."""

    edge_provenances: List[str] = field(default_factory=list)
    """Per-edge provenance strings along the path (from Edge.provenance)."""

    edge_features: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    """Raw feature components for parameter learning: (sim, cs, etw, nd, hd)."""

    beta_alpha: float = 1.0
    """Beta distribution α — accumulated effective successes (sum of edge weights)."""

    beta_beta: float = 1.0
    """Beta distribution β — accumulated effective failures (sum of 1-weight per edge)."""

    @property
    def posterior_mean(self) -> float:
        """E[Beta(α, β)] = α / (α + β). Equals 0.5 at initialization."""
        return self.beta_alpha / (self.beta_alpha + self.beta_beta)

    @property
    def score_variance(self) -> float:
        """Var[Beta(α, β)] = αβ / ((α+β)² * (α+β+1))."""
        a, b = self.beta_alpha, self.beta_beta
        s = a + b
        return (a * b) / (s * s * (s + 1))

    def sample_score(self, rng: "np.random.Generator") -> float:
        """Thompson sample from Beta(α, β). Used in probabilistic beam selection."""
        return float(rng.beta(self.beta_alpha, self.beta_beta))

    def __post_init__(self):
        # Confirm seen_entities is populated if nodes is provided
        if not self.seen_entities and self.nodes:
            for i in range(0, len(self.nodes), 2):
                self.seen_entities.add(self.nodes[i])

    @property
    def head(self) -> str:
        """First entity in the path (the seed)."""
        return self.nodes[0] if self.nodes else ""

    @property
    def tail(self) -> str:
        """Last entity in the path (the current frontier)."""
        return self.nodes[-1] if self.nodes else ""

    @property
    def entity_nodes(self) -> List[str]:
        """All entity nodes in the path (every other element from index 0)."""
        return [self.nodes[i] for i in range(0, len(self.nodes), 2)]

    @property
    def hop_depth(self) -> int:
        return len(self.attention_weights)

    @property
    def path_confidence(self) -> float:
        """
        Minimum edge confidence along the path (weakest-link propagation).
        A path is only as confident as its least-certain edge.
        Returns 1.0 when no confidence data is present.
        """
        if not self.edge_confidences:
            return 1.0
        return min(self.edge_confidences)

    def copy_with_extension(
        self,
        rel: str,
        v: str,
        v_cid: int,
        v_emb: np.ndarray,
        weight: float,
        coherence_score: float,
        edge_confidence: float = 1.0,
        edge_provenance: str = "",
        prior_scale: float = 1.0,
        features: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> "TraversalPath":
        """
        Create a new path by extending this one with an edge.
        Aggregates embedding: ReLU(w * v_emb + h) + h (residual) followed by LayerNorm.
        """
        # 1. New sequence
        new_nodes = self.nodes + [rel, v]
        new_seen  = self.seen_entities | {v}
        new_cseq  = self.community_sequence + [v_cid]
        
        # 2. Embedding aggregation (STEP 3)
        # ReLU(w * v_emb + h) + h
        h_agg = np.maximum(0.0, weight * v_emb + self.embedding) + self.embedding
        
        # 3. LayerNorm (approximated via L2-norm normalization)
        norm = float(np.linalg.norm(h_agg))
        if norm > 0:
            h_agg = h_agg / norm
            
        new_features = self.edge_features + ([features] if features else [])
            
        return TraversalPath(
            nodes=new_nodes,
            seen_entities=new_seen,
            embedding=h_agg,
            score=self.score * weight * coherence_score,
            attention_weights=self.attention_weights + [weight],
            community_sequence=new_cseq,
            edge_confidences=self.edge_confidences + [edge_confidence],
            edge_provenances=self.edge_provenances + [edge_provenance],
            edge_features=new_features,
            beta_alpha=self.beta_alpha + weight * prior_scale,
            beta_beta=self.beta_beta + (1.0 - weight) * prior_scale,
        )

    def __repr__(self) -> str:
        chain = " -> ".join(
            f"[{self.nodes[i]}]" if i % 2 == 0 else self.nodes[i]
            for i in range(len(self.nodes))
        )
        return f"TraversalPath(score={self.score:.4f}, path={chain})"

    def to_dict(self) -> Dict:
        """Serialize for federated reasoning (API transmission)."""
        return {
            "nodes": self.nodes,
            "score": self.score,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "attention_weights": self.attention_weights,
            "community_sequence": self.community_sequence,
            "edge_confidences": self.edge_confidences,
            "edge_provenances": self.edge_provenances,
            "edge_features": [list(f) for f in self.edge_features],
            "beta_alpha": self.beta_alpha,
            "beta_beta": self.beta_beta,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TraversalPath":
        """Deserialize from federated reasoning (API reception)."""
        emb = np.array(d["embedding"], dtype=np.float32) if d.get("embedding") else None
        return cls(
            nodes=d["nodes"],
            score=d["score"],
            embedding=emb,
            attention_weights=d.get("attention_weights", []),
            community_sequence=d.get("community_sequence", []),
            edge_confidences=d.get("edge_confidences", []),
            edge_provenances=d.get("edge_provenances", []),
            edge_features=[tuple(f) for f in d.get("edge_features", [])],
            beta_alpha=d.get("beta_alpha", 1.0),
            beta_beta=d.get("beta_beta", 1.0),
        )


class BeamTraversal:
    """
    Beam-search traversal using CSA attention weights.

    Algorithm (Section 5.1 STEP 3):
      - Initialize beam from seed entities
      - For each hop 1..max_hop:
          - For each path in beam, expand to all neighbors via adapter
          - Score each candidate extension with CSA.compute_weight
          - Aggregate embedding: ReLU(w * v_emb + h) + residual + LayerNorm
          - Prune candidates to beam_width by path score
      - Return all paths explored (caller selects top-K answers)
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        csa_engine: CSAEngine,
        beam_width: int = 10,
        max_hop: int = 3,
        max_neighbors: int = 100,
        max_budget: int = 10000,
        edge_type_weights: Optional[Dict[str, float]] = None,
        governor: Optional[ResourceGovernor] = None,
        probabilistic: bool = False,
        seed: int = 42,
        warm_start_strength: float = 0.0,
        beam_widths: Optional[Dict[int, int]] = None,
        cvt_passthrough: bool = False,
    ):
        self.adapter            = adapter
        self.csa                = csa_engine
        self.beam_width         = beam_width
        self.max_hop            = max_hop
        self.max_neighbors      = max_neighbors
        self.max_budget         = max_budget
        self.edge_type_weights  = {BRIDGE_RELATION: 1.0, **(edge_type_weights or {})}
        self.expansions         = 0
        self.governor           = governor or ResourceGovernor()
        self.bridge_engine: Optional[BridgeTwinEngine] = None
        self.insight_engine = None  # Optional[InsightEngine] — set after construction
        self.probabilistic      = probabilistic
        self._rng               = np.random.default_rng(seed)
        self.warm_start_strength = warm_start_strength
        # Per-hop beam width overrides: {hop_number: width}.
        # Allows wider beams at deeper intermediate hops without changing the
        # base beam_width used for the first hop.  Only applied to intermediate
        # hops (hop < max_hop); the terminal hop is never pruned.
        self._beam_widths: Dict[int, int] = beam_widths or {}
        # When True, traversal collapses Freebase CVT mediator nodes (opaque
        # /m/ or /g/ MIDs) into transparent relay hops: A→CVT→B is treated as
        # a single hop from A to B, scored on A↔B semantic similarity.
        self.cvt_passthrough: bool = cvt_passthrough

    def traverse(
        self,
        seeds: List[str],
        query_time: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
        community_merger=None,
    ) -> List[TraversalPath]:
        """
        Run beam traversal from the given seed entity IDs.

        Parameters
        ----------
        seeds           : list of entity IDs to start from
        query_time      : Unix timestamp for temporal filtering.
        query_embedding : Optional[np.ndarray] of the question text.
        community_merger: Optional[QueryGuidedCommunityMerger] instance.
        """
        emb_dim = self._infer_dim()
        self.expansions = 0

        # Hole 1 — Mid-Flight Community Swap: snapshot the community map once
        # at query start so all CSA computations for this query use a consistent
        # partition, even if GlobalRebalancer commits a new map mid-traversal.
        _cmap = getattr(self.adapter, "community_map", None)
        
        # Phase 29: Context-Aware Community Merging
        if community_merger is not None and query_embedding is not None and _cmap is not None:
            _cmap = community_merger.merge(_cmap, query_embedding, self.adapter)

        if self.csa is not None and _cmap is not None:
            self.csa.set_query_snapshot(dict(_cmap))
        try:
            return self._traverse_inner(seeds, emb_dim, query_time)
        finally:
            if self.csa is not None:
                self.csa.clear_query_snapshot()

    def _traverse_inner(
        self,
        seeds: List[str],
        emb_dim: int,
        query_time: Optional[float],
    ) -> "List[TraversalPath]":
        # Initialize beam from seed entities
        beam: List[TraversalPath] = []
        for seed in seeds:
            emb = self.adapter.get_embedding(seed)
            if emb is None:
                emb = np.zeros(emb_dim, dtype=np.float32)

            # STEP 2 LayerNorm (initial)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            cid = self.adapter.get_community(seed)
            beam.append(
                TraversalPath(
                    nodes=[seed],
                    seen_entities={seed},
                    embedding=emb.copy(),
                    score=1.0,
                    community_sequence=[cid] if cid >= 0 else [],
                )
            )

        all_paths: List[TraversalPath] = list(beam)

        for hop in range(1, self.max_hop + 1):
            candidates: List[TraversalPath] = []

            for path in beam:
                # Security & Resource check: Computational Budget & RAM pressure
                if not self.governor.can_expand(self.expansions, self.max_budget):
                    break

                edges = self.adapter.get_neighbors(
                    path.tail,
                    max_neighbors=self.max_neighbors,
                    context_embedding=path.embedding,
                )
                self.expansions += 1

                # Hoist source embedding fetch: same for all edges from this path
                eu = self.adapter.get_embedding(path.tail)

                for edge in edges:
                    v = edge.target_id

                    # Temporal validity filter
                    if not is_valid_at(edge, query_time):
                        continue

                    # Avoid revisiting entities already in the path (O(1) now)
                    if v in path.seen_entities:
                        continue

                    # ----------------------------------------------------------
                    # CVT passthrough: collapse opaque Freebase mediator nodes.
                    # When v is a CVT/MID node (/m/xxx, /g/xxx) we expand one
                    # level further and treat each A→CVT→B pair as a single
                    # hop from A to B, preserving the combined relation label.
                    # A small CVT_HOP_PENALTY is applied via edge_confidence.
                    # ----------------------------------------------------------
                    if self.cvt_passthrough and _is_cvt_node(v):
                        cvt_edges = self.adapter.get_neighbors(
                            v, max_neighbors=self.max_neighbors
                        )
                        next_steps: List[Tuple[str, str, float, str]] = []
                        for ce in cvt_edges:
                            vv = ce.target_id
                            if vv not in path.seen_entities and not _is_cvt_node(vv):
                                next_steps.append((
                                    vv,
                                    f"{edge.relation_type}|{ce.relation_type}",
                                    min(edge.confidence, ce.confidence) * CVT_HOP_PENALTY,
                                    ce.provenance,
                                ))
                    else:
                        next_steps = [
                            (v, edge.relation_type, edge.confidence, edge.provenance)
                        ]

                    for (v_eff, rel_eff, conf_eff, prov_eff) in next_steps:
                        if v_eff in path.seen_entities:
                            continue

                        ev = self.adapter.get_embedding(v_eff)

                        # Single call: weight + all feature components (community_score
                        # is memoized per query, embeddings pre-fetched above).
                        # Falls back to compute_weight() when compute_weight_with_features
                        # is unavailable or returns a non-tuple (e.g. mocks, old subclasses).
                        _cwf = getattr(self.csa, 'compute_weight_with_features', None)
                        _cwf_result = _cwf(
                            path.tail, v_eff, hop=hop,
                            edge_type=rel_eff,
                            edge_type_weights=self.edge_type_weights,
                            normalized_distance=0.0,
                            eu=eu, ev=ev,
                        ) if _cwf is not None else None
                        if isinstance(_cwf_result, tuple) and len(_cwf_result) == 6:
                            w, sim, cs, etw, nd, hd = _cwf_result
                        else:
                            w = float(self.csa.compute_weight(
                                path.tail, v_eff, hop=hop,
                                edge_type=rel_eff,
                                edge_type_weights=self.edge_type_weights,
                                normalized_distance=0.0,
                            ))
                            sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0
                            cs  = float(self.csa.community_score(path.tail, v_eff)) if hasattr(self.csa, 'community_score') else 0.5
                            etw = self.edge_type_weights.get(rel_eff, 0.0)
                            nd  = 0.0
                            hd  = 1.0 / (1.0 + hop)

                        # Path metadata and extension
                        v_emb = ev if ev is not None else np.zeros(emb_dim, dtype=np.float32)
                        v_cid = self.adapter.get_community(v_eff)

                        # Bridge twin detection: record cross-community crossings
                        # and let the engine create a structural relay if warranted.
                        u_cid = self.adapter.get_community(path.tail)
                        if (
                            self.bridge_engine is not None
                            and u_cid != v_cid
                            and u_cid >= 0
                            and v_cid >= 0
                            and rel_eff != BRIDGE_RELATION
                        ):
                            self.bridge_engine.record_crossing(
                                node_id=v_eff,
                                source_community=v_cid,
                                dest_community=u_cid,
                                adapter=self.adapter,
                            )

                        # Mark bridge twin use so its idle timer resets
                        if (
                            self.bridge_engine is not None
                            and rel_eff == BRIDGE_RELATION
                        ):
                            self.bridge_engine.record_twin_use(v_eff)

                        # Hot path: record cross-community crossing for InsightEngine
                        if (
                            self.insight_engine is not None
                            and u_cid != v_cid
                            and u_cid >= 0
                            and v_cid >= 0
                        ):
                            self.insight_engine.record_crossing(
                                u=path.tail,
                                v=v_eff,
                                u_cid=u_cid,
                                v_cid=v_cid,
                                path_score=path.score,
                                path=path,
                            )

                        # Compute community coherence for the candidate step
                        coh = community_coherence(path.community_sequence + [v_cid])

                        # Warm-start: amplify the first-hop beta update to reduce cold-start variance
                        _prior_scale = 1.0
                        if (
                            self.probabilistic
                            and self.warm_start_strength > 0.0
                            and len(path.nodes) == 1
                        ):
                            _prior_scale = 1.0 + self.warm_start_strength

                        new_path = path.copy_with_extension(
                            rel=rel_eff,
                            v=v_eff,
                            v_cid=v_cid,
                            v_emb=v_emb,
                            weight=w,
                            coherence_score=coh,
                            edge_confidence=conf_eff,
                            edge_provenance=prov_eff,
                            prior_scale=_prior_scale,
                            features=(sim, cs, etw, nd, hd),
                        )
                        candidates.append(new_path)

            if not candidates:
                break

            # At the terminal hop, skip pruning — all reachable endpoints are kept
            # for the answer extractor to score and deduplicate. Pruning here discards
            # valid answers with zero benefit (no further expansion occurs).
            hop_bw = self._beam_widths.get(hop, self.beam_width)
            if self.probabilistic:
                _rng = self._rng
                if hop < self.max_hop and len(candidates) > hop_bw:
                    beam = heapq.nlargest(
                        hop_bw, candidates,
                        key=lambda p: p.sample_score(_rng),
                    )
                else:
                    beam = sorted(candidates,
                                  key=lambda p: p.sample_score(_rng),
                                  reverse=True)
            else:
                if hop < self.max_hop and len(candidates) > hop_bw:
                    beam = heapq.nlargest(hop_bw, candidates, key=lambda p: p.score)
                else:
                    beam = sorted(candidates, key=lambda p: p.score, reverse=True)

            all_paths.extend(beam)

        return all_paths

    def _infer_dim(self) -> int:
        """Infer embedding dimension from the adapter."""
        # Use a sample entity if possible
        try:
            # Try some common entities or just pick one
            for seed in self.adapter.find_entities("", top_k=1):
                emb = self.adapter.get_embedding(seed.id)
                if emb is not None:
                    return len(emb)
        except Exception:
            pass
        return 384  # Default for sentence-transformers

class AsyncBeamTraversal(BeamTraversal):
    """
    Asynchronous version of BeamTraversal that yields paths as they are found.
    Allows for real-time reasoning visualization and lower TTFT (Time To First Trace).
    """

    async def traverse_stream(
        self,
        seeds: List[str],
        query_time: Optional[float] = None,
    ):
        """
        Run beam traversal and yield paths hop-by-hop.
        Yields: List[TraversalPath] at each hop completion.

        query_time : Unix timestamp for temporal filtering (see traverse()).
        """
        emb_dim = self._infer_dim()
        self.expansions = 0

        # Hole 1 — Mid-Flight Community Swap: snapshot at stream start.
        _cmap = getattr(self.adapter, "community_map", None)
        if self.csa is not None and _cmap is not None:
            self.csa.set_query_snapshot(dict(_cmap))
        try:
            async for hop_result in self._traverse_stream_inner(seeds, emb_dim, query_time):
                yield hop_result
        finally:
            if self.csa is not None:
                self.csa.clear_query_snapshot()

    async def _traverse_stream_inner(self, seeds, emb_dim, query_time):
        import asyncio
        # Initialize beam from seed entities
        beam: List[TraversalPath] = []
        for seed in seeds:
            emb = self.adapter.get_embedding(seed)
            if emb is None:
                emb = np.zeros(emb_dim, dtype=np.float32)

            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            cid = self.adapter.get_community(seed)
            path = TraversalPath(
                nodes=[seed],
                seen_entities={seed},
                embedding=emb.copy(),
                score=1.0,
                community_sequence=[cid] if cid >= 0 else [],
            )
            beam.append(path)

        # Yield depth-0 (seeds)
        yield beam

        for hop in range(1, self.max_hop + 1):
            candidates: List[TraversalPath] = []

            for path in beam:
                if not self.governor.can_expand(self.expansions, self.max_budget):
                    break

                # Non-blocking neighbor fetch (if adapter supports it,
                # otherwise runs in thread pool or just regular call)
                edges = self.adapter.get_neighbors(
                    path.tail,
                    max_neighbors=self.max_neighbors,
                    context_embedding=path.embedding,
                )
                self.expansions += 1

                # Small yield to prevent event loop starvation on huge graphs
                if self.expansions % 100 == 0:
                    await asyncio.sleep(0)

                # Hoist source embedding fetch: same for all edges from this path
                eu = self.adapter.get_embedding(path.tail)

                for edge in edges:
                    v = edge.target_id

                    # Temporal validity filter
                    if not is_valid_at(edge, query_time):
                        continue

                    if v in path.seen_entities:
                        continue

                    # CVT passthrough (same logic as sync traversal)
                    if self.cvt_passthrough and _is_cvt_node(v):
                        cvt_edges = self.adapter.get_neighbors(
                            v, max_neighbors=self.max_neighbors
                        )
                        next_steps: List[Tuple[str, str, float, str]] = []
                        for ce in cvt_edges:
                            vv = ce.target_id
                            if vv not in path.seen_entities and not _is_cvt_node(vv):
                                next_steps.append((
                                    vv,
                                    f"{edge.relation_type}|{ce.relation_type}",
                                    min(edge.confidence, ce.confidence) * CVT_HOP_PENALTY,
                                    ce.provenance,
                                ))
                    else:
                        next_steps = [
                            (v, edge.relation_type, edge.confidence, edge.provenance)
                        ]

                    for (v_eff, rel_eff, conf_eff, prov_eff) in next_steps:
                        if v_eff in path.seen_entities:
                            continue

                        ev = self.adapter.get_embedding(v_eff)

                        # Single call: weight + all feature components (with fallback)
                        _cwf = getattr(self.csa, 'compute_weight_with_features', None)
                        _cwf_result = _cwf(
                            path.tail, v_eff, hop=hop,
                            edge_type=rel_eff,
                            edge_type_weights=self.edge_type_weights,
                            normalized_distance=0.0,
                            eu=eu, ev=ev,
                        ) if _cwf is not None else None
                        if isinstance(_cwf_result, tuple) and len(_cwf_result) == 6:
                            w, sim, cs, etw, nd, hd = _cwf_result
                        else:
                            w = float(self.csa.compute_weight(
                                path.tail, v_eff, hop=hop,
                                edge_type=rel_eff,
                                edge_type_weights=self.edge_type_weights,
                                normalized_distance=0.0,
                            ))
                            sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0
                            cs  = float(self.csa.community_score(path.tail, v_eff)) if hasattr(self.csa, 'community_score') else 0.5
                            etw = self.edge_type_weights.get(rel_eff, 0.0)
                            nd  = 0.0
                            hd  = 1.0 / (1.0 + hop)

                        v_emb  = ev if ev is not None else np.zeros(emb_dim, dtype=np.float32)
                        v_cid  = self.adapter.get_community(v_eff)
                        u_cid_ = self.adapter.get_community(path.tail)

                        # Hot path: async traversal also feeds InsightEngine
                        if (
                            self.insight_engine is not None
                            and u_cid_ != v_cid
                            and u_cid_ >= 0
                            and v_cid >= 0
                        ):
                            self.insight_engine.record_crossing(
                                u=path.tail,
                                v=v_eff,
                                u_cid=u_cid_,
                                v_cid=v_cid,
                                path_score=path.score,
                                path=path,
                            )

                        coh = community_coherence(path.community_sequence + [v_cid])

                        _prior_scale = 1.0
                        if (
                            self.probabilistic
                            and self.warm_start_strength > 0.0
                            and len(path.nodes) == 1
                        ):
                            _prior_scale = 1.0 + self.warm_start_strength

                        new_path = path.copy_with_extension(
                            rel=rel_eff,
                            v=v_eff,
                            v_cid=v_cid,
                            v_emb=v_emb,
                            weight=w,
                            coherence_score=coh,
                            edge_confidence=conf_eff,
                            edge_provenance=prov_eff,
                            prior_scale=_prior_scale,
                            features=(sim, cs, etw, nd, hd),
                        )
                        candidates.append(new_path)

            if not candidates:
                break

            # At the terminal hop, skip pruning (same as sync version)
            hop_bw = self._beam_widths.get(hop, self.beam_width)
            if self.probabilistic:
                _rng = self._rng
                if hop < self.max_hop and len(candidates) > hop_bw:
                    beam = heapq.nlargest(
                        hop_bw, candidates,
                        key=lambda p: p.sample_score(_rng),
                    )
                else:
                    beam = sorted(candidates,
                                  key=lambda p: p.sample_score(_rng),
                                  reverse=True)
            else:
                if hop < self.max_hop and len(candidates) > hop_bw:
                    beam = heapq.nlargest(hop_bw, candidates, key=lambda p: p.score)
                else:
                    beam = sorted(candidates, key=lambda p: p.score, reverse=True)

            # Yield this layer's best paths
            yield beam




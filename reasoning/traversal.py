"""
Beam-search attention traversal - the forward pass of CEREBRUM (Section 5.1.05050).

BeamTraversal walks the graph hop-by-hop from seed entities, scoring each
candidate next-hop using CSA attention weights and pruning to beam_width at
each step. Returns all explored paths for downstream ranking.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, TYPE_CHECKING
import heapq
import logging
import math
import time as _time

import numpy as np

_log = logging.getLogger("cerebrum.traversal")

from core.graph_adapter import GraphAdapter, Edge
from core.attention_engine import CSAEngine
from core.telemetry import NeuralEvent
from core.bridge_engine import BridgeTwinEngine, BRIDGE_RELATION
from core.resource_governor import ResourceGovernor
from core.task_queue import BridgeTask, TaskQueue
from core.reasoning_logit import ReasoningLogit # New: Unified logit framework
from reasoning.path_scorer import community_coherence

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reasoning.trace import ReasoningTrace

# Phase 68: CSA parameter mapping for hormonal overrides
CSA_PARAM_KEYS = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'mu', 'theta']

# Default parameters for CSAEngine (alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)
_DEFAULT_INIT_PARAMS = (0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0)

#: Fixed-point scale for quantized traversal (Phase 61).
#: Scores are stored as uint8 [0, 255] representing [0.0, 1.0].
QUANT_SCALE: int = 255

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
    Identifying them lets BeamTraversal collapse A->CVT->B into a single hop
    scored on the A<->B semantic similarity instead.
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


def is_valid_at(edge, query_time: Optional[float], hard_filter: bool = True) -> bool:
    """
    Return True if the edge is temporally active at query_time.

    Rules:
      - query_time is None  -> no temporal filter, always valid
      - valid_from is None  -> edge has no start constraint
      
    An edge is active when query_time >= valid_from.
    If hard_filter is True, it also checks query_time <= valid_to.
    When hard_filter is False, valid_to is used for decay scoring instead of pruning.
    """
    if query_time is None:
        return True
    vf = getattr(edge, "valid_from", None)
    if vf is not None and query_time < vf:
        return False
        
    if hard_filter:
        vt = getattr(edge, "valid_to", None)
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

    score: float = 0.997
    """Running product-of-weights path score."""

    q_score: int = 255
    """Quantized uint8 score [0, 255] (Phase 61)."""

    quantized: bool = False
    """Whether this path uses quantized uint8 scoring."""

    attention_weights: List[float] = field(default_factory=list)
    """Attention weight assigned at each hop."""

    community_sequence: List[int] = field(default_factory=list)
    """Community ID of each entity node visited."""

    edge_confidences: List[float] = field(default_factory=list)
    """Per-edge confidence values along the path (from Edge.confidence)."""

    edge_provenances: List[str] = field(default_factory=list)
    """Per-edge provenance strings along the path (from Edge.provenance)."""

    edge_features: List[Tuple[float, float, float, float, float, float, float, float, float]] = field(default_factory=list)
    """Raw feature components for parameter learning: (sim, cs, etw, nd, hd, pr, td, nr, grounding)."""

    beta_alpha: float = 1.0
    """Beta distribution alpha - accumulated effective successes (sum of edge weights)."""

    beta_beta: float = 1.0
    """Beta distribution beta - accumulated effective failures (sum of 1-weight per edge)."""

    @property
    def posterior_mean(self) -> float:
        """E[Beta(alpha, beta)] = alpha / (alpha + beta). Equals 0.5 at initialization."""
        return self.beta_alpha / (self.beta_alpha + self.beta_beta)

    @property
    def score_variance(self) -> float:
        """Var[Beta(alpha, beta)] = alphabeta / ((alpha+beta)^2 * (alpha+beta+1))."""
        a, b = self.beta_alpha, self.beta_beta
        s = a + b
        return (a * b) / (s * s * (s + 1))

    def sample_score(self, rng: "np.random.Generator") -> float:
        """Thompson sample from Beta(alpha, beta). Used in probabilistic beam selection."""
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

    def mean_edge_features(self) -> Tuple[float, ...]:
        """Return mean feature vector across all hops; zeros if no features recorded."""
        if not self.edge_features:
            return tuple([0.0] * 10)
        n = len(self.edge_features[0])
        return tuple(sum(f[i] for f in self.edge_features) / len(self.edge_features) for i in range(n))

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
        features: Optional[Tuple[float, float, float, float, float, float, float, float, float]] = None,
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
            
        new_q_score = self.q_score
        if self.quantized:
            # Fixed-point update: q_next = (q_curr * int(w * 255)) >> 8
            # (Approximates float multiplication)
            w_int = int(round(weight * QUANT_SCALE))
            new_q_score = (self.q_score * w_int) // QUANT_SCALE

        return TraversalPath(
            nodes=new_nodes,
            seen_entities=new_seen,
            embedding=h_agg,
            score=self.score * weight * coherence_score,
            q_score=new_q_score,
            quantized=self.quantized,
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

    def _calculate_conflict_entropy(self, candidates: List["TraversalPath"]) -> float:
        """Phase 149: Measure the distribution entropy of candidate scores."""
        if not candidates:
            return 0.0
        scores = np.array([p.score for p in candidates], dtype=np.float32)
        total = np.sum(scores)
        if total == 0:
            return 0.0
        probs = scores / (total + 1e-9)
        return -float(np.sum(probs * np.log2(probs + 1e-9)))



    def __init__(
        self,
        adapter: GraphAdapter,
        csa_engine: CSAEngine,
        graph=None,
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
        symbolic_validator = None,
        calibration_engine = None,
        task_queue: Optional[TaskQueue] = None,
        quantized: bool = False,
        penultimate_decay: float = 0.0,
        **kwargs,
    ):
        self.graph              = graph
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
        self.insight_engine = None
        self.probabilistic      = probabilistic
        self._rng               = np.random.default_rng(seed)
        self.warm_start_strength = warm_start_strength
        self._beam_widths: Dict[int, int] = beam_widths or {}
        self.cvt_passthrough: bool = cvt_passthrough
        self.symbolic_validator = symbolic_validator
        self.calibration_engine = calibration_engine
        self.uncertainty_log: List[Dict[str, Any]] = []
        self.task_queue = task_queue
        self._partial_paths: List[TraversalPath] = []
        self.quantized = quantized
        self.epistemic_gaps = []  # Phase 150
        # Phase 151: PenultimateGate — score-gap filter at hop max_hop-1
        self.penultimate_decay: float = penultimate_decay
        # Phase 68: Store hormonal overrides for CSA parameters
        self.csa_overrides = {k: v for k, v in kwargs.items() if k in CSA_PARAM_KEYS}
        # Phase 124: Causal edge index + bonus - set by CerebrumGraph.build()
        self._causal_edge_index: set = set()
        self.causal_bonus: float = float(kwargs.get("causal_bonus", 0.3))
        # Phase 146: Terminal Relation Boost - applied only at the final hop
        self.terminal_relation_boost: Dict[str, float] = dict(kwargs.get("terminal_relation_boost", {}) or {})
        # Phase 156: Penultimate Relation Boost - separate relation(s) to boost at hop N-1
        self.penultimate_relation_boost: Dict[str, float] = dict(kwargs.get("penultimate_relation_boost", {}) or {})
        # Phase 99: Thalamic Gating - WM priming map
        self.node_priming: Dict[str, float] = {}
        self.priming_boost: float = 0.3
        # Phase 100: Lateral Inhibition - winner-take-all community NMS
        self.lateral_inhibition_ratio: float = 0.0
        # Phase 106: Thalamic Gating - context-driven branch killing
        self.thalamic_threshold: float = 0.0
        # Phase 117: Prefrontal Bridge (Symbolic Verification)
        self.symbolic_validator = symbolic_validator
        
        # Phase 108: Thalamofrontal Feedback Loop - dynamic metabolic gating
        self.dynamic_thalamic_gating: bool = False
        self.gating_sensitivity: float = 0.1
        # Phase 101: Emotional Valence - aversive/appetitive edge scoring
        self._valence_engine: Optional[Any] = None


    def traverse(
        self,
        seeds: List[str],
        query_time: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
        community_merger=None,
        trace_info: Optional["ReasoningTrace"] = None,
        node_priming: Optional[Dict[str, float]] = None,
        pruning_callback: Optional[callable] = None,
    ) -> List[TraversalPath]:
        """
        Run beam traversal from the given seed entity IDs.

        Parameters
        ----------
        seeds           : list of entity IDs to start from
        query_time      : Unix timestamp for temporal filtering.
        query_embedding : Optional[np.ndarray] of the question text.
        community_merger: Optional[QueryGuidedCommunityMerger] instance.
        trace_info      : Optional ReasoningTrace to populate (Phase 62).
        node_priming    : Optional per-node attention boost map (Phase 99).
        pruning_callback: Optional func(top_score) -> bool (Phase 139).
        """
        emb_dim = self._infer_dim()
        self.expansions = 0
        _t0 = _time.perf_counter()

        _log.info(
            "Traversal START  seeds=%s  max_hop=%d  beam_width=%d  max_budget=%d",
            seeds, self.max_hop, self.beam_width, self.max_budget,
        )

        # Hole 1 - Mid-Flight Community Swap: snapshot the community map once
        # at query start so all CSA computations for this query use a consistent
        # partition, even if GlobalRebalancer commits a new map mid-traversal.
        _cmap = getattr(self.adapter, "community_map", None)

        # Phase 29: Context-Aware Community Merging
        if community_merger is not None and query_embedding is not None and _cmap is not None:
            _cmap = community_merger.merge(_cmap, query_embedding, self.adapter)

        if self.csa is not None:
            if _cmap is not None:
                self.csa.set_query_snapshot(dict(_cmap))
            self.csa.set_query_time(query_time)
        _active_priming = node_priming if node_priming is not None else self.node_priming
        try:
            # Phase 62: Explainable Reasoning Trace (ERT)
            paths = self._traverse_inner(seeds, emb_dim, query_time,
                                         trace_info=trace_info,
                                         node_priming=_active_priming,
                                         pruning_callback=pruning_callback)
            _dt = (_time.perf_counter() - _t0) * 1000
            _log.info(
                "Traversal END    paths=%d  expansions=%d  elapsed=%.1fms",
                len(paths), self.expansions, _dt,
            )
            return paths
        finally:
            if self.csa is not None:
                self.csa.clear_query_snapshot()
                self.csa.set_query_time(None)

    def _traverse_inner(
        self,
        seeds: List[str],
        emb_dim: int,
        query_time: Optional[float],
        trace_info: Optional["ReasoningTrace"] = None,
        node_priming: Optional[Dict[str, float]] = None,
        pruning_callback: Optional[callable] = None,
    ) -> "List[TraversalPath]":
        # Initialize beam from seed entities
        beam: List[TraversalPath] = []
        _get_emb = self.adapter.get_embedding
        _get_comm = self.adapter.get_community

        for seed in seeds:
            emb = _get_emb(seed)
            if emb is None:
                emb = np.zeros(emb_dim, dtype=np.float32)

            # STEP 2 LayerNorm (initial)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            cid = _get_comm(seed)
            path = TraversalPath(
                nodes=[seed],
                seen_entities={seed},
                embedding=emb.copy(),
                score=1.0,
                q_score=QUANT_SCALE,
                quantized=self.quantized,
                community_sequence=[cid] if cid >= 0 else [],
            )
            beam.append(path)

        # Phase 62: Record hop 0 (seeds) in trace
        if trace_info:
            trace_info.add_hop(hop=0, winners=beam, competitors=[], total_count=len(beam), beam_width=len(beam))

        all_paths: List[TraversalPath] = list(beam)
        _log.debug("Beam initialised with %d seed path(s)", len(beam))

        # Phase 134: per-query embedding/community caches (survive across hops)
        emb_cache: Dict[str, np.ndarray] = {}
        comm_cache: Dict[str, int] = {}
        # Pre-warm with seed embeddings - avoids re-fetch when seeds appear as u
        for _sp in beam:
            emb_cache[_sp.nodes[0]] = _sp.embedding
            if _sp.community_sequence:
                comm_cache[_sp.nodes[0]] = _sp.community_sequence[0]

        # Hoist method lookups out of all loops
        _cwf = getattr(self.csa, "compute_weight_with_features", None)
        _get_params = getattr(self.csa, "get_current_params", None)
        _valence_eng = getattr(self, "_valence_engine", None)
        _get_valence = getattr(_valence_eng, "get_valence", None) if _valence_eng else None

        # GWS Phase 110: Integration for blackboard signaling
        gws = getattr(self, "global_workspace", None)

        # Phase 111: Active Inference Prior Generation
        prior = None
        predictive_coder = getattr(self, "predictive_coder", None)
        if predictive_coder is not None and query_time is not None:
             # We use the seed for prior projection
             prior = predictive_coder.get_prior_for_query(seeds[0])

        for hop in range(1, self.max_hop + 1):
            # Phase 139: Early exit via pruning callback (Cross-Branch Pruning)
            if pruning_callback is not None:
                _top_score = max(p.score for p in beam) if beam else 0.0
                if not pruning_callback(_top_score):
                    _log.debug("Phase 139: Branch pruned by callback at hop %d", hop)
                    break

            candidates: List[TraversalPath] = []

            # Phase 110: GWS Bypass Gate Check
            gws_bypass = False
            if gws is not None and gws.blackboard.qsize() > 50: # Critical threshold
                gws_bypass = True

            for path in beam:
                u = path.tail
                
                # Phase 111: Proactive Bias
                expected_rel = None
                if prior is not None and len(path.nodes) < len(prior.patterns[0]):
                    expected_rel = prior.patterns[0][len(path.nodes)]

                # Hoist source embedding and community fetch (use cache when available)
                if u not in emb_cache:
                    emb_cache[u] = _get_emb(u)
                eu = emb_cache[u]
                if u not in comm_cache:
                    comm_cache[u] = _get_comm(u)
                u_cid = comm_cache[u]
                
                # Hoist current params for this source node
                if _get_params:
                    csa_params_base = _get_params(u)
                else:
                    csa_params_base = _DEFAULT_INIT_PARAMS

                # Apply dynamic hormonal overrides once per path
                if self.csa_overrides:
                    p_list = list(csa_params_base)
                    for i, k in enumerate(CSA_PARAM_KEYS):
                        if k in self.csa_overrides:
                            p_list[i] = self.csa_overrides[k]
                    csa_params = tuple(p_list)
                else:
                    csa_params = csa_params_base

                edges = self.adapter.get_neighbors(
                    u,
                    max_neighbors=self.max_neighbors,
                    context_embedding=path.embedding,
                )
                self.expansions += 1

                # Phase 134: accumulate all effective next-steps for batch scoring
                _batch_steps: List[Tuple[str, str, float, str, Any, Any, float]] = []

                for edge in edges:
                    v = edge.target_id

                    # Temporal validity filter
                    if not is_valid_at(edge, query_time, hard_filter=not self.csa.use_temporal_decay):
                        continue

                    # Avoid revisiting entities already in the path (O(1) now)
                    if v in path.seen_entities:
                        continue

                    # Symbolic Guardrails (Phase 34)
                    if self.symbolic_validator is not None:
                        if not self.symbolic_validator.validate_step(u, edge.relation_type, v, path=path):
                            continue

                    # ----------------------------------------------------------
                    # CVT passthrough logic
                    # ----------------------------------------------------------
                    if self.cvt_passthrough and _is_cvt_node(v):
                        cvt_edges = self.adapter.get_neighbors(
                            v, max_neighbors=self.max_neighbors
                        )
                        for ce in cvt_edges:
                            vv = ce.target_id
                            if vv not in path.seen_entities and not _is_cvt_node(vv):
                                _rel = f"{edge.relation_type}|{ce.relation_type}"
                                _pb = 1.5 if (expected_rel and _rel == expected_rel) else 1.0
                                _batch_steps.append((
                                    vv, _rel,
                                    min(edge.confidence, ce.confidence) * CVT_HOP_PENALTY,
                                    ce.provenance, ce.valid_from, ce.valid_to, _pb,
                                ))
                    else:
                        _pb = 1.5 if (expected_rel and edge.relation_type == expected_rel) else 1.0
                        _batch_steps.append((
                            v, edge.relation_type, edge.confidence,
                            edge.provenance, edge.valid_from, edge.valid_to, _pb,
                        ))

                # Phase 134: vectorized batch scoring (one matrix multiply per path per hop)
                # isinstance guard: MagicMock auto-creates any attr, causing silent failures
                _cwb = (getattr(self.csa, "compute_weights_batch", None)
                        if isinstance(self.csa, CSAEngine) else None)
                _prior_scale = (
                    1.0 + self.warm_start_strength
                    if (self.probabilistic and self.warm_start_strength > 0.0 and len(path.nodes) == 1)
                    else 0.500
                )
                _scored_steps: List[Tuple[Any, int, str]] = []  # (new_path, v_cid, rel_eff)

                if _batch_steps and _cwb is not None:
                    # --- Batch path: vectorized CSA scoring ---
                    _v_list   = [s[0] for s in _batch_steps]
                    _rel_list = [s[1] for s in _batch_steps]
                    _vt_list  = [s[5] for s in _batch_steps]

                    _ev_list: List[Optional[np.ndarray]] = []
                    for _v in _v_list:
                        if _v not in emb_cache:
                            emb_cache[_v] = _get_emb(_v)
                        _ev_list.append(emb_cache[_v])

                    _logits = _cwb(
                        u=u, v_list=_v_list, hop=hop,
                        edge_types=_rel_list, valid_tos=_vt_list,
                        eu=eu, ev_list=_ev_list,
                        edge_type_weights=self.edge_type_weights,
                    )

                    for _i, (v_eff, rel_eff, conf_eff, prov_eff, vf_eff, vt_eff, prior_bias) in enumerate(_batch_steps):
                        _logit = _logits[_i]
                        w = _logit.score(csa_params) * prior_bias
                        features_vector = _logit.to_vector()

                        if self._causal_edge_index and (u, v_eff) in self._causal_edge_index:
                            w *= (1.0 + self.causal_bonus)
                        if node_priming and v_eff in node_priming:
                            w = w * (1.0 + self.priming_boost * node_priming[v_eff])
                        if _get_valence:
                            _val = _get_valence(u, v_eff, rel_eff)
                            if _val < 0.0:
                                w = w * (1.0 + _valence_eng.valence_weight * _val)
                        if self.terminal_relation_boost:
                            if hop == self.max_hop:
                                # Phase 148: Negative Terminal Boost (Penalty for non-target relations)
                                # If a target is known, unmatched relations are crushed (0.1x).
                                boost = self.terminal_relation_boost.get(rel_eff, 0.01)
                                w *= boost
                            elif hop == self.max_hop - 1:
                                if self.penultimate_relation_boost:
                                    # Phase 156: dedicated r2 boost (different relation from r3)
                                    pb = self.penultimate_relation_boost.get(rel_eff, 1.0)
                                    if pb > 1.0:
                                        w *= pb
                                else:
                                    tb = self.terminal_relation_boost.get(rel_eff, 1.0)
                                    if tb > 1.0:
                                        w *= tb ** 0.5  # penultimate cascade: 3.0 -> ~1.73x

                        if v_eff not in comm_cache:
                            comm_cache[v_eff] = _get_comm(v_eff)
                        v_cid = comm_cache[v_eff]
                        v_emb = _ev_list[_i] if _ev_list[_i] is not None else np.zeros(emb_dim, dtype=np.float32)
                        coh = community_coherence(path.community_sequence + [v_cid])

                        new_path = path.copy_with_extension(
                            rel=rel_eff, v=v_eff, v_cid=v_cid, v_emb=v_emb,
                            weight=w, coherence_score=coh,
                            edge_confidence=conf_eff, edge_provenance=prov_eff,
                            prior_scale=_prior_scale,
                            features=tuple(features_vector) if features_vector is not None else None,
                        )
                        _scored_steps.append((new_path, v_cid, rel_eff))

                elif _batch_steps:
                    # --- Fallback: per-step scoring (test mocks, older CSAEngine) ---
                    for (v_eff, rel_eff, conf_eff, prov_eff, vf_eff, vt_eff, prior_bias) in _batch_steps:
                        if v_eff in emb_cache:
                            ev = emb_cache[v_eff]
                        else:
                            ev = _get_emb(v_eff)
                            emb_cache[v_eff] = ev

                        if v_eff in comm_cache:
                            v_cid = comm_cache[v_eff]
                        else:
                            v_cid = _get_comm(v_eff)
                            comm_cache[v_eff] = v_cid

                        _cwf_result = _cwf(
                            u, v_eff, hop=hop,
                            edge_type=rel_eff,
                            edge_type_weights=self.edge_type_weights,
                            normalized_distance=0.0,
                            valid_from=vf_eff,
                            valid_to=vt_eff,
                            eu=eu, ev=ev,
                        ) if _cwf is not None else None

                        w: float
                        features_vector: Optional[np.ndarray] = None

                        if isinstance(_cwf_result, ReasoningLogit):
                            logit_obj = _cwf_result
                            w = logit_obj.score(csa_params)
                            features_vector = logit_obj.to_vector()
                        else:
                            w = float(self.csa.compute_weight(
                                u, v_eff, hop=hop,
                                edge_type=rel_eff,
                                edge_type_weights=self.edge_type_weights,
                                normalized_distance=0.0,
                                valid_from=vf_eff,
                                valid_to=vt_eff,
                            ))
                            sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0
                            cs  = float(self.csa.community_score(u, v_eff)) if hasattr(self.csa, "community_score") else 0.5
                            etw = self.edge_type_weights.get(rel_eff, 0.0)
                            hd  = 1.0 / (1.0 + hop)
                            features_vector = ReasoningLogit(sim, cs, etw, 0.0, hd, 0.0, 0.0, 0.0, 0.0, 1.0).to_vector()

                        w = w * prior_bias

                        if self._causal_edge_index and (u, v_eff) in self._causal_edge_index:
                            w *= (1.0 + self.causal_bonus)
                        if node_priming and v_eff in node_priming:
                            w = w * (1.0 + self.priming_boost * node_priming[v_eff])
                        if _get_valence:
                            _val = _get_valence(u, v_eff, rel_eff)
                            if _val < 0.0:
                                w = w * (1.0 + _valence_eng.valence_weight * _val)
                        if self.terminal_relation_boost:
                            if hop == self.max_hop:
                                # Phase 148: Negative Terminal Boost (Penalty for non-target relations)
                                # If a target is known, unmatched relations are crushed (0.1x).
                                boost = self.terminal_relation_boost.get(rel_eff, 0.01)
                                w *= boost
                            elif hop == self.max_hop - 1:
                                if self.penultimate_relation_boost:
                                    # Phase 156: dedicated r2 boost (different relation from r3)
                                    pb = self.penultimate_relation_boost.get(rel_eff, 1.0)
                                    if pb > 1.0:
                                        w *= pb
                                else:
                                    tb = self.terminal_relation_boost.get(rel_eff, 1.0)
                                    if tb > 1.0:
                                        w *= tb ** 0.5  # penultimate cascade: 3.0 -> ~1.73x

                        v_emb = ev if ev is not None else np.zeros(emb_dim, dtype=np.float32)
                        coh = community_coherence(path.community_sequence + [v_cid])

                        new_path = path.copy_with_extension(
                            rel=rel_eff, v=v_eff, v_cid=v_cid, v_emb=v_emb,
                            weight=w, coherence_score=coh,
                            edge_confidence=conf_eff, edge_provenance=prov_eff,
                            prior_scale=_prior_scale,
                            features=tuple(features_vector) if features_vector is not None else None,
                        )
                        _scored_steps.append((new_path, v_cid, rel_eff))

                # Phase 39: Task dispatch - shared for both scoring paths
                for new_path, v_cid, rel_eff in _scored_steps:
                    if self.task_queue is not None:
                        if (self.bridge_engine is not None and
                            u_cid != v_cid and u_cid >= 0 and v_cid >= 0 and
                            rel_eff != BRIDGE_RELATION):
                            self.task_queue.enqueue(BridgeTask(
                                node_id=new_path.tail,
                                source_community=v_cid,
                                dest_community=u_cid,
                            ))

                        if (self.bridge_engine is not None and rel_eff == BRIDGE_RELATION):
                            self.task_queue.enqueue(BridgeTask(
                                node_id=new_path.tail,
                                source_community=-1,
                                dest_community=-1,
                                is_twin_use=True,
                            ))

                        if (self.insight_engine is not None and
                            u_cid != v_cid and u_cid >= 0 and v_cid >= 0):
                            self.task_queue.enqueue(BridgeTask(
                                node_id=new_path.tail,
                                source_community=u_cid,
                                dest_community=v_cid,
                                path_score=new_path.score,
                                path=new_path,
                                u_node=path.tail,
                                v_node=new_path.tail,
                            ))
                    else:
                        # Fallback to synchronous calls if no task queue (e.g. testing)
                        if (
                            self.bridge_engine is not None
                            and u_cid != v_cid
                            and u_cid >= 0
                            and v_cid >= 0
                            and rel_eff != BRIDGE_RELATION
                        ):
                            self.bridge_engine.record_crossing(
                                node_id=new_path.tail,
                                source_community=v_cid,
                                dest_community=u_cid,
                                adapter=self.adapter,
                            )

                        if (
                            self.bridge_engine is not None
                            and rel_eff == BRIDGE_RELATION
                        ):
                            self.bridge_engine.record_twin_use(new_path.tail)

                        if (
                            self.insight_engine is not None
                            and u_cid != v_cid
                            and u_cid >= 0
                            and v_cid >= 0
                        ):
                            self.insight_engine.record_crossing(
                                u=path.tail,
                                v=new_path.tail,
                                u_cid=u_cid,
                                v_cid=v_cid,
                                path_score=path.score,
                                path=path,
                            )

                    candidates.append(new_path)

            _log.debug(
                "Hop %d: beam=%d  candidates=%d  expansions_so_far=%d",
                hop, len(beam), len(candidates), self.expansions,
            )
            if not candidates:
                _log.debug("Hop %d: no candidates - stopping early", hop)
                break

            # Phase 37: Self-Doubt (Calibration)
            if self.calibration_engine is not None and len(candidates) >= 2:
                weights = [p.attention_weights[-1] for p in candidates]
                tails   = [p.tail for p in candidates]
                result  = self.calibration_engine.calibrate_hop(weights, tails)
                
                if result.is_uncertain:
                    self.uncertainty_log.append({
                        "hop": hop,
                        "entropy": result.entropy,
                        "candidates": result.top_candidates,
                        "message": result.message
                    })
                    # Penalize all candidates by the uncertainty multiplier
                    for p in candidates:
                        p.score *= result.confidence_multiplier

            # At the terminal hop, skip pruning - all reachable endpoints are kept
            # for the answer extractor to score and deduplicate. Pruning here discards
            # valid answers with zero benefit (no further expansion occurs).
            hop_bw = self._beam_widths.get(hop, self.beam_width)

            # Phase 108: Thalamofrontal Feedback Loop
            # Dynamically adjust the gating threshold for the NEXT hop based on 
            # the current candidates' quality and metabolic pressure.
            if self.dynamic_thalamic_gating and candidates:
                avg_score = sum(p.score for p in candidates) / len(candidates)
                # Tighten gate if average quality is high; loosen if low.
                # Inspired by the ALARM theory (Ruhr University Bochum, 2025).
                self.thalamic_threshold = avg_score * self.gating_sensitivity

            beam = self._prune_candidates(candidates, hop)
            
            # Phase 62: Explainable Reasoning Trace (ERT)
            if trace_info:
                winners_set = set(id(p) for p in beam)
                competitors = [p for p in candidates if id(p) not in winners_set]
                # Sort competitors by score for the trace (winners already sorted by _prune_candidates)
                competitors.sort(key=lambda p: p.score, reverse=True)
                trace_info.add_hop(hop, beam, competitors, len(candidates), hop_bw)

            all_paths.extend(beam)
            self._partial_paths = list(all_paths)  # checkpoint: survives mid-hop exceptions

        return all_paths

    def _apply_lateral_inhibition(
        self, candidates: "List[TraversalPath]"
    ) -> "List[TraversalPath]":
        """Phase 100: Non-Maximum Suppression grouped by tail community.

        Within each community group, lower-ranked paths are suppressed
        proportionally to their rank. ratio=0 -> no change; ratio=1 -> strict
        one-winner-per-community. The winner (rank 0) is never suppressed.
        """
        ratio = self.lateral_inhibition_ratio
        if ratio <= 0.0:
            return candidates

        # Group by tail community ID
        groups: Dict[int, List[TraversalPath]] = {}
        for p in candidates:
            cid = self.adapter.get_community(p.tail) if p.tail else -1
            groups.setdefault(cid, []).append(p)

        result = []
        for group in groups.values():
            group.sort(key=lambda p: p.score, reverse=True)
            n = len(group)
            for rank, p in enumerate(group):
                if rank == 0:
                    result.append(p)
                else:
                    suppressed_score = p.score * (1.0 - ratio * rank / n)
                    # Create a shallow copy with suppressed score
                    import copy
                    sp = copy.copy(p)
                    sp.score = max(0.0, suppressed_score)
                    result.append(sp)
        return result

    def _thalamic_gating(
        self,
        candidates: "List[TraversalPath]",
    ) -> "List[TraversalPath]":
        """Phase 106: Thalamic Context Gating.
        
        Prunes paths that fall below the thalamic_threshold. This acts as an 
        early-exit mechanism for unproductive branches before they reach the 
        expensive sorting/NMS phase. 
        
        The threshold is compared against the path's product-of-grounding scores.
        """
        if self.thalamic_threshold <= 0.0:
            return candidates
            
        return [p for p in candidates if p.score >= self.thalamic_threshold]

    def _prune_candidates(
        self,
        candidates: "List[TraversalPath]",
        hop: int,
    ) -> "List[TraversalPath]":
        """
        Prune *candidates* to beam width and return the surviving beam.

        # Phase 149: Monitor reasoning entropy (Cingulate Engine)
        entropy = self._calculate_conflict_entropy(candidates)
        self._last_entropy = entropy
        if entropy > 2.0:
            _log.debug("CingulateEngine: High conflict entropy=%.4f hop=%d", entropy, hop)

        Override in subclasses to inject alternative scoring strategies
        (e.g. Engram-steered pruning).  The default implementation sorts by
        raw path score (or Thompson-sampled Beta score in probabilistic mode).

        Phase 106: Thalamic context gating is applied as a hard-filter early-exit.
        Phase 100: Lateral inhibition (community-grouped NMS) is applied
        before final selection when lateral_inhibition_ratio > 0.

        The terminal hop is never pruned - all candidates are returned sorted
        so the answer extractor has the full frontier to deduplicate.
        """
        hop_bw = self._beam_widths.get(hop, self.beam_width)
        
        # Phase 106: Early-exit gating
        if self.thalamic_threshold > 0.0:
            candidates = self._thalamic_gating(candidates)
            if not candidates:
                return []


        # Phase 150: Detect Epistemic Gaps (Frontal Engine)
        if hop > 1 and candidates and not any(p.score > 0.4 for p in candidates):
            # Identify the top potential but "grounding-starved" paths
            for p in sorted(candidates, key=lambda x: x.score, reverse=True)[:2]:
                self.epistemic_gaps.append({
                    "source": p.nodes[0],
                    "target": p.tail,
                    "score": p.score,
                    "hop": hop
                })

        # Phase 151: PenultimateGate — score-gap filter at the penultimate hop.
        # Drops candidates whose score falls below decay_factor * best_score,
        # preventing weak middle-hop paths from polluting the final hop expansion.
        # Only fires at hop == max_hop - 1 (not terminal, not earlier hops).
        if (
            self.penultimate_decay > 0.0
            and hop == self.max_hop - 1
            and candidates
        ):
            best = max(p.score for p in candidates)
            floor = self.penultimate_decay * best
            candidates = [p for p in candidates if p.score >= floor]

        if self.lateral_inhibition_ratio > 0.0:
            candidates = self._apply_lateral_inhibition(candidates)
        if self.probabilistic:
            _rng = self._rng
            
            # Helper for sampling: if quantized, we approximate float score from q_score
            # because Thompson sampling needs the Beta(alpha, beta) which we 
            # still maintain in float for accuracy.
            def _get_key(p: TraversalPath) -> float:
                return p.sample_score(_rng)

            if hop < self.max_hop and len(candidates) > hop_bw:
                return heapq.nlargest(
                    hop_bw, candidates,
                    key=_get_key,
                )
            return sorted(candidates,
                          key=_get_key,
                          reverse=True)
        else:
            # Deterministic: use q_score if quantized, otherwise score
            if self.quantized:
                if hop < self.max_hop and len(candidates) > hop_bw:
                    return heapq.nlargest(hop_bw, candidates, key=lambda p: p.q_score)
                return sorted(candidates, key=lambda p: p.q_score, reverse=True)
            else:
                if hop < self.max_hop and len(candidates) > hop_bw:
                    return heapq.nlargest(hop_bw, candidates, key=lambda p: p.score)
                return sorted(candidates, key=lambda p: p.score, reverse=True)

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
        trace_info: Optional["ReasoningTrace"] = None,
        node_priming: Optional[Dict[str, float]] = None,
    ):
        """
        Run beam traversal and yield paths hop-by-hop.
        Yields: List[TraversalPath] at each hop completion.

        query_time : Unix timestamp for temporal filtering (see traverse()).
        trace_info : Optional ReasoningTrace to populate (Phase 62).
        """
        emb_dim = self._infer_dim()
        self.expansions = 0

        # Hole 1 - Mid-Flight Community Swap: snapshot at stream start.
        _cmap = getattr(self.adapter, "community_map", None)
        if self.csa is not None:
            if _cmap is not None:
                self.csa.set_query_snapshot(dict(_cmap))
            self.csa.set_query_time(query_time)
        try:
            async for hop_result in self._traverse_stream_inner(seeds, emb_dim, query_time, trace_info=trace_info, node_priming=node_priming):
                yield hop_result
        finally:
            if self.csa is not None:
                self.csa.clear_query_snapshot()
                self.csa.set_query_time(None)

    async def _traverse_stream_inner(self, seeds, emb_dim, query_time, trace_info=None, node_priming=None):
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
                q_score=QUANT_SCALE,
                quantized=self.quantized,
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
                    if not is_valid_at(edge, query_time, hard_filter=not self.csa.use_temporal_decay):
                        continue

                    if v in path.seen_entities:
                        continue

                    # Symbolic Guardrails (Phase 34)
                    if self.symbolic_validator is not None:
                        if not self.symbolic_validator.validate_step(path.tail, edge.relation_type, v, path=path):
                            continue

                    # CVT passthrough (same logic as sync traversal)
                    if self.cvt_passthrough and _is_cvt_node(v):
                        cvt_edges = self.adapter.get_neighbors(
                            v, max_neighbors=self.max_neighbors
                        )
                        next_steps: List[Tuple[str, str, float, str, Optional[float], Optional[float]]] = []
                        for ce in cvt_edges:
                            vv = ce.target_id
                            if vv not in path.seen_entities and not _is_cvt_node(vv):
                                next_steps.append((
                                    vv,
                                    f"{edge.relation_type}|{ce.relation_type}",
                                    min(edge.confidence, ce.confidence) * CVT_HOP_PENALTY,
                                    ce.provenance,
                                    ce.valid_from,
                                    ce.valid_to,
                                ))
                    else:
                        next_steps = [
                            (v, edge.relation_type, edge.confidence, edge.provenance, edge.valid_from, edge.valid_to)
                        ]

                    for (v_eff, rel_eff, conf_eff, prov_eff, vf_eff, vt_eff) in next_steps:
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
                            valid_from=vf_eff,
                            valid_to=vt_eff,
                            eu=eu, ev=ev,
                        ) if _cwf is not None else None

                        w: float
                        features_vector: Optional[np.ndarray] = None

                        if isinstance(_cwf_result, ReasoningLogit):
                            logit_obj = _cwf_result
                            # Calculate weight using parameters, default to neutral if learner not present
                            csa_params = self.csa.get_current_params(path.tail) if hasattr(self.csa, 'get_current_params') else _DEFAULT_INIT_PARAMS
                            
                            # Phase 68: Apply dynamic hormonal overrides
                            if self.csa_overrides:
                                p_list = list(csa_params)
                                for i, k in enumerate(CSA_PARAM_KEYS):
                                    if k in self.csa_overrides:
                                        p_list[i] = self.csa_overrides[k]
                                csa_params = tuple(p_list)

                            w = logit_obj.score(csa_params)
                            features_vector = logit_obj.to_vector()
                        else:
                            # Fallback for older CSAEngine or mocks (should be deprecated soon)
                            w = float(self.csa.compute_weight(
                                path.tail, v_eff, hop=hop,
                                edge_type=rel_eff,
                                edge_type_weights=self.edge_type_weights,
                                normalized_distance=0.0,
                                valid_from=vf_eff,
                                valid_to=vt_eff,
                            ))
                            # Reconstruct minimal features for compatibility
                            sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0
                            cs  = float(self.csa.community_score(path.tail, v_eff)) if hasattr(self.csa, 'community_score') else 0.5
                            etw = self.edge_type_weights.get(rel_eff, 0.0)
                            nd  = 0.0
                            hd  = 1.0 / (1.0 + hop)
                            pr_v = 0.0
                            td = 0.0
                            nr_v = 0.0
                            grounding = 0.500
                            # Positional order: sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding
                            features_vector = ReasoningLogit(sim, cs, etw, nd, hd, pr_v, td, nr_v, 0.0, grounding).to_vector()

                        # Phase 99: Thalamic Gating - WM priming boost (async parity)
                        if node_priming and v_eff in node_priming:
                            w = w * (1.0 + self.priming_boost * node_priming[v_eff])

                        # Phase 101: Emotional Valence - aversive edge penalty (async parity)
                        _valence_engine = getattr(self, "_valence_engine", None)
                        if _valence_engine is not None:
                            _val = _valence_engine.get_valence(path.tail, v_eff, rel_eff)
                            if _val < 0.0:
                                w = w * (1.0 + _valence_engine.valence_weight * _val)

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
                            features=tuple(features_vector) if features_vector is not None else None,
                        )

                        # Phase 63: Telemetry Pulse
                        # We pass the graph reference during construction for async stream cases
                        if hasattr(self, 'graph') and self.graph:
                            self.graph.emit(NeuralEvent.pulse(
                                source=path.tail,
                                target=v_eff,
                                relation=rel_eff,
                                weight=new_path.score,
                                hop=hop,
                                features=list(features_vector) if features_vector is not None else None
                            ))

                        candidates.append(new_path)
            if not candidates:
                break

            # Phase 37: Self-Doubt (Calibration)
            if self.calibration_engine is not None and len(candidates) >= 2:
                weights = [p.attention_weights[-1] for p in candidates]
                tails   = [p.tail for p in candidates]
                result  = self.calibration_engine.calibrate_hop(weights, tails)
                
                if result.is_uncertain:
                    self.uncertainty_log.append({
                        "hop": hop,
                        "entropy": result.entropy,
                        "candidates": result.top_candidates,
                        "message": result.message
                    })
                    # Penalize all candidates by the uncertainty multiplier
                    for p in candidates:
                        p.score *= result.confidence_multiplier

            # At the terminal hop, skip pruning (same as sync version)
            hop_bw = self._beam_widths.get(hop, self.beam_width)
            beam = self._prune_candidates(candidates, hop)
            
            # Phase 62: Explainable Reasoning Trace (ERT)
            if trace_info:
                winners_set = set(id(p) for p in beam)
                competitors = [p for p in candidates if id(p) not in winners_set]
                competitors.sort(key=lambda p: p.score, reverse=True)
                trace_info.add_hop(hop, beam, competitors, len(candidates), hop_bw)

            # Yield this layer's best paths
            yield beam




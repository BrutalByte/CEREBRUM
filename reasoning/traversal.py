"""
Beam-search attention traversal — the forward pass of Parallax (Section 5.1).

BeamTraversal walks the graph hop-by-hop from seed entities, scoring each
candidate next-hop using CSA attention weights and pruning to beam_width at
each step. Returns all explored paths for downstream ranking.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import heapq

import numpy as np

from core.graph_adapter import GraphAdapter
from core.attention_engine import CSAEngine
from core.resource_governor import ResourceGovernor
from reasoning.path_scorer import community_coherence


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

    def copy_with_extension(
        self, 
        rel: str, 
        v: str, 
        v_cid: int, 
        v_emb: np.ndarray, 
        weight: float,
        coherence_score: float,
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
            
        return TraversalPath(
            nodes=new_nodes,
            seen_entities=new_seen,
            embedding=h_agg,
            score=self.score * weight * coherence_score,
            attention_weights=self.attention_weights + [weight],
            community_sequence=new_cseq,
        )

    def __repr__(self) -> str:
        chain = " -> ".join(
            f"[{self.nodes[i]}]" if i % 2 == 0 else self.nodes[i]
            for i in range(len(self.nodes))
        )
        return f"TraversalPath(score={self.score:.4f}, path={chain})"


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
        max_neighbors: int = 50,
        max_budget: int = 1000,
        edge_type_weights: Optional[Dict[str, float]] = None,
        governor: Optional[ResourceGovernor] = None,
    ):
        self.adapter            = adapter
        self.csa                = csa_engine
        self.beam_width         = beam_width
        self.max_hop            = max_hop
        self.max_neighbors      = max_neighbors
        self.max_budget         = max_budget
        self.edge_type_weights  = edge_type_weights or {}
        self.expansions         = 0
        self.governor           = governor or ResourceGovernor()

    def traverse(self, seeds: List[str]) -> List[TraversalPath]:
        """
        Run beam traversal from the given seed entity IDs.
        """
        emb_dim = self._infer_dim()
        self.expansions = 0

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
                )
                self.expansions += 1

                for edge in edges:
                    v = edge.target_id

                    # Avoid revisiting entities already in the path (O(1) now)
                    if v in path.seen_entities:
                        continue

                    # CSA attention weight
                    w = self.csa.compute_weight(
                        path.tail,
                        v,
                        hop=hop,
                        edge_type=edge.relation_type,
                        edge_type_weights=self.edge_type_weights,
                    )

                    # Path metadata and extension
                    v_emb = self.adapter.get_embedding(v)
                    if v_emb is None:
                        v_emb = np.zeros(emb_dim, dtype=np.float32)
                    
                    v_cid = self.adapter.get_community(v)
                    
                    # Compute community coherence for the candidate step
                    coh = community_coherence(path.community_sequence + [v_cid])
                    
                    new_path = path.copy_with_extension(
                        rel=edge.relation_type,
                        v=v,
                        v_cid=v_cid,
                        v_emb=v_emb,
                        weight=w,
                        coherence_score=coh
                    )
                    candidates.append(new_path)

            if not candidates:
                break

            # Efficiently pick top B candidates
            if len(candidates) > self.beam_width:
                beam = heapq.nlargest(self.beam_width, candidates, key=lambda p: p.score)
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

    async def traverse_stream(self, seeds: List[str]):
        """
        Run beam traversal and yield paths hop-by-hop.
        Yields: List[TraversalPath] at each hop completion.
        """
        import asyncio
        emb_dim = self._infer_dim()
        self.expansions = 0

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
                )
                self.expansions += 1
                
                # Small yield to prevent event loop starvation on huge graphs
                if self.expansions % 100 == 0:
                    await asyncio.sleep(0)

                for edge in edges:
                    v = edge.target_id
                    if v in path.seen_entities:
                        continue

                    w = self.csa.compute_weight(
                        path.tail,
                        v,
                        hop=hop,
                        edge_type=edge.relation_type,
                        edge_type_weights=self.edge_type_weights,
                    )

                    v_emb = self.adapter.get_embedding(v)
                    if v_emb is None:
                        v_emb = np.zeros(emb_dim, dtype=np.float32)
                    
                    v_cid = self.adapter.get_community(v)
                    coh = community_coherence(path.community_sequence + [v_cid])
                    
                    new_path = path.copy_with_extension(
                        rel=edge.relation_type,
                        v=v,
                        v_cid=v_cid,
                        v_emb=v_emb,
                        weight=w,
                        coherence_score=coh
                    )
                    candidates.append(new_path)

            if not candidates:
                break

            # Pick top B candidates for next hop
            if len(candidates) > self.beam_width:
                beam = heapq.nlargest(self.beam_width, candidates, key=lambda p: p.score)
            else:
                beam = sorted(candidates, key=lambda p: p.score, reverse=True)
            
            # Yield this layer's best paths
            yield beam




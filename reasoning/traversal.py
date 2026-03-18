"""
Beam-search attention traversal — the forward pass of Parallax (Section 5.1).

BeamTraversal walks the graph hop-by-hop from seed entities, scoring each
candidate next-hop using CSA attention weights and pruning to beam_width at
each step. Returns all explored paths for downstream ranking.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

from core.graph_adapter import GraphAdapter
from core.attention_engine import CSAEngine
from reasoning.path_scorer import community_coherence


@dataclass
class TraversalPath:
    """A single reasoning path through the knowledge graph."""

    nodes: List[str] = field(default_factory=list)
    """
    Alternating sequence: [entity, relation, entity, relation, entity, ...].
    Entity nodes are at even indices; relation labels at odd indices.
    """

    embedding: Optional[np.ndarray] = None
    """Current aggregated embedding (updated at each hop)."""

    score: float = 1.0
    """Running product-of-weights path score."""

    attention_weights: List[float] = field(default_factory=list)
    """Attention weight assigned at each hop."""

    community_sequence: List[int] = field(default_factory=list)
    """Community ID of each entity node visited."""

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
        embeddings: Dict[str, np.ndarray],
        communities: Dict[str, int],
        beam_width: int = 10,
        max_hop: int = 3,
        max_neighbors: int = 50,
        edge_type_weights: Optional[Dict[str, float]] = None,
    ):
        self.adapter            = adapter
        self.csa                = csa_engine
        self.embeddings         = embeddings
        self.communities        = communities
        self.beam_width         = beam_width
        self.max_hop            = max_hop
        self.max_neighbors      = max_neighbors
        self.edge_type_weights  = edge_type_weights or {}

    def traverse(self, seeds: List[str]) -> List[TraversalPath]:
        """
        Run beam traversal from the given seed entity IDs.

        Parameters
        ----------
        seeds : list of entity IDs to start from (STEP 1 output)

        Returns
        -------
        All paths explored across all hops, including seed-only depth-0 paths.
        Call reasoning.answer_extractor.extract() to get top-K ranked answers.
        """
        emb_dim = self._infer_dim()

        # Initialize beam from seed entities
        beam: List[TraversalPath] = []
        for seed in seeds:
            emb = self.embeddings.get(seed, np.zeros(emb_dim, dtype=np.float32))
            cid = self.communities.get(seed, -1)
            beam.append(
                TraversalPath(
                    nodes=[seed],
                    embedding=emb.copy(),
                    score=1.0,
                    community_sequence=[cid] if cid >= 0 else [],
                )
            )

        all_paths: List[TraversalPath] = list(beam)

        for hop in range(1, self.max_hop + 1):
            candidates: List[TraversalPath] = []

            for path in beam:
                edges = self.adapter.get_neighbors(
                    path.tail,
                    max_neighbors=self.max_neighbors,
                )

                for edge in edges:
                    v = edge.target_id

                    # Avoid revisiting entities already in the path
                    if v in path.entity_nodes:
                        continue

                    # CSA attention weight
                    w = self.csa.compute_weight(
                        path.tail,
                        v,
                        hop=hop,
                        edge_type=edge.relation_type,
                        edge_type_weights=self.edge_type_weights,
                    )

                    # Embedding aggregation with residual + LayerNorm
                    v_emb   = self.embeddings.get(v, np.zeros(emb_dim, dtype=np.float32))
                    h_new   = np.maximum(0.0, w * v_emb + path.embedding) + path.embedding
                    norm    = float(np.linalg.norm(h_new))
                    if norm > 0:
                        h_new = h_new / norm

                    # Path metadata
                    new_nodes  = path.nodes + [edge.relation_type, v]
                    new_cseq   = path.community_sequence + [self.communities.get(v, -1)]
                    coh        = community_coherence(new_cseq)
                    new_score  = path.score * w * coh

                    candidates.append(
                        TraversalPath(
                            nodes=new_nodes,
                            embedding=h_new,
                            score=new_score,
                            attention_weights=path.attention_weights + [w],
                            community_sequence=new_cseq,
                        )
                    )

            if not candidates:
                break

            candidates.sort(key=lambda p: p.score, reverse=True)
            beam = candidates[: self.beam_width]
            all_paths.extend(beam)

        return all_paths

    def _infer_dim(self) -> int:
        """Infer embedding dimension from the first available vector."""
        for v in self.embeddings.values():
            return len(v)
        return 64

"""
MetaRelationTraversal â€” Phase 217: Relational Abstraction Layer.

The human brain reasons about relations between relations: "X causes Y" is
itself a concept you can reason over.  This module lifts relation types to
first-class graph citizens and provides a beam-search traversal over the
meta-relation graph (the graph of graphs).

Use cases
---------
- "What kinds of connections link disease to gene?" â†’
  traverse the meta-graph starting from 'disease_*' relations.
- Explain *why* a query traversal followed a certain path: which relation
  type transitions are most common for this query type?
- Improve TRB initialisation: seed STRB from the top-k meta-paths from
  the query's detected first relation type.

Architecture
------------
MetaRelationTraversal operates on the meta-graph built by
NetworkXAdapter.build_meta_graph() (or any adapter implementing
get_meta_neighbors()).  It uses the same beam-search skeleton as
BeamTraversal but over relation-type nodes instead of entity nodes.

The score for a meta-path step is:
    score = prev_score * edge_weight * depth_decay
where edge_weight is the TF-IDF-normalised co-occurrence weight from the
meta-graph and depth_decay = 1 / (1 + hop) discourages very long chains.
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter

logger = logging.getLogger("cerebrum.meta_relation")


@dataclass
class MetaPath:
    """A path through the meta-relation graph (a chain of relation types)."""
    relations: List[str]          # sequence of relation type names
    meta_relations: List[str]     # meta-relation labels connecting them
    score: float = 1.0
    hops: int = 0

    def extend(self, next_rel: str, meta_rel: str, step_weight: float, hop: int) -> "MetaPath":
        depth_decay = 1.0 / (1.0 + hop)
        return MetaPath(
            relations=self.relations + [next_rel],
            meta_relations=self.meta_relations + [meta_rel],
            score=self.score * step_weight * depth_decay,
            hops=hop,
        )

    @property
    def tail(self) -> Optional[str]:
        return self.relations[-1] if self.relations else None

    def __repr__(self) -> str:
        chain = " â†’ ".join(
            f"{r}[{m}]" if m else r
            for r, m in zip(self.relations, [""] + self.meta_relations)
        )
        return f"MetaPath({chain}, score={self.score:.4f})"


class MetaRelationTraversal:
    """
    Beam-search traversal over the meta-relation graph.

    Parameters
    ----------
    adapter
        Any GraphAdapter whose get_meta_neighbors() is implemented.
        NetworkXAdapter builds the meta-graph lazily on first call.
    beam_width
        Number of meta-paths to keep at each hop.
    max_hop
        Maximum depth of the meta-path search.
    min_weight
        Prune meta-edges with weight below this floor (noise reduction).
    """

    def __init__(
        self,
        adapter: "GraphAdapter",
        beam_width: int = 10,
        max_hop: int = 3,
        min_weight: float = 0.01,
    ) -> None:
        self.adapter = adapter
        self.beam_width = beam_width
        self.max_hop = max_hop
        self.min_weight = min_weight

    def traverse(self, seed_relation: str, max_hop: Optional[int] = None) -> List[MetaPath]:
        """
        Beam-search starting from seed_relation in the meta-graph.

        Returns all MetaPath objects explored, sorted by score descending.
        """
        max_hop = max_hop or self.max_hop
        beam: List[MetaPath] = [MetaPath(relations=[seed_relation], meta_relations=[], score=1.0)]
        all_paths: List[MetaPath] = list(beam)

        for hop in range(1, max_hop + 1):
            candidates: List[MetaPath] = []
            for path in beam:
                tail = path.tail
                if tail is None:
                    continue
                for meta_edge in self.adapter.get_meta_neighbors(tail):
                    if meta_edge.weight < self.min_weight:
                        continue
                    next_rel = meta_edge.target_relation
                    if next_rel in path.relations:
                        continue  # no cycles
                    candidates.append(path.extend(
                        next_rel=next_rel,
                        meta_rel=meta_edge.meta_relation,
                        step_weight=meta_edge.weight,
                        hop=hop,
                    ))

            if not candidates:
                break

            # Prune to beam_width
            if len(candidates) > self.beam_width:
                beam = heapq.nlargest(self.beam_width, candidates, key=lambda p: p.score)
            else:
                beam = sorted(candidates, key=lambda p: p.score, reverse=True)

            all_paths.extend(beam)

        all_paths.sort(key=lambda p: p.score, reverse=True)
        return all_paths

    def explain_query(self, query: str, adapter: Optional["GraphAdapter"] = None) -> List[MetaPath]:
        """
        Identify which relation sequences best explain the semantics of a query.

        If the adapter has sentence-transformers embeddings (STRB), finds the
        most semantically similar relation type to the query and uses it as the
        seed for meta-path traversal.  Falls back to traversal from all relation
        types if no embeddings are available.
        """
        _adapter = adapter or self.adapter

        # Try STRB: encode query and find best-matching relation type
        seed_relation: Optional[str] = None
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            edge_types: List[str] = []
            try:
                edge_types = _adapter.get_edge_types()
            except NotImplementedError:
                pass

            if edge_types:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                q_emb = model.encode(query, normalize_embeddings=True)
                rel_embs = model.encode(
                    [r.replace("_", " ") for r in edge_types],
                    normalize_embeddings=True,
                )
                sims = rel_embs @ q_emb
                seed_relation = edge_types[int(sims.argmax())]
                logger.debug("explain_query: seed_relation='%s' for query='%s'", seed_relation, query)
        except ImportError:
            pass

        if seed_relation:
            return self.traverse(seed_relation)

        # Fallback: collect all relation types and traverse from each, return merged
        all_paths: List[MetaPath] = []
        try:
            for rel in _adapter.get_edge_types():
                all_paths.extend(self.traverse(rel, max_hop=2))
        except NotImplementedError:
            pass
        all_paths.sort(key=lambda p: p.score, reverse=True)
        return all_paths[:self.beam_width * self.max_hop]

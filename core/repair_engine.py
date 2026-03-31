"""
IncompletenessRepairEngine — Phase 28B
=======================================

Synthesizes missing edges in incomplete Knowledge Graphs by detecting
"near-miss" dead-end paths from traversal output and inferring the most
likely missing connections.

Why this works
--------------
The IKGWQ incompleteness protocol removes edges *incident to answer nodes*.
These are exactly the "last hop" edges that BeamTraversal fails to traverse.
A high-confidence path that terminates at a low-degree node is a strong
signal that the graph is broken at that point — the system reached somewhere
interesting but couldn't go further.

Algorithm
---------
1. **Dead-end detection**: scan all traversal paths; identify tails that
   (a) have path.score >= min_path_score and
   (b) have graph degree <= dead_end_max_degree.
   These are "near-miss" endpoints — the traversal got stuck here.

2. **Candidate generation**: for each dead-end node, find semantically
   similar nodes via embedding cosine similarity over community members.
   Nodes in the same or adjacent community are preferred.

3. **Scoring**: rank candidates by
       combined_score = embedding_similarity(dead_end, candidate)
                       * relation_prior.score(path_to_dead_end)
   If a KGE engine is provided, the KGE plausibility score is multiplied in.

4. **Edge synthesis**: add top-K synthetic edges to the graph with
   ``synthesized=True`` and ``confidence=combined_score``.

5. Return the augmented graph and count of synthesized edges.

Integration
-----------
Typical use in ikgwq_eval.py::

    from core.repair_engine import IncompletenessRepairEngine

    repair = IncompletenessRepairEngine(
        adapter,
        relation_prior=graph_prior,
        kge_engine=kge,          # optional
    )
    G_repaired, n_synth = repair.repair(paths, G_incomplete)
    # Rebuild adapter/traversal on G_repaired and re-evaluate
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from core.graph_adapter import GraphAdapter

# Lazy import to avoid circular deps — only needed if a KGE engine is passed
# from core.kge_engine import _BaseKGEEngine

SYNTH_RELATION = "__synthesized__"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class IncompletenessRepairEngine:
    """
    Synthesize missing edges by detecting high-confidence dead-end paths.

    Parameters
    ----------
    adapter
        The graph adapter wrapping the incomplete graph.  Must have
        ``community_map`` and ``embeddings`` set (standard after
        ``build_full_pipeline``).
    relation_prior
        Optional :class:`~reasoning.relation_path_prior.GraphRelationPrior`
        used to weight synthesis candidates by relation plausibility.
    kge_engine
        Optional trained KGE engine (TransE / RotatE).  When provided,
        ``predict_links`` scores are multiplied into the candidate ranking.
    dead_end_max_degree
        Path tails with graph degree <= this threshold are treated as
        potential dead-ends caused by edge removal.  Default 2.
    min_path_score
        Only paths with score >= this threshold are considered as dead-ends.
        Filters out low-quality traversal noise.  Default 0.05.
    max_candidates
        Maximum neighbourhood nodes to consider per dead-end.  Default 30.
    max_synth_per_node
        Maximum synthetic edges to add per dead-end node.  Default 3.
    confidence_threshold
        Minimum combined score for a synthetic edge to be added.  Default 0.2.
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        relation_prior=None,
        kge_engine=None,
        dead_end_max_degree: int = 2,
        min_path_score: float = 0.05,
        max_candidates: int = 30,
        max_synth_per_node: int = 3,
        confidence_threshold: float = 0.2,
    ):
        self.adapter              = adapter
        self.relation_prior       = relation_prior
        self.kge_engine           = kge_engine
        self.dead_end_max_degree  = dead_end_max_degree
        self.min_path_score       = min_path_score
        self.max_candidates       = max_candidates
        self.max_synth_per_node   = max_synth_per_node
        self.confidence_threshold = confidence_threshold

        # Build reverse community map: cid → [node_ids]
        self._community_members: Dict[int, List[str]] = {}
        cmap = getattr(adapter, "community_map", {}) or {}
        for node, cid in cmap.items():
            self._community_members.setdefault(cid, []).append(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(
        self,
        paths: list,
        G: nx.Graph,
    ) -> Tuple[nx.Graph, int]:
        """
        Detect dead-end paths and synthesize likely missing edges.

        Parameters
        ----------
        paths
            Output of ``BeamTraversal.traverse()`` on the incomplete graph.
        G
            The incomplete NetworkX graph (not modified; a copy is returned).

        Returns
        -------
        (augmented_graph, n_synthesized)
            ``augmented_graph`` is a copy of G with synthetic edges added.
            ``n_synthesized`` is the number of new edges.
        """
        dead_ends = self._detect_dead_ends(paths, G)
        if not dead_ends:
            return G.copy(), 0

        G_aug = G.copy()
        n_synth = 0
        already_added: Set[Tuple[str, str]] = set()

        for dead_end_node, path in dead_ends:
            candidates = self._find_candidates(dead_end_node, path, G)
            added = 0
            for candidate, score in candidates:
                if added >= self.max_synth_per_node:
                    break
                if score < self.confidence_threshold:
                    break
                edge_key = (dead_end_node, candidate)
                rev_key  = (candidate, dead_end_node)
                if edge_key in already_added or rev_key in already_added:
                    continue
                if G_aug.has_edge(dead_end_node, candidate):
                    continue
                G_aug.add_edge(
                    dead_end_node,
                    candidate,
                    relation=SYNTH_RELATION,
                    confidence=float(score),
                    synthesized=True,
                )
                already_added.add(edge_key)
                n_synth += 1
                added += 1

        return G_aug, n_synth

    # ------------------------------------------------------------------
    # Dead-end detection
    # ------------------------------------------------------------------

    def _detect_dead_ends(
        self,
        paths: list,
        G: nx.Graph,
    ) -> List[Tuple[str, object]]:
        """
        Return (tail_node, path) pairs that look like dead-ends.

        Criteria:
          - path.hop_depth >= 1  (at least one hop was made)
          - path.score >= min_path_score
          - G.degree(tail) <= dead_end_max_degree
          - tail is not the seed (nodes[0])
        """
        seen_dead_ends: Set[str] = set()
        result: List[Tuple[str, object]] = []

        # Sort by score descending so we process the strongest signals first
        sorted_paths = sorted(paths, key=lambda p: getattr(p, "score", 0.0), reverse=True)

        for path in sorted_paths:
            tail = getattr(path, "tail", "")
            hop_depth = getattr(path, "hop_depth", 0)
            score = getattr(path, "score", 0.0)

            if not tail or hop_depth < 1:
                continue
            if score < self.min_path_score:
                continue
            if tail in seen_dead_ends:
                continue
            if not G.has_node(tail):
                continue
            if G.degree(tail) > self.dead_end_max_degree:
                continue

            seen_dead_ends.add(tail)
            result.append((tail, path))

        return result

    # ------------------------------------------------------------------
    # Candidate generation and scoring
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        dead_end: str,
        path: object,
        G: nx.Graph,
    ) -> List[Tuple[str, float]]:
        """
        Find and rank candidate nodes to connect to dead_end.

        Scoring:
          combined = embedding_sim(dead_end, candidate)
                   * prior_score(path)
                   [* kge_score(dead_end, candidate)] if kge_engine provided
        """
        dead_emb = self.adapter.get_embedding(dead_end)
        if dead_emb is None:
            return []

        # Restrict search to community neighbourhood
        cid = self.adapter.get_community(dead_end)
        pool: List[str] = list(self._community_members.get(cid, []))

        # Also include adjacent-community members if pool is small
        if len(pool) < self.max_candidates:
            cmap = getattr(self.adapter, "community_map", {}) or {}
            # Pull nodes from nearby communities (rough proximity: shared neighbours)
            for nbr in list(G.neighbors(dead_end)):
                nbr_cid = cmap.get(nbr, -1)
                if nbr_cid >= 0 and nbr_cid != cid:
                    pool.extend(self._community_members.get(nbr_cid, []))
            pool = list(dict.fromkeys(pool))  # deduplicate, preserve order

        # Filter: no self, no already-connected, no seed
        existing_nbrs: Set[str] = set(G.neighbors(dead_end))
        seed = getattr(path, "nodes", [""])[0] if getattr(path, "nodes", None) else ""
        pool = [
            n for n in pool
            if n != dead_end and n not in existing_nbrs and n != seed
        ][:self.max_candidates]

        if not pool:
            return []

        # Relation prior weight for this path
        prior_score = 1.0
        if self.relation_prior is not None:
            try:
                prior_score = float(self.relation_prior.score(path))
            except Exception:
                prior_score = 0.5
        prior_score = max(prior_score, 0.01)  # guard against zero

        # Embedding similarities
        scored: List[Tuple[str, float]] = []
        for candidate in pool:
            cand_emb = self.adapter.get_embedding(candidate)
            if cand_emb is None:
                continue
            sim = _cosine_sim(dead_emb, cand_emb)
            combined = sim * prior_score
            # Optionally multiply in KGE plausibility
            if self.kge_engine is not None:
                kge_score = self._kge_score(dead_end, candidate)
                combined *= max(kge_score, 0.01)
            scored.append((candidate, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _kge_score(self, head: str, tail: str) -> float:
        """
        Return a rank-based plausibility score in [0, 1] from the KGE engine.

        Uses ``predict_links`` to get the top-20 candidates and converts the
        candidate's rank to a score via ``1 / (1 + rank)``:
          rank 0 (best) → 1.0,  rank 1 → 0.5,  rank 9 → 0.1.

        Returns 0.5 (neutral) if the engine is unavailable or raises.
        Returns 0.0 if the candidate does not appear in the top-20.
        """
        if self.kge_engine is None:
            return 0.5
        try:
            if not hasattr(self.kge_engine, "predict_links"):
                return 0.5
            predictions = self.kge_engine.predict_links(head_entity=head, top_k=20)
            for rank, (h, r, t, _score) in enumerate(predictions):
                if t == tail:
                    return 1.0 / (1.0 + rank)  # rank 0 → 1.0, rank 1 → 0.5, …
            return 0.0  # not in top-20 predictions
        except Exception:
            return 0.5

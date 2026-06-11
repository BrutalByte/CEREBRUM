"""
Phase 236: Training-Free Path Schema Prediction.

A PathSchemaIndex enumerates all 2-hop relation schemas (r1, r2) present in a
knowledge graph and embeds them with a sentence encoder.  At query time, cosine
similarity between the question embedding and schema embeddings predicts which
relation path the question is asking about.  The predicted schemas are then
executed as targeted 2-hop traversals — a second reasoning channel that runs
in parallel to the beam and adds high-precision answer candidates the beam may
have pruned.

This is CEREBRUM's first *predictive* reasoning signal: every prior signal
(TRB, community hypothesis, relation-name index) steers or re-ranks AFTER the
beam has already traversed.  PathSchemaIndex forms a goal-directed hypothesis
*before* any traversal and acts on it directly.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


def _rel_to_text(rel: str) -> str:
    """Freebase last-segment tokenization: 'person.person.place_of_birth' → 'place of birth'."""
    return rel.split(".")[-1].replace("_", " ")


class PathSchemaIndex:
    """
    Index of 2-hop relation schemas (r1, r2) with dense embedding lookup.

    Build once per graph load; build time dominated by sentence encoding.
    Per-query cost: one filtered matrix-vector product + O(top_k × fan_out) lookups.

    Key design: predict_schemas_for_seed() filters candidates to schemas whose r1
    is actually present as an outgoing relation from the seed entity before ranking
    by cosine similarity.  Without this filter, the top-ranked schema may be
    semantically similar but structurally inapplicable (the seed has no r1 edge).
    """

    def __init__(self) -> None:
        self._schemas:       List[Tuple[str, str]]  = []
        self._schema_matrix: np.ndarray              = np.empty((0, 1), dtype=np.float16)
        self._counts:        np.ndarray              = np.empty((0,),   dtype=np.int32)
        self._by_r1:         Dict[str, List[int]]   = {}  # r1 → sorted list of schema indices

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        adapter,
        embedding_engine,
        min_count:       int                    = 3,
        skip_relations:  Optional[FrozenSet[str]] = None,
    ) -> "PathSchemaIndex":
        """
        Enumerate 2-hop schemas from the graph and embed them.

        Args:
            adapter:          NetworkXAdapter (or any adapter exposing ._G)
            embedding_engine: EmbeddingEngine with encode(List[str]) → np.ndarray
            min_count:        discard schemas that appear fewer than this many times
            skip_relations:   relation types to exclude from r1 and r2
        """
        skip = skip_relations or frozenset()
        G    = adapter.to_networkx()

        # Enumerate (r1, r2) pairs via intermediate nodes.
        # For each node B: count all (r1, r2) pairs where A→(r1)→B→(r2)→C.
        schema_counts: Counter = Counter()

        for mid in G.nodes():
            in_rels: List[str] = []
            for _, _, data in G.in_edges(mid, data=True):
                r = data.get("relation", "RELATED_TO")
                if r not in skip:
                    in_rels.append(r)

            if not in_rels:
                continue

            out_rels: List[str] = []
            for _, _, data in G.out_edges(mid, data=True):
                r = data.get("relation", "RELATED_TO")
                if r not in skip:
                    out_rels.append(r)

            if not out_rels:
                continue

            for r1 in in_rels:
                for r2 in out_rels:
                    schema_counts[(r1, r2)] += 1

        # Filter by min_count
        filtered = [(schema, cnt) for schema, cnt in schema_counts.items()
                    if cnt >= min_count]
        if not filtered:
            return self

        schemas, counts = zip(*filtered)

        # Embed schema texts as last-segment tokenized strings
        schema_texts = [f"{_rel_to_text(r1)} {_rel_to_text(r2)}" for r1, r2 in schemas]
        vecs = embedding_engine.encode(schema_texts).astype(np.float16)

        # L2-normalise in float32 precision, then store as float16
        vecs_f32 = vecs.astype(np.float32)
        norms    = np.linalg.norm(vecs_f32, axis=1, keepdims=True)
        norms    = np.where(norms < 1e-9, 1.0, norms)
        vecs_f32 /= norms

        self._schemas       = list(schemas)
        self._schema_matrix = vecs_f32.astype(np.float16)
        self._counts        = np.array(counts, dtype=np.int32)

        # Build r1 → [schema indices] lookup for seed-filtered prediction
        by_r1: Dict[str, List[int]] = {}
        for idx_i, (r1, _) in enumerate(self._schemas):
            by_r1.setdefault(r1, []).append(idx_i)
        self._by_r1 = by_r1
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def predict_schemas(
        self,
        question_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Return top-k (r1, r2, score) schemas most similar to the question embedding.

        The question_embedding must be L2-normalised (as produced by encode_one/encode_query).
        Returns an empty list when the index is empty or the embedding is zero.
        """
        if not self._schemas:
            return []

        qvec = question_embedding.astype(np.float32)
        norm = float(np.linalg.norm(qvec))
        if norm < 1e-9:
            return []
        qvec /= norm

        scores: np.ndarray = self._schema_matrix.astype(np.float32) @ qvec
        k = min(top_k, len(self._schemas))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [(self._schemas[i][0], self._schemas[i][1], float(scores[i]))
                for i in top_idx]

    def predict_schemas_for_seed(
        self,
        question_embedding: np.ndarray,
        seed_outgoing_rels: Set[str],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Like predict_schemas, but restricts candidates to schemas whose r1 is present
        as an outgoing relation from the seed entity.

        This eliminates the common failure mode where a semantically similar schema
        (e.g., 'library.collection.language / language.countries_spoken_in') predicts
        the language hop-2 correctly but expects a non-existent hop-1 relation from the
        seed (because the seed IS the language, not a collection that links to it).

        Args:
            question_embedding:  L2-normalised query vector
            seed_outgoing_rels:  set of relation types the seed entity has as outgoing edges
            top_k:               how many schemas to return
        """
        if not self._schemas or not seed_outgoing_rels:
            return []

        # Collect all schema indices whose r1 exists on the seed
        candidate_idx: List[int] = []
        for r1 in seed_outgoing_rels:
            candidate_idx.extend(self._by_r1.get(r1, []))

        if not candidate_idx:
            return []

        qvec = question_embedding.astype(np.float32)
        norm = float(np.linalg.norm(qvec))
        if norm < 1e-9:
            return []
        qvec /= norm

        # Rank candidates by cosine similarity
        cand = np.array(candidate_idx, dtype=np.int32)
        scores: np.ndarray = self._schema_matrix[cand].astype(np.float32) @ qvec
        k = min(top_k, len(cand))
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local  = top_local[np.argsort(-scores[top_local])]

        return [(self._schemas[cand[i]][0], self._schemas[cand[i]][1], float(scores[i]))
                for i in top_local]

    def predict_schemas_conditional(
        self,
        question_embedding: np.ndarray,
        seed_outgoing_rels: Set[str],
        hop1_r2_rels: Set[str],
        top_k: int = 5,
        structural_bonus: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """
        Phase 247: Condition r2 selection on actual hop-1 structural evidence.

        After the beam completes hop-1 and populates _expansion_cache, the caller
        builds hop1_r2_rels = union of outgoing relation types from all hop-1
        intermediates. This method re-ranks the r1-filtered schema candidates by:

            combined_score = cosine_sim × (1 + structural_bonus  if r2 ∈ hop1_r2_rels
                                           else cosine_sim)

        Schemas whose r2 is confirmed present on actual hop-1 intermediates receive
        a boost, biasing selection toward paths that are both semantically appropriate
        AND structurally reachable from this specific seed.
        """
        if not self._schemas or not seed_outgoing_rels:
            return []

        candidate_idx: List[int] = []
        for r1 in seed_outgoing_rels:
            candidate_idx.extend(self._by_r1.get(r1, []))

        if not candidate_idx:
            return []

        qvec = question_embedding.astype(np.float32)
        norm = float(np.linalg.norm(qvec))
        if norm < 1e-9:
            return []
        qvec /= norm

        cand = np.array(candidate_idx, dtype=np.int32)
        cosine_scores: np.ndarray = self._schema_matrix[cand].astype(np.float32) @ qvec

        # Structural confirmation: r2 is actually present on a hop-1 intermediate
        structural_mask = np.array(
            [float(self._schemas[cand[i]][1] in hop1_r2_rels) for i in range(len(cand))],
            dtype=np.float32,
        )
        combined = cosine_scores * (1.0 + structural_bonus * structural_mask)

        k = min(top_k, len(cand))
        top_local = np.argpartition(-combined, k - 1)[:k]
        top_local  = top_local[np.argsort(-combined[top_local])]

        return [(self._schemas[cand[i]][0], self._schemas[cand[i]][1], float(combined[i]))
                for i in top_local]

    def execute_schemas(
        self,
        seed_entity:    str,
        schemas:        List[Tuple[str, str, float]],
        adapter,
        skip_rels:      Optional[FrozenSet[str]] = None,
        max_hop1:       int = 2000,
    ) -> List[Tuple[str, float]]:
        """
        Execute each (r1, r2) schema as a 2-hop traversal from seed_entity.

        Returns a deduplicated list of (answer_entity, schema_score), excluding:
        - seed_entity itself
        - intermediate hop-1 nodes (relay nodes)
        - Freebase MID-style relay nodes (/m/... or m.)

        Highest schema_score wins when the same entity appears in multiple schemas.
        """
        skip = skip_rels or frozenset()
        seen_intermediates: Set[str] = set()
        results: Dict[str, float] = {}  # entity_id → best schema_score

        for r1, r2, schema_score in schemas:
            # Hop 1: seed → r1 → intermediates
            hop1_edges = adapter.get_neighbors(
                seed_entity, edge_types=[r1], max_neighbors=max_hop1
            )
            intermediates = [
                e.target_id for e in hop1_edges
                if e.target_id != seed_entity
            ]
            seen_intermediates.update(intermediates)

            for mid in intermediates:
                # Hop 2: mid → r2 → answers
                hop2_edges = adapter.get_neighbors(
                    mid, edge_types=[r2], max_neighbors=max_hop1
                )
                for e2 in hop2_edges:
                    eid = e2.target_id
                    if eid == seed_entity:
                        continue
                    if eid in seen_intermediates:
                        continue
                    # Best score wins (list may contain multiple schemas)
                    if schema_score > results.get(eid, -1.0):
                        results[eid] = schema_score

        # Return sorted by schema score descending
        return sorted(results.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def schema_count(self) -> int:
        return len(self._schemas)

    def top_schemas_by_count(self, n: int = 20) -> List[Tuple[str, str, int]]:
        """Return the n most frequent (r1, r2, count) schemas — useful for diagnostics."""
        order = np.argsort(-self._counts)[:n]
        return [(self._schemas[i][0], self._schemas[i][1], int(self._counts[i]))
                for i in order]

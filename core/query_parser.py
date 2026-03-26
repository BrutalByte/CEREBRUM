"""
QueryParser — CEREBRUM native language layer (Phase 18b).

Parses a natural language question into a structured graph query
(seed entity + relation hints + hop depth) using only the knowledge
graph's own vocabulary.  No external LLM, no training data, no internet.

How it works
------------
1. Entity linking
   The question is embedded with the same EmbeddingEngine used by the
   graph.  Cosine similarity against all entity embeddings finds the
   best-matching seed entity.  Falls back to exact / fuzzy string match
   when embeddings are unavailable.

2. Relation extraction
   The graph's relation type vocabulary is extracted at init time.
   Each relation type is lowercased and split into keywords.  The
   question is scanned for those keywords and scored by overlap.
   The highest-scoring relation types become hints passed to the
   beam traversal.

3. Hop inference
   Temporal markers ("through", "via", "path", "indirectly") suggest
   multi-hop.  Direct phrasing ("what does X do", "who is") suggests
   1-hop.  Default is 2.

Usage
-----
    from core.query_parser import QueryParser
    from adapters.networkx_adapter import NetworkXAdapter
    from core.embedding_engine import RandomEngine

    adapter = NetworkXAdapter.from_csv("tests/fixtures/toy_graph.csv")
    engine  = RandomEngine(dim=64)
    parser  = QueryParser(adapter, engine)

    q = parser.parse("What did newton influence?")
    print(q.seed_entity_id)       # "newton"
    print(q.relation_hints)       # ["INFLUENCED"]
    print(q.hop_hint)             # 1
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Relation keyword map
# Maps question keywords → candidate relation types.
# Built from the graph's actual vocabulary at runtime; this table provides
# the seed hints that bootstrap the vocabulary scan.
# ---------------------------------------------------------------------------

_KEYWORD_SEEDS: Dict[str, List[str]] = {
    "influenc":     ["INFLUENCED"],
    "caus":         ["CAUSES", "CAUSED_BY"],
    "prevent":      ["PREVENTS"],
    "treat":        ["TREATS"],
    "inhibit":      ["INHIBITS"],
    "activat":      ["ACTIVATES"],
    "encod":        ["ENCODES"],
    "express":      ["EXPRESSED_IN"],
    "bind":         ["BINDS"],
    "interact":     ["INTERACTS_WITH"],
    "wrote":        ["WROTE", "AUTHORED"],
    "written":      ["WROTE", "WRITTEN_BY"],
    "author":       ["AUTHORED"],
    "publish":      ["PUBLISHED"],
    "found":        ["FOUNDED"],
    "discover":     ["DISCOVERED"],
    "invent":       ["INVENTED"],
    "proved":       ["PROVED"],
    "disprove":     ["DISPROVED"],
    "refut":        ["REFUTES"],
    "support":      ["SUPPORTS"],
    "cit":          ["CITED"],
    "contribut":    ["CONTRIBUTED_TO"],
    "studi":        ["STUDIED"],
    "know":         ["KNOWS"],
    "married":      ["MARRIED_TO"],
    "work":         ["WORKED_WITH"],
    "collaborat":   ["COLLABORATED_WITH"],
    "employ":       ["EMPLOYED_BY"],
    "mentor":       ["MENTORED"],
    "student":      ["STUDENT_OF"],
    "correspond":   ["CORRESPONDED_WITH"],
    "oppos":        ["OPPOSED"],
    "part of":      ["PART_OF"],
    "member":       ["MEMBER_OF"],
    "located":      ["LOCATED_IN"],
    "precede":      ["PRECEDES"],
    "follow":       ["FOLLOWS"],
    "starred":      ["STARRED_IN"],
    "direct":       ["DIRECTED", "DIRECTED_BY"],
    "produced":     ["PRODUCED_BY"],
    "genre":        ["HAS_GENRE"],
    "acted":        ["STARRED_IN"],
    "appear":       ["STARRED_IN"],
    "cast":         ["STARRED_IN"],
    "film":         ["STARRED_IN", "DIRECTED"],
    "movie":        ["STARRED_IN", "DIRECTED"],
}

# Question words that suggest multi-hop reasoning
_MULTI_HOP_MARKERS = frozenset([
    "through", "via", "indirectly", "path", "chain", "connection",
    "link", "eventually", "ultimately", "how many hops", "trace",
    "bridge", "between",
])

# Question types
_QUESTION_WORDS = {
    "who":   "person",
    "what":  "thing",
    "where": "place",
    "when":  "time",
    "why":   "reason",
    "how":   "method",
    "which": "thing",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Structured graph query extracted from a natural language question."""

    raw_question: str
    """The original question string."""

    seed_entity_id: Optional[str]
    """Best-matched entity ID in the graph, or None if not found."""

    seed_entity_label: str
    """Human-readable label of the matched entity."""

    seed_entity_score: float
    """Similarity score of the entity match (0.0–1.0)."""

    relation_hints: List[str]
    """Relation types the question appears to be asking about."""

    hop_hint: int
    """Inferred traversal depth (1, 2, or 3)."""

    question_type: str
    """Detected question word type: 'person', 'thing', 'place', etc."""

    candidates: List[Tuple[str, str, float]] = field(default_factory=list)
    """Top-5 entity candidates: [(entity_id, label, score), ...]"""


# ---------------------------------------------------------------------------
# QueryParser
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Parses natural language questions into structured graph queries.

    Parameters
    ----------
    adapter : GraphAdapter
        The loaded knowledge graph.  Used for entity label lookup and
        relation type vocabulary extraction.
    embedding_engine : EmbeddingEngine
        Used to embed the question for semantic entity linking.
        Should be the same engine used to build the graph's embeddings.
    top_k_entities : int
        Number of candidate entities to return alongside the best match.
    min_entity_score : float
        Minimum cosine similarity to accept an entity match (0.0–1.0).
    """

    def __init__(
        self,
        adapter,
        embedding_engine,
        top_k_entities: int = 5,
        min_entity_score: float = 0.10,
    ) -> None:
        self._adapter        = adapter
        self._engine         = embedding_engine
        self._top_k          = top_k_entities
        self._min_score      = min_entity_score
        self._rel_vocab      = self._build_relation_vocab()
        self._entity_index   = None  # built lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, question: str) -> ParsedQuery:
        """
        Parse a natural language question into a ParsedQuery.

        Parameters
        ----------
        question : str
            Free-text question, e.g. "What did Newton influence?"

        Returns
        -------
        ParsedQuery
            seed_entity_id is None if no entity could be matched.
        """
        q_lower = question.lower().strip()

        entity_id, entity_label, score, candidates = self._link_entity(question)
        relations  = self._extract_relations(q_lower)
        hop        = self._infer_hops(q_lower)
        qtype      = self._question_type(q_lower)

        return ParsedQuery(
            raw_question=question,
            seed_entity_id=entity_id,
            seed_entity_label=entity_label,
            seed_entity_score=score,
            relation_hints=relations,
            hop_hint=hop,
            question_type=qtype,
            candidates=candidates,
        )

    def relation_vocabulary(self) -> List[str]:
        """Return all relation types found in the graph."""
        seen = set()
        result = []
        for rel_types in self._rel_vocab.values():
            for rt in rel_types:
                if rt not in seen:
                    seen.add(rt)
                    result.append(rt)
        return result

    # ------------------------------------------------------------------
    # Entity linking
    # ------------------------------------------------------------------

    def _link_entity(
        self,
        question: str,
    ) -> Tuple[Optional[str], str, float, List[Tuple[str, str, float]]]:
        """
        Find the graph entity best matching the question.

        Strategy:
        1. Extract noun phrases / capitalised tokens as candidate strings.
        2. Try exact label match against graph entities.
        3. Semantic search: embed question, cosine-compare against entity
           embeddings.  Return top-k candidates.
        """
        # Step 1: try exact / near-exact match on capitalised tokens
        exact_id, exact_label, exact_score = self._exact_match(question)
        if exact_score >= 0.95:
            return exact_id, exact_label, exact_score, [(exact_id, exact_label, exact_score)]

        # Step 2: semantic search via embeddings
        sem_results = self._semantic_search(question)

        # Merge: prefer exact if score is competitive
        if exact_id and sem_results:
            best_sem_score = sem_results[0][2]
            if exact_score >= best_sem_score * 0.8:
                # Exact match is close enough — put it first
                sem_results = [(exact_id, exact_label, exact_score)] + [
                    r for r in sem_results if r[0] != exact_id
                ]

        if not sem_results:
            return None, "", 0.0, []

        best_id, best_label, best_score = sem_results[0]
        if best_score < self._min_score:
            return None, "", best_score, sem_results

        return best_id, best_label, best_score, sem_results[:self._top_k]

    def _exact_match(self, question: str) -> Tuple[Optional[str], str, float]:
        """
        Try to find an entity whose label appears verbatim in the question.
        Handles mixed case.  Returns (id, label, score) or (None, '', 0.0).
        """
        q_lower = question.lower()
        best_id    = None
        best_label = ""
        best_len   = 0

        try:
            # find_entities returns top matches by string similarity
            results = self._adapter.find_entities(question, top_k=10)
            for ent in results:
                lbl_lower = ent.label.lower()
                if lbl_lower in q_lower and len(lbl_lower) > best_len:
                    best_id    = ent.id
                    best_label = ent.label
                    best_len   = len(lbl_lower)
        except Exception:
            pass

        if best_id:
            # Score by label length relative to question length (longer = more specific)
            score = min(1.0, best_len / max(len(q_lower.split()), 1) * 0.9 + 0.5)
            return best_id, best_label, score

        return None, "", 0.0

    def _semantic_search(
        self,
        question: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Embed the question and compute cosine similarity against entity
        embeddings stored on the adapter.

        Returns list of (entity_id, label, similarity) sorted descending.
        """
        embeddings = getattr(self._adapter, "embeddings", None)
        if not embeddings:
            return []

        # Embed the question (single vector)
        q_vec = self._engine.encode_one(question).astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-8:
            return []
        q_vec = q_vec / q_norm

        # Build entity matrix lazily
        if self._entity_index is None:
            self._entity_index = self._build_entity_index(embeddings)

        ids, labels, matrix = self._entity_index
        # Cosine similarity: matrix rows are already normalised
        sims = matrix @ q_vec  # shape (N,)

        top_k = min(self._top_k, len(ids))
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim >= self._min_score:
                results.append((ids[idx], labels[idx], sim))
        return results

    def _build_entity_index(
        self,
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Build a normalised float32 matrix of entity embeddings for fast
        cosine search.  Cached after first build.
        """
        ids    = list(embeddings.keys())
        vecs   = np.array([embeddings[i].astype(np.float32) for i in ids])
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        vecs   = vecs / norms

        labels = []
        for eid in ids:
            try:
                ent = self._adapter.get_entity(eid)
                labels.append(ent.label if ent and ent.label else eid)
            except Exception:
                labels.append(eid)

        return ids, labels, vecs

    # ------------------------------------------------------------------
    # Relation extraction
    # ------------------------------------------------------------------

    def _build_relation_vocab(self) -> Dict[str, List[str]]:
        """
        Extract all relation types from the graph and build a keyword
        → [relation_type] mapping.

        Falls back to _KEYWORD_SEEDS when the graph cannot provide
        relation types.
        """
        vocab: Dict[str, List[str]] = dict(_KEYWORD_SEEDS)

        try:
            # Collect relation types by sampling edges from the adapter
            G = self._adapter.to_networkx()
            rel_types = set()
            for _, _, data in G.edges(data=True):
                rt = data.get("relation_type") or data.get("relation") or ""
                if rt:
                    rel_types.add(rt.upper())

            for rt in rel_types:
                # Break CamelCase and underscore-separated into keywords
                words = re.sub(r"([A-Z])", r" \1", rt).lower().split()
                words += rt.lower().split("_")
                for w in words:
                    stem = w[:6]  # crude stem
                    if stem not in vocab:
                        vocab[stem] = []
                    if rt not in vocab[stem]:
                        vocab[stem].append(rt)
        except Exception:
            pass

        return vocab

    def _extract_relations(self, q_lower: str) -> List[str]:
        """
        Score each relation type in the vocabulary against the question.
        Returns top relation types in score order.
        """
        scores: Dict[str, float] = {}
        for keyword, rel_types in self._rel_vocab.items():
            if keyword in q_lower:
                for rt in rel_types:
                    scores[rt] = scores.get(rt, 0.0) + 1.0

        if not scores:
            return []

        ranked = sorted(scores, key=lambda r: scores[r], reverse=True)
        return ranked[:5]

    # ------------------------------------------------------------------
    # Hop inference
    # ------------------------------------------------------------------

    def _infer_hops(self, q_lower: str) -> int:
        """
        Infer traversal depth from question phrasing.

        - 1-hop: direct questions ("who is", "what does X do")
        - 2-hop: default for most multi-part questions
        - 3-hop: explicit multi-hop markers ("through", "via", "path")
        """
        tokens = set(q_lower.split())
        if tokens & _MULTI_HOP_MARKERS:
            return 3
        # Explicit numbers
        if "two hop" in q_lower or "2-hop" in q_lower or "2 hop" in q_lower:
            return 2
        if "three hop" in q_lower or "3-hop" in q_lower or "3 hop" in q_lower:
            return 3
        # Short direct questions (≤6 words) suggest 1-hop
        if len(q_lower.split()) <= 6:
            return 1
        return 2

    def _question_type(self, q_lower: str) -> str:
        for word, qtype in _QUESTION_WORDS.items():
            if q_lower.startswith(word + " ") or q_lower.startswith(word + "'"):
                return qtype
        return "thing"

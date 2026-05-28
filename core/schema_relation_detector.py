"""
SchemaAwareRelationDetector — Phase 201

Zero-shot, schema-derived relation detection for multi-hop KGQA.

Replaces keyword-based detect_target_relation() with cosine similarity between
question sub-phrases and KB relation phrase embeddings. Works on any knowledge
graph — MetaQA's 9 relations, FreeBase's 6,000+, or any domain KB. No
hardcoded relation names, no domain-specific templates.

Algorithm
---------
Build time (once per KB load):
  - Collect all relation type strings from the graph adapter.
  - Convert each to a natural-language phrase ("directed_by" → "directed by").
  - Encode all phrases with the provided EmbeddingEngine.
  - Store as an [N × dim] float32 matrix.

Query time (per question):
  - detect_terminal():  embed text AFTER entity bracket (or full question
                        with entity stripped) → cosine match → R3 (answer
                        relation / TRB target).
  - detect_initial():   embed `window` words BEFORE entity bracket → cosine
                        match → R1 (bridge relation / FHRB target).

Usage
-----
    srd = SchemaAwareRelationDetector()
    srd.build(adapter.get_edge_types(), embedding_engine)

    r1 = srd.detect_initial(question)   # bridge relation (first hop)
    r2 = srd.detect_terminal(question)  # answer relation  (terminal hop)
    r1, r2 = srd.detect_path(question)  # both at once
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.embedding_engine import EmbeddingEngine


_ENTITY_RE = re.compile(r"\[([^\]]+)\]")


class SchemaAwareRelationDetector:
    """
    Embedding-based, KB-agnostic relation detector for multi-hop KGQA.

    Build once at KB load time; query once per question. Thread-safe after
    build() completes (no mutable state during detect_* calls).
    """

    def __init__(self) -> None:
        self._relation_names: List[str] = []
        self._rel_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._built: bool = False
        self._embedding_engine: Optional["EmbeddingEngine"] = None

    # ------------------------------------------------------------------
    # Build-time
    # ------------------------------------------------------------------

    @staticmethod
    def _to_phrase(rel: str) -> str:
        return rel.replace("_", " ")

    def build(self, relation_types: List[str], embedding_engine: "EmbeddingEngine") -> None:
        """
        Precompute relation phrase embeddings.

        Parameters
        ----------
        relation_types   : list of relation names from the KB schema
        embedding_engine : EmbeddingEngine (SentenceEngine in production)
        """
        if not relation_types or embedding_engine is None:
            return
        self._embedding_engine = embedding_engine
        phrases = [self._to_phrase(r) for r in relation_types]
        try:
            vecs = embedding_engine.encode(phrases)
        except Exception:
            return
        vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._rel_matrix = vecs / norms
        self._relation_names = list(relation_types)
        self._built = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a short text fragment; returns L2-normalised float32 vector."""
        if not self._built or not text.strip():
            return None
        try:
            fn = getattr(self._embedding_engine, "encode_query", None)
            vecs = fn([text]) if fn is not None else self._embedding_engine.encode([text])
            v = np.array(vecs[0], dtype=np.float32).flatten()
            norm = np.linalg.norm(v)
            if norm < 1e-9:
                return None
            return v / norm
        except Exception:
            return None

    def _top_match(
        self,
        query_vec: np.ndarray,
        exclude: Optional[str] = None,
        min_gap: float = 0.0,
    ) -> Optional[str]:
        """Return best-matching relation or None if top1−top2 gap < min_gap."""
        rel, gap = self._top_match_with_gap(query_vec, exclude=exclude)
        return rel if gap >= min_gap else None

    def _top_match_with_gap(
        self,
        query_vec: np.ndarray,
        exclude: Optional[str] = None,
    ) -> Tuple[Optional[str], float]:
        """Return (best_relation, top1−top2_gap). gap=-1 when matrix empty."""
        if self._rel_matrix.shape[0] == 0:
            return None, -1.0
        sims = self._rel_matrix @ query_vec
        if exclude is not None and exclude in self._relation_names:
            sims = sims.copy()
            sims[self._relation_names.index(exclude)] = -2.0
        order = np.argsort(sims)[::-1]
        top1 = float(sims[order[0]])
        top2 = float(sims[order[1]]) if len(order) > 1 else top1
        return self._relation_names[int(order[0])], top1 - top2

    @staticmethod
    def _entity_bounds(question: str) -> Optional[Tuple[int, int]]:
        """Return (open_bracket, close_bracket) character positions."""
        m = _ENTITY_RE.search(question)
        return (m.start(), m.end()) if m else None

    # ------------------------------------------------------------------
    # Query-time detection
    # ------------------------------------------------------------------

    # WH-question starters that put the answer type near the beginning.
    # This is a universal property of English questions, not domain-specific.
    _WH_STARTERS = frozenset({"what", "who", "which", "whose", "whom"})

    # Temporal question starters: "when" signals a date/year/time relation.
    # Using just the starter word isolates the temporal signal from surrounding
    # content words that describe the traversal path (not the answer type).
    _TEMPORAL_STARTERS = frozenset({"when"})

    def detect_terminal(
        self,
        question: str,
        exclude: Optional[str] = None,
        min_gap: float = 0.0,
        n_head: int = 5,
        n_tail: int = 4,
    ) -> Optional[str]:
        """
        Detect the terminal (answer) relation from the question.

        Strategy:
        - For WH-questions ("what/who/which...") the answer type is near the
          start: use the head n_head words.
        - Otherwise: compare head vs tail of the entity-stripped question and
          return whichever has the higher top1−top2 confidence gap.

        Parameters
        ----------
        question  : raw question string (entity in square brackets)
        exclude   : relation to skip (e.g. already assigned as R1)
        min_gap   : minimum top1−top2 similarity gap to fire
        n_head    : words from start of question to embed as head candidate
        n_tail    : words from end of question to embed as tail candidate
        """
        if not self._built:
            return None
        bounds = self._entity_bounds(question)
        text = (
            (question[:bounds[0]] + question[bounds[1]:]).strip()
            if bounds is not None
            else question
        )
        words = text.split()
        if not words:
            return None
        first = words[0].lower()

        # Temporal question ("when..."): the answer is a date or year.
        # Embed "year" as a compact, KB-agnostic temporal proxy — it matches
        # release_year, publication_year, fiscal_year, etc. in any knowledge
        # graph without encoding any domain-specific relation name. The word
        # "when" itself maps poorly in BGE embedding space (false-positive to
        # in_language), while "year" gives a clear temporal signal (gap≈0.16).
        if first in self._TEMPORAL_STARTERS:
            t_vec = self._embed("year")
            if t_vec is None:
                return None
            return self._top_match(t_vec, exclude=exclude, min_gap=min_gap)

        # WH-question: answer type is near the beginning
        if first in self._WH_STARTERS:
            head_vec = self._embed(" ".join(words[:n_head]))
            if head_vec is None:
                return None
            return self._top_match(head_vec, exclude=exclude, min_gap=min_gap)

        # Non-WH: compare head vs tail confidence and return more confident
        head_vec = self._embed(" ".join(words[:n_head]))
        tail_vec = self._embed(" ".join(words[-n_tail:]))
        head_rel, head_gap = (
            self._top_match_with_gap(head_vec, exclude=exclude)
            if head_vec is not None else (None, -1.0)
        )
        tail_rel, tail_gap = (
            self._top_match_with_gap(tail_vec, exclude=exclude)
            if tail_vec is not None else (None, -1.0)
        )
        if tail_gap > head_gap:
            return tail_rel if tail_gap >= min_gap else None
        return head_rel if head_gap >= min_gap else None

    # Short words that carry no relation signal (function words / filler)
    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "do",
        "did", "does", "have", "has", "had", "in", "of", "to", "for",
        "on", "at", "by", "with", "as", "its", "or", "and", "not",
        "that", "this", "these", "those", "it", "he", "she", "they",
        "which", "who", "what", "where", "when", "how", "from",
    })

    def _content_words(self, tokens: List[str], min_len: int = 4) -> List[str]:
        """Return tokens that are long enough and not stopwords."""
        return [
            t for t in tokens
            if len(t) >= min_len and t.lower() not in self._STOP_WORDS
        ]

    def detect_initial(
        self,
        question: str,
        exclude: Optional[str] = None,
        min_gap: float = 0.0,
        scan_window: int = 8,
        suffix_max_words: int = 4,
        phrase_windows: Tuple[int, ...] = (4, 5),
    ) -> Optional[str]:
        """
        Detect the initial (bridge) relation connecting the seed entity to the
        first intermediate entity in a multi-hop path.

        Scans all content words in the prefix and compares them individually
        against relation embeddings. The content word with the highest
        top1−top2 confidence gap determines R1. This is robust to filler
        phrases (e.g. "with the film") that would corrupt a fixed-window
        approach.

        Also evaluates:
        - Multi-word phrases of size `phrase_windows` from the prefix end.
        - The first `suffix_max_words` words after the entity bracket.

        Parameters
        ----------
        question        : raw question string (entity in square brackets)
        exclude         : relation to skip (typically the detected R3)
        min_gap         : minimum confidence gap to fire (0 = always return)
        scan_window     : how many prefix words to include in the content scan
        suffix_max_words: max words after entity to include in suffix candidate
        phrase_windows  : multi-word phrase sizes to also evaluate from prefix
        """
        if not self._built:
            return None
        bounds = self._entity_bounds(question)
        if bounds is None:
            return None

        best_rel: Optional[str] = None
        best_gap: float = -1.0

        prefix_words = question[:bounds[0]].strip().split()
        scan_words   = prefix_words[-scan_window:] if len(prefix_words) > scan_window else prefix_words

        # --- single content-word scan (batch encode) ---
        content = self._content_words(scan_words)
        if content:
            try:
                fn = getattr(self._embedding_engine, "encode_query", None)
                vecs = fn(content) if fn is not None else self._embedding_engine.encode(content)
                vecs = np.array(vecs, dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1.0, norms)
                vecs = vecs / norms
            except Exception:
                vecs = np.empty((0, 0))
            if vecs.ndim == 2 and vecs.shape[0] > 0:
                exclude_idx = (
                    self._relation_names.index(exclude)
                    if exclude is not None and exclude in self._relation_names
                    else -1
                )
                for v in vecs:
                    sims = self._rel_matrix @ v
                    if exclude_idx >= 0:
                        sims = sims.copy()
                        sims[exclude_idx] = -2.0
                    order = np.argsort(sims)[::-1]
                    gap = float(sims[order[0]]) - float(sims[order[1]])
                    if gap > best_gap:
                        best_gap = gap
                        best_rel = self._relation_names[int(order[0])]

        # --- multi-word phrase candidates from prefix end ---
        for w in phrase_windows:
            if len(prefix_words) < w:
                continue
            text = " ".join(prefix_words[-w:])
            vec = self._embed(text)
            if vec is None:
                continue
            rel, gap = self._top_match_with_gap(vec, exclude=exclude)
            if gap > best_gap:
                best_rel, best_gap = rel, gap

        # --- suffix candidate (first few words after entity) ---
        suffix_words = question[bounds[1]:].strip().split()
        if suffix_words:
            suf_text = " ".join(suffix_words[:suffix_max_words])
            vec = self._embed(suf_text)
            if vec is not None:
                rel, gap = self._top_match_with_gap(vec, exclude=exclude)
                if gap > best_gap:
                    best_rel, best_gap = rel, gap

        return best_rel if best_gap >= min_gap else None

    def detect_path(
        self,
        question: str,
        min_gap: float = 0.0,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect (R1, R2) = (initial relation, terminal relation).

        Detects terminal first, then initial with terminal excluded, to
        reduce cross-contamination between the two signals.

        Returns (None, None) when the detector has not been built.
        """
        r2 = self.detect_terminal(question, min_gap=min_gap)
        r1 = self.detect_initial(question, exclude=r2, min_gap=min_gap)
        return r1, r2

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._built

    def relation_names(self) -> List[str]:
        return list(self._relation_names)

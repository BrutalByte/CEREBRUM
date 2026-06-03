"""
Engram-Steered Beam Traversal (Phase 55).

EngramTraversal extends BeamTraversal with a persistent relation-pattern
cache derived from previous successful reasoning paths.  After each query the
compressed Engram relation sequence is stored; on subsequent queries those
patterns bias the beam pruning step, effectively "caching" logical structure.

Design
------
Engram
    Thread-safe store:  relation_sequence_tuple → success_count.
    Records the relation-type sequences of *completed* high-confidence paths.
    Provides an affinity score (0..1) for any prefix sequence.

EngramTraversal(BeamTraversal)
    Identical to BeamTraversal except:
    - Between candidate expansion and beam pruning at each hop, each candidate
      path receives an affinity boost proportional to how well its emerging
      relation sequence matches cached patterns.
    - Boost formula: effective_score = score * (1 + engram_strength * affinity)
    - After traversal, the caller should call .record_answers(answers) to
      update the cache with successful paths from this query.

Usage
-----
    from reasoning.engram_traversal import EngramTraversal, Engram

    cache = Engram()
    traversal = EngramTraversal(adapter, csa, cache=cache, engram_strength=0.3)

    paths = traversal.traverse(seeds)
    answers = extract(paths, ...)
    traversal.record_answers(answers, min_score=0.5)   # feed cache
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reasoning.traversal import BeamTraversal, TraversalPath
from core.graph_adapter import GraphAdapter
from core.attention_engine import CSAEngine

_log = logging.getLogger("cerebrum.engram")


# ---------------------------------------------------------------------------
# Engram relation shorthand (mirrors EngramVerbalizer._SHORTHAND)
# ---------------------------------------------------------------------------

_SHORTHAND: Dict[str, str] = {
    "CAUSES": "!",
    "CAUSED_BY": "<-!",
    "TREATS": "+",
    "INHIBITS": "-",
    "STARRED_IN": "*",
    "DIRECTED_BY": "^",
    "RELEASE_YEAR": "@",
    "INFLUENCED": "~",
    "MEMBER_OF": "€",
    "PART_OF": "⊂",
    "REM_SYNTHESIZED": "≈",
}


def _compress_rel(rel: str) -> str:
    """Compress a relation type to its Engram shorthand, or keep as-is."""
    return _SHORTHAND.get(rel.upper(), rel)


def _path_rel_sequence(path: TraversalPath) -> Tuple[str, ...]:
    """
    Extract the ordered relation sequence from a TraversalPath.

    TraversalPath.nodes is  [entity, rel, entity, rel, entity, ...].
    Odd-indexed elements are relation types.
    """
    nodes = path.nodes
    return tuple(_compress_rel(nodes[i]) for i in range(1, len(nodes), 2))


# ---------------------------------------------------------------------------
# Engram
# ---------------------------------------------------------------------------

class Engram:
    """
    Thread-safe cache of relation-sequence patterns from successful paths.

    Internally stores:
        _counts: {rel_tuple -> int}   — raw success counts per full sequence
        _prefix: {rel_tuple -> int}   — aggregated counts for each prefix

    The prefix index allows fast O(depth) affinity lookup during traversal.
    """

    def __init__(self, max_patterns: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._prefix: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._max_patterns = max_patterns
        self._max_count: int = 1  # tracks maximum stored count for normalization

    def record(self, rel_sequence: Tuple[str, ...], weight: int = 1) -> None:
        """
        Record a successful relation sequence and all its prefixes.

        Parameters
        ----------
        rel_sequence : tuple of relation shorthand strings
        weight       : how many successes to credit (default 1)
        """
        if not rel_sequence:
            return
        with self._lock:
            # Evict oldest (least common) if over capacity
            if len(self._counts) >= self._max_patterns:
                min_key = min(self._counts, key=self._counts.__getitem__)
                old_seq = min_key
                del self._counts[old_seq]
                # Rebuild prefix map on eviction (rare)
                self._prefix.clear()
                for seq, cnt in self._counts.items():
                    for k in range(1, len(seq) + 1):
                        self._prefix[seq[:k]] += cnt

            self._counts[rel_sequence] += weight
            self._max_count = max(self._max_count, self._counts[rel_sequence])
            for k in range(1, len(rel_sequence) + 1):
                self._prefix[rel_sequence[:k]] += weight

    def affinity(self, rel_prefix: Tuple[str, ...]) -> float:
        """
        Return a score in [0, 1] indicating how well *rel_prefix* matches
        previously-cached patterns.

        Looks up the longest matching prefix in the cache.
        Returns 0.0 if no match; 1.0 for the single highest-count pattern.
        """
        if not rel_prefix or self._max_count == 0:
            return 0.0
        with self._lock:
            # Walk from longest to shortest prefix looking for any match
            for k in range(len(rel_prefix), 0, -1):
                cnt = self._prefix.get(rel_prefix[:k], 0)
                if cnt > 0:
                    # Normalize by max known count; scale by prefix length ratio
                    # (longer matches get higher affinity)
                    depth_bonus = k / len(rel_prefix)
                    return min(1.0, (cnt / self._max_count) * depth_bonus)
        return 0.0

    def size(self) -> int:
        """Return the number of distinct full sequences in the cache."""
        with self._lock:
            return len(self._counts)

    def top_patterns(self, n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """Return the n most frequent patterns as [(sequence, count)] list."""
        with self._lock:
            items = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
            return items[:n]

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._prefix.clear()
            self._max_count = 1

    # ------------------------------------------------------------------
    # Persistence — survives process restarts
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the cache to a JSON file.

        Serialises ``_counts`` as a list of [rel_sequence, count] pairs.
        The prefix index is derived and not stored (rebuilt on load).

        Parameters
        ----------
        path : file path to write (created with parents if needed)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "version":      1,
                "max_patterns": self._max_patterns,
                "counts":       [[list(seq), cnt] for seq, cnt in self._counts.items()],
            }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _log.info("Engram saved: %d patterns → %s", len(data["counts"]), p)

    @classmethod
    def load(cls, path: str) -> "Engram":
        """
        Load a previously saved cache from *path*.

        Returns an empty cache if the file does not exist.
        The prefix index is rebuilt from the stored counts.
        """
        p = Path(path)
        if not p.exists():
            _log.info("Engram: no file at %s — starting empty", p)
            return cls()
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        cache = cls(max_patterns=data.get("max_patterns", 1000))
        for seq_list, cnt in data.get("counts", []):
            seq = tuple(seq_list)
            # Bypass eviction logic during restore (counts are already stable)
            cache._counts[seq] = cnt
            cache._max_count = max(cache._max_count, cnt)
            for k in range(1, len(seq) + 1):
                cache._prefix[seq[:k]] += cnt
        _log.info("Engram loaded: %d patterns ← %s", len(cache._counts), p)
        return cache

    def save_if_path(self, path: Optional[str]) -> None:
        """Convenience: call save(path) only if path is not None."""
        if path:
            self.save(path)


# ---------------------------------------------------------------------------
# EngramTraversal
# ---------------------------------------------------------------------------

class EngramTraversal(BeamTraversal):
    """
    Beam traversal variant that uses an Engram to bias beam pruning.

    At each hop, before pruning candidates to beam_width, each candidate path
    receives a score boost:

        effective_score = path.score * (1 + engram_strength * affinity)

    where affinity ∈ [0, 1] is how well the candidate's relation prefix
    matches previously-cached patterns.  Paths with no matching pattern
    receive no boost and are pruned on their raw CSA score.

    Parameters
    ----------
    cache           : Engram instance (shared across queries for persistence)
    engram_strength : multiplicative boost ceiling (default 0.3 → max 30% boost)
    All other parameters are forwarded to BeamTraversal.
    """

    def __init__(
        self,
        *args,
        cache: Optional[Engram] = None,
        engram_strength: float = 0.3,
        fast_binding: bool = True,
        fast_novelty_threshold: float = 0.1,
        fast_score_threshold: float = 0.7,
        fast_weight: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache = cache or Engram()
        self.engram_strength = engram_strength
        self._fast_binding_engine = None
        if fast_binding:
            from reasoning.speedtalk_cache import FastBindingEngine
            self._fast_binding_engine = FastBindingEngine(
                engram=self.cache,
                novelty_threshold=fast_novelty_threshold,
                score_threshold=fast_score_threshold,
                fast_weight=fast_weight,
            )

    def _boosted_score(self, path: TraversalPath) -> float:
        """Compute the Engram-boosted effective score for beam pruning."""
        rel_seq = _path_rel_sequence(path)
        if not rel_seq:
            return path.score
        aff = self.cache.affinity(rel_seq)
        return path.score * (1.0 + self.engram_strength * aff)

    def _prune_candidates(
        self,
        candidates: List[TraversalPath],
        hop: int,
    ) -> List[TraversalPath]:
        """
        Engram-steered pruning: rank by Engram-boosted score instead of raw score.

        Candidates whose relation-sequence prefix matches a cached successful
        pattern receive a multiplicative boost proportional to the cache
        affinity, biasing the beam toward known-productive reasoning chains.
        At the terminal hop all candidates are returned (sorted) so the
        answer extractor retains the full frontier.
        """
        import heapq
        hop_bw = self._beam_widths.get(hop, self.beam_width)
        if hop < self.max_hop and len(candidates) > hop_bw:
            return heapq.nlargest(hop_bw, candidates, key=self._boosted_score)
        return sorted(candidates, key=self._boosted_score, reverse=True)

    def record_answers(
        self,
        answers,
        min_score: float = 0.3,
    ) -> None:
        """
        Feed successful answers back into the Engram.

        Call this after each query to accumulate relation patterns.

        Parameters
        ----------
        answers   : list of Answer objects (from reasoning.answer_extractor)
        min_score : only record answers with score >= this threshold
        """
        for ans in answers:
            if ans.score < min_score:
                continue
            path = getattr(ans, "best_path", None)
            if path is None:
                continue
            rel_seq = _path_rel_sequence(path)
            if rel_seq:
                # Weight by score so high-confidence paths influence more
                weight = max(1, int(ans.score * 10))
                self.cache.record(rel_seq, weight=weight)

                # Phase 219-A: fast binding for novel high-confidence paths
                if self._fast_binding_engine is not None:
                    existing_aff = self.cache.affinity(rel_seq)
                    if self._fast_binding_engine.evaluate(rel_seq, existing_aff, ans.score):
                        self._fast_binding_engine.bind(rel_seq)



"""
CandidateRegistry — TTL-Aware Candidate Deduplication with Nomination Boost (Phase 73 Batch B).

Replaces the flat ``_evaluated_pairs: Set[Tuple[str, str]]`` in ResearchAgent with
a richer structure that:

  1. Tracks how many times each (source, target) pair has been independently
     nominated across scan cycles.  A pair nominated by multiple mechanisms
     (ANN scan + structural hole + insight engine) is more likely to represent
     a genuine missing link.

  2. Enforces TTL-gated re-evaluation.  After a pair has been through the
     HypothesisEngine, it is blocked until its TTL expires.  This prevents
     wasted computation while still allowing the agent to revisit pairs after
     the graph has evolved sufficiently.

  3. Applies a log-scale ``nomination_boost`` multiplier to ``discovery_potential``
     so multi-nominated candidates rise to the top of the priority queue.

Nomination boost formula
------------------------
  boost = min(1.0 + 0.5 * log2(nomination_count), nomination_boost_cap)

  nomination_count  |  boost
  ------------------|--------
        1           |  1.00 ×
        2           |  1.50 ×
        4           |  2.00 ×
        8           |  2.50 ×
       16+          |  3.00 ×  (capped)

Memory management
-----------------
When the number of entries exceeds ``max_entries``, the oldest entries (by
``last_seen``) are evicted via a deque-based LRU approach.  ``prune()`` provides
an explicit age-based eviction pass for long-running deployments.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """
    Per-(source, target) pair record in the CandidateRegistry.
    """

    source_id: str
    target_id: str

    first_seen: float = field(default_factory=time.time)
    """Timestamp of the first nomination."""

    last_seen: float = field(default_factory=time.time)
    """Timestamp of the most recent nomination."""

    last_evaluated: Optional[float] = None
    """Timestamp of the last HypothesisEngine run on this pair.  None = never evaluated."""

    nomination_count: int = 1
    """Number of times this pair has been independently nominated."""

    ttl: float = 3600.0
    """Seconds to block re-evaluation after ``last_evaluated``."""


# ---------------------------------------------------------------------------
# CandidateRegistry
# ---------------------------------------------------------------------------

class CandidateRegistry:
    """
    TTL-aware registry for ResearchAgent candidate deduplication.

    Parameters
    ----------
    default_ttl
        Seconds to block re-evaluation after a pair has been through the
        HypothesisEngine.  Default 3600 (1 hour).
    max_entries
        LRU cap.  Oldest-by-last_seen entries are evicted when exceeded.
        Default 10,000.
    nomination_boost_cap
        Maximum nomination_boost multiplier.  Default 3.0.
    """

    def __init__(
        self,
        default_ttl: float = 3600.0,
        max_entries: int = 10_000,
        nomination_boost_cap: float = 3.0,
    ) -> None:
        if default_ttl < 0:
            raise ValueError("default_ttl must be >= 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if nomination_boost_cap < 1.0:
            raise ValueError("nomination_boost_cap must be >= 1.0")

        self.default_ttl          = default_ttl
        self.max_entries          = max_entries
        self.nomination_boost_cap = nomination_boost_cap

        # key → RegistryEntry; ordered-dict for O(1) LRU eviction
        self._entries: Dict[Tuple[str, str], RegistryEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, candidate: Any) -> RegistryEntry:
        """
        Upsert a candidate into the registry.

        - First nomination → creates a new ``RegistryEntry``.
        - Subsequent nominations → increments ``nomination_count`` and updates
          ``last_seen``.

        Parameters
        ----------
        candidate
            Any object with ``source_id`` and ``target_id`` string attributes
            (e.g. ``ResearchCandidate``).

        Returns
        -------
        The current (possibly freshly created) ``RegistryEntry``.
        """
        key = self._key(candidate)
        now = time.time()

        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                entry.nomination_count += 1
                entry.last_seen = now
                # Refresh LRU position
                del self._entries[key]
                self._entries[key] = entry
            else:
                entry = RegistryEntry(
                    source_id=candidate.source_id,
                    target_id=candidate.target_id,
                    first_seen=now,
                    last_seen=now,
                    ttl=self.default_ttl,
                )
                self._entries[key] = entry
                self._evict_if_needed()

        return entry

    def should_evaluate(self, candidate: Any) -> bool:
        """
        Return True if this pair should be sent through the HypothesisEngine.

        A pair should be evaluated when:
        - It has never been evaluated (``last_evaluated`` is None), OR
        - Its TTL has expired since ``last_evaluated``.

        A pair not yet in the registry is always considered evaluable.
        """
        key = self._key(candidate)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return True
        if entry.last_evaluated is None:
            return True
        return (time.time() - entry.last_evaluated) >= entry.ttl

    def get_nomination_boost(self, candidate: Any) -> float:
        """
        Return the log-scale sampling multiplier for this candidate's
        discovery_potential based on its nomination_count.

        Formula: ``min(1.0 + 0.5 * log2(nomination_count), cap)``

        Returns 1.0 (no boost) for candidates not yet in the registry.
        """
        key = self._key(candidate)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return 1.0
        n = max(1, entry.nomination_count)
        boost = 1.0 + 0.5 * math.log2(n)
        return min(boost, self.nomination_boost_cap)

    def mark_evaluated(self, candidate: Any) -> None:
        """
        Record that this pair has just been through the HypothesisEngine.
        Resets the TTL countdown from now.

        Safe to call on candidates not yet in the registry (no-op).
        """
        key = self._key(candidate)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_evaluated = time.time()

    def prune(self, max_age_seconds: Optional[float] = None) -> int:
        """
        Evict stale entries from the registry.

        Parameters
        ----------
        max_age_seconds
            Entries whose ``last_seen`` is older than this many seconds are
            removed.  When None, uses ``default_ttl * 24`` (24 TTL-cycles).

        Returns
        -------
        Number of entries evicted.
        """
        if max_age_seconds is None:
            max_age_seconds = self.default_ttl * 24
        cutoff = time.time() - max_age_seconds
        evicted = 0

        with self._lock:
            stale_keys = [
                k for k, e in self._entries.items()
                if e.last_seen < cutoff
            ]
            for k in stale_keys:
                del self._entries[k]
                evicted += 1

        return evicted

    def stats(self) -> Dict[str, Any]:
        """Return a summary dict for monitoring and observability."""
        with self._lock:
            total = len(self._entries)
            never_eval = sum(
                1 for e in self._entries.values() if e.last_evaluated is None
            )
            multi_nom = sum(
                1 for e in self._entries.values() if e.nomination_count > 1
            )
            max_nom = max(
                (e.nomination_count for e in self._entries.values()),
                default=0,
            )
        return {
            "total_entries": total,
            "never_evaluated": never_eval,
            "multi_nominated": multi_nom,
            "max_nomination_count": max_nom,
            "default_ttl": self.default_ttl,
            "max_entries": self.max_entries,
            "nomination_boost_cap": self.nomination_boost_cap,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(candidate: Any) -> Tuple[str, str]:
        return (candidate.source_id, candidate.target_id)

    def _evict_if_needed(self) -> None:
        """Evict the oldest entry (by insertion order) if over capacity. Caller holds lock."""
        while len(self._entries) > self.max_entries:
            # Dict preserves insertion order in Python 3.7+; first key is oldest
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]

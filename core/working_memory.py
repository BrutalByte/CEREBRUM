"""Phase 95 — Working Memory Buffer.

Short-term episodic buffer of recent reasoning results. Thread-safe sliding
window with recency-weighted Jaccard relevance scoring.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    timestamp: float
    seeds: List[str]
    answers: List[str]                   # top answer entity_ids
    top_score: float
    soliton_index: Optional[float]
    prediction_error: Optional[float]
    source: str                          # "query" | "active_inference" | "goal_directed" | "discovery_cycle"


_DECAY_LAMBDA = 0.01  # recency decay rate: exp(-λ * seconds_ago)


class WorkingMemoryBuffer:
    """Thread-safe sliding window of MemoryEntry records.

    Relevance scoring: recency_weight × Jaccard(entry_tokens, query_seeds)
    where recency_weight = exp(-λ × age_in_seconds).
    """

    def __init__(self, maxlen: int = 50) -> None:
        self._maxlen = maxlen
        self._buffer: deque[MemoryEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def most_relevant(self, seeds: List[str], k: int = 5) -> List[MemoryEntry]:
        """Return up to k entries ranked by recency-weighted Jaccard similarity."""
        query_tokens = set(seeds)
        now = time.time()
        with self._lock:
            entries = list(self._buffer)

        if not entries:
            return []

        scored: List[tuple[float, MemoryEntry]] = []
        for entry in entries:
            age = max(0.0, now - entry.timestamp)
            recency = math.exp(-_DECAY_LAMBDA * age)
            entry_tokens = set(entry.seeds) | set(entry.answers)
            if query_tokens or entry_tokens:
                jaccard = len(query_tokens & entry_tokens) / len(query_tokens | entry_tokens) if (query_tokens | entry_tokens) else 0.0
            else:
                jaccard = 0.0
            scored.append((recency * jaccard, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    def recent(self, n: int = 10) -> List[MemoryEntry]:
        """Return up to n most recent entries, newest first."""
        with self._lock:
            entries = list(self._buffer)
        return list(reversed(entries[-n:]))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            buf = list(self._buffer)
        if not buf:
            return {"count": 0, "oldest_ts": None, "newest_ts": None, "maxlen": self._maxlen}
        return {
            "count": len(buf),
            "oldest_ts": buf[0].timestamp,
            "newest_ts": buf[-1].timestamp,
            "maxlen": self._maxlen,
        }

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

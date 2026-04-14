"""
ProvenanceLedger — Graph Provenance & Rollback (Phase 76).

Every edge materialized by the AutonomousDiscoveryLoop (or a manual approve()
call) is recorded here with its batch_id, finding_id, cycle_number, and edge
triples.  The ledger enables:

  - Per-finding rollback: remove exactly the edges added by one approval.
  - Per-cycle rollback: remove all edges materialized in a given loop cycle.
  - Audit trail: list recent batches with materialization timestamp.

Design notes
------------
- The ledger is in-memory with a configurable ``max_batches`` cap (LRU eviction).
- Edges are stored as (source, target, relation) tuples.
- ``rollback_*`` methods call ``adapter.remove_edge()`` and mark batches as
  rolled back.  If the adapter does not expose ``remove_edge``, they raise
  ``NotImplementedError`` rather than silently doing nothing.
- Thread-safe via a single ``threading.Lock``.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EdgeRecord:
    """A single edge triple recorded during materialization."""
    source: str
    target: str
    relation: str


@dataclass
class BatchRecord:
    """All edges materialized by a single approve() call."""
    batch_id: str
    """Unique identifier — typically the finding_id."""

    finding_id: str
    cycle_number: Optional[int]
    materialized_at: float
    edges: List[EdgeRecord]
    rolled_back: bool = False


# ---------------------------------------------------------------------------
# ProvenanceLedger
# ---------------------------------------------------------------------------

class ProvenanceLedger:
    """
    Records and manages provenance for edges materialized via ResearchAgent.

    Parameters
    ----------
    max_batches:
        Maximum number of batch records to retain (LRU — oldest evicted first).
    """

    def __init__(self, max_batches: int = 500) -> None:
        self._max_batches = max_batches
        # OrderedDict preserves insertion order; oldest entries are at the front
        self._batches: OrderedDict[str, BatchRecord] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_batch(
        self,
        batch_id: str,
        finding_id: str,
        edges: List[Tuple[str, str, str]],
        cycle_number: Optional[int] = None,
    ) -> BatchRecord:
        """
        Record a batch of materialized edges.

        Parameters
        ----------
        batch_id:
            Unique key (typically ``finding_id``).
        finding_id:
            The ResearchFinding that was approved.
        edges:
            List of ``(source, target, relation)`` triples that were added.
        cycle_number:
            AutonomousDiscoveryLoop cycle index, or ``None`` for manual approvals.

        Returns
        -------
        BatchRecord
            The newly created record.
        """
        record = BatchRecord(
            batch_id=batch_id,
            finding_id=finding_id,
            cycle_number=cycle_number,
            materialized_at=time.time(),
            edges=[EdgeRecord(s, t, r) for s, t, r in edges],
        )
        with self._lock:
            if batch_id in self._batches:
                # Idempotent re-record: update in place, move to end
                self._batches.move_to_end(batch_id)
                self._batches[batch_id] = record
            else:
                self._batches[batch_id] = record
                # Evict oldest if over capacity
                while len(self._batches) > self._max_batches:
                    evicted_id, evicted = next(iter(self._batches.items()))
                    del self._batches[evicted_id]
                    logger.debug("ProvenanceLedger: evicted batch %s.", evicted_id)
        return record

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_batch(self, batch_id: str, adapter) -> int:
        """
        Remove all edges in batch ``batch_id`` from the graph adapter.

        Parameters
        ----------
        batch_id:
            The batch to roll back (must match a recorded ``batch_id``).
        adapter:
            The graph adapter exposing ``remove_edge(u, v, relation)``.

        Returns
        -------
        int
            Number of edges removed.

        Raises
        ------
        ValueError
            If ``batch_id`` is not found in the ledger.
        NotImplementedError
            If the adapter does not expose ``remove_edge``.
        """
        if not hasattr(adapter, "remove_edge"):
            raise NotImplementedError(
                f"Adapter {type(adapter).__name__} does not implement remove_edge()."
            )
        with self._lock:
            record = self._batches.get(batch_id)
            if record is None:
                raise ValueError(f"ProvenanceLedger: batch_id {batch_id!r} not found.")
            if record.rolled_back:
                logger.warning("ProvenanceLedger: batch %s already rolled back.", batch_id)
                return 0
            edges_to_remove = list(record.edges)

        removed = 0
        for edge in edges_to_remove:
            try:
                adapter.remove_edge(edge.source, edge.target, edge.relation)
                removed += 1
            except Exception:
                logger.warning(
                    "ProvenanceLedger: could not remove edge %s -[%s]-> %s.",
                    edge.source, edge.relation, edge.target,
                )

        with self._lock:
            if batch_id in self._batches:
                self._batches[batch_id].rolled_back = True

        logger.info("ProvenanceLedger: rolled back batch %s (%d edges).", batch_id, removed)
        return removed

    def rollback_cycle(self, cycle_number: int, adapter) -> int:
        """
        Remove all edges materialized during ``cycle_number``.

        Returns the total number of edges removed across all batches in that cycle.
        """
        if not hasattr(adapter, "remove_edge"):
            raise NotImplementedError(
                f"Adapter {type(adapter).__name__} does not implement remove_edge()."
            )
        with self._lock:
            batch_ids = [
                bid for bid, rec in self._batches.items()
                if rec.cycle_number == cycle_number and not rec.rolled_back
            ]

        total = 0
        for bid in batch_ids:
            total += self.rollback_batch(bid, adapter)
        logger.info("ProvenanceLedger: rolled back cycle %d (%d edges).", cycle_number, total)
        return total

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def list_batches(self, n: int = 20) -> List[BatchRecord]:
        """Return the *n* most recent batch records (newest first)."""
        with self._lock:
            records = list(self._batches.values())
        return list(reversed(records))[:n]

    def get_batch(self, batch_id: str) -> Optional[BatchRecord]:
        """Return a single batch record by ID, or ``None`` if not found."""
        with self._lock:
            return self._batches.get(batch_id)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        with self._lock:
            total_edges = sum(len(r.edges) for r in self._batches.values())
            rolled_back = sum(1 for r in self._batches.values() if r.rolled_back)
            cycles = {r.cycle_number for r in self._batches.values() if r.cycle_number is not None}
            return {
                "total_batches": len(self._batches),
                "total_edges_recorded": total_edges,
                "batches_rolled_back": rolled_back,
                "cycles_seen": sorted(cycles),
                "max_batches": self._max_batches,
            }

"""
GlobalRebalancer — periodic modularity-drift detection and full DSCF re-optimization.

IncrementalCommunityUpdater (core/stream_engine.py) handles local subgraph updates
efficiently, but accumulated incremental changes cause modularity Q to drift silently
over time. GlobalRebalancer solves this by:

  1. Measuring modularity Q every N events
  2. Comparing to the last known Q (drift detection)
  3. Triggering a full DSCF re-run in a background thread when drift exceeds threshold
  4. Rate-limiting re-runs to prevent thrashing

Usage
-----
    from core.rebalancer import GlobalRebalancer
    from adapters.stream_adapter import StreamAdapter

    adapter = StreamAdapter(...)
    rebalancer = GlobalRebalancer(adapter, check_every_n_events=500, drift_threshold=0.05)
    adapter2 = StreamAdapter(..., rebalancer=rebalancer)

Or pass to an existing StreamAdapter (it will be called in ingest()):
    stream_adapter = StreamAdapter(..., rebalancer=rebalancer)
"""
from __future__ import annotations

import threading
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("cerebrum.rebalancer")


def _partitions_from_map(community_map: Dict[str, int]) -> List[frozenset]:
    """Convert {node -> cid} dict to List[frozenset] for modularity()."""
    groups: Dict[int, list] = {}
    for node, cid in community_map.items():
        groups.setdefault(cid, []).append(node)
    return [frozenset(members) for members in groups.values()]


def _community_map_from_partitions(partitions: List[frozenset]) -> Dict[str, int]:
    """Convert List[frozenset] to {node -> cid} dict."""
    result: Dict[str, int] = {}
    for cid, part in enumerate(partitions):
        for node in part:
            result[node] = cid
    return result


class GlobalRebalancer:
    """
    Background monitor that detects modularity Q drift and triggers a full
    DSCF re-optimization when drift exceeds threshold.

    Parameters
    ----------
    adapter                 : NetworkXAdapter-compatible adapter with ``._G``,
                              ``.community_map``, and ``._lock``.
    check_every_n_events    : How many ingest events between Q measurements.
    drift_threshold         : Absolute ΔQ that triggers a full re-run.
    min_rebalance_interval  : Minimum seconds between full re-runs (rate limit).
    n_dscf_trials           : Number of DSCF runs; the best-Q result is kept.
    dscf_seed               : Random seed for reproducibility (not directly
                              used by dscf_communities, but stored for logging).
    """

    def __init__(
        self,
        adapter,
        check_every_n_events: int = 200,
        drift_threshold: float = 0.05,
        min_rebalance_interval: float = 60.0,
        n_dscf_trials: int = 3,
        dscf_seed: int = 42,
        bridge_engine=None,
        pruning_enabled: bool = True,
    ) -> None:
        self._adapter = adapter
        self._check_every = check_every_n_events
        self._drift_threshold = drift_threshold
        self._min_interval = min_rebalance_interval
        self._n_trials = n_dscf_trials
        self._dscf_seed = dscf_seed
        self._bridge_engine = bridge_engine
        self._pruning_enabled = pruning_enabled

        from core.synaptic_pruner import SynapticPruner
        self._pruner = SynapticPruner(adapter)
        self._lock = threading.RLock()
        self._event_counter: int = 0
        self._last_q: float = 0.0
        self._last_q_initialized: bool = False
        self._last_rebalance_time: float = 0.0
        self._rebalance_count: int = 0
        self._rebalance_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def last_q(self) -> float:
        """Most recently measured modularity Q."""
        return self._last_q

    @property
    def rebalance_count(self) -> int:
        """Number of full DSCF re-runs completed."""
        return self._rebalance_count

    @property
    def event_counter(self) -> int:
        """Current event counter value (resets after each Q check)."""
        return self._event_counter

    def record_event(self) -> None:
        """
        Called after each ingest event (by StreamAdapter).

        Increments the internal counter. When the counter reaches
        ``check_every_n_events``, measures Q and triggers a re-run if needed.
        """
        with self._lock:
            self._event_counter += 1
            if self._event_counter >= self._check_every:
                self._event_counter = 0
                self._check_drift()

    def _compute_q(self) -> float:
        """Measure current modularity Q from the adapter's live graph."""
        import networkx as nx

        G = self._adapter._G
        community_map = self._adapter.community_map

        if G.number_of_edges() == 0 or not community_map:
            return 0.0

        # Build undirected view for modularity (nx.community.modularity needs undirected)
        G_und = G.to_undirected() if G.is_directed() else G

        parts = _partitions_from_map(community_map)
        # Filter out nodes not in current graph
        valid_parts = [frozenset(n for n in p if n in G_und) for p in parts]
        valid_parts = [p for p in valid_parts if p]

        if not valid_parts:
            return 0.0

        try:
            return float(nx.community.modularity(G_und, valid_parts))
        except Exception as exc:
            logger.debug("modularity computation failed: %s", exc)
            return 0.0

    def _check_drift(self, dry_run: bool = False) -> float:
        """
        Measure current Q, compute ΔQ from last known Q.

        Parameters
        ----------
        dry_run : If True, return ΔQ without triggering a rebalance.

        Returns
        -------
        Absolute ΔQ (float). Zero when no prior Q baseline exists.
        """
        current_q = self._compute_q()

        if not self._last_q_initialized:
            self._last_q = current_q
            self._last_q_initialized = True
            return 0.0

        delta_q = abs(current_q - self._last_q)

        if not dry_run:
            now = time.time()
            rate_limited = (now - self._last_rebalance_time) < self._min_interval
            if delta_q > self._drift_threshold and not rate_limited:
                logger.info(
                    "Q drift detected: last_q=%.4f current_q=%.4f ΔQ=%.4f — "
                    "scheduling full rebalance",
                    self._last_q, current_q, delta_q,
                )
                self._last_q = current_q
                self._full_rebalance()

        return delta_q

    def _full_rebalance(self) -> None:
        """Run best-of-N DSCF on the full graph in a background daemon thread."""
        # Don't launch if one is already running
        if self._rebalance_thread is not None and self._rebalance_thread.is_alive():
            logger.debug("Rebalance already in progress — skipping launch")
            return

        self._last_rebalance_time = time.time()
        thread = threading.Thread(target=self._rebalance_worker, daemon=True)
        self._rebalance_thread = thread
        thread.start()

    def _rebalance_worker(self) -> None:
        """Worker: run best-of-N DSCF and commit result under adapter lock."""
        try:
            self._rebalance_worker_inner()
        except Exception as exc:
            logger.error(
                "Rebalance worker crashed unexpectedly: %s", exc, exc_info=True
            )

    def _rebalance_worker_inner(self) -> None:
        """Inner rebalance logic — called by _rebalance_worker inside a top-level guard."""
        from core.community_engine import dscf_communities, modularity_score

        G = self._adapter._G
        G_und = G.to_undirected() if G.is_directed() else G

        if G_und.number_of_nodes() == 0:
            return

        # Best-of-N trials
        best_partitions = None
        best_q = float("-inf")

        for _ in range(self._n_trials):
            try:
                partitions = dscf_communities(G_und)
                q = modularity_score(G_und, partitions)
                if q > best_q:
                    best_q = q
                    best_partitions = partitions
            except Exception as exc:
                logger.warning("DSCF trial failed: %s", exc)

        if best_partitions is None:
            logger.warning("All DSCF trials failed — rebalance aborted")
            return

        new_map = _community_map_from_partitions(best_partitions)

        # Commit under the adapter's lock
        lock = getattr(self._adapter, "_lock", None)
        if lock is not None:
            with lock:
                self._adapter.community_map = new_map
        else:
            self._adapter.community_map = new_map

        # Post-rebalance hook: notify BridgeTwinEngine to prune stale records
        if self._bridge_engine is not None:
            try:
                pruned = self._bridge_engine.on_rebalance(new_map)
                if pruned:
                    logger.info("Bridge hook: pruned %d stale bridge records", pruned)
            except Exception as exc:
                logger.warning("Bridge on_rebalance hook failed: %s", exc)

        with self._lock:
            self._last_q = best_q
            self._rebalance_count += 1

        # Post-rebalance: Synaptic Pruning (Phase 61)
        if self._pruning_enabled:
            try:
                pruned_edges = self._pruner.prune()
                if pruned_edges > 0:
                    logger.info("SynapticPruner: pruned %d low-utility edges", pruned_edges)
            except Exception as exc:
                logger.warning("SynapticPruner failed during rebalance: %s", exc)

        logger.info(
            "Full rebalance #%d complete: Q=%.4f, communities=%d",
            self._rebalance_count, best_q, len(best_partitions),
        )

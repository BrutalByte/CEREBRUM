"""
OscillationEngine — Phase 219-B: Neural Oscillation / DSCF Synchronization.

The hippocampus uses theta/gamma coupling to bind temporal sequences and keep
active memory representations fresh.  The analog here: monitor query frequency
per community (theta cycle) and trigger partial DSCF re-runs on "hot" communities
much more frequently than the global drift-based rebalance.

Architecture
------------
- theta_period : number of queries between oscillation ticks (global).
- gamma_ratio  : partial DSCF re-runs per theta cycle for hot communities.
- hot_threshold: fraction-of-queries threshold for marking a community hot.

Query → record_query(touched_communities)
Every theta_period queries → trigger partial DSCF on hot communities
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger("cerebrum.oscillation")


class OscillationEngine:
    """
    Theta/gamma oscillation synchronization for DSCF community maintenance.

    Parameters
    ----------
    theta_period    : Queries per theta cycle (triggers hot-community rebalance).
    gamma_ratio     : Gamma cycles per theta (partial DSCF sub-runs per hot community).
    hot_threshold   : EMA fraction-of-queries touching a community to flag it hot.
    ema_alpha       : EMA decay for per-community query frequency estimates.
    """

    def __init__(
        self,
        theta_period: int = 50,
        gamma_ratio: int = 5,
        hot_threshold: float = 0.1,
        ema_alpha: float = 0.1,
    ) -> None:
        self.theta_period = theta_period
        self.gamma_ratio = gamma_ratio
        self.hot_threshold = hot_threshold
        self.ema_alpha = ema_alpha

        self._lock = threading.Lock()
        self._query_count: int = 0
        self._community_ema: Dict[int, float] = defaultdict(float)
        self._last_rebalance_time: float = 0.0
        self._partial_rebalance_count: int = 0

    def record_query(self, touched_communities: Set[int]) -> None:
        """Update EMA frequency counters for all communities touched by a query."""
        with self._lock:
            self._query_count += 1
            all_communities = set(self._community_ema.keys()) | touched_communities
            for cid in all_communities:
                hit = 1.0 if cid in touched_communities else 0.0
                prev = self._community_ema[cid]
                self._community_ema[cid] = prev * (1.0 - self.ema_alpha) + hit * self.ema_alpha

    def get_hot_communities(self) -> Set[int]:
        """Return community IDs whose EMA query frequency exceeds hot_threshold."""
        with self._lock:
            return {cid for cid, freq in self._community_ema.items()
                    if freq >= self.hot_threshold}

    def should_rebalance(self, query_count: Optional[int] = None) -> bool:
        """Return True every theta_period queries."""
        with self._lock:
            count = query_count if query_count is not None else self._query_count
            return (count > 0) and (count % self.theta_period == 0)

    def get_gamma_subgraph(self, community_id: int, adapter) -> Optional["nx.Graph"]:
        """Extract the local subgraph for a community (for partial DSCF re-run)."""
        import networkx as nx
        try:
            G = adapter._G
            community_map = adapter.community_map
            nodes = [n for n, cid in community_map.items() if cid == community_id]
            if not nodes:
                return None
            return G.subgraph(nodes).copy()
        except Exception as exc:
            logger.debug("get_gamma_subgraph(%d) failed: %s", community_id, exc)
            return None

    def partial_rebalance(self, adapter, rebalancer=None) -> int:
        """
        Run partial DSCF on hot communities only.

        Returns the number of communities re-partitioned.
        """
        from core.community_engine import dscf_communities

        hot = self.get_hot_communities()
        if not hot:
            return 0

        rebalanced = 0
        for cid in hot:
            subgraph = self.get_gamma_subgraph(cid, adapter)
            if subgraph is None or subgraph.number_of_nodes() < 3:
                continue
            G_und = subgraph.to_undirected() if subgraph.is_directed() else subgraph
            try:
                for _ in range(self.gamma_ratio):
                    partitions = dscf_communities(G_und)
                    # Map new sub-partitions back to global community map
                    lock = getattr(adapter, "_lock", None)
                    if lock is not None:
                        with lock:
                            self._apply_sub_partitions(adapter, partitions)
                    else:
                        self._apply_sub_partitions(adapter, partitions)
                rebalanced += 1
                logger.debug("OscillationEngine: partial DSCF community %d (%d nodes)",
                             cid, subgraph.number_of_nodes())
            except Exception as exc:
                logger.warning("Partial DSCF failed for community %d: %s", cid, exc)

        with self._lock:
            self._partial_rebalance_count += rebalanced
            self._last_rebalance_time = time.time()

        if rebalanced:
            logger.info("OscillationEngine: rebalanced %d hot communities (gamma=%d)",
                        rebalanced, self.gamma_ratio)
        return rebalanced

    def _apply_sub_partitions(self, adapter, partitions) -> None:
        """Overwrite community assignments for nodes in sub-partitions."""
        base_cid = max(adapter.community_map.values(), default=0) + 1
        for i, part in enumerate(partitions):
            new_cid = base_cid + i
            for node in part:
                adapter.community_map[node] = new_cid

    @property
    def query_count(self) -> int:
        with self._lock:
            return self._query_count

    @property
    def partial_rebalance_count(self) -> int:
        with self._lock:
            return self._partial_rebalance_count

    def status(self) -> dict:
        """Return a human-readable status dict for the /oscillations/status endpoint."""
        with self._lock:
            hot = {cid: round(freq, 4)
                   for cid, freq in self._community_ema.items()
                   if freq >= self.hot_threshold}
            return {
                "query_count": self._query_count,
                "theta_period": self.theta_period,
                "gamma_ratio": self.gamma_ratio,
                "hot_threshold": self.hot_threshold,
                "hot_communities": hot,
                "partial_rebalance_count": self._partial_rebalance_count,
                "last_rebalance_time": self._last_rebalance_time,
            }

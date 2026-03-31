"""
Phase 11 — Real-Time Streaming Graph Engine.

Core abstractions for live, mutable Knowledge Graph reasoning over
streaming data sources: sensors, signals, video detections, log events,
IoT feeds, WebSocket streams, etc.

Architecture
------------
                Stream Source (sensor / file / WebSocket / MQTT / ...)
                        |
                  StreamEvent (source, relation, target, timestamp, metadata)
                        |
                SlidingWindowBuffer  ← evicts stale events by time or count
                        |
                StreamAdapter  ←  live NetworkX graph (thread-safe)
                        |
          IncrementalCommunityUpdater  ← re-runs DSCF only on affected subgraph
                        |
              CSAEngine + AsyncBeamTraversal  ← existing reasoning stack
                        |
                   /query/stream SSE  ← consumers

Key Design Decisions
--------------------
- Thread safety: a single ``threading.RLock`` guards all graph mutations.
  Reasoning queries hold the lock for reads; ingest holds it for writes.
- Incremental DSCF: affected nodes + their radius-2 ego-network are
  extracted. DSCF runs on that subgraph only. Results are merged back
  using the existing merge_small_communities union-find approach.
- Sliding window supports two independent limits that work together:
    time_window_seconds  — evict edges older than N seconds
    max_edges            — keep only the most recent N edges (LRU-cap)
- Event metadata is preserved as edge attributes for downstream use.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# StreamEvent — the unit of streaming data
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """
    A single streaming observation expressed as a (source, relation, target) triple.

    All stream sources (sensors, video, logs, WebSocket) produce StreamEvents.
    The discretizer converts raw signals into this canonical form.

    Parameters
    ----------
    source    : source entity ID (e.g. "sensor_42", "camera_north", "user_101")
    relation  : edge label (e.g. "READS", "DETECTS", "PRECEDES", "CO_ACTIVATES")
    target    : target entity ID (e.g. "temp_HIGH", "person_0", "event_timeout")
    timestamp : Unix timestamp (float); defaults to now
    metadata  : arbitrary key-value payload from the source (raw value, confidence, etc.)
    ttl       : time-to-live in seconds; 0 = use the window default; -1 = permanent
    """
    source:    str
    relation:  str
    target:    str
    timestamp: float = field(default_factory=time.time)
    metadata:  Dict[str, Any] = field(default_factory=dict)
    ttl:       float = 0.0

    def edge_key(self) -> Tuple[str, str, str]:
        """Canonical (source, relation, target) identifier."""
        return (self.source, self.relation, self.target)


# ---------------------------------------------------------------------------
# SlidingWindowBuffer — time- and count-bounded event deque
# ---------------------------------------------------------------------------

class SlidingWindowBuffer:
    """
    Maintains a bounded deque of StreamEvents sorted by insertion order.

    Two independent limits work together:
      ``time_window_seconds`` — evict events older than this many seconds.
      ``max_edges``           — hard cap on total stored events (LRU eviction).

    The buffer tracks which (source, target) graph edges are still "live"
    (i.e. have at least one event within the window). When the last event
    for a given edge is evicted, the edge becomes stale.

    Thread safety: external callers must hold their own lock.
    """

    def __init__(
        self,
        time_window_seconds: float = 60.0,
        max_edges: int = 10_000,
    ):
        self.time_window_seconds = time_window_seconds
        self.max_edges = max_edges

        # Ordered queue of (timestamp, event) for O(1) front-eviction
        self._queue: deque[StreamEvent] = deque()

        # Reference-count map: edge_key → count of live events for that edge
        self._edge_refs: Dict[Tuple[str, str, str], int] = {}

    # -- Public API ----------------------------------------------------------

    def push(self, event: StreamEvent) -> List[StreamEvent]:
        """
        Add a new event. Returns a list of evicted events (if any).

        Eviction happens in two passes:
          1. Time-evict: remove events older than time_window_seconds.
          2. Count-cap:  if still over max_edges, drop the oldest.
        """
        self._queue.append(event)
        key = event.edge_key()
        self._edge_refs[key] = self._edge_refs.get(key, 0) + 1

        evicted = self._evict_stale(time.time())
        evicted += self._evict_to_cap()
        return evicted

    def evict_now(self) -> List[StreamEvent]:
        """Force an eviction pass (call periodically from ingestion loop)."""
        evicted = self._evict_stale(time.time())
        evicted += self._evict_to_cap()
        return evicted

    def live_edges(self) -> Set[Tuple[str, str, str]]:
        """Return the set of (source, relation, target) keys still in the window."""
        return {k for k, v in self._edge_refs.items() if v > 0}

    def stale_edges(self) -> Set[Tuple[str, str, str]]:
        """Return edge keys whose reference count has dropped to zero."""
        return {k for k, v in self._edge_refs.items() if v <= 0}

    def __len__(self) -> int:
        return len(self._queue)

    # -- Internal ------------------------------------------------------------

    def _evict_stale(self, now: float) -> List[StreamEvent]:
        evicted = []
        cutoff = now - self.time_window_seconds
        while self._queue and self._queue[0].timestamp < cutoff:
            ev = self._queue.popleft()
            # Honour per-event TTL override
            if ev.ttl < 0:  # permanent — skip eviction
                self._queue.appendleft(ev)
                break
            self._decrement_ref(ev)
            evicted.append(ev)
        return evicted

    def _evict_to_cap(self) -> List[StreamEvent]:
        evicted = []
        while len(self._queue) > self.max_edges:
            ev = self._queue.popleft()
            if ev.ttl < 0:
                self._queue.appendleft(ev)
                break
            self._decrement_ref(ev)
            evicted.append(ev)
        return evicted

    def _decrement_ref(self, ev: StreamEvent) -> None:
        key = ev.edge_key()
        count = self._edge_refs.get(key, 0)
        if count <= 1:
            self._edge_refs.pop(key, None)
        else:
            self._edge_refs[key] = count - 1


# ---------------------------------------------------------------------------
# IncrementalCommunityUpdater — subgraph-scoped DSCF re-runs
# ---------------------------------------------------------------------------

class IncrementalCommunityUpdater:
    """
    Runs community detection incrementally when the graph changes.

    Instead of re-running DSCF over the full graph on every event
    (which is O(N * max_iter)), this updater:

    1. Identifies the set of affected nodes (new / removed edges).
    2. Extracts the ego-network (radius ``neighborhood_radius``) around
       those nodes — typically a few hundred nodes even on large graphs.
    3. Re-runs DSCF on that subgraph.
    4. Merges the new assignments back into the global community_map.

    Parameters
    ----------
    neighborhood_radius : int
        BFS depth for ego-network extraction. 2 captures second-order
        effects without blowing up on high-degree hub nodes.
    min_events_before_update : int
        Batch N events before triggering a re-run. Prevents thrashing
        on high-frequency feeds (e.g. 1000 events/sec).
    max_subgraph_size : int
        If the affected subgraph exceeds this size, skip incremental
        update and schedule a full re-run instead.
    """

    def __init__(
        self,
        neighborhood_radius: int = 2,
        min_events_before_update: int = 10,
        max_subgraph_size: int = 2000,
        use_lpa: bool = True,
    ):
        self.neighborhood_radius = neighborhood_radius
        self.min_events_before_update = min_events_before_update
        self.max_subgraph_size = max_subgraph_size
        self.use_lpa = use_lpa
        """
        When True (default), incremental subgraph updates use LPA instead of
        DSCF.  LPA is faster and avoids the post-pass connectivity splitting
        that causes DSCF to over-fragment star-topology subgraphs.  Full
        re-runs triggered by ``run_full()`` still use best_of_n_dscf.
        """

        self._pending_nodes: Set[str] = set()
        self._events_since_update: int = 0
        self._full_rerun_scheduled: bool = False

    def mark_affected(self, nodes: List[str]) -> None:
        """Register nodes as affected by recent graph mutations."""
        self._pending_nodes.update(nodes)
        self._events_since_update += 1

    def should_update(self) -> bool:
        """True when enough events have accumulated to warrant a re-run."""
        return (
            self._events_since_update >= self.min_events_before_update
            and bool(self._pending_nodes)
        )

    def full_rerun_needed(self) -> bool:
        return self._full_rerun_scheduled

    def run(
        self,
        G,  # networkx Graph
        community_map: Dict[str, int],
        resolution: float = 1.0,
    ) -> Dict[str, int]:
        """
        Run incremental community detection on the affected subgraph.

        Returns an updated community_map (same dict, mutated in-place).
        """
        from core.community_engine import dscf_communities, merge_small_communities

        affected = set(self._pending_nodes)
        self._pending_nodes.clear()
        self._events_since_update = 0
        self._full_rerun_scheduled = False

        # 1. Build ego-network around affected nodes
        ego_nodes: Set[str] = set(affected)
        frontier = set(affected)
        for _ in range(self.neighborhood_radius):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in G:
                    next_frontier.update(G.neighbors(node))
            ego_nodes.update(next_frontier)
            frontier = next_frontier - ego_nodes

        ego_nodes = {n for n in ego_nodes if n in G}

        if len(ego_nodes) > self.max_subgraph_size:
            # Too large — schedule a full re-run at idle time
            self._full_rerun_scheduled = True
            return community_map

        if not ego_nodes:
            return community_map

        # 2. Extract subgraph and run community detection
        subgraph = G.subgraph(ego_nodes).copy()
        if G.is_directed():
            subgraph = subgraph.to_undirected()
        if subgraph.number_of_edges() == 0:
            # Isolated new nodes — assign each to its own community
            max_cid = max(community_map.values(), default=-1) + 1
            for node in ego_nodes:
                if node not in community_map:
                    community_map[node] = max_cid
                    max_cid += 1
            return community_map

        if self.use_lpa:
            from core.community_engine import lpa_communities
            new_parts = lpa_communities(subgraph)
        else:
            new_parts = dscf_communities(subgraph, resolution=resolution, max_iter=20)
        new_sub_map: Dict[str, int] = {}
        for i, part in enumerate(new_parts):
            for node in part:
                new_sub_map[node] = i

        # 3. Remap local IDs to avoid collisions with global community IDs
        max_cid = max(community_map.values(), default=-1) + 1
        for node, local_cid in new_sub_map.items():
            community_map[node] = max_cid + local_cid

        # 4. Merge small communities in the updated region
        if len(new_parts) > 3:
            community_map = merge_small_communities(community_map, G, min_size=2)

        return community_map

    def run_full(
        self,
        G,
        resolution: float = 1.0,
    ) -> Dict[str, int]:
        """
        Full DSCF re-run over the entire graph. Used when the incremental
        subgraph was too large, or on explicit request.
        """
        from core.community_engine import best_of_n_dscf, merge_small_communities

        G_undirected = G.to_undirected() if G.is_directed() else G
        parts = best_of_n_dscf(G_undirected, n_trials=1, resolution=resolution, max_iter=50,
                                use_multiprocessing=False)
        community_map = {node: i for i, members in enumerate(parts) for node in members}

        n_nodes = G.number_of_nodes()
        min_size = max(2, n_nodes // 200) if n_nodes > 500 else 2
        if min_size > 2:
            community_map = merge_small_communities(community_map, G, min_size=min_size)

        self._full_rerun_scheduled = False
        return community_map


# ---------------------------------------------------------------------------
# StreamStats — lightweight telemetry for the ingestion loop
# ---------------------------------------------------------------------------

class StreamStats:
    """Rolling statistics for the stream ingestion loop."""

    def __init__(self, window_seconds: float = 10.0):
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self.total_ingested: int = 0
        self.total_evicted: int = 0
        self.total_community_updates: int = 0
        self._lock = threading.Lock()

    def record_event(self) -> None:
        now = time.time()
        with self._lock:
            self._timestamps.append(now)
            self.total_ingested += 1
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    def record_eviction(self, count: int) -> None:
        with self._lock:
            self.total_evicted += count

    def record_community_update(self) -> None:
        with self._lock:
            self.total_community_updates += 1

    @property
    def events_per_second(self) -> float:
        with self._lock:
            if not self._timestamps:
                return 0.0
            span = time.time() - self._timestamps[0]
            return len(self._timestamps) / max(span, 1e-6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_per_second": round(self.events_per_second, 2),
            "total_ingested": self.total_ingested,
            "total_evicted": self.total_evicted,
            "total_community_updates": self.total_community_updates,
        }

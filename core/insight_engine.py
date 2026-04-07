"""
InsightEngine — Eureka/AHA moment detection and memory strengthening (Phase 15).

Biological analogy: insight learning creates stronger, faster memories than
gradual repetition because it is driven by *surprise* — the gap between
expected and actual outcome fires a dopamine burst that strengthens the
entire causal chain simultaneously.

Three-tier trigger architecture:
  Hot path  — O(1) ring-buffer append during every BeamTraversal crossing.
               Called inline; zero compute overhead.
  Warm path — Daemon thread drains ring buffer at a configurable rate,
               computes surprise scores, fires InsightEvents. ~0.01% CPU.
  Cold path — Community boundary scan on idle/schedule; finds latent insights
               that traversal hasn't yet reached. Milliseconds, periodic.

InsightEvents produce:
  - INSIGHT_LINK edge (confidence=0.85, weight=2.0) — resists REM pruning.
  - Reward propagation: every edge on the insight path gets a Hebbian boost
    proportional to the surprise magnitude.
"""
from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.graph_adapter import GraphAdapter

INSIGHT_RELATION   = "INSIGHT_LINK"
INSIGHT_CONFIDENCE = 0.85
INSIGHT_WEIGHT     = 2.0
HEBBIAN_DELTA      = 0.01   # per-traversal confidence increment along path
MIN_BASELINE_OBS   = 5      # observations required before surprise is meaningful


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """One cross-community crossing recorded by the hot path."""
    u: str
    v: str
    u_cid: int
    v_cid: int
    path_score: float
    path: Any           # TraversalPath (or None for cold-path candidates)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InsightEvent:
    """
    A detected insight — a cross-community connection whose path score
    exceeded the rolling baseline by more than salience_threshold.

    The bridging_node is the entity that made the insight possible: the
    unexpected intermediate that suddenly connected two previously distant
    knowledge regions.
    """
    bridging_node: str
    """The entity that bridged the community gap."""

    source: str
    """Start of the path that triggered the insight."""

    target: str
    """End of the path (the answer entity)."""

    insight_score: float
    """Combined surprise × explanatory-power score, in [0, 1]."""

    explanatory_power: float
    """Fraction of node pairs newly connected by this insight edge."""

    community_leap: int
    """Number of distinct community boundaries crossed in the path."""

    path: Any
    """TraversalPath that triggered the event (None for cold-path events)."""

    edge_created: bool = False
    """True if an INSIGHT_LINK edge was materialized in the graph."""

    timestamp: float = field(default_factory=time.time)

    # Phase 16 — validation fields (defaults preserve backward compatibility)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    """Unique short ID for this event (used by InsightValidator and MetaInsightEngine)."""

    validation_status: str = "pending"
    """
    Validation state set by InsightValidator.
    Values: "pending" | "bilateral" | "corroborated" | "unilateral" | "isolated"
    """

    corroboration_count: int = 0
    """Number of independent alternate paths that also reach the insight target."""

    def __repr__(self) -> str:
        return (
            f"InsightEvent(bridge={self.bridging_node!r}, "
            f"score={self.insight_score:.3f}, "
            f"power={self.explanatory_power:.3f}, "
            f"edge={'yes' if self.edge_created else 'no'})"
        )


# ---------------------------------------------------------------------------
# InsightEngine
# ---------------------------------------------------------------------------

class InsightEngine:
    """
    Detects and materializes insight events across three compute tiers.

    Parameters
    ----------
    adapter : GraphAdapter
        Graph adapter. Must expose ``adapter._G`` (NetworkXAdapter) and
        ``adapter.get_community()`` / ``adapter.get_embedding()``.
    salience_threshold : float
        Minimum surprise (path_score − baseline) to trigger an InsightEvent.
        Default 0.35 — roughly one standard deviation above typical scores.
    insight_confidence : float
        Confidence assigned to materialized INSIGHT_LINK edges. Default 0.85.
    insight_weight : float
        Traversal weight for INSIGHT_LINK edges. Default 2.0 — beam strongly
        prefers paths that have already produced insight.
    hebbian_delta : float
        Per-insight confidence increment for edges along the triggering path.
    ring_buffer_size : int
        Maximum hot-path candidates buffered before oldest are dropped.
    drain_rate : int
        Candidates processed per second by the warm-path daemon thread.
    cold_scan_interval : float
        Seconds between cold-path boundary scans. None = disabled.
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        salience_threshold: float = 0.35,
        insight_confidence: float = INSIGHT_CONFIDENCE,
        insight_weight: float = INSIGHT_WEIGHT,
        hebbian_delta: float = HEBBIAN_DELTA,
        ring_buffer_size: int = 1000,
        drain_rate: int = 100,
        cold_scan_interval: Optional[float] = 3600.0,
    ):
        self.adapter              = adapter
        self.salience_threshold   = salience_threshold
        self.insight_confidence   = insight_confidence
        self.insight_weight       = insight_weight
        self.hebbian_delta        = hebbian_delta
        self.drain_rate           = drain_rate
        self.cold_scan_interval   = cold_scan_interval

        self._lock      = threading.RLock()
        self._buffer: Deque[_Candidate] = collections.deque(maxlen=ring_buffer_size)
        self._events: Deque[InsightEvent] = collections.deque(maxlen=500)

        # Rolling baseline: (min_cid, max_cid) → running mean path score.
        # Capped at _BASELINE_MAX entries via LRU eviction (OrderedDict) to
        # prevent unbounded memory growth for graphs with many communities.
        _BASELINE_MAX = 10_000
        self._baselines: collections.OrderedDict = collections.OrderedDict()
        self._baseline_counts: collections.OrderedDict = collections.OrderedDict()
        self._baseline_max = _BASELINE_MAX

        self._paused    = False
        self._stopped   = False
        self._warm_thread: Optional[threading.Thread] = None
        self._cold_timer: Optional[threading.Timer]   = None

        self._start_warm_path()
        if cold_scan_interval is not None:
            self._schedule_cold()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # --- Hot path (called inline by BeamTraversal, O(1)) ----------------

    def record_crossing(
        self,
        u: str,
        v: str,
        u_cid: int,
        v_cid: int,
        path_score: float,
        path: Any,
    ) -> None:
        """
        Record a cross-community edge crossing. Called by BeamTraversal for
        every hop where source_community != target_community.
        O(1) — just appends to the ring buffer.
        """
        candidate = _Candidate(
            u=u, v=v,
            u_cid=u_cid, v_cid=v_cid,
            path_score=path_score,
            path=path,
        )
        with self._lock:
            self._buffer.append(candidate)

    # --- Warm path (daemon thread) ---------------------------------------

    def pause(self) -> None:
        """Pause warm-path processing. Hot-path recording continues."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume warm-path processing."""
        with self._lock:
            self._paused = False

    # --- Cold path (on-demand or scheduled) -----------------------------

    def scan_boundaries(self) -> List[InsightEvent]:
        """
        Cold path: scan adjacent community boundary pairs for latent insights.
        Finds high-similarity unconnected node pairs at community edges.
        Returns list of InsightEvents (edges materialized if score >= threshold).
        """

        G   = self._get_graph()
        
        # Phase 19 fix: Insight Decay (Hole 1).
        # Prevent "Recursive Insight" feedback loops by decaying old insight
        # edges. If they aren't reinforced by new traversals or Hebbian updates,
        # they eventually fade and are pruned.
        self._decay_existing_insights(G)

        events: List[InsightEvent] = []

        # Index nodes by community
        community_nodes: Dict[int, List[str]] = collections.defaultdict(list)
        for node in G.nodes():
            cid = self.adapter.get_community(node)
            if cid >= 0:
                community_nodes[cid].append(node)

        # Find adjacent community pairs (share at least one cross-community edge)
        adj_pairs: set = set()
        for u, v in G.edges():
            cid_u = self.adapter.get_community(u)
            cid_v = self.adapter.get_community(v)
            if cid_u >= 0 and cid_v >= 0 and cid_u != cid_v:
                adj_pairs.add((min(cid_u, cid_v), max(cid_u, cid_v)))

        for cid_a, cid_b in adj_pairs:
            nodes_a = community_nodes[cid_a]
            nodes_b = community_nodes[cid_b]

            # Collect unit-normalised embeddings for both communities in bulk.
            # Batch cosine similarity is then a single matrix multiply instead
            # of an O(|A|×|B|) nested Python loop.
            def _collect(node_list):
                ids, vecs = [], []
                for n in node_list:
                    e = self.adapter.get_embedding(n)
                    if e is None:
                        continue
                    norm = float(np.linalg.norm(e))
                    if norm > 0:
                        ids.append(n)
                        vecs.append(e.astype(np.float32) / norm)
                return ids, vecs

            ids_a, vecs_a = _collect(nodes_a)
            ids_b, vecs_b = _collect(nodes_b)
            if not ids_a or not ids_b:
                continue

            # mat_a: (|A|, dim),  mat_b: (|B|, dim)
            # sim_matrix: (|A|, |B|) — all cosine sims in one BLAS call
            mat_a = np.stack(vecs_a)                       # (|A|, dim)
            mat_b = np.stack(vecs_b)                       # (|B|, dim)
            sim_matrix = mat_a @ mat_b.T                   # (|A|, |B|)

            # Only evaluate pairs above threshold (discount 30 % for cold path)
            threshold_sim = self.salience_threshold / 0.7
            above = np.argwhere(sim_matrix >= threshold_sim)

            for ai, bi in above:
                u = ids_a[ai]
                v = ids_b[bi]
                if G.has_edge(u, v) or G.has_edge(v, u):
                    continue

                sim           = float(sim_matrix[ai, bi])
                insight_score = sim * 0.7
                if insight_score < self.salience_threshold:
                    continue

                power = self._explanatory_power(G, u, v)
                insight_score = min(1.0, (insight_score + power) / 2.0)

                event = InsightEvent(
                    bridging_node=u,
                    source=u,
                    target=v,
                    insight_score=insight_score,
                    explanatory_power=power,
                    community_leap=1,
                    path=None,
                    edge_created=False,
                )

                if not self._already_materialized(G, u, v):
                    self._materialize(event, G)
                    event.edge_created = True

                with self._lock:
                    self._events.append(event)
                events.append(event)

        return events

    def _decay_existing_insights(
        self, G, decay_rate: float = 0.95, min_conf: float = 0.2
    ) -> int:
        """
        Decay confidence of all INSIGHT_LINK edges; prune if too low.
        Returns number of edges pruned.
        """
        to_remove = []
        # Note: Iterating all edges is O(E). For massive graphs, this should be
        # optimized to use a dedicated index of insight edges.
        # For v1.2, this linear scan on the cold-path thread is acceptable.
        for u, v, data in G.edges(data=True):
            if data.get("relation") == INSIGHT_RELATION:
                # Decay confidence
                current_conf = data.get("confidence", 1.0)
                new_conf = current_conf * decay_rate
                
                if new_conf < min_conf:
                    to_remove.append((u, v))
                else:
                    data["confidence"] = new_conf
        
        for u, v in to_remove:
            G.remove_edge(u, v)
            
        return len(to_remove)

    # --- Results --------------------------------------------------------

    def recent_events(self, n: int = 20) -> List[InsightEvent]:
        """Return the last n InsightEvents (most recent last)."""
        with self._lock:
            items = list(self._events)
        return items[-n:]

    @property
    def total_events(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def buffer_capacity(self) -> int:
        return self._buffer.maxlen  # type: ignore[return-value]

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def stop(self) -> None:
        """Shut down warm-path thread and cold-path timer cleanly."""
        with self._lock:
            self._stopped = True
            if self._cold_timer is not None:
                self._cold_timer.cancel()

    # ------------------------------------------------------------------
    # Internal — warm path
    # ------------------------------------------------------------------

    def _start_warm_path(self) -> None:
        t = threading.Thread(target=self._drain_loop, daemon=True, name="InsightWarm")
        self._warm_thread = t
        t.start()

    def _drain_loop(self) -> None:
        """Warm-path daemon: drains ring buffer at drain_rate candidates/sec."""
        sleep_per_item = 1.0 / max(1, self.drain_rate)
        while True:
            with self._lock:
                if self._stopped:
                    return
                paused = self._paused
                candidate = self._buffer.popleft() if (not paused and self._buffer) else None

            if candidate is not None:
                self._evaluate_candidate(candidate)
                time.sleep(sleep_per_item)
            else:
                time.sleep(0.01)  # idle — check again in 10ms

    def _evaluate_candidate(self, c: _Candidate) -> None:
        """Compute surprise for a candidate and fire InsightEvent if warranted."""
        key = (min(c.u_cid, c.v_cid), max(c.u_cid, c.v_cid))

        with self._lock:
            n = self._baseline_counts.get(key, 0)
            baseline = self._baselines.get(key, c.path_score)

        if n < MIN_BASELINE_OBS:
            # Not enough history — record as baseline observation, don't fire
            self._update_baseline(key, c.path_score)
            return

        surprise = max(0.0, c.path_score - baseline)
        self._update_baseline(key, c.path_score)

        if surprise < self.salience_threshold:
            return

        # Surprise threshold crossed — compute full insight score
        G = self._get_graph()
        power = self._explanatory_power(G, c.u, c.v)
        insight_score = min(1.0, (surprise + power) / 2.0)

        # Count community leaps along the path
        community_leap = 1
        if c.path is not None:
            cids = getattr(c.path, "community_sequence", [])
            community_leap = sum(
                1 for i in range(len(cids) - 1) if cids[i] != cids[i + 1]
            )

        event = InsightEvent(
            bridging_node=c.v,
            source=c.path.head if c.path else c.u,
            target=c.path.tail if c.path else c.v,
            insight_score=insight_score,
            explanatory_power=power,
            community_leap=community_leap,
            path=c.path,
            edge_created=False,
        )

        if not self._already_materialized(G, c.u, c.v):
            self._materialize(event, G)
            event.edge_created = True

        if c.path is not None:
            self._propagate_reward(G, c.path, insight_score)

        with self._lock:
            self._events.append(event)

    # ------------------------------------------------------------------
    # Internal — materialization and reward
    # ------------------------------------------------------------------

    def _materialize(self, event: InsightEvent, G) -> None:
        """Add an INSIGHT_LINK edge to the graph for a confirmed insight."""
        u, v = event.source, event.target
        # Try forward direction first, then reverse (same policy as ContradictionEngine)
        for a, b in [(u, v), (v, u)]:
            if not G.has_edge(a, b):
                G.add_edge(
                    a, b,
                    relation=INSIGHT_RELATION,
                    confidence=self.insight_confidence,
                    weight=self.insight_weight,
                    provenance="insight",
                    insight_score=round(event.insight_score, 4),
                    explanatory_power=round(event.explanatory_power, 4),
                )
                break

    def _propagate_reward(self, G, path, insight_score: float) -> None:
        """
        Hebbian boost: increase confidence on every edge along the insight path.
        Delta is proportional to insight_score so stronger insights leave
        stronger traces. Confidence is capped at 1.0.
        """
        nodes = path.nodes
        delta = self.hebbian_delta * insight_score
        for i in range(1, len(nodes), 2):
            u = nodes[i - 1]
            v = nodes[i + 1]
            if G.has_edge(u, v):
                data = G.get_edge_data(u, v)
                old  = data.get("confidence", 1.0)
                data["confidence"] = min(1.0, old + delta)

    # ------------------------------------------------------------------
    # Internal — cold path scheduling
    # ------------------------------------------------------------------

    def _schedule_cold(self) -> None:
        def _fire():
            try:
                self.scan_boundaries()
            finally:
                with self._lock:
                    if not self._stopped and self.cold_scan_interval is not None:
                        self._schedule_cold()

        with self._lock:
            if self.cold_scan_interval is not None:
                self._cold_timer = threading.Timer(self.cold_scan_interval, _fire)
                self._cold_timer.daemon = True
                self._cold_timer.start()

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _get_graph(self):
        G = getattr(self.adapter, "_G", None)
        if G is None:
            raise AttributeError(
                "InsightEngine requires adapter._G (NetworkXAdapter). "
                "Override _get_graph() for other backends."
            )
        return G

    def _update_baseline(self, key: Tuple[int, int], score: float) -> None:
        with self._lock:
            n   = self._baseline_counts.get(key, 0)
            old = self._baselines.get(key, score)
            # Move key to end (most-recently-used) for LRU ordering
            self._baselines[key]       = (old * n + score) / (n + 1)
            self._baseline_counts[key] = n + 1
            self._baselines.move_to_end(key)
            self._baseline_counts.move_to_end(key)
            # Evict oldest entry when over capacity
            if len(self._baselines) > self._baseline_max:
                self._baselines.popitem(last=False)
                self._baseline_counts.popitem(last=False)

    def _explanatory_power(self, G, u: str, v: str) -> float:
        """
        Fraction of node pairs newly connected if edge u→v is added.

        If u and v are already in the same weakly-connected component the
        edge adds an alternative path but connects no new pairs → returns a
        small epsilon. Otherwise returns comp_u_size × comp_v_size / total_pairs.
        """
        import networkx as nx

        N = G.number_of_nodes()
        if N < 2:
            return 0.0
        total_pairs = N * (N - 1) / 2.0

        G_und = G.to_undirected() if G.is_directed() else G

        try:
            connected = nx.has_path(G_und, u, v)
        except nx.NodeNotFound:
            return 0.0

        if connected:
            # Already reachable — marginal power only
            return 1.0 / total_pairs

        try:
            comp_u = len(nx.node_connected_component(G_und, u))
            comp_v = len(nx.node_connected_component(G_und, v))
        except nx.NodeNotFound:
            return 0.0

        return (comp_u * comp_v) / total_pairs

    def _already_materialized(self, G, u: str, v: str) -> bool:
        """True if an INSIGHT_LINK edge already exists between u and v."""
        for a, b in [(u, v), (v, u)]:
            if G.has_edge(a, b):
                data = G.get_edge_data(a, b) or {}
                if data.get("relation") == INSIGHT_RELATION:
                    return True
        return False

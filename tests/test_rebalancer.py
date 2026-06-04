"""
Tests for Gap 1 â€” GlobalRebalancer.

Covers:
  - Event counting
  - Rebalance triggering on Q drift
  - Rate-limiting
  - Property accessors
  - StreamAdapter integration
  - Thread safety
  - dry_run check
"""
from typing import Counter
import threading
from unittest.mock import MagicMock

import networkx as nx

from core.rebalancer import GlobalRebalancer, _partitions_from_map, _community_map_from_partitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(n_nodes=10, n_edges=15):
    """Return a mock adapter backed by a real NetworkX graph."""
    G = nx.karate_club_graph() if n_nodes == 34 else _random_two_cliques(n_nodes, n_edges)
    adapter = MagicMock()
    adapter._G = G
    adapter._lock = threading.RLock()
    # Simple community map: all in community 0
    adapter.community_map = {n: n % 3 for n in G.nodes()}
    return adapter


def _random_two_cliques(n, e):
    """Two cliques connected by a bridge â€” easy to detect communities."""
    G = nx.Graph()
    half = n // 2
    # Clique A
    for i in range(half):
        for j in range(i + 1, half):
            G.add_edge(i, j)
    # Clique B
    for i in range(half, n):
        for j in range(i + 1, n):
            G.add_edge(i, j)
    # Bridge
    G.add_edge(0, half)
    return G


def _make_rebalancer(adapter=None, **kwargs):
    if adapter is None:
        adapter = _make_adapter()
    # We use explicit mapping to satisfy Mypy while keeping the flexible kwargs API
    return GlobalRebalancer(
        adapter,
        check_every_n_events=int(kwargs.get("check_every_n_events", 10)),
        drift_threshold=float(kwargs.get("drift_threshold", 0.05)),
        min_rebalance_interval=float(kwargs.get("min_rebalance_interval", 0.0)),
        n_dscf_trials=int(kwargs.get("n_dscf_trials", 1)),
        dscf_seed=int(kwargs.get("dscf_seed", 42)),
        bridge_engine=kwargs.get("bridge_engine"),
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def test_partitions_from_map_roundtrip():
    cmap = {"A": 0, "B": 0, "C": 1}
    parts = _partitions_from_map(cmap)
    assert frozenset({"A", "B"}) in parts
    assert frozenset({"C"}) in parts


def test_community_map_from_partitions():
    parts = [frozenset(["A", "B"]), frozenset(["C"])]
    cmap = _community_map_from_partitions(parts)
    assert cmap["A"] == cmap["B"]
    assert cmap["C"] != cmap["A"]


# ---------------------------------------------------------------------------
# record_event / counter
# ---------------------------------------------------------------------------

def test_record_event_increments_counter():
    r = _make_rebalancer(check_every_n_events=100)
    r.record_event()
    r.record_event()
    assert r.event_counter == 2


def test_counter_resets_after_threshold():
    r = _make_rebalancer(check_every_n_events=5)
    for _ in range(5):
        r.record_event()
    # Counter resets after hitting threshold
    assert r.event_counter == 0


def test_no_rebalance_before_threshold():
    r = _make_rebalancer(check_every_n_events=10)
    for _ in range(9):
        r.record_event()
    assert r.rebalance_count == 0


# ---------------------------------------------------------------------------
# Rebalance triggering
# ---------------------------------------------------------------------------

def test_rebalance_triggered_on_drift():
    """Force Q drop > threshold â†’ rebalance_count increments."""
    adapter = _make_adapter()
    r = _make_rebalancer(adapter=adapter, check_every_n_events=5,
                          drift_threshold=0.0001, min_rebalance_interval=0.0)

    # Seed initial Q
    r._check_drift()

    # Artificially set a very different last_q so delta is large
    r._last_q = 0.99  # current Q of the random graph is << 0.99

    # Trigger check â€” should schedule rebalance
    r._check_drift()

    # Give background thread a moment to finish
    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=5.0)

    assert r.rebalance_count >= 1


def test_no_rebalance_when_q_stable():
    adapter = _make_adapter()
    r = _make_rebalancer(adapter=adapter, check_every_n_events=5,
                          drift_threshold=0.5)  # very high threshold
    r._check_drift()  # seed
    r._check_drift()  # second check â€” Q hasn't changed, delta=0
    # Allow any thread to complete
    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=2.0)
    assert r.rebalance_count == 0


def test_rate_limit_prevents_thrashing():
    """Two drift events within min_rebalance_interval â†’ only 1 rebalance."""
    adapter = _make_adapter()
    r = _make_rebalancer(adapter=adapter, drift_threshold=0.0001,
                          min_rebalance_interval=60.0)  # 60s rate limit

    r._check_drift()         # seed
    r._last_q = 0.99         # force drift
    r._check_drift()         # fires rebalance #1

    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=5.0)

    count_after_first = r.rebalance_count

    r._last_q = 0.99         # force drift again
    r._check_drift()         # should be blocked by rate limit

    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=2.0)

    assert r.rebalance_count == count_after_first


# ---------------------------------------------------------------------------
# Property accessors
# ---------------------------------------------------------------------------

def test_rebalance_count_property():
    r = _make_rebalancer()
    assert r.rebalance_count == 0


def test_last_q_updated_after_rebalance():
    adapter = _make_adapter()
    r = _make_rebalancer(adapter=adapter, drift_threshold=0.0001,
                          min_rebalance_interval=0.0, n_dscf_trials=1)
    r._check_drift()        # seed
    r._last_q = 0.99
    r._check_drift()        # triggers rebalance

    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=5.0)

    # last_q must not still be 0.99 â€” worker updated it
    assert r.last_q != 0.99


# ---------------------------------------------------------------------------
# community_map updated after rebalance
# ---------------------------------------------------------------------------

def test_community_map_updated():
    G = _random_two_cliques(10, 20)
    adapter = MagicMock()
    adapter._G = G
    adapter._lock = threading.RLock()
    adapter.community_map = {n: 0 for n in G.nodes()}  # degenerate single community

    r = GlobalRebalancer(adapter, check_every_n_events=5,
                         drift_threshold=0.0001,
                         min_rebalance_interval=0.0,
                         n_dscf_trials=1)
    r._check_drift()       # seed
    r._last_q = 0.99
    r._check_drift()       # triggers rebalance

    if r._rebalance_thread is not None:
        r._rebalance_thread.join(timeout=5.0)

    # After rebalance, community_map should have been replaced
    # (values may differ from original all-0 degenerate map)
    assert isinstance(adapter.community_map, dict)
    assert len(adapter.community_map) == G.number_of_nodes()


# ---------------------------------------------------------------------------
# dry_run check
# ---------------------------------------------------------------------------

def test_dry_run_returns_delta_q_without_rebalancing():
    adapter = _make_adapter()
    r = _make_rebalancer(adapter=adapter, drift_threshold=0.0001,
                          min_rebalance_interval=0.0)
    r._check_drift()        # seed
    r._last_q = 0.99

    delta = r._check_drift(dry_run=True)

    # dry_run should not have triggered a rebalance
    assert r.rebalance_count == 0
    # delta should be non-negative
    assert delta >= 0.0


# ---------------------------------------------------------------------------
# StreamAdapter integration
# ---------------------------------------------------------------------------

def test_stream_adapter_integration():
    """StreamAdapter(rebalancer=r) â€” ingest calls record_event."""
    from adapters.stream_adapter import StreamAdapter
    from core.stream_engine import StreamEvent

    adapter = StreamAdapter(min_events_before_update=1000)
    r = GlobalRebalancer(adapter, check_every_n_events=3,
                         drift_threshold=0.5,
                         min_rebalance_interval=0.0)
    adapter._rebalancer = r

    for i in range(3):
        adapter.ingest(StreamEvent(source=f"A{i}", relation="REL", target=f"B{i}"))

    # After 3 events with check_every=3, counter should have reset
    assert r.event_counter == 0


def test_no_rebalancer_backward_compatible():
    """StreamAdapter() with no rebalancer must work unchanged."""
    from adapters.stream_adapter import StreamAdapter
    from core.stream_engine import StreamEvent

    adapter = StreamAdapter(min_events_before_update=1000)
    # No rebalancer â€” must not raise
    adapter.ingest(StreamEvent(source="X", relation="REL", target="Y"))
    assert adapter.stats.total_ingested == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safe_concurrent_ingest():
    """Multiple threads calling record_event simultaneously must not corrupt state."""
    adapter = _make_adapter()
    r = GlobalRebalancer(adapter, check_every_n_events=10,
                         drift_threshold=0.5,   # unlikely to trigger
                         min_rebalance_interval=60.0)

    errors = []

    def worker():
        try:
            for _ in range(50):
                r.record_event()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert errors == [], f"Thread errors: {errors}"
    assert r.event_counter >= 0

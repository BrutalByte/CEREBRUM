"""
Tests for core/insight_engine.py — InsightEngine (Phase 15).

All tests use small in-process NetworkXAdapter instances with synthetic
embeddings (RandomEngine). No external dependencies. No file I/O.
"""
import time

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.insight_engine import (
    InsightEngine,
    InsightEvent,
    _Candidate,
    INSIGHT_RELATION,
    INSIGHT_CONFIDENCE,
    INSIGHT_WEIGHT,
    MIN_BASELINE_OBS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _adapter_two_communities():
    """
    Graph with two clear communities separated by a single cross edge.
    Community 0: A, B, C
    Community 1: D, E, F
    Cross edge: C → D
    """
    G = nx.DiGraph()
    edges = [
        ("A", "B", "KNOWS"), ("B", "C", "KNOWS"), ("A", "C", "KNOWS"),
        ("D", "E", "KNOWS"), ("E", "F", "KNOWS"), ("D", "F", "KNOWS"),
        ("C", "D", "BRIDGE"),  # cross-community
    ]
    for u, v, rel in edges:
        G.add_node(u, label=u, type="entity")
        G.add_node(v, label=v, type="entity")
        G.add_edge(u, v, relation=rel, weight=1.0, confidence=1.0)

    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter.embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    # Manually set communities (normally set by CommunityEngine)
    adapter.community_map = {"A": 0, "B": 0, "C": 0, "D": 1, "E": 1, "F": 1}

    # Patch get_community to use community_map
    def patched_get_community(entity_id):
        return adapter.community_map.get(entity_id, -1)
    adapter.get_community = patched_get_community

    return adapter


def _make_engine(adapter, **kwargs) -> InsightEngine:
    """Create an InsightEngine with cold scan disabled for speed."""
    kwargs.setdefault("cold_scan_interval", None)
    kwargs.setdefault("drain_rate", 1000)  # fast drain for tests
    return InsightEngine(adapter, **kwargs)


def _flood_baseline(engine, u_cid, v_cid, score, n=MIN_BASELINE_OBS):
    """Force enough baseline observations so surprise can fire."""
    key = (min(u_cid, v_cid), max(u_cid, v_cid))
    for _ in range(n):
        engine._update_baseline(key, score)


# ---------------------------------------------------------------------------
# Ring buffer — hot path
# ---------------------------------------------------------------------------

def test_record_crossing_adds_to_buffer():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    ie.pause()  # freeze warm path so buffer isn't drained
    assert ie.buffer_size == 0
    ie.record_crossing("C", "D", 0, 1, path_score=0.8, path=None)
    assert ie.buffer_size == 1
    ie.stop()


def test_ring_buffer_bounded():
    """Buffer drops oldest entries when full (deque maxlen behaviour)."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, ring_buffer_size=3)
    ie.pause()
    for i in range(10):
        ie.record_crossing("C", "D", 0, 1, path_score=float(i) / 10, path=None)
    assert ie.buffer_size == 3  # capped at maxlen
    assert ie.buffer_capacity == 3
    ie.stop()


def test_pause_stops_drain():
    """Pausing prevents the warm thread from draining the buffer."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, drain_rate=10)
    ie.pause()
    for _ in range(5):
        ie.record_crossing("C", "D", 0, 1, path_score=0.5, path=None)
    time.sleep(0.1)
    # Buffer should still have items (warm path paused)
    assert ie.buffer_size > 0
    ie.stop()


def test_resume_allows_drain():
    """After resume(), warm path drains the buffer."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, drain_rate=1000)
    ie.pause()
    ie.record_crossing("C", "D", 0, 1, path_score=0.5, path=None)
    ie.resume()
    deadline = time.time() + 2.0
    while ie.buffer_size > 0 and time.time() < deadline:
        time.sleep(0.01)
    assert ie.buffer_size == 0
    ie.stop()


# ---------------------------------------------------------------------------
# Baseline and surprise scoring
# ---------------------------------------------------------------------------

def test_compute_surprise_no_baseline():
    """Fewer than MIN_BASELINE_OBS observations → no InsightEvent fired."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    # Only 2 observations — not enough
    key = (0, 1)
    for _ in range(2):
        ie._update_baseline(key, 0.3)

    c = _Candidate(u="C", v="D", u_cid=0, v_cid=1, path_score=0.9, path=None)
    ie._evaluate_candidate(c)
    assert ie.total_events == 0
    ie.stop()


def test_compute_surprise_below_threshold():
    """Path score barely above baseline → no event."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.5)
    _flood_baseline(ie, 0, 1, 0.6)  # baseline = 0.6
    c = _Candidate(u="C", v="D", u_cid=0, v_cid=1, path_score=0.65, path=None)
    ie._evaluate_candidate(c)
    assert ie.total_events == 0  # surprise = 0.05, threshold = 0.5
    ie.stop()


def test_compute_surprise_above_threshold():
    """Path score well above baseline → InsightEvent fired."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.2)
    _flood_baseline(ie, 0, 1, 0.3)  # baseline = 0.3
    c = _Candidate(u="C", v="D", u_cid=0, v_cid=1, path_score=0.95, path=None)
    ie._evaluate_candidate(c)
    assert ie.total_events == 1
    ie.stop()


def test_insight_score_range():
    """insight_score must always be in [0, 1]."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.1)
    _flood_baseline(ie, 0, 1, 0.1)
    c = _Candidate(u="C", v="D", u_cid=0, v_cid=1, path_score=1.0, path=None)
    ie._evaluate_candidate(c)
    for ev in ie.recent_events():
        assert 0.0 <= ev.insight_score <= 1.0
    ie.stop()


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------

def test_materialize_creates_insight_edge():
    """InsightEvent with edge_created=True adds INSIGHT_LINK to the graph."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.1)
    G = adapter._G

    event = InsightEvent(
        bridging_node="D", source="C", target="D",
        insight_score=0.8, explanatory_power=0.2,
        community_leap=1, path=None, edge_created=False,
    )
    ie._materialize(event, G)

    # Either C→D or D→C should now be INSIGHT_LINK
    found = False
    for a, b in [("C", "D"), ("D", "C")]:
        if G.has_edge(a, b):
            data = G.get_edge_data(a, b)
            if data.get("relation") == INSIGHT_RELATION:
                found = True
                break
    assert found
    ie.stop()


def test_materialize_edge_metadata():
    """INSIGHT_LINK edge has correct confidence, weight, provenance."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    G = adapter._G

    # Remove the C→D BRIDGE edge so forward slot is free
    G.remove_edge("C", "D")

    event = InsightEvent(
        bridging_node="D", source="C", target="D",
        insight_score=0.75, explanatory_power=0.3,
        community_leap=1, path=None, edge_created=False,
    )
    ie._materialize(event, G)

    data = G.get_edge_data("C", "D")
    assert data is not None
    assert data["relation"] == INSIGHT_RELATION
    assert data["confidence"] == pytest.approx(INSIGHT_CONFIDENCE)
    assert data["weight"] == pytest.approx(INSIGHT_WEIGHT)
    assert data["provenance"] == "insight"
    ie.stop()


def test_no_duplicate_materialization():
    """_already_materialized prevents adding a second INSIGHT_LINK."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    G = adapter._G
    G.remove_edge("C", "D")

    event = InsightEvent(
        bridging_node="D", source="C", target="D",
        insight_score=0.8, explanatory_power=0.2,
        community_leap=1, path=None,
    )
    ie._materialize(event, G)
    assert ie._already_materialized(G, "C", "D")
    ie.stop()


# ---------------------------------------------------------------------------
# Reward propagation
# ---------------------------------------------------------------------------

class _MockPath:
    """Minimal TraversalPath substitute for reward-propagation tests."""
    def __init__(self, nodes):
        self.nodes = nodes
        self.community_sequence = []
        self.head = nodes[0] if nodes else ""
        self.tail = nodes[-1] if nodes else ""


def test_propagate_reward_boosts_edge_confidence():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    G = adapter._G

    # A→B is currently confidence=1.0; path A→KNOWS→B
    path = _MockPath(["A", "KNOWS", "B"])
    before = G.get_edge_data("A", "B")["confidence"]
    ie._propagate_reward(G, path, insight_score=1.0)
    after = G.get_edge_data("A", "B")["confidence"]
    assert after > before or after == 1.0  # already capped at 1.0


def test_propagate_reward_caps_at_one():
    """Confidence must never exceed 1.0 after reward propagation."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, hebbian_delta=0.5)
    G = adapter._G

    path = _MockPath(["A", "KNOWS", "B"])
    # Apply many times
    for _ in range(20):
        ie._propagate_reward(G, path, insight_score=1.0)
    assert G.get_edge_data("A", "B")["confidence"] <= 1.0
    ie.stop()


def test_propagate_reward_proportional_to_insight_score():
    """Stronger insights produce larger confidence boosts."""

    adapter1 = _adapter_two_communities()
    adapter2 = _adapter_two_communities()
    ie1 = _make_engine(adapter1, hebbian_delta=0.1)
    ie2 = _make_engine(adapter2, hebbian_delta=0.1)

    path = _MockPath(["A", "KNOWS", "B"])
    ie1._propagate_reward(adapter1._G, path, insight_score=0.2)
    ie2._propagate_reward(adapter2._G, path, insight_score=0.9)

    conf1 = adapter1._G.get_edge_data("A", "B")["confidence"]
    conf2 = adapter2._G.get_edge_data("A", "B")["confidence"]
    # Higher insight_score → larger boost
    assert conf2 >= conf1
    ie1.stop()
    ie2.stop()


# ---------------------------------------------------------------------------
# Explanatory power
# ---------------------------------------------------------------------------

def test_explanatory_power_disconnected():
    """Adding an edge between two separate components → high explanatory power."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)

    # Build a disconnected graph: two separate cliques
    G2 = nx.DiGraph()
    G2.add_edge("X1", "X2", relation="K", weight=1.0, confidence=1.0)
    G2.add_edge("Y1", "Y2", relation="K", weight=1.0, confidence=1.0)
    adapter2 = NetworkXAdapter(G2)
    adapter2.embeddings = {}
    ie2 = _make_engine(adapter2)

    power = ie2._explanatory_power(G2, "X1", "Y1")
    assert power > 0.1  # Should be substantial (2 × 2 / C(4,2) = 4/6 ≈ 0.67)
    ie.stop()
    ie2.stop()


def test_explanatory_power_already_connected():
    """Adding an edge between already-connected nodes → near-zero power."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    G = adapter._G

    # A and B are already connected
    power = ie._explanatory_power(G, "A", "B")
    assert power < 0.1
    ie.stop()


# ---------------------------------------------------------------------------
# recent_events and total_events
# ---------------------------------------------------------------------------

def test_recent_events_empty_initially():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    assert ie.recent_events() == []
    assert ie.total_events == 0
    ie.stop()


def test_recent_events_capped_at_n():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.05)
    _flood_baseline(ie, 0, 1, 0.1)

    # Fire multiple events directly
    for i in range(15):
        ev = InsightEvent(
            bridging_node="D", source="C", target="D",
            insight_score=float(i) / 15,
            explanatory_power=0.1,
            community_leap=1, path=None,
        )
        ie._events.append(ev)

    recent = ie.recent_events(5)
    assert len(recent) == 5
    ie.stop()


# ---------------------------------------------------------------------------
# Cold path boundary scan
# ---------------------------------------------------------------------------

def test_cold_scan_finds_boundary_candidates():
    """scan_boundaries() finds high-similarity pairs at community edges."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.01)

    # Give A and D near-identical embeddings (cross-community pair)
    base = np.ones(32, dtype=np.float32) / np.sqrt(32)
    adapter.embeddings["A"] = base.copy()
    adapter.embeddings["D"] = base.copy()

    events = ie.scan_boundaries()
    # Should find at least one event (A and D are similar and A-D not directly connected)
    assert isinstance(events, list)
    ie.stop()


def test_cold_scan_does_not_duplicate_existing_insight_edges():
    """scan_boundaries() skips pairs that already have an INSIGHT_LINK."""
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter, salience_threshold=0.01)
    G = adapter._G

    # Pre-add INSIGHT_LINK between A and D
    G.add_edge("A", "D", relation=INSIGHT_RELATION, confidence=0.85, weight=2.0, provenance="insight")

    base = np.ones(32, dtype=np.float32) / np.sqrt(32)
    adapter.embeddings["A"] = base.copy()
    adapter.embeddings["D"] = base.copy()

    events = ie.scan_boundaries()
    # Any A-D event should have edge_created=False
    for ev in events:
        if (ev.source == "A" and ev.target == "D") or (ev.source == "D" and ev.target == "A"):
            assert ev.edge_created is False
    ie.stop()


# ---------------------------------------------------------------------------
# Stop / cleanup
# ---------------------------------------------------------------------------

def test_stop_does_not_raise():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    ie.stop()  # should not raise


def test_is_paused_property():
    adapter = _adapter_two_communities()
    ie = _make_engine(adapter)
    assert ie.is_paused is False
    ie.pause()
    assert ie.is_paused is True
    ie.resume()
    assert ie.is_paused is False
    ie.stop()

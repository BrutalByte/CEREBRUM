"""
Tests for core/meta_insight_engine.py — MetaInsightEngine (Phase 16).

All tests use synthetic InsightEvents — no graph adapter or traversal needed.
MetaInsightEngine operates entirely on InsightEvent metadata.
"""
import time

import pytest

from core.insight_engine import InsightEvent
from core.meta_insight_engine import MetaInsightEngine, MetaInsightEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(source, target, bridging=None, score=0.8, community_leap=1, ts=None):
    """Create a synthetic InsightEvent."""
    return InsightEvent(
        bridging_node=bridging or target,
        source=source,
        target=target,
        insight_score=score,
        explanatory_power=0.2,
        community_leap=community_leap,
        path=None,
        edge_created=True,
        timestamp=ts if ts is not None else time.time(),
    )


def _engine(**kwargs) -> MetaInsightEngine:
    kwargs.setdefault("chain_score_threshold", 0.1)  # low for deterministic tests
    kwargs.setdefault("temporal_window", 300.0)
    kwargs.setdefault("meta_depth", 2)
    return MetaInsightEngine(**kwargs)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_empty_state():
    meta = _engine()
    assert meta.total_meta_events == 0
    assert meta.recent_meta_events() == []
    nodes, edges = meta.insight_graph_size
    assert nodes == 0 and edges == 0


def test_single_event_no_meta():
    """One event → no meta-events (need at least two to pair)."""
    meta = _engine()
    ev = _event("A", "B")
    fired = meta.observe(ev)
    assert fired == []
    assert meta.total_meta_events == 0


# ---------------------------------------------------------------------------
# Chain detection (depth-1)
# ---------------------------------------------------------------------------

def test_chain_fires_when_a_target_equals_b_source():
    """
    Insight A: X→Y and Insight B: Y→Z form a chain.
    A.target == B.source → connection_type='chain'.
    """
    meta = _engine()
    ev_a = _event("X", "Y")
    ev_b = _event("Y", "Z")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    chain_events = [e for e in fired if e.connection_type == "chain"]
    assert len(chain_events) >= 1
    assert chain_events[0].depth == 1


def test_chain_fires_bidirectionally():
    """
    B.target == A.source also counts as a chain (reverse direction).
    Insight A: Y→X and Insight B: X→Y (loop) → reverse chain.
    """
    meta = _engine()
    ev_a = _event("Y", "X")
    ev_b = _event("X", "Y")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    chain_events = [e for e in fired if e.connection_type == "chain"]
    assert len(chain_events) >= 1


def test_no_chain_when_entities_disjoint():
    """Insights with completely disjoint entities → no chain connection."""
    meta = _engine()
    ev_a = _event("A", "B")
    ev_b = _event("C", "D")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    chain_events = [e for e in fired if e.connection_type == "chain"]
    assert chain_events == []


# ---------------------------------------------------------------------------
# Shared entity detection
# ---------------------------------------------------------------------------

def test_shared_entity_fires_on_same_source():
    """Two insights starting from the same source entity."""
    meta = _engine()
    ev_a = _event("Hub", "X")
    ev_b = _event("Hub", "Y")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    shared = [e for e in fired if e.connection_type == "shared_entity"]
    assert len(shared) >= 1


def test_shared_entity_fires_on_same_target():
    """Two insights ending at the same target entity."""
    meta = _engine()
    ev_a = _event("X", "Sink")
    ev_b = _event("Y", "Sink")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    shared = [e for e in fired if e.connection_type == "shared_entity"]
    assert len(shared) >= 1


def test_shared_entity_fires_on_bridging_node_match():
    """Bridging node of one insight matches source of another."""
    meta = _engine()
    ev_a = _event("A", "B", bridging="Pivot")
    ev_b = _event("Pivot", "C")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    shared = [e for e in fired if e.connection_type == "shared_entity"]
    assert len(shared) >= 1


def test_no_shared_entity_when_all_distinct():
    """No common entities across two insights → no shared_entity connection."""
    meta = _engine()
    ev_a = _event("A", "B", bridging="M")
    ev_b = _event("C", "D", bridging="N")
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    shared = [e for e in fired if e.connection_type == "shared_entity"]
    assert shared == []


# ---------------------------------------------------------------------------
# Community overlap detection
# ---------------------------------------------------------------------------

def test_community_overlap_same_leap_count():
    """Two insights crossing 2 community boundaries → community_overlap."""
    meta = _engine()
    ev_a = _event("A", "B", community_leap=2)
    ev_b = _event("C", "D", community_leap=2)
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    overlap = [e for e in fired if e.connection_type == "community_overlap"]
    assert len(overlap) >= 1


def test_no_community_overlap_different_leap_counts():
    """Different community_leap values → no community_overlap."""
    meta = _engine()
    ev_a = _event("A", "B", community_leap=1)
    ev_b = _event("C", "D", community_leap=3)
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    overlap = [e for e in fired if e.connection_type == "community_overlap"]
    assert overlap == []


def test_no_community_overlap_when_leap_zero():
    """community_leap=0 means intra-community — no overlap connection."""
    meta = _engine()
    ev_a = _event("A", "B", community_leap=0)
    ev_b = _event("C", "D", community_leap=0)
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    overlap = [e for e in fired if e.connection_type == "community_overlap"]
    assert overlap == []


# ---------------------------------------------------------------------------
# Temporal cluster detection
# ---------------------------------------------------------------------------

def test_temporal_cluster_within_window():
    """Events fired within temporal_window → temporal_cluster connection."""
    now = time.time()
    meta = _engine(temporal_window=60.0)
    ev_a = _event("A", "B", ts=now)
    ev_b = _event("C", "D", ts=now + 30.0)  # 30s later — within 60s window
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    cluster = [e for e in fired if e.connection_type == "temporal_cluster"]
    assert len(cluster) >= 1


def test_no_temporal_cluster_outside_window():
    """Events fired outside the temporal window → no temporal_cluster."""
    now = time.time()
    meta = _engine(temporal_window=10.0)
    ev_a = _event("A", "B", ts=now)
    ev_b = _event("C", "D", ts=now + 60.0)  # 60s later — outside 10s window
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    cluster = [e for e in fired if e.connection_type == "temporal_cluster"]
    assert cluster == []


# ---------------------------------------------------------------------------
# Score threshold
# ---------------------------------------------------------------------------

def test_threshold_filters_low_score_events():
    """Events with score below threshold are not recorded in meta_events."""
    meta = _engine(chain_score_threshold=0.9)
    ev_a = _event("A", "B", score=0.2)
    ev_b = _event("B", "C", score=0.2)  # chain, but avg score = 0.2 < 0.9
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    # chain connection should exist in the graph but not fired as a MetaInsightEvent
    # (score below threshold)
    recorded_chain = [e for e in fired if e.connection_type == "chain"]
    assert recorded_chain == []


def test_threshold_allows_high_score_events():
    """Events with score above threshold are recorded."""
    meta = _engine(chain_score_threshold=0.5)
    ev_a = _event("A", "B", score=0.9)
    ev_b = _event("B", "C", score=0.9)
    meta.observe(ev_a)
    fired = meta.observe(ev_b)

    recorded_chain = [e for e in fired if e.connection_type == "chain"]
    assert len(recorded_chain) >= 1


# ---------------------------------------------------------------------------
# InsightGraph state
# ---------------------------------------------------------------------------

def test_insight_graph_nodes_grow_with_observations():
    meta = _engine()
    assert meta.insight_graph_size[0] == 0

    meta.observe(_event("A", "B"))
    assert meta.insight_graph_size[0] == 1

    meta.observe(_event("C", "D"))
    assert meta.insight_graph_size[0] == 2


def test_insight_graph_edges_added_on_connection():
    """InsightGraph gets an edge when two insights are related."""
    meta = _engine()
    meta.observe(_event("A", "B"))
    meta.observe(_event("B", "C"))  # chain → edge added

    _, edges = meta.insight_graph_size
    assert edges >= 1


def test_export_insight_graph_structure():
    """export_insight_graph() returns dict with nodes/edges lists."""
    meta = _engine()
    meta.observe(_event("A", "B"))
    meta.observe(_event("B", "C"))

    data = meta.export_insight_graph()
    assert "nodes" in data and "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    assert len(data["nodes"]) == 2


def test_export_node_fields():
    """Each exported node has required fields."""
    meta = _engine()
    ev = _event("A", "B", score=0.75, community_leap=2)
    meta.observe(ev)

    data = meta.export_insight_graph()
    node = data["nodes"][0]
    assert node["id"] == ev.id
    assert node["source_entity"] == "A"
    assert node["target_entity"] == "B"
    assert node["insight_score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Depth-2 higher-order detection
# ---------------------------------------------------------------------------

def test_depth_2_fires_for_three_event_chain():
    """
    Three chained insights A→B, B→C, C→D.
    After observing C→D, the engine should detect the depth-2 pattern
    [A→B] → [B→C] → [C→D].
    """
    meta = _engine(chain_score_threshold=0.1, meta_depth=2)
    ev_ab = _event("A", "B", score=0.9)
    ev_bc = _event("B", "C", score=0.9)
    ev_cd = _event("C", "D", score=0.9)

    meta.observe(ev_ab)
    meta.observe(ev_bc)
    fired = meta.observe(ev_cd)

    depth2 = [e for e in fired if e.depth == 2]
    assert len(depth2) >= 1


def test_depth_2_chain_ids_contains_three_ids():
    """Depth-2 MetaInsightEvent.chain_ids has exactly 3 entries."""
    meta = _engine(chain_score_threshold=0.1, meta_depth=2)
    ev1 = _event("A", "B", score=0.9)
    ev2 = _event("B", "C", score=0.9)
    ev3 = _event("C", "D", score=0.9)

    meta.observe(ev1)
    meta.observe(ev2)
    fired = meta.observe(ev3)

    depth2 = [e for e in fired if e.depth == 2]
    if depth2:
        assert len(depth2[0].chain_ids) == 3


def test_no_depth_2_when_meta_depth_is_1():
    """meta_depth=1 disables higher-order detection."""
    meta = _engine(chain_score_threshold=0.1, meta_depth=1)
    meta.observe(_event("A", "B", score=0.9))
    meta.observe(_event("B", "C", score=0.9))
    fired = meta.observe(_event("C", "D", score=0.9))

    depth2 = [e for e in fired if e.depth == 2]
    assert depth2 == []


# ---------------------------------------------------------------------------
# Ring buffer / recent_meta_events
# ---------------------------------------------------------------------------

def test_recent_meta_events_capped():
    """recent_meta_events(n) returns at most n events."""
    meta = _engine(chain_score_threshold=0.01, max_meta_events=100)

    # Feed many chaining events to generate many meta-events
    prev = _event("node_0", "node_1", score=0.9)
    meta.observe(prev)
    for i in range(1, 20):
        curr = _event(f"node_{i}", f"node_{i+1}", score=0.9)
        meta.observe(curr)

    recent = meta.recent_meta_events(5)
    assert len(recent) <= 5


def test_ring_buffer_caps_at_max_meta_events():
    """MetaInsightEvents beyond max_meta_events drop the oldest."""
    meta = _engine(chain_score_threshold=0.01, max_meta_events=3)

    prev = _event("n_0", "n_1", score=0.9)
    meta.observe(prev)
    for i in range(1, 10):
        curr = _event(f"n_{i}", f"n_{i+1}", score=0.9)
        meta.observe(curr)

    assert meta.total_meta_events <= 3


def test_total_meta_events_increments():
    """total_meta_events increases as meta-events are fired."""
    meta = _engine()
    assert meta.total_meta_events == 0

    meta.observe(_event("A", "B"))
    meta.observe(_event("B", "C"))  # fires chain meta-event
    assert meta.total_meta_events >= 1


# ---------------------------------------------------------------------------
# MetaInsightEvent structure
# ---------------------------------------------------------------------------

def test_meta_insight_event_has_unique_id():
    """Two MetaInsightEvents have different IDs."""
    ev_a = MetaInsightEvent(
        insight_a_id="x", insight_b_id="y",
        connection_type="chain", meta_score=0.8,
    )
    ev_b = MetaInsightEvent(
        insight_a_id="x", insight_b_id="z",
        connection_type="chain", meta_score=0.8,
    )
    assert ev_a.id != ev_b.id


def test_meta_insight_event_timestamp_set():
    """MetaInsightEvent.timestamp is populated automatically."""
    before = time.time()
    ev = MetaInsightEvent(
        insight_a_id="a", insight_b_id="b",
        connection_type="shared_entity", meta_score=0.5,
    )
    after = time.time()
    assert before <= ev.timestamp <= after


def test_meta_insight_event_repr():
    """MetaInsightEvent.__repr__ includes type and depth."""
    ev = MetaInsightEvent(
        insight_a_id="abc", insight_b_id="def",
        connection_type="chain", meta_score=0.75, depth=2,
    )
    r = repr(ev)
    assert "chain" in r
    assert "depth=2" in r

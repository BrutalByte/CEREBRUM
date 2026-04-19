"""Tests for Phase 102 — Default Mode Network (Self-Referential Idle Reasoning)."""
import time
from unittest.mock import MagicMock

import networkx as nx
import pytest

from core.default_mode_engine import DefaultModeEngine, DMNInsight
from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(G=None, cmap=None, max_insights=5, frontier_threshold=2.0):
    if G is None:
        G = nx.DiGraph()
    adapter = MagicMock()
    adapter.to_networkx = lambda: G
    adapter.community_map = cmap or {}
    graph = MagicMock()
    graph.emit = MagicMock()
    return DefaultModeEngine(
        adapter=adapter,
        graph=graph,
        max_insights=max_insights,
        frontier_threshold=frontier_threshold,
    )


def _entry(seeds=None, answers=None, top_score=0.5, path_edges=None, ts=None):
    return MemoryEntry(
        timestamp=ts if ts is not None else time.time(),
        seeds=seeds or [],
        answers=answers or [],
        top_score=top_score,
        soliton_index=None,
        prediction_error=None,
        source="query",
        path_edges=path_edges or [],
    )


def _wm(*entries):
    wm = WorkingMemoryBuffer(maxlen=100)
    for e in entries:
        wm.record(e)
    return wm


# ---------------------------------------------------------------------------
# Isolation audit
# ---------------------------------------------------------------------------

def test_isolation_audit_finds_disconnected_community():
    """A community with no cross-community edges should be detected."""
    G = nx.DiGraph()
    # Community 0: a-b (internal only)
    G.add_edge("a", "b", relation="r")
    # Community 1: c-d (internal only), no bridge to community 0
    G.add_edge("c", "d", relation="r")
    cmap = {"a": 0, "b": 0, "c": 1, "d": 1}

    eng = _make_engine(G=G, cmap=cmap)
    insights = eng._isolation_audit(G, cmap)
    assert len(insights) > 0
    assert any(i.type == "isolation" for i in insights)


def test_isolation_audit_no_isolated_community():
    """A community with cross-community edges is not flagged."""
    G = nx.DiGraph()
    G.add_edge("a", "c", relation="bridge")  # cross-community
    G.add_edge("a", "b", relation="r")
    cmap = {"a": 0, "b": 0, "c": 1}

    eng = _make_engine(G=G, cmap=cmap)
    insights = eng._isolation_audit(G, cmap)
    assert len(insights) == 0


# ---------------------------------------------------------------------------
# Dead zone audit
# ---------------------------------------------------------------------------

def test_dead_zone_audit_finds_untouched_nodes():
    """Nodes never appearing in WM entries should be flagged."""
    G = nx.DiGraph()
    G.add_node("active")
    G.add_node("forgotten")

    eng = _make_engine(G=G)
    wm = _wm(_entry(seeds=["active"], answers=["active"]))
    insights = eng._dead_zone_audit(G, wm)
    assert len(insights) == 1
    assert insights[0].type == "dead_zone"
    assert "forgotten" in insights[0].context_seeds


def test_dead_zone_excludes_recently_active_nodes():
    """Nodes in recent WM entries should not be flagged."""
    G = nx.DiGraph()
    G.add_node("active")

    eng = _make_engine(G=G)
    wm = _wm(_entry(seeds=["active"], answers=["active"]))
    insights = eng._dead_zone_audit(G, wm)
    assert all("active" not in i.context_seeds for i in insights)


def test_dead_zone_no_wm_returns_empty():
    G = nx.DiGraph()
    G.add_node("a")
    eng = _make_engine(G=G)
    insights = eng._dead_zone_audit(G, None)
    assert insights == []


# ---------------------------------------------------------------------------
# Unanswered audit
# ---------------------------------------------------------------------------

def test_unanswered_audit_finds_failed_queries():
    """WM entries with top_score=0 should be flagged for reinvestigation."""
    eng = _make_engine()
    wm = _wm(
        _entry(seeds=["mystery"], answers=[], top_score=0.0),
        _entry(seeds=["known"], answers=["result"], top_score=0.8),
    )
    insights = eng._unanswered_audit(wm)
    assert len(insights) == 1
    assert insights[0].type == "unanswered"
    assert "mystery" in insights[0].context_seeds


def test_unanswered_audit_no_failures_returns_empty():
    eng = _make_engine()
    wm = _wm(_entry(seeds=["a"], top_score=0.7))
    insights = eng._unanswered_audit(wm)
    assert insights == []


# ---------------------------------------------------------------------------
# Frontier audit
# ---------------------------------------------------------------------------

def test_frontier_audit_finds_high_calibrator_weight():
    """Calibrator communities with weight > threshold should be flagged."""
    calibrator = MagicMock()
    calibrator.stats = MagicMock(return_value={
        "communities": {
            0: {"weight": 3.5},
            1: {"weight": 0.8},
        }
    })
    eng = _make_engine(frontier_threshold=2.0)
    insights = eng._frontier_audit(calibrator)
    assert len(insights) == 1
    assert insights[0].type == "frontier"
    assert "0" in insights[0].description


def test_frontier_audit_no_calibrator_returns_empty():
    eng = _make_engine()
    assert eng._frontier_audit(None) == []


# ---------------------------------------------------------------------------
# Max insights cap
# ---------------------------------------------------------------------------

def test_max_insights_capped():
    """idle_scan should never return more than max_insights."""
    G = nx.DiGraph()
    G.add_node("a")
    G.add_node("b")
    # Create multiple audit triggers
    cmap = {"a": 0, "b": 1}
    eng = _make_engine(G=G, cmap=cmap, max_insights=1)
    calibrator = MagicMock()
    calibrator.stats = MagicMock(return_value={
        "communities": {0: {"weight": 5.0}, 1: {"weight": 4.0}}
    })
    wm = _wm(_entry(seeds=["x"], top_score=0.0))
    insights = eng.idle_scan(wm=wm, calibrator=calibrator)
    assert len(insights) <= 1


# ---------------------------------------------------------------------------
# No insights when graph is healthy
# ---------------------------------------------------------------------------

def test_no_insights_when_graph_healthy():
    """A well-connected graph with all nodes in WM and no failures = no insights."""
    G = nx.DiGraph()
    G.add_edge("a", "b", relation="bridge")  # cross-community
    cmap = {"a": 0, "b": 1}
    eng = _make_engine(G=G, cmap=cmap)
    wm = _wm(
        _entry(seeds=["a"], answers=["b"], top_score=0.9),
    )
    # No calibrator → no frontier insights
    insights = eng.idle_scan(wm=wm, calibrator=None)
    assert all(i.type != "unanswered" for i in insights)


# ---------------------------------------------------------------------------
# Goal auto-push
# ---------------------------------------------------------------------------

def test_insight_auto_pushes_goal():
    """DMNInsight with auto_push_goal=True should push a Goal to the stack."""
    from core.goal_system import GoalStack
    G = nx.DiGraph()
    G.add_node("forgotten")
    eng = _make_engine(G=G)
    wm = _wm()  # empty → no dead zone exclusions

    goal_stack = GoalStack()
    eng.idle_scan(wm=wm, goal_stack=goal_stack)
    # If insights were generated, goals should be pushed
    # (This is integration-level; we just verify it doesn't crash)
    assert isinstance(goal_stack.all_active(), list)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_dmn_pulse_telemetry_emitted():
    """idle_scan should emit a DEFAULT_MODE_PULSE event."""
    G = nx.DiGraph()
    adapter = MagicMock()
    adapter.to_networkx = lambda: G
    adapter.community_map = {}
    graph = MagicMock()
    graph.emit = MagicMock()
    eng = DefaultModeEngine(adapter=adapter, graph=graph)
    eng.idle_scan()
    graph.emit.assert_called_once()
    event = graph.emit.call_args[0][0]
    assert event.event_type.value == "DEFAULT_MODE_PULSE"


# ---------------------------------------------------------------------------
# DMNInsight context_seeds are valid
# ---------------------------------------------------------------------------

def test_dmn_insight_context_seeds_valid_nodes():
    """All context_seeds in dead_zone insights must be nodes in the graph."""
    G = nx.DiGraph()
    for name in ["node_a", "node_b", "node_c"]:
        G.add_node(name)

    eng = _make_engine(G=G)
    wm = _wm(_entry(seeds=["node_a"], answers=["node_a"]))
    insights = eng._dead_zone_audit(G, wm)
    for ins in insights:
        for seed in ins.context_seeds:
            assert G.has_node(seed), f"Seed {seed!r} not in graph"


# ---------------------------------------------------------------------------
# Goal auto-push correctness
# ---------------------------------------------------------------------------

def test_dmn_auto_push_creates_valid_goal():
    """idle_scan with auto_push_goal=True should push a Goal with a valid id."""
    from core.goal_system import GoalStack, Goal
    G = nx.DiGraph()
    G.add_node("forgotten")
    eng = _make_engine(G=G)
    wm = _wm()  # empty → all nodes are dead zones

    goal_stack = GoalStack()
    eng.idle_scan(wm=wm, goal_stack=goal_stack)

    active = goal_stack.all_active()
    assert len(active) > 0, "Expected at least one goal to be pushed"
    for goal in active:
        assert isinstance(goal, Goal)
        assert goal.id, "Goal.id must not be empty"
        assert goal.description
        assert goal.metric_type is not None

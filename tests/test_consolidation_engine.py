"""Tests for core/consolidation_engine.py and Phase 96 Hebbian replay."""
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.consolidation_engine import ConsolidationEngine, ConsolidationResult
from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_adapter(edges=None):
    """Build a NetworkXAdapter backed by a simple DiGraph."""
    G = nx.DiGraph()
    for (u, rel, v, w) in (edges or []):
        G.add_edge(u, v, relation=rel, weight=w)
    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.update_edge_weight = NetworkXAdapter.update_edge_weight.__get__(adapter, NetworkXAdapter)
    return adapter


def _make_multi_adapter(edges=None):
    """Build a NetworkXAdapter backed by a MultiDiGraph."""
    G = nx.MultiDiGraph()
    for (u, rel, v, w) in (edges or []):
        G.add_edge(u, v, relation=rel, weight=w)
    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.update_edge_weight = NetworkXAdapter.update_edge_weight.__get__(adapter, NetworkXAdapter)
    return adapter


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


def _engine(adapter, min_score=0.4, max_weight=2.0, delta=0.1):
    graph = MagicMock()
    graph.emit = MagicMock()
    return ConsolidationEngine(
        adapter=adapter,
        graph=graph,
        min_score=min_score,
        max_weight=max_weight,
        hebbian_delta=delta,
    )


# ---------------------------------------------------------------------------
# update_edge_weight — NetworkXAdapter
# ---------------------------------------------------------------------------

def test_update_edge_weight_simple_graph():
    adapter = _make_simple_adapter([("a", "knows", "b", 1.0)])
    n = adapter.update_edge_weight("a", "b", "knows", delta=0.2, max_weight=2.0)
    assert n == 1
    assert abs(adapter._G["a"]["b"]["weight"] - 1.2) < 1e-6


def test_update_edge_weight_multigraph():
    adapter = _make_multi_adapter([("a", "knows", "b", 1.0), ("a", "likes", "b", 0.5)])
    n = adapter.update_edge_weight("a", "b", "likes", delta=0.3, max_weight=2.0)
    assert n == 1
    # find the "likes" edge
    weights = {v["relation"]: v["weight"] for v in adapter._G["a"]["b"].values()}
    assert abs(weights["likes"] - 0.8) < 1e-6
    assert abs(weights["knows"] - 1.0) < 1e-6  # unchanged


def test_update_edge_weight_unknown_edge():
    adapter = _make_simple_adapter([("a", "knows", "b", 1.0)])
    n = adapter.update_edge_weight("x", "y", "knows", delta=0.1)
    assert n == 0


def test_update_edge_weight_wrong_relation():
    adapter = _make_simple_adapter([("a", "knows", "b", 1.0)])
    n = adapter.update_edge_weight("a", "b", "hates", delta=0.1)
    assert n == 0
    assert adapter._G["a"]["b"]["weight"] == 1.0  # unchanged


def test_update_edge_weight_capped_at_max():
    adapter = _make_simple_adapter([("a", "knows", "b", 1.9)])
    adapter.update_edge_weight("a", "b", "knows", delta=0.5, max_weight=2.0)
    assert adapter._G["a"]["b"]["weight"] == 2.0


# ---------------------------------------------------------------------------
# MemoryEntry path_edges field
# ---------------------------------------------------------------------------

def test_working_memory_path_edges():
    wm = WorkingMemoryBuffer(maxlen=10)
    e = _entry(path_edges=[("a", "knows", "b"), ("b", "likes", "c")])
    wm.record(e)
    recent = wm.recent(5)
    assert recent[0].path_edges == [("a", "knows", "b"), ("b", "likes", "c")]


def test_memory_entry_path_edges_default_empty():
    e = _entry()
    assert e.path_edges == []


# ---------------------------------------------------------------------------
# ConsolidationEngine.consolidate
# ---------------------------------------------------------------------------

def test_consolidate_strengthens_edges():
    adapter = _make_simple_adapter([("newton", "discovered", "gravity", 1.0)])
    eng = _engine(adapter, min_score=0.4, delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=0.8, path_edges=[("newton", "discovered", "gravity")]))

    result = eng.consolidate(wm, k=5)

    assert result.entries_replayed == 1
    assert result.edges_strengthened == 1
    expected = min(2.0, 1.0 + 0.1 * 0.8)
    assert abs(adapter._G["newton"]["gravity"]["weight"] - expected) < 1e-6


def test_consolidate_score_proportional_delta():
    adapter = _make_simple_adapter([
        ("a", "r", "b", 1.0),
        ("c", "r", "d", 1.0),
    ])
    eng = _engine(adapter, min_score=0.3, delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=0.4, path_edges=[("a", "r", "b")]))
    wm.record(_entry(top_score=0.9, path_edges=[("c", "r", "d")]))

    eng.consolidate(wm, k=5)

    w_low = adapter._G["a"]["b"]["weight"]
    w_high = adapter._G["c"]["d"]["weight"]
    assert w_high > w_low  # higher-score entry → larger delta


def test_consolidate_weight_capped_at_max():
    adapter = _make_simple_adapter([("a", "r", "b", 1.95)])
    eng = _engine(adapter, max_weight=2.0, delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=1.0, path_edges=[("a", "r", "b")]))

    eng.consolidate(wm)
    assert adapter._G["a"]["b"]["weight"] == 2.0


def test_consolidate_below_threshold_skipped():
    adapter = _make_simple_adapter([("a", "r", "b", 1.0)])
    eng = _engine(adapter, min_score=0.7, delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=0.3, path_edges=[("a", "r", "b")]))

    result = eng.consolidate(wm)
    assert result.entries_replayed == 0
    assert result.edges_strengthened == 0
    assert adapter._G["a"]["b"]["weight"] == 1.0  # unchanged


def test_consolidate_empty_wm():
    adapter = _make_simple_adapter()
    eng = _engine(adapter)
    wm = WorkingMemoryBuffer(maxlen=10)

    result = eng.consolidate(wm)
    assert result.entries_replayed == 0
    assert result.edges_strengthened == 0
    assert result.mean_delta == 0.0


def test_consolidate_no_path_edges_skipped():
    adapter = _make_simple_adapter([("a", "r", "b", 1.0)])
    eng = _engine(adapter, min_score=0.3)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=0.9, path_edges=[]))  # no path_edges

    result = eng.consolidate(wm)
    assert result.entries_replayed == 0
    assert result.edges_strengthened == 0


def test_consolidate_k_cap():
    adapter = _make_simple_adapter([
        (f"a{i}", "r", f"b{i}", 1.0) for i in range(10)
    ])
    eng = _engine(adapter, min_score=0.3, delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=20)
    for i in range(10):
        wm.record(_entry(top_score=0.8, path_edges=[(f"a{i}", "r", f"b{i}")]))

    result = eng.consolidate(wm, k=3)
    assert result.entries_replayed == 3


def test_consolidate_emits_telemetry():
    adapter = _make_simple_adapter([("a", "r", "b", 1.0)])
    graph = MagicMock()
    graph.emit = MagicMock()
    eng = ConsolidationEngine(adapter=adapter, graph=graph, min_score=0.3, hebbian_delta=0.1)
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(top_score=0.8, path_edges=[("a", "r", "b")]))

    eng.consolidate(wm)
    graph.emit.assert_called_once()
    event = graph.emit.call_args[0][0]
    assert event.event_type.value == "CONSOLIDATION_PULSE"


def test_consolidate_result_duration_positive():
    adapter = _make_simple_adapter()
    eng = _engine(adapter)
    wm = WorkingMemoryBuffer(maxlen=10)
    result = eng.consolidate(wm)
    assert result.duration >= 0.0

"""Tests for core/synaptic_decay_engine.py — Phase 97 LTD / Synaptic Homeostasis."""
import time
from unittest.mock import MagicMock

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.synaptic_decay_engine import DecayResult, SynapticDecayEngine
from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(edges):
    """Build a minimal adapter backed by a DiGraph."""
    G = nx.DiGraph()
    for (u, rel, v, w) in edges:
        G.add_edge(u, v, relation=rel, weight=w)
    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.update_edge_weight = NetworkXAdapter.update_edge_weight.__get__(adapter, NetworkXAdapter)
    adapter.to_networkx = lambda: G
    return adapter


def _make_engine(adapter, **kwargs):
    graph = MagicMock()
    graph.emit = MagicMock()
    return SynapticDecayEngine(adapter=adapter, graph=graph, **kwargs)


def _entry(path_edges=None, top_score=0.5, source="query"):
    return MemoryEntry(
        timestamp=time.time(),
        seeds=[],
        answers=[],
        top_score=top_score,
        soliton_index=None,
        prediction_error=None,
        source=source,
        path_edges=path_edges or [],
    )


# ---------------------------------------------------------------------------
# DecayResult fields
# ---------------------------------------------------------------------------

def test_decay_result_fields():
    adapter = _make_adapter([("a", "r", "b", 1.0)])
    eng = _make_engine(adapter)
    result = eng.decay()
    assert hasattr(result, "edges_processed")
    assert hasattr(result, "edges_decayed")
    assert hasattr(result, "edges_resisted")
    assert hasattr(result, "mean_delta")
    assert hasattr(result, "duration")
    assert result.duration >= 0.0


# ---------------------------------------------------------------------------
# Decay direction
# ---------------------------------------------------------------------------

def test_decay_reduces_high_weight():
    """An edge above baseline should decay toward baseline."""
    adapter = _make_adapter([("a", "r", "b", 1.5)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    eng.decay()
    assert adapter._G["a"]["b"]["weight"] < 1.5


def test_decay_raises_low_weight():
    """An edge below baseline should be nudged upward."""
    adapter = _make_adapter([("a", "r", "b", 0.6)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1, min_weight=0.0)
    eng.decay()
    assert adapter._G["a"]["b"]["weight"] > 0.6


def test_decay_baseline_stable():
    """An edge exactly at baseline produces near-zero delta."""
    adapter = _make_adapter([("a", "r", "b", 1.0)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    eng.decay()
    assert abs(adapter._G["a"]["b"]["weight"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Floor enforcement
# ---------------------------------------------------------------------------

def test_decay_floor_enforced():
    """Weight must never drop below min_weight."""
    adapter = _make_adapter([("a", "r", "b", 0.51)])
    eng = _make_engine(adapter, baseline_weight=0.0, decay_rate=1.0, min_weight=0.5)
    eng.decay()
    assert adapter._G["a"]["b"]["weight"] >= 0.5


def test_min_weight_param_networkx():
    """NetworkXAdapter.update_edge_weight respects min_weight floor."""
    adapter = _make_adapter([("a", "r", "b", 0.6)])
    n = adapter.update_edge_weight("a", "b", "r", delta=-0.5, min_weight=0.4)
    assert n == 1
    assert adapter._G["a"]["b"]["weight"] >= 0.4


# ---------------------------------------------------------------------------
# Traversal frequency resistance
# ---------------------------------------------------------------------------

def test_decay_resistance_proportional_to_frequency():
    """Frequently-traversed edge decays less than an unused edge."""
    adapter = _make_adapter([
        ("hot", "r", "b", 1.5),
        ("cold", "r", "b", 1.5),
    ])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    wm = WorkingMemoryBuffer(maxlen=20)
    for _ in range(10):
        wm.record(_entry(path_edges=[("hot", "r", "b")]))

    eng.decay(wm)

    hot_w = adapter._G["hot"]["b"]["weight"]
    cold_w = adapter._G["cold"]["b"]["weight"]
    # hot edge resists decay → stays higher
    assert hot_w > cold_w


def test_decay_empty_wm_uniform_decay():
    """With no WM, all edges decay uniformly (zero frequency)."""
    adapter = _make_adapter([("a", "r", "b", 1.5), ("c", "r", "d", 1.5)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    eng.decay()
    wa = adapter._G["a"]["b"]["weight"]
    wc = adapter._G["c"]["d"]["weight"]
    assert abs(wa - wc) < 1e-9


# ---------------------------------------------------------------------------
# Source-weighted frequency (Phase 98 Gap 8)
# ---------------------------------------------------------------------------

def test_decay_source_weights_dissonance_accelerates():
    """Dissonance edges decay faster than approval edges (same traversal count).

    Mechanism: dissonance multiplier (0.5) < approval multiplier (3.0).
    Lower multiplier → lower effective frequency → lower resistance → more decay.
    """
    adapter = _make_adapter([
        ("a", "r", "b", 1.5),  # traversed with source="approval"
        ("c", "r", "d", 1.5),  # traversed with source="dissonance"
    ])
    wm = WorkingMemoryBuffer(maxlen=40)
    for _ in range(4):
        wm.record(_entry(path_edges=[("a", "r", "b")], source="approval"))
        wm.record(_entry(path_edges=[("c", "r", "d")], source="dissonance"))

    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    eng.decay(wm)

    approval_w = adapter._G["a"]["b"]["weight"]
    dissonance_w = adapter._G["c"]["d"]["weight"]
    # approval: multiplier=3.0 → high effective freq → high resistance → less decay
    # dissonance: multiplier=0.5 → low effective freq → low resistance → more decay
    assert approval_w > dissonance_w

    # Verify frequency map directly: approval entries contribute more freq than dissonance
    wm_approval = WorkingMemoryBuffer(maxlen=20)
    wm_dissonance = WorkingMemoryBuffer(maxlen=20)
    for _ in range(4):
        wm_approval.record(_entry(path_edges=[("a", "r", "b")], source="approval"))
        wm_dissonance.record(_entry(path_edges=[("c", "r", "d")], source="dissonance"))
    assert (
        eng._build_frequency_map(wm_approval)[("a", "r", "b")]
        > eng._build_frequency_map(wm_dissonance)[("c", "r", "d")]
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def test_decay_emits_telemetry():
    adapter = _make_adapter([("a", "r", "b", 1.0)])
    graph = MagicMock()
    graph.emit = MagicMock()
    eng = SynapticDecayEngine(adapter=adapter, graph=graph)
    eng.decay()
    graph.emit.assert_called_once()
    event = graph.emit.call_args[0][0]
    assert event.event_type.value == "SYNAPTIC_DECAY"


def test_decay_with_none_wm():
    """decay(wm=None) should run with uniform frequency (no WM data) and not raise."""
    adapter = _make_adapter([("a", "r", "b", 1.5), ("c", "r", "d", 0.5)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.1)
    result = eng.decay(wm=None)
    assert isinstance(result, DecayResult)
    # Both edges decay uniformly (no frequency data)
    assert result.edges_processed >= 0
    wa = adapter._G["a"]["b"]["weight"]
    wc = adapter._G["c"]["d"]["weight"]
    # Above-baseline edge decayed toward 1.0; below-baseline edge grew toward 1.0
    assert wa < 1.5
    assert wc > 0.5


# ---------------------------------------------------------------------------
# Integration: LTP then LTD converges
# ---------------------------------------------------------------------------

def test_decay_engine_integrates_with_consolidation():
    """After LTP boosts a weight, LTD should decay it back toward baseline over passes."""
    adapter = _make_adapter([("a", "r", "b", 1.0)])
    eng = _make_engine(adapter, baseline_weight=1.0, decay_rate=0.2, min_weight=0.5)

    # Simulate LTP: manually boost the edge
    adapter.update_edge_weight("a", "b", "r", delta=0.5, max_weight=2.0)
    assert adapter._G["a"]["b"]["weight"] > 1.0

    # Run 20 decay passes with no WM (full decay)
    for _ in range(20):
        eng.decay()

    # Should converge toward baseline
    assert abs(adapter._G["a"]["b"]["weight"] - 1.0) < 0.05

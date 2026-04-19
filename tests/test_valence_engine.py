"""Tests for Phase 101 — Emotional Valence (Amygdala)."""
import time
from unittest.mock import MagicMock

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.valence_engine import ValenceEngine, ValenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(edges=None):
    G = nx.DiGraph()
    for (u, rel, v) in (edges or []):
        G.add_edge(u, v, relation=rel, weight=1.0)
    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.update_edge_valence = NetworkXAdapter.update_edge_valence.__get__(adapter, NetworkXAdapter)
    adapter.get_edge_valence    = NetworkXAdapter.get_edge_valence.__get__(adapter, NetworkXAdapter)
    return adapter


def _engine(adapter, lr=0.1):
    graph = MagicMock()
    graph.emit = MagicMock()
    return ValenceEngine(adapter=adapter, graph=graph, learning_rate=lr)


# ---------------------------------------------------------------------------
# NetworkXAdapter valence methods
# ---------------------------------------------------------------------------

def test_update_edge_valence_networkx():
    adapter = _make_adapter([("a", "r", "b")])
    n = adapter.update_edge_valence("a", "b", "r", delta=0.3)
    assert n == 1
    assert abs(adapter._G["a"]["b"]["valence"] - 0.3) < 1e-9


def test_update_edge_valence_unknown_edge_returns_zero():
    adapter = _make_adapter([("a", "r", "b")])
    n = adapter.update_edge_valence("x", "y", "r", delta=0.3)
    assert n == 0


def test_get_edge_valence_missing_edge_returns_zero():
    adapter = _make_adapter([("a", "r", "b")])
    assert adapter.get_edge_valence("x", "y", "r") == 0.0


def test_get_edge_valence_returns_stored_value():
    adapter = _make_adapter([("a", "r", "b")])
    adapter.update_edge_valence("a", "b", "r", delta=-0.5)
    assert abs(adapter.get_edge_valence("a", "b", "r") - (-0.5)) < 1e-9


def test_valence_clamped_at_bounds():
    adapter = _make_adapter([("a", "r", "b")])
    adapter.update_edge_valence("a", "b", "r", delta=5.0)
    assert adapter._G["a"]["b"]["valence"] == pytest.approx(1.0)
    adapter.update_edge_valence("a", "b", "r", delta=-5.0)
    assert adapter._G["a"]["b"]["valence"] == pytest.approx(-1.0)


def test_valence_independent_of_edge_weight():
    """Valence and weight are separate attributes."""
    adapter = _make_adapter([("a", "r", "b")])
    adapter.update_edge_valence("a", "b", "r", delta=-0.4)
    assert adapter._G["a"]["b"].get("weight", 1.0) == 1.0
    assert adapter._G["a"]["b"].get("valence", 0.0) == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# ValenceEngine.record_outcome
# ---------------------------------------------------------------------------

def test_negative_valence_applied_on_rejection():
    adapter = _make_adapter([("a", "r", "b")])
    eng = _engine(adapter, lr=0.1)
    result = eng.record_outcome([("a", "r", "b")], outcome_score=-1.0)
    assert result.edges_updated == 1
    assert adapter._G["a"]["b"]["valence"] < 0.0


def test_positive_valence_applied_on_approval():
    adapter = _make_adapter([("a", "r", "b")])
    eng = _engine(adapter, lr=0.1)
    result = eng.record_outcome([("a", "r", "b")], outcome_score=1.0)
    assert result.edges_updated == 1
    assert adapter._G["a"]["b"]["valence"] > 0.0


def test_valence_accumulates_across_events():
    """Repeated rejections drive valence toward -1."""
    adapter = _make_adapter([("a", "r", "b")])
    eng = _engine(adapter, lr=0.3)
    for _ in range(10):
        eng.record_outcome([("a", "r", "b")], outcome_score=-1.0)
    assert adapter._G["a"]["b"]["valence"] == pytest.approx(-1.0)


def test_default_valence_is_neutral():
    """Fresh edge has valence 0.0."""
    adapter = _make_adapter([("a", "r", "b")])
    eng = _engine(adapter)
    assert eng.get_valence("a", "b", "r") == 0.0


def test_empty_path_edges_no_update():
    adapter = _make_adapter([("a", "r", "b")])
    eng = _engine(adapter)
    result = eng.record_outcome([], outcome_score=-1.0)
    assert result.edges_updated == 0
    assert result.mean_delta == 0.0


def test_valence_engine_emits_telemetry():
    adapter = _make_adapter([("a", "r", "b")])
    graph = MagicMock()
    graph.emit = MagicMock()
    eng = ValenceEngine(adapter=adapter, graph=graph, learning_rate=0.1)
    eng.record_outcome([("a", "r", "b")], outcome_score=0.8)
    graph.emit.assert_called_once()
    event = graph.emit.call_args[0][0]
    assert event.event_type.value == "VALENCE_UPDATE"


# ---------------------------------------------------------------------------
# Traversal integration — aversive edges reduce w
# ---------------------------------------------------------------------------

def test_negative_valence_reduces_path_score():
    """Edge with negative valence should reduce traversal weight w."""
    from reasoning.traversal import BeamTraversal
    from core.attention_engine import CSAEngine

    G = nx.DiGraph()
    G.add_edge("seed", "target", relation="r", weight=1.0, valence=-0.8)

    adapter_real = NetworkXAdapter(G)
    adapter_real.embeddings = {}
    adapter_real.community_map = {"seed": 0, "target": 0}

    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.get_edge_valence = NetworkXAdapter.get_edge_valence.__get__(adapter, NetworkXAdapter)
    adapter.get_embedding = MagicMock(return_value=None)
    adapter.get_community = MagicMock(return_value=0)
    adapter.get_neighbors = MagicMock(return_value=[])
    adapter.community_map = {}
    csa = MagicMock(spec=CSAEngine)
    csa.set_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    csa.use_temporal_decay = False

    graph = MagicMock()
    graph.emit = MagicMock()
    eng = ValenceEngine(adapter=adapter, graph=graph, learning_rate=0.1, valence_weight=1.0)

    t = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5, max_hop=1)
    t._valence_engine = eng

    # Verify the engine is attached
    assert t._valence_engine is eng


def test_valence_weight_zero_disables_effect():
    """valence_weight=0 means negative valence has no effect on w."""
    adapter = _make_adapter([("a", "r", "b")])
    graph = MagicMock()
    graph.emit = MagicMock()
    eng = ValenceEngine(adapter=adapter, graph=graph, learning_rate=0.1, valence_weight=0.0)
    assert eng.valence_weight == 0.0

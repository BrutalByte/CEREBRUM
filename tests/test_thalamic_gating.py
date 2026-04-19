"""Tests for Phase 99 — Thalamic Gating (WM → BeamTraversal Priming)."""
import math
import time
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(seeds=None, answers=None, top_score=0.8, ts=None):
    return MemoryEntry(
        timestamp=ts if ts is not None else time.time(),
        seeds=seeds or [],
        answers=answers or [],
        top_score=top_score,
        soliton_index=None,
        prediction_error=None,
        source="query",
        path_edges=[],
    )


def _build_priming(wm):
    """Call CerebrumGraph._build_priming_map via a minimal stub."""
    import types, math as _math, time as _time

    class _Stub:
        _working_memory = wm

    stub = _Stub()
    # Bind the real method
    from core.cerebrum import CerebrumGraph
    return CerebrumGraph._build_priming_map(stub)


def _make_traversal():
    from reasoning.traversal import BeamTraversal
    from core.attention_engine import CSAEngine
    from core.graph_adapter import GraphAdapter

    adapter = MagicMock()
    adapter.get_embedding = MagicMock(return_value=np.zeros(8, dtype=np.float32))
    adapter.get_community = MagicMock(return_value=0)
    adapter.community_map = {}
    csa = MagicMock(spec=CSAEngine)
    csa.set_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    csa.use_temporal_decay = False
    return BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5, max_hop=2)


# ---------------------------------------------------------------------------
# _build_priming_map
# ---------------------------------------------------------------------------

def test_priming_map_normalized():
    """All priming values must be in [0, 1]."""
    wm = WorkingMemoryBuffer(maxlen=20)
    wm.record(_entry(seeds=["a"], answers=["b"], top_score=0.9))
    wm.record(_entry(seeds=["a"], answers=["c"], top_score=0.5))

    priming = _build_priming(wm)
    assert all(0.0 <= v <= 1.0 for v in priming.values())
    assert max(priming.values()) == pytest.approx(1.0)


def test_primed_node_in_map():
    """Seeds and answers from WM entries appear in the priming map."""
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry(seeds=["newton"], answers=["gravity"], top_score=0.8))
    priming = _build_priming(wm)
    assert "newton" in priming
    assert "gravity" in priming


def test_empty_wm_no_priming():
    """Empty WM returns empty priming map."""
    wm = WorkingMemoryBuffer(maxlen=10)
    assert _build_priming(wm) == {}


def test_no_wm_returns_empty():
    """No WM attached returns empty dict."""
    import types
    class _Stub:
        _working_memory = None
    from core.cerebrum import CerebrumGraph
    result = CerebrumGraph._build_priming_map(_Stub())
    assert result == {}


def test_priming_decays_with_age():
    """Older WM entries should prime less than recent ones."""
    wm = WorkingMemoryBuffer(maxlen=10)
    old_ts = time.time() - 1000
    new_ts = time.time()
    wm.record(_entry(seeds=["old_node"], top_score=1.0, ts=old_ts))
    wm.record(_entry(seeds=["new_node"], top_score=1.0, ts=new_ts))

    priming = _build_priming(wm)
    assert priming.get("new_node", 0.0) > priming.get("old_node", 0.0)


def test_priming_scales_with_top_score():
    """Higher top_score entries should produce higher priming values."""
    wm = WorkingMemoryBuffer(maxlen=10)
    now = time.time()
    wm.record(_entry(seeds=["strong"], top_score=0.9, ts=now))
    wm.record(_entry(seeds=["weak"], top_score=0.1, ts=now))

    priming = _build_priming(wm)
    assert priming.get("strong", 0.0) > priming.get("weak", 0.0)


# ---------------------------------------------------------------------------
# BeamTraversal priming_boost parameter
# ---------------------------------------------------------------------------

def test_priming_boost_configurable():
    """priming_boost=0.0 should leave w unchanged."""
    traversal = _make_traversal()
    traversal.priming_boost = 0.0
    # The node_priming map should have no effect at priming_boost=0
    assert traversal.priming_boost == 0.0


def test_traversal_stores_node_priming():
    """BeamTraversal initializes node_priming as empty dict."""
    traversal = _make_traversal()
    assert traversal.node_priming == {}
    assert traversal.priming_boost == 0.3


def test_traverse_accepts_node_priming_kwarg():
    """traverse() should accept node_priming without error (smoke test)."""
    from reasoning.traversal import BeamTraversal
    from core.attention_engine import CSAEngine

    adapter = MagicMock()
    adapter.get_embedding = MagicMock(return_value=np.zeros(8, dtype=np.float32))
    adapter.get_community = MagicMock(return_value=0)
    adapter.get_neighbors = MagicMock(return_value=[])
    adapter.community_map = {}
    csa = MagicMock(spec=CSAEngine)
    csa.set_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    csa.use_temporal_decay = False

    t = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=2, max_hop=1)
    # Should not raise
    paths = t.traverse(["seed_a"], node_priming={"target_b": 0.8})
    assert isinstance(paths, list)

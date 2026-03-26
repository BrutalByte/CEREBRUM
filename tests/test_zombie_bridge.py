"""
tests/test_zombie_bridge.py
Hole 1 — Zombie Bridge fix: BridgeTwinEngine.on_rebalance hook.

Validates that on_rebalance() prunes stale BridgeRecords after a GlobalRebalancer
DSCF re-run produces a new community_map with different community IDs.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest

from core.bridge_engine import BridgeTwinEngine, BridgeRecord
from core.rebalancer import GlobalRebalancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> BridgeTwinEngine:
    return BridgeTwinEngine(n_min=2, similarity_threshold=0.0)


def _inject_bridge(engine: BridgeTwinEngine, original: str, twin: str,
                   src_cid: int, dst_cid: int) -> None:
    """Directly inject a BridgeRecord into the engine (bypasses graph)."""
    record = BridgeRecord(
        original_id=original,
        twin_id=twin,
        source_community=src_cid,
        destination_community=dst_cid,
        traversal_count=5,
    )
    engine._bridges[twin] = record
    engine._bridge_index[(original, dst_cid)] = twin
    engine._candidates[(original, dst_cid)] = 5


# ---------------------------------------------------------------------------
# BridgeTwinEngine.on_rebalance — core behavior
# ---------------------------------------------------------------------------

def test_on_rebalance_empty_bridges():
    """No bridges → returns 0, no crash."""
    engine = _make_engine()
    n = engine.on_rebalance({"A": 0, "B": 1})
    assert n == 0


def test_on_rebalance_returns_count():
    """Return value equals number of stale bridges pruned."""
    engine = _make_engine()
    # Two bridges: one stale, one valid
    _inject_bridge(engine, "A", "A::twin::5", src_cid=0, dst_cid=5)
    _inject_bridge(engine, "B", "B::twin::1", src_cid=2, dst_cid=1)

    # New map: community 5 is gone; A is now in cid 3; B::twin::1 stays at 1, B at 2
    new_map = {"A": 3, "B": 2, "B::twin::1": 1}
    n = engine.on_rebalance(new_map)
    assert n == 1  # A::twin::5 pruned (both src and dst stale)


def test_on_rebalance_prunes_stale_source_community():
    """Source community ID changed → bridge pruned."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::2", src_cid=0, dst_cid=2)

    # A's community changed from 0 → 7
    new_map = {"A": 7, "A::twin::2": 2}
    n = engine.on_rebalance(new_map)
    assert n == 1
    assert "A::twin::2" not in engine._bridges


def test_on_rebalance_prunes_stale_dest_community():
    """Destination community ID changed for the twin node → bridge pruned."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::2", src_cid=0, dst_cid=2)

    # twin's community changed from 2 → 8 (dest stale)
    new_map = {"A": 0, "A::twin::2": 8}
    n = engine.on_rebalance(new_map)
    assert n == 1
    assert "A::twin::2" not in engine._bridges


def test_on_rebalance_keeps_valid_bridges():
    """Bridge whose src and dst communities survive → kept."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::2", src_cid=0, dst_cid=2)

    # Both IDs still match in new map
    new_map = {"A": 0, "A::twin::2": 2}
    n = engine.on_rebalance(new_map)
    assert n == 0
    assert "A::twin::2" in engine._bridges


def test_on_rebalance_all_stale():
    """All bridges stale → all pruned, _bridges and _bridge_index empty."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::5", src_cid=0, dst_cid=5)
    _inject_bridge(engine, "B", "B::twin::3", src_cid=1, dst_cid=3)

    # New map has completely different IDs
    new_map = {"A": 10, "B": 11}
    n = engine.on_rebalance(new_map)
    assert n == 2
    assert len(engine._bridges) == 0
    assert len(engine._bridge_index) == 0


def test_on_rebalance_updates_bridge_index():
    """Stale entry removed from _bridge_index."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::5", src_cid=0, dst_cid=5)

    new_map = {"A": 99}  # dst community 5 gone
    engine.on_rebalance(new_map)
    assert ("A", 5) not in engine._bridge_index


def test_on_rebalance_leaves_candidates_intact():
    """_candidates crossing counts are NOT cleared by on_rebalance."""
    engine = _make_engine()
    _inject_bridge(engine, "A", "A::twin::5", src_cid=0, dst_cid=5)

    new_map = {"A": 99}
    engine.on_rebalance(new_map)
    # _candidates still has the crossing count
    assert engine._candidates.get(("A", 5), 0) == 5


# ---------------------------------------------------------------------------
# GlobalRebalancer bridge_engine param
# ---------------------------------------------------------------------------

def test_rebalancer_bridge_engine_param():
    """Constructor stores bridge_engine correctly."""
    bridge = _make_engine()
    adapter = MagicMock()
    adapter.community_map = {}
    adapter._G = MagicMock()
    r = GlobalRebalancer(adapter, bridge_engine=bridge)
    assert r._bridge_engine is bridge


def test_rebalancer_no_hook_when_none():
    """bridge_engine=None (default) → no AttributeError during rebalance worker."""
    adapter = MagicMock()
    adapter.community_map = {"A": 0}
    adapter._G = MagicMock()
    adapter._G.number_of_nodes.return_value = 0

    r = GlobalRebalancer(adapter, bridge_engine=None)
    # _rebalance_worker with empty graph should return early — no crash
    r._rebalance_worker()


def test_rebalancer_calls_hook_on_rebalance_worker():
    """After _rebalance_worker commits new map, bridge hook is called."""
    import networkx as nx
    G = nx.path_graph(4)
    adapter = MagicMock()
    adapter._G = G
    adapter._lock = threading.RLock()
    adapter.community_map = {str(n): 0 for n in G.nodes()}

    bridge_engine = MagicMock()
    bridge_engine.on_rebalance.return_value = 0

    r = GlobalRebalancer(adapter, n_dscf_trials=1, bridge_engine=bridge_engine)
    r._rebalance_worker()

    # on_rebalance should have been called once with the new community_map
    bridge_engine.on_rebalance.assert_called_once()
    called_map = bridge_engine.on_rebalance.call_args[0][0]
    assert isinstance(called_map, dict)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_on_rebalance_thread_safety():
    """Concurrent on_rebalance and _inject_bridge calls don't corrupt state."""
    engine = _make_engine()
    for i in range(20):
        _inject_bridge(engine, f"N{i}", f"N{i}::twin::{i}", src_cid=i % 3, dst_cid=i)

    errors = []

    def rebalance_loop():
        new_map = {f"N{i}": i % 3 for i in range(20)}
        for _ in range(50):
            try:
                engine.on_rebalance(new_map)
            except Exception as e:
                errors.append(e)

    def inject_loop():
        for i in range(100, 120):
            try:
                _inject_bridge(engine, f"M{i}", f"M{i}::twin::{i}", src_cid=0, dst_cid=i)
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=rebalance_loop)
    t2 = threading.Thread(target=inject_loop)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert len(errors) == 0, f"Thread safety errors: {errors}"

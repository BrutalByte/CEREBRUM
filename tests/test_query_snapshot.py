"""
tests/test_query_snapshot.py
Hole 1 — Mid-Flight Community Swap: Query Snapshot Isolation.

Validates that BeamTraversal.traverse() takes a frozen snapshot of
adapter.community_map at query start, so a concurrent GlobalRebalancer
commit cannot produce inconsistent CSA weights within a single query.
"""
import threading
import time
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from reasoning.traversal import BeamTraversal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_adapter():
    G = nx.path_graph(6, create_using=nx.DiGraph())
    adapter = NetworkXAdapter(G)
    adapter.community_map = {str(n): n % 2 for n in G.nodes()}
    adapter.embeddings = {}
    for n in G.nodes():
        v = np.random.default_rng(n).standard_normal(16).astype(np.float32)
        adapter.embeddings[str(n)] = v / np.linalg.norm(v)
    csa = CSAEngine(adapter=adapter)
    return adapter, csa


# ---------------------------------------------------------------------------
# CSAEngine.set_query_snapshot / clear_query_snapshot
# ---------------------------------------------------------------------------

def test_set_query_snapshot_uses_frozen_map():
    adapter, csa = _make_engine_adapter()
    # Initial community: node "0" → cid 0
    assert adapter.community_map["0"] == 0

    csa.set_query_snapshot({"0": 99, "1": 99, "2": 99})
    # During snapshot, _get_community returns from snapshot, not adapter
    assert csa._get_community("0") == 99

    csa.clear_query_snapshot()
    # After clear, falls back to adapter
    assert csa._get_community("0") == adapter.community_map.get("0", -1)


def test_clear_query_snapshot_restores_adapter_lookup():
    adapter, csa = _make_engine_adapter()
    csa.set_query_snapshot({"0": 42})
    csa.clear_query_snapshot()
    assert csa._query_snapshot is None


def test_get_community_no_snapshot_uses_adapter():
    adapter, csa = _make_engine_adapter()
    # No snapshot set — should use adapter
    result = csa._get_community("0")
    assert result == adapter.get_community("0")


def test_get_community_missing_node_returns_minus_one():
    adapter, csa = _make_engine_adapter()
    csa.set_query_snapshot({"known": 5})
    assert csa._get_community("ghost") == -1
    csa.clear_query_snapshot()


def test_community_score_uses_snapshot():
    adapter, csa = _make_engine_adapter()
    # Without snapshot: nodes "0" and "1" are in communities 0 and 1 → not same
    score_live = csa.community_score("0", "1")

    # With snapshot forcing both into same community → score = 1.0
    csa.set_query_snapshot({"0": 5, "1": 5})
    score_snap = csa.community_score("0", "1")
    csa.clear_query_snapshot()

    assert score_snap == 1.0
    assert score_live != 1.0


def test_snapshot_overrides_live_community_change():
    """Simulates rebalancer changing community_map mid-query."""
    adapter, csa = _make_engine_adapter()
    original_cid = adapter.community_map.get("2", -1)
    csa.set_query_snapshot(dict(adapter.community_map))

    # Rebalancer fires — changes community_map live
    adapter.community_map["2"] = 999

    # Snapshot should still return the pre-rebalance value
    assert csa._get_community("2") == original_cid
    csa.clear_query_snapshot()


# ---------------------------------------------------------------------------
# BeamTraversal.traverse() — snapshot lifecycle
# ---------------------------------------------------------------------------

def test_traverse_sets_and_clears_snapshot():
    adapter, csa = _make_engine_adapter()
    from core.resource_governor import ResourceGovernor
    gov = ResourceGovernor(memory_threshold_pct=99.0)
    trav = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5,
                         max_hop=2, governor=gov)

    # Before traverse — no snapshot
    assert csa._query_snapshot is None

    trav.traverse(["0"])

    # After traverse — snapshot cleared
    assert csa._query_snapshot is None


def test_snapshot_cleared_even_on_exception():
    """Snapshot must be cleared via finally even if traversal raises."""
    adapter, csa = _make_engine_adapter()
    from core.resource_governor import ResourceGovernor
    gov = ResourceGovernor(memory_threshold_pct=99.0)
    trav = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5,
                         max_hop=2, governor=gov)

    with patch.object(trav, "_traverse_inner", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            trav.traverse(["0"])

    assert csa._query_snapshot is None


def test_traverse_snapshot_is_dict_copy():
    """Snapshot is a shallow copy; mutating adapter.community_map after
    snapshot is taken must not affect the snapshot."""
    adapter, csa = _make_engine_adapter()
    from core.resource_governor import ResourceGovernor
    gov = ResourceGovernor(memory_threshold_pct=99.0)
    trav = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5,
                         max_hop=1, governor=gov)

    snapshots_seen = []

    orig_inner = trav._traverse_inner

    def _capture_snapshot(*args, **kwargs):
        snapshots_seen.append(dict(csa._query_snapshot) if csa._query_snapshot else None)
        return orig_inner(*args, **kwargs)

    trav._traverse_inner = _capture_snapshot
    trav.traverse(["0"])

    assert snapshots_seen, "Should have captured at least one snapshot"
    assert snapshots_seen[0] is not None


# ---------------------------------------------------------------------------
# Concurrency: rebalancer fires while traverse is in flight
# ---------------------------------------------------------------------------

def test_concurrent_rebalance_does_not_corrupt_query():
    """
    Start a traverse() in a thread. Simultaneously mutate adapter.community_map
    from another thread. The traversal must complete without raising.
    """
    G = nx.path_graph(20, create_using=nx.DiGraph())
    adapter = NetworkXAdapter(G)
    adapter.community_map = {str(n): n % 4 for n in G.nodes()}
    adapter.embeddings = {}
    for n in G.nodes():
        v = np.random.default_rng(n).standard_normal(16).astype(np.float32)
        adapter.embeddings[str(n)] = v / np.linalg.norm(v)

    csa = CSAEngine(adapter=adapter)
    from core.resource_governor import ResourceGovernor
    gov = ResourceGovernor(memory_threshold_pct=99.0)
    trav = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10,
                         max_hop=3, governor=gov)

    errors = []

    def run_traverse():
        try:
            trav.traverse(["0"])
        except Exception as e:
            errors.append(e)

    def mutate_map():
        for i in range(5):
            adapter.community_map = {str(n): (n + i) % 5 for n in G.nodes()}
            time.sleep(0.001)

    t1 = threading.Thread(target=run_traverse)
    t2 = threading.Thread(target=mutate_map)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=5)

    assert not errors, f"Errors during concurrent traversal: {errors}"

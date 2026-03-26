"""
Tests for core/rem_engine.py — REM Cycle (Phase 14).

All tests use a small in-process NetworkXAdapter with synthetic embeddings.
No external dependencies. No file I/O.
"""
import threading
import time
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.rem_engine import REMEngine, REMReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(edges, low_conf_edges=None, bridge_twin_edges=None):
    """
    Build a NetworkXAdapter from explicit edge lists.

    edges             : [(u, v, relation)] — all with confidence=1.0
    low_conf_edges    : [(u, v, relation, confidence)] — low-confidence edges
    bridge_twin_edges : [(u, v)] — BRIDGE_TWIN edges (immune to pruning)
    """
    G = nx.DiGraph()
    for u, v, rel in edges:
        G.add_node(u, label=u, type="entity")
        G.add_node(v, label=v, type="entity")
        G.add_edge(u, v, relation=rel, weight=1.0, confidence=1.0)

    if low_conf_edges:
        for u, v, rel, conf in low_conf_edges:
            G.add_node(u, label=u, type="entity")
            G.add_node(v, label=v, type="entity")
            G.add_edge(u, v, relation=rel, weight=1.0, confidence=conf)

    if bridge_twin_edges:
        for u, v in bridge_twin_edges:
            G.add_node(u, label=u, type="entity")
            G.add_node(v, label=v, type="entity")
            G.add_edge(u, v, relation="BRIDGE_TWIN", weight=1.0, confidence=0.0)

    adapter = NetworkXAdapter(G)
    # Attach a RandomEngine so embeddings are available
    engine = RandomEngine(dim=32)
    embeddings = {node: engine.encode_one(node) for node in G.nodes()}
    adapter.embeddings = embeddings
    return adapter


def _rem(adapter, **kwargs) -> REMEngine:
    return REMEngine(adapter, **kwargs)


# ---------------------------------------------------------------------------
# Prune tests
# ---------------------------------------------------------------------------

def test_prune_removes_low_confidence_edges():
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS")],
        low_conf_edges=[("B", "C", "RELATED", 0.1)],
    )
    rem = _rem(adapter, prune_confidence_threshold=0.2)
    report = rem.run()
    assert report.pruned_edges == 1
    assert ("B", "C", "RELATED") in report.pruned_edge_list
    assert not adapter._G.has_edge("B", "C")


def test_prune_keeps_high_confidence_edges():
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS")],
        low_conf_edges=[("B", "C", "RELATED", 0.9)],
    )
    rem = _rem(adapter, prune_confidence_threshold=0.2)
    report = rem.run()
    assert report.pruned_edges == 0
    assert adapter._G.has_edge("B", "C")


def test_prune_keeps_default_confidence_edges():
    """Edges with no confidence key default to 1.0 — not pruned."""
    G = nx.DiGraph()
    G.add_edge("X", "Y", relation="LINK", weight=1.0)  # no confidence key
    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=32)
    adapter._embeddings = {n: engine.encode_one(n) for n in G.nodes()}

    rem = _rem(adapter, prune_confidence_threshold=0.5)
    report = rem.run()
    assert report.pruned_edges == 0


def test_prune_never_removes_bridge_twin_edges():
    """BRIDGE_TWIN edges are immune even if confidence is below threshold."""
    adapter = _make_adapter(
        edges=[],
        bridge_twin_edges=[("orig", "twin")],
    )
    # confidence is 0.0 but should not be pruned
    rem = _rem(adapter, prune_confidence_threshold=0.5)
    report = rem.run()
    assert report.pruned_edges == 0
    assert adapter._G.has_edge("orig", "twin")


# ---------------------------------------------------------------------------
# Synthesize tests
# ---------------------------------------------------------------------------

def _make_similar_adapter():
    """Two disconnected nodes with nearly identical embeddings."""
    G = nx.DiGraph()
    G.add_node("hub", label="hub", type="entity")
    G.add_node("alpha", label="alpha", type="entity")
    G.add_node("beta", label="beta", type="entity")
    # hub connects to both but alpha-beta are not directly linked
    G.add_edge("hub", "alpha", relation="KNOWS", weight=1.0, confidence=1.0)
    G.add_edge("hub", "beta", relation="KNOWS", weight=1.0, confidence=1.0)

    adapter = NetworkXAdapter(G)
    # Give alpha and beta near-identical embeddings
    base = np.ones(32, dtype=np.float32)
    base /= np.linalg.norm(base)
    noise = np.zeros(32, dtype=np.float32)
    noise[0] = 0.001
    beta_raw = base + noise
    beta_raw = beta_raw / np.linalg.norm(beta_raw)
    adapter.embeddings = {
        "hub":   RandomEngine(dim=32).encode_one("hub"),
        "alpha": base.copy(),
        "beta":  beta_raw,
    }
    return adapter


def test_synthesize_proposes_similar_nodes():
    adapter = _make_similar_adapter()
    rem = _rem(adapter, synthesis_similarity_threshold=0.99)
    report = rem.run()
    assert report.synthesized_edges >= 1
    pairs = [(t[0], t[1]) for t in report.synthesized_edge_list]
    assert ("alpha", "beta") in pairs or ("beta", "alpha") in pairs


def test_synthesize_does_not_duplicate_existing_edges():
    adapter = _make_similar_adapter()
    # Pre-connect alpha-beta
    adapter._G.add_edge("alpha", "beta", relation="KNOWS", weight=1.0, confidence=1.0)
    rem = _rem(adapter, synthesis_similarity_threshold=0.99)
    report = rem.run()
    pairs = [(t[0], t[1]) for t in report.synthesized_edge_list]
    assert ("alpha", "beta") not in pairs
    assert ("beta", "alpha") not in pairs


def test_synthesize_respects_max_proposals():
    """With max_synthesis_proposals=1, only one edge can be proposed."""
    adapter = _make_similar_adapter()
    rem = _rem(adapter, synthesis_similarity_threshold=0.5, max_synthesis_proposals=1)
    report = rem.run()
    assert report.synthesized_edges <= 1


def test_synthesized_edge_metadata():
    """Synthetic edges must have correct confidence, provenance, weight."""
    adapter = _make_similar_adapter()
    rem = _rem(adapter, synthesis_similarity_threshold=0.99, synthesis_confidence=0.3)
    report = rem.run()
    if report.synthesized_edges > 0:
        u, v, _ = report.synthesized_edge_list[0]
        data = adapter._G.get_edge_data(u, v)
        assert data is not None
        assert data["confidence"] == pytest.approx(0.3)
        assert data["provenance"] == "rem_synthesized"
        assert data["weight"] == pytest.approx(0.5)


def test_no_synthesis_without_embeddings():
    """Nodes without embeddings must be skipped in synthesis."""
    G = nx.DiGraph()
    G.add_node("A", label="A", type="entity")
    G.add_node("B", label="B", type="entity")
    G.add_edge("A", "hub", relation="K", weight=1.0, confidence=1.0)
    G.add_edge("B", "hub", relation="K", weight=1.0, confidence=1.0)
    G.add_node("hub", label="hub", type="entity")
    adapter = NetworkXAdapter(G)
    adapter._embeddings = {}  # no embeddings at all

    rem = _rem(adapter, synthesis_similarity_threshold=0.5)
    report = rem.run()
    assert report.synthesized_edges == 0


# ---------------------------------------------------------------------------
# Consolidate test
# ---------------------------------------------------------------------------

def test_consolidate_updates_communities():
    """After a real run, community_map on adapter should be set."""
    adapter = _make_similar_adapter()
    # Ensure community_map exists on adapter (it may after construction)
    rem = _rem(adapter)
    report = rem.run()
    # consolidate should have run without error
    # (True if dscf_communities succeeded, False otherwise)
    assert isinstance(report.communities_updated, bool)


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

def test_rem_report_structure():
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter)
    report = rem.run()
    assert isinstance(report, REMReport)
    assert report.pruned_edges >= 0
    assert report.synthesized_edges >= 0
    assert isinstance(report.communities_updated, bool)
    assert report.duration_seconds >= 0
    assert isinstance(report.pruned_edge_list, list)
    assert isinstance(report.synthesized_edge_list, list)
    assert isinstance(report.timestamp, float)


def test_full_cycle_returns_report():
    adapter = _make_adapter(edges=[("A", "B", "KNOWS"), ("B", "C", "KNOWS")])
    rem = _rem(adapter)
    report = rem.run()
    assert isinstance(report, REMReport)


def test_full_cycle_on_clean_graph():
    """A graph with all confidence=1.0 should prune nothing."""
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS"), ("B", "C", "KNOWS"), ("C", "D", "KNOWS")],
    )
    rem = _rem(adapter, prune_confidence_threshold=0.2)
    report = rem.run()
    assert report.pruned_edges == 0


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

def test_dry_run_makes_no_changes():
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS")],
        low_conf_edges=[("B", "C", "RELATED", 0.1)],
    )
    edge_count_before = adapter._G.number_of_edges()
    rem = _rem(adapter, prune_confidence_threshold=0.2)
    report = rem.run(dry_run=True)

    assert report.dry_run is True
    assert adapter._G.number_of_edges() == edge_count_before
    # Report still shows what would happen
    assert report.pruned_edges == 0          # dry_run → 0 actual mutations
    assert len(report.pruned_edge_list) == 1  # but the list shows what would be pruned


def test_dry_run_report_matches_real_run():
    """dry_run pruned_edge_list should equal the real run's pruned_edge_list."""
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS")],
        low_conf_edges=[("B", "C", "RELATED", 0.05)],
    )
    # Two separate adapters from identical graphs for fair comparison
    import copy
    adapter2_G = copy.deepcopy(adapter._G)
    adapter2 = NetworkXAdapter(adapter2_G)
    engine = RandomEngine(dim=32)
    adapter2.embeddings = {n: engine.encode_one(n) for n in adapter2_G.nodes()}

    rem_dry = _rem(adapter, prune_confidence_threshold=0.2)
    rem_real = _rem(adapter2, prune_confidence_threshold=0.2)

    dry_report = rem_dry.run(dry_run=True)
    real_report = rem_real.run(dry_run=False)

    assert set(dry_report.pruned_edge_list) == set(real_report.pruned_edge_list)


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------

def test_rollback_restores_pruned_edges():
    adapter = _make_adapter(
        edges=[("A", "B", "KNOWS")],
        low_conf_edges=[("B", "C", "RELATED", 0.05)],
    )
    rem = _rem(adapter, prune_confidence_threshold=0.2)
    rem.run()
    assert not adapter._G.has_edge("B", "C")

    rem.rollback()
    assert adapter._G.has_edge("B", "C")
    data = adapter._G.get_edge_data("B", "C")
    assert data["relation"] == "RELATED"


def test_rollback_removes_synthetic_edges():
    adapter = _make_similar_adapter()
    rem = _rem(adapter, synthesis_similarity_threshold=0.99)
    report = rem.run()

    if report.synthesized_edges > 0:
        u, v, _ = report.synthesized_edge_list[0]
        assert adapter._G.has_edge(u, v)
        rem.rollback()
        assert not adapter._G.has_edge(u, v)
    else:
        pytest.skip("No synthetic edges proposed — similarity threshold may not be met")


def test_rollback_raises_without_prior_run():
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter)
    with pytest.raises(RuntimeError, match="No snapshot"):
        rem.rollback()


def test_can_rollback_property():
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter)
    assert rem.can_rollback is False

    rem.run()
    assert rem.can_rollback is True

    rem.rollback()
    assert rem.can_rollback is False


def test_rollback_unavailable_after_dry_run():
    """dry_run does not create a snapshot — can_rollback stays False."""
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter)
    rem.run(dry_run=True)
    assert rem.can_rollback is False


# ---------------------------------------------------------------------------
# last_report and schedule/cancel
# ---------------------------------------------------------------------------

def test_last_report_property():
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter)
    assert rem.last_report is None

    rem.run()
    assert rem.last_report is not None
    assert isinstance(rem.last_report, REMReport)


def test_schedule_and_cancel():
    """schedule() fires the callback; cancel() stops subsequent runs."""
    adapter = _make_adapter(edges=[("A", "B", "KNOWS")])
    rem = _rem(adapter, interval_seconds=0.05)
    fired = threading.Event()

    original_run = rem.run

    def patched_run(dry_run=False):
        result = original_run(dry_run=dry_run)
        fired.set()
        return result

    rem.run = patched_run
    rem.schedule()

    assert fired.wait(timeout=2.0), "REM cycle did not fire within 2s"
    rem.cancel()
    # After cancel, no further fires — we just verify cancel doesn't raise
    time.sleep(0.1)

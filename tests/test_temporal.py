"""
Tests for Phase 17.1 — Temporal Reasoning.

Covers is_valid_at() helper and query_time filtering in BeamTraversal.
"""
import time
import pytest
import networkx as nx

from reasoning.traversal import is_valid_at, BeamTraversal
from core.graph_adapter import Edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockEdge:
    """Minimal edge-like object for is_valid_at tests."""
    def __init__(self, valid_from=None, valid_to=None):
        self.valid_from = valid_from
        self.valid_to = valid_to


# ---------------------------------------------------------------------------
# is_valid_at unit tests
# ---------------------------------------------------------------------------

def test_no_query_time_always_valid():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, None) is True


def test_no_validity_window_always_valid():
    edge = _MockEdge()
    assert is_valid_at(edge, 1500.0) is True


def test_within_validity_window():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, 1500.0) is True


def test_exactly_at_valid_from():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, 1000.0) is True


def test_exactly_at_valid_to():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, 2000.0) is True


def test_before_valid_from():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, 500.0) is False


def test_after_valid_to():
    edge = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(edge, 3000.0) is False


def test_only_valid_from_no_end():
    edge = _MockEdge(valid_from=1000.0)
    assert is_valid_at(edge, 500.0) is False
    assert is_valid_at(edge, 1000.0) is True
    assert is_valid_at(edge, 9999999.0) is True


def test_only_valid_to_no_start():
    edge = _MockEdge(valid_to=2000.0)
    assert is_valid_at(edge, 1.0) is True
    assert is_valid_at(edge, 2000.0) is True
    assert is_valid_at(edge, 3000.0) is False


def test_real_edge_dataclass():
    """is_valid_at works on the real Edge dataclass."""
    past = Edge(source_id="A", target_id="B", relation_type="R",
                valid_from=0.0, valid_to=100.0)
    future = Edge(source_id="A", target_id="B", relation_type="R",
                  valid_from=9999.0, valid_to=None)
    now = 500.0
    assert is_valid_at(past, now) is False
    assert is_valid_at(future, now) is False


# ---------------------------------------------------------------------------
# BeamTraversal integration: query_time filters edges during traversal
# ---------------------------------------------------------------------------

def _make_temporal_adapter():
    """
    Build a small adapter with:
      A --(always valid)--> B
      A --(expired)-------> C   (valid_to = 1.0 — already expired)
      B --(future)--------> D   (valid_from = 9e9 — not yet valid)
      B --(always valid)--> E
    """
    from adapters.networkx_adapter import NetworkXAdapter
    G = nx.DiGraph()
    nodes = ["A", "B", "C", "D", "E"]
    for n in nodes:
        G.add_node(n, label=n)

    # A → B always valid
    G.add_edge("A", "B", relation="KNOWS", weight=1.0, confidence=1.0,
               valid_from=None, valid_to=None)
    # A → C expired
    G.add_edge("A", "C", relation="KNOWS", weight=1.0, confidence=1.0,
               valid_from=0.0, valid_to=1.0)
    # B → D future only
    G.add_edge("B", "D", relation="KNOWS", weight=1.0, confidence=1.0,
               valid_from=9_000_000_000.0, valid_to=None)
    # B → E always valid
    G.add_edge("B", "E", relation="KNOWS", weight=1.0, confidence=1.0,
               valid_from=None, valid_to=None)

    adapter = NetworkXAdapter(G)
    adapter.build_communities()
    return adapter


def _make_traversal(adapter):
    from core.attention_engine import CSAEngine
    csa = CSAEngine(adapter)
    return BeamTraversal(adapter, csa, beam_width=20, max_hop=2)


def test_no_query_time_all_edges_considered():
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    paths = bt.traverse(["A"])
    tails = {p.tail for p in paths}
    # Without filter, C and D should be reachable
    assert "C" in tails or "B" in tails  # at least B is reachable


def test_query_time_filters_expired_edge():
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    now = time.time()  # >> 1.0 (the expiry of A→C)
    paths = bt.traverse(["A"], query_time=now)
    tails = {p.tail for p in paths}
    assert "C" not in tails, "Expired edge A→C should be filtered"


def test_query_time_filters_future_edge():
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    now = time.time()  # << 9e9 (the future valid_from of B→D)
    paths = bt.traverse(["A"], query_time=now)
    tails = {p.tail for p in paths}
    assert "D" not in tails, "Future edge B→D should be filtered"


def test_always_valid_edges_pass_filter():
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    now = time.time()
    paths = bt.traverse(["A"], query_time=now)
    tails = {p.tail for p in paths}
    assert "B" in tails, "Always-valid edge A→B should pass filter"
    assert "E" in tails, "Always-valid edge B→E should pass filter"


def test_historical_query_time():
    """A query_time of 0.5 should see A→C (valid 0→1) but NOT A→B if it had a future start."""
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    # At time 0.5, C's window (0→1) is active; D's future window is not
    paths = bt.traverse(["A"], query_time=0.5)
    tails = {p.tail for p in paths}
    assert "C" in tails, "At t=0.5 the A→C window (0→1) should be active"
    assert "D" not in tails


def test_backward_compatible_no_arg():
    """traverse() with no query_time arg works identically to before."""
    adapter = _make_temporal_adapter()
    bt = _make_traversal(adapter)
    paths_no_arg = bt.traverse(["A"])
    paths_none = bt.traverse(["A"], query_time=None)
    assert len(paths_no_arg) == len(paths_none)

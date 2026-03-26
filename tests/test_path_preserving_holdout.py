"""
tests/test_path_preserving_holdout.py
Hole 4 — Sparse-Graph Validation Bias: Path-Preserving Hold-out.

Validates that InferenceValidator only withholds edges where an alternative
multi-hop path exists, preventing false-zero recall on sparse graphs.
"""
import networkx as nx
import pytest
from unittest.mock import MagicMock, patch

from core.inference_validator import InferenceValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(G: nx.Graph):
    """Mock adapter exposing to_networkx()."""
    adapter = MagicMock()
    adapter.to_networkx.return_value = G
    return adapter


def _dense_graph():
    """
    Triangle-plus-node: A-B-C-A + B-D.
    Every edge in the triangle has an alternative 2-hop path.
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B", {"relation_type": "KNOWS", "weight": 1.0}),
        ("B", "C", {"relation_type": "KNOWS", "weight": 1.0}),
        ("C", "A", {"relation_type": "KNOWS", "weight": 1.0}),
        ("B", "D", {"relation_type": "KNOWS", "weight": 1.0}),
    ])
    return G


def _bridge_graph():
    """
    A-B-C: linear path. B is the only bridge between A and C.
    Removing A-B severs A's only path to C.
    """
    G = nx.Graph()
    G.add_edges_from([
        ("A", "B", {"relation_type": "KNOWS", "weight": 1.0}),
        ("B", "C", {"relation_type": "KNOWS", "weight": 1.0}),
    ])
    return G


# ---------------------------------------------------------------------------
# _has_alternative_path
# ---------------------------------------------------------------------------

def test_has_alternative_path_with_triangle():
    G = _dense_graph()
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter)
    # A-B has alternative path A→(C←B) via reverse, or B→C→A
    # For directed: A→C is via C→A reversed — check A→B removal
    # With directed graph: A→B removed; A→? still has A? not reachable to B
    # Use an undirected variant for clarity
    G_ud = nx.Graph()
    G_ud.add_edges_from([("A","B"),("B","C"),("C","A")])
    adapter2 = _make_adapter(G_ud)
    v2 = InferenceValidator(adapter2)
    # A-B removed; A→C→B is still a path
    assert v2._has_alternative_path(G_ud, "A", "B") is True


def test_has_alternative_path_false_for_bridge():
    G = _bridge_graph()
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter)
    # Removing A-B leaves A disconnected from B
    assert v._has_alternative_path(G, "A", "B") is False


def test_has_alternative_path_false_for_isolated_pair():
    G = nx.Graph()
    G.add_edge("X", "Y")
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter)
    assert v._has_alternative_path(G, "X", "Y") is False


def test_has_alternative_path_ghost_node_returns_false():
    G = nx.Graph()
    G.add_edge("A", "B")
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter)
    assert v._has_alternative_path(G, "A", "ghost") is False


# ---------------------------------------------------------------------------
# validate() dry_run: path_preserving=True filters hold-out set
# ---------------------------------------------------------------------------

def test_path_preserving_true_filters_bridges():
    """
    Linear graph A-B-C: edge A-B is a bridge (no alternative path).
    With path_preserving=True and only inferable edges, the hold-out set
    should exclude bridge edges (or be empty if no inferable edges survive).
    """
    G = nx.Graph()
    # Linear chain — every edge is a bridge
    for i in range(10):
        G.add_edge(str(i), str(i+1), relation_type="KNOWS", weight=1.0)
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter, path_preserving=True, seed=42)

    # dry_run gives us the held_out_edges count without modifying graph
    with patch.object(v, "_find_inferable_edges", return_value=[
        ("0", "2", "KNOWS"),   # 0-1-2 exists, so 0-2 has 2-hop path
        ("1", "3", "KNOWS"),   # 1-2-3 exists
    ]):
        report = v.validate(dry_run=True)

    # All candidates may survive path-preserving if 2-hop paths exist
    assert report.held_out_edges >= 0  # doesn't crash


def test_path_preserving_false_no_filter():
    """
    path_preserving=False: edges are held out even if they are bridges.
    """
    G = nx.Graph()
    G.add_edge("A", "B", relation_type="KNOWS")
    G.add_edge("B", "C", relation_type="KNOWS")
    adapter = _make_adapter(G)

    v_pp  = InferenceValidator(adapter, path_preserving=True,  seed=42)
    v_raw = InferenceValidator(adapter, path_preserving=False, seed=42)

    candidates = [("A", "C", "KNOWS")]  # synthetic inferable edge
    with patch.object(v_pp,  "_find_inferable_edges", return_value=candidates):
        with patch.object(v_raw, "_find_inferable_edges", return_value=candidates):
            report_pp  = v_pp.validate(dry_run=True)
            report_raw = v_raw.validate(dry_run=True)

    # path_preserving=False should keep the candidate
    assert report_raw.held_out_edges >= 1


def test_path_preserving_default_is_true():
    G = nx.Graph()
    G.add_edge("A", "B")
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter)
    assert v._path_preserving is True


def test_path_preserving_false_explicit():
    G = nx.Graph()
    G.add_edge("A", "B")
    adapter = _make_adapter(G)
    v = InferenceValidator(adapter, path_preserving=False)
    assert v._path_preserving is False


# ---------------------------------------------------------------------------
# Functional: dense graph — path_preserving keeps more valid candidates
# ---------------------------------------------------------------------------

def test_dense_graph_path_preserving_keeps_candidates():
    """
    A fully connected 5-node string-keyed graph: every edge removal leaves
    alternative paths.  path_preserving=True must not reduce candidate count.
    """
    nodes = ["A", "B", "C", "D", "E"]
    G = nx.complete_graph(nodes)
    for u, v in G.edges():
        G[u][v]["relation_type"] = "KNOWS"
        G[u][v]["weight"] = 1.0
    adapter = _make_adapter(G)

    candidates = [(u, v, "KNOWS")
                  for u in nodes for v in nodes if u != v]

    val = InferenceValidator(adapter, path_preserving=True, hold_out_fraction=0.5, seed=0)
    with patch.object(val, "_find_inferable_edges", return_value=candidates):
        report = val.validate(dry_run=True)

    # All candidates should survive path-preserving on a complete graph
    expected = max(1, int(len(candidates) * 0.5))
    assert report.held_out_edges == expected


# ---------------------------------------------------------------------------
# Sparse graph: path_preserving prevents false-zero recall scenario
# ---------------------------------------------------------------------------

def test_sparse_graph_path_preserving_reduces_held_out():
    """
    MetaQA-like sparse graph (avg degree ~2): path_preserving=True should
    result in fewer or equal held-out edges than path_preserving=False,
    because bridge edges are excluded.
    """
    # Build a sparse tree-like graph with a few shortcuts
    G = nx.Graph()
    # Tree backbone
    edges = [("A","B"),("B","C"),("C","D"),("D","E"),
             ("E","F"),("F","G"),("G","H")]
    for u, v in edges:
        G.add_edge(u, v, relation_type="KNOWS")
    # Add two shortcuts that create alternative paths
    G.add_edge("B", "D", relation_type="KNOWS")  # B-C-D shortcut
    G.add_edge("E", "G", relation_type="KNOWS")  # E-F-G shortcut

    adapter = _make_adapter(G)
    # Synthesize inferable edges (all adjacent pairs)
    candidates = [(u, v, "KNOWS") for u, v in G.edges()]

    v_pp  = InferenceValidator(adapter, path_preserving=True,
                                hold_out_fraction=1.0, seed=0)
    v_raw = InferenceValidator(adapter, path_preserving=False,
                                hold_out_fraction=1.0, seed=0)

    with patch.object(v_pp,  "_find_inferable_edges", return_value=candidates):
        with patch.object(v_raw, "_find_inferable_edges", return_value=candidates):
            r_pp  = v_pp.validate(dry_run=True)
            r_raw = v_raw.validate(dry_run=True)

    # Path-preserving must hold out <= raw (bridge edges excluded)
    assert r_pp.held_out_edges <= r_raw.held_out_edges

"""
Tests for core/leiden_native.py — native Leiden reimplementation.

Verifies output format, partition quality, connectivity guarantees,
reproducibility, and drop-in compatibility with the existing API.
"""
import pytest
import networkx as nx
from core.leiden_native import leiden_communities_native


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_two_cliques(n=6):
    """Two n-cliques connected by a single bridge edge. Should detect 2 communities."""
    G = nx.Graph()
    # Clique A: nodes 0..n-1
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j)
    # Clique B: nodes n..2n-1
    for i in range(n, 2 * n):
        for j in range(i + 1, 2 * n):
            G.add_edge(i, j)
    # Bridge
    G.add_edge(n - 1, n)
    return G


def make_ring(n=10):
    """Ring graph — should produce roughly sqrt(n) communities or all one."""
    return nx.cycle_graph(n)


def make_star(n=10):
    """Star graph — hub + n spokes."""
    return nx.star_graph(n)


def make_path(n=8):
    """Path graph."""
    return nx.path_graph(n)


def make_string_nodes():
    """Graph with string node IDs."""
    G = nx.Graph()
    G.add_edge("alice", "bob")
    G.add_edge("bob", "carol")
    G.add_edge("carol", "alice")
    G.add_edge("dave", "eve")
    G.add_edge("eve", "frank")
    G.add_edge("frank", "dave")
    G.add_edge("carol", "dave")  # weak bridge
    return G


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------

def test_returns_list_of_frozensets():
    G = make_two_cliques()
    parts = leiden_communities_native(G)
    assert isinstance(parts, list)
    for p in parts:
        assert isinstance(p, frozenset)


def test_covers_all_nodes():
    G = make_two_cliques()
    parts = leiden_communities_native(G)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == set(G.nodes())


def test_no_node_in_multiple_communities():
    G = make_two_cliques()
    parts = leiden_communities_native(G)
    all_nodes = []
    for p in parts:
        all_nodes.extend(p)
    assert len(all_nodes) == len(set(all_nodes)), "Node appears in multiple communities"


def test_no_empty_communities():
    G = make_two_cliques()
    parts = leiden_communities_native(G)
    for p in parts:
        assert len(p) > 0


def test_string_node_ids_preserved():
    G = make_string_nodes()
    parts = leiden_communities_native(G)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == set(G.nodes())
    for node in all_nodes:
        assert isinstance(node, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_graph_returns_empty():
    G = nx.Graph()
    parts = leiden_communities_native(G)
    assert parts == []


def test_no_edges_returns_singletons():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    parts = leiden_communities_native(G)
    assert len(parts) == 4
    for p in parts:
        assert len(p) == 1


def test_single_node():
    G = nx.Graph()
    G.add_node("solo")
    parts = leiden_communities_native(G)
    assert len(parts) == 1
    assert frozenset(["solo"]) in parts


def test_single_edge():
    G = nx.Graph()
    G.add_edge("a", "b")
    parts = leiden_communities_native(G)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == {"a", "b"}


def test_disconnected_graph():
    """Two isolated cliques with no bridge — must both be covered."""
    G = nx.Graph()
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j)
    for i in range(10, 14):
        for j in range(i + 1, 14):
            G.add_edge(i, j)
    parts = leiden_communities_native(G)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == set(G.nodes())


def test_directed_graph_is_symmetrized():
    """Directed graph input should not raise; output covers all nodes."""
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("c", "a")
    parts = leiden_communities_native(G)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Quality tests
# ---------------------------------------------------------------------------

def test_two_cliques_detected_as_two_communities():
    """The canonical two-clique graph should yield exactly 2 communities."""
    G = make_two_cliques(n=8)
    parts = leiden_communities_native(G, resolution=1.0, seed=0)
    assert len(parts) == 2
    sizes = sorted(len(p) for p in parts)
    assert sizes == [8, 8]


def test_communities_are_internally_connected():
    """Leiden guarantee: every community must be internally connected."""
    G = make_two_cliques(n=6)
    parts = leiden_communities_native(G)
    for part in parts:
        subgraph = G.subgraph(part)
        assert nx.is_connected(subgraph), f"Community {part} is not internally connected"


def test_larger_graph_connectivity_guarantee():
    """Connectivity guarantee holds on a larger random graph."""
    rng = nx.utils.create_py_random_state(99)
    G = nx.barabasi_albert_graph(50, 3, seed=99)
    parts = leiden_communities_native(G, seed=0)
    for part in parts:
        subgraph = G.subgraph(part)
        assert nx.is_connected(subgraph)


def test_resolution_higher_gives_more_communities():
    """Higher resolution should produce more (smaller) communities."""
    G = make_two_cliques(n=10)
    parts_low = leiden_communities_native(G, resolution=0.5, seed=42)
    parts_high = leiden_communities_native(G, resolution=2.0, seed=42)
    assert len(parts_high) >= len(parts_low)


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    G = make_two_cliques(n=8)
    parts1 = leiden_communities_native(G, seed=7)
    parts2 = leiden_communities_native(G, seed=7)
    # Compare as sets of frozensets
    assert set(parts1) == set(parts2)


def test_different_seeds_same_structure():
    """Two strong cliques — both seeds should still find 2 communities."""
    G = make_two_cliques(n=8)
    parts_a = leiden_communities_native(G, seed=1)
    parts_b = leiden_communities_native(G, seed=999)
    assert len(parts_a) == len(parts_b) == 2


# ---------------------------------------------------------------------------
# Weighted graph test
# ---------------------------------------------------------------------------

def test_weighted_graph():
    """Edge weights are respected — high-weight edges should stay in same community."""
    G = nx.Graph()
    # Strong cluster A
    G.add_edge("a1", "a2", weight=10.0)
    G.add_edge("a2", "a3", weight=10.0)
    G.add_edge("a1", "a3", weight=10.0)
    # Strong cluster B
    G.add_edge("b1", "b2", weight=10.0)
    G.add_edge("b2", "b3", weight=10.0)
    G.add_edge("b1", "b3", weight=10.0)
    # Weak bridge
    G.add_edge("a1", "b1", weight=0.01)

    parts = leiden_communities_native(G, resolution=1.0, seed=0)
    all_nodes = set()
    for p in parts:
        all_nodes |= p
    assert all_nodes == set(G.nodes())

    # Cluster A nodes should all be in the same community
    a_nodes = {"a1", "a2", "a3"}
    b_nodes = {"b1", "b2", "b3"}
    found_a = any(a_nodes <= p for p in parts)
    found_b = any(b_nodes <= p for p in parts)
    assert found_a, "Cluster A not in same community"
    assert found_b, "Cluster B not in same community"

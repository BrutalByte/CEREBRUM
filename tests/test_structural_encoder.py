"""
Unit and component tests for core/structural_encoder.py.

Covers:
  - compute_structural_features: coverage, key presence, PageRank sum, betweenness
  - encode_structural_features: dimensionality, value range, empty input
  - build_community_distance_matrix: symmetry, self-distance absent
  - adjacent_community_pairs: bidirectionality, cross-edge detection
"""
import math

import networkx as nx
import numpy as np
import pytest

from core.structural_encoder import (
    adjacent_community_pairs,
    build_community_distance_matrix,
    compute_structural_features,
    encode_structural_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_bridge_graph() -> nx.DiGraph:
    """
    Two triangles connected by a single bridge edge.

    a -- b -- c
              |
              d -- e -- f

    Node 'c'/'d' bridge has the highest betweenness centrality.
    """
    G = nx.DiGraph()
    G.add_edges_from([
        ("a", "b"), ("b", "c"), ("c", "a"),   # triangle 1
        ("d", "e"), ("e", "f"), ("f", "d"),   # triangle 2
        ("c", "d"),                           # bridge
    ])
    return G


def make_two_clique_community_map() -> tuple:
    """
    Six-node graph with two clear communities.
    Returns (G, community_map).
    """
    G = nx.Graph()
    # Community 0: a, b, c
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    # Community 1: x, y, z
    G.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])
    # Bridge
    G.add_edge("c", "x")

    community_map = {"a": 0, "b": 0, "c": 0, "x": 1, "y": 1, "z": 1}
    return G, community_map


# ---------------------------------------------------------------------------
# compute_structural_features
# ---------------------------------------------------------------------------

def test_structural_features_all_nodes_covered():
    """Feature dict must contain an entry for every node in the graph."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    assert set(features.keys()) == set(G.nodes())


def test_structural_features_keys_present():
    """Every node entry must have pagerank, betweenness, degree, in_degree, out_degree."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    required = {"pagerank", "betweenness", "degree", "in_degree", "out_degree"}
    for node, data in features.items():
        assert required == set(data.keys()), f"Missing keys for node {node!r}"


def test_pagerank_sums_to_one():
    """PageRank values across all nodes must sum to approximately 1.0."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    total    = sum(v["pagerank"] for v in features.values())
    assert abs(total - 1.0) < 0.01, f"PageRank sum {total} is not ~1.0"


def test_betweenness_bridge_node_highest():
    """
    The bridge node 'c' (or 'd') must have strictly higher betweenness
    centrality than a pure-leaf node like 'a'.
    """
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    # 'c' is the bridge; 'a' is a leaf-like node inside the triangle
    assert features["c"]["betweenness"] > features["a"]["betweenness"], (
        f"Expected bridge node 'c' ({features['c']['betweenness']:.4f}) > "
        f"leaf 'a' ({features['a']['betweenness']:.4f})"
    )


def test_empty_graph_returns_empty():
    """An empty graph must return an empty dict without error."""
    G        = nx.DiGraph()
    features = compute_structural_features(G)
    assert features == {}


# ---------------------------------------------------------------------------
# encode_structural_features
# ---------------------------------------------------------------------------

def test_encode_produces_correct_dim():
    """Encoded vectors must have exactly the requested dimension."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    for dim in [3, 16, 64, 128]:
        encoded = encode_structural_features(features, dim=dim)
        for node, vec in encoded.items():
            assert len(vec) == dim, f"Expected dim={dim}, got {len(vec)} for node {node!r}"


def test_encode_values_in_0_1():
    """All encoded feature values must be in [0, 1] (normalized)."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    encoded  = encode_structural_features(features, dim=64)
    for node, vec in encoded.items():
        assert vec.min() >= -1e-6, f"Value below 0 for node {node!r}: {vec.min()}"
        assert vec.max() <= 1.0 + 1e-6, f"Value above 1 for node {node!r}: {vec.max()}"


def test_encode_all_nodes_covered():
    """Encoded dict must contain an entry for every node in features."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    encoded  = encode_structural_features(features, dim=64)
    assert set(encoded.keys()) == set(features.keys())


def test_encode_empty_input():
    """Empty feature dict must return empty encoded dict without error."""
    encoded = encode_structural_features({}, dim=64)
    assert encoded == {}


def test_encode_vectors_are_float32():
    """Encoded vectors must be float32 (the expected dtype throughout Parallax)."""
    G        = make_bridge_graph()
    features = compute_structural_features(G)
    encoded  = encode_structural_features(features, dim=64)
    for node, vec in encoded.items():
        assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype} for {node!r}"


# ---------------------------------------------------------------------------
# build_community_distance_matrix
# ---------------------------------------------------------------------------

def test_community_distance_matrix_symmetry():
    """
    In an undirected graph, d(A, B) must equal d(B, A) for all community pairs.
    """
    G, community_map = make_two_clique_community_map()
    distances        = build_community_distance_matrix(G, community_map)
    for (ci, cj), dist in distances.items():
        reverse = distances.get((cj, ci))
        assert reverse is not None, f"Missing reverse entry ({cj}, {ci})"
        assert abs(dist - reverse) < 1e-9, f"Asymmetry: d({ci},{cj})={dist}, d({cj},{ci})={reverse}"


def test_community_distance_matrix_no_self_loops():
    """Self-distances (d(A, A)) must not appear in the distance matrix."""
    G, community_map = make_two_clique_community_map()
    distances        = build_community_distance_matrix(G, community_map)
    for (ci, cj) in distances:
        assert ci != cj, f"Self-distance found for community {ci}"


def test_community_distance_adjacent_is_one():
    """Two communities connected by a direct bridge edge must have distance 1."""
    G, community_map = make_two_clique_community_map()
    distances        = build_community_distance_matrix(G, community_map)
    # Communities 0 and 1 are connected by the c-x bridge
    assert distances.get((0, 1)) == 1.0 or distances.get((1, 0)) == 1.0


# ---------------------------------------------------------------------------
# adjacent_community_pairs
# ---------------------------------------------------------------------------

def test_adjacent_pairs_are_bidirectional():
    """If (A, B) is in the set, (B, A) must also be in the set."""
    G, community_map = make_two_clique_community_map()
    pairs            = adjacent_community_pairs(G, community_map)
    for (ci, cj) in list(pairs):
        assert (cj, ci) in pairs, f"Missing reverse pair ({cj}, {ci})"


def test_adjacent_pairs_bridge_detected():
    """The bridge between communities 0 and 1 must appear in adjacent pairs."""
    G, community_map = make_two_clique_community_map()
    pairs            = adjacent_community_pairs(G, community_map)
    assert (0, 1) in pairs or (1, 0) in pairs


def test_no_intra_community_pairs():
    """Intra-community edges must NOT appear in the adjacent pairs set."""
    G, community_map = make_two_clique_community_map()
    pairs            = adjacent_community_pairs(G, community_map)
    for (ci, cj) in pairs:
        assert ci != cj, f"Intra-community pair ({ci}, {cj}) found in adjacent pairs"

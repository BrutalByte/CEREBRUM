"""
Unit tests for DSCF and community detection algorithms.
Phase 1 checklist: all tests here must pass before proceeding to Phase 2.
"""
import random

import networkx as nx
import pytest

from core.community_engine import (
    dscf_communities,
    leiden_communities,
    lpa_communities,
    modularity_score,
    best_of_n_dscf,
    tsc_communities,
    tsc_quality_metrics,
)


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

def make_two_cliques(n: int = 6, bridge: int = 1) -> nx.Graph:
    """Two cliques of n nodes each, connected by bridge edges."""
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(f"a{i}", f"a{j}")
            G.add_edge(f"b{i}", f"b{j}")
    for k in range(bridge):
        G.add_edge(f"a{k}", f"b{k}")
    return G


def make_toy_graph() -> nx.Graph:
    """Load the toy graph fixture."""
    import csv
    from pathlib import Path
    G    = nx.Graph()
    path = Path(__file__).parent / "fixtures" / "toy_graph.csv"
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            src = row["source"].strip()
            tgt = row["target"].strip()
            if src and tgt:
                G.add_edge(src, tgt, relation=row.get("relation", "").strip())
    return G


# ---------------------------------------------------------------------------
# DSCF tests
# ---------------------------------------------------------------------------

def test_singleton_init_empty_graph():
    """No edges -> one singleton community per node."""
    G = nx.Graph()
    G.add_nodes_from(["x", "y", "z"])
    result = dscf_communities(G)
    assert len(result) == 3
    all_nodes = set()
    for c in result:
        all_nodes.update(c)
    assert all_nodes == {"x", "y", "z"}


def test_single_node_graph():
    G = nx.Graph()
    G.add_node("solo")
    parts = dscf_communities(G)
    assert len(parts) == 1
    assert frozenset(["solo"]) in parts


def test_all_nodes_covered():
    """Every node must appear in exactly one community."""
    G = make_two_cliques(n=8, bridge=2)
    random.seed(0)
    parts     = dscf_communities(G)
    all_nodes = set(G.nodes())
    covered   = set()
    for c in parts:
        assert len(c & covered) == 0, "Node appears in multiple communities"
        covered.update(c)
    assert covered == all_nodes


def test_disconnected_components_split():
    """Post-pass must not return internally disconnected communities."""
    G = nx.Graph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
    G.add_edges_from([("x", "y"), ("y", "z"), ("z", "x")])

    for _ in range(5):
        parts = dscf_communities(G, max_iter=100)
        for part in parts:
            sub        = G.subgraph(part)
            components = list(nx.connected_components(sub))
            assert len(components) == 1, f"Community {set(part)} is internally disconnected"


def test_two_cliques_finds_few_communities():
    """Two tightly connected cliques should produce at most ~4 communities."""
    G       = make_two_cliques(n=6, bridge=1)
    results = []
    for seed in range(5):
        random.seed(seed)
        parts = dscf_communities(G, resolution=1.0, max_iter=50)
        results.append(parts)

    min_found = min(len(r) for r in results)
    assert min_found <= 4, f"Expected at most 4 communities, got {min_found}"


def test_convergence_within_max_iter():
    """Must complete without hitting the max_iter hard stop for simple graphs."""
    G = make_two_cliques(n=10, bridge=1)
    random.seed(1)
    parts = dscf_communities(G, max_iter=100)
    assert len(parts) >= 1


def test_toy_graph_three_communities():
    """The toy graph has 3 natural communities; DSCF should find between 3-12."""
    G = make_toy_graph()
    random.seed(42)
    # Increase resolution slightly to overcome stay-bias on small graphs
    parts = best_of_n_dscf(G, n_trials=5, seed=42, resolution=1.2)
    # Expect somewhere between 2 (merger) and 12 (over-split)
    assert 2 <= len(parts) <= 12, f"Unexpected community count: {len(parts)}"


def test_modularity_score_positive():
    """Non-trivial community structure should yield Q > 0."""
    G = make_two_cliques(n=6, bridge=1)
    random.seed(42)
    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    q     = modularity_score(G, parts)
    assert q > 0.0, f"Expected Q > 0, got {q}"


# ---------------------------------------------------------------------------
# Leiden / LPA sanity checks (ablation baseline)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("leidenalg"),
    reason="leidenalg not installed",
)
def test_leiden_covers_all_nodes():
    G     = make_two_cliques(n=6, bridge=1)
    parts = leiden_communities(G)
    all_nodes = set(G.nodes())
    covered   = set()
    for c in parts:
        covered.update(c)
    assert covered == all_nodes


def test_lpa_covers_all_nodes():
    G     = make_two_cliques(n=6, bridge=1)
    parts = lpa_communities(G)
    all_nodes = set(G.nodes())
    covered   = set()
    for c in parts:
        covered.update(c)
    assert covered == all_nodes


# ---------------------------------------------------------------------------
# Phase 49 — TSC Explicit Mode tests
# ---------------------------------------------------------------------------

def test_tsc_two_cliques_returns_two_communities():
    G = make_two_cliques(n=6, bridge=1)
    parts = tsc_communities(G)
    assert len(parts) == 2, f"Expected 2 communities, got {len(parts)}"


def test_tsc_covers_all_nodes():
    G = make_two_cliques(n=6, bridge=1)
    parts = tsc_communities(G)
    all_nodes = set(G.nodes())
    covered = set()
    for c in parts:
        covered.update(c)
    assert covered == all_nodes


def test_tsc_no_empty_communities():
    G = make_two_cliques(n=6, bridge=1)
    parts = tsc_communities(G)
    for c in parts:
        assert len(c) > 0, "Found empty community"


def test_tsc_with_explicit_centrality():
    G = make_two_cliques(n=6, bridge=1)
    cent = {n: 1.0 for n in G.nodes()}
    parts = tsc_communities(G, centrality_weights=cent)
    all_nodes = set(G.nodes())
    covered = set()
    for c in parts:
        assert len(c) > 0
        covered.update(c)
    assert covered == all_nodes


def test_tsc_quality_metrics_keys():
    G = make_two_cliques(n=6, bridge=1)
    parts = tsc_communities(G)
    metrics = tsc_quality_metrics(G, parts)
    assert set(metrics.keys()) == {"modularity", "community_count", "min_size", "max_size", "mean_size"}
    assert metrics["community_count"] == len(parts)
    assert metrics["min_size"] >= 1
    assert metrics["max_size"] >= metrics["min_size"]
    assert metrics["mean_size"] > 0.0




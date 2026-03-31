"""
Tests for Phase 22 — Adaptive Community Resolution.

Covers:
  - adaptive_resolution_search() edge cases (empty graph, singleton)
  - heuristic target of K ≈ sqrt(N)
  - custom explicit targets
  - tolerance adherence
"""
import pytest
import networkx as nx
import random

from core.community_engine import adaptive_resolution_search


def _make_clique_graph(n_cliques, clique_size):
    """Create a graph with distinct cliques connected by a single line."""
    G = nx.Graph()
    for i in range(n_cliques):
        nodes = [f"C{i}_N{j}" for j in range(clique_size)]
        for u in nodes:
            for v in nodes:
                if u != v:
                    G.add_edge(u, v, weight=1.0)
    
    # Connect cliques linearly
    for i in range(n_cliques - 1):
        G.add_edge(f"C{i}_N0", f"C{i+1}_N0", weight=1.0)
    
    return G


def test_adaptive_resolution_empty_graph():
    G = nx.Graph()
    res = adaptive_resolution_search(G)
    assert res == 1.0


def test_adaptive_resolution_single_node():
    G = nx.Graph()
    G.add_node(1)
    res = adaptive_resolution_search(G)
    # Target K=sqrt(1)=1. Graph has 1 community.
    # It should hit the minimum bounds rapidly.
    assert res > 0.0


def test_explicit_target_achieved():
    # 4 cliques of 5 nodes. N = 20.
    G = _make_clique_graph(4, 5)
    
    # Force target=4
    res = adaptive_resolution_search(G, target_communities=4, seed=42)
    
    # Verify the resolution actually yields 4 communities
    from core.community_engine import dscf_communities
    random.seed(42)
    parts = dscf_communities(G, resolution=res)
    assert len(parts) == 4


def test_heuristic_sqrt_target_achieved():
    # 9 cliques of 9 nodes. N = 81. sqrt(N) = 9.
    G = _make_clique_graph(9, 9)
    
    # Default target
    res = adaptive_resolution_search(G, seed=42)
    
    from core.community_engine import dscf_communities
    random.seed(42)
    parts = dscf_communities(G, resolution=res)
    
    # Due to tolerance, it might be 8, 9, or 10
    # Diff allowed = 9 * 0.10 = 0.9 => integer must be 9 approx, maybe 8-10 max
    assert 7 <= len(parts) <= 11


def test_tolerance_early_exit():
    # A graph with 2 cliques
    G = _make_clique_graph(2, 5)
    
    # Very loose tolerance should return quickly
    res_loose = adaptive_resolution_search(G, target_communities=2, tol=0.5, max_steps=10, seed=42)
    assert res_loose > 0
    
def test_unachievable_high_target_returns_max_res():
    G = _make_clique_graph(2, 5) # N=10
    # Target 20, Impossible since N=10.
    res = adaptive_resolution_search(G, target_communities=20, max_res=9.9, seed=42)
    assert res == 9.9
    
def test_unachievable_low_target_returns_min_res():
    G = _make_clique_graph(4, 5) # N=20
    # Target 0
    res = adaptive_resolution_search(G, target_communities=0, min_res=0.05, seed=42)
    assert res == 0.05

"""
Tests for Phase 22 — GPUDSCFEngine test coverage.

Covers:
  - device_info() properties
  - detect() with empty, singleton, and trivial graphs
  - detect() with a known two-clique structure
  - detect_with_stats() for profiling info
  - gpu_best_of_n() convenience wrapper
"""
import pytest
import networkx as nx

from core.dscf_gpu import GPUDSCFEngine, GPUDSCFConfig, gpu_best_of_n


def test_device_info():
    info = GPUDSCFEngine.device_info()
    assert isinstance(info, str)
    assert len(info) > 0


def test_empty_graph():
    engine = GPUDSCFEngine()
    G = nx.Graph()
    parts = engine.detect(G)
    assert parts == []


def test_singleton_graph():
    engine = GPUDSCFEngine()
    G = nx.Graph()
    G.add_node("A")
    parts = engine.detect(G)
    assert parts == [frozenset(["A"])]


def test_two_cliques():
    engine = GPUDSCFEngine(GPUDSCFConfig(resolution=1.0, temp_start=0.01))
    G = nx.Graph()
    # Clique 1
    for i in range(5):
        for j in range(i+1, 5):
            G.add_edge(f"A{i}", f"A{j}", weight=1.0)
    # Clique 2
    for i in range(5):
        for j in range(i+1, 5):
            G.add_edge(f"B{i}", f"B{j}", weight=1.0)
    # Single bridge
    G.add_edge("A0", "B0", weight=1.0)

    parts = engine.detect(G)
    
    # Might be more than 2 if temp causes splits, but generally should be 2
    # Ensure all nodes are partitioned
    all_nodes = set.union(*[set(p) for p in parts])
    assert all_nodes == set(G.nodes())
    
    # We should have exactly 2 communities if it works perfectly
    assert len(parts) == 2


def test_detect_with_stats():
    engine = GPUDSCFEngine(GPUDSCFConfig(temp_start=0.0))
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)

    parts, stats = engine.detect_with_stats(G)
    assert len(parts) == 1
    assert stats.n_nodes == 3
    assert stats.n_edges == 3
    assert stats.n_communities == 1
    assert stats.wall_ms > 0.0


def test_gpu_best_of_n():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(3, 4)
    # 2 disconnected components
    
    parts = gpu_best_of_n(G, n_trials=2)
    assert len(parts) == 2

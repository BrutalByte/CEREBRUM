"""
Unit tests for GraphBridgeEngine:
  - Proactive cross-component bridge synthesis.
"""
import networkx as nx
import pytest
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.graph_bridge import GraphBridgeEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adapter(triples, directed=True):
    """Build a NetworkXAdapter from a list of (src, rel, tgt) tuples."""
    G = nx.DiGraph() if directed else nx.Graph()
    for s, r, o in triples:
        G.add_edge(s, o, relation=r, confidence=1.0, weight=1.0)
    return NetworkXAdapter(G)


# ---------------------------------------------------------------------------
# GraphBridgeEngine tests
# ---------------------------------------------------------------------------

class TestGraphBridgeEngine:

    def test_apply_bridges_disconnected_components(self):
        """
        Bridges two disconnected components if they have similar labels.
        Component 1: (A -r-> B)
        Component 2: (C -r-> D)
        If A and C have the same label, they should be bridged.
        """
        # Create two disconnected components
        G = nx.DiGraph()
        G.add_node("node1", label="apple")
        G.add_node("node2", label="banana")
        G.add_edge("node1", "node2", relation="is_a", confidence=1.0, weight=1.0)

        G.add_node("node3", label="apple")  # Same label as node1
        G.add_node("node4", label="cherry")
        G.add_edge("node3", "node4", relation="is_a", confidence=1.0, weight=1.0)

        adapter = NetworkXAdapter(G)
        # RandomEngine uses label hash for seed; identical labels = identical vectors
        engine = RandomEngine(dim=64)
        
        bridge_engine = GraphBridgeEngine(min_similarity=0.9, top_k=1)
        n = bridge_engine.apply(adapter, engine)

        assert n >= 1
        # Check if bridge edge exists between node1 and node3
        assert G.has_edge("node1", "node3") or G.has_edge("node3", "node1")
        
        edge_data = G.get_edge_data("node1", "node3")
        assert edge_data["relation"] == "bridge:similar"
        assert edge_data["synthetic"] is True
        assert "rule:bridge_embed" in edge_data["provenance"]
        assert "sim:1.0000" in edge_data["provenance"]

    def test_apply_skips_same_component(self):
        """Should not add bridge edges between nodes already in the same component."""
        G = nx.DiGraph()
        G.add_node("n1", label="apple")
        G.add_node("n2", label="apple")
        G.add_edge("n1", "n2", relation="related", confidence=1.0, weight=1.0)
        
        adapter = NetworkXAdapter(G)
        engine = RandomEngine(dim=64)
        bridge_engine = GraphBridgeEngine(min_similarity=0.9)
        
        n = bridge_engine.apply(adapter, engine)
        assert n == 0, "Should not bridge nodes in the same component"

    def test_max_degree_respected(self):
        """Should only bridge nodes with degree <= max_degree."""
        G = nx.DiGraph()
        # Component 1: n1 has degree 3 (out: n2, n3, n4)
        G.add_node("n1", label="apple")
        for i in [2, 3, 4]:
            G.add_edge("n1", f"n{i}", relation="r")
            
        # Component 2: n5 has degree 1
        G.add_node("n5", label="apple")
        G.add_edge("n5", "n6", relation="r")
        
        adapter = NetworkXAdapter(G)
        engine = RandomEngine(dim=64)
        
        # With max_degree=1, n1 (deg 3) should be ignored as a frontier node
        bridge_engine = GraphBridgeEngine(max_degree=1, min_similarity=0.9)
        n = bridge_engine.apply(adapter, engine)
        
        assert n == 0
        assert not G.has_edge("n1", "n5")

    def test_max_bridges_cap(self):
        """Should respect the max_bridges limit."""
        G = nx.DiGraph()
        # Create many small components
        for i in range(10):
            G.add_node(f"src_{i}", label="target")
            G.add_node(f"tgt_{i}", label="other")
            G.add_edge(f"src_{i}", f"tgt_{i}", relation="r")
            
        adapter = NetworkXAdapter(G)
        engine = RandomEngine(dim=64)
        
        # Each src_i has label "target", so they are all similar.
        # But max_bridges=3 should limit the number of added edges.
        bridge_engine = GraphBridgeEngine(max_bridges=3, min_similarity=0.9)
        n = bridge_engine.apply(adapter, engine)
        
        # Each bridge adds 2 edges if directed (forward and backward)
        # So max_bridges=3 means at most 3 edges total.
        assert n <= 3

    def test_no_bridging_if_already_connected(self):
        """If graph is a single component, no bridging should occur."""
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="r")
        G.add_node("A", label="x")
        G.add_node("B", label="x")
        
        adapter = NetworkXAdapter(G)
        engine = RandomEngine(dim=64)
        bridge_engine = GraphBridgeEngine()
        
        n = bridge_engine.apply(adapter, engine)
        assert n == 0

    def test_describe(self):
        engine = GraphBridgeEngine(min_similarity=0.85, top_k=3)
        desc = engine.describe()
        assert "min_sim=0.85" in desc
        assert "top_k=3" in desc

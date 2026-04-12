import numpy as np
import networkx as nx
import pytest
from core.rem_engine import REMEngine
from core.graph_adapter import GraphAdapter

class MockAdapter(GraphAdapter):
    def __init__(self, G, embeddings):
        self._G = G
        self.embeddings = embeddings
    def get_community(self, node_id): return 0
    def get_embedding(self, node_id): return self.embeddings.get(node_id)
    def find_entities(self, query, top_k=5): return []
    def find_similar(self, node_id, top_k=5): return []
    def get_entity(self, node_id): return None
    def get_neighbors(self, node_id): return []
    def to_networkx(self): return self._G
    def add_edge(self, u, v, relation, confidence=1.0, provenance="", synthetic=False): pass

def test_wormhole_synthesis():
    # Create two disconnected components
    # Comp 1: Apple -- Fruit
    # Comp 2: Orange
    G = nx.Graph()
    G.add_edge("apple", "fruit", relation="related_to", confidence=0.9)
    G.add_node("orange")
    
    # Embeddings: Apple and Orange are very similar
    embeddings = {
        "apple":  np.array([1.0, 0.1, 0.0]),
        "fruit":  np.array([0.9, 0.2, 0.0]),
        "orange": np.array([0.98, 0.15, 0.0]),
    }
    
    adapter = MockAdapter(G, embeddings)
    # Threshold 0.95 for cross-component
    rem = REMEngine(adapter, cross_component_similarity_threshold=0.95)
    
    # Run synthesis
    report = rem.run(dry_run=True)
    
    # Check if apple <-> orange wormhole is proposed
    # (Similarity is ~0.99)
    wormholes = [p for p in report.synthesized_edge_list if p[2] == "rem_synthesized_wormhole"]
    assert len(wormholes) > 0
    
    # Ensure apple and orange were chosen
    found = False
    for u, v, rel in wormholes:
        if (u == "apple" and v == "orange") or (u == "orange" and v == "apple"):
            found = True
            break
    assert found

def test_no_wormhole_if_connected():
    # If they are already connected (even long path), regular synthesis might catch them
    # but wormhole phase skips same-component.
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("c", "d")
    G.add_edge("d", "e") # a and e are 4 hops apart
    
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([1.0, 0.0]),
        "c": np.array([1.0, 0.0]),
        "d": np.array([1.0, 0.0]),
        "e": np.array([1.0, 0.0]),
    }
    
    adapter = MockAdapter(G, embeddings)
    rem = REMEngine(adapter, synthesis_similarity_threshold=0.9, cross_component_similarity_threshold=0.99)
    
    report = rem.run(dry_run=True)
    
    # Should NOT have wormholes because it's a single component
    wormholes = [p for p in report.synthesized_edge_list if p[2] == "rem_synthesized_wormhole"]
    assert len(wormholes) == 0

if __name__ == "__main__":
    pytest.main([__file__])

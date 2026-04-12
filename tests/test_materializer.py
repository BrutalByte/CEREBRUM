import pytest
from core.graph_adapter import GraphAdapter
from core.hypothesis_engine import HypothesisEngine
from core.hypothesis_materializer import HypothesisMaterializer
from adapters.networkx_adapter import NetworkXAdapter
import networkx as nx

def test_materializer():
    # Setup graph
    g = nx.DiGraph()
    g.add_node("A", label="Node A", type="entity")
    g.add_node("B", label="Node B", type="entity")
    adapter = NetworkXAdapter(g)
    
    # Materializer mocks engine
    engine = HypothesisEngine(adapter)
    materializer = HypothesisMaterializer(adapter, engine)
    
    candidate = {"source": "A", "target": "B", "relation": "LIKES"}
    
    success = materializer.materialize(candidate, confidence=0.9)
    assert success is True
    
    # Check if edge exists in underlying graph
    assert g.has_edge("A", "B")
    assert g.edges["A", "B"]["relation"] == "LIKES"

def test_materializer_invalid_input():
    g = nx.DiGraph()
    adapter = NetworkXAdapter(g)
    engine = HypothesisEngine(adapter)
    materializer = HypothesisMaterializer(adapter, engine)
    
    success = materializer.materialize({"source": "A"}, confidence=0.5)
    assert success is False

import pytest
import networkx as nx
import numpy as np
from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from reasoning.traversal import BeamTraversal, QUANT_SCALE

@pytest.fixture
def quant_setup():
    G = nx.Graph()
    G.add_node("A", label="Seed", type="entity")
    G.add_node("B", label="Target", type="entity")
    G.add_edge("A", "B", relation="related_to", weight=1.0)
    
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"A": 0, "B": 0}
    adapter.embeddings = {"A": np.random.rand(16), "B": np.random.rand(16)}
    adapter._csa_metadata = {"distances": {0: {0: 0.0}}, "adjacent_pairs": set()}
    
    csa = CSAEngine(adapter)
    return adapter, csa

def test_quantized_traversal_logic(quant_setup):
    adapter, csa = quant_setup
    
    # 1. Standard Traversal
    bt_std = BeamTraversal(adapter, csa, quantized=False, max_hop=1)
    paths_std = bt_std.traverse(["A"])
    
    # 2. Quantized Traversal
    bt_quant = BeamTraversal(adapter, csa, quantized=True, max_hop=1)
    paths_quant = bt_quant.traverse(["A"])
    
    assert len(paths_std) == len(paths_quant)
    
    # Check quantized path
    p_q = [p for p in paths_quant if len(p.nodes) > 1][0]
    assert p_q.quantized is True
    assert p_q.q_score <= QUANT_SCALE
    # For a weight of ~1.0, q_score should be near 255
    assert p_q.q_score > 200 

def test_quantized_multi_hop(quant_setup):
    adapter, csa = quant_setup
    # Add one more hop
    adapter._G.add_node("C", label="End", type="entity")
    adapter._G.add_edge("B", "C", relation="leads_to", weight=1.0)
    adapter.community_map["C"] = 0
    adapter.embeddings["C"] = np.random.rand(16)
    
    bt_quant = BeamTraversal(adapter, csa, quantized=True, max_hop=2)
    paths = bt_quant.traverse(["A"])
    
    # Find 2-hop path
    p2 = [p for p in paths if len(p.nodes) == 5][0] # A -> R -> B -> R -> C
    assert p2.q_score <= QUANT_SCALE
    assert p2.quantized is True

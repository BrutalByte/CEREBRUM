import pytest
import numpy as np
import networkx as nx
from core.attention_engine import CSAEngine
from adapters.networkx_adapter import NetworkXAdapter

def test_temporal_sliding_window():
    G = nx.Graph()
    G.add_edge("A", "B", relation="works_at", valid_to=100.0)
    G.add_edge("A", "C", relation="works_at", valid_to=190.0)
    
    adapter = NetworkXAdapter(G)
    # 384-dim random embeddings
    for node in G.nodes():
        adapter.embeddings[node] = np.random.rand(384)
    
    # query_time = 200
    # "A" -> "C" (valid_to=190, elapsed=10) -> higher td
    # "A" -> "B" (valid_to=100, elapsed=100) -> lower td
    engine = CSAEngine(adapter, use_temporal_decay=True)
    engine.set_query_time(200.0)
    
    w_ab = engine.compute_weight("A", "B", hop=1, edge_type="works_at", valid_to=100.0)
    w_ac = engine.compute_weight("A", "C", hop=1, edge_type="works_at", valid_to=190.0)
    
    assert w_ac > w_ab, f"Newer edge {w_ac} should be stronger than older edge {w_ab}"

def test_temporal_window_penalty():
    G = nx.Graph()
    G.add_edge("A", "B", relation="works_at", valid_to=100.0)
    G.add_edge("A", "C", relation="works_at", valid_to=190.0)
    
    adapter = NetworkXAdapter(G)
    for node in G.nodes():
        adapter.embeddings[node] = np.zeros(384)
    
    # window = 50. 
    # "A" -> "C" (elapsed=10) is inside window.
    # "A" -> "B" (elapsed=100) is outside window (penalty 0.1x).
    engine = CSAEngine(adapter, use_temporal_decay=True, temporal_window_size=50.0)
    engine.set_query_time(200.0)
    
    logit_ab = engine.compute_weight_with_features("A", "B", hop=1, edge_type="works_at", valid_to=100.0)
    logit_ac = engine.compute_weight_with_features("A", "C", hop=1, edge_type="works_at", valid_to=190.0)
    
    # td_ab = exp(-0.5 * 100) * 0.1
    # td_ac = exp(-0.5 * 10)
    assert logit_ac.td > logit_ab.td
    assert logit_ab.td < 0.001 # Should be very small due to penalty + decay

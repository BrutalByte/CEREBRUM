import math
import numpy as np
import pytest
from core.attention_engine import CSAEngine
from core.graph_adapter import GraphAdapter
from core.reasoning_logit import ReasoningLogit

class MockAdapter(GraphAdapter):
    def get_community(self, node_id: str) -> int: return 0
    def get_embedding(self, node_id: str) -> np.ndarray: return np.zeros(10)
    def find_entities(self, query, top_k=5): return []
    def find_similar(self, node_id, top_k=5): return []
    def get_entity(self, node_id): return None
    def get_neighbors(self, node_id): return []
    def to_networkx(self): return None
    def add_edge(self, u, v, relation, confidence=1.0, provenance="", synthetic=False): pass

def test_temporal_scoring_recency_bias():
    adapter = MockAdapter()
    csa = CSAEngine(adapter=adapter, use_temporal_decay=True, eta=1.0)
    csa.set_query_time(1000.0)
    
    # Recent edge: valid_to = 999.0 (elapsed = 1.0)
    # Old edge: valid_to = 500.0 (elapsed = 500.0)
    
    # We expect newer edges to have HIGHER scores (raw logit)
    # Score = ... + eta * exp(-lambda * elapsed)
    
    # compute_weight
    w_recent = csa.compute_weight("u", "v", hop=1, valid_to=999.0)
    w_old = csa.compute_weight("u", "v", hop=1, valid_to=500.0)
    
    assert w_recent > w_old
    print(f"Weights: Recent={w_recent:.4f}, Old={w_old:.4f}")

def test_temporal_decay_feature_in_logit():
    adapter = MockAdapter()
    # Use high eta in CSA to make it obvious, but ReasoningLogit should store RAW feature
    csa = CSAEngine(adapter=adapter, use_temporal_decay=True, eta=5.0)
    csa.set_query_time(100.0)
    
    # valid_to = 90.0, elapsed = 10.0
    # λ default for "" is 0.5 (self.lambda_decay)
    # expected_td = exp(-0.5 * 10.0) = exp(-5) approx 0.0067
    
    logit = csa.compute_weight_with_features("u", "v", hop=1, valid_to=90.0)
    
    expected_td = math.exp(-0.5 * 10.0)
    assert pytest.approx(logit.td, rel=1e-5) == expected_td
    
    # ReasoningLogit.score should apply the eta from params
    # params: (alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)
    params = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0) # Only eta=2.0, theta=0.0
    # raw_score = 2.0 * logit.td
    expected_raw = 2.0 * expected_td
    expected_sigmoid = 1.0 / (1.0 + math.exp(-expected_raw))
    
    assert pytest.approx(logit.score(params), rel=1e-5) == expected_sigmoid

def test_node_recency_scoring():
    adapter = MockAdapter()
    node_recency = {"v": 0.8}
    csa = CSAEngine(adapter=adapter, node_recency=node_recency, iota=2.0)
    
    # compute_weight
    # Score = ... + iota * nr_v
    # Default alpha=0.4, beta=0.4, gamma=0.1, delta=0.05, epsilon=0.05, zeta=0.1, eta=0.1, iota=2.0, mu=0.1, theta=1.0
    # pr_v = 0, td = 0, sim = 0, cs = 1.0 (same comm), etw = 0, nd = 0, hd = 0.5 (hop 1)
    # raw = beta*cs + epsilon*hd + iota*nr_v + theta*grounding
    #     = 0.4*1.0 + 0.05*0.5 + 2.0*0.8 + 1.0*1.0 = 0.4 + 0.025 + 1.6 + 1.0 = 3.025
    
    w = csa.compute_weight("u", "v", hop=1)
    expected_raw = 3.025
    expected_w = 1.0 / (1.0 + math.exp(-expected_raw))
    
    assert pytest.approx(w, rel=1e-5) == expected_w

def test_temporal_decay_disabled():
    adapter = MockAdapter()
    csa = CSAEngine(adapter=adapter, use_temporal_decay=False)
    csa.set_query_time(100.0)
    
    logit = csa.compute_weight_with_features("u", "v", hop=1, valid_to=90.0)
    assert logit.td == 0.0

if __name__ == "__main__":
    pytest.main([__file__])

"""
Unit tests for Community-Structured Attention (CSA).
"""
import math

import numpy as np

from core.attention_engine import CSAEngine, _cosine_sim, _sigmoid
from core.graph_adapter import GraphAdapter


# ---------------------------------------------------------------------------
# Mock Adapter for Testing
# ---------------------------------------------------------------------------

class MockAdapter(GraphAdapter):
    def __init__(self, communities, embeddings):
        self.communities = communities
        self.embeddings = embeddings

    def get_entity(self, entity_id): return None
    def get_neighbors(self, entity_id, edge_types=None, max_neighbors=50, context_embedding=None): return []
    def find_entities(self, query, top_k=10): return []
    def to_networkx(self): return None
    def get_community(self, entity_id): return self.communities.get(entity_id, -1)
    def get_embedding(self, entity_id): return self.embeddings.get(entity_id)
    def find_similar(self, embedding, top_k=10): return []
    def add_edge(self, u, v, relation, confidence=1.0, provenance="", synthetic=False): pass
    def get_degree(self, entity_id: str) -> int: return 0

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_engine() -> CSAEngine:
    """
    Three communities:
      0 : a, b, c  (science cluster — embeddings point along x-axis)
      1 : x, y, z  (history cluster — embeddings point along y-axis)
      2 : m         (bridge node   — diagonal embedding)

    Community graph: 0 adjacent to 1, 1 adjacent to 2, 0 and 2 NOT adjacent.
    """
    communities = {
        "a": 0, "b": 0, "c": 0,
        "x": 1, "y": 1, "z": 1,
        "m": 2,
    }
    embeddings = {
        "a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "c": np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32),
        "x": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.8, 0.2, 0.0], dtype=np.float32),
        "m": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
    }
    adapter = MockAdapter(communities, embeddings)
    engine = CSAEngine(adapter=adapter)
    engine.set_community_graph(
        community_distances={(0, 2): 2.0, (2, 0): 2.0},
        adjacent_pairs={(0, 1), (1, 0), (1, 2), (2, 1)},
    )
    return engine


# ---------------------------------------------------------------------------
# community_score tests
# ---------------------------------------------------------------------------

def test_same_community_score_is_one():
    engine = make_engine()
    assert engine.community_score("a", "b") == 1.0
    assert engine.community_score("x", "y") == 1.0
    assert engine.community_score("b", "c") == 1.0


def test_adjacent_community_score_is_half():
    engine = make_engine()
    # 0 and 1 are adjacent
    assert engine.community_score("a", "x") == 0.5
    assert engine.community_score("y", "m") == 0.5


def test_distant_community_score_uses_exp_decay():
    engine = make_engine()
    # Communities 0 and 2 are distance 2 apart
    score = engine.community_score("a", "m")
    expected = math.exp(-0.5 * 2.0)   # lambda=0.5, d=2
    assert abs(score - expected) < 0.01


def test_unknown_community_returns_neutral():
    engine = make_engine()
    # "ghost" is not in communities dict
    score = engine.community_score("a", "ghost")
    assert score == 0.5


# ---------------------------------------------------------------------------
# compute_weight tests
# ---------------------------------------------------------------------------

def test_weight_is_in_0_1():
    engine = make_engine()
    for u in ["a", "x", "m"]:
        for v in ["b", "y", "m"]:
            if u != v:
                w = engine.compute_weight(u, v, hop=1)
                assert 0.0 < w < 1.0, f"Weight out of (0,1) for {u}->{v}: {w}"


def test_same_community_weight_higher_than_cross():
    engine = make_engine()
    w_same  = engine.compute_weight("a", "b", hop=1)   # both in community 0
    w_cross = engine.compute_weight("a", "x", hop=1)   # communities 0 and 1
    assert w_same > w_cross


def test_hop_decay_decreases_weight():
    engine = make_engine()
    w1 = engine.compute_weight("a", "b", hop=1)
    w5 = engine.compute_weight("a", "b", hop=5)
    assert w1 > w5, "Weight should decrease with hop depth"


def test_missing_embeddings_use_zero_sim():
    """Nodes without embeddings should still produce a valid weight."""
    communities = {"u": 0, "v": 0}
    embeddings  = {}   # empty — no vectors
    adapter = MockAdapter(communities, embeddings)
    engine = CSAEngine(adapter=adapter)
    w = engine.compute_weight("u", "v", hop=1)
    assert 0.0 < w < 1.0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_cosine_sim_parallel():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b)) < 1e-6


def test_cosine_sim_zero_vector():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert _cosine_sim(a, b) == 0.0


def test_sigmoid_midpoint():
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_monotone():
    vals = [_sigmoid(x) for x in [-5, -1, 0, 1, 5]]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# Phase 134: batch scoring parity
# ---------------------------------------------------------------------------

def test_batch_csa_parity():
    """Batch scores must match per-edge scores within float32 precision."""
    engine = make_engine()
    u = "a"
    v_list = ["b", "c", "m", "x"]
    hop = 1
    edge_types = ["KNOWS", "KNOWS", "BRIDGE", "RELATED"]
    valid_tos = [None, None, None, None]
    eu = engine.adapter.get_embedding(u)
    ev_list = [engine.adapter.get_embedding(v) for v in v_list]

    batch_logits = engine.compute_weights_batch(
        u=u, v_list=v_list, hop=hop,
        edge_types=edge_types, valid_tos=valid_tos,
        eu=eu, ev_list=ev_list,
    )
    params = engine.get_current_params(u)
    batch_scores = [l.score(params) for l in batch_logits]

    for i, v in enumerate(v_list):
        per_logit = engine.compute_weight_with_features(
            u, v, hop, edge_type=edge_types[i], eu=eu, ev=ev_list[i]
        )
        per_score = per_logit.score(params)
        assert abs(batch_scores[i] - per_score) < 1e-5, (
            f"Mismatch for ({u},{v}): batch={batch_scores[i]:.6f} per={per_score:.6f}"
        )

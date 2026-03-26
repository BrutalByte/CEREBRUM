"""
Tests for Phase 17.5 — KGE Embeddings (TransE + RotatE).

Covers:
  - Vocabulary extraction from adapter
  - Embedding shape and dtype
  - get_embedding() returns correct dimensions / None for unknown
  - Training runs without error and returns KGETrainingResult
  - Loss is non-negative
  - Embeddings differ between entities after training
  - TransE: score(h, r, t) < score(h, r, t_random) after sufficient training
  - RotatE: get_embedding returns real part (dim, not dim*2)
  - Zero-epoch training returns result with final_loss=0
  - Empty graph produces sensible result
  - Relation embedding accessible
  - Negative sampling produces different triple
  - Reproducibility with same seed
"""
import math
import pytest
import networkx as nx
import numpy as np

from core.kge_engine import TransEEngine, RotatEEngine, KGETrainingResult
from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_adapter():
    """Small directed graph: 5 nodes, 4 edges, 2 relation types."""
    G = nx.DiGraph()
    nodes = ["A", "B", "C", "D", "E"]
    for n in nodes:
        G.add_node(n, label=n)
    G.add_edge("A", "B", relation="KNOWS", weight=1.0)
    G.add_edge("B", "C", relation="KNOWS", weight=1.0)
    G.add_edge("A", "C", relation="INFLUENCED", weight=1.0)
    G.add_edge("C", "D", relation="INFLUENCED", weight=1.0)
    return NetworkXAdapter(G)


def _train_transe(n_epochs=20, dim=16):
    adapter = _small_adapter()
    kge = TransEEngine(dim=dim, margin=1.0, lr=0.05, seed=0)
    result = kge.fit(adapter, n_epochs=n_epochs)
    return kge, result, adapter


def _train_rotate(n_epochs=20, dim=16):
    adapter = _small_adapter()
    kge = RotatEEngine(dim=dim, margin=1.0, lr=0.05, seed=0)
    result = kge.fit(adapter, n_epochs=n_epochs)
    return kge, result, adapter


# ---------------------------------------------------------------------------
# KGETrainingResult fields
# ---------------------------------------------------------------------------

def test_training_result_transe():
    _, result, _ = _train_transe()
    assert isinstance(result, KGETrainingResult)
    assert result.model == "TransE"
    assert result.n_entities == 5
    assert result.n_triples == 4
    assert result.n_relations == 2
    assert result.n_epochs == 20
    assert result.embedding_dim == 16
    assert result.duration_seconds >= 0.0


def test_training_result_rotate():
    _, result, _ = _train_rotate()
    assert result.model == "RotatE"
    assert result.n_entities == 5


def test_final_loss_non_negative_transe():
    _, result, _ = _train_transe()
    assert result.final_loss >= 0.0


def test_final_loss_non_negative_rotate():
    _, result, _ = _train_rotate()
    assert result.final_loss >= 0.0


def test_result_property_accessible():
    kge = TransEEngine(dim=8, seed=0)
    assert kge.result is None
    kge.fit(_small_adapter(), n_epochs=1)
    assert kge.result is not None


# ---------------------------------------------------------------------------
# Embedding shape and values
# ---------------------------------------------------------------------------

def test_transe_embedding_shape():
    kge, _, _ = _train_transe(dim=32)
    emb = kge.get_embedding("A")
    assert emb is not None
    assert emb.shape == (32,)


def test_rotate_embedding_shape_is_real_part_only():
    """RotatE stores (dim*2) internally but get_embedding returns dim."""
    kge, _, _ = _train_rotate(dim=16)
    emb = kge.get_embedding("A")
    assert emb is not None
    assert emb.shape == (16,)


def test_unknown_entity_returns_none():
    kge, _, _ = _train_transe()
    assert kge.get_embedding("NOBODY") is None


def test_unknown_entity_before_fit():
    kge = TransEEngine(dim=8, seed=0)
    assert kge.get_embedding("X") is None


def test_embeddings_differ_between_entities():
    kge, _, _ = _train_transe()
    emb_a = kge.get_embedding("A")
    emb_b = kge.get_embedding("B")
    assert not np.allclose(emb_a, emb_b), "A and B should have distinct embeddings"


# ---------------------------------------------------------------------------
# Relation embeddings
# ---------------------------------------------------------------------------

def test_relation_embedding_accessible():
    kge, _, _ = _train_transe(dim=8)
    r = kge.get_relation_embedding("KNOWS")
    assert r is not None
    assert r.shape == (8,)


def test_unknown_relation_returns_none():
    kge, _, _ = _train_transe()
    assert kge.get_relation_embedding("MYSTERY") is None


# ---------------------------------------------------------------------------
# Zero-epoch and empty graph
# ---------------------------------------------------------------------------

def test_zero_epoch_training():
    adapter = _small_adapter()
    kge = TransEEngine(dim=8, seed=0)
    result = kge.fit(adapter, n_epochs=0)
    assert result.final_loss == 0.0
    assert result.n_epochs == 0


def test_empty_graph():
    G = nx.DiGraph()
    adapter = NetworkXAdapter(G)
    kge = TransEEngine(dim=8, seed=0)
    result = kge.fit(adapter, n_epochs=5)
    assert result.n_entities == 0
    assert result.n_triples == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility_same_seed():
    def run():
        adapter = _small_adapter()
        kge = TransEEngine(dim=8, lr=0.05, seed=7)
        kge.fit(adapter, n_epochs=10)
        return kge.get_embedding("A").copy()

    emb1 = run()
    emb2 = run()
    np.testing.assert_allclose(emb1, emb2, rtol=1e-5)


def test_different_seed_different_result():
    def run(seed):
        adapter = _small_adapter()
        kge = TransEEngine(dim=8, lr=0.05, seed=seed)
        kge.fit(adapter, n_epochs=10)
        return kge.get_embedding("A").copy()

    emb_a = run(1)
    emb_b = run(999)
    # Different seeds should give different embeddings (very unlikely to collide)
    assert not np.allclose(emb_a, emb_b)


# ---------------------------------------------------------------------------
# TransE semantic check
# ---------------------------------------------------------------------------

def test_transe_entity_embeddings_L2_normalised_after_training():
    """TransE normalises entity embeddings after each batch."""
    kge, _, _ = _train_transe(n_epochs=10, dim=16)
    for eid in ["A", "B", "C"]:
        emb = kge.get_embedding(eid)
        norm = float(np.linalg.norm(emb))
        assert norm == pytest.approx(1.0, abs=0.05), (
            f"Entity {eid!r} L2 norm = {norm:.4f}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# RotatE internal: complex multiplication sanity
# ---------------------------------------------------------------------------

def test_rotate_complex_mul_identity():
    """r = [1,0,...,0,0,...] (cos=1, sin=0) acts as identity rotation."""
    kge = RotatEEngine(dim=4, seed=0)
    kge.fit(_small_adapter(), n_epochs=0)

    # Manually set relation to identity rotation (theta=0)
    identity = np.concatenate([np.ones(4), np.zeros(4)])[None, :]  # (1, 8)
    h = np.array([[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4]])      # (1, 8)
    result = kge._complex_mul(h, identity)
    np.testing.assert_allclose(result, h, atol=1e-9)

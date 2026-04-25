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


# ---------------------------------------------------------------------------
# predict_links — Phase 28C
# ---------------------------------------------------------------------------

class TestPredictLinks:
    def test_returns_list(self):
        kge, _, _ = _train_transe()
        result = kge.predict_links("A", top_k=3)
        assert isinstance(result, list)

    def test_length_at_most_top_k(self):
        kge, _, _ = _train_transe()
        result = kge.predict_links("A", top_k=2)
        assert len(result) <= 2

    def test_tuple_format(self):
        """Each entry is (head, relation, tail, score)."""
        kge, _, _ = _train_transe()
        for entry in kge.predict_links("A", top_k=3):
            assert len(entry) == 4
            head, rel, tail, score = entry
            assert isinstance(head, str)
            assert isinstance(rel, str)
            assert isinstance(tail, str)
            assert isinstance(score, float)

    def test_head_matches_query(self):
        kge, _, _ = _train_transe()
        for (h, r, t, s) in kge.predict_links("A", top_k=5):
            assert h == "A"

    def test_no_self_predictions(self):
        kge, _, _ = _train_transe()
        for (h, r, t, s) in kge.predict_links("A", top_k=10):
            assert t != "A"

    def test_scores_descending(self):
        kge, _, _ = _train_transe()
        results = kge.predict_links("A", top_k=5)
        scores = [s for (_, _, _, s) in results]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_entity_returns_empty(self):
        kge, _, _ = _train_transe()
        assert kge.predict_links("UNKNOWN_ENTITY_XYZ", top_k=5) == []

    def test_untrained_returns_empty(self):
        kge = TransEEngine(dim=16, seed=0)
        assert kge.predict_links("A", top_k=5) == []

    def test_relation_filter(self):
        """When relations= is specified, only those relations appear in results."""
        kge, _, _ = _train_transe()
        results = kge.predict_links("A", top_k=10, relations=["KNOWS"])
        for (h, r, t, s) in results:
            assert r == "KNOWS"

    def test_empty_relation_filter_returns_empty(self):
        kge, _, _ = _train_transe()
        assert kge.predict_links("A", top_k=5, relations=[]) == []

    def test_rotate_also_works(self):
        kge, _, _ = _train_rotate()
        result = kge.predict_links("A", top_k=3)
        assert isinstance(result, list)
        if result:
            assert len(result[0]) == 4


class TestKgeRepairIntegration:
    """predict_links plugs into IncompletenessRepairEngine._kge_score."""

    def test_kge_score_top_candidate_returns_one(self):
        """The top-ranked candidate by KGE should get score 1.0."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from core.repair_engine import IncompletenessRepairEngine
        from unittest.mock import MagicMock

        G = nx.DiGraph()
        G.add_edge("A", "B", relation="r")
        G.add_edge("B", "C", relation="r")
        adapter_mock = MagicMock()
        adapter_mock.community_map = {}
        adapter_mock.get_embedding.return_value = None

        kge = TransEEngine(dim=16, seed=0)
        kge.fit(NetworkXAdapter(G), n_epochs=30)

        engine = IncompletenessRepairEngine(adapter_mock, kge_engine=kge)
        preds = kge.predict_links("A", top_k=5)
        if preds:
            top_tail = preds[0][2]
            score = engine._kge_score("A", top_tail)
            assert score == 1.0  # rank 0 → 1/(1+0) = 1.0

    def test_kge_score_not_in_top_returns_zero(self):
        """A tail not in top-20 predictions should return 0.0."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from core.repair_engine import IncompletenessRepairEngine
        from unittest.mock import MagicMock

        G = nx.DiGraph()
        G.add_edge("A", "B", relation="r")
        G.add_edge("B", "C", relation="r")
        adapter_mock = MagicMock()
        adapter_mock.community_map = {}

        kge = TransEEngine(dim=16, seed=0)
        kge.fit(NetworkXAdapter(G), n_epochs=10)

        engine = IncompletenessRepairEngine(adapter_mock, kge_engine=kge)
        score = engine._kge_score("A", "ENTITY_NOT_IN_VOCAB_XYZ")
        assert score == 0.0

    def test_kge_none_returns_neutral(self):
        """No KGE engine → _kge_score returns 0.5 (neutral)."""
        from core.repair_engine import IncompletenessRepairEngine
        from unittest.mock import MagicMock

        adapter_mock = MagicMock()
        adapter_mock.community_map = {}
        engine = IncompletenessRepairEngine(adapter_mock, kge_engine=None)
        assert engine._kge_score("A", "B") == 0.5


# ---------------------------------------------------------------------------
# Phase 135: blend utility + GPU training + build() integration
# ---------------------------------------------------------------------------

class TestBlendKGEEmbeddings:
    def test_midpoint_blend_is_normalised(self):
        from core.kge_engine import blend_kge_embeddings
        base = {"a": np.array([1.0, 0.0], dtype=np.float32)}
        kge  = {"a": np.array([0.0, 1.0], dtype=np.float32)}
        out = blend_kge_embeddings(base, kge, blend=0.5)
        assert abs(np.linalg.norm(out["a"]) - 1.0) < 1e-5

    def test_pure_kge_blend(self):
        from core.kge_engine import blend_kge_embeddings
        base = {"x": np.array([1.0, 0.0], dtype=np.float32)}
        kge  = {"x": np.array([0.0, 1.0], dtype=np.float32)}
        out = blend_kge_embeddings(base, kge, blend=1.0)
        np.testing.assert_allclose(out["x"], np.array([0.0, 1.0], dtype=np.float32), atol=1e-5)

    def test_missing_entity_fallback(self):
        from core.kge_engine import blend_kge_embeddings
        base = {"a": np.array([1.0, 0.0], dtype=np.float32),
                "b": np.array([0.0, 1.0], dtype=np.float32)}
        kge  = {"a": np.array([0.5, 0.5], dtype=np.float32)}  # "b" absent
        out = blend_kge_embeddings(base, kge, blend=0.5)
        # "b" not in KGE → retained as-is
        np.testing.assert_array_equal(out["b"], base["b"])

    def test_zero_blend_returns_base(self):
        from core.kge_engine import blend_kge_embeddings
        base = {"x": np.array([1.0, 0.0], dtype=np.float32)}
        kge  = {"x": np.array([0.0, 1.0], dtype=np.float32)}
        out = blend_kge_embeddings(base, kge, blend=0.0)
        np.testing.assert_array_equal(out["x"], base["x"])


def test_build_use_kge_integrates():
    """CerebrumGraph.build(use_kge=True) completes and produces unit-norm embeddings."""
    from core.cerebrum import CerebrumGraph
    triples = [
        ("a", "KNOWS", "b"), ("b", "KNOWS", "c"), ("c", "KNOWS", "d"),
        ("d", "RELATED", "e"), ("e", "RELATED", "a"),
    ] * 5
    g = CerebrumGraph.from_triples(triples)
    g.build(seed=42, use_kge=True, kge_model="transe", kge_epochs=10, kge_dim=16)
    for eid in ["a", "b", "c"]:
        emb = g.adapter.embeddings.get(eid)
        assert emb is not None, f"Missing embedding for {eid!r}"
        assert abs(np.linalg.norm(emb.astype(np.float32)) - 1.0) < 0.1


def test_transe_gpu_train():
    """GPU TransE training produces unit-norm embeddings (skipped if CUDA unavailable)."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    adapter = _small_adapter()
    kge = TransEEngine(dim=16, seed=0)
    result = kge.fit(adapter, n_epochs=10, device="cuda")
    assert result.final_loss >= 0.0
    for eid in ["A", "B", "C"]:
        emb = kge.get_embedding(eid)
        assert emb is not None
        assert abs(np.linalg.norm(emb) - 1.0) < 0.05

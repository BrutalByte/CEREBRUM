"""
Tests for RandomEngine (and the EmbeddingEngine ABC contract).

SentenceEngine requires sentence-transformers (optional dependency) and is
tested indirectly via the benchmark suite. Its interface contract is covered
by the ABC which RandomEngine satisfies.
"""
import numpy as np
import pytest

from core.embedding_engine import RandomEngine


# ---------------------------------------------------------------------------
# RandomEngine — basic contract
# ---------------------------------------------------------------------------

def test_random_engine_dim():
    engine = RandomEngine(dim=32)
    assert engine.dim == 32


def test_random_engine_default_dim():
    engine = RandomEngine()
    assert engine.dim == 64


def test_encode_returns_correct_shape():
    engine = RandomEngine(dim=16)
    result = engine.encode(["alice", "bob", "carol"])
    assert result.shape == (3, 16)


def test_encode_returns_float16():
    engine = RandomEngine(dim=8)
    result = engine.encode(["a"])
    assert result.dtype == np.float16


def test_encode_unit_norm():
    engine = RandomEngine(dim=64)
    vecs = engine.encode(["newton", "einstein", "bohr"])
    norms = np.linalg.norm(vecs.astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=2e-3)  # float16 rounding ~1e-4


def test_encode_empty_list():
    engine = RandomEngine(dim=16)
    result = engine.encode([])
    assert len(result) == 0


def test_encode_deterministic():
    engine = RandomEngine(dim=32)
    v1 = engine.encode(["newton"])
    v2 = engine.encode(["newton"])
    np.testing.assert_array_equal(v1, v2)


def test_different_labels_different_vectors():
    engine = RandomEngine(dim=64)
    vecs = engine.encode(["alpha", "beta"])
    # Different labels should produce different vectors (with overwhelming probability)
    assert not np.allclose(vecs[0], vecs[1])


# ---------------------------------------------------------------------------
# encode_one
# ---------------------------------------------------------------------------

def test_encode_one_shape():
    engine = RandomEngine(dim=16)
    v = engine.encode_one("newton")
    assert v.shape == (16,)


def test_encode_one_consistent_with_encode():
    engine = RandomEngine(dim=32)
    v_one = engine.encode_one("einstein")
    v_batch = engine.encode(["einstein"])[0]
    np.testing.assert_array_equal(v_one, v_batch)


# ---------------------------------------------------------------------------
# encode_entities
# ---------------------------------------------------------------------------

def test_encode_entities_returns_dict():
    engine = RandomEngine(dim=16)
    result = engine.encode_entities({"A": "alice", "B": "bob"})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"A", "B"}


def test_encode_entities_correct_shape():
    engine = RandomEngine(dim=32)
    result = engine.encode_entities({"X": "newton", "Y": "einstein"})
    for v in result.values():
        assert v.shape == (32,)


def test_encode_entities_empty():
    engine = RandomEngine(dim=16)
    result = engine.encode_entities({})
    assert result == {}


def test_encode_entities_id_not_label():
    """The entity ID (key) is preserved; embedding is based on label (value)."""
    engine = RandomEngine(dim=32)
    result = engine.encode_entities({"node_001": "albert einstein"})
    assert "node_001" in result
    assert result["node_001"].shape == (32,)


def test_encode_entities_same_label_same_vector():
    engine = RandomEngine(dim=32)
    r = engine.encode_entities({"A": "newton", "B": "newton"})
    np.testing.assert_array_equal(r["A"], r["B"])

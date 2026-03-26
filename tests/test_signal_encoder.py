"""
Tests for Gap 2 — Cross-Modal Alignment (core/signal_encoder.py).

Covers:
  - Output shape and L2-normalization for both encoders
  - Determinism and distinctness of encodings
  - Alignment learning (Procrustes)
  - Edge cases: constant signals, dim truncation/padding
  - min_anchors enforcement
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from core.signal_encoder import (
    StatisticalSignalEncoder,
    SpectralSignalEncoder,
    _l2_normalize,
    _statistical_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32

def _make_adapter(entity_dim=DIM, n_entities=10, seed=0):
    """Mock adapter with real unit-norm embeddings."""
    rng = np.random.default_rng(seed)
    embeddings = {}
    for i in range(n_entities):
        v = rng.standard_normal(entity_dim).astype(np.float32)
        v /= np.linalg.norm(v)
        embeddings[f"e{i}"] = v

    adapter = MagicMock()
    adapter.get_embedding.side_effect = lambda eid: embeddings.get(eid)
    return adapter, embeddings


def _random_signal(length=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length).astype(np.float32)


# ---------------------------------------------------------------------------
# StatisticalSignalEncoder — output shape and normalization
# ---------------------------------------------------------------------------

def test_statistical_encoder_output_shape():
    enc = StatisticalSignalEncoder(entity_dim=DIM)
    sig = _random_signal()
    emb = enc.encode_signal(sig)
    assert emb.shape == (DIM,)


def test_statistical_encoder_l2_normalized():
    enc = StatisticalSignalEncoder(entity_dim=DIM)
    sig = _random_signal()
    emb = enc.encode_signal(sig)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_statistical_features_constant_signal():
    """All-zero / constant signal must not raise and must produce finite output."""
    enc = StatisticalSignalEncoder(entity_dim=DIM)
    zero_sig = np.zeros(50, dtype=np.float32)
    emb = enc.encode_signal(zero_sig)
    # If the zero vector falls through, l2_normalize returns zeros — that's fine
    assert emb.shape == (DIM,)
    assert np.all(np.isfinite(emb))


# ---------------------------------------------------------------------------
# SpectralSignalEncoder — output shape and normalization
# ---------------------------------------------------------------------------

def test_spectral_encoder_output_shape():
    enc = SpectralSignalEncoder(entity_dim=DIM)
    sig = _random_signal()
    emb = enc.encode_signal(sig)
    assert emb.shape == (DIM,)


def test_spectral_encoder_l2_normalized():
    enc = SpectralSignalEncoder(entity_dim=DIM)
    sig = _random_signal()
    emb = enc.encode_signal(sig)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_spectral_encoder_dim_truncation():
    """Signal longer than entity_dim — FFT bins must be truncated to entity_dim."""
    enc = SpectralSignalEncoder(entity_dim=8)
    sig = _random_signal(length=200)
    emb = enc.encode_signal(sig)
    assert emb.shape == (8,)


def test_spectral_encoder_dim_padding():
    """Very short signal — FFT output shorter than entity_dim must be zero-padded."""
    enc = SpectralSignalEncoder(entity_dim=64)
    sig = _random_signal(length=4)   # rfft → 3 bins << 64
    emb = enc.encode_signal(sig)
    assert emb.shape == (64,)
    assert np.all(np.isfinite(emb))


# ---------------------------------------------------------------------------
# Shared behavioral tests (both encoders)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_different_signals_different_embeddings(EncoderCls):
    enc = EncoderCls(entity_dim=DIM)
    sig1 = _random_signal(length=100, seed=1)
    sig2 = _random_signal(length=100, seed=2)
    emb1 = enc.encode_signal(sig1)
    emb2 = enc.encode_signal(sig2)
    # Cosine distance > 0 (not identical)
    cos_sim = float(np.dot(emb1, emb2))
    assert cos_sim < 0.999


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_same_signal_deterministic(EncoderCls):
    enc = EncoderCls(entity_dim=DIM)
    sig = _random_signal()
    emb1 = enc.encode_signal(sig)
    emb2 = enc.encode_signal(sig)
    np.testing.assert_array_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# Alignment tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_learn_alignment_returns_anchor_count(EncoderCls):
    n_anchors = DIM
    adapter, _ = _make_adapter(entity_dim=DIM, n_entities=n_anchors)
    enc = EncoderCls(entity_dim=DIM, namespace="")  # no prefix — adapter stores bare IDs
    signals = [_random_signal(seed=i) for i in range(n_anchors)]
    entity_ids = [f"e{i}" for i in range(n_anchors)]
    n = enc.learn_alignment(signals, entity_ids, adapter, min_anchors=3)
    assert n == n_anchors


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_alignment_reduces_distance(EncoderCls):
    """
    After alignment, the total sum-of-squared distances (Frobenius norm)
    between anchor signal embeddings and their target entity embeddings must
    decrease. Procrustes minimises ||AR - B||_F, so this is strictly guaranteed
    for the training anchors regardless of dimensionality.
    """
    # Use DIM anchors so the problem is fully constrained
    n_anchors = DIM
    adapter, entity_embeds = _make_adapter(entity_dim=DIM, n_entities=n_anchors)
    enc = EncoderCls(entity_dim=DIM, namespace="")  # no prefix — adapter stores bare IDs
    signals = [_random_signal(seed=i, length=128) for i in range(n_anchors)]
    entity_ids = [f"e{i}" for i in range(n_anchors)]

    # Pre-alignment sum of squared distances
    pre_sq = sum(
        float(np.linalg.norm(entity_embeds[eid] - enc.encode_signal(sig)) ** 2)
        for sig, eid in zip(signals, entity_ids)
    )

    enc.learn_alignment(signals, entity_ids, adapter, min_anchors=3)

    # Post-alignment sum of squared distances
    post_sq = sum(
        float(np.linalg.norm(entity_embeds[eid] - enc.encode_signal(sig)) ** 2)
        for sig, eid in zip(signals, entity_ids)
    )

    assert post_sq <= pre_sq + 1e-3, (
        f"Alignment increased Frobenius norm: pre={pre_sq:.4f} post={post_sq:.4f}"
    )


def test_min_anchors_enforced():
    """Fewer valid anchors than min_anchors must raise ValueError."""
    # Adapter returns None for all embeddings
    adapter = MagicMock()
    adapter.get_embedding.return_value = None

    enc = StatisticalSignalEncoder(entity_dim=DIM)
    with pytest.raises(ValueError, match="Not enough valid anchor pairs"):
        enc.learn_alignment(
            [_random_signal()],
            ["ghost_entity"],
            adapter,
            min_anchors=3,
        )


def test_min_anchors_exactly_met():
    """Exactly min_anchors valid pairs should not raise."""
    adapter, _ = _make_adapter(entity_dim=DIM, n_entities=3)
    enc = StatisticalSignalEncoder(entity_dim=DIM, namespace="")
    signals = [_random_signal(seed=i) for i in range(3)]
    entity_ids = [f"e{i}" for i in range(3)]
    # Should not raise
    n = enc.learn_alignment(signals, entity_ids, adapter, min_anchors=3)
    assert n == 3


# ---------------------------------------------------------------------------
# l2_normalize helper
# ---------------------------------------------------------------------------

def test_l2_normalize_unit_vector():
    v = np.array([3.0, 4.0], dtype=np.float32)
    result = _l2_normalize(v)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


def test_l2_normalize_zero_vector():
    v = np.zeros(5, dtype=np.float32)
    result = _l2_normalize(v)
    np.testing.assert_array_equal(result, v)

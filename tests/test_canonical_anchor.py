"""
tests/test_canonical_anchor.py
Hole 3 — Recursive Alignment Drift: Canonical Basis Anchor.

Validates that SignalEncoder.learn_alignment() uses canonical_embeddings
when provided, ensuring all alignments target the same fixed root space
rather than chaining through different adapters' embedding spaces.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from core.signal_encoder import StatisticalSignalEncoder, SpectralSignalEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32

def _rng_emb(seed: int, dim: int = DIM) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def _rng_signal(seed: int, length: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(length).astype(np.float32)

def _make_adapter(n: int = DIM, seed: int = 0) -> MagicMock:
    """Adapter whose get_embedding returns deterministic unit embeddings."""
    embeds = {f"signal:e{i}": _rng_emb(seed * 100 + i) for i in range(n)}
    m = MagicMock()
    m.get_embedding.side_effect = lambda eid: embeds.get(eid)
    return m

def _canonical_dict(n: int = DIM, seed: int = 99) -> dict:
    return {f"signal:e{i}": _rng_emb(seed * 100 + i) for i in range(n)}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_no_canonical_embeddings_uses_adapter():
    enc = StatisticalSignalEncoder(entity_dim=DIM)
    assert enc._canonical_embeddings is None


def test_canonical_embeddings_stored():
    canon = _canonical_dict()
    enc = StatisticalSignalEncoder(entity_dim=DIM, canonical_embeddings=canon)
    assert enc._canonical_embeddings is canon


def test_spectral_encoder_stores_canonical():
    canon = _canonical_dict()
    enc = SpectralSignalEncoder(entity_dim=DIM, canonical_embeddings=canon)
    assert enc._canonical_embeddings is canon


# ---------------------------------------------------------------------------
# learn_alignment with canonical_embeddings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_embeddings_used_instead_of_adapter(EncoderCls):
    """When canonical_embeddings is set, adapter.get_embedding must NOT be called."""
    canon = _canonical_dict(n=10)
    enc = EncoderCls(entity_dim=DIM, namespace="signal",
                     canonical_embeddings=canon)
    signals = [_rng_signal(i) for i in range(10)]
    entity_ids = [f"e{i}" for i in range(10)]

    adapter = MagicMock()
    adapter.get_embedding.return_value = _rng_emb(0)

    enc.learn_alignment(signals, entity_ids, adapter=adapter, min_anchors=3)
    adapter.get_embedding.assert_not_called()


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_adapter_none_allowed(EncoderCls):
    """Passing adapter=None is valid when canonical_embeddings covers all anchors."""
    canon = _canonical_dict(n=10)
    enc = EncoderCls(entity_dim=DIM, namespace="signal",
                     canonical_embeddings=canon)
    signals = [_rng_signal(i) for i in range(10)]
    entity_ids = [f"e{i}" for i in range(10)]

    n = enc.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)
    assert n == 10


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_alignment_differs_from_adapter_alignment(EncoderCls):
    """Alignment learned against canonical space must differ from adapter space."""
    # Create two distinct target embedding spaces
    canon  = _canonical_dict(n=DIM, seed=11)
    adapt  = _make_adapter(n=DIM, seed=22)

    signals    = [_rng_signal(i) for i in range(DIM)]
    entity_ids = [f"e{i}" for i in range(DIM)]

    enc_canon = EncoderCls(entity_dim=DIM, namespace="signal",
                           canonical_embeddings=canon)
    enc_adapt = EncoderCls(entity_dim=DIM, namespace="signal")

    enc_canon.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)
    enc_adapt.learn_alignment(signals, entity_ids, adapter=adapt, min_anchors=3)

    assert enc_canon._alignment_R is not None
    assert enc_adapt._alignment_R is not None
    # Rotation matrices should differ (different target spaces)
    assert not np.allclose(enc_canon._alignment_R, enc_adapt._alignment_R, atol=1e-3)


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_alignment_stable_across_two_calls(EncoderCls):
    """
    Calling learn_alignment twice with the same canonical_embeddings should
    yield the same rotation matrix (idempotent when anchors are the same).
    """
    canon = _canonical_dict(n=DIM)
    signals    = [_rng_signal(i) for i in range(DIM)]
    entity_ids = [f"e{i}" for i in range(DIM)]

    enc1 = EncoderCls(entity_dim=DIM, namespace="signal", canonical_embeddings=canon)
    enc2 = EncoderCls(entity_dim=DIM, namespace="signal", canonical_embeddings=canon)

    enc1.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)
    enc2.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)

    np.testing.assert_allclose(enc1._alignment_R, enc2._alignment_R, atol=1e-5)


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_missing_anchor_skipped(EncoderCls):
    """Anchors whose IDs are absent from canonical_embeddings are skipped."""
    canon = {f"signal:e{i}": _rng_emb(i) for i in range(5)}  # only 5 anchors
    enc = EncoderCls(entity_dim=DIM, namespace="signal",
                     canonical_embeddings=canon)

    signals = [_rng_signal(i) for i in range(10)]
    entity_ids = [f"e{i}" for i in range(10)]  # e5-e9 missing from canon

    n = enc.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)
    assert n == 5


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_too_few_anchors_raises(EncoderCls):
    """Fewer canonical anchors than min_anchors raises ValueError."""
    canon = {f"signal:e{i}": _rng_emb(i) for i in range(2)}
    enc = EncoderCls(entity_dim=DIM, namespace="signal",
                     canonical_embeddings=canon)
    signals    = [_rng_signal(i) for i in range(2)]
    entity_ids = [f"e{i}" for i in range(2)]

    with pytest.raises(ValueError, match="Not enough valid anchor pairs"):
        enc.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)


# ---------------------------------------------------------------------------
# encode_signal with canonical alignment applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_encode_signal_applies_canonical_rotation(EncoderCls):
    """After learn_alignment with canonical, encode_signal output differs from raw."""
    canon = _canonical_dict(n=DIM)
    enc = EncoderCls(entity_dim=DIM, namespace="signal", canonical_embeddings=canon)
    signals = [_rng_signal(i) for i in range(DIM)]
    entity_ids = [f"e{i}" for i in range(DIM)]
    enc.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)

    sig = _rng_signal(999)
    raw_enc = EncoderCls(entity_dim=DIM)
    emb_raw    = raw_enc.encode_signal(sig)
    emb_aligned = enc.encode_signal(sig)

    # Rotation changes the embedding
    assert not np.allclose(emb_raw, emb_aligned, atol=1e-3)


@pytest.mark.parametrize("EncoderCls", [StatisticalSignalEncoder, SpectralSignalEncoder])
def test_canonical_bare_id_fallback(EncoderCls):
    """If signal:X not in canonical but X is, encoder accepts bare ID."""
    # Store embedding under bare key, not prefixed
    canon = {f"e{i}": _rng_emb(i) for i in range(DIM)}
    enc = EncoderCls(entity_dim=DIM, namespace="signal", canonical_embeddings=canon)
    signals    = [_rng_signal(i) for i in range(DIM)]
    entity_ids = [f"e{i}" for i in range(DIM)]
    n = enc.learn_alignment(signals, entity_ids, adapter=None, min_anchors=3)
    assert n == DIM

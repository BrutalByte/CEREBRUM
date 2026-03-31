"""
Cross-Modal Alignment — encode non-textual signals into the entity embedding space.

THALAMUS handles text and ID normalization (core/thalamus.py). This module extends
THALAMUS to handle non-textual signals (sensor waveforms, time-series, audio, etc.)
by projecting them into the same embedding space as entity embeddings so they can be
queried via the standard GraphAdapter interface.

Two implementations are provided:

  StatisticalSignalEncoder  — hand-crafted features, no ML dependencies
  SpectralSignalEncoder     — FFT-based, well-suited for waveforms

Both support optional Procrustes alignment to the entity embedding space
(same SVD pattern as FederatedAdapter.align_embeddings in adapters/federated_adapter.py).

Usage
-----
    # Stand-alone, no alignment:
    encoder = StatisticalSignalEncoder(entity_dim=64)
    emb = encoder.encode_signal(waveform)   # shape (64,), L2-normalized

    # With alignment (recommended — maps signal space → entity space):
    encoder.learn_alignment(anchor_signals, anchor_entity_ids, adapter)
    emb = encoder.encode_signal(waveform)   # now in entity embedding space

    # Store and query via adapter:
    adapter.embeddings["sensor_42"] = emb
    adapter.get_embedding("sensor_42")      # standard API
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# SignalEncoder ABC
# ---------------------------------------------------------------------------

class SignalEncoder(ABC):
    """
    Abstract base class for signal encoders.

    A SignalEncoder converts a 1-D NumPy array (e.g. a sensor waveform or time
    series) into an L2-normalized vector in the entity embedding space.

    Subclasses implement:
      - ``dim`` property — output dimensionality
      - ``encode_signal(signal, metadata)`` — raw encoding (pre-alignment)
      - ``_raw_encode(signal)`` — internal method called by encode_signal

    After calling ``learn_alignment()``, ``encode_signal()`` automatically applies
    the Procrustes rotation R that maps the signal embedding space to the entity
    embedding space.
    """

    def __init__(
        self,
        entity_dim: int,
        namespace: str = "signal",
        canonical_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._entity_dim = entity_dim
        self._alignment_R: Optional[np.ndarray] = None  # shape (entity_dim, entity_dim)
        self._namespace = namespace
        # Hole 3 — Recursive Alignment Drift: Canonical Basis Anchor.
        # When provided, learn_alignment() looks up target embeddings here instead
        # of from the per-query adapter.  All alignments therefore target the same
        # fixed "root space", preventing geometric drift when encoders are chained
        # across federated hops or multiple cross-modal projections.
        self._canonical_embeddings: Optional[Dict[str, np.ndarray]] = canonical_embeddings

    def get_namespaced_id(self, entity_id: str) -> str:
        """Return the entity_id with this encoder's namespace prefix applied."""
        return f"{self._namespace}:{entity_id}" if self._namespace else entity_id

    @property
    def dim(self) -> int:
        """Output dimensionality of encoded embeddings."""
        return self._entity_dim

    @abstractmethod
    def _raw_encode(self, signal: np.ndarray) -> np.ndarray:
        """
        Encode signal to a unit-norm vector of shape (entity_dim,).
        Called internally by encode_signal() before optional alignment rotation.
        """
        ...

    def encode_signal(
        self,
        signal: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Encode a signal to a unit-norm vector in the entity embedding space.

        Parameters
        ----------
        signal   : 1-D NumPy array (any length, any dtype)
        metadata : optional dict of signal metadata (passed to subclasses that
                   use it; ignored by default implementations)

        Returns
        -------
        np.ndarray of shape (entity_dim,), L2-normalized.
        """
        signal = np.asarray(signal, dtype=np.float32).ravel()
        emb = self._raw_encode(signal)

        # Apply Procrustes alignment if learned
        if self._alignment_R is not None:
            # If learn_alignment finds R s.t. A R ≈ B (row vectors),
            # then for a column vector x: R.T @ x ≈ y.
            emb = self._alignment_R.T @ emb

        return _l2_normalize(emb)

    def learn_alignment(
        self,
        anchor_signals: List[np.ndarray],
        anchor_entity_ids: List[str],
        adapter,
        min_anchors: int = 3,
    ) -> int:
        """
        Fit a Procrustes rotation R: signal_embed_space → entity_embed_space.

        Uses SVD Procrustes — the same pattern as
        ``FederatedAdapter.align_embeddings`` in ``adapters/federated_adapter.py``.

        Parameters
        ----------
        anchor_signals     : list of 1-D signal arrays (known waveforms)
        anchor_entity_ids  : entity IDs whose embeddings serve as targets
        adapter            : any GraphAdapter with ``get_embedding(entity_id)``
        min_anchors        : minimum paired anchor count; raises ValueError if fewer

        Returns
        -------
        int — number of anchor pairs actually used (may be < len(anchor_signals)
        if some entities have no embedding).
        """
        if len(anchor_signals) != len(anchor_entity_ids):
            raise ValueError(
                "anchor_signals and anchor_entity_ids must have the same length"
            )

        # Gather valid pairs (signal embed + entity embed).
        # Hole 3 — Canonical Basis Anchor: when canonical_embeddings is set,
        # target embeddings are drawn from that fixed dict rather than the
        # adapter, so all alignments target the same root embedding space.
        sig_embeds = []
        ent_embeds = []
        for sig, eid in zip(anchor_signals, anchor_entity_ids):
            ns_eid = self.get_namespaced_id(eid)
            if self._canonical_embeddings is not None:
                ent_emb = self._canonical_embeddings.get(ns_eid)
                if ent_emb is None:
                    ent_emb = self._canonical_embeddings.get(eid)
            else:
                ent_emb = adapter.get_embedding(ns_eid) if adapter is not None else None
            if ent_emb is None:
                continue
            sig_embeds.append(self._raw_encode(
                np.asarray(sig, dtype=np.float32).ravel()
            ))
            ent_embeds.append(np.asarray(ent_emb, dtype=np.float32).ravel())

        n_pairs = len(sig_embeds)
        if n_pairs < min_anchors:
            raise ValueError(
                f"Not enough valid anchor pairs: got {n_pairs}, need {min_anchors}. "
                "Check that anchor entity IDs exist in the adapter."
            )

        # A: signal embeddings (n_pairs × entity_dim)
        # B: entity embeddings (n_pairs × entity_dim)
        A = np.stack(sig_embeds, axis=0)   # (n, d)
        B = np.stack(ent_embeds, axis=0)   # (n, d)

        # Procrustes SVD: min ||A R - B||_F  → R = U Vᵀ where M = Aᵀ B = U Σ Vᵀ
        M = A.T @ B                         # (d, d)
        U, _s, Vt = np.linalg.svd(M)
        R = (U @ Vt).astype(np.float32)    # (d, d) orthogonal rotation

        self._alignment_R = R
        return n_pairs


# ---------------------------------------------------------------------------
# StatisticalSignalEncoder
# ---------------------------------------------------------------------------

_N_STAT_FEATURES = 16   # mean, std, min, max, range, zcr, energy, peak_count, 8 fft bins


class StatisticalSignalEncoder(SignalEncoder):
    """
    Hand-crafted statistical feature encoder — no ML dependencies.

    Features (16 total):
      mean, std, min, max, range, zero-crossing rate, energy,
      peak count, first 8 FFT magnitude bins

    A fixed random projection matrix W (entity_dim × 16) maps the feature
    vector into the entity embedding dimension — the same approach as
    RandomEngine in ``core/embedding_engine.py``.

    Parameters
    ----------
    entity_dim : output embedding dimension (must match the adapter's embeddings)
    seed       : random seed for the projection matrix
    """

    def __init__(
        self,
        entity_dim: int = 64,
        seed: int = 42,
        namespace: str = "signal",
        canonical_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__(entity_dim, namespace=namespace, canonical_embeddings=canonical_embeddings)
        rng = np.random.default_rng(seed)
        # W shape: (entity_dim, n_features)
        self._W = rng.standard_normal((entity_dim, _N_STAT_FEATURES)).astype(np.float32)
        # L2-normalize columns so projection preserves unit norms
        col_norms = np.linalg.norm(self._W, axis=0, keepdims=True)
        col_norms = np.where(col_norms == 0, 1.0, col_norms)
        self._W /= col_norms

    def _raw_encode(self, signal: np.ndarray) -> np.ndarray:
        features = _statistical_features(signal)   # (15,)
        projected = self._W @ features             # (entity_dim,)
        return _l2_normalize(projected)


# ---------------------------------------------------------------------------
# SpectralSignalEncoder
# ---------------------------------------------------------------------------

class SpectralSignalEncoder(SignalEncoder):
    """
    FFT-based encoder — good for waveforms and periodic signals.

    Algorithm:
      1. ``np.fft.rfft`` → magnitude spectrum
      2. Log-scale compression: ``log(1 + |spectrum|)``
      3. Pad or truncate to entity_dim
      4. L2-normalize

    Parameters
    ----------
    entity_dim : output embedding dimension
    """

    def __init__(
        self,
        entity_dim: int = 64,
        namespace: str = "signal",
        canonical_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__(entity_dim, namespace=namespace, canonical_embeddings=canonical_embeddings)

    def _raw_encode(self, signal: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(signal))          # (n//2+1,)
        log_spectrum = np.log1p(spectrum).astype(np.float32)

        d = self._entity_dim
        n = len(log_spectrum)
        if n >= d:
            emb = log_spectrum[:d]
        else:
            emb = np.zeros(d, dtype=np.float32)
            emb[:n] = log_spectrum

        return _l2_normalize(emb)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of v; returns zero vector if norm is zero."""
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _statistical_features(signal: np.ndarray) -> np.ndarray:
    """
    Compute 15 hand-crafted statistical features from a 1-D signal.
    Handles edge cases (constant, empty) without raising.
    """
    if len(signal) == 0:
        return np.zeros(_N_STAT_FEATURES, dtype=np.float32)

    s = signal.astype(np.float64)
    mean  = float(np.mean(s))
    std   = float(np.std(s))
    smin  = float(np.min(s))
    smax  = float(np.max(s))
    rng   = smax - smin

    # Zero-crossing rate
    if len(s) > 1:
        zcr = float(np.sum(np.diff(np.sign(s)) != 0)) / (len(s) - 1)
    else:
        zcr = 0.0

    # Signal energy (mean squared amplitude)
    energy = float(np.mean(s ** 2))

    # Peak count (simple local maxima)
    if len(s) >= 3:
        diffs = np.diff(s)
        peak_count = float(np.sum((diffs[:-1] > 0) & (diffs[1:] <= 0)))
    else:
        peak_count = 0.0

    # First 8 FFT magnitude bins (log-compressed)
    spectrum = np.abs(np.fft.rfft(s))
    fft_bins = np.log1p(spectrum[:8]) if len(spectrum) >= 8 else \
        np.pad(np.log1p(spectrum), (0, 8 - len(spectrum)))
    fft_bins = fft_bins[:8].astype(np.float64)

    features = np.array(
        [mean, std, smin, smax, rng, zcr, energy, peak_count] + list(fft_bins),
        dtype=np.float32,
    )
    return features

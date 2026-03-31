"""
Phase 17.5 — Knowledge Graph Embedding Engines.

Provides two KGE models trained directly on graph triples:

TransE  — Bordes et al. 2013.  Models relations as vector translations:
          score(h, r, t) = -‖h + r − t‖₂

RotatE  — Sun et al. 2019.  Models relations as rotations in complex space:
          score(h, r, t) = -‖h ∘ r − t‖₂  (complex element-wise product)

Both use margin ranking loss with uniform negative sampling (corrupt either
head or tail), trained with mini-batch SGD.

After training, call .get_embedding(entity_id) to retrieve a numpy vector
compatible with the GraphAdapter embedding interface.

Usage::

    from core.kge_engine import TransEEngine
    kge = TransEEngine(dim=64, margin=1.0, lr=0.01)
    kge.fit(adapter, n_epochs=100)
    emb = kge.get_embedding("Newton")
    # Pass to CSAEngine via adapter or directly
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KGETrainingResult:
    """Summary of a KGE training run."""

    model: str              # "TransE" or "RotatE"
    n_entities: int
    n_relations: int
    n_triples: int
    n_epochs: int
    final_loss: float
    duration_seconds: float
    embedding_dim: int

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KGETrainingResult(model={self.model!r}, entities={self.n_entities}, "
            f"triples={self.n_triples}, loss={self.final_loss:.6f}, "
            f"epochs={self.n_epochs})"
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseKGEEngine:
    """
    Shared infrastructure for KGE models.

    Subclasses implement _score_batch() and _model_name.
    """

    _model_name: str = "BaseKGE"

    def __init__(
        self,
        dim: int = 64,
        margin: float = 1.0,
        lr: float = 0.01,
        batch_size: int = 128,
        seed: Optional[int] = 42,
    ):
        self.dim        = dim
        self.margin     = margin
        self.lr         = lr
        self.batch_size = batch_size
        self._rng       = random.Random(seed)
        self._np_rng    = np.random.default_rng(seed)

        # Populated by fit()
        self._entity_ids: List[str]     = []
        self._relation_ids: List[str]   = []
        self._entity_index: Dict[str, int]   = {}
        self._relation_index: Dict[str, int] = {}
        self._entity_emb:   Optional[np.ndarray] = None  # (n_ent, dim) or (n_ent, dim*2) for RotatE
        self._relation_emb: Optional[np.ndarray] = None
        self._triples: List[Tuple[int, int, int]] = []   # (head_idx, rel_idx, tail_idx)
        self._result: Optional[KGETrainingResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[KGETrainingResult]:
        return self._result

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Return the learned embedding for entity_id, or None if unknown / not trained."""
        if self._entity_emb is None:
            return None
        idx = self._entity_index.get(entity_id)
        if idx is None:
            return None
        return self._entity_emb[idx].copy()

    def get_relation_embedding(self, relation: str) -> Optional[np.ndarray]:
        """Return the learned relation vector, or None if unknown / not trained."""
        if self._relation_emb is None:
            return None
        idx = self._relation_index.get(relation)
        if idx is None:
            return None
        return self._relation_emb[idx].copy()

    def predict_links(
        self,
        head_entity: str,
        top_k: int = 10,
        relations: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Predict the most plausible missing tail entities for ``head_entity``.

        For each known relation (or the subset specified by ``relations``),
        scores all entities as potential tails using the trained model's
        scoring function and returns the top-K (head, relation, tail, score)
        triples sorted by descending plausibility score (higher = more
        plausible; internally the model produces *distances* that are
        negated so the caller always sees higher = better).

        Parameters
        ----------
        head_entity
            Source entity ID.  Must be in the training vocabulary.
        top_k
            Number of (rel, tail, score) candidates to return.
        relations
            Optional list of relation types to consider.  If None, all
            known relations are used.

        Returns
        -------
        list of (head, relation, tail, score) tuples, highest score first.
        Returns an empty list if the model has not been trained or the
        head_entity is unknown.
        """
        if self._entity_emb is None or self._relation_emb is None:
            return []
        h_idx = self._entity_index.get(head_entity)
        if h_idx is None:
            return []

        rel_names = relations if relations is not None else self._relation_ids
        if not rel_names:
            return []

        h_emb = self._entity_emb[h_idx]          # (dim,)
        n_ent = len(self._entity_ids)

        all_scored: List[Tuple[str, str, str, float]] = []

        for rel in rel_names:
            r_idx = self._relation_index.get(rel)
            if r_idx is None:
                continue
            r_emb = self._relation_emb[r_idx]    # (dim,)

            # Broadcast score over all tail candidates at once
            # _score_batch expects (batch, dim) arrays
            h_batch = np.broadcast_to(h_emb, (n_ent, len(h_emb)))
            r_batch = np.broadcast_to(r_emb, (n_ent, len(r_emb)))
            t_batch = self._entity_emb             # (n_ent, dim)

            raw_scores = self._score_batch(h_batch, r_batch, t_batch)
            # raw_scores are *distances* (lower = more plausible for TransE/RotatE).
            # Negate so that higher = more plausible.
            plausibility = -raw_scores

            # Grab top-k indices for this relation
            if len(plausibility) <= top_k:
                best_idxs = np.argsort(plausibility)[::-1]
            else:
                best_idxs = np.argpartition(plausibility, -top_k)[-top_k:]
                best_idxs = best_idxs[np.argsort(plausibility[best_idxs])[::-1]]

            for t_idx in best_idxs:
                tail = self._entity_ids[int(t_idx)]
                if tail == head_entity:
                    continue  # skip self-prediction
                score = float(plausibility[t_idx])
                all_scored.append((head_entity, rel, tail, score))

        # Global top-k across all relations
        all_scored.sort(key=lambda x: x[3], reverse=True)
        return all_scored[:top_k]

    def fit(self, adapter, n_epochs: int = 100) -> KGETrainingResult:
        """
        Train on all triples extracted from the adapter.

        Parameters
        ----------
        adapter  : GraphAdapter — must implement get_all_entities() and get_neighbors().
        n_epochs : number of passes over the training triples.

        Returns
        -------
        KGETrainingResult
        """
        t0 = time.monotonic()
        self._build_vocab(adapter)
        self._init_embeddings()
        triples = self._triples

        if not triples or n_epochs == 0:
            result = KGETrainingResult(
                model=self._model_name,
                n_entities=len(self._entity_ids),
                n_relations=len(self._relation_ids),
                n_triples=len(triples),
                n_epochs=0,
                final_loss=0.0,
                duration_seconds=time.monotonic() - t0,
                embedding_dim=self.dim,
            )
            self._result = result
            return result

        final_loss = 0.0
        for epoch in range(n_epochs):
            self._rng.shuffle(triples)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, len(triples), self.batch_size):
                batch = triples[start : start + self.batch_size]
                epoch_loss += self._train_batch(batch)
                n_batches  += 1
            final_loss = epoch_loss / max(n_batches, 1)

        result = KGETrainingResult(
            model=self._model_name,
            n_entities=len(self._entity_ids),
            n_relations=len(self._relation_ids),
            n_triples=len(triples),
            n_epochs=n_epochs,
            final_loss=final_loss,
            duration_seconds=time.monotonic() - t0,
            embedding_dim=self.dim,
        )
        self._result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vocab(self, adapter) -> None:
        """Extract all entities and relations from the adapter."""
        entities = adapter.get_all_entities()
        entity_set = {e.id for e in entities}
        self._entity_ids   = sorted(entity_set)
        self._entity_index = {eid: i for i, eid in enumerate(self._entity_ids)}

        relations: set = set()
        triples: List[Tuple[int, Any, int]] = []
        for entity in entities:
            for edge in adapter.get_neighbors(entity.id):
                rel = edge.relation_type
                relations.add(rel)
                if edge.target_id not in self._entity_index:
                    # Target not in vocab — skip
                    continue
                triples.append((
                    self._entity_index[entity.id],
                    -1,  # placeholder — filled after relation index built
                    self._entity_index[edge.target_id],
                ))
                # Store relation string for index
                triples[-1] = (
                    self._entity_index[entity.id],
                    rel,
                    self._entity_index[edge.target_id],
                )

        self._relation_ids   = sorted(relations)
        self._relation_index = {r: i for i, r in enumerate(self._relation_ids)}

        # Convert relation strings to indices
        self._triples = [
            (h, self._relation_index[r], t)
            for h, r, t in triples
            if r in self._relation_index
        ]

    def _init_embeddings(self) -> None:
        """Initialise entity and relation embeddings (subclass may override)."""
        n_ent = len(self._entity_ids)
        n_rel = len(self._relation_ids)
        scale = 6.0 / math.sqrt(self.dim)
        self._entity_emb   = self._np_rng.uniform(-scale, scale, (n_ent, self.dim))
        self._relation_emb = self._np_rng.uniform(-scale, scale, (n_rel, self.dim))
        self._normalise_entities()

    def _normalise_entities(self) -> None:
        """L2-normalise all entity embeddings (TransE convention)."""
        if self._entity_emb is None:
            return
        norms = np.linalg.norm(self._entity_emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._entity_emb /= norms

    def _corrupt(self, h: int, r: int, t: int) -> Tuple[int, int, int]:
        """Uniform negative sample: replace head or tail randomly."""
        n = len(self._entity_ids)
        if self._rng.random() < 0.5:
            return (self._rng.randint(0, n - 1), r, t)
        return (h, r, self._rng.randint(0, n - 1))

    def _score_batch(
        self,
        h_emb: np.ndarray,
        r_emb: np.ndarray,
        t_emb: np.ndarray,
    ) -> np.ndarray:
        """Compute scores for a batch. Lower = more plausible (for TransE/RotatE)."""
        raise NotImplementedError

    def _train_batch(self, batch: List[Tuple[int, int, int]]) -> float:
        """SGD step on one batch. Returns mean margin loss."""
        n = len(batch)
        h_idx = np.array([t[0] for t in batch])
        r_idx = np.array([t[1] for t in batch])
        t_idx = np.array([t[2] for t in batch])

        neg_batch = [self._corrupt(*triple) for triple in batch]
        nh_idx = np.array([t[0] for t in neg_batch])
        nt_idx = np.array([t[2] for t in neg_batch])

        if self._entity_emb is None or self._relation_emb is None:
            return 0.0

        h_emb  = self._entity_emb[h_idx]
        r_emb  = self._relation_emb[r_idx]
        t_emb  = self._entity_emb[t_idx]
        nh_emb = self._entity_emb[nh_idx]
        nt_emb = self._entity_emb[nt_idx]

        pos_scores = self._score_batch(h_emb, r_emb, t_emb)
        neg_scores_h = self._score_batch(nh_emb, r_emb, t_emb)
        neg_scores_t = self._score_batch(h_emb, r_emb, nt_emb)
        neg_scores = np.minimum(neg_scores_h, neg_scores_t)

        # Margin ranking loss: L = max(0, margin + pos - neg)
        # (for TransE/RotatE, lower score = more plausible, so pos < neg ideally)
        loss_vec = np.maximum(0.0, self.margin + pos_scores - neg_scores)
        loss = float(loss_vec.mean())

        # Gradient: update only violating triples
        violating = loss_vec > 0.0
        if not violating.any():
            return loss

        # Approximate gradient update (simplified SGD on embedding norms)
        grad_scale = self.lr / max(n, 1)
        self._gradient_update(
            h_idx, r_idx, t_idx,
            nh_idx, nt_idx,
            h_emb, r_emb, t_emb, nh_emb, nt_emb,
            violating, grad_scale,
        )
        self._normalise_entities()
        return loss

    def _gradient_update(
        self,
        h_idx, r_idx, t_idx,
        nh_idx, nt_idx,
        h_emb, r_emb, t_emb, nh_emb, nt_emb,
        violating, grad_scale,
    ) -> None:
        """Subclass-specific gradient update step."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TransE
# ---------------------------------------------------------------------------

class TransEEngine(_BaseKGEEngine):
    """
    TransE: relations as vector translations.
    score(h, r, t) = ‖h + r − t‖₂  (lower = more plausible)
    """

    _model_name = "TransE"

    def _score_batch(self, h_emb, r_emb, t_emb) -> np.ndarray:
        diff = h_emb + r_emb - t_emb
        return np.linalg.norm(diff, axis=1)

    def _gradient_update(
        self,
        h_idx, r_idx, t_idx,
        nh_idx, nt_idx,
        h_emb, r_emb, t_emb, nh_emb, nt_emb,
        violating, grad_scale,
    ) -> None:
        if self._entity_emb is None or self._relation_emb is None:
            return
        # Gradient of ‖h + r - t‖ w.r.t. h, r, t
        pos_diff = h_emb + r_emb - t_emb
        pos_norm = np.linalg.norm(pos_diff, axis=1, keepdims=True)
        pos_norm = np.where(pos_norm == 0, 1.0, pos_norm)
        pos_grad = pos_diff / pos_norm   # unit vector in direction of diff

        neg_diff_t = h_emb + r_emb - nt_emb
        neg_norm_t = np.linalg.norm(neg_diff_t, axis=1, keepdims=True)
        neg_norm_t = np.where(neg_norm_t == 0, 1.0, neg_norm_t)
        neg_grad_t = neg_diff_t / neg_norm_t

        v = violating.astype(float)[:, None]

        # Positive triple: decrease ‖h + r - t‖ → push h+r toward t
        np.add.at(self._entity_emb,   h_idx,  -grad_scale * v * pos_grad)
        np.add.at(self._relation_emb, r_idx,  -grad_scale * v * pos_grad)
        np.add.at(self._entity_emb,   t_idx,   grad_scale * v * pos_grad)

        # Negative triple: increase score → push h+r away from nt
        np.add.at(self._entity_emb,   h_idx,   grad_scale * v * neg_grad_t)
        np.add.at(self._relation_emb, r_idx,   grad_scale * v * neg_grad_t)
        np.add.at(self._entity_emb,   nt_idx, -grad_scale * v * neg_grad_t)


# ---------------------------------------------------------------------------
# RotatE
# ---------------------------------------------------------------------------

class RotatEEngine(_BaseKGEEngine):
    """
    RotatE: relations as rotations in complex space.
    score(h, r, t) = ‖h ∘ r − t‖  where ∘ is element-wise complex multiplication.

    Embeddings are stored as (n, dim*2) arrays: first dim entries = real part,
    last dim entries = imaginary part.  Relation embeddings are unit-modulus
    complex numbers: r = exp(iθ), stored as [cos θ, sin θ] pairs.
    """

    _model_name = "RotatE"

    def _init_embeddings(self) -> None:
        n_ent = len(self._entity_ids)
        n_rel = len(self._relation_ids)
        scale = 6.0 / math.sqrt(self.dim)

        # Entity: complex (real + imag), shape (n_ent, dim*2)
        self._entity_emb = self._np_rng.uniform(-scale, scale, (n_ent, self.dim * 2))
        # Relation: angles θ, stored as (cos θ, sin θ), shape (n_rel, dim*2)
        theta = self._np_rng.uniform(-math.pi, math.pi, (n_rel, self.dim))
        self._relation_emb = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)

    def _normalise_entities(self) -> None:
        pass  # RotatE doesn't require L2-normalisation of entity vectors

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Return only the real part as the usable embedding vector."""
        if self._entity_emb is None:
            return None
        idx = self._entity_index.get(entity_id)
        if idx is None:
            return None
        return self._entity_emb[idx, : self.dim].copy()

    def _complex_mul(
        self, h: np.ndarray, r: np.ndarray
    ) -> np.ndarray:
        """Element-wise complex multiplication h ∘ r."""
        h_re, h_im = h[:, :self.dim], h[:, self.dim:]
        r_re, r_im = r[:, :self.dim], r[:, self.dim:]
        re_out = h_re * r_re - h_im * r_im
        im_out = h_re * r_im + h_im * r_re
        return np.concatenate([re_out, im_out], axis=1)

    def _score_batch(self, h_emb, r_emb, t_emb) -> np.ndarray:
        hr = self._complex_mul(h_emb, r_emb)
        diff = hr - t_emb
        return np.linalg.norm(diff, axis=1)

    def _gradient_update(
        self,
        h_idx, r_idx, t_idx,
        nh_idx, nt_idx,
        h_emb, r_emb, t_emb, nh_emb, nt_emb,
        violating, grad_scale,
    ) -> None:
        if self._entity_emb is None or self._relation_emb is None:
            return
        v = violating.astype(float)[:, None]

        # Positive: decrease ‖h∘r - t‖
        pos_hr   = self._complex_mul(h_emb, r_emb)
        pos_diff = pos_hr - t_emb
        pos_norm = np.linalg.norm(pos_diff, axis=1, keepdims=True)
        pos_norm = np.where(pos_norm == 0, 1.0, pos_norm)
        pos_g    = pos_diff / pos_norm

        np.add.at(self._entity_emb, h_idx, -grad_scale * v * pos_g)
        np.add.at(self._entity_emb, t_idx,  grad_scale * v * pos_g)

        # Negative (corrupt tail): increase ‖h∘r - nt‖
        neg_hr   = self._complex_mul(h_emb, r_emb)
        neg_diff = neg_hr - nt_emb
        neg_norm = np.linalg.norm(neg_diff, axis=1, keepdims=True)
        neg_norm = np.where(neg_norm == 0, 1.0, neg_norm)
        neg_g    = neg_diff / neg_norm

        np.add.at(self._entity_emb, h_idx,   grad_scale * v * neg_g)
        np.add.at(self._entity_emb, nt_idx, -grad_scale * v * neg_g)

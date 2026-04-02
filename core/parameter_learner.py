"""
Phase 17.4 — Learned CSA Parameters.

CSAParameterLearner optimises the five attention coefficients
(alpha, beta, gamma, delta, epsilon) via numerical gradient descent
on a path-ranking loss.

Loss (pairwise margin ranking):
  For each (positive_path, negative_path) pair drawn from training data,
  we want score(positive) > score(negative) + margin.
  L = sum(max(0, margin - score(pos) + score(neg)))

Gradient is approximated with symmetric finite differences:
  dL/dθ_i ≈ (L(θ + h*e_i) - L(θ - h*e_i)) / (2h)

Usage::

    learner = CSAParameterLearner(adapter)
    learner.fit(training_pairs)  # list of (positive_path, negative_path)
    alpha, beta, gamma, delta, epsilon = learner.params
    csa = CSAEngine(adapter, alpha=alpha, beta=beta, ...)
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.attention_engine import CSAEngine, _sigmoid


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LearningResult:
    """Outcome of a CSAParameterLearner.fit() call."""

    params: Tuple[float, float, float, float, float]
    """Learned (alpha, beta, gamma, delta, epsilon)."""

    initial_loss: float
    final_loss: float
    n_iterations: int
    duration_seconds: float
    converged: bool

    def __repr__(self) -> str:  # pragma: no cover
        a, b, g, d, e = self.params
        return (
            f"LearningResult(α={a:.4f}, β={b:.4f}, γ={g:.4f}, δ={d:.4f}, ε={e:.4f}, "
            f"loss={self.final_loss:.6f}, iters={self.n_iterations}, "
            f"converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

_PARAM_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon")
_DEFAULT_INIT = (0.4, 0.4, 0.1, 0.05, 0.05)


class MetaParameterLearner:
    """
    Adaptive Parameter Learning — Phase 22 (Milestone 4).
    
    Maintains community-specific overrides for CSA attention coefficients.
    Adapts parameters online from query feedback using local gradient descent.
    """

    def __init__(
        self,
        global_prior: Tuple[float, float, float, float, float] = _DEFAULT_INIT,
        learning_rate: float = 0.05,
        momentum: float = 0.9,
    ):
        self.global_prior = np.array(global_prior, dtype=np.float32)
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # {community_id -> parameter_vector}
        self.community_overrides: Dict[int, np.ndarray] = {}
        # {community_id -> accumulated_gradient}
        self._velocity: Dict[int, np.ndarray] = {}

    def get_params(self, community_id: int) -> Tuple[float, float, float, float, float]:
        """Return the parameter vector for a specific community."""
        p = self.community_overrides.get(community_id, self.global_prior)
        return tuple(p.tolist())

    def update_from_feedback(
        self, 
        path, 
        reward: float, 
        margin: float = 0.1
    ):
        """
        Perform a local gradient update based on a single path and its reward.
        
        reward: 1.0 for positive feedback, -1.0 for negative.
        """
        edge_features = getattr(path, "edge_features", None)
        if not edge_features:
            return

        cseq = getattr(path, "community_sequence", [])
        if not cseq:
            return

        # Simple online SGD update
        # dL/dθ = -reward * ∇score
        for k, (sim, cs, etw, nd, hd) in enumerate(edge_features):
            cid = cseq[k] if k < len(cseq) else -1
            if cid < 0:
                continue
            
            # Current params for this community
            theta = self.community_overrides.get(cid, self.global_prior.copy())
            
            # Local score gradient (approximated for sigmoid)
            # score = log(sigmoid(theta dot features))
            # d_score/d_theta = (1 - sigmoid) * features
            raw = np.dot(theta, np.array([sim, cs, etw, -nd, hd]))
            sig = _sigmoid(raw)
            grad = (1.0 - sig) * np.array([sim, cs, etw, -nd, hd])
            
            # Update step
            delta = self.learning_rate * reward * grad
            
            # Apply momentum
            v = self._velocity.get(cid, np.zeros(5, dtype=np.float32))
            v = self.momentum * v + delta
            self._velocity[cid] = v
            
            theta += v
            # Clip
            theta = np.clip(theta, 0.0, 2.0)
            self.community_overrides[cid] = theta


class CSAParameterLearner:
    """
    Learn CSA attention coefficients from pairwise path-ranking supervision.

    Parameters
    ----------
    adapter          : GraphAdapter — used to create CSAEngine instances for scoring.
    init_params      : starting (α,β,γ,δ,ε); defaults to paper zero-shot values.
    learning_rate    : gradient descent step size.
    max_iterations   : iteration cap.
    margin           : ranking margin; paths scored margin apart are "correctly ranked".
    finite_diff_h    : finite-difference step for numerical gradient.
    tolerance        : stop early when loss improvement < tolerance.
    clip             : keep each parameter in [clip_lo, clip_hi].
    """

    def __init__(
        self,
        adapter,
        init_params: Tuple[float, float, float, float, float] = _DEFAULT_INIT,
        learning_rate: float = 0.01,
        max_iterations: int = 200,
        margin: float = 0.1,
        finite_diff_h: float = 1e-4,
        tolerance: float = 1e-6,
        clip: Tuple[float, float] = (0.0, 2.0),
    ):
        self.adapter       = adapter
        self._params       = list(init_params)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.margin        = margin
        self.h             = finite_diff_h
        self.tolerance     = tolerance
        self.clip_lo, self.clip_hi = clip

        self._result: Optional[LearningResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def params(self) -> Tuple[float, float, float, float, float]:
        """Current (alpha, beta, gamma, delta, epsilon)."""
        return tuple(self._params)  # type: ignore[return-value]

    @property
    def result(self) -> Optional[LearningResult]:
        """The most recent LearningResult, or None before fit() is called."""
        return self._result

    def fit(
        self,
        training_pairs: List[Tuple],
        *,
        verbose: bool = False,
    ) -> LearningResult:
        """
        Optimise parameters on a list of (positive_path, negative_path) pairs.

        Each element of training_pairs must be a 2-tuple of objects that have:
          - .attention_weights  : list[float]
          - .community_sequence : list[int]

        The score used is the same log-product attention score used in
        PathScorer.score_path() for the attention component:
          score = sum(log(max(w, 1e-9)) for w in path.attention_weights)

        More precisely, we rebuild a lightweight scalar score directly from
        the raw CSA formula components so the gradient sees how the parameters
        affect individual edge weights.

        Parameters
        ----------
        training_pairs : list of (positive_path, negative_path)
        verbose        : if True, print loss every 20 iterations

        Returns
        -------
        LearningResult
        """
        if not training_pairs:
            result = LearningResult(
                params=self.params,
                initial_loss=0.0,
                final_loss=0.0,
                n_iterations=0,
                duration_seconds=0.0,
                converged=True,
            )
            self._result = result
            return result

        t0 = time.monotonic()
        initial_loss = self._compute_loss(training_pairs)
        prev_loss = initial_loss
        n_iter = 0
        converged = False

        for n_iter in range(1, self.max_iterations + 1):
            grad = self._numerical_gradient(training_pairs)

            # Gradient descent step
            for i in range(5):
                self._params[i] -= self.learning_rate * grad[i]
                # Clip to keep parameters in valid range
                self._params[i] = max(self.clip_lo, min(self.clip_hi, self._params[i]))

            loss = self._compute_loss(training_pairs)
            if verbose and n_iter % 20 == 0:  # pragma: no cover
                print(f"  iter {n_iter:4d}  loss={loss:.6f}")

            if abs(prev_loss - loss) < self.tolerance:
                converged = True
                break
            prev_loss = loss

        final_loss = self._compute_loss(training_pairs)
        result = LearningResult(
            params=self.params,
            initial_loss=initial_loss,
            final_loss=final_loss,
            n_iterations=n_iter,
            duration_seconds=time.monotonic() - t0,
            converged=converged,
        )
        self._result = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_engine(self, params: List[float]) -> CSAEngine:
        a, b, g, d, e = params
        return CSAEngine(
            adapter=self.adapter,
            alpha=a,
            beta=b,
            gamma=g,
            delta=d,
            epsilon=e,
        )

    def _score_path(self, path, engine: CSAEngine) -> float:
        """
        Score a single path using this engine's attention weights.

        If the path has pre-computed attention_weights, use log-product of those.
        Otherwise fall back to 0.0 (neutral).
        """
        weights = getattr(path, "attention_weights", None)
        if not weights:
            return 0.0
        return sum(math.log(max(w, 1e-9)) for w in weights)

    def _score_path_parametric(self, path, params: List[float]) -> float:
        """
        Re-score a path by re-computing edge weights with given params.

        Reads (u, v, edge_type) tuples from path.edge_triples if available,
        otherwise falls back to the stored attention_weights rescaled by
        the ratio of new/default sigmoid inputs.

        For paths that carry raw edge feature tuples (sim, cs, etw, nd, hd):
          score = sum(log(sigmoid(a*sim + b*cs + g*etw - d*nd + e*hd)))
        """
        edge_features = getattr(path, "edge_features", None)
        if edge_features:
            a, b, g, d, e = params
            total = 0.0
            for k, feat in enumerate(edge_features):
                # Handle both legacy (5) and updated (7) edge feature tuples
                if len(feat) == 7:
                    sim, cs, etw, nd, hd, pr_v, td = feat
                else:
                    sim, cs, etw, nd, hd = feat
                
                raw = a * sim + b * cs + g * etw - d * nd + e * hd
                total += math.log(max(_sigmoid(raw), 1e-9))
            return total
        # Fallback: use stored weights (no gradient signal through params)
        return self._score_path(path, self._make_engine(params))

    def _compute_loss(self, training_pairs: List[Tuple]) -> float:
        """
        Pairwise margin ranking loss.
        L = mean(max(0, margin - score(pos) + score(neg)))
        """
        total = 0.0
        for pos_path, neg_path in training_pairs:
            s_pos = self._score_path_parametric(pos_path, self._params)
            s_neg = self._score_path_parametric(neg_path, self._params)
            total += max(0.0, self.margin - s_pos + s_neg)
        return total / max(len(training_pairs), 1)

    def _numerical_gradient(self, training_pairs: List[Tuple]) -> List[float]:
        """
        Symmetric finite-difference gradient of the loss w.r.t. each parameter.
        """
        grad = []
        for i in range(5):
            original = self._params[i]

            self._params[i] = original + self.h
            loss_plus = self._compute_loss(training_pairs)

            self._params[i] = original - self.h
            loss_minus = self._compute_loss(training_pairs)

            self._params[i] = original
            grad.append((loss_plus - loss_minus) / (2.0 * self.h))
        return grad

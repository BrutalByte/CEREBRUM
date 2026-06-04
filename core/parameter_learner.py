"""
Phase 45 — Learned CSA Parameters (10-parameter upgrade).

CSAParameterLearner optimises all ten attention coefficients
(alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)
via numerical gradient descent on a path-ranking loss.

The ten parameters correspond to the Phase 43 CSAEngine formula:

  a(u, v, k) = sigmoid(
      alpha   * sim          # semantic similarity
    + beta    * cs           # community score
    + gamma   * etw          # edge-type weight
    - delta   * nd           # normalised distance penalty
    + epsilon * hd           # hop decay
    + zeta    * pr_v         # PageRank prior
    + eta     * td           # temporal decay
    + iota    * nr_v         # node recency
    - mu      * sd           # synthesis-density penalty
    + theta   * grounding    # confidence / grounding score
  )

Loss (pairwise margin ranking):
  For each (positive_path, negative_path) pair drawn from training data,
  we want score(positive) > score(negative) + margin.
  L = sum(max(0, margin - score(pos) + score(neg)))

Gradient is approximated with symmetric finite differences:
  dL/dθ_i ≈ (L(θ + h*e_i) - L(θ - h*e_i)) / (2h)

Edge feature tuple format (10 elements):
  (sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding)

Shorter tuples (e.g. legacy 5-element) are zero-padded for
backward compatibility.

Usage::

    learner = CSAParameterLearner(adapter)
    learner.fit(training_pairs)  # list of (positive_path, negative_path)
    params = learner.params      # 10-tuple
    csa = CSAEngine(adapter, alpha=params[0], beta=params[1], ...)
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.attention_engine import CSAEngine, _sigmoid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAM_NAMES = (
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "iota", "mu", "theta",
)
# Defaults match CSAEngine.__init__ defaults (Phase 43)
_DEFAULT_INIT: Tuple[float, ...] = (
    0.4,   # alpha  — semantic similarity
    0.4,   # beta   — community score
    0.1,   # gamma  — edge-type weight
    0.05,  # delta  — distance penalty (applied with − sign)
    0.05,  # epsilon — hop decay
    0.1,   # zeta   — PageRank prior
    0.1,   # eta    — temporal decay
    0.05,  # iota   — node recency
    0.1,   # mu     — synthesis-density penalty (applied with − sign)
    1.0,   # theta  — grounding / confidence
)

# Signs applied to each feature in the dot-product formula.
# Positive = add, negative = subtract.
_FEATURE_SIGNS = np.array([1, 1, 1, -1, 1, 1, 1, 1, -1, 1], dtype=np.float32)

_N_PARAMS = len(_PARAM_NAMES)  # 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LearningResult:
    """Outcome of a CSAParameterLearner.fit() call."""

    params: Tuple[float, ...]
    """Learned 10-parameter vector (alpha … theta)."""

    initial_loss: float
    final_loss: float
    n_iterations: int
    duration_seconds: float
    converged: bool

    def __repr__(self) -> str:  # pragma: no cover
        names = _PARAM_NAMES
        pairs = ", ".join(f"{n}={v:.4f}" for n, v in zip(names, self.params))
        return (
            f"LearningResult({pairs}, "
            f"loss={self.final_loss:.6f}, iters={self.n_iterations}, "
            f"converged={self.converged})"
        )


# ---------------------------------------------------------------------------
# MetaParameterLearner
# ---------------------------------------------------------------------------

class MetaParameterLearner:
    """
    Adaptive Parameter Learning — Phase 22 / Phase 45 upgrade.

    Maintains community-specific overrides for all 10 CSA attention
    coefficients.  Adapts parameters online from query feedback using
    local gradient descent with momentum.
    """

    def __init__(
        self,
        global_prior: Tuple[float, ...] = _DEFAULT_INIT,
        learning_rate: float = 0.05,
        momentum: float = 0.9,
    ):
        self.global_prior = np.array(global_prior, dtype=np.float32)
        self._n = len(self.global_prior)
        self.learning_rate = learning_rate
        self.learning_rate_scale = 1.0  # Phase 88 metabolic scaling
        self.momentum = momentum

        # {community_id -> parameter_vector}
        self.community_overrides: Dict[int, np.ndarray] = {}
        # {community_id -> accumulated_gradient}
        self._velocity: Dict[int, np.ndarray] = {}

    def get_params(self, community_id: int) -> Tuple[float, ...]:
        """Return the parameter vector for a specific community."""
        p = self.community_overrides.get(community_id, self.global_prior)
        return tuple(p.tolist())

    def update_from_feedback(
        self,
        path,
        reward: float,
        margin: float = 0.1,
    ) -> None:
        """
        Perform a local gradient update based on a single path and its reward.

        reward: 1.0 for positive feedback, -1.0 for negative.

        Edge features are read from path.edge_features as 10-element tuples
        (sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding).
        Shorter legacy tuples are zero-padded to 10 elements.
        """
        edge_features = getattr(path, "edge_features", None)
        if not edge_features:
            return

        cseq = getattr(path, "community_sequence", [])
        if not cseq:
            return

        signs = _FEATURE_SIGNS[:self._n]

        # Simple online SGD update: dL/dθ = -reward * ∇score
        for k, feat_raw in enumerate(edge_features):
            cid = cseq[k] if k < len(cseq) else -1
            if cid < 0:
                continue

            # Pad to self._n if shorter (backward compat)
            feat = np.zeros(self._n, dtype=np.float32)
            feat[: len(feat_raw)] = feat_raw

            # Current params for this community
            theta = self.community_overrides.get(cid, self.global_prior.copy())

            # Local score gradient (approximated for sigmoid)
            # score = log(sigmoid(theta ⊙ signs · features))
            # d_score/d_theta_i = (1 - sigmoid) * sign_i * feat_i
            signed_feat = signs * feat
            raw = float(np.dot(theta, signed_feat))
            sig = _sigmoid(raw)
            grad = (1.0 - sig) * signed_feat

            # Update step (with metabolic scaling)
            eff_lr = self.learning_rate * self.learning_rate_scale
            delta = eff_lr * reward * grad

            # Apply momentum
            v = self._velocity.get(cid, np.zeros(self._n, dtype=np.float32))
            v = self.momentum * v + delta
            self._velocity[cid] = v

            theta = theta + v
            theta = np.clip(theta, 0.0, 2.0)
            self.community_overrides[cid] = theta

    def triplet_update(
        self,
        positive_features: list,
        negative_features: list,
        community_id: int = -1,
        margin: float = 0.2,
    ) -> None:
        """
        Phase 127: contrastive margin ranking update.

        Minimises: L = max(0, margin - score(pos) + score(neg))
        Gradient via finite differences on the 10-param vector for the given community.
        No-op when the positive already scores higher than negative by >= margin.

        Parameters
        ----------
        positive_features : 10-element feature tuple for the correct path
        negative_features : 10-element feature tuple for the hard negative path
        community_id      : community to update (-1 = global_prior)
        margin            : target score gap (default 0.2)
        """
        if not positive_features or not negative_features:
            return

        pos = np.zeros(self._n, dtype=np.float32)
        neg = np.zeros(self._n, dtype=np.float32)
        pos[: len(positive_features)] = positive_features[: self._n]
        neg[: len(negative_features)] = negative_features[: self._n]

        signs = _FEATURE_SIGNS[: self._n]
        theta = self.community_overrides.get(community_id, self.global_prior.copy())

        def _score(feat: np.ndarray) -> float:
            return _sigmoid(float(np.dot(theta * signs, feat)))

        s_pos = _score(pos)
        s_neg = _score(neg)
        loss = max(0.0, margin - s_pos + s_neg)
        if loss == 0.0:
            return  # already well-separated — no update needed

        # Finite-difference gradient: dL/dθ_i ≈ (L(θ+h) - L(θ-h)) / (2h)
        h = 1e-4
        grad = np.zeros(self._n, dtype=np.float32)
        for i in range(self._n):
            theta_p = theta.copy(); theta_p[i] += h
            theta_n = theta.copy(); theta_n[i] -= h

            def _loss_at(t: np.ndarray) -> float:
                sp = _sigmoid(float(np.dot(t * signs, pos)))
                sn = _sigmoid(float(np.dot(t * signs, neg)))
                return max(0.0, margin - sp + sn)

            grad[i] = (_loss_at(theta_p) - _loss_at(theta_n)) / (2 * h)

        eff_lr = self.learning_rate * self.learning_rate_scale
        delta = -eff_lr * grad  # gradient descent
        v = self._velocity.get(community_id, np.zeros(self._n, dtype=np.float32))
        v = self.momentum * v + delta
        self._velocity[community_id] = v
        theta = np.clip(theta + v, 0.0, 2.0)
        if community_id >= 0:
            self.community_overrides[community_id] = theta
        else:
            self.global_prior = theta

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the learned state to a JSON-compatible dict.

        Schema::

            {
                "global_prior": [float, ...],   # length 10
                "learning_rate": float,
                "momentum": float,
                "community_overrides": {"<cid>": [float, ...]},
            }
        """
        return {
            "global_prior": self.global_prior.tolist(),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "community_overrides": {
                str(cid): vec.tolist()
                for cid, vec in self.community_overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetaParameterLearner":
        """
        Restore a MetaParameterLearner from a dict produced by ``to_dict()``.
        """
        obj = cls(
            global_prior=tuple(data["global_prior"]),
            learning_rate=data.get("learning_rate", 0.05),
            momentum=data.get("momentum", 0.9),
        )
        obj.community_overrides = {
            int(cid): np.array(vec, dtype=np.float32)
            for cid, vec in data.get("community_overrides", {}).items()
        }
        return obj


# ---------------------------------------------------------------------------
# CSAParameterLearner
# ---------------------------------------------------------------------------

class CSAParameterLearner:
    """
    Learn all 10 CSA attention coefficients from pairwise path-ranking
    supervision.

    Parameters
    ----------
    adapter          : GraphAdapter — used to create CSAEngine instances.
    init_params      : starting 10-tuple; defaults to Phase 43 zero-shot values.
    learning_rate    : gradient descent step size.
    max_iterations   : iteration cap.
    margin           : ranking margin; paths scored margin apart are correct.
    finite_diff_h    : finite-difference step for numerical gradient.
    tolerance        : stop early when loss improvement < tolerance.
    clip             : keep each parameter in [clip_lo, clip_hi].
    """

    def __init__(
        self,
        adapter,
        init_params: Tuple[float, ...] = _DEFAULT_INIT,
        learning_rate: float = 0.01,
        max_iterations: int = 200,
        margin: float = 0.1,
        finite_diff_h: float = 1e-4,
        tolerance: float = 1e-6,
        clip: Tuple[float, float] = (0.0, 2.0),
    ):
        self.adapter        = adapter
        self._params        = list(init_params)
        self.learning_rate  = learning_rate
        self.max_iterations = max_iterations
        self.margin         = margin
        self.h              = finite_diff_h
        self.tolerance      = tolerance
        self.clip_lo, self.clip_hi = clip

        self._result: Optional[LearningResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def params(self) -> Tuple[float, ...]:
        """Current 10-parameter tuple (alpha … theta)."""
        return tuple(self._params)

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

        Each path object should carry:
          - .edge_features      : list of 10-tuples (or shorter, zero-padded)
          - .attention_weights  : list[float]  (fallback when edge_features absent)

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
            for i in range(len(self._params)):
                self._params[i] -= self.learning_rate * grad[i]
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
        n = len(params)
        kwargs = {}
        names = _PARAM_NAMES
        for i, name in enumerate(names):
            if i < n:
                kwargs[name] = params[i]
        return CSAEngine(adapter=self.adapter, **kwargs)

    def _score_path(self, path, engine: CSAEngine) -> float:
        """Score a path via log-product of stored attention_weights."""
        weights = getattr(path, "attention_weights", None)
        if not weights:
            return 0.0
        return sum(math.log(max(w, 1e-9)) for w in weights)

    def _score_path_parametric(self, path, params: List[float]) -> float:
        """
        Re-score a path using given params against its edge features.

        Edge features format: 10-tuple
          (sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding)
        Shorter tuples are zero-padded for backward compatibility.

        score = sum over edges of log(sigmoid(params ⊙ signs · features))
        where signs = [+1,+1,+1,-1,+1,+1,+1,+1,-1,+1]
        """
        edge_features = getattr(path, "edge_features", None)
        if edge_features:
            n = len(params)
            signs = _FEATURE_SIGNS[:n]
            p = np.array(params, dtype=np.float64)
            total = 0.0
            for feat_raw in edge_features:
                feat = np.zeros(n, dtype=np.float64)
                feat[: len(feat_raw)] = feat_raw
                raw = float(np.dot(p, signs * feat))
                total += math.log(max(_sigmoid(raw), 1e-9))
            return total
        # Fallback: use stored weights (no gradient signal through params)
        return self._score_path(path, self._make_engine(params))

    def _compute_loss(self, training_pairs: List[Tuple]) -> float:
        """Pairwise margin ranking loss."""
        total = 0.0
        for pos_path, neg_path in training_pairs:
            s_pos = self._score_path_parametric(pos_path, self._params)
            s_neg = self._score_path_parametric(neg_path, self._params)
            total += max(0.0, self.margin - s_pos + s_neg)
        return total / max(len(training_pairs), 1)

    def _numerical_gradient(self, training_pairs: List[Tuple]) -> List[float]:
        """Symmetric finite-difference gradient of the loss w.r.t. each param."""
        grad = []
        for i in range(len(self._params)):
            original = self._params[i]

            self._params[i] = original + self.h
            loss_plus = self._compute_loss(training_pairs)

            self._params[i] = original - self.h
            loss_minus = self._compute_loss(training_pairs)

            self._params[i] = original
            grad.append((loss_plus - loss_minus) / (2.0 * self.h))
        return grad


# ---------------------------------------------------------------------------
# Phase 129: Platt Scaling — calibrated confidence output
# ---------------------------------------------------------------------------

class PlattCalibration:
    """Two-parameter sigmoid calibration: P = 1 / (1 + exp(A*score + B)).

    Fitted via gradient descent on accumulated (raw_score, correct:bool) pairs.
    Requires at least MIN_SAMPLES pairs before fit() has any effect.
    """
    MIN_SAMPLES = 20

    def __init__(self) -> None:
        self.A: float = 1.0   # slope initialised to identity scaling
        self.B: float = 0.0   # bias initialised to zero shift
        self._samples: List[Tuple[float, float]] = []   # (raw_score, label)
        self._fitted: bool = False

    def record(self, raw_score: float, correct: bool) -> None:
        """Accumulate one feedback sample."""
        self._samples.append((raw_score, 1.0 if correct else 0.0))

    def transform(self, raw_score: float) -> float:
        """Map a raw sigmoid score to a calibrated probability in (0, 1)."""
        if not self._fitted:
            return float(raw_score)
        val = max(-500.0, min(500.0, self.A * raw_score + self.B))
        return 1.0 / (1.0 + math.exp(val))

    def ece(self, bins: int = 10) -> float:
        """Expected Calibration Error across ``bins`` confidence bins."""
        if len(self._samples) < 2:
            return -1.0
        total = len(self._samples)
        bin_size = 1.0 / bins
        ece_sum = 0.0
        for b in range(bins):
            lo = b * bin_size
            hi = lo + bin_size
            bucket = [
                (self.transform(s), y)
                for s, y in self._samples
                if lo <= self.transform(s) < hi
            ]
            if not bucket:
                continue
            n_b = len(bucket)
            mean_conf = sum(p for p, _ in bucket) / n_b
            mean_acc  = sum(y for _, y in bucket) / n_b
            ece_sum += (n_b / total) * abs(mean_conf - mean_acc)
        return round(ece_sum, 6)

    def drift_score(self) -> float:
        """Absolute ECE change since last fit().  Returns 0.0 if not yet fitted."""
        if not self._fitted:
            return 0.0
        current_ece = self.ece()
        baseline = getattr(self, "_baseline_ece", 0.0)
        if current_ece < 0 or baseline < 0:
            return 0.0
        return round(abs(current_ece - baseline), 6)

    def fit(self, lr: float = 0.1, iterations: int = 200) -> bool:
        # Override to update _baseline_ece after a successful fit.
        # Delegate to parent implementation via direct code.
        if len(self._samples) < self.MIN_SAMPLES:
            return False
        A, B = self.A, self.B
        for _ in range(iterations):
            dA = dB = 0.0
            for s, y in self._samples:
                p = 1.0 / (1.0 + math.exp(max(-500.0, min(500.0, A * s + B))))
                err = p - y
                dA += err * s
                dB += err
            n = len(self._samples)
            A -= lr * dA / n
            B -= lr * dB / n
        self.A, self.B = A, B
        self._fitted = True
        self._baseline_ece = self.ece()
        self._last_fit_ts = time.monotonic()
        return True

    def to_dict(self) -> dict:
        return {
            "A": self.A, "B": self.B, "fitted": self._fitted,
            "n_samples": len(self._samples),
            "ece": self.ece() if self._fitted else -1.0,
            "drift_score": self.drift_score(),
            "last_fit_ts": getattr(self, "_last_fit_ts", None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibration":
        obj = cls()
        obj.A = float(d.get("A", 1.0))
        obj.B = float(d.get("B", 0.0))
        obj._fitted = bool(d.get("fitted", False))
        return obj

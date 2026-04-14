"""
AutoApprover — Automated Research Finding Decision Engine (Phase 71).

Makes approve/reject/review decisions on ResearchFinding objects autonomously,
replacing the manual POST /research/approve|reject workflow at scale.

Decision tiers (ordered):
  1. Hard gates      — deterministic safety rules (blocked statuses, validation)
  2. Logistic SGD    — online classifier trained from human decision history
  3. LLM fallback    — optional semantic evaluation for uncertain zone
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("cerebrum.auto_approver")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# Literature status → ordinal score (higher = more novel / actionable)
_STATUS_ORDINAL: Dict[str, float] = {
    "novel":           1.00,
    "active_research": 0.75,
    "unvalidated":     0.50,
    "contested":       0.25,
    "established":     0.00,
}

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class AutoApprovalPolicy:
    """All configurable thresholds and safety parameters for AutoApprover."""

    approve_threshold: float = 0.80
    """P(approve) ≥ this → auto-approve."""

    reject_threshold: float = 0.20
    """P(approve) ≤ this → auto-reject."""

    min_training_examples: int = 10
    """Cold-start guard: below this count all decisions are 'review'."""

    max_auto_per_scan: int = 10
    """Maximum automatic decisions emitted per scan cycle."""

    require_validation: bool = True
    """Will not auto-approve a finding that has no ValidationReport."""

    blocked_statuses: Set[str] = field(default_factory=lambda: {"established", "contested"})
    """Findings with these literature_status values are always hard-rejected."""

    learning_rate: float = 0.05
    """SGD step size for online weight updates."""

    audit_capacity: int = 500
    """Maximum number of AutoDecision records retained in the audit deque."""


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass
class AutoDecision:
    """Returned by AutoApprover.decide()."""

    action: str
    """One of: 'approve' | 'reject' | 'review'."""

    confidence: float
    """Classifier P(approve) in [0, 1].  0.5 before cold-start ends."""

    reason: str
    """Human-readable explanation for the decision."""

    features: List[float]
    """The 12-dimensional feature vector used by the classifier."""

    finding_id: str = ""


# ---------------------------------------------------------------------------
# AutoApprover
# ---------------------------------------------------------------------------

_N_FEATURES = 16


class AutoApprover:
    """
    Automated approve/reject/review decision engine for ResearchFindings.

    Parameters
    ----------
    policy
        ``AutoApprovalPolicy`` instance controlling all thresholds.
    llm_fn
        Optional callable ``(prompt: str) -> str``.  When provided and the
        classifier lands in the uncertain zone, the LLM is queried for a
        semantic YES/NO assessment.  Any LLM bridge adapter works here.
    """

    def __init__(
        self,
        policy: Optional[AutoApprovalPolicy] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.policy = policy or AutoApprovalPolicy()
        self._llm_fn = llm_fn

        # Weight vector + bias (logistic regression, online SGD)
        self._w: np.ndarray = np.zeros(_N_FEATURES, dtype=np.float64)
        self._bias: float = 0.0

        # Training counter
        self._n_trained: int = 0

        # Decision counters
        self._n_approve: int = 0
        self._n_reject: int = 0
        self._n_review: int = 0

        # Audit log
        self.audit_log: deque = deque(maxlen=self.policy.audit_capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, finding: Any) -> AutoDecision:
        """
        Evaluate a ResearchFinding and return an AutoDecision.

        Decision priority:
          1. Cold-start guard
          2. Hard gates (blocked status, require_validation)
          3. Logistic classifier
          4. LLM fallback (uncertain zone only)
        """
        features = self._extract_features(finding)
        finding_id = getattr(finding, "finding_id", "")

        # 1. Cold-start guard
        if self._n_trained < self.policy.min_training_examples:
            decision = AutoDecision(
                action="review",
                confidence=0.5,
                reason=f"cold_start: only {self._n_trained}/{self.policy.min_training_examples} training examples",
                features=features,
                finding_id=finding_id,
            )
            self._record(decision)
            return decision

        # 2. Hard gates
        lit_status = getattr(finding, "literature_status", "unvalidated")
        if lit_status in self.policy.blocked_statuses:
            decision = AutoDecision(
                action="reject",
                confidence=0.0,
                reason=f"hard_gate: literature_status '{lit_status}' is blocked",
                features=features,
                finding_id=finding_id,
            )
            self._record(decision)
            return decision

        if self.policy.require_validation and getattr(finding, "validation_report", None) is None:
            decision = AutoDecision(
                action="review",
                confidence=0.5,
                reason="hard_gate: require_validation=True but no ValidationReport",
                features=features,
                finding_id=finding_id,
            )
            self._record(decision)
            return decision

        # 3. Logistic classifier
        feat_arr = np.array(features, dtype=np.float64)
        raw = float(np.dot(self._w, feat_arr)) + self._bias
        p = _sigmoid(raw)

        if p >= self.policy.approve_threshold:
            action, reason = "approve", f"classifier: p={p:.3f} >= {self.policy.approve_threshold}"
        elif p <= self.policy.reject_threshold:
            action, reason = "reject", f"classifier: p={p:.3f} <= {self.policy.reject_threshold}"
        else:
            action, reason = "review", f"classifier: p={p:.3f} in uncertain zone"

        # 4. LLM fallback for uncertain zone
        if action == "review" and self._llm_fn is not None:
            action, reason = self._llm_fallback(finding, p, features)

        decision = AutoDecision(
            action=action,
            confidence=p,
            reason=reason,
            features=features,
            finding_id=finding_id,
        )
        self._record(decision)
        return decision

    def fit(self, finding: Any, approved: bool) -> None:
        """
        Online SGD update from a confirmed human (or auto-confirmed) decision.

        Parameters
        ----------
        finding
            The ResearchFinding that was decided.
        approved
            True if the finding was approved; False if rejected.
        """
        features = self._extract_features(finding)
        feat_arr = np.array(features, dtype=np.float64)

        raw = float(np.dot(self._w, feat_arr)) + self._bias
        p = _sigmoid(raw)
        target = 1.0 if approved else 0.0
        error = target - p

        lr = self.policy.learning_rate
        self._w += lr * error * feat_arr
        self._bias += lr * error
        self._n_trained += 1

    def stats(self) -> Dict[str, Any]:
        """Return a summary dict of learned state and decision counts."""
        return {
            "n_trained": self._n_trained,
            "n_approve": self._n_approve,
            "n_reject": self._n_reject,
            "n_review": self._n_review,
            "weights": self._w.tolist(),
            "bias": self._bias,
            "policy": {
                "approve_threshold": self.policy.approve_threshold,
                "reject_threshold": self.policy.reject_threshold,
                "min_training_examples": self.policy.min_training_examples,
                "max_auto_per_scan": self.policy.max_auto_per_scan,
                "require_validation": self.policy.require_validation,
                "blocked_statuses": list(self.policy.blocked_statuses),
                "learning_rate": self.policy.learning_rate,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Checkpoint: serialise weights and counters to a JSON-compatible dict."""
        return {
            "w": self._w.tolist(),
            "bias": self._bias,
            "n_trained": self._n_trained,
            "n_approve": self._n_approve,
            "n_reject": self._n_reject,
            "n_review": self._n_review,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], policy: Optional[AutoApprovalPolicy] = None) -> "AutoApprover":
        """Restore an AutoApprover from a checkpoint dict."""
        approver = cls(policy=policy)
        w = data.get("w", [])
        if len(w) == _N_FEATURES:
            approver._w = np.array(w, dtype=np.float64)
        approver._bias = float(data.get("bias", 0.0))
        approver._n_trained = int(data.get("n_trained", 0))
        approver._n_approve = int(data.get("n_approve", 0))
        approver._n_reject = int(data.get("n_reject", 0))
        approver._n_review = int(data.get("n_review", 0))
        return approver

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, finding: Any) -> List[float]:
        """
        Build the 16-dimensional feature vector from a ResearchFinding.

        Index  Feature
        -----  -------
        0      best_confidence                           [0,1]
        1      candidate.discovery_potential             [0,1]
        2      candidate.gap_score                       [0,1]
        3      candidate.community_distance / 5          [0,1]
        4      1 - candidate.local_density               [0,1]
        5      literature_status ordinal                 [0,1]
        6      validation_report.novelty_score           [0,1] (0.5 if absent)
        7      metadata["engram_affinity"]               [0,1] (0.0 if absent)
        8      best_proposal.path_count / 5              [0,1]
        9      best_proposal.contradiction_score         [0,1] (0.0 if absent)
        10     seeded_by == "structural_hole"            {0,1}
        11     seeded_by == "embedding_scan"             {0,1}
        --- TriangulationReport (Phase 72) ---
        12     triangulation.reverse_confidence          [0,1] (0.0 if absent)
        13     triangulation.strategy_agreement          [0,1] (0.5 if absent)
        14     triangulation.mean_path_independence      [0,1] (0.5 if absent)
        15     triangulation.semantic_type_score         [0,1] (0.5 if absent)
        """
        cand = getattr(finding, "candidate", None)

        # 0: best_confidence
        f0 = float(getattr(finding, "best_confidence", 0.0))

        # 1: discovery_potential
        f1 = float(getattr(cand, "discovery_potential", 0.0)) if cand else 0.0

        # 2: gap_score
        f2 = float(getattr(cand, "gap_score", 0.0)) if cand else 0.0

        # 3: community_distance (normalised)
        cd = float(getattr(cand, "community_distance", 0)) if cand else 0.0
        f3 = min(1.0, cd / 5.0)

        # 4: sparsity (1 - local_density)
        ld = float(getattr(cand, "local_density", 0.0)) if cand else 0.0
        f4 = 1.0 - min(1.0, ld)

        # 5: literature_status ordinal
        lit = getattr(finding, "literature_status", "unvalidated")
        f5 = _STATUS_ORDINAL.get(lit, 0.5)

        # 6: novelty_score from ValidationReport
        report = getattr(finding, "validation_report", None)
        f6 = float(getattr(report, "novelty_score", 0.5)) if report is not None else 0.5

        # 7: engram_affinity from metadata
        meta = getattr(finding, "metadata", {}) or {}
        f7 = float(meta.get("engram_affinity", 0.0))

        # 8: path_count from best proposal (normalised)
        proposals = getattr(finding, "proposals", []) or []
        best_prop = max(proposals, key=lambda p: getattr(p, "confidence", 0.0), default=None)
        pc = float(getattr(best_prop, "path_count", 0)) if best_prop else 0.0
        f8 = min(1.0, pc / 5.0)

        # 9: contradiction_score from best proposal
        f9 = float(getattr(best_prop, "contradiction_score", 0.0)) if best_prop else 0.0
        f9 = min(1.0, f9)

        # 10: structural_hole flag
        seeded_by = getattr(cand, "seeded_by", "") if cand else ""
        f10 = 1.0 if seeded_by == "structural_hole" else 0.0

        # 11: embedding_scan flag
        f11 = 1.0 if seeded_by == "embedding_scan" else 0.0

        # 12–15: TriangulationReport (Phase 72)
        # Neutral defaults when no TriangulationEngine is attached — backwards-compatible.
        tri = meta.get("triangulation", None)
        f12 = float(getattr(tri, "reverse_confidence",     0.0)) if tri is not None else 0.0
        f13 = float(getattr(tri, "strategy_agreement",     0.5)) if tri is not None else 0.5
        f14 = float(getattr(tri, "mean_path_independence", 0.5)) if tri is not None else 0.5
        f15 = float(getattr(tri, "semantic_type_score",    0.5)) if tri is not None else 0.5

        return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_fallback(self, finding: Any, p: float, features: List[float]) -> tuple:
        """Invoke LLM for semantic YES/NO evaluation. Returns (action, reason)."""
        cand = getattr(finding, "candidate", None)
        proposals = getattr(finding, "proposals", []) or []
        best_prop = max(proposals, key=lambda pr: getattr(pr, "confidence", 0.0), default=None)

        source_label = getattr(cand, "source_id", "?") if cand else "?"
        target_label = getattr(cand, "target_id", "?") if cand else "?"
        derived_rel = getattr(best_prop, "derived_relation", "?") if best_prop else "?"
        derivation = getattr(best_prop, "derivation_text", "") if best_prop else ""
        lit_status = getattr(finding, "literature_status", "unvalidated")
        report = getattr(finding, "validation_report", None)
        hit_count = getattr(report, "hit_count", 0) if report else 0
        confidence = getattr(finding, "best_confidence", 0.0)
        contradiction = getattr(best_prop, "contradiction_score", 0.0) if best_prop else 0.0

        prompt = (
            "You are reviewing a proposed knowledge graph edge for a research system.\n\n"
            f"Source: {source_label}\n"
            f"Target: {target_label}\n"
            f"Proposed relation: {derived_rel}\n"
            f"Reasoning: {derivation}\n"
            f"Literature status: {lit_status} ({hit_count} literature hits)\n"
            f"Path confidence: {confidence:.2f}\n"
            f"Contradiction score: {contradiction:.2f}\n\n"
            "Should this edge be added to the knowledge graph?\n"
            "Reply with exactly one word: YES or NO."
        )

        try:
            response = self._llm_fn(prompt)
            answer = str(response).strip().upper()
            if answer == "YES":
                return "approve", f"llm_fallback: YES (classifier p={p:.3f})"
            elif answer == "NO":
                return "reject", f"llm_fallback: NO (classifier p={p:.3f})"
            else:
                logger.warning("AutoApprover LLM returned unexpected: %r", response)
                return "review", f"llm_fallback: unexpected response '{answer}'"
        except Exception as exc:
            logger.error("AutoApprover LLM fallback failed: %s", exc)
            return "review", f"llm_fallback: error ({exc})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, decision: AutoDecision) -> None:
        """Append decision to audit log and increment counters."""
        self.audit_log.append(decision)
        if decision.action == "approve":
            self._n_approve += 1
        elif decision.action == "reject":
            self._n_reject += 1
        else:
            self._n_review += 1

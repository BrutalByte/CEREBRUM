"""
SelfAwarenessEngine — Phase 220: Explicit Epistemic Self-Assessment.

CEREBRUM v2.72 knows *where* its knowledge came from, *how confident* it is,
and *which signals* drove each answer.  This module synthesises those scattered
signals into a single, queryable SelfAwarenessReport returned alongside every
query result.

The report answers five questions the system can now answer about itself:

  1. How confident am I?      (answer_confidence, epistemic_uncertainty)
  2. How good is my evidence? (evidence_quality, corroboration)
  3. Do I contradict myself?  (contradiction_detected, contradiction_count)
  4. What drove my answer?    (dominant_signal, signal_breakdown)
  5. Do I actually know this? (knowledge_gap, gap_reason)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning.answer_extractor import Answer
    from reasoning.traversal import TraversalPath


# CSA feature index → human-readable signal name
_SIGNAL_NAMES: Tuple[str, ...] = (
    "semantic_similarity",
    "community_structure",
    "edge_type_weight",
    "path_length_penalty",
    "hop_decay",
    "pagerank_authority",
    "temporal_freshness",
    "node_recency",
    "synthesis_density",
    "source_credibility",
)

# Thresholds
_MIN_CONFIDENT_SCORE: float = 0.20   # below this = knowledge_gap
_MIN_CORROBORATION:   int   = 2      # below this = weakly corroborated
_HIGH_UNCERTAINTY:    float = 0.08   # beta-variance above this = uncertain


@dataclass
class SelfAwarenessReport:
    """
    A structured self-assessment synthesised from a single query result.

    Fields
    ------
    answer_confidence
        Score margin between top answer and runner-up, normalised to [0, 1].
        1.0 = decisive lead; 0.0 = top two answers tied.
    epistemic_uncertainty
        Beta-distribution variance of the winning path's score accumulation.
        High variance → the system explored many directions before committing.
    evidence_quality
        Mean source-credibility (grounding) across all edges in the best path.
        Reflects provenance quality, not just structural path quality.
    corroboration
        Number of independent first-intermediate-hop branches that converged
        on the top answer (branch_count).  ≥ 2 = corroborated; 1 = single path.
    contradiction_detected
        True if the AnswerExtractor flagged cross-path contradictions in the
        top-5 answers.
    contradiction_count
        Number of contradiction flags across the top answer set.
    knowledge_gap
        True if the top answer score is below the confident-answer threshold,
        meaning the system does not have enough information to answer reliably.
    gap_reason
        Human-readable explanation when knowledge_gap is True.
    signal_breakdown
        Mean contribution of each CSA feature dimension across all edges in
        the best path: {signal_name: mean_value}.
    dominant_signal
        The CSA feature that contributed most on average to the best path.
    causal_fraction
        Fraction of best-path edges whose relation type is registered as causal
        by the SymbolicValidator (via CausalDiscoveryEngine).
    path_length
        Number of hops in the best path.
    summary
        One-sentence plain-English self-assessment of the answer quality.
    """

    answer_confidence:    float = 0.0
    epistemic_uncertainty: float = 0.0
    evidence_quality:     float = 1.0
    corroboration:        int   = 1
    contradiction_detected: bool  = False
    contradiction_count:  int   = 0
    knowledge_gap:        bool  = False
    gap_reason:           str   = ""
    signal_breakdown:     Dict[str, float] = field(default_factory=dict)
    dominant_signal:      str   = ""
    causal_fraction:      float = 0.0
    path_length:          int   = 0
    summary:              str   = ""
    gap_recovery:         bool  = False
    """True if query() triggered a recovery retry due to knowledge_gap."""
    calibration_ece:      float = -1.0
    """Expected Calibration Error from PlattCalibration. −1 = not yet calibrated."""
    contradiction_resolved: bool = False
    """True if at least one contradiction was resolved (e.g. by credibility scoring)."""
    resolution_method:    str   = ""
    """How contradiction was resolved: 'credibility' or ''."""

    def to_dict(self) -> dict:
        return {
            "answer_confidence":     round(self.answer_confidence, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "evidence_quality":      round(self.evidence_quality, 4),
            "corroboration":         self.corroboration,
            "contradiction_detected": self.contradiction_detected,
            "contradiction_count":   self.contradiction_count,
            "contradiction_resolved": self.contradiction_resolved,
            "resolution_method":     self.resolution_method,
            "knowledge_gap":         self.knowledge_gap,
            "gap_reason":            self.gap_reason,
            "gap_recovery":          self.gap_recovery,
            "calibration_ece":       round(self.calibration_ece, 4) if self.calibration_ece >= 0 else -1,
            "dominant_signal":       self.dominant_signal,
            "signal_breakdown":      {k: round(v, 4) for k, v in self.signal_breakdown.items()},
            "causal_fraction":       round(self.causal_fraction, 4),
            "path_length":           self.path_length,
            "summary":               self.summary,
        }


class SelfAwarenessEngine:
    """
    Synthesises a SelfAwarenessReport from query results.

    Parameters
    ----------
    causal_relations
        Set of relation types registered as causal (from SymbolicValidator or
        CausalDiscoveryEngine).  Used to compute causal_fraction.
    min_confident_score
        Top answer scores below this are flagged as knowledge_gap.
    """

    def __init__(
        self,
        causal_relations: Optional[set] = None,
        min_confident_score: float = _MIN_CONFIDENT_SCORE,
    ) -> None:
        self.causal_relations: set = causal_relations or set()
        self.min_confident_score = min_confident_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        answers: "List[Answer]",
        all_paths: Optional["List[TraversalPath]"] = None,
    ) -> SelfAwarenessReport:
        """
        Build a SelfAwarenessReport from a completed query's answers.

        Parameters
        ----------
        answers   : List[Answer] returned by CerebrumGraph.query()
        all_paths : Optional list of all TraversalPath objects explored.
                    When provided, signal_breakdown and epistemic_uncertainty
                    are computed from the full beam rather than just the
                    best path's stored data.
        """
        if not answers:
            return SelfAwarenessReport(
                knowledge_gap=True,
                gap_reason="no answers returned",
                summary="I have no answer for this query.",
            )

        top = answers[0]
        best_path: Optional["TraversalPath"] = getattr(top, "best_path", None)

        # 1. Answer confidence — score margin
        confidence = self._compute_confidence(answers)

        # 2. Epistemic uncertainty — Beta variance of best path
        uncertainty = self._compute_uncertainty(best_path)

        # 3. Evidence quality — mean grounding across best path edges
        evidence_quality = self._compute_evidence_quality(best_path)

        # 4. Corroboration
        corroboration = getattr(top, "branch_count", 1)

        # 5. Contradiction
        contra_flags = getattr(top, "contradiction_flags", []) or []
        for ans in answers[1:5]:
            contra_flags = contra_flags + (getattr(ans, "contradiction_flags", []) or [])
        contradiction_count = len(contra_flags)

        # 6. Knowledge gap
        knowledge_gap, gap_reason = self._detect_gap(top, answers)

        # 7. Signal breakdown + dominant signal
        signal_breakdown, dominant_signal = self._compute_signals(best_path)

        # 8. Causal fraction
        causal_fraction = self._compute_causal_fraction(best_path)

        # 9. Path length
        path_length = getattr(best_path, "hop_depth", 0) if best_path else 0

        # 10. Human-readable summary
        summary = self._build_summary(
            top, confidence, uncertainty, evidence_quality,
            corroboration, contradiction_count, knowledge_gap,
            gap_reason, dominant_signal, causal_fraction,
        )

        # Check if any contradictions were resolved by credibility scoring
        _contra_resolved = False
        _resolution_method = ""
        for ans in answers[:5]:
            for flag in getattr(ans, "contradiction_flags", []) or []:
                if getattr(flag, "resolution_status", "") == "resolved_by_credibility":
                    _contra_resolved = True
                    _resolution_method = "credibility"
                    break
            if _contra_resolved:
                break

        return SelfAwarenessReport(
            answer_confidence=confidence,
            epistemic_uncertainty=uncertainty,
            evidence_quality=evidence_quality,
            corroboration=corroboration,
            contradiction_detected=contradiction_count > 0,
            contradiction_count=contradiction_count,
            knowledge_gap=knowledge_gap,
            gap_reason=gap_reason,
            signal_breakdown=signal_breakdown,
            dominant_signal=dominant_signal,
            causal_fraction=causal_fraction,
            path_length=path_length,
            summary=summary,
            contradiction_resolved=_contra_resolved,
            resolution_method=_resolution_method,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self, answers: "List[Answer]") -> float:
        top_score = answers[0].score if answers else 0.0
        second_score = answers[1].score if len(answers) > 1 else 0.0
        if top_score <= 0:
            return 0.0
        margin = (top_score - second_score) / (top_score + 1e-9)
        return round(min(1.0, max(0.0, margin)), 4)

    def _compute_uncertainty(self, path: Optional["TraversalPath"]) -> float:
        if path is None:
            return 1.0
        alpha = getattr(path, "beta_alpha", 1.0)
        beta  = getattr(path, "beta_beta",  1.0)
        total = alpha + beta + 1e-9
        variance = (alpha * beta) / (total * total * (total + 1.0))
        return round(float(variance), 6)

    def _compute_evidence_quality(self, path: Optional["TraversalPath"]) -> float:
        if path is None:
            return 0.5
        feats = getattr(path, "edge_features", None)
        if not feats:
            # Fall back to path_confidence
            return round(float(getattr(path, "path_confidence", 1.0)), 4)
        # grounding is index 8 in the 9-tuple (sim, cs, etw, nd, hd, pr, td, nr, grounding)
        groundings = [f[8] for f in feats if len(f) > 8]
        if not groundings:
            return 1.0
        return round(float(sum(groundings) / len(groundings)), 4)

    def _detect_gap(
        self, top: "Answer", answers: "List[Answer]"
    ) -> Tuple[bool, str]:
        score = getattr(top, "score", 0.0)
        if score < self.min_confident_score:
            return True, f"top answer score {score:.3f} < threshold {self.min_confident_score:.3f}"
        consensus = getattr(top, "consensus_score", score)
        if consensus < 0.05:
            return True, "no beam paths converged on this answer"
        return False, ""

    def _compute_signals(
        self, path: Optional["TraversalPath"]
    ) -> Tuple[Dict[str, float], str]:
        if path is None:
            return {}, ""
        feats = getattr(path, "edge_features", None)
        if not feats:
            return {}, ""

        n = len(_SIGNAL_NAMES)
        totals = [0.0] * n
        count = 0
        for f in feats:
            for i in range(min(n, len(f))):
                totals[i] += float(f[i])
            count += 1

        if count == 0:
            return {}, ""

        breakdown = {_SIGNAL_NAMES[i]: round(totals[i] / count, 4) for i in range(n)}
        # synthesis_density is a *penalty* — negate for "contribution" ranking
        ranked = sorted(
            breakdown.items(),
            key=lambda kv: -kv[1] if kv[0] != "synthesis_density" else kv[1],
        )
        dominant = ranked[0][0] if ranked else ""
        return breakdown, dominant

    def _compute_causal_fraction(self, path: Optional["TraversalPath"]) -> float:
        if path is None or not self.causal_relations:
            return 0.0
        nodes = getattr(path, "nodes", [])
        # nodes is alternating [entity, relation, entity, ...]
        relations = [nodes[i] for i in range(1, len(nodes), 2)]
        if not relations:
            return 0.0
        causal = sum(1 for r in relations if r in self.causal_relations)
        return round(causal / len(relations), 4)

    def _build_summary(
        self,
        top: "Answer",
        confidence: float,
        uncertainty: float,
        evidence_quality: float,
        corroboration: int,
        contradiction_count: int,
        knowledge_gap: bool,
        gap_reason: str,
        dominant_signal: str,
        causal_fraction: float,
    ) -> str:
        if knowledge_gap:
            return f"Knowledge gap: {gap_reason}."

        # Confidence descriptor
        if confidence > 0.5:
            conf_desc = "high confidence"
        elif confidence > 0.2:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence"

        # Evidence descriptor
        if evidence_quality > 0.85:
            ev_desc = "authoritative sources"
        elif evidence_quality > 0.60:
            ev_desc = "mixed-quality sources"
        else:
            ev_desc = "low-credibility sources"

        # Corroboration descriptor
        corr_desc = f"{corroboration} independent path{'s' if corroboration != 1 else ''}"

        # Signal descriptor
        sig_map = {
            "community_structure":   "structural proximity",
            "semantic_similarity":   "semantic similarity",
            "source_credibility":    "source credibility",
            "temporal_freshness":    "temporal freshness",
            "pagerank_authority":    "hub authority",
            "edge_type_weight":      "relation specificity",
            "hop_decay":             "path depth",
        }
        sig_desc = sig_map.get(dominant_signal, dominant_signal.replace("_", " "))

        contra_note = f" ({contradiction_count} contradiction{'s' if contradiction_count != 1 else ''} detected)" if contradiction_count else ""
        causal_note = f"; {int(causal_fraction * 100)}% of path edges are causal" if causal_fraction > 0 else ""

        return (
            f"{conf_desc.capitalize()} answer driven by {sig_desc}, "
            f"supported by {corr_desc} via {ev_desc}{causal_note}{contra_note}."
        )

    @classmethod
    def from_symbolic_validator(cls, validator, **kwargs) -> "SelfAwarenessEngine":
        """Convenience: build engine pre-loaded with causal relations from SymbolicValidator."""
        from core.symbolic_engine import ConstraintType
        causal = {
            c.params.get("relation", "")
            for c in getattr(validator, "constraints", [])
            if c.constraint_type == ConstraintType.CAUSAL_ORDERING
        }
        return cls(causal_relations=causal, **kwargs)

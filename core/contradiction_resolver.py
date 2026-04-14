"""
ContradictionResolver — Evidence-Weight Classifier for Contested Findings (Phase 73 Batch B).

When HypothesisEngine returns proposals that carry a non-trivial contradiction_score,
a simple confidence-threshold reject would discard findings that might represent genuine
revision opportunities: the proposed evidence could be *stronger* than whatever opposing
signal the engine detected.

This module classifies contested findings deterministically from already-computed fields —
no additional traversal, no extra API calls.

Resolution taxonomy
-------------------
"clean"               contradiction_score < min_contradiction_score.
                      Fast path; no further work.
"revision_candidate"  net_evidence_score > revision_threshold.
                      Proposed Noisy-OR outweighs opposing evidence.
                      Existing graph knowledge may be stale or wrong.
"contested"           |net_evidence_score| ≤ threshold.
                      Roughly equal evidence on both sides; surface for human review.
"discardable"         net_evidence_score < discard_threshold.
                      Opposing evidence dominates; reject before AutoApprover runs.

Design invariant
----------------
Absence of structural precedent is NOT evidence of impossibility.  A proposed edge
whose relation type has never appeared between these community-distance nodes might
simply be novel.  This resolver never penalises novelty — it only acts when explicit
contradiction_score evidence is present and exceeds min_contradiction_score.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger("cerebrum.contradiction_resolver")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContradictionRecord:
    """
    Evidence-weight classification record for a contested finding.

    Stored in ``finding.metadata["contradiction_resolution"]``.
    Accessible by AutoApprover as a hard-gate check.
    """

    finding_id: str
    """ID of the ResearchFinding this record belongs to."""

    contradiction_score: float
    """Max contradiction_score across all evaluated proposals [0, 1]."""

    proposed_noisy_or: float
    """
    Noisy-OR aggregation of all proposal confidences:
    ``1 - prod(1 - c_i)`` for c_i in proposal.confidence.
    Represents the combined probability that at least one proposed path is correct.
    """

    net_evidence_score: float
    """
    ``proposed_noisy_or - contradiction_score``.
    Positive  = proposed evidence is stronger than opposing signal.
    Negative  = opposing signal dominates.
    """

    resolution: str
    """One of: "clean" | "revision_candidate" | "contested" | "discardable"."""

    revision_weight: float
    """
    ``proposed_noisy_or / max(contradiction_score, 0.01)``.
    How many times stronger is the proposed evidence vs. the opposing signal.
    Only meaningful when resolution != "clean".
    """


# ---------------------------------------------------------------------------
# ContradictionResolver
# ---------------------------------------------------------------------------

class ContradictionResolver:
    """
    Classifies a ResearchFinding as clean / revision_candidate / contested /
    discardable based on the ratio of proposed vs. opposing evidence.

    Parameters
    ----------
    revision_threshold
        net_evidence_score above this value → "revision_candidate".
        Default 0.15 — proposed evidence must be meaningfully stronger.
    discard_threshold
        net_evidence_score below this value → "discardable".
        Default -0.15 — opposing evidence must be meaningfully stronger.
    min_contradiction_score
        Skip analysis (return "clean") when the maximum contradiction_score
        across all proposals is below this floor.
        Default 0.10 — ignore noise-level contradiction signals.
    """

    def __init__(
        self,
        revision_threshold: float = 0.15,
        discard_threshold: float = -0.15,
        min_contradiction_score: float = 0.10,
    ) -> None:
        if revision_threshold <= 0:
            raise ValueError("revision_threshold must be positive")
        if discard_threshold >= 0:
            raise ValueError("discard_threshold must be negative")
        if min_contradiction_score < 0:
            raise ValueError("min_contradiction_score must be >= 0")

        self.revision_threshold      = revision_threshold
        self.discard_threshold       = discard_threshold
        self.min_contradiction_score = min_contradiction_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        finding: Any,
        proposals: List[Any],
    ) -> ContradictionRecord:
        """
        Classify a finding based on the evidence-weight of its proposals.

        Parameters
        ----------
        finding
            The ``ResearchFinding``.  Only ``finding_id`` is read; the
            caller already has the filtered proposal list.
        proposals
            List of ``HypothesisProposal`` objects (confidence ≥ min_confidence).
            Each proposal is expected to carry:
            - ``confidence``       float [0, 1]
            - ``contradiction_score`` float [0, 1]  (0.0 if absent)

        Returns
        -------
        ContradictionRecord
        """
        finding_id = getattr(finding, "finding_id", "")

        if not proposals:
            return ContradictionRecord(
                finding_id=finding_id,
                contradiction_score=0.0,
                proposed_noisy_or=0.0,
                net_evidence_score=0.0,
                resolution="clean",
                revision_weight=1.0,
            )

        # Gather per-proposal fields
        confidences     = [float(getattr(p, "confidence", 0.0))           for p in proposals]
        contra_scores   = [float(getattr(p, "contradiction_score", 0.0))  for p in proposals]

        max_contra      = max(contra_scores, default=0.0)
        proposed_nor    = self._noisy_or(confidences)

        # --- Fast path: no meaningful contradiction ---
        if max_contra < self.min_contradiction_score:
            logger.debug(
                "Resolver %s: clean (max_contra=%.3f < threshold=%.3f)",
                finding_id, max_contra, self.min_contradiction_score,
            )
            return ContradictionRecord(
                finding_id=finding_id,
                contradiction_score=max_contra,
                proposed_noisy_or=proposed_nor,
                net_evidence_score=proposed_nor - max_contra,
                resolution="clean",
                revision_weight=proposed_nor / max(max_contra, 0.01),
            )

        net             = proposed_nor - max_contra
        rev_weight      = proposed_nor / max(max_contra, 0.01)

        if net > self.revision_threshold:
            resolution = "revision_candidate"
        elif net < self.discard_threshold:
            resolution = "discardable"
        else:
            resolution = "contested"

        logger.debug(
            "Resolver %s: %s (nor=%.3f contra=%.3f net=%.3f rev_weight=%.3f)",
            finding_id, resolution, proposed_nor, max_contra, net, rev_weight,
        )

        return ContradictionRecord(
            finding_id=finding_id,
            contradiction_score=max_contra,
            proposed_noisy_or=proposed_nor,
            net_evidence_score=round(net, 4),
            resolution=resolution,
            revision_weight=round(rev_weight, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _noisy_or(confidences: List[float]) -> float:
        """
        Noisy-OR aggregation: 1 - prod(1 - c_i).

        Represents the probability that at least one of the independent
        hypotheses is correct, assuming independent failure modes.
        Clamps each confidence to [0, 1] before computation.
        """
        if not confidences:
            return 0.0
        product = 1.0
        for c in confidences:
            product *= 1.0 - max(0.0, min(1.0, c))
        return round(1.0 - product, 4)

"""
Phase 37 — Calibration Layer (Self-Doubt & Entropy).

Implements a calibration layer that monitors the "certainty" of the reasoning
engine. If the attention weights for a hop are too similar (high entropy), the
system flags a "Self-Doubt" signal, suggesting that the answer may be a guess.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np


@dataclass
class CalibrationResult:
    """The result of an uncertainty analysis on a reasoning hop."""
    
    entropy: float
    is_uncertain: bool
    confidence_multiplier: float
    top_candidates: List[str]
    message: str = ""


class CalibrationEngine:
    """
    Analyzes attention distributions to detect ambiguity.
    """

    def __init__(self, entropy_threshold: float = 0.8, min_candidates: int = 3):
        """
        Parameters
        ----------
        entropy_threshold : float
            Threshold (0.0 to 1.0) above which a distribution is considered 'uncertain'.
            1.0 means a perfectly uniform distribution (maximum doubt).
        min_candidates : int
            Minimum number of candidates to consider for entropy calculation.
        """
        self.entropy_threshold = entropy_threshold
        self.min_candidates = min_candidates

    def calibrate_hop(self, weights: List[float], candidate_ids: List[str]) -> CalibrationResult:
        """
        Calculate the normalized entropy of the attention distribution.
        """
        if len(weights) < 2:
            return CalibrationResult(0.0, False, 1.0, candidate_ids)

        # Normalize weights to a probability distribution
        w_sum = sum(weights)
        if w_sum <= 0:
            return CalibrationResult(1.0, True, 0.1, candidate_ids, "Zero weight sum")

        probs = [w / w_sum for w in weights]
        
        # Calculate Shannon entropy: H = -sum(p * log2(p))
        # Normalized by log2(N) to stay in range [0, 1]
        entropy = -sum(p * math.log2(max(p, 1e-9)) for p in probs)
        max_entropy = math.log2(len(probs))
        
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        is_uncertain = norm_entropy > self.entropy_threshold
        
        # Confidence multiplier: 1.0 if certain, decays as entropy increases
        # multiplier = 1.0 - (norm_entropy^2)
        multiplier = float(np.clip(1.0 - (norm_entropy ** 2), 0.1, 1.0))
        
        # Get top candidates for clarification
        sorted_indices = np.argsort(weights)[::-1]
        top_ids = [candidate_ids[i] for i in sorted_indices[:3]]
        
        msg = ""
        if is_uncertain:
            msg = f"High ambiguity detected (entropy {norm_entropy:.2f}). System is uncertain between: {', '.join(top_ids)}"

        return CalibrationResult(
            entropy=norm_entropy,
            is_uncertain=is_uncertain,
            confidence_multiplier=multiplier,
            top_candidates=top_ids,
            message=msg
        )

"""
MetacognitiveMonitor — Phase 121: Epistemic Uncertainty Integration.

Composes 9 existing epistemic signals into a unified EpistemicState per
reasoning call. Enables the system to self-report "I don't know" with a
calibrated uncertainty estimate rather than only emitting point-estimate scores.

Signal sources:
  - PredictiveCoder   → prediction_error, soliton_index
  - CerebellarEngine  → dissonance (path_score - consensus_score gap)
  - CalibrationEngine → hop_entropy (Shannon entropy of CSA weights)
  - path_scorer       → path_confidence (weakest-link edge confidence)
  - WorkingMemory     → historical_pe_mean, calibration_drift

Composite formula:
  epistemic_uncertainty    = 0.40 * PE + 0.25 * hop_entropy + 0.20 * dissonance
                           + 0.15 * (1 - path_confidence)
  confidence_in_uncertainty = 0.40 * soliton_index + 0.35 * (1 - dissonance)
                            + 0.25 * path_confidence
"""
from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.predictive_coder import PredictiveCoder
    from core.cerebellar_engine import CerebellarEngine
    from core.calibration_engine import CalibrationEngine
    from core.chemical_modulator import ChemicalModulator
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger("cerebrum.metacognition")

# Weights for epistemic_uncertainty composite
_W_PE = 0.40
_W_ENTROPY = 0.25
_W_DISSONANCE = 0.20
_W_GROUNDING = 0.15  # applied to (1 - path_confidence)

# Weights for confidence_in_uncertainty composite
_W_SOLITON = 0.40
_W_CONSENSUS = 0.35
_W_PATH_CONF = 0.25

_DISSONANCE_THRESHOLD = 0.35
_ENTROPY_THRESHOLD = 0.80
_GROUNDED_THRESHOLD = 0.70


@dataclass
class EpistemicState:
    """Holistic epistemic uncertainty summary for one reasoning event."""

    # --- Raw signals [0, 1] ---
    prediction_error: float = 0.5       # PE from PredictiveCoder; 0=perfect, 1=miss
    soliton_index: float = 0.0          # Prior stability; higher = more stable
    hop_entropy: float = 0.5            # Mean CSA weight entropy; higher = more ambiguous
    path_confidence: float = 1.0        # Weakest-link edge confidence
    dissonance: float = 0.0             # path_score - consensus_score gap
    consensus_score: float = 1.0        # Multi-strategy agreement

    # --- Composite scores [0, 1] ---
    epistemic_uncertainty: float = 0.5  # Higher = less confident
    confidence_in_uncertainty: float = 0.5  # Credence in above estimate

    # --- Decision flags ---
    is_dissonant: bool = False          # dissonance >= 0.35
    is_ambiguous: bool = False          # hop_entropy >= 0.80
    is_grounded: bool = True            # path_confidence >= 0.70

    # --- Historical calibration ---
    historical_pe_mean: Optional[float] = None
    calibration_drift: Optional[float] = None  # |current_pe - historical_mean|

    def to_dict(self) -> dict:
        return {
            "prediction_error": round(self.prediction_error, 4),
            "soliton_index": round(self.soliton_index, 4),
            "hop_entropy": round(self.hop_entropy, 4),
            "path_confidence": round(self.path_confidence, 4),
            "dissonance": round(self.dissonance, 4),
            "consensus_score": round(self.consensus_score, 4),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 4),
            "confidence_in_uncertainty": round(self.confidence_in_uncertainty, 4),
            "is_dissonant": self.is_dissonant,
            "is_ambiguous": self.is_ambiguous,
            "is_grounded": self.is_grounded,
            "historical_pe_mean": round(self.historical_pe_mean, 4) if self.historical_pe_mean is not None else None,
            "calibration_drift": round(self.calibration_drift, 4) if self.calibration_drift is not None else None,
        }


class MetacognitiveMonitor:
    """
    Integrates distributed epistemic signals into a unified EpistemicState.

    All engine parameters are optional — the monitor degrades gracefully when
    sub-engines are unavailable, substituting neutral default values.

    Parameters
    ----------
    predictive_coder    : PredictiveCoder (Phase 69/111) — PE + soliton signals
    cerebellar_engine   : CerebellarEngine (Phase 59) — dissonance events
    calibration_engine  : CalibrationEngine (Phase 37) — hop entropy
    chemical_modulator  : ChemicalModulator (Phase 68) — metabolic context (unused
                          in scoring formula but available on EpistemicState)
    working_memory      : WorkingMemoryBuffer (Phase 95) — historical PE calibration
    history_window      : number of past assessments to track for calibration history
    """

    def __init__(
        self,
        predictive_coder: Optional["PredictiveCoder"] = None,
        cerebellar_engine: Optional["CerebellarEngine"] = None,
        calibration_engine: Optional["CalibrationEngine"] = None,
        chemical_modulator: Optional["ChemicalModulator"] = None,
        working_memory: Optional["WorkingMemoryBuffer"] = None,
        history_window: int = 100,
    ) -> None:
        self.pc = predictive_coder
        self.cb = cerebellar_engine
        self.ce = calibration_engine
        self.cm = chemical_modulator
        self.wm = working_memory
        # Rolling history of (predicted_epistemic_uncertainty, was_correct)
        self._history: Deque[Tuple[float, bool]] = deque(maxlen=history_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        paths: List[Any],
        answers: List[Any],
        trace: Optional[Any] = None,
        seed_ids: Optional[List[str]] = None,
    ) -> EpistemicState:
        """
        Synthesize all available epistemic signals into a unified EpistemicState.

        Parameters
        ----------
        paths       : list of TraversalPath objects from BeamTraversal
        answers     : list of Answer objects from AnswerExtractor
        trace       : Optional ReasoningTrace (carries PE, soliton_index)
        seed_ids    : query seeds (used for working memory lookup)
        """
        pe = self._get_pe(trace)
        soliton = self._get_soliton(trace, seed_ids)
        hop_entropy = self._compute_hop_entropy(paths)
        path_conf = self._compute_path_confidence(answers)
        dissonance, consensus = self._compute_dissonance_consensus(answers, seed_ids)
        hist_pe, cal_drift = self._compute_calibration_drift(pe)

        eu = self._epistemic_uncertainty(pe, hop_entropy, dissonance, path_conf)
        ciu = self._confidence_in_uncertainty(soliton, dissonance, path_conf)

        state = EpistemicState(
            prediction_error=pe,
            soliton_index=soliton,
            hop_entropy=hop_entropy,
            path_confidence=path_conf,
            dissonance=dissonance,
            consensus_score=consensus,
            epistemic_uncertainty=eu,
            confidence_in_uncertainty=ciu,
            is_dissonant=dissonance >= _DISSONANCE_THRESHOLD,
            is_ambiguous=hop_entropy >= _ENTROPY_THRESHOLD,
            is_grounded=path_conf >= _GROUNDED_THRESHOLD,
            historical_pe_mean=hist_pe,
            calibration_drift=cal_drift,
        )
        logger.debug(
            "EpistemicState: EU=%.3f CIU=%.3f PE=%.3f soliton=%.3f dissonance=%.3f",
            eu, ciu, pe, soliton, dissonance,
        )
        return state

    def record_outcome(self, epistemic_state: EpistemicState, was_correct: bool) -> None:
        """Update calibration history after ground-truth is known."""
        self._history.append((epistemic_state.epistemic_uncertainty, was_correct))

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def _get_pe(self, trace: Optional[Any]) -> float:
        if trace is not None:
            pe = getattr(trace, "prediction_error", None)
            if pe is not None:
                return float(pe)
        # Fallback: check PredictiveCoder directly (not available without prior)
        return 0.5  # neutral default when no engram is active

    def _get_soliton(self, trace: Optional[Any], seed_ids: Optional[List[str]]) -> float:
        if trace is not None:
            sol = getattr(trace, "soliton_index", None)
            if sol is not None:
                return float(sol)
        if self.pc is not None and seed_ids:
            for seed in seed_ids:
                val = self.pc.soliton_stability_map.get(seed)
                if val is not None:
                    return float(val)
        return 0.0

    def _compute_hop_entropy(self, paths: List[Any]) -> float:
        """Average Shannon entropy of attention weights across all path hops."""
        if self.ce is None or not paths:
            return 0.5
        entropies = []
        for path in paths:
            weights = getattr(path, "attention_weights", [])
            if len(weights) < 2:
                continue
            try:
                result = self.ce.calibrate_hop(weights, [str(i) for i in range(len(weights))])
                entropies.append(result.entropy)
            except Exception:
                pass
        return float(sum(entropies) / len(entropies)) if entropies else 0.5

    def _compute_path_confidence(self, answers: List[Any]) -> float:
        """Weakest-link path confidence from best answer."""
        if not answers:
            return 1.0
        best = answers[0]
        conf = getattr(best, "path_confidence", None)
        if conf is not None:
            return float(conf)
        # Fallback: use score as proxy
        return float(getattr(best, "score", 1.0))

    def _compute_dissonance_consensus(
        self, answers: List[Any], seed_ids: Optional[List[str]]
    ) -> Tuple[float, float]:
        """Return (max_dissonance, mean_consensus_score) from answers."""
        if not answers:
            return 0.0, 1.0

        dissonances = []
        consensus_scores = []
        for ans in answers:
            path_score = getattr(ans, "path_score", getattr(ans, "score", 0.0))
            cons = getattr(ans, "consensus_score", path_score)
            gap = max(0.0, float(path_score) - float(cons))
            dissonances.append(gap)
            consensus_scores.append(float(cons))

        return (
            max(dissonances) if dissonances else 0.0,
            sum(consensus_scores) / len(consensus_scores) if consensus_scores else 1.0,
        )

    def _compute_calibration_drift(
        self, current_pe: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute historical PE mean and drift from WorkingMemory."""
        if self.wm is None:
            return None, None
        try:
            entries = self.wm.recent(50)
            pe_vals = [
                e.prediction_error
                for e in entries
                if e.prediction_error is not None
            ]
            if not pe_vals:
                return None, None
            hist_mean = sum(pe_vals) / len(pe_vals)
            drift = abs(current_pe - hist_mean)
            return hist_mean, drift
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Composite formulas
    # ------------------------------------------------------------------

    @staticmethod
    def _epistemic_uncertainty(
        pe: float, hop_entropy: float, dissonance: float, path_conf: float
    ) -> float:
        eu = (
            _W_PE * pe
            + _W_ENTROPY * hop_entropy
            + _W_DISSONANCE * dissonance
            + _W_GROUNDING * (1.0 - path_conf)
        )
        return max(0.0, min(1.0, eu))

    @staticmethod
    def _confidence_in_uncertainty(
        soliton: float, dissonance: float, path_conf: float
    ) -> float:
        # Low dissonance = high consensus = high credence in estimates
        ciu = (
            _W_SOLITON * soliton
            + _W_CONSENSUS * (1.0 - dissonance)
            + _W_PATH_CONF * path_conf
        )
        return max(0.0, min(1.0, ciu))

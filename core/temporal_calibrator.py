"""
Temporal Bias Calibrator (Phase 55).

Tunes the ``eta`` (temporal decay, td) and ``iota`` (node recency, nr_v)
parameters in the CSA formula to maximise Recall@N on a labelled validation
set.  CEREBRUM's verifiable paths serve as the ground truth — the calibrator
uses gradient-free grid search (fast, no training data needed) over a small
parameter grid.

Aligns temporal scoring with empirical Recall@K targets derived from
labelled validation sets or external benchmark results.

Usage
-----
    from core.temporal_calibrator import TemporalCalibrator

    calibrator = TemporalCalibrator(graph, recall_at=5)
    calibrator.add_example(query_nodes=["newton"], correct_answer="gravity")
    ...

    result = calibrator.calibrate(eta_range=(0.0, 1.0), iota_range=(0.0, 1.0))
    print(result.best_eta, result.best_iota, result.recall)
    calibrator.apply(graph)   # writes best params into graph._csa
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("cerebrum.temporal_calibrator")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationExample:
    """A single validation example for temporal calibration."""
    query_nodes: List[str]
    correct_answers: List[str]  # one or more acceptable entity IDs
    top_k: int = 5
    max_hop: int = 3
    beam_width: int = 10


@dataclass
class CalibrationResult:
    """Output of TemporalCalibrator.calibrate()."""
    best_eta: float
    best_iota: float
    recall: float            # Recall@N on the validation set
    n_examples: int
    n_correct: int
    grid_scores: List[Tuple[float, float, float]] = field(default_factory=list)
    """List of (eta, iota, recall) for every grid point evaluated."""


# ---------------------------------------------------------------------------
# TemporalCalibrator
# ---------------------------------------------------------------------------

class TemporalCalibrator:
    """
    Grid-search calibrator for the temporal logit parameters eta and iota.

    The calibrator runs CEREBRUM beam-search on each validation example,
    computes Recall@top_k (whether any correct answer appears in the top-K
    results), and returns the (eta, iota) pair that maximises recall.

    Parameters
    ----------
    graph       : CerebrumGraph — must be built before calling calibrate()
    recall_at   : K for Recall@K metric (default 5)
    grid_steps  : number of grid points per axis (default 5 → 5×5 = 25 evals)
    """

    def __init__(
        self,
        graph,
        recall_at: int = 5,
        grid_steps: int = 5,
    ) -> None:
        self.graph = graph
        self.recall_at = recall_at
        self.grid_steps = grid_steps
        self._examples: List[CalibrationExample] = []

    # ------------------------------------------------------------------
    # Example management
    # ------------------------------------------------------------------

    def add_example(
        self,
        query_nodes: List[str],
        correct_answers: List[str],
        top_k: int = 5,
        max_hop: int = 3,
        beam_width: int = 10,
    ) -> "TemporalCalibrator":
        """
        Register a labelled validation example.

        Parameters
        ----------
        query_nodes     : seed entity IDs for the query
        correct_answers : acceptable answer entity IDs (any match = correct)
        top_k           : beam search top_k for this example
        """
        self._examples.append(CalibrationExample(
            query_nodes=query_nodes,
            correct_answers=correct_answers,
            top_k=top_k,
            max_hop=max_hop,
            beam_width=beam_width,
        ))
        return self

    def add_examples(self, examples: List[CalibrationExample]) -> "TemporalCalibrator":
        self._examples.extend(examples)
        return self

    def clear_examples(self) -> None:
        self._examples.clear()

    def n_examples(self) -> int:
        return len(self._examples)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        eta_range:  Tuple[float, float] = (0.0, 1.0),
        iota_range: Tuple[float, float] = (0.0, 1.0),
    ) -> CalibrationResult:
        """
        Run grid-search over (eta, iota) and return the best configuration.

        Parameters
        ----------
        eta_range  : (min, max) for the eta (temporal decay) grid
        iota_range : (min, max) for the iota (node recency) grid

        Returns
        -------
        CalibrationResult with best_eta, best_iota, and achieved recall.
        """
        if not self._examples:
            raise ValueError(
                "No calibration examples registered. Call add_example() first."
            )
        if not self.graph._built:
            raise RuntimeError(
                "Graph has not been built. Call graph.build() first."
            )

        eta_vals  = np.linspace(eta_range[0],  eta_range[1],  self.grid_steps)
        iota_vals = np.linspace(iota_range[0], iota_range[1], self.grid_steps)

        best_eta   = float(eta_vals[0])
        best_iota  = float(iota_vals[0])
        best_recall = -1.0
        grid_scores: List[Tuple[float, float, float]] = []

        for eta, iota in itertools.product(eta_vals, iota_vals):
            recall = self._eval_recall(float(eta), float(iota))
            grid_scores.append((float(eta), float(iota), recall))
            logger.debug("eta=%.3f iota=%.3f → recall@%d=%.4f",
                         eta, iota, self.recall_at, recall)
            if recall > best_recall:
                best_recall = recall
                best_eta    = float(eta)
                best_iota   = float(iota)

        n_correct = int(round(best_recall * len(self._examples)))
        logger.info(
            "Calibration complete: best eta=%.3f iota=%.3f recall@%d=%.4f (%d/%d)",
            best_eta, best_iota, self.recall_at, best_recall,
            n_correct, len(self._examples),
        )
        return CalibrationResult(
            best_eta=best_eta,
            best_iota=best_iota,
            recall=best_recall,
            n_examples=len(self._examples),
            n_correct=n_correct,
            grid_scores=grid_scores,
        )

    def _eval_recall(self, eta: float, iota: float) -> float:
        """Evaluate Recall@K for the given (eta, iota) on all examples."""
        # Temporarily set CSA parameters
        csa = self.graph._csa
        orig_eta  = csa.eta
        orig_iota = csa.iota
        csa.eta   = eta
        csa.iota  = iota

        try:
            hits = 0
            for ex in self._examples:
                answers = self.graph.query(
                    ex.query_nodes,
                    top_k=self.recall_at,
                    max_hop=ex.max_hop,
                    beam_width=ex.beam_width,
                )
                predicted_ids = {a.entity_id for a in answers}
                correct_set   = set(ex.correct_answers)
                if predicted_ids & correct_set:
                    hits += 1
            return hits / len(self._examples)
        finally:
            # Always restore original values
            csa.eta  = orig_eta
            csa.iota = orig_iota

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, result: Optional[CalibrationResult] = None) -> None:
        """
        Write the best (eta, iota) values into the graph's CSA engine.

        Parameters
        ----------
        result : CalibrationResult from calibrate().  If None, calibrate()
                 is called first with default ranges.
        """
        if result is None:
            result = self.calibrate()
        self.graph._csa.eta  = result.best_eta
        self.graph._csa.iota = result.best_iota
        logger.info(
            "Applied calibrated temporal params: eta=%.3f iota=%.3f (recall@%d=%.4f)",
            result.best_eta, result.best_iota, self.recall_at, result.recall,
        )

    # ------------------------------------------------------------------
    # Convenience: recall measurement without changing CSA params
    # ------------------------------------------------------------------

    def measure_recall(self, eta: Optional[float] = None, iota: Optional[float] = None) -> float:
        """
        Measure current Recall@K on the registered examples.

        Uses the graph's current CSA eta/iota unless overrides are provided.
        """
        csa = self.graph._csa
        _eta  = eta  if eta  is not None else csa.eta
        _iota = iota if iota is not None else csa.iota
        return self._eval_recall(_eta, _iota)

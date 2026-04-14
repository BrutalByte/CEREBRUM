"""
PredictiveCodingEngine — Active Inference for CEREBRUM (Phase 69).

Before traversal, generates a *prior path* by reading top patterns from the
Engram cache and propagating them forward from the seed entities. After
traversal, computes a Prediction Error (PE) — the Jaccard divergence between
the prior's relation sequence and the best actual result. PE drives all
downstream regulatory components:

  - High PE → ChemicalModulator arousal/novelty spike (widen beam, explore more)
  - Low PE  → ChemicalModulator reinforcement signal (reward stability)

The ``soliton_index`` tracks the running coherence of predictions for a given
seed set over recent calls. A prior that consistently yields low PE behaves as
a soliton — a self-reinforcing, self-localising wave that maintains its
structure through propagation (Coherence Field Theory, UCFT 2025). High
soliton_index means the system has converged on a stable internal model of
this reasoning domain.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cerebrum.predictive_coder")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PriorPath:
    """A predicted path generated before traversal from Engram patterns."""
    seeds: List[str]
    predicted_relations: List[str]   # Most likely relation sequence (from Engram)
    predicted_nodes: List[str]       # Best-effort entity trace through those relations
    confidence: float                # Normalized Engram affinity [0, 1]
    soliton_index: float             # Running prediction stability for this seed set [0, 1]


@dataclass
class PredictionResult:
    """Outcome of comparing a prior against actual traversal results."""
    prior: Optional[PriorPath]
    prediction_error: float          # Jaccard divergence [0, 1]; 0 = perfect prediction
    soliton_stability: float         # 1 - mean(recent PEs) for this seed set [0, 1]
    reinforcement_signal: float      # 1 - PE; directly usable by ChemicalModulator


# ---------------------------------------------------------------------------
# PredictiveCodingEngine
# ---------------------------------------------------------------------------

class PredictiveCodingEngine:
    """
    Generates priors from the Engram cache before traversal and computes
    Prediction Error (PE) after traversal.

    The PE is a Jaccard distance on relation sequences — how much did the
    actual traversal deviate from the expected reasoning pattern?

    Parameters
    ----------
    engram      : Engram (or SpeedTalkEngram) instance.
    adapter     : GraphAdapter — used to trace predicted nodes through relations.
    history_len : Number of recent PE values averaged for soliton_index (default 20).
    """

    def __init__(self, engram: Any, adapter: Any, history_len: int = 20) -> None:
        self.engram       = engram
        self.adapter      = adapter
        self._history_len = history_len
        # seed_key → deque of recent prediction errors
        self._pe_history: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, seeds: List[str]) -> Optional[PriorPath]:
        """
        Generate a prior path for the given seeds using the Engram cache.

        Returns None on cold start (empty Engram) — callers must handle this
        gracefully; ``update()`` returns PE=0.0 when prior is None.
        """
        if self.engram.size() == 0:
            return None

        top = self.engram.top_patterns(n=5)
        if not top:
            return None

        # Highest-count pattern becomes the prior relation sequence
        best_seq, best_count = top[0]
        total_count = sum(cnt for _, cnt in top)
        confidence = best_count / max(total_count, 1)

        predicted_nodes = self._trace_nodes(seeds, list(best_seq))

        key = self._seed_key(seeds)
        sol = self._soliton_index(key)

        logger.debug(
            "Prior generated: rels=%s conf=%.3f soliton=%.3f",
            list(best_seq), confidence, sol,
        )

        return PriorPath(
            seeds=list(seeds),
            predicted_relations=list(best_seq),
            predicted_nodes=predicted_nodes,
            confidence=confidence,
            soliton_index=sol,
        )

    def update(
        self,
        prior: Optional[PriorPath],
        actual_paths: List[Any],  # List[TraversalPath]
    ) -> PredictionResult:
        """
        Compare prior against actual traversal results, update PE history, and
        return a PredictionResult containing all downstream signals.

        Safe to call with ``prior=None`` (cold start) — returns PE=0.0.
        """
        pe = self.compute_pe(prior, actual_paths)

        if prior is not None:
            key     = self._seed_key(prior.seeds)
            history = self._pe_history.setdefault(key, deque(maxlen=self._history_len))
            history.append(pe)
            sol = self._soliton_index(key)
        else:
            sol = 0.0

        logger.debug("PE=%.3f  soliton_stability=%.3f", pe, sol)

        return PredictionResult(
            prior=prior,
            prediction_error=round(pe, 4),
            soliton_stability=round(sol, 4),
            reinforcement_signal=round(1.0 - pe, 4),
        )

    def compute_pe(
        self,
        prior: Optional[PriorPath],
        actual_paths: List[Any],
    ) -> float:
        """
        Jaccard distance between the prior's relation set and the best actual
        path's relation set.

        - Returns 0.0 when prior is None (cold start — no penalty).
        - Returns 1.0 when actual_paths is empty (traversal found nothing).
        """
        if prior is None:
            return 0.0
        if not actual_paths:
            return 1.0

        prior_rels = set(prior.predicted_relations)
        if not prior_rels:
            return 0.0

        # Best-scoring actual path
        best_path  = max(actual_paths, key=lambda p: float(p.score))
        actual_rels = set(
            best_path.nodes[i]
            for i in range(1, len(best_path.nodes), 2)
        )

        if not actual_rels and not prior_rels:
            return 0.0

        intersection = len(prior_rels & actual_rels)
        union        = len(prior_rels | actual_rels)
        return round(1.0 - (intersection / union if union > 0 else 0.0), 4)

    def soliton_stats(self) -> Dict[str, float]:
        """
        Return per-seed-key soliton stability scores for observability.
        """
        return {
            key: round(self._soliton_index(key), 4)
            for key in self._pe_history
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_key(self, seeds: List[str]) -> str:
        return "|".join(sorted(seeds))

    def _soliton_index(self, key: str) -> float:
        """
        Coherence metric: 1 - mean(recent PEs) for this seed set.

        Inspired by soliton field theory (UCFT 2025): a self-reinforcing wave
        that maintains its shape through propagation has minimal dispersion.
        High soliton_index = the system's predictions are self-consistent and
        structurally stable across repeated calls for this reasoning domain.
        """
        history = self._pe_history.get(key)
        if not history:
            return 0.0
        return max(0.0, 1.0 - (sum(history) / len(history)))

    def _trace_nodes(self, seeds: List[str], relations: List[str]) -> List[str]:
        """
        Best-effort forward trace through the adapter for ``len(relations)`` hops
        starting from seeds. Used to populate PriorPath.predicted_nodes.

        Does not require strict relation matching — this is exploratory tracing,
        not ground-truth path selection.
        """
        current = list(seeds)
        trace   = list(seeds)

        for _ in relations:
            next_nodes: List[str] = []
            for node in current:
                try:
                    neighbors  = self.adapter.get_neighbors(node, max_neighbors=5)
                    next_nodes.extend(e.target_id for e in neighbors[:3])
                except Exception:
                    pass

            if next_nodes:
                current = next_nodes[:5]
                trace.extend(current)
            else:
                break  # Cannot follow further — partial trace is fine

        return trace

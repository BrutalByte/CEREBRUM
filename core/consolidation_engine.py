"""
ConsolidationEngine — Phase 96: Memory Consolidation (Hebbian Replay).

During idle loop cycles, replays high-quality WorkingMemory entries and
applies Hebbian weight boosts to the edges on those paths. Edges the system
repeatedly traverses to produce good answers grow permanently stronger —
the KG analogue of hippocampal replay / sleep consolidation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger("cerebrum.consolidation")


@dataclass
class ConsolidationResult:
    """Outcome summary for a single consolidation pass."""
    entries_replayed: int
    edges_strengthened: int
    mean_delta: float
    duration: float


class ConsolidationEngine:
    """Hebbian replay engine.

    Takes the top-k highest-scoring WorkingMemory entries that carry
    ``path_edges`` and applies a quality-proportional weight boost to each
    traversed edge via ``adapter.update_edge_weight()``.

    Parameters
    ----------
    adapter       : graph adapter (must implement update_edge_weight)
    graph         : CerebrumGraph — used for telemetry emit()
    min_score     : minimum top_score an entry must have to be replayed
    max_weight    : upper bound on any edge weight after boosting
    hebbian_delta : base weight increment per replay; scaled by entry.top_score
    """

    def __init__(
        self,
        adapter: Any,
        graph: Any,
        min_score: float = 0.6,
        max_weight: float = 2.0,
        hebbian_delta: float = 0.05,
        pe_weight: float = 0.3,
    ) -> None:
        self.adapter = adapter
        self.graph = graph
        self.min_score = min_score
        self.max_weight = max_weight
        self.hebbian_delta = hebbian_delta
        self.pe_weight = pe_weight  # Gap 2: PE salience weight

    def consolidate(
        self,
        wm: "WorkingMemoryBuffer",
        k: int = 5,
        reinforcement_scale: float = 1.0,
    ) -> ConsolidationResult:
        """Replay up to k WM entries and Hebbian-boost their traversed edges.

        Only entries whose top_score >= min_score AND that carry non-empty
        path_edges are eligible. Entries are ranked by salience = top_score +
        pe_weight * prediction_error (Gap 2). Delta is further scaled by
        reinforcement_scale from ChemicalModulator (Gap 1).
        """
        t0 = time.time()

        candidates = [
            e for e in wm.recent(50)
            if e.top_score >= self.min_score and e.path_edges
        ]
        # Gap 2: rank by salience = score + PE contribution
        candidates.sort(
            key=lambda e: e.top_score + self.pe_weight * (e.prediction_error or 0.0),
            reverse=True,
        )
        candidates = candidates[:k]

        total_strengthened = 0
        total_delta_sum = 0.0

        for entry in candidates:
            # Gap 1: scale delta by dopamine reinforcement signal
            delta = self.hebbian_delta * entry.top_score * reinforcement_scale
            for (u, rel, v) in entry.path_edges:
                try:
                    n = self.adapter.update_edge_weight(
                        u, v, rel, delta=delta, max_weight=self.max_weight
                    )
                    total_strengthened += n
                    total_delta_sum += delta * n
                except Exception as exc:
                    logger.debug("update_edge_weight failed for (%s,%s,%s): %s", u, rel, v, exc)

        mean_delta = (total_delta_sum / total_strengthened) if total_strengthened else 0.0
        result = ConsolidationResult(
            entries_replayed=len(candidates),
            edges_strengthened=total_strengthened,
            mean_delta=mean_delta,
            duration=time.time() - t0,
        )

        self._emit(result)
        logger.debug(
            "Consolidation: replayed=%d strengthened=%d mean_delta=%.4f",
            result.entries_replayed, result.edges_strengthened, result.mean_delta,
        )
        return result

    def _emit(self, result: ConsolidationResult) -> None:
        try:
            from core.telemetry import NeuralEvent
            self.graph.emit(NeuralEvent.consolidation_pulse(
                entries_replayed=result.entries_replayed,
                edges_strengthened=result.edges_strengthened,
                mean_delta=result.mean_delta,
            ))
        except Exception as exc:
            logger.debug("CONSOLIDATION_PULSE emit failed: %s", exc)

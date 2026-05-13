"""
ConsolidationEngine — Phase 96 & Phase 172.

Phase 96: Memory Consolidation (Hebbian Replay).
During idle loop cycles, replays high-quality WorkingMemory entries and
applies Hebbian weight boosts to the edges on those paths.

Phase 172: REM Cycle Daemon.
Analyzes reasoning traces from QueryLog and materializes high-utility shortcut paths.
"""
from __future__ import annotations

import logging
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.working_memory import WorkingMemoryBuffer
    from core.persistence import QueryLog
    from core.graph_adapter import GraphAdapter

logger = logging.getLogger("cerebrum.consolidation")


@dataclass
class ConsolidationResult:
    """Outcome summary for a single consolidation pass."""
    entries_replayed: int
    edges_strengthened: int
    mean_delta: float
    duration: float


class ConsolidationEngine:
    """
    Handles both Hebbian Replay (Phase 96) and Shortcut Synthesis (Phase 172).

    Parameters
    ----------
    adapter       : graph adapter
    graph         : CerebrumGraph — used for telemetry emit()
    query_log     : QueryLog instance for Phase 172
    threshold     : frequency threshold for shortcut materialization (Phase 172)
    min_score     : minimum top_score an entry must have to be replayed (Phase 96)
    max_weight    : upper bound on any edge weight after boosting (Phase 96)
    hebbian_delta : base weight increment per replay (Phase 96)
    """

    def __init__(
        self,
        adapter: Any,
        graph: Any,
        query_log: Optional[Any] = None,
        threshold: int = 5,
        min_score: float = 0.6,
        max_weight: float = 2.0,
        hebbian_delta: float = 0.05,
        pe_weight: float = 0.3,
    ) -> None:
        self.adapter = adapter
        self.graph = graph
        self.query_log = query_log
        self.threshold = threshold
        self.min_score = min_score
        self.max_weight = max_weight
        self.hebbian_delta = hebbian_delta
        self.pe_weight = pe_weight

    def consolidate(
        self,
        wm: "WorkingMemoryBuffer",
        k: int = 5,
        reinforcement_scale: float = 1.0,
    ) -> ConsolidationResult:
        """Phase 96: Replay up to k WM entries and Hebbian-boost their traversed edges."""
        t0 = time.time()

        candidates = [
            e for e in wm.recent(50)
            if e.top_score >= self.min_score and e.path_edges
        ]
        # Rank by salience = score + PE contribution
        candidates.sort(
            key=lambda e: e.top_score + self.pe_weight * (e.prediction_error or 0.0),
            reverse=True,
        )
        candidates = candidates[:k]

        total_strengthened = 0
        total_delta_sum = 0.0

        for entry in candidates:
            # Scale delta by dopamine reinforcement signal
            delta = self.hebbian_delta * entry.top_score * reinforcement_scale
            for (u, rel, v) in entry.path_edges:
                try:
                    # adapter must implement update_edge_weight
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

    async def run_rem_cycle(self):
        """Phase 172: Analyzes logs and synthesizes shortcut edges."""
        if not self.query_log:
            logger.warning("REM Cycle: No QueryLog provided, skipping.")
            return

        logger.info("REM Cycle: Starting consolidation...")
        traces = self.query_log.get_recent_traces(limit=1000)
        
        # Identify high-frequency paths (multi-hop patterns)
        path_counts: Dict[tuple, int] = {}
        for trace in traces:
            if len(trace.path) > 2:
                # Store as tuple for dictionary key
                path_tuple = tuple(trace.path)
                path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1
        
        # Materialize paths meeting threshold
        materialized_count = 0
        for path, count in path_counts.items():
            if count >= self.threshold:
                self._materialize_path(path)
                materialized_count += 1
                
        logger.info(f"REM Cycle: Consolidation complete. Materialized {materialized_count} shortcut(s).")

    def _materialize_path(self, path: tuple):
        """Synthesize a shortcut edge A->Z."""
        src, tgt = path[0], path[-1]
        logger.info(f"REM Cycle: Materializing shortcut {src} -> {tgt}")
        self.adapter.add_edge(
            u=src,
            v=tgt,
            relation="REM_SHORTCUT",
            confidence=0.9,
            provenance="REM_Consolidation",
            synthetic=True
        )
        # Emit SYNAPTOGENESIS event for telemetry
        try:
            from core.telemetry import NeuralEvent
            self.graph.emit(NeuralEvent.synaptogenesis(
                u=src, v=tgt, relation="REM_SHORTCUT", confidence=0.9
            ))
        except Exception as exc:
            logger.debug("SYNAPTOGENESIS emit failed: %s", exc)

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

    def track_path_success(self, path: List[str], outcome_score: float):
        """Track path success to bias future Engram predictive priors."""
        # Update success metrics for Engrams
        if self.query_log:
            self.query_log.log_success(path, outcome_score)
        logger.debug(f"ConsolidationEngine: Tracked success for path of len {len(path)}")

"""
SleepCycleOrchestrator — Phase 119: Offline Consolidation / Sleep Cycle.

Coordinates all idle-state self-organization engines into a single atomic
offline pass, analogous to biological sleep-dependent memory consolidation.

Five sequential sub-phases:
  1. Engram Consolidation  — promote high-frequency Engram patterns to canonical
  2. WM Replay (Hebbian)   — strengthen edges from successful WorkingMemory entries
  3. REM Synthesis         — materialize shortcut edges from QueryLog traces
  4. Synaptic Decay        — homeostatic weight normalization (LTD)
  5. DMN Scan              — structural self-audit; push insights to GoalStack
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning.engram_consolidation import EngramConsolidator
    from core.consolidation_engine import ConsolidationEngine
    from core.synaptic_decay_engine import SynapticDecayEngine
    from core.default_mode_engine import DefaultModeEngine, DMNInsight
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger("cerebrum.sleep_cycle")


@dataclass
class SleepReport:
    """Summary of one complete sleep cycle."""
    started_at: float
    duration_seconds: float = 0.0
    engrams_promoted: int = 0
    wm_entries_replayed: int = 0
    edges_strengthened: int = 0
    rem_shortcuts_added: int = 0
    edges_decayed: int = 0
    dmn_insights: int = 0
    dry_run: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "engrams_promoted": self.engrams_promoted,
            "wm_entries_replayed": self.wm_entries_replayed,
            "edges_strengthened": self.edges_strengthened,
            "rem_shortcuts_added": self.rem_shortcuts_added,
            "edges_decayed": self.edges_decayed,
            "dmn_insights": self.dmn_insights,
            "dry_run": self.dry_run,
            "error": self.error,
        }


class SleepCycleOrchestrator:
    """
    Coordinates all offline self-organization engines into a single sleep pass.

    Parameters
    ----------
    adapter               : graph adapter
    engram_consolidator   : EngramConsolidator (Phase 64)
    consolidation_engine  : ConsolidationEngine (Phase 96 + 112)
    synaptic_decay_engine : SynapticDecayEngine (Phase 97)
    default_mode_engine   : DefaultModeEngine (Phase 102)
    working_memory        : WorkingMemoryBuffer (optional, passed to sub-engines)
    goal_stack            : GoalStack (optional, receives DMN insights)
    calibrator            : DiscoveryCalibrator (optional, used by DMN frontier audit)
    """

    def __init__(
        self,
        adapter: Any,
        engram_consolidator: Optional["EngramConsolidator"] = None,
        consolidation_engine: Optional["ConsolidationEngine"] = None,
        synaptic_decay_engine: Optional["SynapticDecayEngine"] = None,
        default_mode_engine: Optional["DefaultModeEngine"] = None,
        working_memory: Optional["WorkingMemoryBuffer"] = None,
        goal_stack: Optional[Any] = None,
        calibrator: Optional[Any] = None,
    ) -> None:
        self.adapter = adapter
        self.engram_consolidator = engram_consolidator
        self.consolidation_engine = consolidation_engine
        self.synaptic_decay_engine = synaptic_decay_engine
        self.default_mode_engine = default_mode_engine
        self.working_memory = working_memory
        self.goal_stack = goal_stack
        self.calibrator = calibrator

        self._lock = threading.Lock()
        self._last_report: Optional[SleepReport] = None
        self._sleeping = False
        self._timer: Optional[threading.Timer] = None
        self._last_activity_at: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_activity(self) -> None:
        """Call on every query/event to reset the idle timer."""
        self._last_activity_at = time.time()

    def run(self, dry_run: bool = False) -> SleepReport:
        """Execute all 5 sleep sub-phases sequentially. Thread-safe.

        Parameters
        ----------
        dry_run : if True, skip mutations (decay/shortcuts); still audits.
        """
        with self._lock:
            if self._sleeping:
                logger.info("SleepCycle: already running, skipping.")
                return self._last_report or SleepReport(
                    started_at=time.time(), duration_seconds=0.0, dry_run=dry_run
                )
            self._sleeping = True

        t0 = time.time()
        report = SleepReport(started_at=t0, dry_run=dry_run)
        logger.info("SleepCycle: starting%s", " (dry_run)" if dry_run else "")

        try:
            # Phase 1 — Engram Consolidation
            report.engrams_promoted = self._run_engram_consolidation(dry_run)

            # Phase 2 — Working Memory Hebbian Replay
            wm_result = self._run_wm_replay(dry_run)
            report.wm_entries_replayed = wm_result[0]
            report.edges_strengthened = wm_result[1]

            # Phase 3 — REM Synthesis (async shim via sync wrapper)
            report.rem_shortcuts_added = self._run_rem_synthesis(dry_run)

            # Phase 4 — Synaptic Decay
            report.edges_decayed = self._run_synaptic_decay(dry_run)

            # Phase 5 — DMN Scan
            report.dmn_insights = self._run_dmn_scan()

        except Exception as exc:
            logger.exception("SleepCycle: error during sleep pass: %s", exc)
            report.error = str(exc)
        finally:
            report.duration_seconds = time.time() - t0
            self._last_report = report
            self._sleeping = False

        logger.info(
            "SleepCycle: complete in %.2fs — engrams=%d wm=%d edges_str=%d "
            "shortcuts=%d decayed=%d dmn=%d",
            report.duration_seconds, report.engrams_promoted,
            report.wm_entries_replayed, report.edges_strengthened,
            report.rem_shortcuts_added, report.edges_decayed, report.dmn_insights,
        )
        return report

    def schedule(self, idle_threshold_seconds: float = 300.0) -> None:
        """Start background idle watcher. Triggers run() after idle_threshold_seconds."""
        self._idle_threshold = idle_threshold_seconds
        self._start_idle_watch()

    def cancel(self) -> None:
        """Cancel any pending scheduled sleep."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @property
    def last_report(self) -> Optional[SleepReport]:
        return self._last_report

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    # ------------------------------------------------------------------
    # Sub-phase implementations
    # ------------------------------------------------------------------

    def _run_engram_consolidation(self, dry_run: bool) -> int:
        if self.engram_consolidator is None:
            return 0
        try:
            if dry_run:
                # Count what would be promoted without mutating
                count = sum(
                    1 for seq, cnt in self.engram_consolidator.cache._counts.items()
                    if cnt >= self.engram_consolidator.min_success_threshold
                    and seq not in self.engram_consolidator.canonical_patterns
                )
                return count
            return self.engram_consolidator.consolidate()
        except Exception as exc:
            logger.warning("SleepCycle[engram]: %s", exc)
            return 0

    def _run_wm_replay(self, dry_run: bool):
        if self.consolidation_engine is None or self.working_memory is None:
            return 0, 0
        try:
            if dry_run:
                candidates = [
                    e for e in self.working_memory.recent(50)
                    if e.top_score >= self.consolidation_engine.min_score and e.path_edges
                ]
                return min(len(candidates), 5), 0
            result = self.consolidation_engine.consolidate(self.working_memory)
            return result.entries_replayed, result.edges_strengthened
        except Exception as exc:
            logger.warning("SleepCycle[wm_replay]: %s", exc)
            return 0, 0

    def _run_rem_synthesis(self, dry_run: bool) -> int:
        if self.consolidation_engine is None:
            return 0
        try:
            import asyncio
            if dry_run:
                return 0
            # Run the async REM cycle in a new event loop (we're in a thread)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.consolidation_engine.run_rem_cycle())
            finally:
                loop.close()
            # Return count is not exposed by run_rem_cycle; approximate from log
            return 0
        except Exception as exc:
            logger.warning("SleepCycle[rem]: %s", exc)
            return 0

    def _run_synaptic_decay(self, dry_run: bool) -> int:
        if self.synaptic_decay_engine is None:
            return 0
        try:
            if dry_run:
                return 0
            result = self.synaptic_decay_engine.decay(self.working_memory)
            return result.edges_decayed
        except Exception as exc:
            logger.warning("SleepCycle[decay]: %s", exc)
            return 0

    def _run_dmn_scan(self) -> int:
        if self.default_mode_engine is None:
            return 0
        try:
            insights = self.default_mode_engine.idle_scan(
                wm=self.working_memory,
                goal_stack=self.goal_stack,
                calibrator=self.calibrator,
            )
            return len(insights)
        except Exception as exc:
            logger.warning("SleepCycle[dmn]: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Idle watcher
    # ------------------------------------------------------------------

    def _start_idle_watch(self) -> None:
        self.cancel()
        delay = getattr(self, "_idle_threshold", 300.0)
        self._timer = threading.Timer(delay, self._on_idle_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_idle_timer(self) -> None:
        idle_secs = time.time() - self._last_activity_at
        threshold = getattr(self, "_idle_threshold", 300.0)
        if idle_secs >= threshold:
            logger.info("SleepCycle: idle %.0fs >= threshold %.0fs — triggering sleep.", idle_secs, threshold)
            self.run()
        # Reschedule
        self._start_idle_watch()

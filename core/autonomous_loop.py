"""
AutonomousDiscoveryLoop — Phase 74.

Closes the full discover → validate → approve → materialize loop by running
ResearchAgent.scan_once() on a configurable timer, processing each finding
through the attached AutoApprover, and tracking cycle health via a circuit
breaker that pauses autonomous materialization when approval rates collapse.

Design decisions
----------------
- Uses threading.Event + threading.Thread rather than asyncio so it can run
  alongside FastAPI without requiring a running event loop at startup.
- Circuit breaker is a sliding window over the last N decisions; if the
  approval fraction drops below min_approval_rate the loop pauses and alerts
  callers via status(). It auto-resets when the window clears.
- dry_run=True runs every cycle but skips approve() / reject() calls so the
  loop can be safely trialled against a live graph without materialising edges.
- approver_checkpoint_path: if set, AutoApprover weights are persisted after
  every cycle that changed the decision distribution.

Usage
-----
    from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig
    from core.research_agent import ResearchAgent

    loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=300))
    loop.start()
    ...
    loop.stop()
    print(loop.status())
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    """Tunable parameters for AutonomousDiscoveryLoop."""

    cycle_interval: float = 300.0
    """Seconds between scan cycles. Default 5 minutes."""

    max_materializations_per_cycle: int = 5
    """Hard cap on approve() calls per cycle (regardless of AutoApprover score)."""

    min_approval_rate: float = 0.10
    """Circuit-breaker threshold. If the rolling window approval fraction falls
    below this, the loop pauses materialization and sets tripped=True."""

    circuit_breaker_window: int = 20
    """Number of recent AutoApprover decisions tracked for the approval rate."""

    dry_run: bool = False
    """If True, run cycles but never call approve() or reject() on the agent."""

    approver_checkpoint_path: Optional[str] = None
    """If set, AutoApprover.to_dict() is persisted here after each cycle
    that changes the decision distribution. Pass None to disable."""


# ---------------------------------------------------------------------------
# Cycle record
# ---------------------------------------------------------------------------

@dataclass
class CycleRecord:
    """Outcome summary for a single autonomous scan cycle."""

    cycle_number: int
    started_at: float
    duration_seconds: float
    findings_seen: int
    """Total findings returned by scan_once() this cycle."""

    auto_approved: int
    """Findings the AutoApprover (or loop) approved + materialized."""

    auto_rejected: int
    """Findings the AutoApprover rejected (not materialized)."""

    sent_to_review: int
    """Findings below confidence threshold, queued for human review."""

    edges_added: int
    """Total graph edges materialized this cycle."""

    circuit_breaker_tripped: bool = False
    """True if the circuit breaker paused materialization this cycle."""

    dry_run: bool = False


# ---------------------------------------------------------------------------
# AutonomousDiscoveryLoop
# ---------------------------------------------------------------------------

class AutonomousDiscoveryLoop:
    """
    Autonomous discovery loop that periodically calls ResearchAgent.scan_once()
    and processes each finding through the attached AutoApprover.

    Parameters
    ----------
    agent:
        A configured ResearchAgent instance.  Must have an AutoApprover
        attached (``agent._auto_approver``) — if not, findings are queued for
        human review rather than auto-decided.
    config:
        LoopConfig controlling timing, circuit breaker, and materialization cap.
    """

    def __init__(self, agent, config: Optional[LoopConfig] = None) -> None:
        self._agent = agent
        self._config = config or LoopConfig()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._cycle_number = 0
        self._total_approved = 0
        self._total_rejected = 0
        self._total_review = 0
        self._total_edges = 0
        self._total_cycles = 0
        self._started_at: Optional[float] = None
        self._last_cycle_at: Optional[float] = None

        # Sliding window for circuit breaker: True=approved, False=rejected/review
        self._decision_window: Deque[bool] = deque(maxlen=self._config.circuit_breaker_window)
        self._circuit_tripped = False

        self._recent_cycles: Deque[CycleRecord] = deque(maxlen=50)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop thread (idempotent)."""
        with self._lock:
            if self._running:
                logger.debug("AutonomousDiscoveryLoop: already running.")
                return
            self._running = True
            self._started_at = time.time()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop_body,
                name="autonomous-discovery-loop",
                daemon=True,
            )
            self._thread.start()
        logger.info("AutonomousDiscoveryLoop: started (interval=%.0fs, dry_run=%s).",
                    self._config.cycle_interval, self._config.dry_run)

    def stop(self) -> None:
        """Signal the loop to stop and wait for the current cycle to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(self._config.cycle_interval + 5, 30))
        logger.info("AutonomousDiscoveryLoop: stopped.")

    def run_cycle(self) -> CycleRecord:
        """
        Execute a single discovery cycle synchronously.

        Calls scan_once(), processes findings through AutoApprover (if attached),
        enforces circuit breaker + per-cycle materialization cap.
        Returns a CycleRecord summarizing the cycle outcome.
        """
        with self._lock:
            self._cycle_number += 1
            cycle_num = self._cycle_number

        t0 = time.time()
        cfg = self._config
        aa = getattr(self._agent, "_auto_approver", None)

        findings = self._agent.scan_once()

        auto_approved = 0
        auto_rejected = 0
        sent_to_review = 0
        edges_added = 0
        circuit_tripped = False

        # Re-check circuit breaker state before deciding anything this cycle
        self._update_circuit_breaker()

        for finding in findings:
            if auto_approved >= cfg.max_materializations_per_cycle:
                # Cap reached — remaining findings go to review queue
                sent_to_review += 1
                continue

            if aa is None:
                # No AutoApprover — findings accumulate in agent's ring buffer
                sent_to_review += 1
                continue

            # --- AutoApprover decision ---
            from core.auto_approver import AutoDecision  # local import to avoid circular
            decision: AutoDecision = aa.decide(finding)

            if decision.action == "approve":
                self._decision_window.append(True)
                self._update_circuit_breaker()

                if self._circuit_tripped:
                    circuit_tripped = True
                    sent_to_review += 1
                    logger.warning(
                        "AutonomousDiscoveryLoop cycle %d: circuit breaker tripped, "
                        "skipping materialization of %s.",
                        cycle_num, finding.finding_id,
                    )
                    continue

                if not cfg.dry_run:
                    try:
                        n = self._agent.approve(finding.finding_id,
                                                cycle_number=cycle_num)
                        edges_added += n
                        aa.fit(finding, approved=True)
                    except ValueError:
                        # Finding may have been consumed by a concurrent caller
                        pass
                auto_approved += 1

            elif decision.action == "reject":
                self._decision_window.append(False)
                self._update_circuit_breaker()
                if not cfg.dry_run:
                    try:
                        self._agent.reject(finding.finding_id)
                        aa.fit(finding, approved=False)
                    except ValueError:
                        pass
                auto_rejected += 1

            else:  # "review"
                sent_to_review += 1

        duration = time.time() - t0

        record = CycleRecord(
            cycle_number=cycle_num,
            started_at=t0,
            duration_seconds=duration,
            findings_seen=len(findings),
            auto_approved=auto_approved,
            auto_rejected=auto_rejected,
            sent_to_review=sent_to_review,
            edges_added=edges_added,
            circuit_breaker_tripped=circuit_tripped,
            dry_run=cfg.dry_run,
        )

        with self._lock:
            self._total_cycles += 1
            self._total_approved += auto_approved
            self._total_rejected += auto_rejected
            self._total_review += sent_to_review
            self._total_edges += edges_added
            self._last_cycle_at = t0
            self._recent_cycles.append(record)

        # Persist AutoApprover checkpoint if configured and decisions were made
        if (not cfg.dry_run
                and cfg.approver_checkpoint_path
                and aa is not None
                and (auto_approved + auto_rejected) > 0):
            self._save_checkpoint(aa)

        logger.info(
            "AutonomousDiscoveryLoop cycle %d: seen=%d approved=%d rejected=%d "
            "review=%d edges=%d duration=%.2fs circuit=%s",
            cycle_num, len(findings), auto_approved, auto_rejected,
            sent_to_review, edges_added, duration,
            "TRIPPED" if circuit_tripped else "ok",
        )

        return record

    def configure(self, config: LoopConfig) -> None:
        """Replace the current configuration. Thread-safe."""
        with self._lock:
            self._config = config
            # Resize decision window to new circuit_breaker_window
            new_window: Deque[bool] = deque(
                list(self._decision_window)[-config.circuit_breaker_window:],
                maxlen=config.circuit_breaker_window,
            )
            self._decision_window = new_window
        logger.info("AutonomousDiscoveryLoop: configuration updated.")

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of loop health and cumulative statistics."""
        with self._lock:
            window = list(self._decision_window)
            approval_rate = (sum(window) / len(window)) if window else None
            return {
                "running": self._running,
                "cycle_interval": self._config.cycle_interval,
                "max_materializations_per_cycle": self._config.max_materializations_per_cycle,
                "min_approval_rate": self._config.min_approval_rate,
                "circuit_breaker_window": self._config.circuit_breaker_window,
                "dry_run": self._config.dry_run,
                "circuit_breaker_tripped": self._circuit_tripped,
                "current_approval_rate": approval_rate,
                "total_cycles": self._total_cycles,
                "total_approved": self._total_approved,
                "total_rejected": self._total_rejected,
                "total_review": self._total_review,
                "total_edges_added": self._total_edges,
                "started_at": self._started_at,
                "last_cycle_at": self._last_cycle_at,
                "recent_cycles": [vars(r) for r in self._recent_cycles],
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loop_body(self) -> None:
        """Background thread: run cycles on the configured interval."""
        while not self._stop_event.is_set():
            try:
                self.run_cycle()
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: unhandled error in cycle.")

            # Wait for the next cycle or until stop is signalled
            self._stop_event.wait(timeout=self._config.cycle_interval)

    def _update_circuit_breaker(self) -> None:
        """Recalculate circuit_tripped from current decision window. Not locked."""
        window = self._decision_window
        if len(window) < 3:
            # Not enough data to trip; remain open
            self._circuit_tripped = False
            return
        rate = sum(window) / len(window)
        self._circuit_tripped = rate < self._config.min_approval_rate

    def _save_checkpoint(self, aa) -> None:
        """Write AutoApprover checkpoint to disk. Errors are logged, not raised."""
        import json
        path = self._config.approver_checkpoint_path
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(aa.to_dict(), fh, indent=2)
            logger.debug("AutonomousDiscoveryLoop: AutoApprover checkpoint saved → %s", path)
        except Exception:
            logger.exception("AutonomousDiscoveryLoop: failed to save checkpoint to %s.", path)

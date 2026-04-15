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

    auto_rollback_on_trip: bool = False
    """If True, and the circuit breaker trips during a cycle, automatically
    call ProvenanceLedger.rollback_cycle() to undo any edges materialized
    during that cycle.  Requires the ResearchAgent to have a ProvenanceLedger
    attached (via set_provenance_ledger()) and an adapter with remove_edge().
    No-op in dry_run mode."""

    adaptive_tuning: bool = False
    """If True, the loop reads from the attached DiscoveryCalibrator after
    each cycle and adjusts both ``max_materializations_per_cycle`` (higher
    when the graph is underexplored) and the inter-cycle sleep interval
    (shorter when many communities are uncharted).  Original config values
    act as the neutral point; min/max bounds prevent extremes."""

    adaptive_min_cap: int = 1
    """Lower bound on the effective materialization cap when adaptive_tuning
    is active.  Must be >= 1."""

    adaptive_max_cap: int = 20
    """Upper bound on the effective materialization cap when adaptive_tuning
    is active."""

    adaptive_min_interval: float = 60.0
    """Minimum inter-cycle sleep when adaptive_tuning shrinks the interval."""

    adaptive_max_interval: float = 7200.0
    """Maximum inter-cycle sleep when adaptive_tuning extends the interval."""


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

    edges_rolled_back: int = 0
    """Edges removed by auto-rollback when the circuit breaker tripped."""

    effective_cap: int = 0
    """The max_materializations_per_cycle value actually used this cycle
    (may differ from config when adaptive_tuning is active)."""

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
        self._next_interval: float = self._config.cycle_interval  # Phase 82 adaptive
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

        # Phase 82: compute adaptive cap (and stash effective interval for _loop_body)
        effective_cap, effective_interval = self._compute_adaptive_params()
        with self._lock:
            self._next_interval = effective_interval

        findings = self._agent.scan_once()

        auto_approved = 0
        auto_rejected = 0
        sent_to_review = 0
        edges_added = 0
        circuit_tripped = False

        # Re-check circuit breaker state before deciding anything this cycle
        self._update_circuit_breaker()

        for finding in findings:
            if auto_approved >= effective_cap:
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

        # Phase 79: auto-rollback if circuit breaker tripped this cycle
        edges_rolled_back = 0
        if circuit_tripped and cfg.auto_rollback_on_trip and not cfg.dry_run and edges_added > 0:
            ledger = getattr(self._agent, "_provenance_ledger", None)
            adapter = getattr(self._agent, "_adapter", None)
            if ledger is not None and adapter is not None:
                try:
                    rolled = ledger.rollback_cycle(cycle_num, adapter)
                    edges_rolled_back = rolled
                    edges_added = max(0, edges_added - rolled)
                    logger.warning(
                        "AutonomousDiscoveryLoop cycle %d: circuit breaker tripped — "
                        "auto-rolled back %d edges.",
                        cycle_num, rolled,
                    )
                except Exception:
                    logger.exception(
                        "AutonomousDiscoveryLoop cycle %d: auto-rollback failed.",
                        cycle_num,
                    )

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
            edges_rolled_back=edges_rolled_back,
            effective_cap=effective_cap,
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
                "auto_rollback_on_trip": self._config.auto_rollback_on_trip,
                "adaptive_tuning": self._config.adaptive_tuning,
                "adaptive_effective_interval": self._next_interval if self._config.adaptive_tuning else self._config.cycle_interval,
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

            # Phase 82: use adaptive interval if tuning is enabled, else config value
            with self._lock:
                interval = self._next_interval if self._config.adaptive_tuning else self._config.cycle_interval
            self._stop_event.wait(timeout=interval)

    def _compute_adaptive_params(self) -> tuple:
        """
        Read from the DiscoveryCalibrator (if attached) and return
        ``(effective_cap, effective_interval)`` adjusted for current graph coverage.

        Signal: mean community weight from ``calibrator.stats()``.
          - weight > 1.0 → underexplored  → increase cap, shorten interval
          - weight < 1.0 → saturated       → decrease cap, lengthen interval
          - weight ≈ 1.0 → neutral         → no change

        Scale is linear; capped to ``[adaptive_min_cap, adaptive_max_cap]``
        and ``[adaptive_min_interval, adaptive_max_interval]``.

        Returns the base config values unchanged when:
          - ``adaptive_tuning`` is False
          - no calibrator is attached
          - calibrator has no community data yet
        """
        cfg = self._config
        base_cap = cfg.max_materializations_per_cycle
        base_interval = cfg.cycle_interval

        if not cfg.adaptive_tuning:
            return base_cap, base_interval

        calibrator = getattr(self._agent, "_calibrator", None)
        if calibrator is None:
            return base_cap, base_interval

        try:
            stats = calibrator.stats()
        except Exception:
            return base_cap, base_interval

        communities = stats.get("communities", {})
        if not communities:
            return base_cap, base_interval

        weights = [v["weight"] for v in communities.values() if "weight" in v]
        if not weights:
            return base_cap, base_interval

        mean_weight = sum(weights) / len(weights)
        # mean_weight is normalised around 1.0 (neutral)

        effective_cap = int(round(base_cap * mean_weight))
        effective_cap = max(cfg.adaptive_min_cap,
                            min(cfg.adaptive_max_cap, effective_cap))

        # Interval: shorter when graph is underexplored (mean_weight > 1),
        # longer when saturated (mean_weight < 1).  Invert the scale.
        if mean_weight > 0:
            effective_interval = base_interval / mean_weight
        else:
            effective_interval = base_interval
        effective_interval = max(cfg.adaptive_min_interval,
                                 min(cfg.adaptive_max_interval, effective_interval))

        if effective_cap != base_cap or abs(effective_interval - base_interval) > 1.0:
            logger.debug(
                "AutonomousDiscoveryLoop adaptive: mean_weight=%.3f "
                "cap %d→%d  interval %.0f→%.0f",
                mean_weight, base_cap, effective_cap,
                base_interval, effective_interval,
            )

        return effective_cap, effective_interval

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

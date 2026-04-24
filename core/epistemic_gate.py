"""
EpistemicGate — Phase 122: Epistemic Gating.

Converts the MetacognitiveMonitor's EpistemicState into concrete runtime
decisions.  Rather than leaving EU/CIU as passive metrics in the query
response, the gate maps them to four actions:

  suppress    EU >= suppress_threshold  → low_confidence=True on the response
  warn        CIU < credence_threshold  → epistemic_warning string on the response
  research    EU >= research_threshold  → triggered_research=True (server fires
                                          ResearchAgent.scan_once() asynchronously)
  sleep       EU >= sleep_threshold     → triggered_sleep=True (server schedules
                                          SleepCycleOrchestrator)

EpistemicGate.evaluate() is a pure, synchronous function — it returns a
GateDecision with flags and never fires side effects itself.  The server layer
reads the flags and fires the appropriate async tasks.  This keeps the gate
fully testable without a live event loop.

Research cooldown (default 60 s) prevents flooding the ResearchAgent with
triggers on consecutive high-EU queries.  The gate tracks last_triggered_at
internally with a threading.Lock so it is safe under concurrent API requests.
"""
from __future__ import annotations

import dataclasses
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("cerebrum.epistemic_gate")


@dataclass
class GateConfig:
    """Threshold configuration for EpistemicGate decisions."""

    suppress_threshold: float = 0.75
    """EU >= this value marks the response low_confidence."""

    research_threshold: float = 0.70
    """EU >= this value triggers a ResearchAgent scan (subject to cooldown)."""

    sleep_threshold: float = 0.80
    """EU >= this value requests a SleepCycle run."""

    credence_threshold: float = 0.30
    """CIU < this value adds an epistemic_warning to the response."""

    research_cooldown: float = 60.0
    """Minimum seconds between consecutive research triggers."""

    enabled: bool = True
    """Master switch — False passes all queries through with no gating."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GateConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class GateDecision:
    """Output of EpistemicGate.evaluate() for one reasoning call."""

    eu: float = 0.0
    ciu: float = 0.0

    low_confidence: bool = False
    """EU >= suppress_threshold.  Server should annotate response."""

    epistemic_warning: Optional[str] = None
    """Set when CIU < credence_threshold.  Explains low credence to caller."""

    triggered_research: bool = False
    """Server should fire ResearchAgent.scan_once() for this query's seeds."""

    triggered_sleep: bool = False
    """Server should schedule SleepCycleOrchestrator.run()."""

    action_log: List[str] = field(default_factory=list)
    """Human-readable record of every gate decision made."""

    def to_dict(self) -> dict:
        return {
            "eu": round(self.eu, 4),
            "ciu": round(self.ciu, 4),
            "low_confidence": self.low_confidence,
            "epistemic_warning": self.epistemic_warning,
            "triggered_research": self.triggered_research,
            "triggered_sleep": self.triggered_sleep,
            "action_log": self.action_log,
        }


class EpistemicGate:
    """
    Converts EpistemicState signals into runtime gating decisions.

    Parameters
    ----------
    config : GateConfig — threshold configuration (defaults apply if None)
    """

    def __init__(self, config: Optional[GateConfig] = None) -> None:
        self.config = config or GateConfig()
        self._lock = threading.Lock()
        self._last_research_at: float = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate(self, state: Any) -> GateDecision:
        """
        Evaluate an EpistemicState and return a GateDecision.

        Parameters
        ----------
        state : EpistemicState — output of MetacognitiveMonitor.assess()

        Returns
        -------
        GateDecision with all flags set.  Never raises.
        """
        eu = float(getattr(state, "epistemic_uncertainty", 0.0))
        ciu = float(getattr(state, "confidence_in_uncertainty", 0.5))
        decision = GateDecision(eu=eu, ciu=ciu)

        if not self.config.enabled:
            decision.action_log.append("gate disabled — pass-through")
            return decision

        cfg = self.config

        # ── Suppress (low_confidence flag) ────────────────────────────
        if eu >= cfg.suppress_threshold:
            decision.low_confidence = True
            decision.action_log.append(
                f"suppress: EU={eu:.3f} >= threshold={cfg.suppress_threshold}"
            )

        # ── Credence warning ──────────────────────────────────────────
        if ciu < cfg.credence_threshold:
            decision.epistemic_warning = (
                f"Uncertainty estimate has low credence (CIU={ciu:.3f} < "
                f"{cfg.credence_threshold:.3f}). Attach PredictiveCoder and "
                "CerebellarEngine to MetacognitiveMonitor for better calibration."
            )
            decision.action_log.append(
                f"warn: CIU={ciu:.3f} < credence_threshold={cfg.credence_threshold}"
            )

        # ── Research trigger (with cooldown) ─────────────────────────
        if eu >= cfg.research_threshold:
            now = time.monotonic()
            with self._lock:
                elapsed = now - self._last_research_at
                if elapsed >= cfg.research_cooldown:
                    decision.triggered_research = True
                    self._last_research_at = now
                    decision.action_log.append(
                        f"research_triggered: EU={eu:.3f} >= "
                        f"threshold={cfg.research_threshold}"
                    )
                else:
                    remaining = cfg.research_cooldown - elapsed
                    decision.action_log.append(
                        f"research_skipped: cooldown {remaining:.0f}s remaining"
                    )

        # ── Sleep trigger ─────────────────────────────────────────────
        if eu >= cfg.sleep_threshold:
            decision.triggered_sleep = True
            decision.action_log.append(
                f"sleep_triggered: EU={eu:.3f} >= threshold={cfg.sleep_threshold}"
            )

        logger.debug(
            "EpistemicGate: EU=%.3f CIU=%.3f low_conf=%s research=%s sleep=%s",
            eu, ciu, decision.low_confidence,
            decision.triggered_research, decision.triggered_sleep,
        )
        return decision

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_config(self, **kwargs: Any) -> None:
        """Partial update of GateConfig fields by keyword."""
        valid = {f.name for f in dataclasses.fields(GateConfig)}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self.config, k, v)
            else:
                logger.warning("EpistemicGate.update_config: unknown field %r", k)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "last_research_triggered_ago_seconds": round(
                time.monotonic() - self._last_research_at, 1
            ),
        }

"""
MetaOrchestrator — Phase 260.

Master daemon that starts and coordinates all autonomous sub-systems:
  - SelfEvaluator   (performance measurement)
  - ParameterDaemon (parameter re-derivation on graph change)
  - AutonomousDiscoveryLoop (knowledge gap filling)
  - AutonomousResearcher    (code mutation + benchmark)

Events flow through a shared EventBus (thread-safe queue). The orchestrator
routes events:

  IMPROVED   → log, update best-params snapshot
  PLATEAU    → trigger ResearchAgent.scan_once() for gap filling
  REGRESSION → pause AutonomousDiscoveryLoop materialization,
               check ProvenanceLedger for rollback candidates
  GRAPH_CHANGED → notify ParameterDaemon to re-derive params

Circuit breaker: two consecutive REGRESSION events pause all materialization
and set circuit_open=True. Reset manually via reset_circuit() or REST.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.self_evaluator import (
    SelfEvaluator, SelfEvaluatorConfig,
    EvalEvent, IMPROVED, PLATEAU, REGRESSION,
)
from core.parameter_daemon import ParameterDaemon, GRAPH_CHANGED

logger = logging.getLogger(__name__)


# ── Event bus ─────────────────────────────────────────────────────────────────

@dataclass
class BusEvent:
    kind:    str
    payload: Any = None
    ts:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class EventBus:
    """Simple thread-safe publish/subscribe queue."""

    def __init__(self) -> None:
        self._q: queue.Queue[BusEvent] = queue.Queue()
        self._listeners: dict[str, list[Callable[[BusEvent], None]]] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def subscribe(self, kind: str, fn: Callable[[BusEvent], None]) -> None:
        with self._lock:
            self._listeners.setdefault(kind, []).append(fn)

    def publish(self, kind: str, payload: Any = None) -> None:
        self._q.put(BusEvent(kind=kind, payload=payload))

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._dispatch, name="event-bus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._q.put(BusEvent(kind="__stop__"))  # unblock queue
        if self._thread:
            self._thread.join(timeout=5)

    def _dispatch(self) -> None:
        while self._running:
            try:
                event = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if event.kind == "__stop__":
                break
            with self._lock:
                listeners = list(self._listeners.get(event.kind, []))
            for fn in listeners:
                try:
                    fn(event)
                except Exception:
                    logger.exception("EventBus: listener error on %s.", event.kind)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    # SelfEvaluator
    eval_dataset:         str   = "metaqa"
    eval_n_questions:     int   = 100
    eval_interval_s:      float = 600.0
    eval_plateau_window:  int   = 3
    embeddings:           str   = "sentence"

    # ParameterDaemon
    param_validation_n:   int   = 20
    param_q_delta_gate:   float = 0.05
    param_edge_delta_gate: float = 0.01

    # Circuit breaker
    max_consecutive_regressions: int = 2


# ── MetaOrchestrator ──────────────────────────────────────────────────────────

class MetaOrchestrator:
    """
    Wires SelfEvaluator + ParameterDaemon + AutonomousDiscoveryLoop into
    one coordinated self-improvement loop.
    """

    def __init__(
        self,
        graph: Any,                          # CerebrumGraph
        config: Optional[OrchestratorConfig] = None,
        discovery_loop: Optional[Any] = None,    # AutonomousDiscoveryLoop
        params_getter: Optional[Callable[[], dict]] = None,
        params_setter: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._graph          = graph
        self._config         = config or OrchestratorConfig()
        self._discovery_loop = discovery_loop
        self._params_getter  = params_getter or self._default_params_getter
        self._params_setter  = params_setter or self._default_params_setter

        self._bus      = EventBus()
        self._running  = False
        self._lock     = threading.Lock()

        self._consecutive_regressions = 0
        self._circuit_open            = False
        self._started_at: Optional[float] = None
        self._events_routed: int = 0

        self._evaluator = SelfEvaluator(
            config=SelfEvaluatorConfig(
                dataset          = self._config.eval_dataset,
                n_questions      = self._config.eval_n_questions,
                interval_seconds = self._config.eval_interval_s,
                plateau_window   = self._config.eval_plateau_window,
                embeddings       = self._config.embeddings,
            ),
            params_getter=self._params_getter,
        )
        self._evaluator.add_listener(self._on_eval_event)

        self._param_daemon = ParameterDaemon(
            graph_getter   = lambda: self._graph,
            params_setter  = self._params_setter,
            params_getter  = self._params_getter,
            config_dataset = self._config.eval_dataset,
            validation_n   = self._config.param_validation_n,
            q_delta_gate   = self._config.param_q_delta_gate,
            edge_delta_gate= self._config.param_edge_delta_gate,
            embeddings     = self._config.embeddings,
        )

        # Wire bus → param daemon
        self._bus.subscribe(GRAPH_CHANGED, self._on_graph_changed)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running  = True
            self._started_at = time.time()

        self._bus.start()
        self._evaluator.start()
        if self._discovery_loop is not None:
            try:
                self._discovery_loop.start()
            except Exception:
                logger.exception("MetaOrchestrator: failed to start AutonomousDiscoveryLoop.")
        logger.info("MetaOrchestrator: started.")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False

        self._evaluator.stop()
        if self._discovery_loop is not None:
            try:
                self._discovery_loop.stop()
            except Exception:
                pass
        self._bus.stop()
        logger.info("MetaOrchestrator: stopped.")

    def notify_graph_changed(self, edge_count: int) -> None:
        """Called by AutonomousDiscoveryLoop (or any graph mutation) after edges are added."""
        self._bus.publish(GRAPH_CHANGED, {"edge_count": edge_count})

    def trigger_eval(self) -> Optional[Any]:
        """Manually trigger one self-eval (returns EvalResult)."""
        return self._evaluator.trigger()

    def reset_circuit(self) -> None:
        """Clear circuit breaker and resume materialization."""
        with self._lock:
            self._consecutive_regressions = 0
            self._circuit_open = False
        if self._discovery_loop is not None:
            try:
                from core.autonomous_loop import LoopConfig
                cfg = getattr(self._discovery_loop, "_config", None)
                if cfg is not None:
                    cfg.dry_run = False
            except Exception:
                pass
        logger.info("MetaOrchestrator: circuit breaker reset.")

    def status(self) -> dict:
        with self._lock:
            return {
                "running":                  self._running,
                "circuit_open":             self._circuit_open,
                "consecutive_regressions":  self._consecutive_regressions,
                "events_routed":            self._events_routed,
                "started_at":               datetime.fromtimestamp(self._started_at, tz=timezone.utc).isoformat(timespec="seconds") if self._started_at else None,
                "evaluator":                self._evaluator.status(),
            }

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_eval_event(self, event: EvalEvent) -> None:
        with self._lock:
            self._events_routed += 1

        if event.kind == IMPROVED:
            with self._lock:
                self._consecutive_regressions = 0
            logger.info("MetaOrchestrator: IMPROVED (H@1=%.4f +%.4f).", event.result.h1, event.delta)

        elif event.kind == PLATEAU:
            logger.info("MetaOrchestrator: PLATEAU — triggering ResearchAgent scan.")
            self._trigger_research_scan()

        elif event.kind == REGRESSION:
            with self._lock:
                self._consecutive_regressions += 1
                n = self._consecutive_regressions
                threshold = self._config.max_consecutive_regressions

            logger.warning(
                "MetaOrchestrator: REGRESSION (%d/%d) — H@1=%.4f (%.4f).",
                n, threshold, event.result.h1, event.delta,
            )

            if n >= threshold:
                self._open_circuit()
            else:
                self._check_rollback()

    def _on_graph_changed(self, bus_event: BusEvent) -> None:
        edge_count = (bus_event.payload or {}).get("edge_count", -1)
        self._param_daemon.handle_graph_changed(edge_count)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _trigger_research_scan(self) -> None:
        if self._discovery_loop is None:
            return
        agent = getattr(self._discovery_loop, "_agent", None)
        if agent is None:
            return
        threading.Thread(
            target=self._safe_scan, args=(agent,), daemon=True, name="research-scan"
        ).start()

    def _safe_scan(self, agent: Any) -> None:
        try:
            findings = agent.scan_once()
            logger.info("MetaOrchestrator: ResearchAgent found %d candidates.", len(findings))
        except Exception:
            logger.exception("MetaOrchestrator: ResearchAgent scan failed.")

    def _open_circuit(self) -> None:
        with self._lock:
            self._circuit_open = True
        logger.error(
            "MetaOrchestrator: circuit breaker OPEN — pausing materialization. "
            "Call reset_circuit() to resume."
        )
        if self._discovery_loop is not None:
            try:
                cfg = getattr(self._discovery_loop, "_config", None)
                if cfg is not None:
                    cfg.dry_run = True
            except Exception:
                pass

    def _check_rollback(self) -> None:
        """Ask ProvenanceLedger to rollback the most recent cycle."""
        if self._discovery_loop is None:
            return
        agent  = getattr(self._discovery_loop, "_agent", None)
        ledger = getattr(agent, "_provenance_ledger", None) if agent else None
        if ledger is None:
            return
        adapter = getattr(agent, "_adapter", None)
        if adapter is None:
            return
        cycle = getattr(self._discovery_loop, "_cycle_number", None)
        if cycle and cycle > 0:
            try:
                rolled = ledger.rollback_cycle(cycle, adapter)
                logger.info("MetaOrchestrator: rolled back %d edges from cycle %d.", rolled, cycle)
            except Exception:
                logger.exception("MetaOrchestrator: rollback failed.")

    # ── Default param accessors ───────────────────────────────────────────────

    def _default_params_getter(self) -> dict:
        try:
            adapter = getattr(self._graph, "adapter", self._graph)
            csa     = getattr(adapter, "_csa_engine", None) or getattr(self._graph, "_csa", None)
            if csa is None:
                return {}
            return {
                "trb_factor":   getattr(csa, "trb_factor",   None),
                "gamma":        getattr(csa, "gamma",         None),
                "beta":         getattr(csa, "beta",          None),
                "idf_weight":   getattr(csa, "idf_weight",   None),
                "vote_weight":  getattr(csa, "vote_weight",  None),
                "beam_width":   getattr(csa, "beam_width",   None),
            }
        except Exception:
            return {}

    def _default_params_setter(self, params: dict) -> None:
        try:
            adapter = getattr(self._graph, "adapter", self._graph)
            csa     = getattr(adapter, "_csa_engine", None) or getattr(self._graph, "_csa", None)
            if csa is None:
                return
            for k, v in params.items():
                if v is not None and hasattr(csa, k):
                    setattr(csa, k, v)
        except Exception:
            logger.exception("MetaOrchestrator: params_setter failed.")

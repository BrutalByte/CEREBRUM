"""
AutonomousDiscoveryLoop — Phase 74.

Closes the full discover → validate → approve → materialize loop by running
ResearchAgent.scan_once() on a configurable timer, processing each finding
through the attached AutoApprover, and tracking cycle health via a circuit
breaker that pauses autonomous materialization when approval rates collapse.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.active_inference import ActiveInferenceEngine

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
    during that cycle."""

    adaptive_tuning: bool = False
    """If True, the loop adjusts timing and cap based on DiscoveryCalibrator."""

    adaptive_min_cap: int = 1
    adaptive_max_cap: int = 20
    adaptive_min_interval: float = 60.0
    adaptive_max_interval: float = 7200.0

    # Phase 93: Active Inference (Daydreaming)
    active_inference: bool = False
    active_inference_interval: float = 60.0
    active_inference_floor: float = 0.2

    # Phase 94: Self-Modifying GUI
    gui_adaptation: bool = False
    gui_widget_path: str = "/Game/UI/WBP_CerebrumHUD"
    gui_toolkit_url: str = "http://localhost:3000"

    # Phase 95: Working Memory + Goal System
    working_memory: bool = False
    working_memory_maxlen: int = 50

    # Phase 96: Memory Consolidation (Hebbian Replay)
    consolidation: bool = False
    consolidation_min_score: float = 0.6
    consolidation_k: int = 5
    consolidation_max_weight: float = 2.0
    consolidation_hebbian_delta: float = 0.05

    # Phase 97: Synaptic Decay (LTD / Synaptic Homeostasis)
    synaptic_decay: bool = False
    decay_rate: float = 0.01
    decay_baseline: float = 1.0
    decay_min_weight: float = 0.5
    decay_resistance_k: float = 5.0
    decay_interval: float = 600.0

    # Phase 101: Emotional Valence (Amygdala)
    valence_learning: bool = False

    # Phase 102: Default Mode Network (self-referential idle reasoning)
    default_mode: bool = False
    default_mode_idle_threshold: float = 120.0
    default_mode_max_insights: int = 3

    # Phase 105: Recursive Self-Synthesis
    autonomous_research: bool = False
    research_interval: int = 600 # Every 10 mins
    recursive_synthesis: bool = True
    metaplasticity: bool = True

    # Phase 119: Sleep Cycle (offline consolidation)
    sleep_cycle: bool = False
    sleep_idle_threshold: float = 300.0


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
    auto_approved: int
    auto_rejected: int
    sent_to_review: int
    edges_added: int
    circuit_breaker_tripped: bool = False
    edges_rolled_back: int = 0
    effective_cap: int = 0
    dry_run: bool = False


# ---------------------------------------------------------------------------
# AutonomousDiscoveryLoop
# ---------------------------------------------------------------------------

class AutonomousDiscoveryLoop:
    """
    Autonomous discovery loop that periodically calls ResearchAgent.scan_once()
    and processes each finding through the attached AutoApprover.
    """

    def __init__(self, agent, config: Optional[LoopConfig] = None) -> None:
        self._agent = agent
        self._config = config or LoopConfig()
        self._orchestrator = None  # Phase 260: set by MetaOrchestrator.attach_loop()

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

        # Sliding window for circuit breaker
        self._decision_window: Deque[bool] = deque(maxlen=self._config.circuit_breaker_window)
        self._circuit_tripped = False

        self._recent_cycles: Deque[CycleRecord] = deque(maxlen=50)
        self._next_interval: float = self._config.cycle_interval
        self._lock = threading.Lock()

        # Phase 93
        graph = getattr(agent._adapter, "graph", agent._adapter)
        self._inference_engine = ActiveInferenceEngine(
            graph=graph,
            metabolic_floor=self._config.active_inference_floor
        )
        self._total_inference_pulses = 0
        self._last_inference_at: Optional[float] = None

        # Phase 95: Working Memory + Goal System
        self._wm = None
        self._goal_stack = None
        self._goal_evaluator = None
        self._suppress_inference: bool = False
        if self._config.working_memory:
            try:
                from core.working_memory import WorkingMemoryBuffer
                from core.goal_system import GoalStack, GoalEvaluator
                self._wm = WorkingMemoryBuffer(maxlen=self._config.working_memory_maxlen)
                self._goal_stack = GoalStack()
                self._goal_evaluator = GoalEvaluator(graph, self, self._wm)
                if hasattr(graph, "attach_working_memory"):
                    graph.attach_working_memory(self._wm)
                if hasattr(graph, "attach_goal_stack"):
                    graph.attach_goal_stack(self._goal_stack)
                # Phase 98 Gaps 3/4/5: attach WM to agent subsystems
                for attr in ("_insight_engine", "_cerebellar_engine"):
                    engine = getattr(agent, attr, None)
                    if engine is not None and hasattr(engine, "set_working_memory"):
                        engine.set_working_memory(self._wm)
                if hasattr(agent, "set_working_memory"):
                    agent.set_working_memory(self._wm)
                logger.info("AutonomousDiscoveryLoop: WorkingMemory + GoalStack enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init working memory/goal system.")

        # Phase 96: Memory Consolidation
        self._consolidation_engine = None
        self._total_consolidations = 0
        self._total_edges_strengthened = 0
        if self._config.consolidation:
            try:
                from core.consolidation_engine import ConsolidationEngine
                self._consolidation_engine = ConsolidationEngine(
                    adapter=graph.adapter,
                    graph=graph,
                    min_score=self._config.consolidation_min_score,
                    max_weight=self._config.consolidation_max_weight,
                    hebbian_delta=self._config.consolidation_hebbian_delta,
                )
                logger.info("AutonomousDiscoveryLoop: ConsolidationEngine enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init ConsolidationEngine.")

        # Phase 97: Synaptic Decay (LTD)
        self._decay_engine = None
        self._total_decay_passes = 0
        self._total_edges_decayed = 0
        self._last_decay_at: float = 0.0
        if self._config.synaptic_decay:
            try:
                from core.synaptic_decay_engine import SynapticDecayEngine
                self._decay_engine = SynapticDecayEngine(
                    adapter=graph.adapter,
                    graph=graph,
                    baseline_weight=self._config.decay_baseline,
                    decay_rate=self._config.decay_rate,
                    min_weight=self._config.decay_min_weight,
                    resistance_k=self._config.decay_resistance_k,
                )
                logger.info("AutonomousDiscoveryLoop: SynapticDecayEngine enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init SynapticDecayEngine.")

        # Phase 101: Emotional Valence
        self._valence_engine = None
        if self._config.valence_learning:
            try:
                from core.valence_engine import ValenceEngine
                self._valence_engine = ValenceEngine(
                    adapter=graph.adapter,
                    graph=graph,
                )
                if hasattr(graph, "attach_valence_engine"):
                    graph.attach_valence_engine(self._valence_engine)
                if hasattr(agent, "set_valence_engine"):
                    agent.set_valence_engine(self._valence_engine)
                logger.info("AutonomousDiscoveryLoop: ValenceEngine enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init ValenceEngine.")

        # Phase 102: Default Mode Network
        self._dmn_engine = None
        self._total_dmn_pulses = 0
        self._last_query_at: float = time.time()
        if self._config.default_mode:
            try:
                from core.default_mode_engine import DefaultModeEngine
                self._dmn_engine = DefaultModeEngine(
                    adapter=graph.adapter,
                    graph=graph,
                    max_insights=self._config.default_mode_max_insights,
                )
                logger.info("AutonomousDiscoveryLoop: DefaultModeEngine enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init DefaultModeEngine.")

        # Phase 105: Recursive Self-Synthesis
        self._researcher = None
        if self._config.autonomous_research:
            try:
                from core.autonomous_researcher import AutonomousResearcher
                self._researcher = AutonomousResearcher(modulator=getattr(graph, "modulator", None))
                logger.info("AutonomousDiscoveryLoop: AutonomousResearcher enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init AutonomousResearcher.")

        # Phase 119: Sleep Cycle (offline consolidation)
        self._sleep_orchestrator = None
        if self._config.sleep_cycle:
            try:
                from core.sleep_cycle import SleepCycleOrchestrator
                self._sleep_orchestrator = SleepCycleOrchestrator(
                    adapter=graph.adapter,
                    engram_consolidator=None,  # wired externally via attach_sleep_cycle
                    consolidation_engine=self._consolidation_engine,
                    synaptic_decay_engine=self._decay_engine,
                    default_mode_engine=self._dmn_engine,
                    working_memory=self._wm,
                    goal_stack=self._goal_stack,
                )
                logger.info("AutonomousDiscoveryLoop: SleepCycleOrchestrator enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init SleepCycleOrchestrator.")

        # Phase 94: GUI Adaptation
        self._gui_engine = None
        if self._config.gui_adaptation:
            try:
                from api.ue_toolkit_client import UEToolkitClient
                from core.gui_adaptation_engine import GUIAdaptationEngine
                toolkit = UEToolkitClient(self._config.gui_toolkit_url)
                emit_fn = getattr(graph, "emit", None)
                self._gui_engine = GUIAdaptationEngine(
                    toolkit=toolkit,
                    emit_fn=emit_fn,
                    widget_path=self._config.gui_widget_path,
                )
                logger.info("AutonomousDiscoveryLoop: GUIAdaptationEngine enabled.")
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: failed to init GUIAdaptationEngine.")

    def start(self) -> None:
        """Start the background loop thread (idempotent)."""
        with self._lock:
            if self._running:
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
        logger.info("AutonomousDiscoveryLoop: started (interval=%.0fs, active_inference=%s).",
                    self._config.cycle_interval, self._config.active_inference)

    def stop(self) -> None:
        """Signal the loop to stop and wait for the current cycle to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        logger.info("AutonomousDiscoveryLoop: stopped.")

    def run_cycle(self) -> CycleRecord:
        """Execute a single discovery cycle synchronously."""
        with self._lock:
            self._cycle_number += 1
            cycle_num = self._cycle_number

        t0 = time.time()
        cfg = self._config
        aa = getattr(self._agent, "_auto_approver", None)

        effective_cap, effective_interval = self._compute_adaptive_params()
        with self._lock:
            self._next_interval = effective_interval

        findings = self._agent.scan_once()

        auto_approved = 0
        auto_rejected = 0
        sent_to_review = 0
        edges_added = 0
        circuit_tripped = False

        self._update_circuit_breaker()

        for finding in findings:
            if auto_approved >= effective_cap:
                sent_to_review += 1
                continue
            if aa is None:
                sent_to_review += 1
                continue

            from core.auto_approver import AutoDecision
            decision: AutoDecision = aa.decide(finding)

            if decision.action == "approve":
                self._decision_window.append(True)
                self._update_circuit_breaker()

                if self._circuit_tripped:
                    circuit_tripped = True
                    sent_to_review += 1
                    continue

                if not cfg.dry_run:
                    try:
                        n = self._agent.approve(finding.finding_id, cycle_number=cycle_num)
                        edges_added += n
                        aa.fit(finding, approved=True)
                    except ValueError:
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
            else:
                sent_to_review += 1

        duration = time.time() - t0
        edges_rolled_back = 0
        if circuit_tripped and cfg.auto_rollback_on_trip and not cfg.dry_run and edges_added > 0:
            ledger = getattr(self._agent, "_provenance_ledger", None)
            adapter = getattr(self._agent, "_adapter", None)
            if ledger is not None and adapter is not None:
                try:
                    rolled = ledger.rollback_cycle(cycle_num, adapter)
                    edges_rolled_back = rolled
                    edges_added = max(0, edges_added - rolled)
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop cycle %d: auto-rollback failed.", cycle_num)

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

        if (not cfg.dry_run and cfg.approver_checkpoint_path and aa is not None and (auto_approved + auto_rejected) > 0):
            self._save_checkpoint(aa)

        # Phase 260: notify MetaOrchestrator when edges were added
        if edges_added > 0 and self._orchestrator is not None:
            try:
                adapter = getattr(self._agent, "_adapter", None)
                edge_count = adapter.graph.number_of_edges() if adapter and hasattr(getattr(adapter, "graph", None), "number_of_edges") else -1
                self._orchestrator.notify_graph_changed(edge_count)
            except Exception:
                logger.debug("AutonomousDiscoveryLoop: could not notify orchestrator.", exc_info=True)

        return record

    def configure(self, config: LoopConfig) -> None:
        """Hot-update loop configuration without restarting the thread."""
        with self._lock:
            old_research = self._config.autonomous_research
            self._config = config
            self._next_interval = config.cycle_interval
            self._decision_window = deque(
                self._decision_window, maxlen=config.circuit_breaker_window
            )
            
            # Phase 105: Hot-update researcher
            if self._config.autonomous_research:
                if self._researcher is None:
                    try:
                        from core.autonomous_researcher import AutonomousResearcher
                        graph = getattr(self._agent._adapter, "graph", self._agent._adapter)
                        self._researcher = AutonomousResearcher(
                            modulator=getattr(graph, "modulator", None),
                            recursive_synthesis=self._config.recursive_synthesis,
                            metaplasticity=self._config.metaplasticity
                        )
                        logger.info("AutonomousDiscoveryLoop: AutonomousResearcher enabled via reconfig.")
                    except Exception:
                        logger.exception("AutonomousDiscoveryLoop: failed to init researcher on reconfig.")
                else:
                    # Update existing researcher flags
                    self._researcher.recursive_synthesis = self._config.recursive_synthesis
                    self._researcher.metaplasticity = self._config.metaplasticity
            else:
                self._researcher = None

    def push_goal(self, goal: Any) -> None:
        """Push a user-defined goal onto the goal stack (requires working_memory=True)."""
        if self._goal_stack is None:
            raise RuntimeError("working_memory must be True in LoopConfig to use goals.")
        self._goal_stack.push(goal)

    def get_goal_stack(self) -> Optional[Any]:
        """Return the GoalStack, or None if working_memory is not enabled."""
        return self._goal_stack

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of loop health and cumulative statistics."""
        with self._lock:
            window = list(self._decision_window)
            approval_rate = (sum(window) / len(window)) if window else None
            cfg = self._config
            s: Dict[str, Any] = {
                "running": self._running,
                "cycle_interval": cfg.cycle_interval,
                "max_materializations_per_cycle": cfg.max_materializations_per_cycle,
                "min_approval_rate": cfg.min_approval_rate,
                "circuit_breaker_window": cfg.circuit_breaker_window,
                "dry_run": cfg.dry_run,
                "auto_rollback_on_trip": cfg.auto_rollback_on_trip,
                "adaptive_tuning": cfg.adaptive_tuning,
                "active_inference_enabled": cfg.active_inference,
                "gui_adaptation_enabled": cfg.gui_adaptation,
                "total_inference_pulses": self._total_inference_pulses,
                "last_inference_at": self._last_inference_at,
                "circuit_breaker_tripped": self._circuit_tripped,
                "current_approval_rate": approval_rate,
                "total_cycles": self._total_cycles,
                "total_approved": self._total_approved,
                "total_rejected": self._total_rejected,
                "total_review": self._total_review,
                "total_edges_added": self._total_edges,
                "started_at": self._started_at,
                "last_cycle_at": self._last_cycle_at,
                "adaptive_effective_interval": self._next_interval if cfg.adaptive_tuning else cfg.cycle_interval,
                "recent_cycles": [vars(r) for r in self._recent_cycles],
                "working_memory_enabled": cfg.working_memory,
                "working_memory_size": len(self._wm._buffer) if self._wm is not None else 0,
                "active_goals": len(self._goal_stack.all_active()) if self._goal_stack is not None else 0,
                "inference_suppressed": self._suppress_inference,
                "consolidation_enabled": cfg.consolidation,
                "total_consolidations": self._total_consolidations,
                "total_edges_strengthened": self._total_edges_strengthened,
                "valence_learning_enabled": cfg.valence_learning,
                "synaptic_decay_enabled": cfg.synaptic_decay,
                "total_decay_passes": self._total_decay_passes,
                "total_edges_decayed": self._total_edges_decayed,
                "default_mode_enabled": cfg.default_mode,
                "total_dmn_pulses": self._total_dmn_pulses,
                "autonomous_research_enabled": cfg.autonomous_research,
                "research_interval": cfg.research_interval,
                "recursive_synthesis_enabled": cfg.recursive_synthesis,
                "metaplasticity_enabled": cfg.metaplasticity,
            }
            return s

    def _loop_body(self) -> None:
        """Background thread: run cycles and active inference."""
        last_inference_at = 0.0
        while not self._stop_event.is_set():
            now = time.time()
            # 1. Discovery Cycle
            try:
                self.run_cycle()
            except Exception:
                logger.exception("AutonomousDiscoveryLoop: cycle failed.")

            # 2. Phase 95: Record cycle to WM + evaluate goals
            if self._wm is not None:
                try:
                    from core.working_memory import MemoryEntry
                    self._wm.record(MemoryEntry(
                        timestamp=now,
                        seeds=[],
                        answers=[],
                        top_score=0.0,
                        soliton_index=None,
                        prediction_error=None,
                        source="discovery_cycle",
                    ))
                    self._suppress_inference = self._goal_evaluator.evaluate(self._goal_stack)
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: goal evaluation failed.")

            # 2b. Phase 96/98: Memory Consolidation (Hebbian replay + Gap 1 reinforcement)
            if self._consolidation_engine is not None and self._wm is not None:
                try:
                    graph = getattr(self._agent, "_adapter", None)
                    graph = getattr(graph, "graph", graph)
                    mod = getattr(graph, "modulator", None)
                    reinforcement_scale = 1.0
                    if mod is not None:
                        reinforcement_scale = getattr(mod, "state", {}).get("reinforcement", 1.0)
                    result = self._consolidation_engine.consolidate(
                        self._wm,
                        k=self._config.consolidation_k,
                        reinforcement_scale=reinforcement_scale,
                    )
                    with self._lock:
                        self._total_consolidations += 1
                        self._total_edges_strengthened += result.edges_strengthened
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: consolidation failed.")

            # 2c. Phase 98 Gap 6: DiscoveryCalibrator → GoalStack
            if self._goal_stack is not None:
                try:
                    calibrator = getattr(self._agent, "_calibrator", None)
                    if calibrator is not None:
                        stats = calibrator.stats()
                        for cid, info in stats.get("communities", {}).items():
                            if info.get("weight", 1.0) > 3.0:
                                goal_desc = f"explore_community_{cid}"
                                active = self._goal_stack.all_active()
                                if not any(g.description == goal_desc for g in active):
                                    from core.goal_system import make_goal
                                    self._goal_stack.push(make_goal(
                                        description=goal_desc,
                                        metric_type="approval_rate",
                                        target_value=0.15,
                                        priority=6,
                                    ))
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: calibrator→goal wiring failed.")

            # 2d. Phase 98 Gap 7: pass goal-hinted entities to ResearchAgent
            if self._goal_stack is not None and self._goal_evaluator is not None:
                try:
                    hints = self._goal_evaluator.get_context_seeds(self._goal_stack, self._wm)
                    if hasattr(self._agent, "set_goal_hints"):
                        self._agent.set_goal_hints(hints or [])
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: goal→research wiring failed.")

            # 2e. Phase 97: Synaptic Decay (LTD) — runs on its own slower timer
            if self._decay_engine is not None:
                if (now - self._last_decay_at) >= self._config.decay_interval:
                    try:
                        result = self._decay_engine.decay(self._wm)
                        with self._lock:
                            self._total_decay_passes += 1
                            self._total_edges_decayed += result.edges_decayed
                        self._last_decay_at = now
                    except Exception:
                        logger.exception("AutonomousDiscoveryLoop: synaptic decay failed.")

            # 2f. Phase 102: Default Mode Network — idle self-reflection
            if self._dmn_engine is not None:
                # Derive last real query time from WM entries (source="query"),
                # since _last_query_at is set at init and CerebrumGraph.query()
                # has no reference back to this loop.
                last_real_query = self._last_query_at
                if self._wm is not None:
                    for _entry in self._wm.recent(20):
                        if _entry.source == "query" and _entry.timestamp > last_real_query:
                            last_real_query = _entry.timestamp
                idle_secs = now - last_real_query
                if idle_secs >= self._config.default_mode_idle_threshold:
                    try:
                        graph = getattr(self._agent, "_adapter", None)
                        graph = getattr(graph, "graph", graph)
                        mod = getattr(graph, "modulator", None)
                        arousal = getattr(mod, "state", {}).get("arousal", 1.0) if mod else 1.0
                        if arousal < 1.5:
                            calibrator = getattr(self._agent, "_calibrator", None)
                            insights = self._dmn_engine.idle_scan(
                                wm=self._wm,
                                goal_stack=self._goal_stack,
                                calibrator=calibrator,
                            )
                            # Phase 105: Recursive Self-Synthesis
                            if self._researcher is not None and insights:
                                self._researcher.process_dmn_insights(insights)
                                
                            with self._lock:
                                self._total_dmn_pulses += 1
                    except Exception:
                        logger.exception("AutonomousDiscoveryLoop: DMN idle scan failed.")

            # 3. Active Inference (Daydreaming)
            if self._config.active_inference and (now - last_inference_at) >= self._config.active_inference_interval:
                try:
                    context_seeds = None
                    if self._goal_stack is not None and not self._suppress_inference:
                        context_seeds = self._goal_evaluator.get_context_seeds(self._goal_stack, self._wm)
                    if not self._suppress_inference:
                        res = self._inference_engine.step(context_seeds=context_seeds)
                        if res:
                            with self._lock:
                                self._total_inference_pulses += 1
                                self._last_inference_at = time.time()
                            last_inference_at = now
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: active inference failed.")

            # 4. GUI Adaptation (Phase 94)
            if self._gui_engine is not None:
                try:
                    from core.gui_adaptation_engine import SignalSnapshot
                    graph = getattr(self._agent._adapter, "graph", self._agent._adapter)
                    mod = getattr(graph, "modulator", None)
                    pc  = getattr(graph, "predictive_coder", None)
                    loop_status = self.status()
                    snapshot = SignalSnapshot(
                        timestamp=time.time(),
                        arousal=getattr(mod, "state", {}).get("arousal", 1.0) if mod else 1.0,
                        reinforcement=getattr(mod, "state", {}).get("reinforcement", 1.0) if mod else 1.0,
                        soliton_index=None,
                        approval_rate=loop_status.get("current_approval_rate"),
                        circuit_breaker_tripped=loop_status.get("circuit_breaker_tripped", False),
                        total_inference_pulses=loop_status.get("total_inference_pulses", 0),
                    )
                    # Enrich soliton_index from predictive coder if available
                    if pc:
                        try:
                            stats = pc.soliton_stats()
                            if stats:
                                snapshot.soliton_index = sum(stats.values()) / len(stats)
                        except Exception:
                            pass
                    self._gui_engine.record(snapshot)
                    fired = self._gui_engine.step()
                    if fired:
                        logger.info("GUIAdaptationEngine fired rules: %s", fired)
                except Exception:
                    logger.exception("AutonomousDiscoveryLoop: GUI adaptation failed.")

            with self._lock:
                interval = self._next_interval if self._config.adaptive_tuning else self._config.cycle_interval
            self._stop_event.wait(timeout=min(interval, self._config.active_inference_interval))

    def _compute_adaptive_params(self) -> tuple:
        cfg = self._config
        base_cap = cfg.max_materializations_per_cycle
        base_interval = cfg.cycle_interval
        if not cfg.adaptive_tuning: return base_cap, base_interval
        calibrator = getattr(self._agent, "_calibrator", None)
        if calibrator is None: return base_cap, base_interval
        try:
            stats = calibrator.stats()
            communities = stats.get("communities", {})
            weights = [v["weight"] for v in communities.values() if "weight" in v]
            if not weights: return base_cap, base_interval
            mean_weight = sum(weights) / len(weights)
            effective_cap = max(cfg.adaptive_min_cap, min(cfg.adaptive_max_cap, int(round(base_cap * mean_weight))))
            effective_interval = max(cfg.adaptive_min_interval, min(cfg.adaptive_max_interval, base_interval / (mean_weight or 1.0)))
            return effective_cap, effective_interval
        except Exception:
            return base_cap, base_interval

    def _update_circuit_breaker(self) -> None:
        window = self._decision_window
        if len(window) < 3:
            self._circuit_tripped = False
            return
        rate = sum(window) / len(window)
        self._circuit_tripped = rate < self._config.min_approval_rate

    def _save_checkpoint(self, aa) -> None:
        import json
        path = self._config.approver_checkpoint_path
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(aa.to_dict(), fh, indent=2)
        except Exception:
            logger.exception("AutonomousDiscoveryLoop: failed to save checkpoint.")


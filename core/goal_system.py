"""Phase 95 — Goal System.

Goal-directed reasoning: Goal dataclass, GoalStack (priority queue with built-in
homeostasis rules), and GoalEvaluator (measures metrics + drives inference direction).

Standing homeostasis rules (always active, auto-reset after cooldown):
  AROUSAL_CAP       — suppress inference when system is over-excited
  SOLITON_HEALTH    — keep prediction stability above 0.75
  REINFORCEMENT_FLOOR — keep positive feedback above 0.5
  DISCOVERY_QUALITY — warn when approval rate drops below 0.15
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger(__name__)


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class GoalMetricType(str, Enum):
    SOLITON_INDEX = "soliton_index"         # mean soliton_stats() value
    APPROVAL_RATE = "approval_rate"         # loop approval rate
    REINFORCEMENT = "reinforcement"         # ChemicalModulator reinforcement
    ENTITY_DISCOVERED = "entity_discovered" # target entity present in graph
    AROUSAL_BELOW = "arousal_below"         # arousal ≤ target_value (inverted)


_HISTORY_CAP = 200


@dataclass
class Goal:
    id: str
    description: str
    metric_type: GoalMetricType
    target_value: float
    target_entity: Optional[str] = None
    priority: int = 5                         # 1 = highest, 10 = lowest
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    achieved_at: Optional[float] = None
    standing: bool = False                    # True = auto-reset, never user-deletable
    reset_after_seconds: float = 300.0        # cooldown before standing goal re-arms
    progress_history: List[Tuple[float, float]] = field(default_factory=list)
    # internal timestamp used for standing-rule reset cooldown
    _last_achieved_ts: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    def record_progress(self, value: float) -> None:
        self.progress_history.append((time.time(), value))
        if len(self.progress_history) > _HISTORY_CAP:
            self.progress_history = self.progress_history[-_HISTORY_CAP:]


# ---------------------------------------------------------------------------
# Standing homeostasis rules
# ---------------------------------------------------------------------------

def _make_standing_rules() -> List[Goal]:
    return [
        Goal(
            id="standing_arousal_cap",
            description="Keep arousal below 2.0 (suppress inference when over-excited)",
            metric_type=GoalMetricType.AROUSAL_BELOW,
            target_value=2.0,
            priority=1,
            standing=True,
            reset_after_seconds=120.0,
        ),
        Goal(
            id="standing_soliton_health",
            description="Maintain soliton_index mean ≥ 0.75 (stable predictions)",
            metric_type=GoalMetricType.SOLITON_INDEX,
            target_value=0.75,
            priority=2,
            standing=True,
            reset_after_seconds=300.0,
        ),
        Goal(
            id="standing_reinforcement_floor",
            description="Keep reinforcement ≥ 0.5 (positive feedback baseline)",
            metric_type=GoalMetricType.REINFORCEMENT,
            target_value=0.5,
            priority=3,
            standing=True,
            reset_after_seconds=300.0,
        ),
        Goal(
            id="standing_discovery_quality",
            description="Keep approval_rate ≥ 0.15 (research quality gate)",
            metric_type=GoalMetricType.APPROVAL_RATE,
            target_value=0.15,
            priority=4,
            standing=True,
            reset_after_seconds=600.0,
        ),
    ]


# ---------------------------------------------------------------------------
# GoalStack
# ---------------------------------------------------------------------------

class GoalStack:
    """Priority queue of Goals with built-in homeostasis standing rules."""

    def __init__(self) -> None:
        self._standing_rules: List[Goal] = _make_standing_rules()
        self._user_goals: Dict[str, Goal] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, goal: Goal) -> None:
        if goal.standing:
            raise ValueError("Cannot push a standing goal via push(); standing rules are immutable.")
        self._user_goals[goal.id] = goal

    def abandon(self, goal_id: str) -> bool:
        for g in self._standing_rules:
            if g.id == goal_id:
                raise ValueError(f"Standing goal '{goal_id}' cannot be abandoned.")
        goal = self._user_goals.get(goal_id)
        if goal is None:
            return False
        goal.status = GoalStatus.ABANDONED
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def top(self) -> Optional[Goal]:
        """Highest-priority active goal across standing + user goals."""
        active = self.all_active()
        return active[0] if active else None

    def all_active(self) -> List[Goal]:
        """All active goals sorted by priority (ascending = highest first)."""
        goals = [g for g in self._standing_rules if g.status == GoalStatus.ACTIVE]
        goals += [g for g in self._user_goals.values() if g.status == GoalStatus.ACTIVE]
        goals.sort(key=lambda g: g.priority)
        return goals

    def all_goals(self) -> List[Goal]:
        """All goals including achieved/abandoned."""
        return list(self._standing_rules) + list(self._user_goals.values())

    def get(self, goal_id: str) -> Optional[Goal]:
        for g in self._standing_rules:
            if g.id == goal_id:
                return g
        return self._user_goals.get(goal_id)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": g.id,
                "description": g.description,
                "metric_type": g.metric_type.value,
                "target_value": g.target_value,
                "target_entity": g.target_entity,
                "priority": g.priority,
                "status": g.status.value,
                "standing": g.standing,
                "created_at": g.created_at,
                "achieved_at": g.achieved_at,
                "progress_history": g.progress_history[-20:],  # last 20 for serialization
            }
            for g in self.all_goals()
        ]


# ---------------------------------------------------------------------------
# GoalEvaluator
# ---------------------------------------------------------------------------

class GoalEvaluator:
    """Measures goal metrics, evaluates achievement, and drives inference direction."""

    def __init__(self, graph: Any, loop: Any, wm: Optional["WorkingMemoryBuffer"] = None) -> None:
        self._graph = graph
        self._loop = loop
        self._wm = wm

    # ------------------------------------------------------------------
    # Metric measurement
    # ------------------------------------------------------------------

    def measure(self, goal: Goal) -> float:
        """Return the current value of a goal's tracked metric."""
        try:
            if goal.metric_type == GoalMetricType.SOLITON_INDEX:
                pc = getattr(self._graph, "predictive_coder", None)
                if pc is None:
                    return 0.0
                stats = pc.soliton_stats() if callable(getattr(pc, "soliton_stats", None)) else {}
                return float(sum(stats.values()) / len(stats)) if stats else 0.0

            elif goal.metric_type == GoalMetricType.APPROVAL_RATE:
                status = self._loop.status() if callable(getattr(self._loop, "status", None)) else {}
                return float(status.get("approval_rate", 0.0) or 0.0)

            elif goal.metric_type == GoalMetricType.REINFORCEMENT:
                mod = getattr(self._graph, "modulator", None)
                if mod is None:
                    return 0.0
                state = getattr(mod, "state", {})
                return float(state.get("reinforcement", 0.0) if isinstance(state, dict) else 0.0)

            elif goal.metric_type == GoalMetricType.ENTITY_DISCOVERED:
                if goal.target_entity is None:
                    return 0.0
                adapter = getattr(self._graph, "adapter", None)
                if adapter is None:
                    return 0.0
                try:
                    g = adapter.to_networkx()
                    return 1.0 if goal.target_entity in g.nodes else 0.0
                except Exception:
                    return 0.0

            elif goal.metric_type == GoalMetricType.AROUSAL_BELOW:
                mod = getattr(self._graph, "modulator", None)
                if mod is None:
                    return 0.0
                state = getattr(mod, "state", {})
                return float(state.get("arousal", 0.0) if isinstance(state, dict) else 0.0)

        except Exception as exc:
            logger.warning("GoalEvaluator.measure failed for %s: %s", goal.id, exc)
        return 0.0

    def _is_achieved(self, goal: Goal, value: float) -> bool:
        if goal.metric_type == GoalMetricType.AROUSAL_BELOW:
            return value <= goal.target_value
        if goal.metric_type == GoalMetricType.ENTITY_DISCOVERED:
            return value >= 1.0
        return value >= goal.target_value

    # ------------------------------------------------------------------
    # Evaluate all goals for one loop cycle
    # ------------------------------------------------------------------

    def evaluate(self, stack: GoalStack) -> bool:
        """Evaluate all active goals and reset eligible standing rules.

        Returns True if active inference should be suppressed (AROUSAL_CAP breached).
        Emits GOAL_UPDATE events on status transitions.
        """
        suppress_inference = False

        for goal in stack.all_goals():
            # Re-arm standing goals after cooldown
            if goal.standing and goal.status == GoalStatus.ACHIEVED:
                if goal._last_achieved_ts is not None:
                    if time.time() - goal._last_achieved_ts >= goal.reset_after_seconds:
                        goal.status = GoalStatus.ACTIVE
                        goal._last_achieved_ts = None
                        self._emit_goal_update(goal, 0.0)
                continue

            if goal.status != GoalStatus.ACTIVE:
                continue

            value = self.measure(goal)
            goal.record_progress(value)

            if self._is_achieved(goal, value):
                goal.status = GoalStatus.ACHIEVED
                goal.achieved_at = time.time()
                goal._last_achieved_ts = goal.achieved_at
                self._emit_goal_update(goal, value)
                logger.info("Goal achieved: %s (value=%.3f, target=%.3f)", goal.id, value, goal.target_value)
            else:
                # Check if AROUSAL_CAP is breached
                if goal.id == "standing_arousal_cap":
                    suppress_inference = True

        return suppress_inference

    # ------------------------------------------------------------------
    # Context seed selection for goal-directed inference
    # ------------------------------------------------------------------

    def get_context_seeds(
        self,
        stack: GoalStack,
        wm: Optional["WorkingMemoryBuffer"] = None,
    ) -> Optional[List[str]]:
        """Return up to 3 entity seeds to direct active inference toward the top active goal."""
        top = stack.top()
        if top is None:
            return None

        try:
            if top.metric_type == GoalMetricType.ENTITY_DISCOVERED and top.target_entity:
                return [top.target_entity]

            elif top.metric_type == GoalMetricType.SOLITON_INDEX:
                pc = getattr(self._graph, "predictive_coder", None)
                if pc is None:
                    return None
                stats = pc.soliton_stats() if callable(getattr(pc, "soliton_stats", None)) else {}
                if not stats:
                    return None
                # Lowest soliton values = most dissonant = highest PE
                sorted_nodes = sorted(stats.items(), key=lambda kv: kv[1])
                return [node for node, _ in sorted_nodes[:3]]

            elif top.metric_type == GoalMetricType.REINFORCEMENT:
                # Seeds from recent WM entries with high scores
                buf = wm or self._wm
                if buf is None:
                    return None
                recent = buf.recent(10)
                high_score = [e for e in recent if e.top_score >= 0.5]
                seeds = [s for e in high_score for s in e.seeds][:3]
                return seeds or None

        except Exception as exc:
            logger.warning("GoalEvaluator.get_context_seeds failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_goal_update(self, goal: Goal, value: float) -> None:
        try:
            from core.telemetry import NeuralEvent
            event = NeuralEvent.goal_update(
                goal_id=goal.id,
                status=goal.status.value,
                metric_type=goal.metric_type.value,
                metric_value=value,
                target_value=goal.target_value,
            )
            if hasattr(self._graph, "emit"):
                self._graph.emit(event)
        except Exception as exc:
            logger.warning("GoalEvaluator._emit_goal_update failed: %s", exc)


def make_goal(
    description: str,
    metric_type: str,
    target_value: float,
    target_entity: Optional[str] = None,
    priority: int = 5,
) -> Goal:
    """Convenience factory for user-created goals."""
    return Goal(
        id=uuid.uuid4().hex,
        description=description,
        metric_type=GoalMetricType(metric_type),
        target_value=target_value,
        target_entity=target_entity,
        priority=priority,
    )

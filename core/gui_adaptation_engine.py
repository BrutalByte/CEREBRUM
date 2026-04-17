"""
GUIAdaptationEngine — Phase 94: Self-Modifying GUI.

Watches CEREBRUM's metabolic and loop signals and applies structural changes
to the UE5 HUD widget via the ue-llm-toolkit HTTP API.  Also emits
GUI_ADAPTATION telemetry events so the already-running widget can show/hide
panels at runtime without a Blueprint recompile.

Two distinct operations:
  - Structural: toolkit HTTP call → modifies/saves the Blueprint asset
  - Runtime:    emit NeuralEvent(GUI_ADAPTATION) → handled by CerebrumBrain
                via the existing WebSocket channel
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("cerebrum.gui_adaptation")


# ---------------------------------------------------------------------------
# Signal snapshot
# ---------------------------------------------------------------------------

@dataclass
class SignalSnapshot:
    timestamp: float
    arousal: float
    reinforcement: float
    soliton_index: Optional[float] = None
    approval_rate: Optional[float] = None
    circuit_breaker_tripped: bool = False
    total_inference_pulses: int = 0


# ---------------------------------------------------------------------------
# Adaptation action
# ---------------------------------------------------------------------------

@dataclass
class AdaptationAction:
    rule_id: str
    structural: bool          # True → toolkit edit; False → runtime event only
    toolkit_op: Optional[str] = None
    toolkit_params: Optional[Dict[str, Any]] = None
    event_action: Optional[str] = None   # "show" | "hide" | "collapse" | "update"
    event_target: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class AdaptationRule:
    rule_id: str
    check: Callable[["GUIAdaptationEngine"], bool]
    actions: List[AdaptationAction]
    once: bool = True    # if True, only fires once per lifetime; reset on recovery


# ---------------------------------------------------------------------------
# GUIAdaptationEngine
# ---------------------------------------------------------------------------

class GUIAdaptationEngine:
    """
    Signal-watching engine that structurally and dynamically adapts the UE5
    HUD widget based on CEREBRUM's internal state.
    """

    WIDGET_PATH = "/Game/UI/WBP_CerebrumHUD"

    def __init__(
        self,
        toolkit,           # UEToolkitClient (or None for toolkit-less mode)
        emit_fn: Optional[Callable] = None,  # CerebrumGraph.emit or similar
        widget_path: str = WIDGET_PATH,
        history_maxlen: int = 50,
    ) -> None:
        self._toolkit = toolkit
        self._emit_fn = emit_fn
        self._widget_path = widget_path
        self._history: deque[SignalSnapshot] = deque(maxlen=history_maxlen)
        self._applied: Set[str] = set()
        self._rules: List[AdaptationRule] = _build_default_rules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, snapshot: SignalSnapshot) -> None:
        self._history.append(snapshot)

    def step(self) -> List[str]:
        """Evaluate all rules; apply triggered actions. Returns list of fired rule IDs."""
        fired: List[str] = []
        for rule in self._rules:
            if rule.once and rule.rule_id in self._applied:
                continue
            try:
                if rule.check(self):
                    self._apply_rule(rule)
                    fired.append(rule.rule_id)
                    if rule.once:
                        self._applied.add(rule.rule_id)
            except Exception:
                logger.exception("GUIAdaptationEngine: rule %s evaluation failed.", rule.rule_id)
        return fired

    def reset_rule(self, rule_id: str) -> None:
        self._applied.discard(rule_id)

    @property
    def history(self) -> List[SignalSnapshot]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_rule(self, rule: AdaptationRule) -> None:
        for action in rule.actions:
            # 1. Structural toolkit operation
            if action.structural and self._toolkit is not None:
                try:
                    if action.toolkit_op == "add_widget_element":
                        p = action.toolkit_params or {}
                        self._toolkit.add_widget_element(
                            self._widget_path,
                            p.get("element_type", "TextBlock"),
                            p.get("element_name", action.rule_id),
                            p.get("properties"),
                        )
                        self._toolkit.compile_blueprint(self._widget_path)
                        self._toolkit.save_all()
                        logger.info("GUIAdaptation: structural change applied for rule %s", rule.rule_id)
                    elif action.toolkit_op == "set_widget_property":
                        p = action.toolkit_params or {}
                        self._toolkit.set_widget_property(
                            self._widget_path,
                            p.get("element_name", ""),
                            p.get("property_name", ""),
                            p.get("value"),
                        )
                except Exception:
                    logger.exception("GUIAdaptation: toolkit op failed for rule %s", rule.rule_id)

            # 2. Runtime event via telemetry channel
            if action.event_action and self._emit_fn:
                try:
                    from core.telemetry import NeuralEvent
                    event = NeuralEvent.gui_adapt(
                        action=action.event_action,
                        target=action.event_target or "",
                        data=action.event_data or {},
                    )
                    self._emit_fn(event)
                except Exception:
                    logger.exception("GUIAdaptation: emit failed for rule %s", rule.rule_id)

    # ------------------------------------------------------------------
    # Helpers for rule conditions
    # ------------------------------------------------------------------

    def _consecutive_above(self, key: str, threshold: float, n: int) -> bool:
        snap = list(self._history)
        if len(snap) < n:
            return False
        return all(getattr(s, key, 0.0) > threshold for s in snap[-n:])

    def _consecutive_below(self, key: str, threshold: float, n: int) -> bool:
        snap = list(self._history)
        if len(snap) < n:
            return False
        return all(getattr(s, key, 1.0) < threshold for s in snap[-n:])

    def _latest(self, key: str, default: Any = None) -> Any:
        if not self._history:
            return default
        return getattr(self._history[-1], key, default)

    def _was_tripped_then_cleared(self) -> bool:
        snap = list(self._history)
        if len(snap) < 2:
            return False
        prev_tripped = any(s.circuit_breaker_tripped for s in snap[:-1])
        return prev_tripped and not snap[-1].circuit_breaker_tripped


# ---------------------------------------------------------------------------
# Default rule set
# ---------------------------------------------------------------------------

def _build_default_rules() -> List[AdaptationRule]:
    return [
        # Rule 1: Sustained high arousal → add DissonanceMeter panel
        AdaptationRule(
            rule_id="HIGH_AROUSAL",
            check=lambda e: e._consecutive_above("arousal", 2.0, 5),
            actions=[
                AdaptationAction(
                    rule_id="HIGH_AROUSAL",
                    structural=True,
                    toolkit_op="add_widget_element",
                    toolkit_params={
                        "element_type": "ProgressBar",
                        "element_name": "DissonanceMeter",
                        "properties": {
                            "anchor_min": [0.35, 0.0],
                            "anchor_max": [0.65, 0.0],
                            "offset": [0.0, 60.0, 0.0, 20.0],
                            "fill_color": {"r": 1.0, "g": 0.35, "b": 0.05, "a": 1.0},
                        },
                    },
                    event_action="show",
                    event_target="dissonance_meter",
                    event_data={"reason": "sustained_high_arousal"},
                )
            ],
        ),
        # Rule 2: Unstable prior → show relearning badge
        AdaptationRule(
            rule_id="UNSTABLE_PRIOR",
            check=lambda e: e._consecutive_below("soliton_index", 0.3, 3),
            actions=[
                AdaptationAction(
                    rule_id="UNSTABLE_PRIOR",
                    structural=True,
                    toolkit_op="add_widget_element",
                    toolkit_params={
                        "element_type": "TextBlock",
                        "element_name": "RelearningBadge",
                        "properties": {
                            "text": "⟳ Relearning",
                            "font_size": 14,
                            "color": {"r": 1.0, "g": 0.8, "b": 0.0, "a": 1.0},
                            "anchor_min": [0.5, 0.0],
                            "anchor_max": [0.5, 0.0],
                        },
                    },
                    event_action="show",
                    event_target="relearning_badge",
                    event_data={"soliton_index": None},
                )
            ],
        ),
        # Rule 3: Circuit breaker tripped → show warning banner
        AdaptationRule(
            rule_id="CIRCUIT_BREAKER",
            once=False,  # re-fires every cycle it's tripped (for runtime show)
            check=lambda e: bool(e._latest("circuit_breaker_tripped", False)),
            actions=[
                AdaptationAction(
                    rule_id="CIRCUIT_BREAKER",
                    structural=False,
                    event_action="show",
                    event_target="circuit_warning",
                    event_data={"approval_rate": None},
                )
            ],
        ),
        # Rule 4: Circuit breaker cleared → hide warning banner
        AdaptationRule(
            rule_id="CIRCUIT_RECOVERY",
            once=False,
            check=lambda e: e._was_tripped_then_cleared(),
            actions=[
                AdaptationAction(
                    rule_id="CIRCUIT_RECOVERY",
                    structural=False,
                    event_action="hide",
                    event_target="circuit_warning",
                    event_data={},
                )
            ],
        ),
        # Rule 5: Inference milestone (50 pulses) → add inference history panel
        AdaptationRule(
            rule_id="INFERENCE_MILESTONE_50",
            check=lambda e: (e._latest("total_inference_pulses", 0) or 0) >= 50,
            actions=[
                AdaptationAction(
                    rule_id="INFERENCE_MILESTONE_50",
                    structural=True,
                    toolkit_op="add_widget_element",
                    toolkit_params={
                        "element_type": "ScrollBox",
                        "element_name": "InferenceHistoryBox",
                        "properties": {
                            "anchor_min": [0.0, 0.5],
                            "anchor_max": [0.2, 1.0],
                            "offset": [8.0, 8.0, -8.0, -8.0],
                        },
                    },
                    event_action="show",
                    event_target="inference_history_box",
                    event_data={"milestone": 50},
                )
            ],
        ),
        # Rule 6: Low reinforcement sustained → collapse active inference panel
        AdaptationRule(
            rule_id="LOW_REINFORCEMENT",
            check=lambda e: e._consecutive_below("reinforcement", 0.2, 10),
            actions=[
                AdaptationAction(
                    rule_id="LOW_REINFORCEMENT",
                    structural=False,
                    event_action="collapse",
                    event_target="active_inference_panel",
                    event_data={"reason": "low_reinforcement"},
                )
            ],
        ),
    ]

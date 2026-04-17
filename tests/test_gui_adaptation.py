"""Tests for Phase 94: GUIAdaptationEngine."""
import time
import pytest
from unittest.mock import MagicMock, patch
from core.gui_adaptation_engine import GUIAdaptationEngine, SignalSnapshot


def _make_engine(toolkit=None, emit_fn=None, history_maxlen=50):
    if toolkit is None:
        toolkit = MagicMock()
        toolkit.add_widget_element.return_value = True
        toolkit.compile_blueprint.return_value = True
        toolkit.save_all.return_value = True
        toolkit.set_widget_property.return_value = True
    return GUIAdaptationEngine(toolkit=toolkit, emit_fn=emit_fn, history_maxlen=history_maxlen)


def _snap(arousal=1.0, reinforcement=1.0, soliton_index=None,
          approval_rate=None, circuit_breaker_tripped=False, pulses=0):
    return SignalSnapshot(
        timestamp=time.time(),
        arousal=arousal,
        reinforcement=reinforcement,
        soliton_index=soliton_index,
        approval_rate=approval_rate,
        circuit_breaker_tripped=circuit_breaker_tripped,
        total_inference_pulses=pulses,
    )


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_engine_init():
    engine = _make_engine()
    assert engine.history == []
    assert engine._applied == set()


def test_record_snapshot():
    engine = _make_engine()
    s = _snap(arousal=1.5)
    engine.record(s)
    assert engine.history == [s]


def test_signal_history_window():
    engine = _make_engine(history_maxlen=3)
    for i in range(5):
        engine.record(_snap(arousal=float(i)))
    assert len(engine.history) == 3
    assert engine.history[-1].arousal == 4.0


# ---------------------------------------------------------------------------
# Rule: no toolkit (graceful degradation)
# ---------------------------------------------------------------------------

def test_rule_no_toolkit():
    """Engine with None toolkit still evaluates rules without raising."""
    engine = GUIAdaptationEngine(toolkit=None, emit_fn=None)
    for _ in range(6):
        engine.record(_snap(arousal=2.5))
    fired = engine.step()
    assert "HIGH_AROUSAL" in fired  # rule still fires


def test_toolkit_unavailable_skips_structural():
    """Structural action is skipped when toolkit is None; no exception."""
    engine = GUIAdaptationEngine(toolkit=None, emit_fn=None)
    for _ in range(6):
        engine.record(_snap(arousal=2.5))
    engine.step()  # must not raise


# ---------------------------------------------------------------------------
# Rule: HIGH_AROUSAL
# ---------------------------------------------------------------------------

def test_high_arousal_rule_not_enough_history():
    engine = _make_engine()
    for _ in range(3):  # only 3, need 5
        engine.record(_snap(arousal=2.5))
    fired = engine.step()
    assert "HIGH_AROUSAL" not in fired


def test_high_arousal_rule_fires_after_5():
    toolkit = MagicMock()
    toolkit.add_widget_element.return_value = True
    toolkit.compile_blueprint.return_value = True
    toolkit.save_all.return_value = True
    engine = _make_engine(toolkit=toolkit)
    for _ in range(5):
        engine.record(_snap(arousal=2.1))
    fired = engine.step()
    assert "HIGH_AROUSAL" in fired
    toolkit.add_widget_element.assert_called_once()
    toolkit.compile_blueprint.assert_called_once()


def test_high_arousal_rule_idempotent():
    toolkit = MagicMock()
    toolkit.add_widget_element.return_value = True
    toolkit.compile_blueprint.return_value = True
    toolkit.save_all.return_value = True
    engine = _make_engine(toolkit=toolkit)
    for _ in range(5):
        engine.record(_snap(arousal=2.5))
    engine.step()
    # Feed more high-arousal snapshots
    for _ in range(5):
        engine.record(_snap(arousal=2.5))
    engine.step()
    # Still only called once (rule is once=True)
    assert toolkit.add_widget_element.call_count == 1


# ---------------------------------------------------------------------------
# Rule: CIRCUIT_BREAKER
# ---------------------------------------------------------------------------

def test_circuit_breaker_rule_fires():
    emit_calls = []
    engine = _make_engine(emit_fn=lambda e: emit_calls.append(e))
    engine.record(_snap(circuit_breaker_tripped=True))
    fired = engine.step()
    assert "CIRCUIT_BREAKER" in fired
    assert len(emit_calls) == 1
    assert emit_calls[0].payload["action"] == "show"
    assert emit_calls[0].payload["target"] == "circuit_warning"


def test_circuit_breaker_repeats_while_tripped():
    """CIRCUIT_BREAKER is once=False — fires each cycle while tripped."""
    emit_calls = []
    engine = _make_engine(emit_fn=lambda e: emit_calls.append(e))
    for _ in range(3):
        engine.record(_snap(circuit_breaker_tripped=True))
        engine.step()
    assert len(emit_calls) == 3


# ---------------------------------------------------------------------------
# Rule: CIRCUIT_RECOVERY
# ---------------------------------------------------------------------------

def test_circuit_recovery_rule():
    emit_calls = []
    engine = _make_engine(emit_fn=lambda e: emit_calls.append(e))
    engine.record(_snap(circuit_breaker_tripped=True))
    engine.record(_snap(circuit_breaker_tripped=False))
    engine.step()
    targets = [e.payload["target"] for e in emit_calls]
    assert "circuit_warning" in targets
    hide_events = [e for e in emit_calls if e.payload["action"] == "hide"]
    assert hide_events


# ---------------------------------------------------------------------------
# Rule: UNSTABLE_PRIOR
# ---------------------------------------------------------------------------

def test_unstable_prior_rule_fires():
    engine = _make_engine()
    for _ in range(3):
        engine.record(_snap(soliton_index=0.2))
    fired = engine.step()
    assert "UNSTABLE_PRIOR" in fired


def test_unstable_prior_not_enough_snaps():
    engine = _make_engine()
    for _ in range(2):
        engine.record(_snap(soliton_index=0.1))
    fired = engine.step()
    assert "UNSTABLE_PRIOR" not in fired


# ---------------------------------------------------------------------------
# Rule: INFERENCE_MILESTONE
# ---------------------------------------------------------------------------

def test_inference_milestone_fires_at_50():
    engine = _make_engine()
    engine.record(_snap(pulses=50))
    fired = engine.step()
    assert "INFERENCE_MILESTONE_50" in fired


def test_inference_milestone_does_not_fire_below_50():
    engine = _make_engine()
    engine.record(_snap(pulses=49))
    fired = engine.step()
    assert "INFERENCE_MILESTONE_50" not in fired


# ---------------------------------------------------------------------------
# Rule: LOW_REINFORCEMENT
# ---------------------------------------------------------------------------

def test_low_reinforcement_fires_after_10():
    emit_calls = []
    engine = _make_engine(emit_fn=lambda e: emit_calls.append(e))
    for _ in range(10):
        engine.record(_snap(reinforcement=0.1))
    fired = engine.step()
    assert "LOW_REINFORCEMENT" in fired
    collapse = [e for e in emit_calls if e.payload["action"] == "collapse"]
    assert collapse


def test_low_reinforcement_needs_10_consecutive():
    engine = _make_engine()
    for i in range(9):
        engine.record(_snap(reinforcement=0.1))
    fired = engine.step()
    assert "LOW_REINFORCEMENT" not in fired


# ---------------------------------------------------------------------------
# Emit integration
# ---------------------------------------------------------------------------

def test_emit_fn_called_on_runtime_event():
    from core.telemetry import NeuralEventType
    events = []
    engine = _make_engine(emit_fn=lambda e: events.append(e))
    engine.record(_snap(circuit_breaker_tripped=True))
    engine.step()
    assert events
    assert events[0].event_type == NeuralEventType.GUI_ADAPTATION


def test_reset_rule_allows_refire():
    toolkit = MagicMock()
    toolkit.add_widget_element.return_value = True
    toolkit.compile_blueprint.return_value = True
    toolkit.save_all.return_value = True
    engine = _make_engine(toolkit=toolkit)
    for _ in range(5):
        engine.record(_snap(arousal=2.5))
    engine.step()
    assert toolkit.add_widget_element.call_count == 1

    engine.reset_rule("HIGH_AROUSAL")
    for _ in range(5):
        engine.record(_snap(arousal=2.5))
    engine.step()
    assert toolkit.add_widget_element.call_count == 2

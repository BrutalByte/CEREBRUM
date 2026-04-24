"""Tests for EpistemicGate (Phase 122)."""
import time
import pytest
from unittest.mock import MagicMock

from core.epistemic_gate import EpistemicGate, GateConfig, GateDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(eu: float, ciu: float) -> MagicMock:
    s = MagicMock()
    s.epistemic_uncertainty = eu
    s.confidence_in_uncertainty = ciu
    return s


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------

def test_gate_config_defaults():
    cfg = GateConfig()
    assert cfg.suppress_threshold == 0.75
    assert cfg.research_threshold == 0.70
    assert cfg.sleep_threshold == 0.80
    assert cfg.credence_threshold == 0.30
    assert cfg.research_cooldown == 60.0
    assert cfg.enabled is True


def test_gate_config_roundtrip():
    cfg = GateConfig(suppress_threshold=0.60, enabled=False)
    d = cfg.to_dict()
    restored = GateConfig.from_dict(d)
    assert restored.suppress_threshold == 0.60
    assert restored.enabled is False


def test_gate_config_from_dict_ignores_unknown_keys():
    cfg = GateConfig.from_dict({"suppress_threshold": 0.5, "unknown_key": 99})
    assert cfg.suppress_threshold == 0.5


# ---------------------------------------------------------------------------
# Disabled gate
# ---------------------------------------------------------------------------

def test_gate_disabled_returns_passthrough():
    gate = EpistemicGate(GateConfig(enabled=False))
    decision = gate.evaluate(make_state(eu=0.99, ciu=0.01))
    assert decision.low_confidence is False
    assert decision.triggered_research is False
    assert decision.triggered_sleep is False
    assert decision.epistemic_warning is None
    assert any("disabled" in a for a in decision.action_log)


# ---------------------------------------------------------------------------
# Suppress threshold
# ---------------------------------------------------------------------------

def test_low_confidence_above_threshold():
    gate = EpistemicGate(GateConfig(suppress_threshold=0.75))
    decision = gate.evaluate(make_state(eu=0.80, ciu=0.5))
    assert decision.low_confidence is True
    assert decision.eu == pytest.approx(0.80)


def test_low_confidence_below_threshold():
    gate = EpistemicGate(GateConfig(suppress_threshold=0.75))
    decision = gate.evaluate(make_state(eu=0.70, ciu=0.5))
    assert decision.low_confidence is False


def test_low_confidence_exactly_at_threshold():
    gate = EpistemicGate(GateConfig(suppress_threshold=0.75))
    decision = gate.evaluate(make_state(eu=0.75, ciu=0.5))
    assert decision.low_confidence is True


# ---------------------------------------------------------------------------
# Credence warning
# ---------------------------------------------------------------------------

def test_credence_warning_below_threshold():
    gate = EpistemicGate(GateConfig(credence_threshold=0.30))
    decision = gate.evaluate(make_state(eu=0.40, ciu=0.20))
    assert decision.epistemic_warning is not None
    assert "CIU" in decision.epistemic_warning


def test_no_credence_warning_above_threshold():
    gate = EpistemicGate(GateConfig(credence_threshold=0.30))
    decision = gate.evaluate(make_state(eu=0.40, ciu=0.50))
    assert decision.epistemic_warning is None


# ---------------------------------------------------------------------------
# Research trigger
# ---------------------------------------------------------------------------

def test_research_triggered_above_threshold():
    gate = EpistemicGate(GateConfig(research_threshold=0.70, research_cooldown=0.0))
    decision = gate.evaluate(make_state(eu=0.75, ciu=0.5))
    assert decision.triggered_research is True


def test_research_not_triggered_below_threshold():
    gate = EpistemicGate(GateConfig(research_threshold=0.70))
    decision = gate.evaluate(make_state(eu=0.65, ciu=0.5))
    assert decision.triggered_research is False


def test_research_cooldown_blocks_second_trigger():
    gate = EpistemicGate(GateConfig(research_threshold=0.70, research_cooldown=60.0))
    d1 = gate.evaluate(make_state(eu=0.80, ciu=0.5))
    d2 = gate.evaluate(make_state(eu=0.80, ciu=0.5))
    assert d1.triggered_research is True
    assert d2.triggered_research is False
    assert any("cooldown" in a for a in d2.action_log)


def test_research_cooldown_zero_allows_repeat():
    gate = EpistemicGate(GateConfig(research_threshold=0.70, research_cooldown=0.0))
    d1 = gate.evaluate(make_state(eu=0.80, ciu=0.5))
    d2 = gate.evaluate(make_state(eu=0.80, ciu=0.5))
    assert d1.triggered_research is True
    assert d2.triggered_research is True


# ---------------------------------------------------------------------------
# Sleep trigger
# ---------------------------------------------------------------------------

def test_sleep_triggered_above_threshold():
    gate = EpistemicGate(GateConfig(sleep_threshold=0.80))
    decision = gate.evaluate(make_state(eu=0.85, ciu=0.5))
    assert decision.triggered_sleep is True


def test_sleep_not_triggered_below_threshold():
    gate = EpistemicGate(GateConfig(sleep_threshold=0.80))
    decision = gate.evaluate(make_state(eu=0.75, ciu=0.5))
    assert decision.triggered_sleep is False


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------

def test_action_log_records_all_decisions():
    gate = EpistemicGate(GateConfig(
        suppress_threshold=0.70,
        research_threshold=0.70,
        sleep_threshold=0.70,
        credence_threshold=0.99,
        research_cooldown=0.0,
    ))
    decision = gate.evaluate(make_state(eu=0.75, ciu=0.10))
    log_text = " ".join(decision.action_log)
    assert "suppress" in log_text
    assert "research" in log_text
    assert "sleep" in log_text
    assert "warn" in log_text


# ---------------------------------------------------------------------------
# to_dict / GateDecision.to_dict
# ---------------------------------------------------------------------------

def test_gate_decision_to_dict():
    gate = EpistemicGate(GateConfig(suppress_threshold=0.50, research_cooldown=0.0))
    decision = gate.evaluate(make_state(eu=0.60, ciu=0.40))
    d = decision.to_dict()
    assert "eu" in d
    assert "low_confidence" in d
    assert "action_log" in d
    assert isinstance(d["action_log"], list)


def test_gate_to_dict():
    gate = EpistemicGate()
    d = gate.to_dict()
    assert "config" in d
    assert "last_research_triggered_ago_seconds" in d


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------

def test_update_config_partial():
    gate = EpistemicGate()
    gate.update_config(suppress_threshold=0.50, enabled=False)
    assert gate.config.suppress_threshold == 0.50
    assert gate.config.enabled is False
    # Other fields unchanged
    assert gate.config.research_threshold == 0.70


def test_update_config_ignores_unknown():
    gate = EpistemicGate()
    gate.update_config(nonexistent_field=123)
    assert gate.config.suppress_threshold == 0.75  # unchanged


# ---------------------------------------------------------------------------
# Auto-wiring via CerebrumGraph.attach_metacognitive_monitor
# ---------------------------------------------------------------------------

def test_auto_wire_injects_available_engines():
    from core.metacognitive_monitor import MetacognitiveMonitor
    from core.cerebrum import CerebrumGraph
    from unittest.mock import MagicMock, patch

    graph = CerebrumGraph.__new__(CerebrumGraph)
    graph._built = False
    graph._metacognitive_monitor = None
    graph._epistemic_gate = None
    graph._working_memory = MagicMock()

    mock_pc = MagicMock()
    mock_modulator = MagicMock()
    graph.predictive_coder = mock_pc
    graph.modulator = mock_modulator

    monitor = MetacognitiveMonitor()
    assert monitor.pc is None
    assert monitor.cm is None
    assert monitor.wm is None

    graph.attach_metacognitive_monitor(monitor)

    assert monitor.pc is mock_pc
    assert monitor.cm is mock_modulator
    assert monitor.wm is graph._working_memory


def test_auto_wire_does_not_overwrite_existing():
    from core.metacognitive_monitor import MetacognitiveMonitor
    from core.cerebrum import CerebrumGraph

    graph = CerebrumGraph.__new__(CerebrumGraph)
    graph._built = False
    graph._metacognitive_monitor = None
    graph._epistemic_gate = None
    graph._working_memory = None

    existing_pc = MagicMock()
    graph.predictive_coder = MagicMock()
    graph.modulator = MagicMock()

    monitor = MetacognitiveMonitor(predictive_coder=existing_pc)
    graph.attach_metacognitive_monitor(monitor)

    # Should not be overwritten
    assert monitor.pc is existing_pc


# ---------------------------------------------------------------------------
# attach_epistemic_gate
# ---------------------------------------------------------------------------

def test_attach_epistemic_gate():
    from core.cerebrum import CerebrumGraph

    graph = CerebrumGraph.__new__(CerebrumGraph)
    graph._built = False
    graph._epistemic_gate = None

    gate = EpistemicGate()
    graph.attach_epistemic_gate(gate)
    assert graph._epistemic_gate is gate

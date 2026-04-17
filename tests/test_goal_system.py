"""Tests for core/goal_system.py (Phase 95)."""
import time
from unittest.mock import MagicMock, patch

import pytest

from core.goal_system import (
    Goal,
    GoalEvaluator,
    GoalMetricType,
    GoalStack,
    GoalStatus,
    make_goal,
)
from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Goal dataclass
# ---------------------------------------------------------------------------

def test_goal_create_active():
    g = make_goal("test", "reinforcement", 0.5)
    assert g.status == GoalStatus.ACTIVE
    assert g.standing is False
    assert g.achieved_at is None
    assert g.progress_history == []


def test_goal_record_progress_cap():
    g = make_goal("test", "reinforcement", 0.5)
    for i in range(250):
        g.record_progress(float(i))
    assert len(g.progress_history) == 200
    # should retain the last 200
    assert g.progress_history[-1][1] == 249.0


# ---------------------------------------------------------------------------
# GoalStack
# ---------------------------------------------------------------------------

def test_standing_rules_present():
    stack = GoalStack()
    rules = stack._standing_rules
    assert len(rules) == 4
    ids = {r.id for r in rules}
    assert "standing_arousal_cap" in ids
    assert "standing_soliton_health" in ids
    assert "standing_reinforcement_floor" in ids
    assert "standing_discovery_quality" in ids
    assert all(r.standing for r in rules)


def test_goal_stack_priority():
    stack = GoalStack()
    high = make_goal("high", "reinforcement", 0.5, priority=1)
    low = make_goal("low", "reinforcement", 0.5, priority=8)
    stack.push(low)
    stack.push(high)
    # standing rules have priority 1-4; user high also has priority 1
    # top() should return priority=1 goal (could be arousal_cap or user high)
    top = stack.top()
    assert top is not None
    assert top.priority == 1


def test_goal_stack_all_active_sorted():
    stack = GoalStack()
    g5 = make_goal("mid", "reinforcement", 0.5, priority=5)
    g2 = make_goal("high", "reinforcement", 0.5, priority=2)
    stack.push(g5)
    stack.push(g2)
    active = stack.all_active()
    priorities = [g.priority for g in active]
    assert priorities == sorted(priorities)


def test_abandon_user_goal():
    stack = GoalStack()
    g = make_goal("drop", "reinforcement", 0.5)
    stack.push(g)
    result = stack.abandon(g.id)
    assert result is True
    assert stack.get(g.id).status == GoalStatus.ABANDONED
    assert g not in stack.all_active()


def test_standing_rule_not_abandonable():
    stack = GoalStack()
    with pytest.raises(ValueError):
        stack.abandon("standing_arousal_cap")


def test_get_unknown_goal():
    stack = GoalStack()
    assert stack.get("does_not_exist") is None


def test_push_standing_raises():
    stack = GoalStack()
    g = make_goal("test", "reinforcement", 0.5)
    g.standing = True
    with pytest.raises(ValueError):
        stack.push(g)


def test_all_goals_includes_all():
    stack = GoalStack()
    g = make_goal("u", "reinforcement", 0.5)
    stack.push(g)
    all_ids = {g.id for g in stack.all_goals()}
    assert "standing_arousal_cap" in all_ids
    assert g.id in all_ids


# ---------------------------------------------------------------------------
# GoalEvaluator.measure
# ---------------------------------------------------------------------------

def _mock_graph(reinforcement=1.0, arousal=0.5, soliton_stats=None, nodes=None):
    graph = MagicMock()
    graph.modulator.state = {"reinforcement": reinforcement, "arousal": arousal}
    pc = MagicMock()
    pc.soliton_stats.return_value = soliton_stats or {}
    graph.predictive_coder = pc
    adapter = MagicMock()
    nx_graph = MagicMock()
    nx_graph.nodes = set(nodes or [])
    adapter.to_networkx.return_value = nx_graph
    graph.adapter = adapter
    graph.emit = MagicMock()
    return graph


def _mock_loop(approval_rate=0.5):
    loop = MagicMock()
    loop.status.return_value = {"approval_rate": approval_rate}
    return loop


def test_evaluator_soliton_index():
    graph = _mock_graph(soliton_stats={"a": 0.8, "b": 0.6})
    ev = GoalEvaluator(graph, _mock_loop())
    g = make_goal("test", "soliton_index", 0.75)
    val = ev.measure(g)
    assert abs(val - 0.7) < 1e-6  # mean(0.8, 0.6) = 0.7


def test_evaluator_reinforcement():
    graph = _mock_graph(reinforcement=0.9)
    ev = GoalEvaluator(graph, _mock_loop())
    g = make_goal("test", "reinforcement", 0.5)
    assert ev.measure(g) == pytest.approx(0.9)


def test_evaluator_approval_rate():
    graph = _mock_graph()
    ev = GoalEvaluator(graph, _mock_loop(approval_rate=0.25))
    g = make_goal("test", "approval_rate", 0.15)
    assert ev.measure(g) == pytest.approx(0.25)


def test_evaluator_entity_discovered():
    graph = _mock_graph(nodes=["newton", "gravity"])
    ev = GoalEvaluator(graph, _mock_loop())
    g = make_goal("test", "entity_discovered", 1.0, target_entity="newton")
    assert ev.measure(g) == pytest.approx(1.0)
    g2 = make_goal("test", "entity_discovered", 1.0, target_entity="einstein")
    assert ev.measure(g2) == pytest.approx(0.0)


def test_evaluator_arousal_below():
    graph = _mock_graph(arousal=1.5)
    ev = GoalEvaluator(graph, _mock_loop())
    g = make_goal("test", "arousal_below", 2.0)
    assert ev.measure(g) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# GoalEvaluator.evaluate — achievement and standing reset
# ---------------------------------------------------------------------------

def test_goal_achieved_on_evaluate():
    graph = _mock_graph(reinforcement=0.9)
    stack = GoalStack()
    user_goal = make_goal("raise rein", "reinforcement", 0.5)
    stack.push(user_goal)
    ev = GoalEvaluator(graph, _mock_loop())
    ev.evaluate(stack)
    assert user_goal.status == GoalStatus.ACHIEVED
    assert user_goal.achieved_at is not None


def test_goal_not_achieved_when_below_target():
    graph = _mock_graph(reinforcement=0.3)
    stack = GoalStack()
    user_goal = make_goal("raise rein", "reinforcement", 0.5)
    stack.push(user_goal)
    ev = GoalEvaluator(graph, _mock_loop())
    ev.evaluate(stack)
    assert user_goal.status == GoalStatus.ACTIVE


def test_standing_rule_resets_after_cooldown():
    graph = _mock_graph(reinforcement=0.9)
    stack = GoalStack()
    rein_rule = stack.get("standing_reinforcement_floor")
    # Manually mark as achieved with old timestamp
    rein_rule.status = GoalStatus.ACHIEVED
    rein_rule._last_achieved_ts = time.time() - 9999  # long past cooldown

    ev = GoalEvaluator(graph, _mock_loop())
    ev.evaluate(stack)
    assert rein_rule.status == GoalStatus.ACTIVE
    assert rein_rule._last_achieved_ts is None


def test_standing_rule_does_not_reset_before_cooldown():
    graph = _mock_graph(reinforcement=0.9)
    stack = GoalStack()
    rein_rule = stack.get("standing_reinforcement_floor")
    rein_rule.status = GoalStatus.ACHIEVED
    rein_rule._last_achieved_ts = time.time() - 1  # just achieved 1s ago

    ev = GoalEvaluator(graph, _mock_loop())
    ev.evaluate(stack)
    # still achieved (cooldown not elapsed)
    assert rein_rule.status == GoalStatus.ACHIEVED


# ---------------------------------------------------------------------------
# AROUSAL_CAP suppression
# ---------------------------------------------------------------------------

def test_arousal_cap_suppresses_inference():
    graph = _mock_graph(arousal=3.0)  # above 2.0
    stack = GoalStack()
    ev = GoalEvaluator(graph, _mock_loop())
    suppress = ev.evaluate(stack)
    assert suppress is True


def test_arousal_cap_does_not_suppress_when_normal():
    graph = _mock_graph(arousal=1.0)  # below 2.0 → achieved
    stack = GoalStack()
    ev = GoalEvaluator(graph, _mock_loop())
    suppress = ev.evaluate(stack)
    assert suppress is False


def test_arousal_cap_clears_after_achieved():
    # Start breached, then drop arousal
    graph_high = _mock_graph(arousal=3.0)
    stack = GoalStack()
    ev = GoalEvaluator(graph_high, _mock_loop())
    suppress1 = ev.evaluate(stack)
    assert suppress1 is True

    # Reset standing rule manually (simulate cooldown elapsed)
    cap = stack.get("standing_arousal_cap")
    cap.status = GoalStatus.ACTIVE
    cap._last_achieved_ts = None

    graph_low = _mock_graph(arousal=1.0)
    ev2 = GoalEvaluator(graph_low, _mock_loop())
    suppress2 = ev2.evaluate(stack)
    assert suppress2 is False


# ---------------------------------------------------------------------------
# GoalEvaluator.get_context_seeds
# ---------------------------------------------------------------------------

def test_get_context_seeds_entity_discovered():
    graph = _mock_graph(nodes=[])
    stack = GoalStack()
    g = make_goal("find newton", "entity_discovered", 1.0, target_entity="newton")
    stack.push(g)
    # Make entity goal the top by giving it priority=1 (same as arousal cap)
    g.priority = 1
    # Suppress the standing rules first so user goal can come up
    for r in stack._standing_rules:
        r.status = GoalStatus.ACHIEVED
        r._last_achieved_ts = time.time()

    ev = GoalEvaluator(graph, _mock_loop())
    seeds = ev.get_context_seeds(stack)
    assert seeds == ["newton"]


def test_get_context_seeds_reinforcement_uses_wm():
    graph = _mock_graph(reinforcement=0.2)
    wm = WorkingMemoryBuffer(maxlen=10)
    now = time.time()
    wm.record(MemoryEntry(
        timestamp=now - 1,
        seeds=["alpha", "beta"],
        answers=["gamma"],
        top_score=0.8,
        soliton_index=None,
        prediction_error=None,
        source="query",
    ))
    stack = GoalStack()
    # Suppress all standing rules so reinforcement goal is top
    for r in stack._standing_rules:
        r.status = GoalStatus.ACHIEVED
        r._last_achieved_ts = time.time()
    g = make_goal("raise rein", "reinforcement", 0.5, priority=1)
    stack.push(g)

    ev = GoalEvaluator(graph, _mock_loop(), wm=wm)
    seeds = ev.get_context_seeds(stack, wm)
    assert seeds is not None
    assert "alpha" in seeds or "beta" in seeds


def test_get_context_seeds_none_when_no_active():
    stack = GoalStack()
    for r in stack._standing_rules:
        r.status = GoalStatus.ACHIEVED
        r._last_achieved_ts = time.time()
    graph = _mock_graph()
    ev = GoalEvaluator(graph, _mock_loop())
    seeds = ev.get_context_seeds(stack)
    assert seeds is None

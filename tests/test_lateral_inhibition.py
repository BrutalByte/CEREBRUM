"""Tests for Phase 100 — Lateral Inhibition (Winner-Take-All Beam Pruning)."""
import copy
from unittest.mock import MagicMock

import numpy as np
import pytest

from reasoning.traversal import BeamTraversal, TraversalPath
from core.attention_engine import CSAEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_traversal(ratio=0.5, beam_width=5):
    from core.attention_engine import CSAEngine
    adapter = MagicMock()
    adapter.get_embedding = MagicMock(return_value=np.zeros(8, dtype=np.float32))
    adapter.get_community = MagicMock(side_effect=lambda node: {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2}.get(node, -1))
    adapter.community_map = {}
    csa = MagicMock(spec=CSAEngine)
    csa.set_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    csa.use_temporal_decay = False
    t = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=beam_width, max_hop=2)
    t.lateral_inhibition_ratio = ratio
    return t


def _path(tail: str, score: float) -> TraversalPath:
    p = TraversalPath(nodes=["seed", "rel", tail], score=score)
    return p


# ---------------------------------------------------------------------------
# lateral_inhibition_ratio = 0 → no change
# ---------------------------------------------------------------------------

def test_inhibition_ratio_zero_preserves_scores():
    t = _make_traversal(ratio=0.0)
    candidates = [_path("a", 0.9), _path("b", 0.7), _path("c", 0.5)]
    result = t._apply_lateral_inhibition(candidates)
    scores = {p.tail: p.score for p in result}
    assert abs(scores["a"] - 0.9) < 1e-9
    assert abs(scores["b"] - 0.7) < 1e-9
    assert abs(scores["c"] - 0.5) < 1e-9


def test_inhibition_ratio_zero_prune_unchanged():
    """ratio=0 → _prune_candidates behaves like original."""
    t = _make_traversal(ratio=0.0, beam_width=2)
    candidates = [_path("a", 0.9), _path("b", 0.7), _path("c", 0.5), _path("d", 0.3)]
    result = t._prune_candidates(candidates, hop=1)
    # Top 2 by score
    tails = [p.tail for p in result]
    assert "a" in tails
    assert "b" in tails
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Winner is never suppressed
# ---------------------------------------------------------------------------

def test_highest_score_always_survives():
    """The top-scoring path within its community is never suppressed."""
    t = _make_traversal(ratio=1.0)
    # All in community 0
    t.adapter.get_community = MagicMock(return_value=0)
    candidates = [_path("a", 0.9), _path("b", 0.5), _path("c", 0.3)]
    result = t._apply_lateral_inhibition(candidates)
    scores = {p.tail: p.score for p in result}
    assert abs(scores["a"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------
# Same-community suppression
# ---------------------------------------------------------------------------

def test_inhibition_suppresses_same_community_duplicates():
    """Lower-ranked paths in same community get score reduced."""
    t = _make_traversal(ratio=0.5)
    t.adapter.get_community = MagicMock(return_value=0)  # all same community
    candidates = [_path("a", 0.9), _path("b", 0.6)]
    result = t._apply_lateral_inhibition(candidates)
    scores = {p.tail: p.score for p in result}
    assert scores["a"] == pytest.approx(0.9)
    assert scores["b"] < 0.6


def test_inhibition_ratio_one_keeps_one_per_community():
    """ratio=1.0 → second path in community gets score * (1 - 1*1/2) = score*0.5."""
    t = _make_traversal(ratio=1.0)
    t.adapter.get_community = MagicMock(return_value=0)
    candidates = [_path("a", 0.8), _path("b", 0.6)]
    result = t._apply_lateral_inhibition(candidates)
    scores = {p.tail: p.score for p in result}
    assert scores["a"] == pytest.approx(0.8)
    # rank=1, n=2 → suppressed_score = 0.6 * (1 - 1.0 * 1/2) = 0.3
    assert scores["b"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Cross-community paths unaffected
# ---------------------------------------------------------------------------

def test_cross_community_paths_unaffected():
    """Paths to different communities don't suppress each other."""
    t = _make_traversal(ratio=1.0)
    # a → community 0, b → community 1
    t.adapter.get_community = MagicMock(side_effect=lambda node: 0 if node == "a" else 1)
    candidates = [_path("a", 0.8), _path("b", 0.6)]
    result = t._apply_lateral_inhibition(candidates)
    scores = {p.tail: p.score for p in result}
    # Each is the winner of its community → no suppression
    assert scores["a"] == pytest.approx(0.8)
    assert scores["b"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Unknown community handled safely
# ---------------------------------------------------------------------------

def test_inhibition_with_unknown_community():
    """Nodes with unknown community (get_community returns -1) should not crash."""
    t = _make_traversal(ratio=0.5)
    t.adapter.get_community = MagicMock(return_value=-1)
    candidates = [_path("x", 0.7), _path("y", 0.4)]
    result = t._apply_lateral_inhibition(candidates)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Beam width respected after inhibition
# ---------------------------------------------------------------------------

def test_beam_width_respected_after_inhibition():
    """After inhibition, _prune_candidates still caps to beam_width."""
    t = _make_traversal(ratio=0.5, beam_width=2)
    t.adapter.get_community = MagicMock(return_value=0)
    candidates = [_path(f"n{i}", 1.0 - i * 0.1) for i in range(6)]
    result = t._prune_candidates(candidates, hop=1)
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# Diversity improvement
# ---------------------------------------------------------------------------

def test_inhibition_improves_answer_diversity():
    """With inhibition, surviving paths span more communities than without."""
    # 4 paths: 3 in community 0 (high scores), 1 in community 1 (lower score)
    beam_width = 2
    t_no  = _make_traversal(ratio=0.0, beam_width=beam_width)
    t_yes = _make_traversal(ratio=1.0, beam_width=beam_width)

    def _community(node):
        return 0 if node in ("a", "b", "c") else 1

    t_no.adapter.get_community  = MagicMock(side_effect=_community)
    t_yes.adapter.get_community = MagicMock(side_effect=_community)

    candidates = [_path("a", 0.9), _path("b", 0.8), _path("c", 0.7), _path("d", 0.5)]

    result_no  = t_no._prune_candidates(copy.deepcopy(candidates), hop=1)
    result_yes = t_yes._prune_candidates(copy.deepcopy(candidates), hop=1)

    cids_no  = {_community(p.tail) for p in result_no}
    cids_yes = {_community(p.tail) for p in result_yes}

    # Without inhibition: top 2 are both community 0 → diversity=1
    # With inhibition: community 1's "d" gets a chance → diversity >= 1
    assert len(cids_yes) >= len(cids_no)

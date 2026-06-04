"""Tests for Phase 220 SelfAwarenessEngine."""
import pytest
from unittest.mock import MagicMock
from core.self_awareness import SelfAwarenessEngine, SelfAwarenessReport


def _make_answer(score=0.8, branch_count=3, consensus_score=0.75):
    ans = MagicMock()
    ans.score = score
    ans.branch_count = branch_count
    ans.consensus_score = consensus_score
    ans.contradiction_flags = []
    path = MagicMock()
    path.beta_alpha = 8.0
    path.beta_beta = 2.0
    path.hop_depth = 2
    path.path_confidence = 0.9
    path.nodes = ["A", "treats", "B", "causes", "C"]
    # 9-tuple per edge: (sim, cs, etw, nd, hd, pr, td, nr, grounding)
    path.edge_features = [
        (0.3, 0.5, 0.8, 0.1, 0.9, 0.2, 1.0, 0.5, 0.95),
        (0.4, 0.6, 0.7, 0.1, 0.8, 0.3, 1.0, 0.4, 0.90),
    ]
    ans.best_path = path
    return ans


def test_basic_report():
    engine = SelfAwarenessEngine(causal_relations={"treats"})
    top = _make_answer(score=0.8)
    second = _make_answer(score=0.3)
    report = engine.assess([top, second])

    assert isinstance(report, SelfAwarenessReport)
    assert 0 < report.answer_confidence <= 1.0
    assert report.corroboration == 3
    assert not report.knowledge_gap
    assert not report.contradiction_detected
    assert report.evidence_quality > 0.8  # grounding is 0.95 / 0.90
    assert report.dominant_signal != ""
    assert report.summary != ""
    assert report.causal_fraction > 0  # "treats" is causal
    assert report.path_length == 2


def test_knowledge_gap_no_answers():
    engine = SelfAwarenessEngine()
    report = engine.assess([])
    assert report.knowledge_gap
    assert "no answers" in report.gap_reason


def test_knowledge_gap_low_score():
    engine = SelfAwarenessEngine(min_confident_score=0.5)
    ans = _make_answer(score=0.1)
    ans.consensus_score = 0.1
    report = engine.assess([ans])
    assert report.knowledge_gap
    assert "threshold" in report.gap_reason


def test_contradiction_detected():
    engine = SelfAwarenessEngine()
    top = _make_answer()
    top.contradiction_flags = [MagicMock(), MagicMock()]
    report = engine.assess([top])
    assert report.contradiction_detected
    assert report.contradiction_count == 2


def test_to_dict_serialisable():
    engine = SelfAwarenessEngine()
    report = engine.assess([_make_answer()])
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "summary" in d
    assert "signal_breakdown" in d
    assert isinstance(d["signal_breakdown"], dict)


def test_signal_breakdown_keys():
    engine = SelfAwarenessEngine()
    report = engine.assess([_make_answer()])
    assert "community_structure" in report.signal_breakdown
    assert "source_credibility" in report.signal_breakdown
    assert "semantic_similarity" in report.signal_breakdown


def test_from_symbolic_validator():
    validator = MagicMock()
    from core.symbolic_engine import ConstraintType
    c = MagicMock()
    c.constraint_type = ConstraintType.CAUSAL_ORDERING
    c.params = {"relation": "causes"}
    validator.constraints = [c]
    engine = SelfAwarenessEngine.from_symbolic_validator(validator)
    assert "causes" in engine.causal_relations


def test_no_causal_zero_fraction():
    engine = SelfAwarenessEngine(causal_relations=set())
    report = engine.assess([_make_answer()])
    assert report.causal_fraction == 0.0

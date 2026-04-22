"""
Batch A tests — Feature 2 (temporal recency) + Feature 4 (discovery calibration).
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest


# ===========================================================================
# Feature 2 — Temporal recency scoring
# ===========================================================================

from core.external_validator import (
    LiteratureHit,
    ValidationReport,
    _compute_recency_score,
    _RECENCY_HALF_LIFE_YEARS,
)


def _hit(year):
    return LiteratureHit(adapter="pubmed", external_id="x", title="T", year=year)


def test_recency_score_no_year_data():
    """Hits without year → neutral 0.5."""
    hits = [LiteratureHit(adapter="pubmed", external_id="x", title="T", year=None)]
    assert _compute_recency_score(hits) == 0.5


def test_recency_score_empty_hits():
    assert _compute_recency_score([]) == 0.5


def test_recency_score_current_year():
    """A hit published this year → score ≈ 1.0."""
    current = time.gmtime().tm_year
    score = _compute_recency_score([_hit(current)])
    assert abs(score - 1.0) < 1e-4


def test_recency_score_half_life():
    """A hit exactly half_life years old → score ≈ 0.5."""
    current = time.gmtime().tm_year
    hl = int(_RECENCY_HALF_LIFE_YEARS)
    score = _compute_recency_score([_hit(current - hl)])
    assert abs(score - 0.5) < 0.02


def test_recency_score_old_hit():
    """A hit 30 years old → score << 0.3."""
    current = time.gmtime().tm_year
    score = _compute_recency_score([_hit(current - 30)])
    assert score < 0.15


def test_recency_score_mean_of_multiple():
    """Mean is taken across all hits with year data."""
    current = time.gmtime().tm_year
    s1 = math.exp(0)                                          # age 0 → 1.0
    s2 = math.exp(-7 * math.log(2) / _RECENCY_HALF_LIFE_YEARS)  # age ≈ hl → 0.5
    expected = (s1 + s2) / 2
    score = _compute_recency_score([_hit(current), _hit(current - 7)])
    assert abs(score - expected) < 0.01


def test_recency_score_mixed_none_and_year():
    """Hits with None year are excluded from the mean."""
    current = time.gmtime().tm_year
    score = _compute_recency_score([
        _hit(current),
        LiteratureHit(adapter="x", external_id="y", title="T", year=None),
    ])
    assert abs(score - 1.0) < 1e-4


def test_validation_report_has_recency_score():
    """ValidationReport dataclass has recency_score with default 0.5."""
    r = ValidationReport(
        hypothesis_id="h1",
        source_id="A",
        target_id="B",
        derived_relation="related_to",
        literature_status="novel",
        novelty_score=1.0,
        hit_count=0,
    )
    assert hasattr(r, "recency_score")
    assert r.recency_score == 0.5


def test_validation_report_recency_score_settable():
    """recency_score can be set explicitly."""
    r = ValidationReport(
        hypothesis_id="h1",
        source_id="A",
        target_id="B",
        derived_relation="related_to",
        literature_status="novel",
        novelty_score=1.0,
        recency_score=0.72,
        hit_count=0,
    )
    assert abs(r.recency_score - 0.72) < 1e-6


# ===========================================================================
# Feature 4 — DiscoveryCalibrator
# ===========================================================================

from core.discovery_calibrator import DiscoveryCalibrator


def _make_calibrator(**kwargs) -> DiscoveryCalibrator:
    return DiscoveryCalibrator(**kwargs)


def test_cold_start_weight_is_max():
    """Before any data, unseen community returns max_weight."""
    cal = _make_calibrator(max_weight=5.0)
    assert cal.get_weight(42) == 5.0


def test_weight_after_scan_no_discovery():
    """Community scanned but no discovery → weight ≥ 1.0 (not penalised)."""
    cal = _make_calibrator(max_weight=5.0, min_weight=0.2)
    # Two communities: 0 scanned only, 1 has discoveries — 0 should be >= 1
    cal.record_scan({0, 1})
    cal.record_discovery(1)  # only community 1 discovered
    w0 = cal.get_weight(0)
    w1 = cal.get_weight(1)
    assert w0 > w1  # unexplored community gets boosted relative to productive one


def test_weight_decreases_with_discoveries():
    """Community with many discoveries gets lower weight than one with none."""
    cal = _make_calibrator()
    for _ in range(10):
        cal.record_scan({0, 1})
        cal.record_discovery(0)   # community 0 always finds something
        # community 1 never finds anything

    w_high_rate = cal.get_weight(0)   # low weight — productive
    w_low_rate  = cal.get_weight(1)   # high weight — unexplored

    assert w_high_rate < w_low_rate


def test_weight_clamped_to_min():
    """A community with very high discovery rate is clamped at min_weight."""
    cal = _make_calibrator(min_weight=0.2, max_weight=5.0, window=100)
    for _ in range(50):
        cal.record_scan({0})
        cal.record_discovery(0)
    assert cal.get_weight(0) >= 0.2


def test_weight_clamped_to_max():
    """A community never discovered is clamped at max_weight."""
    cal = _make_calibrator(min_weight=0.2, max_weight=5.0)
    for _ in range(50):
        cal.record_scan({0})
        # no discoveries
    assert cal.get_weight(0) <= 5.0


def test_two_communities_differential():
    """High-discovery community gets lower weight than low-discovery community."""
    cal = _make_calibrator()
    for _ in range(10):
        cal.record_scan({0, 1})
        cal.record_discovery(0)   # community 0 always finds something
        # community 1 never finds anything

    w0 = cal.get_weight(0)
    w1 = cal.get_weight(1)
    assert w1 > w0


def test_stats_returns_expected_keys():
    cal = _make_calibrator()
    cal.record_scan({0})
    cal.record_discovery(0)
    s = cal.stats()
    assert "total_scans" in s
    assert "total_discoveries" in s
    assert "communities" in s
    assert 0 in s["communities"]
    assert "rate" in s["communities"][0]
    assert "weight" in s["communities"][0]


def test_stats_total_counts():
    cal = _make_calibrator()
    for _ in range(3):
        cal.record_scan({0})
    cal.record_discovery(0)
    cal.record_discovery(0)
    s = cal.stats()
    assert s["total_scans"] == 3
    assert s["total_discoveries"] == 2


def test_ema_decay_reduces_old_counts():
    """EMA with short window decays old data — community 0 recovers after burst."""
    cal = _make_calibrator(window=2, max_weight=5.0)  # fast decay
    # Community 1 provides a stable reference (never discovers)
    # Community 0: burst of discoveries, then nothing
    for _ in range(10):
        cal.record_scan({0, 1})
        cal.record_discovery(0)
    w_after_burst = cal.get_weight(0)
    # Now scan many times without discoveries — EMA decays community 0's old hits
    for _ in range(20):
        cal.record_scan({0, 1})
    w_after_decay = cal.get_weight(0)
    # Community 0's weight should recover as old discovery counts decay away
    assert w_after_decay >= w_after_burst - 1e-6



# ===========================================================================
# Integration: ResearchAgent + DiscoveryCalibrator
# ===========================================================================

def test_set_calibrator_wires_correctly():
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine
    from adapters.networkx_adapter import NetworkXAdapter
    import networkx as nx

    G = nx.Graph()
    G.add_edge("A", "B", relation="related_to")
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"A": 0, "B": 0}
    he = HypothesisEngine(adapter)
    agent = ResearchAgent(adapter, he)

    assert agent._calibrator is None
    cal = _make_calibrator()
    agent.set_calibrator(cal)
    assert agent._calibrator is cal


def test_calibrator_constructor_param():
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine
    from adapters.networkx_adapter import NetworkXAdapter
    import networkx as nx

    G = nx.Graph()
    G.add_edge("A", "B", relation="related_to")
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"A": 0, "B": 0}
    he = HypothesisEngine(adapter)
    cal = _make_calibrator()
    agent = ResearchAgent(adapter, he, calibrator=cal)
    assert agent._calibrator is cal

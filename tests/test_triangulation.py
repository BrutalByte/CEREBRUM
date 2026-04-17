"""
Tests for TriangulationEngine — Phase 72.

Coverage:
  - evaluate() returns all four scores in [0, 1]
  - P1 reverse_confidence = 0.0 when no reverse path
  - P1 reverse_confidence > 0 when reverse path exists
  - P1 skipped when run_bidirectional=False → 0.0
  - P2 strategy_agreement = 1.0 when all three strategies find proposals
  - P2 strategy_agreement ≈ 0.33 when only primary finds proposals
  - P2 skipped when run_multi_strategy=False → 1.0 or 0.0
  - P3 mean_path_independence = 0.5 when no independence_scores
  - P3 mean_path_independence computed correctly from independence_scores
  - P4 semantic_type_score = 0.5 for unknown relation type
  - P4 semantic_type_score = 1.0 for exact (src_type, tgt_type) match
  - P4 semantic_type_score = 0.65 for one-side type match
  - P4 semantic_type_score = 0.30 for relation known but neither type matches
  - is_SynapticBridge_candidate = True when all four conditions met
  - is_SynapticBridge_candidate = False when reverse_confidence = 0
  - Type index caches on second call (same graph signature)
  - Type index rebuilds when edge count changes
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from core.triangulation_engine import TriangulationEngine, TriangulationReport


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    source_id="A",
    target_id="B",
    community_distance=3,
    seeded_by="structural_hole",
    local_density=0.1,
):
    return SimpleNamespace(
        source_id=source_id,
        target_id=target_id,
        community_distance=community_distance,
        seeded_by=seeded_by,
        local_density=local_density,
    )


def _make_proposal(confidence=0.8, independence_scores=None, derived_relation="related_to"):
    return SimpleNamespace(
        confidence=confidence,
        path_count=3,
        contradiction_score=0.05,
        derived_relation=derived_relation,
        independence_scores=independence_scores or [],
    )


class _MockHypothesisEngine:
    """
    Controllable mock for HypothesisEngine.

    ``responses`` maps (source_id, target_id, max_hop) → list of proposals.
    Unmatched calls return [].
    """
    def __init__(self, responses=None):
        self._responses = responses or {}
        self.call_log: List[dict] = []

    def generate(self, source_id, target_id, max_hop=3, beam_width=10, max_budget=500, **_):
        self.call_log.append({"source": source_id, "target": target_id, "max_hop": max_hop})
        key = (source_id, target_id, max_hop)
        return self._responses.get(key, [])


class _MockEntity:
    def __init__(self, type_="entity"):
        self.type = type_


class _MockAdapter:
    """Minimal adapter with programmable entity types and graph edges."""

    def __init__(self, edges=None, entity_types=None):
        """
        edges: list of (u, v, {relation, ...}) tuples
        entity_types: {entity_id: type_str}
        """
        import networkx as nx
        self._G = nx.MultiDiGraph()
        for u, v, data in (edges or []):
            self._G.add_edge(u, v, **data)
        self._entity_types = entity_types or {}

    def to_networkx(self):
        return self._G

    def get_entity(self, entity_id: str) -> _MockEntity:
        t = self._entity_types.get(entity_id, "entity")
        return _MockEntity(type_=t)


def _make_engine(
    adapter=None,
    hyp_responses=None,
    run_bidirectional=True,
    run_multi_strategy=True,
) -> TriangulationEngine:
    if adapter is None:
        adapter = _MockAdapter()
    hyp = _MockHypothesisEngine(responses=hyp_responses or {})
    return TriangulationEngine(
        adapter=adapter,
        hypothesis_engine=hyp,
        run_bidirectional=run_bidirectional,
        run_multi_strategy=run_multi_strategy,
    )


# ---------------------------------------------------------------------------
# evaluate() return value shape
# ---------------------------------------------------------------------------

def test_evaluate_returns_report():
    engine = _make_engine()
    cand = _make_candidate()
    report = engine.evaluate(cand, [])
    assert isinstance(report, TriangulationReport)


def test_evaluate_all_scores_in_range():
    engine = _make_engine()
    cand = _make_candidate()
    props = [_make_proposal(independence_scores=[0.8, 0.6])]
    report = engine.evaluate(cand, props)
    assert 0.0 <= report.reverse_confidence <= 1.0
    assert 0.0 <= report.strategy_agreement <= 1.0
    assert 0.0 <= report.mean_path_independence <= 1.0
    assert 0.0 <= report.semantic_type_score <= 1.0


# ---------------------------------------------------------------------------
# P1 — Bidirectional path confirmation
# ---------------------------------------------------------------------------

def test_p1_no_reverse_path():
    engine = _make_engine(hyp_responses={})  # no responses → all return []
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, [_make_proposal()])
    assert report.reverse_confidence == 0.0


def test_p1_reverse_path_found():
    proposal = _make_proposal(confidence=0.75)
    engine = _make_engine(hyp_responses={
        ("B", "A", 3): [proposal],
    })
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, [])
    assert abs(report.reverse_confidence - 0.75) < 1e-6


def test_p1_skipped_when_disabled():
    proposal = _make_proposal(confidence=0.9)
    engine = _make_engine(
        hyp_responses={("B", "A", 3): [proposal]},
        run_bidirectional=False,
    )
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, [])
    assert report.reverse_confidence == 0.0


def test_p1_below_min_confidence_not_counted():
    """Reverse proposal below min_confidence threshold is not counted."""
    low_prop = _make_proposal(confidence=0.10)
    engine = _make_engine(hyp_responses={("B", "A", 3): [low_prop]})
    engine._min_confidence = 0.20
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, [])
    assert report.reverse_confidence == 0.0


# ---------------------------------------------------------------------------
# P2 — Multi-strategy agreement
# ---------------------------------------------------------------------------

def test_p2_all_strategies_agree():
    """All three strategies (primary + 2 extra) find proposals → agreement = 1.0."""
    prop = _make_proposal(confidence=0.7)
    responses = {
        ("A", "B", 2): [prop],   # conservative
        ("A", "B", 4): [prop],   # exploratory
    }
    engine = _make_engine(hyp_responses=responses)
    primary = [prop]  # primary (standard, 3-hop) also found
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, primary)
    assert abs(report.strategy_agreement - 1.0) < 1e-6


def test_p2_only_primary():
    """Only the primary (standard) run finds proposals → agreement = 1/3."""
    engine = _make_engine(hyp_responses={})  # extra strategies return []
    primary = [_make_proposal(confidence=0.7)]
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, primary)
    assert abs(report.strategy_agreement - 1/3) < 1e-4


def test_p2_no_strategies_agree():
    """No strategy finds proposals → agreement = 0.0."""
    engine = _make_engine(hyp_responses={})
    cand = _make_candidate(source_id="A", target_id="B")
    report = engine.evaluate(cand, [])  # primary also empty
    assert abs(report.strategy_agreement - 0.0) < 1e-6


def test_p2_skipped_returns_primary_only():
    """run_multi_strategy=False: agreement is 1.0 if primary found, 0.0 otherwise."""
    prop = _make_proposal(confidence=0.7)
    engine = _make_engine(hyp_responses={}, run_multi_strategy=False)
    cand = _make_candidate(source_id="A", target_id="B")
    # Primary found
    report = engine.evaluate(cand, [prop])
    assert abs(report.strategy_agreement - 1.0) < 1e-6
    # Primary not found
    report2 = engine.evaluate(cand, [])
    assert abs(report2.strategy_agreement - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# P3 — Path independence
# ---------------------------------------------------------------------------

def test_p3_no_proposals():
    engine = _make_engine()
    cand = _make_candidate()
    report = engine.evaluate(cand, [])
    assert report.mean_path_independence == 0.5


def test_p3_no_independence_scores_on_proposals():
    engine = _make_engine()
    cand = _make_candidate()
    props = [_make_proposal(independence_scores=[]), _make_proposal(independence_scores=None)]
    report = engine.evaluate(cand, props)
    assert report.mean_path_independence == 0.5


def test_p3_correct_mean():
    engine = _make_engine()
    cand = _make_candidate()
    props = [
        _make_proposal(independence_scores=[1.0, 0.8]),
        _make_proposal(independence_scores=[0.6]),
    ]
    report = engine.evaluate(cand, props)
    expected = (1.0 + 0.8 + 0.6) / 3
    assert abs(report.mean_path_independence - expected) < 1e-6


def test_p3_perfect_independence():
    engine = _make_engine()
    cand = _make_candidate()
    props = [_make_proposal(independence_scores=[1.0, 1.0, 1.0])]
    report = engine.evaluate(cand, props)
    assert abs(report.mean_path_independence - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# P4 — Semantic type consistency
# ---------------------------------------------------------------------------

def test_p4_novel_relation_neutral():
    """Relation not seen in graph → 0.5 (neutral, not penalised)."""
    adapter = _MockAdapter(edges=[], entity_types={"A": "drug", "B": "disease"})
    engine = _make_engine(adapter=adapter)
    cand = _make_candidate(source_id="A", target_id="B")
    props = [_make_proposal(derived_relation="treats")]
    report = engine.evaluate(cand, props)
    assert report.semantic_type_score == 0.5


def test_p4_exact_type_match():
    """(src_type, tgt_type) exactly seen in graph for this relation → 1.0."""
    adapter = _MockAdapter(
        edges=[("X", "Y", {"relation": "treats"})],
        entity_types={"A": "drug", "B": "disease", "X": "drug", "Y": "disease"},
    )
    engine = _make_engine(adapter=adapter)
    cand = _make_candidate(source_id="A", target_id="B")
    props = [_make_proposal(derived_relation="treats")]
    report = engine.evaluate(cand, props)
    assert abs(report.semantic_type_score - 1.0) < 1e-6


def test_p4_one_side_match():
    """Source type matches but target type is new → 0.65."""
    adapter = _MockAdapter(
        edges=[("X", "Y", {"relation": "treats"})],
        entity_types={"A": "drug", "B": "protein", "X": "drug", "Y": "disease"},
    )
    engine = _make_engine(adapter=adapter)
    cand = _make_candidate(source_id="A", target_id="B")
    props = [_make_proposal(derived_relation="treats")]
    report = engine.evaluate(cand, props)
    assert abs(report.semantic_type_score - 0.65) < 1e-4


def test_p4_neither_type_matches():
    """Relation known but neither endpoint type seen → 0.30."""
    adapter = _MockAdapter(
        edges=[("X", "Y", {"relation": "treats"})],
        entity_types={"A": "gene", "B": "pathway", "X": "drug", "Y": "disease"},
    )
    engine = _make_engine(adapter=adapter)
    cand = _make_candidate(source_id="A", target_id="B")
    props = [_make_proposal(derived_relation="treats")]
    report = engine.evaluate(cand, props)
    assert abs(report.semantic_type_score - 0.30) < 1e-4


def test_p4_empty_derived_relation():
    """No derived relation → neutral 0.5."""
    engine = _make_engine()
    cand = _make_candidate()
    report = engine.evaluate(cand, [])  # no proposals → derived_relation = ""
    assert report.semantic_type_score == 0.5


# ---------------------------------------------------------------------------
# SynapticBridge candidate flag
# ---------------------------------------------------------------------------

def test_SynapticBridge_candidate_all_conditions_met():
    prop = _make_proposal(confidence=0.8, independence_scores=[0.9, 0.8])
    rev_prop = _make_proposal(confidence=0.75)
    adapter = _MockAdapter(
        edges=[("X", "Y", {"relation": "related_to"})],
        entity_types={"A": "entity", "B": "entity", "X": "entity", "Y": "entity"},
    )
    responses = {
        ("B", "A", 3): [rev_prop],
        ("A", "B", 2): [prop],
        ("A", "B", 4): [prop],
    }
    engine = _make_engine(adapter=adapter, hyp_responses=responses)
    cand = _make_candidate(source_id="A", target_id="B", community_distance=3)
    report = engine.evaluate(cand, [prop])
    # All four conditions: reverse > 0, community_distance > 2, sem > 0.3, indep > 0.4
    assert report.is_SynapticBridge_candidate is True


def test_SynapticBridge_candidate_no_reverse():
    """No reverse path → not a SynapticBridge."""
    engine = _make_engine(hyp_responses={})
    prop = _make_proposal(independence_scores=[0.9])
    cand = _make_candidate(community_distance=3)
    report = engine.evaluate(cand, [prop])
    assert report.is_SynapticBridge_candidate is False


def test_SynapticBridge_candidate_low_community_distance():
    """Same community → not a SynapticBridge even if reverse path exists."""
    prop = _make_proposal(confidence=0.8, independence_scores=[0.9])
    rev_prop = _make_proposal(confidence=0.75)
    responses = {("B", "A", 3): [rev_prop]}
    engine = _make_engine(hyp_responses=responses)
    cand = _make_candidate(community_distance=1)  # too close
    report = engine.evaluate(cand, [prop])
    assert report.is_SynapticBridge_candidate is False


# ---------------------------------------------------------------------------
# Type index caching
# ---------------------------------------------------------------------------

def test_type_index_cached_on_same_signature():
    adapter = _MockAdapter(
        edges=[("X", "Y", {"relation": "related_to"})],
        entity_types={"X": "drug", "Y": "disease"},
    )
    engine = _make_engine(adapter=adapter)
    # Build index
    idx1 = engine._get_type_index()
    # Second call with same graph
    idx2 = engine._get_type_index()
    assert idx1 is idx2  # same object — returned from cache


def test_type_index_rebuilds_on_graph_change():
    import networkx as nx

    G = nx.MultiDiGraph()
    G.add_edge("X", "Y", relation="related_to")

    class _LiveAdapter:
        def __init__(self, g):
            self._G = g
            self._types = {"X": "drug", "Y": "disease"}

        def to_networkx(self):
            return self._G

        def get_entity(self, eid):
            return _MockEntity(type_=self._types.get(eid, "entity"))

    adapter = _LiveAdapter(G)
    engine = TriangulationEngine(adapter=adapter, hypothesis_engine=_MockHypothesisEngine())

    idx1 = engine._get_type_index()

    # Mutate the graph — this changes edge count
    G.add_edge("X", "Z", relation="causes")
    adapter._types["Z"] = "symptom"

    idx2 = engine._get_type_index()
    assert idx2 is not idx1  # rebuilt
    assert "causes" in idx2

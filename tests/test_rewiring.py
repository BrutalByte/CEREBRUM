"""Tests for Phase 98 — Rewiring Sprint (8 integration gaps)."""
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.consolidation_engine import ConsolidationEngine
from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(edges=None):
    G = nx.DiGraph()
    for (u, rel, v, w) in (edges or []):
        G.add_edge(u, v, relation=rel, weight=w)
    adapter = MagicMock(spec=NetworkXAdapter)
    adapter._G = G
    adapter.update_edge_weight = NetworkXAdapter.update_edge_weight.__get__(adapter, NetworkXAdapter)
    return adapter


def _graph_mock():
    g = MagicMock()
    g.emit = MagicMock()
    return g


def _entry(top_score=0.5, path_edges=None, prediction_error=None, source="query"):
    return MemoryEntry(
        timestamp=time.time(),
        seeds=[],
        answers=[],
        top_score=top_score,
        soliton_index=None,
        prediction_error=prediction_error,
        source=source,
        path_edges=path_edges or [],
    )


def _wm(*entries):
    wm = WorkingMemoryBuffer(maxlen=50)
    for e in entries:
        wm.record(e)
    return wm


# ---------------------------------------------------------------------------
# Gap 1: ChemicalModulator → ConsolidationEngine (reinforcement_scale)
# ---------------------------------------------------------------------------

def test_consolidation_scales_with_reinforcement():
    """High reinforcement_scale → larger Hebbian delta."""
    adapter_high = _make_adapter([("a", "r", "b", 1.0)])
    adapter_low  = _make_adapter([("a", "r", "b", 1.0)])

    eng_high = ConsolidationEngine(adapter=adapter_high, graph=_graph_mock(), min_score=0.3, hebbian_delta=0.1)
    eng_low  = ConsolidationEngine(adapter=adapter_low,  graph=_graph_mock(), min_score=0.3, hebbian_delta=0.1)

    wm = _wm(_entry(top_score=0.8, path_edges=[("a", "r", "b")]))
    eng_high.consolidate(wm, reinforcement_scale=2.0)
    eng_low.consolidate(wm, reinforcement_scale=0.5)

    assert adapter_high._G["a"]["b"]["weight"] > adapter_low._G["a"]["b"]["weight"]


def test_consolidation_reinforcement_default_unchanged():
    """Default reinforcement_scale=1.0 keeps original behavior."""
    adapter = _make_adapter([("a", "r", "b", 1.0)])
    eng = ConsolidationEngine(adapter=adapter, graph=_graph_mock(), min_score=0.3, hebbian_delta=0.1)
    wm = _wm(_entry(top_score=0.8, path_edges=[("a", "r", "b")]))
    result = eng.consolidate(wm)
    assert result.edges_strengthened == 1


# ---------------------------------------------------------------------------
# Gap 2: PredictiveCodingEngine PE → Consolidation salience
# ---------------------------------------------------------------------------

def test_consolidation_pe_boosts_salience():
    """High-PE entry should be replayed over equal-score low-PE entry when k=1."""
    adapter = _make_adapter([
        ("high_pe", "r", "x", 1.0),
        ("low_pe",  "r", "x", 1.0),
    ])
    eng = ConsolidationEngine(adapter=adapter, graph=_graph_mock(), min_score=0.3,
                               hebbian_delta=0.1, pe_weight=0.5)
    wm = _wm(
        _entry(top_score=0.5, path_edges=[("high_pe", "r", "x")], prediction_error=0.8),
        _entry(top_score=0.5, path_edges=[("low_pe",  "r", "x")], prediction_error=0.0),
    )
    eng.consolidate(wm, k=1)
    # Only the high-PE entry should have been replayed
    assert adapter._G["high_pe"]["x"]["weight"] > 1.0
    assert adapter._G["low_pe"]["x"]["weight"] == 1.0


def test_consolidation_pe_weight_zero_no_boost():
    """pe_weight=0 means salience = top_score only."""
    adapter = _make_adapter([
        ("a", "r", "b", 1.0),
        ("c", "r", "d", 1.0),
    ])
    eng = ConsolidationEngine(adapter=adapter, graph=_graph_mock(), min_score=0.3,
                               hebbian_delta=0.1, pe_weight=0.0)
    wm = _wm(
        _entry(top_score=0.6, path_edges=[("a", "r", "b")], prediction_error=1.0),
        _entry(top_score=0.9, path_edges=[("c", "r", "d")], prediction_error=0.0),
    )
    eng.consolidate(wm, k=1)
    # Without PE boost, highest top_score wins
    assert adapter._G["c"]["d"]["weight"] > 1.0
    assert adapter._G["a"]["b"]["weight"] == 1.0


# ---------------------------------------------------------------------------
# Gap 3: InsightEngine → WorkingMemory
# ---------------------------------------------------------------------------

def test_insight_engine_records_to_wm():
    """_propagate_reward with a WM attached should record an 'insight' entry."""
    from core.insight_engine import InsightEngine
    adapter = MagicMock()
    adapter._G = nx.DiGraph()
    adapter._G.add_edge("a", "b", relation="r", confidence=1.0)
    adapter.get_community = MagicMock(return_value=0)
    adapter.get_embedding = MagicMock(return_value=None)

    engine = InsightEngine(adapter=adapter, cold_scan_interval=None)
    wm = WorkingMemoryBuffer(maxlen=20)
    engine.set_working_memory(wm)

    path = MagicMock()
    path.nodes = ["a", "r", "b"]
    engine._propagate_reward(adapter._G, path, insight_score=0.7)

    entries = wm.recent(10)
    assert len(entries) == 1
    assert entries[0].source == "insight"
    assert entries[0].path_edges == [("a", "r", "b")]
    engine.stop()


def test_insight_engine_no_wm_no_error():
    """_propagate_reward without WM attached should not raise."""
    from core.insight_engine import InsightEngine
    adapter = MagicMock()
    adapter._G = nx.DiGraph()
    adapter._G.add_edge("a", "b", relation="r", confidence=1.0)
    engine = InsightEngine(adapter=adapter, cold_scan_interval=None)

    path = MagicMock()
    path.nodes = ["a", "r", "b"]
    engine._propagate_reward(adapter._G, path, insight_score=0.5)  # must not raise
    engine.stop()


# ---------------------------------------------------------------------------
# Gap 4: CerebellarEngine → WorkingMemory
# ---------------------------------------------------------------------------

def test_cerebellar_engine_records_dissonance_to_wm():
    """Dissonance event should be recorded to WM with source='dissonance' and top_score=0."""
    from core.cerebellar_engine import CerebellarEngine
    from reasoning.answer_extractor import Answer

    engine = CerebellarEngine(dissonance_threshold=0.1, min_path_score=0.1)
    wm = WorkingMemoryBuffer(maxlen=20)
    engine.set_working_memory(wm)

    best_path = MagicMock()
    best_path.nodes = ["seed", "rel", "target"]
    ans = Answer(
        entity_id="target",
        score=0.9,
        path_score=0.9,
        consensus_score=0.1,
        best_path=best_path,
    )
    engine.process_results("seed", [ans])

    entries = wm.recent(10)
    assert len(entries) == 1
    assert entries[0].source == "dissonance"
    assert entries[0].top_score == 0.0


# ---------------------------------------------------------------------------
# Gap 5: AutoApprover decisions → WorkingMemory
# ---------------------------------------------------------------------------

def test_research_agent_records_approval_to_wm():
    """approve() should record an 'approval' MemoryEntry for each materialized edge."""
    from core.research_agent import ResearchAgent, ResearchFinding

    adapter = MagicMock()
    hypothesis_engine = MagicMock()
    hypothesis_engine.materialize.return_value = 1

    proposal = MagicMock()
    proposal.source_id = "A"
    proposal.target_id = "B"
    proposal.relation = "causes"
    proposal.confidence = 0.9

    finding = MagicMock(spec=ResearchFinding)
    finding.finding_id = "f1"
    finding.proposals = [proposal]

    agent = ResearchAgent.__new__(ResearchAgent)
    agent._hypothesis_engine = hypothesis_engine
    agent._provenance_ledger = None
    agent._wm = None
    agent._lock = __import__("threading").Lock()
    agent._findings = __import__("collections").deque([finding], maxlen=100)

    wm = WorkingMemoryBuffer(maxlen=20)
    agent.set_working_memory(wm)

    agent._get_finding = MagicMock(return_value=finding)
    agent.approve("f1")

    entries = wm.recent(10)
    assert len(entries) == 1
    assert entries[0].source == "approval"
    assert entries[0].top_score == 0.9
    assert entries[0].path_edges == [("A", "causes", "B")]


# ---------------------------------------------------------------------------
# Gap 7: Goals → ResearchAgent discovery scoring
# ---------------------------------------------------------------------------

def test_goal_hints_boost_research_discovery_potential():
    """Entity appearing in goal_hints should receive 2× potential boost."""
    import collections
    from core.research_agent import ResearchAgent

    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")

    agent = ResearchAgent.__new__(ResearchAgent)
    agent._calibrator = None
    agent._feedback_weights = {}
    agent._comm_distances = {}
    agent._goal_hints = []
    agent._lock = __import__("threading").Lock()

    cmap = {"A": 0, "B": 1}

    potential_no_hint, _ = agent._score_discovery_potential(
        gap_score=0.5, community_dist=1, G=G, u="A", v="B", cmap=cmap
    )

    agent._goal_hints = ["A"]
    potential_with_hint, _ = agent._score_discovery_potential(
        gap_score=0.5, community_dist=1, G=G, u="A", v="B", cmap=cmap
    )

    assert potential_with_hint > potential_no_hint


def test_goal_hints_no_match_unchanged():
    """Nodes not in goal_hints should not be boosted."""
    import collections
    from core.research_agent import ResearchAgent

    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")

    agent = ResearchAgent.__new__(ResearchAgent)
    agent._calibrator = None
    agent._feedback_weights = {}
    agent._comm_distances = {}
    agent._goal_hints = ["C", "D"]  # different nodes
    agent._lock = __import__("threading").Lock()

    cmap = {"A": 0, "B": 1}
    potential_hints, _ = agent._score_discovery_potential(
        gap_score=0.5, community_dist=1, G=G, u="A", v="B", cmap=cmap
    )
    agent._goal_hints = []
    potential_no_hints, _ = agent._score_discovery_potential(
        gap_score=0.5, community_dist=1, G=G, u="A", v="B", cmap=cmap
    )

    assert abs(potential_hints - potential_no_hints) < 1e-9


# ---------------------------------------------------------------------------
# Gap 8: Source weights already in SynapticDecayEngine (verified in test_synaptic_decay.py)
# Confirm source_weights dict is initialized correctly
# ---------------------------------------------------------------------------

def test_decay_source_weights_initialized():
    from core.synaptic_decay_engine import SynapticDecayEngine, _DEFAULT_SOURCE_WEIGHTS
    adapter = MagicMock()
    adapter.to_networkx = lambda: nx.DiGraph()
    eng = SynapticDecayEngine(adapter=adapter, graph=_graph_mock())
    assert eng.source_weights["dissonance"] == _DEFAULT_SOURCE_WEIGHTS["dissonance"]
    assert eng.source_weights["approval"] == _DEFAULT_SOURCE_WEIGHTS["approval"]

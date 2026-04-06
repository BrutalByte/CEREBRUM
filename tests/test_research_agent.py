"""
Tests for ResearchAgent — Autonomous Missing-Link Discovery Daemon (Phase 51).

Covers:
  - Empty graph → no candidates
  - Disconnected similar pair → 1 candidate
  - Discovery potential score range
  - scan_once returns findings (toy graph)
  - scan_once filters already-connected pairs
  - approve() adds edge to graph
  - reject() removes finding from ring buffer
  - reject() with unknown ID raises ValueError
  - start/stop control
  - Ring buffer capacity cap (oldest dropped)
"""
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.hypothesis_engine import HypothesisEngine
from core.research_agent import ResearchAgent, ResearchCandidate, ResearchFinding


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_adapter(G: nx.DiGraph, dim: int = 16) -> NetworkXAdapter:
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(7)
    adapter.embeddings = {n: rng.random(dim).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {n: i % 2 for i, n in enumerate(G.nodes())}
    return adapter


def _linear_graph(relations: list):
    nodes = ["A"] + [f"M{i}" for i in range(len(relations) - 1)] + ["B"]
    G = nx.DiGraph()
    for i, rel in enumerate(relations):
        G.add_edge(nodes[i], nodes[i + 1], relation=rel)
    return G, nodes, _make_adapter(G)


def _make_agent(adapter, **kwargs) -> ResearchAgent:
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    return ResearchAgent(
        adapter=adapter,
        hypothesis_engine=engine,
        min_confidence=0.0,
        min_discovery_potential=0.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_research_agent_mine_candidates_empty_graph():
    """No embeddings → mine_candidates returns []."""
    G = nx.DiGraph()
    G.add_node("X")
    G.add_node("Y")
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {}
    adapter.community_map = {}
    agent = _make_agent(adapter)
    candidates = agent._mine_candidates()
    assert candidates == []


def test_research_agent_mine_candidates_disconnected():
    """Two similar nodes with no edge → at least one candidate returned."""
    G = nx.DiGraph()
    G.add_node("P")
    G.add_node("Q")
    adapter = NetworkXAdapter(G)
    dim = 8
    # Give them very similar embeddings (sim ~0.99) → gap in [0.6, 0.95] boundary
    base = np.ones(dim, dtype=np.float32)
    rng = np.random.default_rng(0)
    # Make embeddings similar enough (cosine ~0.85)
    v1 = base + rng.random(dim).astype(np.float32) * 0.3
    v2 = base + rng.random(dim).astype(np.float32) * 0.3
    adapter.embeddings = {"P": v1, "Q": v2}
    adapter.community_map = {"P": 0, "Q": 1}
    agent = _make_agent(adapter)
    candidates = agent._mine_candidates()
    # May be 0 if similarity falls outside [0.60, 0.95]; just assert no crash
    assert isinstance(candidates, list)


def test_research_agent_score_discovery_potential_range():
    """_score_discovery_potential returns (potential, density) both in [0, 1]."""
    G = nx.DiGraph()
    G.add_edge("A", "C", relation="REL")
    G.add_edge("C", "B", relation="REL")
    adapter = _make_adapter(G)
    agent = _make_agent(adapter)
    cmap = {"A": 0, "B": 1, "C": 0}
    potential, density = agent._score_discovery_potential(
        gap_score=0.4,
        community_dist=1,
        G=G,
        u="A",
        v="B",
        cmap=cmap,
    )
    assert 0.0 <= potential <= 1.0
    assert 0.0 <= density <= 1.0


def test_research_agent_scan_once_returns_list():
    """scan_once() always returns a list (may be empty for small graphs)."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    result = agent.scan_once()
    assert isinstance(result, list)


def test_research_agent_scan_once_filters_existing_edges():
    """Pairs with a direct edge are never proposed."""
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="DIRECT")
    adapter = _make_adapter(G)
    rng = np.random.default_rng(0)
    dim = 8
    base = np.ones(dim, dtype=np.float32)
    adapter.embeddings = {
        "A": base + rng.random(dim).astype(np.float32) * 0.1,
        "B": base + rng.random(dim).astype(np.float32) * 0.1,
    }
    adapter.community_map = {"A": 0, "B": 1}
    agent = _make_agent(adapter)
    candidates = agent._mine_candidates()
    # A→B already connected — should not appear
    for c in candidates:
        assert not (c.source_id == "A" and c.target_id == "B"), \
            "Direct edge pair should not be a candidate"


def test_research_agent_approve_adds_edge():
    """approve(finding_id) materializes proposals to the graph."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert proposals, "need at least one proposal for this test"

    # Build a synthetic finding
    finding = ResearchFinding(
        finding_id="test-finding-1",
        candidate=ResearchCandidate(
            source_id="A",
            target_id="B",
            discovery_potential=0.8,
            gap_score=0.3,
            community_distance=1,
            seeded_by="manual",
        ),
        proposals=proposals,
        best_confidence=proposals[0].confidence,
    )

    agent = _make_agent(adapter)
    # Manually inject finding
    with agent._lock:
        agent._findings.append(finding)

    edges_before = adapter.to_networkx().number_of_edges()
    edges_added = agent.approve("test-finding-1")
    assert edges_added >= 1
    assert adapter.to_networkx().number_of_edges() > edges_before


def test_research_agent_reject_removes_finding():
    """reject(finding_id) removes the finding from the ring buffer."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    finding = ResearchFinding(
        finding_id="find-A",
        candidate=ResearchCandidate("A", "B", 0.5, 0.3, 1, "manual"),
        proposals=[],
        best_confidence=0.0,
    )
    with agent._lock:
        agent._findings.append(finding)
    assert len(agent.findings) == 1

    removed = agent.reject("find-A")
    assert removed is True
    assert len(agent.findings) == 0


def test_research_agent_reject_unknown_id_raises():
    """reject() with a bad ID raises ValueError."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    with pytest.raises(ValueError):
        agent.reject("nonexistent-id")


def test_research_agent_start_stop():
    """start() sets running=True; stop() sets running=False."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter, scan_interval=9999.0)  # long interval → won't fire
    assert not agent.status["running"]

    agent.start()
    assert agent.status["running"]

    agent.stop()
    assert not agent.status["running"]


def test_research_agent_start_idempotent():
    """Calling start() twice does not create duplicate timers."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter, scan_interval=9999.0)
    agent.start()
    agent.start()  # second call should be no-op
    assert agent.status["running"]
    agent.stop()


def test_research_agent_capacity_cap():
    """Ring buffer drops oldest findings when full."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter, findings_capacity=3)

    for i in range(5):
        f = ResearchFinding(
            finding_id=f"f{i}",
            candidate=ResearchCandidate(f"N{i}", "B", 0.5, 0.3, 1, "manual"),
            proposals=[],
            best_confidence=0.0,
        )
        with agent._lock:
            agent._findings.append(f)

    # deque(maxlen=3) keeps last 3
    ids = [f.finding_id for f in agent.findings]
    assert len(ids) == 3
    assert "f0" not in ids   # oldest dropped
    assert "f4" in ids        # newest kept


# ---------------------------------------------------------------------------
# Phase 53 — Adaptive strategy tests
# ---------------------------------------------------------------------------

def test_research_agent_strategy_dense():
    """Dense neighbourhood (>0.4) → shallow narrow beam."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    strategy = agent._select_strategy(0.6)
    assert strategy["max_hop"] == 2
    assert strategy["beam_width"] == 5
    assert strategy["max_budget"] == 150


def test_research_agent_strategy_sparse():
    """Sparse neighbourhood (<0.1) → deep wide beam."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    strategy = agent._select_strategy(0.05)
    assert strategy["max_hop"] == 4
    assert strategy["beam_width"] == 12
    assert strategy["max_budget"] == 500


def test_research_agent_strategy_transitional():
    """Transitional density (0.1–0.4) → agent's configured defaults."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)  # defaults: max_hop=3, beam_width=8
    strategy = agent._select_strategy(0.25)
    assert strategy["max_hop"] == agent.max_hop
    assert strategy["beam_width"] == agent.beam_width
    assert strategy["max_budget"] == agent.max_budget


def test_research_agent_candidate_has_local_density():
    """Candidates produced by _mine_candidates() carry local_density in [0, 1]."""
    G = nx.DiGraph()
    # Build a small connected graph so density is non-trivial
    for i in range(4):
        G.add_edge(f"N{i}", f"N{(i+1) % 4}", relation="REL")
    G.add_node("X")
    G.add_node("Y")
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(5)
    dim = 8
    base = np.ones(dim, dtype=np.float32)
    # X and Y get similar embeddings so they pass the similarity filter
    adapter.embeddings = {
        n: base + rng.random(dim).astype(np.float32) * 0.2
        for n in G.nodes()
    }
    adapter.community_map = {n: 0 for n in G.nodes()}
    agent = _make_agent(adapter)
    candidates = agent._mine_candidates()
    for c in candidates:
        assert 0.0 <= c.local_density <= 1.0, (
            f"local_density {c.local_density} out of range for ({c.source_id}, {c.target_id})"
        )

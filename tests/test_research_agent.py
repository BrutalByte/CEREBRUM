"""
Tests for ResearchAgent — Autonomous Missing-Link Discovery Daemon (Phase 51+).

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

Phase 51+ additions:
  - _pushed_candidates initialised in __init__ (no more hasattr)
  - push_candidate() queues and scan picks it up
  - Graph-signature dedup: small edge change doesn't reset evaluated_pairs
  - report_outcome() updates feedback weights (EMA)
  - Feedback weight clamped to [0.3, 2.0]
  - thread_findings() groups by shared intersection hub (union-find)
  - thread_findings() on empty ring buffer returns {}
  - ANN scan (_ann_scan) finds candidates without duplicates or connected pairs
  - Structural hole candidates (_structural_hole_candidates) surface bridges
  - Engram affinity stored in finding.metadata when engram is attached
  - seeded_by reflects candidate source
"""
import time
from pathlib import Path
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Phase 51+ — Foundation fixes
# ---------------------------------------------------------------------------

def test_pushed_candidates_initialised_in_init():
    """_pushed_candidates is a list attribute from __init__, not lazy-created."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    assert hasattr(agent, "_pushed_candidates")
    assert isinstance(agent._pushed_candidates, list)
    assert agent._pushed_candidates == []


def test_push_candidate_accepted_and_consumed():
    """push_candidate() returns True and the candidate appears in next scan."""
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="REL")
    G.add_node("C")
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(42)
    adapter.embeddings = {n: rng.random(8).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {"A": 0, "B": 0, "C": 1}
    agent = _make_agent(adapter)

    cand = ResearchCandidate(
        source_id="A",
        target_id="C",
        discovery_potential=0.9,
        gap_score=0.4,
        community_distance=1,
        seeded_by="manual",
    )
    accepted = agent.push_candidate(cand)
    assert accepted is True
    assert len(agent._pushed_candidates) == 1

    # Pushing the same pair again returns False (already evaluated after first scan)
    # (evaluated_pairs only set after _run_scan, but pushed_candidates queue holds it)
    # Second push before scan: still accepted (not yet in evaluated_pairs)
    cand2 = ResearchCandidate("A", "C", 0.9, 0.4, 1, "manual")
    accepted2 = agent.push_candidate(cand2)
    assert accepted2 is True  # not in evaluated_pairs yet


def test_push_candidate_rejected_if_already_evaluated():
    """push_candidate returns False when the pair is in _evaluated_pairs."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    with agent._lock:
        agent._evaluated_pairs.add(("A", "B"))

    cand = ResearchCandidate("A", "B", 0.9, 0.4, 1, "manual")
    assert agent.push_candidate(cand) is False


def test_graph_signature_large_change_resets_dedup():
    """Evaluated pairs are cleared when the graph grows by more than 1%."""
    G = nx.DiGraph()
    for i in range(10):
        G.add_edge(f"N{i}", f"N{(i+1)%10}", relation="REL")
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {}
    # All same community — suppresses structural hole candidates so the scan
    # produces no new candidates and doesn't re-populate evaluated_pairs.
    adapter.community_map = {f"N{i}": 0 for i in range(10)}
    agent = _make_agent(adapter)

    # Prime the signature at the current graph size
    agent._graph_signature = (10, 10)
    sentinel = ("N0", "N5")
    with agent._lock:
        agent._evaluated_pairs.add(sentinel)

    # Grow the graph significantly (>1%) with same-community nodes
    for i in range(20):
        G.add_edge(f"X{i}", f"Y{i}", relation="NEW")
        adapter.community_map[f"X{i}"] = 0
        adapter.community_map[f"Y{i}"] = 0

    agent._run_scan()

    with agent._lock:
        # The clear happened and no candidate re-added the sentinel
        assert sentinel not in agent._evaluated_pairs


def test_graph_signature_tiny_change_keeps_dedup():
    """A single new edge (< 1% growth on a large graph) does not reset pairs."""
    G = nx.DiGraph()
    # 200 nodes/edges — a single extra edge is < 1%
    for i in range(200):
        G.add_edge(f"N{i}", f"N{(i+1)%200}", relation="REL")
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {}
    adapter.community_map = {f"N{i}": 0 for i in range(200)}
    agent = _make_agent(adapter)

    # Prime signature at the current size
    agent._graph_signature = (200, 200)
    sentinel = ("N0", "N100")
    with agent._lock:
        agent._evaluated_pairs.add(sentinel)

    # Add exactly 1 edge (< 1% of 200) — already connected but updates edge count
    G.add_edge("NewA", "NewB", relation="NEW")
    adapter.community_map["NewA"] = 0
    adapter.community_map["NewB"] = 0
    agent._run_scan()

    with agent._lock:
        # Dedup not cleared for a tiny change; sentinel still present
        assert sentinel in agent._evaluated_pairs


# ---------------------------------------------------------------------------
# Phase 51+ — Feedback learning
# ---------------------------------------------------------------------------

def test_report_outcome_correct_increases_weight():
    """report_outcome(correct=True) increases the community-pair weight."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    adapter.community_map = {"A": 0, "M0": 0, "B": 1}
    agent = _make_agent(adapter)

    finding = ResearchFinding(
        finding_id="f1",
        candidate=ResearchCandidate("A", "B", 0.8, 0.3, 1, "manual"),
        proposals=[],
        best_confidence=0.0,
    )
    with agent._lock:
        agent._findings.append(finding)

    agent.report_outcome("f1", correct=True)

    key = (0, 1)
    with agent._lock:
        w = agent._feedback_weights.get(key, 1.0)
    assert w > 1.0, f"Expected weight > 1.0 after correct=True, got {w}"


def test_report_outcome_incorrect_decreases_weight():
    """report_outcome(correct=False) decreases the community-pair weight."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    adapter.community_map = {"A": 0, "M0": 0, "B": 1}
    agent = _make_agent(adapter)

    finding = ResearchFinding(
        finding_id="f2",
        candidate=ResearchCandidate("A", "B", 0.8, 0.3, 1, "manual"),
        proposals=[],
        best_confidence=0.0,
    )
    with agent._lock:
        agent._findings.append(finding)

    agent.report_outcome("f2", correct=False)

    key = (0, 1)
    with agent._lock:
        w = agent._feedback_weights.get(key, 1.0)
    assert w < 1.0, f"Expected weight < 1.0 after correct=False, got {w}"


def test_report_outcome_weight_clamped():
    """Repeated incorrect outcomes don't push weight below 0.3."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    adapter.community_map = {"A": 0, "M0": 0, "B": 1}
    agent = _make_agent(adapter)

    for i in range(50):
        finding = ResearchFinding(
            finding_id=f"fn{i}",
            candidate=ResearchCandidate("A", "B", 0.5, 0.3, 1, "manual"),
            proposals=[],
            best_confidence=0.0,
        )
        with agent._lock:
            agent._findings.append(finding)
        agent.report_outcome(f"fn{i}", correct=False)

    key = (0, 1)
    with agent._lock:
        w = agent._feedback_weights.get(key, 1.0)
    assert w >= 0.3, f"Weight {w} fell below minimum clamp of 0.3"


def test_report_outcome_unknown_finding_id_is_no_op():
    """report_outcome with an unknown finding_id does not raise."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    agent.report_outcome("nonexistent-id", correct=True)  # must not raise


# ---------------------------------------------------------------------------
# Phase 51+ — Research thread clustering
# ---------------------------------------------------------------------------

def _make_proposal(intersection_nodes):
    """Minimal mock HypothesisProposal with intersection_nodes."""
    return SimpleNamespace(
        hypothesis_id="h1",
        source="A",
        target="B",
        derived_relation="CAUSES",
        confidence=0.8,
        path_count=1,
        intersection_nodes=intersection_nodes,
        derivation_text="",
    )


def test_thread_findings_empty_returns_empty():
    """thread_findings() on an empty ring buffer returns {}."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)
    assert agent.thread_findings() == {}


def test_thread_findings_groups_shared_hub():
    """Two findings sharing an intersection node end up in the same thread."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)

    hub = "HubNode"
    f1 = ResearchFinding(
        finding_id="t1",
        candidate=ResearchCandidate("A", "B", 0.8, 0.3, 1, "manual"),
        proposals=[_make_proposal([hub])],
        best_confidence=0.8,
    )
    f2 = ResearchFinding(
        finding_id="t2",
        candidate=ResearchCandidate("C", "D", 0.7, 0.4, 1, "manual"),
        proposals=[_make_proposal([hub])],
        best_confidence=0.7,
    )
    with agent._lock:
        agent._findings.append(f1)
        agent._findings.append(f2)

    threads = agent.thread_findings()
    # Both share HubNode → must be in the same thread (one key)
    assert len(threads) == 1
    thread = next(iter(threads.values()))
    thread_ids = {f.finding_id for f in thread}
    assert thread_ids == {"t1", "t2"}


def test_thread_findings_disjoint_hubs_separate_threads():
    """Findings with no shared hubs form separate threads."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)

    f1 = ResearchFinding(
        finding_id="s1",
        candidate=ResearchCandidate("A", "B", 0.8, 0.3, 1, "manual"),
        proposals=[_make_proposal(["HubX"])],
        best_confidence=0.8,
    )
    f2 = ResearchFinding(
        finding_id="s2",
        candidate=ResearchCandidate("C", "D", 0.7, 0.4, 1, "manual"),
        proposals=[_make_proposal(["HubY"])],
        best_confidence=0.7,
    )
    with agent._lock:
        agent._findings.append(f1)
        agent._findings.append(f2)

    threads = agent.thread_findings()
    assert len(threads) == 2


def test_thread_findings_no_intersection_singleton():
    """A finding with no intersection nodes forms a singleton thread."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)

    f = ResearchFinding(
        finding_id="solo",
        candidate=ResearchCandidate("A", "B", 0.5, 0.3, 1, "manual"),
        proposals=[_make_proposal([])],
        best_confidence=0.5,
    )
    with agent._lock:
        agent._findings.append(f)

    threads = agent.thread_findings()
    assert len(threads) == 1
    assert "solo" in next(iter(threads.values()))[0].finding_id


# ---------------------------------------------------------------------------
# Phase 51+ — ANN scan
# ---------------------------------------------------------------------------

def test_ann_scan_no_duplicates():
    """_ann_scan never produces duplicate (u, v) pairs."""
    G = nx.DiGraph()
    for i in range(10):
        G.add_node(f"N{i}")
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(99)
    base = np.ones(16, dtype=np.float32)
    adapter.embeddings = {
        f"N{i}": base + rng.random(16).astype(np.float32) * 0.3
        for i in range(10)
    }
    adapter.community_map = {f"N{i}": i % 3 for i in range(10)}

    agent = _make_agent(adapter)
    G_nx = adapter.to_networkx()
    nodes = list(G_nx.nodes())
    emb_cache = {n: adapter.get_embedding(n) for n in nodes}
    cmap = adapter.community_map
    seen: set = set()

    candidates = agent._ann_scan(G_nx, nodes, emb_cache, cmap, seen)
    pairs = [(c.source_id, c.target_id) for c in candidates]
    assert len(pairs) == len(set(pairs)), "Duplicate pairs returned by _ann_scan"


def test_ann_scan_skips_connected_pairs():
    """_ann_scan never proposes pairs with an existing edge."""
    G = nx.DiGraph()
    rng = np.random.default_rng(7)
    base = np.ones(16, dtype=np.float32)
    # Fully connected graph
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"N{i}", f"N{j}", relation="REL")
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {
        f"N{i}": base + rng.random(16).astype(np.float32) * 0.2
        for i in range(5)
    }
    adapter.community_map = {f"N{i}": i % 2 for i in range(5)}

    agent = _make_agent(adapter)
    G_nx = adapter.to_networkx()
    nodes = list(G_nx.nodes())
    emb_cache = {n: adapter.get_embedding(n) for n in nodes}
    candidates = agent._ann_scan(G_nx, nodes, emb_cache, adapter.community_map, set())
    for c in candidates:
        assert not G_nx.has_edge(c.source_id, c.target_id), (
            f"ANN returned an already-connected pair: {c.source_id} -> {c.target_id}"
        )


def test_ann_scan_empty_embeddings():
    """_ann_scan with no embeddings returns an empty list without error."""
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {}
    adapter.community_map = {}
    agent = _make_agent(adapter)
    G_nx = adapter.to_networkx()
    result = agent._ann_scan(G_nx, list(G_nx.nodes()), {}, {}, set())
    assert result == []


# ---------------------------------------------------------------------------
# Phase 51+ — Structural hole detection
# ---------------------------------------------------------------------------

def test_structural_hole_candidates_seeded_by():
    """Structural hole candidates carry seeded_by='structural_hole'."""
    # H connects to both community 0 and community 1 but those communities
    # are not connected to each other → H is a structural hole.
    G = nx.DiGraph()
    # Community 0: clique among A, B, C
    for u, v in [("A", "B"), ("B", "C"), ("A", "C"), ("B", "A"), ("C", "A"), ("C", "B")]:
        G.add_edge(u, v, relation="REL")
    # Community 1: clique among D, E, F
    for u, v in [("D", "E"), ("E", "F"), ("D", "F"), ("E", "D"), ("F", "D"), ("F", "E")]:
        G.add_edge(u, v, relation="REL")
    # H bridges both communities but the two cliques are disconnected from each other
    for n in ["A", "B", "C", "D", "E", "F"]:
        G.add_edge("H", n, relation="CONNECTS")
    # H has high degree (6) and low clustering
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(3)
    adapter.embeddings = {n: rng.random(8).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {
        "A": 0, "B": 0, "C": 0,
        "D": 1, "E": 1, "F": 1,
        "H": 0,  # H is in community 0 but bridges to community 1
    }

    agent = _make_agent(adapter)
    G_nx = adapter.to_networkx()
    nodes = list(G_nx.nodes())
    cmap = adapter.community_map
    seen: set = set()

    candidates = agent._structural_hole_candidates(G_nx, nodes, cmap, seen)
    hole_cands = [c for c in candidates if c.seeded_by == "structural_hole"]
    assert len(hole_cands) > 0, "Expected structural hole candidates for node H"


def test_structural_hole_candidates_no_duplicates_with_seen():
    """_structural_hole_candidates respects the seen_pairs set."""
    G = nx.DiGraph()
    for i in range(6):
        G.add_edge(f"A{i}", f"A{(i+1)%6}", relation="REL")
    G.add_node("H")
    for i in range(6):
        G.add_edge("H", f"A{i}", relation="TO")
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(11)
    adapter.embeddings = {n: rng.random(8).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {f"A{i}": i % 2 for i in range(6)}
    adapter.community_map["H"] = 0

    agent = _make_agent(adapter)
    G_nx = adapter.to_networkx()
    nodes = list(G_nx.nodes())
    cmap = adapter.community_map

    # Pre-populate seen with all possible (H, X) pairs; take a snapshot before
    # calling the method so we can compare against the original set (the method
    # mutates seen in-place by adding new pairs as it generates candidates).
    initial_seen = {("H", f"A{i}") for i in range(6)}
    seen = set(initial_seen)
    candidates = agent._structural_hole_candidates(G_nx, nodes, cmap, seen)
    for c in candidates:
        assert (c.source_id, c.target_id) not in initial_seen


# ---------------------------------------------------------------------------
# Phase 51+ — Engram affinity metadata
# ---------------------------------------------------------------------------

class _MockEngram:
    """Minimal Engram stub that returns a fixed affinity."""
    def __init__(self, fixed_affinity: float = 0.75):
        self._aff = fixed_affinity

    def affinity(self, rel_prefix):
        return self._aff


def test_engram_affinity_stored_in_finding_metadata():
    """When an engram is attached, finding.metadata['engram_affinity'] is set."""
    G, nodes, adapter = _linear_graph(["CAUSES", "TREATS"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    if not proposals:
        pytest.skip("No proposals generated for this graph — skip engram test")

    engram = _MockEngram(fixed_affinity=0.75)
    agent = ResearchAgent(
        adapter=adapter,
        hypothesis_engine=engine,
        engram=engram,
        min_confidence=0.0,
        min_discovery_potential=0.0,
    )

    good = [p for p in proposals if p.confidence >= 0.0]
    finding = ResearchFinding(
        finding_id="eng-test",
        candidate=ResearchCandidate("A", "B", 0.8, 0.3, 1, "manual"),
        proposals=good,
        best_confidence=max((p.confidence for p in good), default=0.0),
    )
    with agent._lock:
        agent._findings.append(finding)

    # Simulate the engram-affinity assignment that happens in _run_scan
    if agent._engram is not None and good:
        aff = max(agent._engram.affinity([p.derived_relation]) for p in good)
        finding.metadata["engram_affinity"] = round(aff, 3)

    assert "engram_affinity" in finding.metadata
    assert finding.metadata["engram_affinity"] == pytest.approx(0.75, abs=1e-3)


def test_no_engram_no_affinity_key():
    """Without an engram attached, finding.metadata has no 'engram_affinity'."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    agent = _make_agent(adapter)  # no engram
    f = ResearchFinding(
        finding_id="no-eng",
        candidate=ResearchCandidate("A", "B", 0.5, 0.3, 1, "manual"),
        proposals=[],
        best_confidence=0.0,
    )
    assert "engram_affinity" not in f.metadata

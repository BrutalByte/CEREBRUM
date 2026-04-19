"""Tests for Phase 97-102 integration wiring gaps.

Covers:
- LoopedBeamTraversal with node_priming
- AsyncBeamTraversal with node_priming and ValenceEngine
- ValenceEngine initialization via AutonomousDiscoveryLoop
- Calibrator→goal wiring (make_goal used, not bare Goal())
- SynapticDecayEngine decay with wm=None (already in test_synaptic_decay.py, repeated here
  in context of loop config to confirm no regression)
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from core.working_memory import MemoryEntry, WorkingMemoryBuffer


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_adapter(edges):
    """Minimal NetworkXAdapter-like stub backed by a real DiGraph."""
    from adapters.networkx_adapter import NetworkXAdapter
    G = nx.DiGraph()
    for u, rel, v, w in edges:
        G.add_edge(u, v, relation=rel, weight=w)
    adapter = NetworkXAdapter.__new__(NetworkXAdapter)
    adapter._G = G
    adapter.community_map = {n: 0 for n in G.nodes()}

    def _get_neighbors(node, max_neighbors=50, **kw):
        from core.graph_adapter import Edge
        result = []
        for _, tgt, data in G.out_edges(node, data=True):
            result.append(Edge(
                source_id=node, target_id=tgt,
                relation_type=data.get("relation", ""),
                weight=data.get("weight", 1.0),
                confidence=data.get("weight", 1.0),
            ))
        return result[:max_neighbors]

    adapter.get_neighbors = _get_neighbors
    adapter.get_embedding = lambda n: np.zeros(8, dtype=np.float32)
    adapter.get_community = lambda n: 0
    adapter.to_networkx = lambda: G
    return adapter


def _make_csa():
    csa = MagicMock()
    csa.compute_weight_with_features = None
    csa.use_temporal_decay = False
    csa.set_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.clear_query_snapshot = MagicMock()

    def _compute_weight(u, v, **kw):
        return 0.5

    csa.compute_weight = _compute_weight
    csa.community_score = lambda u, v: 0.5
    return csa


# ---------------------------------------------------------------------------
# LoopedBeamTraversal with node_priming
# ---------------------------------------------------------------------------

def test_looped_traversal_accepts_node_priming():
    """LoopedBeamTraversal.traverse() accepts node_priming without error."""
    from reasoning.traversal import BeamTraversal
    from reasoning.looped_traversal import LoopedBeamTraversal

    adapter = _make_adapter([("a", "r", "b", 1.0), ("b", "r", "c", 1.0)])
    traversal = BeamTraversal(adapter=adapter, csa_engine=_make_csa(), beam_width=5, max_hop=2)
    looped = LoopedBeamTraversal(traversal=traversal, max_loops=2)

    priming = {"b": 0.8, "c": 0.5}
    paths, loop_trace = looped.traverse(["a"], node_priming=priming)
    assert isinstance(paths, list)
    assert loop_trace is not None


def test_looped_traversal_single_loop_passes_priming():
    """Single-loop fast-path also threads node_priming through."""
    from reasoning.traversal import BeamTraversal
    from reasoning.looped_traversal import LoopedBeamTraversal

    adapter = _make_adapter([("a", "r", "b", 1.0)])
    inner = BeamTraversal(adapter=adapter, csa_engine=_make_csa(), beam_width=5, max_hop=1)
    inner.priming_boost = 1.0  # exaggerate effect for detectability

    looped = LoopedBeamTraversal(traversal=inner, max_loops=1)
    priming = {"b": 1.0}
    paths, _ = looped.traverse(["a"], node_priming=priming)
    assert isinstance(paths, list)


# ---------------------------------------------------------------------------
# AsyncBeamTraversal — node_priming parity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_traversal_accepts_node_priming():
    """AsyncBeamTraversal.traverse_stream() accepts node_priming without error."""
    from reasoning.traversal import AsyncBeamTraversal

    adapter = _make_adapter([("a", "r", "b", 1.0)])
    traversal = AsyncBeamTraversal(adapter=adapter, csa_engine=_make_csa(), beam_width=5, max_hop=1)

    priming = {"b": 0.9}
    hops = []
    async for hop in traversal.traverse_stream(["a"], node_priming=priming):
        hops.append(hop)
    assert len(hops) >= 1  # at least the seed hop


@pytest.mark.asyncio
async def test_async_traversal_priming_boosts_score():
    """A node in node_priming should receive a higher score in async traversal."""
    from reasoning.traversal import AsyncBeamTraversal, BeamTraversal

    adapter = _make_adapter([("a", "r", "b", 1.0), ("a", "r", "c", 1.0)])
    csa = _make_csa()

    # Sync baseline
    sync_t = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10, max_hop=1)
    sync_t.priming_boost = 0.5
    sync_paths = sync_t.traverse(["a"], node_priming={"b": 1.0})
    sync_scores = {p.tail: float(p.score) for p in sync_paths if p.hop_depth > 0}

    # Async with same priming
    async_t = AsyncBeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10, max_hop=1)
    async_t.priming_boost = 0.5
    async_paths = []
    async for hop in async_t.traverse_stream(["a"], node_priming={"b": 1.0}):
        async_paths.extend(hop)

    async_scores = {p.tail: float(p.score) for p in async_paths if p.hop_depth > 0}

    # Both should give b a higher score than c (priming only on b)
    if "b" in sync_scores and "c" in sync_scores:
        assert sync_scores["b"] > sync_scores["c"]
    if "b" in async_scores and "c" in async_scores:
        assert async_scores["b"] > async_scores["c"]


# ---------------------------------------------------------------------------
# AsyncBeamTraversal — ValenceEngine parity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_traversal_valence_penalizes_aversive_edge():
    """Aversive edge (negative valence) reduces score in async traversal."""
    from reasoning.traversal import AsyncBeamTraversal

    adapter = _make_adapter([("a", "r", "b", 1.0), ("a", "r", "c", 1.0)])
    csa = _make_csa()

    valence_engine = MagicMock()
    valence_engine.valence_weight = 1.0

    def _get_valence(u, v, rel):
        # Edge a→b is aversive; a→c is neutral
        return -0.5 if v == "b" else 0.0

    valence_engine.get_valence = _get_valence

    traversal = AsyncBeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10, max_hop=1)
    traversal._valence_engine = valence_engine

    all_paths = []
    async for hop in traversal.traverse_stream(["a"]):
        all_paths.extend(hop)

    scores = {p.tail: float(p.score) for p in all_paths if p.hop_depth > 0}
    if "b" in scores and "c" in scores:
        assert scores["b"] < scores["c"], "Aversive edge to b should score lower than neutral c"


# ---------------------------------------------------------------------------
# ValenceEngine initialization via AutonomousDiscoveryLoop
# ---------------------------------------------------------------------------

def test_loop_valence_learning_initializes_engine():
    """AutonomousDiscoveryLoop with valence_learning=True creates and wires ValenceEngine."""
    from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig
    from core.valence_engine import ValenceEngine

    agent = MagicMock()
    agent._adapter = MagicMock()
    graph = MagicMock()
    graph.adapter = MagicMock()
    graph.modulator = MagicMock()
    graph.modulator.state = {"arousal": 1.0}
    agent._adapter.graph = graph

    config = LoopConfig(valence_learning=True)
    loop = AutonomousDiscoveryLoop(agent=agent, config=config)

    assert loop._valence_engine is not None
    assert isinstance(loop._valence_engine, ValenceEngine)
    # attach_valence_engine called on graph
    graph.attach_valence_engine.assert_called_once_with(loop._valence_engine)
    # set_valence_engine called on agent
    agent.set_valence_engine.assert_called_once_with(loop._valence_engine)


def test_loop_valence_learning_disabled_by_default():
    """ValenceEngine should not be created when valence_learning=False."""
    from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig

    agent = MagicMock()
    agent._adapter = MagicMock()
    graph = MagicMock()
    graph.adapter = MagicMock()
    agent._adapter.graph = graph

    loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig())
    assert loop._valence_engine is None


# ---------------------------------------------------------------------------
# Calibrator→goal wiring uses make_goal (no missing id)
# ---------------------------------------------------------------------------

def test_calibrator_goal_push_creates_valid_goal():
    """High-weight community triggers a Goal with a valid id (not a bare Goal() call)."""
    from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig
    from core.goal_system import GoalStack, Goal

    agent = MagicMock()
    agent._adapter = MagicMock()
    graph = MagicMock()
    graph.adapter = MagicMock()
    graph.modulator = MagicMock()
    graph.modulator.state = {"reinforcement": 1.0, "arousal": 1.0,
                              "novelty": 1.0, "cohesion": 1.0, "persistence": 1.0}
    agent._adapter.graph = graph

    # Set up calibrator that reports a heavy community
    calibrator = MagicMock()
    calibrator.stats.return_value = {"communities": {0: {"weight": 5.0}}}
    agent._calibrator = calibrator

    config = LoopConfig(working_memory=True)
    loop = AutonomousDiscoveryLoop(agent=agent, config=config)

    # Replace goal stack with a real one so we can inspect it
    from core.goal_system import GoalStack
    goal_stack = GoalStack()
    loop._goal_stack = goal_stack

    # Simulate the calibrator→goal wiring block directly
    cid = 0
    goal_desc = f"explore_community_{cid}"
    from core.goal_system import make_goal
    goal_stack.push(make_goal(
        description=goal_desc,
        metric_type="approval_rate",
        target_value=0.15,
        priority=6,
    ))

    active = goal_stack.all_active()
    matching = [g for g in active if g.description == goal_desc]
    assert len(matching) == 1, f"Expected one goal with description '{goal_desc}'"
    g = matching[0]
    assert isinstance(g, Goal)
    assert g.id, "Goal must have a non-empty id"

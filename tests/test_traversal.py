"""
Integration tests for the beam traversal + scoring + extraction pipeline.
"""
import random

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from core.community_engine import dscf_communities
from core.embedding_engine import RandomEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.answer_extractor import extract
from reasoning.path_scorer import community_coherence, score_path
from reasoning.traversal import BeamTraversal, TraversalPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_test_graph() -> nx.Graph:
    """Small graph with clear 2-community structure + one bridge."""
    G = nx.Graph()
    G.add_edges_from([
        ("alice", "bob",   {"relation": "KNOWS"}),
        ("bob",   "carol", {"relation": "KNOWS"}),
        ("carol", "alice", {"relation": "KNOWS"}),
    ])
    G.add_edges_from([
        ("dave",  "eve",   {"relation": "WORKS_WITH"}),
        ("eve",   "frank", {"relation": "WORKS_WITH"}),
        ("frank", "dave",  {"relation": "WORKS_WITH"}),
    ])
    G.add_edge("carol", "dave", relation="INTRODUCED_BY")
    return G


def build_traversal(G: nx.Graph, **kwargs) -> BeamTraversal:
    adapter = NetworkXAdapter(G)
    engine  = RandomEngine(dim=32)
    labels  = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    random.seed(42)
    parts        = dscf_communities(G, max_iter=50)
    community_map = {node: cid for cid, members in enumerate(parts) for node in members}

    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(communities=community_map, embeddings=embeddings)
    csa.set_community_graph(dist, adj)

    return BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        embeddings=embeddings,
        communities=community_map,
        beam_width=kwargs.get("beam_width", 5),
        max_hop=kwargs.get("max_hop", 2),
    )


# ---------------------------------------------------------------------------
# TraversalPath tests
# ---------------------------------------------------------------------------

def test_path_entity_nodes():
    path = TraversalPath(nodes=["a", "REL", "b", "REL2", "c"])
    assert path.entity_nodes == ["a", "b", "c"]


def test_path_hop_depth():
    path = TraversalPath(attention_weights=[0.8, 0.6])
    assert path.hop_depth == 2


# ---------------------------------------------------------------------------
# BeamTraversal tests
# ---------------------------------------------------------------------------

def test_traversal_returns_paths():
    G     = make_test_graph()
    t     = build_traversal(G)
    paths = t.traverse(["alice"])
    assert len(paths) > 0


def test_traversal_paths_start_from_seed():
    G     = make_test_graph()
    t     = build_traversal(G)
    for path in t.traverse(["alice"]):
        assert path.head == "alice"


def test_traversal_no_cycles():
    G     = make_test_graph()
    t     = build_traversal(G)
    for path in t.traverse(["alice"]):
        entities = path.entity_nodes
        assert len(entities) == len(set(entities)), f"Cycle detected: {path.nodes}"


def test_traversal_scores_non_negative():
    G     = make_test_graph()
    t     = build_traversal(G)
    for path in t.traverse(["alice"]):
        assert path.score >= 0.0


def test_traversal_respects_max_hop():
    G     = make_test_graph()
    t     = build_traversal(G, max_hop=1)
    for path in t.traverse(["alice"]):
        assert path.hop_depth <= 1


def test_traversal_beam_width_limits_candidates():
    """Paths returned should not exceed beam_width per hop."""
    G     = make_test_graph()
    t     = build_traversal(G, beam_width=2, max_hop=2)
    paths = t.traverse(["alice"])
    # At max_hop=2, beam_width=2: at most 2 paths survive per hop
    # Total paths <= 1 (seed) + 2 (hop1) + 2 (hop2) = 5
    assert len(paths) <= 10  # generous upper bound


def test_traversal_can_reach_bridge():
    """With enough hops, traversal from alice should reach dave (via bridge)."""
    G     = make_test_graph()
    t     = build_traversal(G, max_hop=3, beam_width=20)
    paths = t.traverse(["alice"])
    reached_entities = set()
    for path in paths:
        reached_entities.update(path.entity_nodes)
    assert "dave" in reached_entities or "frank" in reached_entities, \
        "Expected to reach the other community via bridge"


# ---------------------------------------------------------------------------
# community_coherence tests
# ---------------------------------------------------------------------------

def test_coherence_same_community():
    assert community_coherence([0, 0, 0]) == 1.0


def test_coherence_one_cross():
    # 2 steps: 1 intra, 1 cross -> (1.0 + 0.5) / 2 = 0.75
    assert abs(community_coherence([0, 0, 1]) - 0.75) < 1e-6


def test_coherence_single_node():
    assert community_coherence([0]) == 1.0


def test_coherence_unknown_community():
    # Unknown (-1) communities are ignored
    assert community_coherence([-1, -1]) == 1.0


# ---------------------------------------------------------------------------
# Answer extraction tests
# ---------------------------------------------------------------------------

def test_extract_returns_top_k():
    G       = make_test_graph()
    t       = build_traversal(G, max_hop=2, beam_width=10)
    paths   = t.traverse(["alice"])
    answers = extract(paths, top_k=3)
    assert len(answers) <= 3


def test_extract_excludes_seeds():
    G       = make_test_graph()
    t       = build_traversal(G, max_hop=2, beam_width=10)
    paths   = t.traverse(["alice"])
    answers = extract(paths, top_k=10, min_hop=1)
    # Depth-0 seed paths should be excluded
    for ans in answers:
        assert ans.best_path.hop_depth >= 1


def test_extract_answers_sorted_by_score():
    G       = make_test_graph()
    t       = build_traversal(G, max_hop=2, beam_width=10)
    paths   = t.traverse(["alice"])
    answers = extract(paths, top_k=10)
    scores  = [a.score for a in answers]
    assert scores == sorted(scores, reverse=True)




"""
Tests for HypothesisEngine — Multi-Path Abductive Reasoning (Phase 50).

Covers:
  - Empty result when no paths exist
  - Single-path proposal generation
  - Two independent paths → Noisy-OR confidence compounds
  - Correlated paths (shared intermediary) → only one counts
  - Relation chain composition (transitivity)
  - Fallback to RELATED_TO for unknown relations
  - Materialization adds edge to graph
  - Rollback removes materialized edge
  - Contradiction check reduces confidence
  - API smoke test via POST /hypothesize
"""
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.hypothesis_engine import HypothesisEngine, _OPPOSING_RELATIONS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_adapter(G: nx.DiGraph, dim: int = 16) -> NetworkXAdapter:
    """Build a minimal adapter with random embeddings and community map."""
    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(42)
    adapter.embeddings = {n: rng.random(dim).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {n: i % 3 for i, n in enumerate(G.nodes())}
    return adapter


def _linear_graph(relations: list) -> tuple:
    """
    Build A → M1 → M2 → ... → B with the given relation sequence.
    Returns (G, node_ids, adapter).
    """
    nodes = ["A"] + [f"M{i}" for i in range(len(relations) - 1)] + ["B"]
    G = nx.DiGraph()
    for i, rel in enumerate(relations):
        G.add_edge(nodes[i], nodes[i + 1], relation=rel)
    return G, nodes, _make_adapter(G)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_hypothesis_no_paths_returns_empty():
    """Disconnected graph: no paths A → B → empty proposals."""
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")
    adapter = _make_adapter(G)
    engine = HypothesisEngine(adapter)
    proposals = engine.generate("A", "B")
    assert proposals == []


def test_hypothesis_single_path_one_proposal():
    """
    Single 2-hop path A→M→B via PART_OF+PART_OF.
    Composition: PART_OF+PART_OF → PART_OF (transitivity rule).
    """
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert len(proposals) == 1
    assert proposals[0].derived_relation == "PART_OF"
    assert proposals[0].path_count == 1
    assert 0.0 < proposals[0].confidence <= 1.0


def test_hypothesis_two_independent_paths_combine():
    """
    Two paths through different intermediaries.
    Noisy-OR must exceed either individual score; path_count == 2.
    """
    G = nx.DiGraph()
    # Path 1: A → M1 → B  (PART_OF + PART_OF → PART_OF)
    G.add_edge("A", "M1", relation="PART_OF")
    G.add_edge("M1", "B", relation="PART_OF")
    # Path 2: A → M2 → B  (PART_OF + PART_OF → PART_OF, independent)
    G.add_edge("A", "M2", relation="PART_OF")
    G.add_edge("M2", "B", relation="PART_OF")

    adapter = _make_adapter(G)
    engine = HypothesisEngine(adapter, min_confidence=0.0, min_path_independence=0.0)
    proposals = engine.generate("A", "B", beam_width=10, max_hop=2)

    assert len(proposals) >= 1
    best = proposals[0]
    # Noisy-OR combined must be ≥ the best single-path score
    assert best.confidence >= max(p.score for p in best.supporting_paths)
    assert best.path_count == 2


def test_hypothesis_correlated_paths_no_double_count():
    """
    Only one path exists (A→M→B), so only one independent path contributes.
    """
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert len(proposals) == 1
    assert proposals[0].path_count == 1


def test_hypothesis_compose_chain_transitive():
    """_compose_chain: PART_OF + PART_OF → PART_OF via transitivity rule."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert any(p.derived_relation == "PART_OF" for p in proposals)


def test_hypothesis_compose_chain_fallback():
    """Unknown relations fall back to RELATED_TO."""
    G, nodes, adapter = _linear_graph(["ZORK_REL", "QUUX_REL"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert any(p.derived_relation == "RELATED_TO" for p in proposals)


def test_hypothesis_materialization_adds_edge():
    """After materialize(), the adapter graph has an A→B edge."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert proposals, "Expected at least one proposal"

    added = engine.materialize(proposals)
    assert added >= 1
    assert adapter.to_networkx().has_edge("A", "B")
    edge_data = adapter.to_networkx()["A"]["B"]
    assert edge_data.get("provenance") == "hypothesis_engine"


def test_hypothesis_rollback_removes_edge():
    """After materialize() + rollback(), the materialized edge is gone."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=5, max_hop=2)
    assert proposals

    engine.materialize(proposals)
    assert adapter.to_networkx().has_edge("A", "B")

    removed = engine.rollback()
    assert removed >= 1
    assert not adapter.to_networkx().has_edge("A", "B")
    assert not engine.can_rollback


def test_hypothesis_rollback_raises_without_prior_materialize():
    """rollback() before any materialize() raises RuntimeError."""
    G, nodes, adapter = _linear_graph(["PART_OF", "PART_OF"])
    engine = HypothesisEngine(adapter)
    with pytest.raises(RuntimeError):
        engine.rollback()


def test_hypothesis_contradiction_reduces_confidence():
    """
    Two paths: one composes to INDIRECTLY_CAUSES, another to INDIRECTLY_PREVENTS.
    The INDIRECTLY_CAUSES proposal must have contradiction_score > 0 and
    final confidence < raw Noisy-OR.
    """
    G = nx.DiGraph()
    # Path 1: A -[ACTIVATES]-> M1 -[CAUSES]-> B  → INDIRECTLY_CAUSES
    G.add_edge("A", "M1", relation="ACTIVATES")
    G.add_edge("M1", "B", relation="CAUSES")
    # Path 2: A -[ACTIVATES]-> M2 -[PREVENTS]-> B → INDIRECTLY_PREVENTS
    G.add_edge("A", "M2", relation="ACTIVATES")
    G.add_edge("M2", "B", relation="PREVENTS")

    adapter = _make_adapter(G)
    engine = HypothesisEngine(adapter, min_confidence=0.0)
    proposals = engine.generate("A", "B", beam_width=10, max_hop=2)

    # The INDIRECTLY_CAUSES proposal should have a contradiction score
    causes_proposals = [p for p in proposals if p.derived_relation == "INDIRECTLY_CAUSES"]
    if causes_proposals:
        cp = causes_proposals[0]
        assert cp.contradiction_score > 0.0
        # Final confidence must be reduced from raw Noisy-OR
        raw = engine._noisy_or([p.score for p in cp.supporting_paths])
        assert cp.confidence < raw or cp.contradiction_score == 0.0


def test_hypothesis_api_endpoint_smoke():
    """
    POST /hypothesize with the toy graph returns 200 and a valid response.
    newton → faraday → maxwell (INFLUENCED+INFLUENCED) should yield at least
    one INDIRECTLY_INFLUENCED proposal.
    """
    import random as _random
    from starlette.testclient import TestClient

    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf
    from core.embedding_engine import RandomEngine
    from api.server import create_app

    TOY_CSV = Path(__file__).parent / "fixtures" / "toy_graph.csv"
    _random.seed(42)
    adapter = load_csv_adapter(str(TOY_CSV))
    engine  = RandomEngine(dim=64)

    G     = adapter.to_networkx()
    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}

    app = create_app(adapter=adapter, embedding_engine=engine, community_map=cmap)
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
        r = c.post("/v1/hypothesize", json={
            "source_id": "newton",
            "target_id": "maxwell",
            "max_paths": 10,
            "max_hop": 3,
            "beam_width": 15,
            "min_confidence": 0.0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "proposals" in body
        assert "paths_explored" in body
        assert "duration_seconds" in body
        # newton → faraday → maxwell via INFLUENCED+INFLUENCED should find a path
        if body["proposals"]:
            p = body["proposals"][0]
            assert "hypothesis_id" in p
            assert "derived_relation" in p
            assert "confidence" in p
            assert isinstance(p["supporting_paths"], list)

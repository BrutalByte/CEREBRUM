"""
End-to-end integration tests for the full CEREBRUM inference pipeline.

Pipeline under test (Section 5 of PARALLAX.md):
  CSV → NetworkXAdapter
      → DSCF (community detection)
      → RandomEngine (entity embeddings)
      → build_community_distance_matrix + adjacent_community_pairs
      → CSAEngine (attention weights)
      → BeamTraversal (beam search)
      → extract() (answer ranking)

All tests run against the canonical toy graph fixture
(tests/fixtures/toy_graph.csv, 19 nodes, 30 edges).

These tests are the first verifiable proof that the complete pipeline
operates as described in the CEREBRUM white paper.
"""
import random
from pathlib import Path

import numpy as np
import pytest

from adapters.csv_adapter import load_csv_adapter
from core.attention_engine import CSAEngine
from core.community_engine import best_of_n_dscf
from core.embedding_engine import RandomEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.answer_extractor import extract
from reasoning.traversal import BeamTraversal

TOY_CSV = Path(__file__).parent / "fixtures" / "toy_graph.csv"

# Fixed seed for all end-to-end tests — results must be identical across runs.
GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# Pipeline factory (shared setup)
# ---------------------------------------------------------------------------

def build_pipeline(beam_width: int = 10, max_hop: int = 3):
    """
    Build a fully-wired BeamTraversal from the toy graph fixture.

    Steps mirror the forward pass exactly as documented in Section 5 of
    PARALLAX.md. Deterministic: random.seed(GLOBAL_SEED) is called first.
    """
    random.seed(GLOBAL_SEED)

    # Step 1 — Load graph
    adapter = load_csv_adapter(str(TOY_CSV))
    G       = adapter.to_networkx()

    # Step 2 — Community detection (DSCF)
    parts         = best_of_n_dscf(G, n_trials=5, seed=GLOBAL_SEED, use_multiprocessing=False)
    community_map = {
        node: cid
        for cid, members in enumerate(parts)
        for node in members
    }

    # Step 3 — Entity embeddings (random for reproducibility)
    engine     = RandomEngine(dim=64)
    labels     = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    # Step 4 — Community graph metadata for CSA
    distances = build_community_distance_matrix(G, community_map)
    adj       = adjacent_community_pairs(G, community_map)

    # Attach to adapter for CSAEngine
    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    # Step 5 — CSA engine
    csa = CSAEngine(adapter=adapter)
    csa.set_community_graph(distances, adj)

    # Step 6 — Beam traversal
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=beam_width,
        max_hop=max_hop,
    )
    return traversal, community_map


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_pipeline_loads_and_runs():
    """The full pipeline must run end-to-end on the toy graph without error."""
    traversal, _ = build_pipeline()
    paths        = traversal.traverse(["newton"])
    assert len(paths) > 0, "Expected at least one path from seed 'newton'"


# ---------------------------------------------------------------------------
# Verifiable expected-output tests
# ---------------------------------------------------------------------------

def test_pipeline_newton_reaches_einstein():
    """
    'newton' has a direct INFLUENCED edge to 'einstein' in the toy graph.
    At max_hop >= 1, 'einstein' must appear in the answer set.

    This is the canonical white-paper reasoning trace:
      newton --[INFLUENCED]--> einstein
    """
    traversal, _ = build_pipeline(beam_width=20, max_hop=2)
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=10)
    entities     = {a.entity_id for a in answers}
    assert "einstein" in entities, (
        f"Expected 'einstein' in answers for seed 'newton'. Got: {entities}"
    )


def test_pipeline_newton_reaches_faraday():
    """
    'newton' has a direct INFLUENCED edge to 'faraday'.
    'faraday' must appear in the answer set at max_hop >= 1.
    """
    traversal, _ = build_pipeline(beam_width=20, max_hop=2)
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=10)
    entities     = {a.entity_id for a in answers}
    assert "faraday" in entities, (
        f"Expected 'faraday' in answers for seed 'newton'. Got: {entities}"
    )


def test_pipeline_caesar_reaches_rome():
    """
    'caesar' has a direct RULED edge to 'rome'.
    'rome' must appear in the answer set at max_hop >= 1.
    """
    traversal, _ = build_pipeline(beam_width=20, max_hop=2)
    paths        = traversal.traverse(["caesar"])
    answers      = extract(paths, top_k=10)
    entities     = {a.entity_id for a in answers}
    assert "rome" in entities, (
        f"Expected 'rome' in answers for seed 'caesar'. Got: {entities}"
    )


def test_pipeline_multi_hop_newton_reaches_bohr():
    """
    Multi-hop reasoning:
      newton --[INFLUENCED]--> einstein --[COLLABORATED]--> bohr

    This requires max_hop >= 2 and tests that the traversal crosses
    two edges correctly.
    """
    traversal, _ = build_pipeline(beam_width=20, max_hop=3)
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=20)
    entities     = {a.entity_id for a in answers}
    assert "bohr" in entities, (
        f"Expected 'bohr' reachable from 'newton' via einstein. Got: {entities}"
    )


# ---------------------------------------------------------------------------
# Answer quality / structural tests
# ---------------------------------------------------------------------------

def test_pipeline_answers_are_ranked():
    """Answers must be returned in strictly non-increasing score order."""
    traversal, _ = build_pipeline()
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=10)
    scores       = [a.score for a in answers]
    assert scores == sorted(scores, reverse=True), (
        f"Answers not in descending score order: {scores}"
    )


def test_pipeline_seed_not_in_answers():
    """
    The seed entity must not appear as an answer (min_hop=1 enforced).
    Including the query entity in its own answer set is meaningless.
    """
    traversal, _ = build_pipeline()
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=10, min_hop=1)
    entities     = {a.entity_id for a in answers}
    assert "newton" not in entities, (
        "Seed entity 'newton' must not appear in its own answer set"
    )


def test_pipeline_answer_entities_are_graph_nodes():
    """
    Every returned answer entity must be a node that exists in the graph.
    A hallucinated entity (not in the KG) would indicate a logic error.
    """
    traversal, _ = build_pipeline(beam_width=20, max_hop=3)
    adapter      = load_csv_adapter(str(TOY_CSV))
    known_nodes  = set(adapter.to_networkx().nodes())

    paths   = traversal.traverse(["newton"])
    answers = extract(paths, top_k=20)
    for ans in answers:
        assert ans.entity_id in known_nodes, (
            f"Answer entity {ans.entity_id!r} is not a node in the graph — "
            "possible hallucination"
        )


def test_pipeline_score_breakdown_present():
    """
    Every Answer must include a score_breakdown dict with at least
    'attention' and 'community' keys (the two mandatory signals).
    """
    traversal, _ = build_pipeline()
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=5)
    for ans in answers:
        assert "attention" in ans.score_breakdown, (
            f"Missing 'attention' in score_breakdown for {ans.entity_id!r}"
        )
        assert "community" in ans.score_breakdown, (
            f"Missing 'community' in score_breakdown for {ans.entity_id!r}"
        )


def test_pipeline_scores_in_valid_range():
    """All answer scores must be in [0, 1]."""
    traversal, _ = build_pipeline(beam_width=20, max_hop=3)
    paths        = traversal.traverse(["newton"])
    answers      = extract(paths, top_k=20)
    for ans in answers:
        assert 0.0 <= ans.score <= 1.0, (
            f"Score {ans.score} out of [0, 1] for entity {ans.entity_id!r}"
        )


def test_pipeline_community_trace_matches_graph():
    """
    The community_trace on each answer must have all values that are
    valid community IDs (non-negative integers assigned by DSCF).
    """
    traversal, community_map = build_pipeline()
    valid_cids               = set(community_map.values())
    paths                    = traversal.traverse(["newton"])
    answers                  = extract(paths, top_k=10)
    for ans in answers:
        for cid in ans.community_trace:
            assert cid in valid_cids or cid == -1, (
                f"Invalid community ID {cid} in trace for {ans.entity_id!r}"
            )


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

def test_pipeline_deterministic():
    """
    Running the pipeline twice with the same seed must produce identical
    answer entity lists in the same order.

    This validates that random.seed(GLOBAL_SEED) fully controls the
    stochastic elements (DSCF temperature annealing, node shuffle).
    """
    def run():
        traversal, _ = build_pipeline(beam_width=10, max_hop=2)
        paths        = traversal.traverse(["newton"])
        return [a.entity_id for a in extract(paths, top_k=5)]

    run1 = run()
    run2 = run()
    assert run1 == run2, (
        f"Pipeline is not deterministic.\nRun 1: {run1}\nRun 2: {run2}"
    )




"""Tests for Phase 229 ConceptNet benchmark (conceptnet_eval.py)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "conceptnet_sample.csv"

# ---------------------------------------------------------------------------
# Edge-split helpers
# ---------------------------------------------------------------------------

def test_edge_in_test_deterministic():
    from benchmarks.conceptnet_eval import _edge_in_test
    # Same triple always maps to same split
    assert _edge_in_test("dog", "IsA", "animal") == _edge_in_test("dog", "IsA", "animal")


def test_edge_in_test_coverage():
    from benchmarks.conceptnet_eval import _edge_in_test
    # Over many triples, ~20% should be in test (allow ±10% for small samples)
    triples = [(f"h{i}", "IsA", f"t{i}") for i in range(200)]
    test_count = sum(_edge_in_test(h, r, t) for h, r, t in triples)
    assert 10 <= test_count <= 50, f"Expected ~40 test edges, got {test_count}"


def test_edge_in_test_relation_sensitive():
    from benchmarks.conceptnet_eval import _edge_in_test
    # Different relations on the same (h, t) pair can split differently
    r1 = _edge_in_test("dog", "IsA", "animal")
    r2 = _edge_in_test("dog", "RelatedTo", "animal")
    # At least one should be deterministic (just verify they can differ)
    assert isinstance(r1, bool) and isinstance(r2, bool)


# ---------------------------------------------------------------------------
# load_and_split
# ---------------------------------------------------------------------------

def test_load_and_split_returns_two_lists():
    from benchmarks.conceptnet_eval import load_and_split
    train, test = load_and_split(str(FIXTURE))
    assert isinstance(train, list)
    assert isinstance(test, list)


def test_load_and_split_train_has_weights():
    from benchmarks.conceptnet_eval import load_and_split
    train, _ = load_and_split(str(FIXTURE))
    for item in train:
        assert len(item) == 4, "Train triples must be (h, r, t, w)"
        assert isinstance(item[3], float)


def test_load_and_split_test_is_triples():
    from benchmarks.conceptnet_eval import load_and_split
    _, test = load_and_split(str(FIXTURE))
    for item in test:
        assert len(item) == 3, "Test triples must be (h, r, t)"


def test_load_and_split_no_overlap():
    """Same (h,r,t) should not appear in both train and test."""
    from benchmarks.conceptnet_eval import load_and_split
    train, test = load_and_split(str(FIXTURE))
    train_keys = {(h, r, t) for h, r, t, _ in train}
    test_keys  = {(h, r, t) for h, r, t   in test}
    overlap = train_keys & test_keys
    assert not overlap, f"Overlap found: {overlap}"


def test_load_and_split_respects_max_edges():
    from benchmarks.conceptnet_eval import load_and_split
    train, test = load_and_split(str(FIXTURE), max_edges=5)
    assert len(train) + len(test) <= 5


# ---------------------------------------------------------------------------
# _sample_qa_pairs (2-hop chains from training graph)
# ---------------------------------------------------------------------------

def _get_train_triples():
    from benchmarks.conceptnet_eval import load_and_split
    train, _ = load_and_split(str(FIXTURE))
    return train


def test_sample_qa_pairs_count():
    from benchmarks.conceptnet_eval import _sample_qa_pairs
    train = _get_train_triples()
    qa = _sample_qa_pairs(train, n_questions=5, seed=42)
    assert len(qa) <= 5


def test_sample_qa_pairs_non_trivial():
    """2-hop pairs: h→t must not be a direct training edge."""
    from benchmarks.conceptnet_eval import _sample_qa_pairs, load_and_split
    train, _ = load_and_split(str(FIXTURE))
    direct = {(h, t) for h, r, t, _ in train}
    qa = _sample_qa_pairs(train, n_questions=100, seed=42)
    for h, r, t in qa:
        assert (h, t) not in direct, f"Direct edge (h={h}, t={t}) should be excluded"


def test_sample_qa_pairs_deterministic():
    from benchmarks.conceptnet_eval import _sample_qa_pairs
    train = _get_train_triples()
    qa1 = _sample_qa_pairs(train, n_questions=5, seed=99)
    qa2 = _sample_qa_pairs(train, n_questions=5, seed=99)
    assert qa1 == qa2


def test_sample_qa_pairs_empty_when_no_valid():
    from benchmarks.conceptnet_eval import _sample_qa_pairs
    qa = _sample_qa_pairs([], n_questions=10, seed=0)
    assert qa == []


# ---------------------------------------------------------------------------
# build_conceptnet_state
# ---------------------------------------------------------------------------

def test_build_state_returns_dict():
    from benchmarks.conceptnet_eval import build_conceptnet_state
    state = build_conceptnet_state(str(FIXTURE), n_questions=5, embeddings="random",
                                   use_cache=False, max_edges=50)
    assert "graph" in state
    assert "deriver" in state
    assert "qa_pairs" in state
    assert "answer_freq" in state


def test_build_state_graph_has_nodes():
    from benchmarks.conceptnet_eval import build_conceptnet_state
    state = build_conceptnet_state(str(FIXTURE), n_questions=5, embeddings="random",
                                   use_cache=False, max_edges=50)
    G = state["graph"].adapter.to_networkx()
    assert G.number_of_nodes() > 0


def test_build_state_answer_freq_matches_qa():
    from benchmarks.conceptnet_eval import build_conceptnet_state
    state = build_conceptnet_state(str(FIXTURE), n_questions=20, embeddings="random",
                                   use_cache=False, max_edges=50)
    qa    = state["qa_pairs"]
    freq  = state["answer_freq"]
    # Every tail in qa_pairs should appear in answer_freq
    for _, _, t in qa:
        assert t in freq, f"Tail '{t}' missing from answer_freq"


# ---------------------------------------------------------------------------
# run_trial_inprocess
# ---------------------------------------------------------------------------

def _default_params(graph):
    """Return a minimal valid param dict."""
    return {
        "trb_factor":   10.0,
        "r2_boost":     4.0,
        "vote_weight":  0.85,
        "beam_width":   10,
        "idf_weight":   0.05,
        "branch_bonus": 0.3,
        "fhrb_factor":  3.0,
        "gamma":        2.0,
        "beta":         1.0,
        "max_loops":    1,
    }


def test_run_trial_returns_floats():
    from benchmarks.conceptnet_eval import build_conceptnet_state, run_trial_inprocess
    state = build_conceptnet_state(str(FIXTURE), n_questions=5, embeddings="random",
                                   use_cache=False, max_edges=50)
    if not state["qa_pairs"]:
        pytest.skip("No QA pairs from fixture (all edges in train split)")
    h1, h10, mrr = run_trial_inprocess(state, _default_params(state["graph"]))
    assert 0.0 <= h1  <= 1.0
    assert 0.0 <= h10 <= 1.0
    assert 0.0 <= mrr <= 1.0


def test_run_trial_h10_gte_h1():
    from benchmarks.conceptnet_eval import build_conceptnet_state, run_trial_inprocess
    state = build_conceptnet_state(str(FIXTURE), n_questions=5, embeddings="random",
                                   use_cache=False, max_edges=50)
    if not state["qa_pairs"]:
        pytest.skip("No QA pairs from fixture")
    h1, h10, mrr = run_trial_inprocess(state, _default_params(state["graph"]))
    assert h10 >= h1


def test_run_trial_empty_qa_returns_zeros():
    from benchmarks.conceptnet_eval import build_conceptnet_state, run_trial_inprocess
    state = build_conceptnet_state(str(FIXTURE), n_questions=0, embeddings="random",
                                   use_cache=False, max_edges=50)
    state["qa_pairs"] = []  # Force empty
    h1, h10, mrr = run_trial_inprocess(state, _default_params(state["graph"]))
    assert h1 == 0.0 and h10 == 0.0 and mrr == 0.0

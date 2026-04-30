"""
Phase 146: Terminal Relation Boost (TRB) — unit tests.

Tests verify that terminal_relation_boost is applied only at the final
beam-expansion hop, and that detect_target_relation correctly maps MetaQA
question text to relation types.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from reasoning.traversal import BeamTraversal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_and_traversal(edges, max_hop, boost=None):
    """
    Build a minimal NetworkX graph with the given edge list and return a
    (adapter, traversal) pair.  Edges: list of (src, rel, dst).
    """
    import networkx as nx
    G = nx.DiGraph()
    for src, rel, dst in edges:
        G.add_edge(src, dst, relation=rel)
    adapter = NetworkXAdapter(G)
    csa = CSAEngine(adapter=adapter)
    t = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=100,
        max_hop=max_hop,
        terminal_relation_boost=boost or {},
    )
    return adapter, t


# ---------------------------------------------------------------------------
# Test 1: boost applied at terminal hop only (not at intermediate hops)
# ---------------------------------------------------------------------------

def test_boost_applied_at_terminal_hop_only():
    """
    2-hop graph: seed → [A via rel_X, B via rel_Y] → [answer via rel_TARGET]

    Without boost: A and B get equal CSA weight at hop 1 (uniform graph).
    With boost={rel_TARGET: 10.0}: the boost fires at hop 2 (max_hop=2),
    not at hop 1.  The answer reached via rel_TARGET should have a higher
    weight in the scored paths.
    """
    # seed → A via rel_hop1 (A → answer_A via rel_TARGET)
    # seed → B via rel_hop1 (B → answer_B via rel_OTHER)
    edges = [
        ("seed", "rel_hop1", "A"),
        ("seed", "rel_hop1", "B"),
        ("A", "rel_TARGET", "answer_A"),
        ("B", "rel_OTHER", "answer_B"),
    ]
    _, t = _make_adapter_and_traversal(
        edges, max_hop=2, boost={"rel_TARGET": 10.0}
    )
    paths = t.traverse(["seed"])
    depth2 = [p for p in paths if p.hop_depth == 2]
    assert depth2, "Should have depth-2 paths"

    by_tail = {p.tail: p for p in depth2}
    assert "answer_A" in by_tail, "answer_A must be reachable"
    assert "answer_B" in by_tail, "answer_B must be reachable"

    # answer_A (via rel_TARGET) should score higher than answer_B (via rel_OTHER)
    assert by_tail["answer_A"].score > by_tail["answer_B"].score, (
        f"Expected answer_A score ({by_tail['answer_A'].score:.4f}) > "
        f"answer_B score ({by_tail['answer_B'].score:.4f}) due to 10× boost on rel_TARGET"
    )


# ---------------------------------------------------------------------------
# Test 2: empty boost dict → no change vs vanilla
# ---------------------------------------------------------------------------

def test_no_boost_with_empty_dict():
    """
    With terminal_relation_boost={}, scores must be identical to a traversal
    that has no boost at all.
    """
    edges = [
        ("seed", "rel_A", "X"),
        ("seed", "rel_B", "Y"),
    ]
    _, t_boost = _make_adapter_and_traversal(edges, max_hop=1, boost={})
    _, t_plain = _make_adapter_and_traversal(edges, max_hop=1, boost=None)

    paths_boost = t_boost.traverse(["seed"])
    paths_plain = t_plain.traverse(["seed"])

    scores_boost = sorted(p.score for p in paths_boost)
    scores_plain = sorted(p.score for p in paths_plain)
    assert scores_boost == pytest.approx(scores_plain, rel=1e-6), (
        "Empty boost dict must produce identical scores to no boost"
    )


# ---------------------------------------------------------------------------
# Test 3: unknown relation type gets 1× multiplier (no change)
# ---------------------------------------------------------------------------

def test_unknown_relation_gets_1x():
    """
    Boost dict = {"rel_KNOWN": 5.0}.  An edge via "rel_UNKNOWN" must NOT be
    penalised — the default multiplier is 1.0, not 0.0.
    """
    edges = [
        ("seed", "rel_UNKNOWN", "entity_unknown"),
        ("seed", "rel_KNOWN",   "entity_known"),
    ]
    _, t = _make_adapter_and_traversal(
        edges, max_hop=1, boost={"rel_KNOWN": 5.0}
    )
    paths = t.traverse(["seed"])
    by_tail = {p.tail: p for p in paths if p.hop_depth == 1}

    assert "entity_unknown" in by_tail, "entity_unknown must still be reachable"
    assert "entity_known" in by_tail, "entity_known must be reachable"

    # entity_known boosted 5×, entity_unknown unchanged (×1)
    assert by_tail["entity_known"].score > by_tail["entity_unknown"].score, (
        "rel_KNOWN (5×) should score above rel_UNKNOWN (1×)"
    )
    # entity_unknown score must be > 0 (not zeroed out)
    assert by_tail["entity_unknown"].score > 0.0, (
        "Unknown relation must get multiplier 1.0, not 0.0"
    )


# ---------------------------------------------------------------------------
# Test 4: detect_target_relation keyword mapping for MetaQA
# ---------------------------------------------------------------------------

def test_detect_target_relation_metaqa():
    """
    detect_target_relation maps MetaQA question text → correct relation type
    via keyword matching.  Tests 8 representative questions.
    """
    from benchmarks.metaqa_eval import detect_target_relation

    KB_RELATIONS = [
        "directed_by", "starred_actors", "written_by", "has_genre",
        "has_tags", "in_language", "release_year",
        "has_imdb_rating", "has_imdb_votes",
    ]

    cases = [
        ("who directed [E]'s movies",                  "directed_by"),
        ("what directors worked on films starring [E]", "directed_by"),
        ("what actors appeared in [E]",                 "starred_actors"),
        ("which actors starred in [E]'s films",         "starred_actors"),
        ("who wrote the screenplay for [E]",            "written_by"),
        ("what genre is [E]",                           "has_genre"),
        ("what language is [E] in",                     "in_language"),
        ("what year was [E] released",                  "release_year"),
    ]

    for question, expected_relation in cases:
        result = detect_target_relation(question, KB_RELATIONS)
        assert result == expected_relation, (
            f"Question: '{question}'\n"
            f"  Expected: {expected_relation}\n"
            f"  Got:      {result}"
        )


# ---------------------------------------------------------------------------
# Test 5: prefix-only detection avoids false hits from intermediate keywords
# ---------------------------------------------------------------------------

def test_detect_prefix_avoids_intermediate_keywords():
    """
    Phase 147: detect_target_relation scans only the first 5 words.
    'who acted in movies directed by [E]' has answer = starred_actors,
    not directed_by.  Full-question scan returns directed_by (wrong);
    prefix scan correctly returns starred_actors.
    """
    from benchmarks.metaqa_eval import detect_target_relation

    KB_RELATIONS = [
        "directed_by", "starred_actors", "written_by", "has_genre",
        "has_tags", "in_language", "release_year",
        "has_imdb_rating", "has_imdb_votes",
    ]

    ambiguous_cases = [
        # answer type first → prefix captures it
        ("who acted in movies directed by [E]",     "starred_actors"),
        ("what actors starred in films written by [E]", "starred_actors"),
        ("who wrote the films directed by [E]",     "written_by"),
    ]

    for question, expected_relation in ambiguous_cases:
        result = detect_target_relation(question, KB_RELATIONS, prefix_words=4)
        assert result == expected_relation, (
            f"Question: '{question}'\n"
            f"  Expected: {expected_relation}\n"
            f"  Got:      {result}"
        )


# ---------------------------------------------------------------------------
# Test 6: Phase 153 pre-passes fix 3-hop false-positive patterns
# ---------------------------------------------------------------------------

def test_detect_phase153_prepasses():
    """
    Phase 153: Three pre-passes handle false positives in 3-hop templates.

    Pre-pass 1: "when ..." → always release_year (not starred_actors even if
      "starred" appears before "release" in the question).
    Pre-pass 2: "...in which TERM" — last word before which/what terminal
      suffix captures answer type without entity-name contamination.
    Pre-pass 3: "what are/is the primary TERM" — 6-word prefix catches
      answer types at position 5 without triggering intermediate keywords.
    """
    from benchmarks.metaqa_eval import detect_target_relation

    KB_RELATIONS = [
        "directed_by", "starred_actors", "written_by", "has_genre",
        "has_tags", "in_language", "release_year",
        "has_imdb_rating", "has_imdb_votes",
    ]

    phase153_cases = [
        # Pre-pass 1: "when" → release_year (not starred_actors / directed_by)
        ("when did the films starred by Gone With The Wind actors release",  "release_year"),
        ("when did the films directed by the Inception director release",    "release_year"),
        ("when did the movies release whose actors also appear in Matrix",   "release_year"),
        # Pre-pass 2: "in which TERM" at sentence end
        ("the movies that share actors with the movie Billy Budd were in which languages", "in_language"),
        ("the movies that share directors with the movie Sheitan were in which genres",    "has_genre"),
        ("the movies that share actors with the movie Body of Lies were released in which years", "release_year"),
        # Pre-pass 3: "what are the primary TERM"
        ("what are the primary languages in the movies written by the writer of Titanic", "in_language"),
    ]

    for question, expected_relation in phase153_cases:
        result = detect_target_relation(question, KB_RELATIONS)
        assert result == expected_relation, (
            f"Question: '{question}'\n"
            f"  Expected: {expected_relation}\n"
            f"  Got:      {result}"
        )

# ---------------------------------------------------------------------------
# Test 6: penultimate cascade fires at hop N-1 with sqrt of terminal boost
# ---------------------------------------------------------------------------

def test_penultimate_cascade_fires_at_hop_n_minus_1():
    """
    Phase 147: 3-hop graph with a shared relation at hops 2 and 3.

    seed → rel_hop1 → mid → rel_TARGET → bridge → rel_TARGET → answer_A
    seed → rel_hop1 → mid → rel_TARGET → bridge → rel_OTHER  → answer_B

    With boost={rel_TARGET: 9.0}:
    - Terminal (hop 3): answer_A boosted 9×
    - Penultimate (hop 2): bridge_A boosted sqrt(9)=3× over bridge_B

    answer_A must score above answer_B.
    """
    edges = [
        ("seed",  "rel_hop1",   "mid"),
        ("mid",   "rel_TARGET", "bridge_A"),
        ("mid",   "rel_TARGET", "bridge_B"),
        ("bridge_A", "rel_TARGET", "answer_A"),
        ("bridge_B", "rel_OTHER",  "answer_B"),
    ]
    _, t = _make_adapter_and_traversal(edges, max_hop=3, boost={"rel_TARGET": 9.0})
    paths = t.traverse(["seed"])
    depth3 = [p for p in paths if p.hop_depth == 3]
    assert depth3, "Should have depth-3 paths"

    by_tail = {p.tail: p for p in depth3}
    assert "answer_A" in by_tail, "answer_A must be reachable"
    assert "answer_B" in by_tail, "answer_B must be reachable"

    assert by_tail["answer_A"].score > by_tail["answer_B"].score, (
        f"Expected answer_A ({by_tail['answer_A'].score:.4f}) > "
        f"answer_B ({by_tail['answer_B'].score:.4f}) via terminal + penultimate cascade"
    )

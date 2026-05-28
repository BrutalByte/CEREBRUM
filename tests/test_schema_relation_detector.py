"""
Unit tests for SchemaAwareRelationDetector (Phase 201).

Uses RandomEngine so tests are fast and GPU-independent.
Structural tests only (build, API, edge cases) — accuracy tests
require SentenceEngine and are covered by benchmarks/metaqa_eval.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.schema_relation_detector import SchemaAwareRelationDetector
from core.embedding_engine import RandomEngine


_RELATIONS = [
    "directed_by", "written_by", "starred_actors", "has_genre",
    "has_tags", "in_language", "release_year",
]

_QUESTIONS = [
    "what genres do the movies that [Actor A] starred in have",
    "the films that share directors with [Film X] were in which languages",
    "when did the movies starred by [Actor B] actors release",
    "who directed the films written by [Writer Z]",
]


@pytest.fixture(scope="module")
def srd_built():
    srd = SchemaAwareRelationDetector()
    eng = RandomEngine(dim=64)
    srd.build(_RELATIONS, eng)
    return srd


# ------------------------------------------------------------------
# Build / state
# ------------------------------------------------------------------

def test_not_built_before_build():
    srd = SchemaAwareRelationDetector()
    assert not srd.is_built


def test_built_after_build(srd_built):
    assert srd_built.is_built


def test_relation_names_preserved(srd_built):
    assert set(srd_built.relation_names()) == set(_RELATIONS)


def test_build_empty_relations():
    srd = SchemaAwareRelationDetector()
    srd.build([], RandomEngine(dim=64))
    assert not srd.is_built  # nothing to build


def test_build_none_engine():
    srd = SchemaAwareRelationDetector()
    srd.build(_RELATIONS, None)
    assert not srd.is_built


# ------------------------------------------------------------------
# detect_terminal
# ------------------------------------------------------------------

def test_detect_terminal_returns_relation(srd_built):
    """With random embeddings the result is non-deterministic but should be a valid relation."""
    r = srd_built.detect_terminal("what genres do the movies that [X] starred in have")
    assert r is None or r in _RELATIONS


def test_detect_terminal_not_built():
    srd = SchemaAwareRelationDetector()
    assert srd.detect_terminal("any question") is None


def test_detect_terminal_no_entity(srd_built):
    """Question without entity bracket still runs (uses full text)."""
    r = srd_built.detect_terminal("what genres do movies have")
    assert r is None or r in _RELATIONS


def test_detect_terminal_min_gap_zero_always_returns(srd_built):
    """min_gap=0 should always return a relation when built."""
    for q in _QUESTIONS:
        r = srd_built.detect_terminal(q, min_gap=0.0)
        assert r in _RELATIONS, f"Expected a relation for '{q[:40]}', got {r}"


def test_detect_terminal_high_min_gap_returns_none(srd_built):
    """min_gap=1.0 (impossible gap) should return None."""
    for q in _QUESTIONS:
        r = srd_built.detect_terminal(q, min_gap=1.0)
        assert r is None


# ------------------------------------------------------------------
# detect_initial
# ------------------------------------------------------------------

def test_detect_initial_returns_relation(srd_built):
    r = srd_built.detect_initial("what genres do the movies that [X] starred in have")
    assert r is None or r in _RELATIONS


def test_detect_initial_not_built():
    srd = SchemaAwareRelationDetector()
    assert srd.detect_initial("any question") is None


def test_detect_initial_no_entity_returns_none(srd_built):
    """Without an entity bracket there is no context to derive R1 from."""
    assert srd_built.detect_initial("what genres do movies have") is None


def test_detect_initial_exclude_removes_relation(srd_built):
    """Excluded relation must not appear in the result."""
    for q in _QUESTIONS:
        for excl in _RELATIONS[:3]:
            r = srd_built.detect_initial(q, exclude=excl, min_gap=0.0)
            assert r != excl or r is None, f"Excluded '{excl}' but got it for '{q[:40]}'"


def test_detect_initial_min_gap_zero_always_returns(srd_built):
    """min_gap=0 returns a relation (not None) for questions with entity."""
    for q in _QUESTIONS:
        r = srd_built.detect_initial(q, min_gap=0.0)
        assert r in _RELATIONS, f"Expected relation for '{q[:40]}', got {r}"


# ------------------------------------------------------------------
# detect_path
# ------------------------------------------------------------------

def test_detect_path_returns_tuple(srd_built):
    r1, r2 = srd_built.detect_path("what genres do the movies that [X] starred in have")
    assert r1 is None or r1 in _RELATIONS
    assert r2 is None or r2 in _RELATIONS


def test_detect_path_not_built():
    srd = SchemaAwareRelationDetector()
    r1, r2 = srd.detect_path("any question")
    assert r1 is None and r2 is None


def test_detect_path_r1_r2_independent(srd_built):
    """R1 and R2 can be the same relation (R1=R3 case), or different. Neither is guaranteed."""
    for q in _QUESTIONS:
        r1, r2 = srd_built.detect_path(q, min_gap=0.0)
        assert r1 in _RELATIONS
        assert r2 in _RELATIONS


# ------------------------------------------------------------------
# content_words helper
# ------------------------------------------------------------------

def test_content_words_filters_stopwords(srd_built):
    tokens = ["what", "the", "directors", "of", "films", "with", "actors"]
    result = srd_built._content_words(tokens, min_len=4)
    assert "what" not in result  # stopword
    assert "the" not in result   # stopword
    assert "of" not in result    # stopword
    assert "directors" in result
    assert "films" in result
    assert "actors" in result


def test_content_words_min_len(srd_built):
    tokens = ["by", "the", "star", "actors"]
    result = srd_built._content_words(tokens, min_len=5)
    assert "actors" in result
    assert "star" not in result  # len=4 < 5


# ------------------------------------------------------------------
# _to_phrase
# ------------------------------------------------------------------

@pytest.mark.parametrize("rel,expected", [
    ("directed_by",    "directed by"),
    ("written_by",     "written by"),
    ("starred_actors", "starred actors"),
    ("has_genre",      "has genre"),
    ("release_year",   "release year"),
])
def test_to_phrase(rel, expected):
    assert SchemaAwareRelationDetector._to_phrase(rel) == expected


# ------------------------------------------------------------------
# Robustness
# ------------------------------------------------------------------

def test_empty_question(srd_built):
    assert srd_built.detect_terminal("") is None
    assert srd_built.detect_initial("") is None


def test_entity_only(srd_built):
    r = srd_built.detect_terminal("[Entity Only]")
    assert r is None or r in _RELATIONS


def test_detect_initial_entity_only(srd_built):
    """Entity only — no prefix/suffix context, should return None or a relation."""
    r = srd_built.detect_initial("[Entity Only]", min_gap=0.0)
    # No prefix words: content word scan is empty. Suffix is also empty.
    # phrase_windows with empty prefix: skipped. Result should be None.
    assert r is None or r in _RELATIONS

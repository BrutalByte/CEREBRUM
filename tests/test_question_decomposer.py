"""Tests for core.question_decomposer (Phase 232)."""
import pytest
from core.question_decomposer import QuestionDecomposer, DecomposedQuestion


@pytest.fixture
def qd():
    return QuestionDecomposer()


# ---------------------------------------------------------------------------
# WH-word and answer_type
# ---------------------------------------------------------------------------

def test_who_question(qd):
    r = qd.decompose("who plays ken barlow in coronation street")
    assert r.wh_word == "who"
    assert r.answer_type == "person"


def test_where_question(qd):
    r = qd.decompose("where was barack obama born")
    assert r.wh_word == "where"
    assert r.answer_type == "place"


def test_when_question(qd):
    r = qd.decompose("when did the beatles form")
    assert r.wh_word == "when"
    assert r.answer_type == "time"


def test_what_question(qd):
    r = qd.decompose("what language do jamaicans speak")
    assert r.wh_word == "what"
    assert r.answer_type == "thing"


def test_how_many_question(qd):
    r = qd.decompose("how many oscars did titanic win")
    assert r.wh_word == "how many"
    assert r.answer_type == "quantity"


def test_how_long_question(qd):
    r = qd.decompose("how long is the great wall of china")
    assert r.wh_word == "how long"
    assert r.answer_type == "duration"


# ---------------------------------------------------------------------------
# Relation keywords
# ---------------------------------------------------------------------------

def test_stopwords_removed(qd):
    r = qd.decompose("who plays ken barlow in coronation street")
    assert "in" not in r.relation_keywords
    assert "the" not in r.relation_keywords
    assert "who" not in r.relation_keywords


def test_verb_lemmatized(qd):
    r = qd.decompose("who directed the dark knight")
    assert "direct" in r.relation_keywords


def test_plays_lemmatized(qd):
    r = qd.decompose("who plays ken barlow in coronation street")
    assert "play" in r.relation_keywords


def test_directed_lemmatized(qd):
    r = qd.decompose("who directed jurassic park")
    assert "direct" in r.relation_keywords
    assert "directed" not in r.relation_keywords


def test_wrote_lemmatized(qd):
    r = qd.decompose("who wrote harry potter")
    assert "write" in r.relation_keywords


# ---------------------------------------------------------------------------
# Temporal constraints
# ---------------------------------------------------------------------------

def test_temporal_before(qd):
    r = qd.decompose("what did james polk do before he was president")
    assert r.has_temporal_constraint is True


def test_temporal_after(qd):
    r = qd.decompose("what happened after the french revolution")
    assert r.has_temporal_constraint is True


def test_no_temporal_constraint(qd):
    r = qd.decompose("who plays romeo in romeo and juliet")
    assert r.has_temporal_constraint is False


# ---------------------------------------------------------------------------
# Comparative / superlative
# ---------------------------------------------------------------------------

def test_comparative_first(qd):
    r = qd.decompose("who was the first person on the moon")
    assert r.is_comparative is True


def test_comparative_oldest(qd):
    r = qd.decompose("what is the oldest university in england")
    assert r.is_comparative is True


def test_not_comparative(qd):
    r = qd.decompose("who wrote moby dick")
    assert r.is_comparative is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_question(qd):
    r = qd.decompose("")
    assert isinstance(r, DecomposedQuestion)
    assert r.answer_type == "thing"


def test_no_wh_word(qd):
    r = qd.decompose("name the capital of france")
    assert r.wh_word == ""
    assert r.answer_type == "thing"

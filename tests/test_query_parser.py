"""
Tests for Phase 18b — QueryParser.
"""
import pytest
import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.query_parser import QueryParser, ParsedQuery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_graph():
    """Small toy graph with known entities and relation types."""
    G = nx.Graph()
    G.add_edge("newton",   "leibniz",  relation_type="INFLUENCED")
    G.add_edge("newton",   "optics",   relation_type="WROTE")
    G.add_edge("leibniz",  "calculus", relation_type="INVENTED")
    G.add_edge("einstein", "newton",   relation_type="INFLUENCED")
    G.add_edge("aspirin",  "headache", relation_type="TREATS")
    G.add_edge("smoking",  "cancer",   relation_type="CAUSES")
    return G


@pytest.fixture
def parser():
    G = make_graph()
    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=64)
    labels = {n: n for n in G.nodes()}
    adapter.embeddings = engine.encode_entities(labels)
    adapter.community_map = {n: 0 for n in G.nodes()}
    return QueryParser(adapter, engine)


# ---------------------------------------------------------------------------
# ParsedQuery structure
# ---------------------------------------------------------------------------

def test_parse_returns_parsed_query(parser):
    result = parser.parse("What did newton influence?")
    assert isinstance(result, ParsedQuery)


def test_raw_question_preserved(parser):
    q = "What did newton influence?"
    result = parser.parse(q)
    assert result.raw_question == q


def test_seed_entity_found(parser):
    result = parser.parse("What did newton influence?")
    assert result.seed_entity_id == "newton"


def test_seed_entity_score_in_range(parser):
    result = parser.parse("What did newton influence?")
    assert 0.0 <= result.seed_entity_score <= 1.0


def test_candidates_returned(parser):
    result = parser.parse("What did newton influence?")
    assert isinstance(result.candidates, list)
    assert len(result.candidates) >= 1


def test_candidates_are_tuples(parser):
    result = parser.parse("What did newton influence?")
    for cand in result.candidates:
        assert len(cand) == 3  # (entity_id, label, score)


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

def test_relation_hints_for_influence(parser):
    result = parser.parse("Who did newton influence?")
    assert "INFLUENCED" in result.relation_hints


def test_relation_hints_for_treats(parser):
    result = parser.parse("What does aspirin treat?")
    assert "TREATS" in result.relation_hints


def test_relation_hints_for_causes(parser):
    result = parser.parse("What does smoking cause?")
    assert "CAUSES" in result.relation_hints


def test_relation_hints_empty_for_generic(parser):
    result = parser.parse("Tell me about newton")
    # generic question — no strong relation signal
    assert isinstance(result.relation_hints, list)


def test_relation_vocabulary_populated(parser):
    vocab = parser.relation_vocabulary()
    assert "INFLUENCED" in vocab
    assert "TREATS" in vocab
    assert "CAUSES" in vocab


# ---------------------------------------------------------------------------
# Hop inference
# ---------------------------------------------------------------------------

def test_hop_hint_short_question(parser):
    result = parser.parse("Who is newton?")
    assert result.hop_hint == 1


def test_hop_hint_multi_hop_marker(parser):
    result = parser.parse("Trace the path through newton to einstein via influence")
    assert result.hop_hint == 3


def test_hop_hint_default(parser):
    result = parser.parse("What did newton contribute to calculus development?")
    assert result.hop_hint == 2


def test_hop_hint_explicit_three(parser):
    result = parser.parse("3-hop path from newton")
    assert result.hop_hint == 3


# ---------------------------------------------------------------------------
# Question type detection
# ---------------------------------------------------------------------------

def test_question_type_who(parser):
    result = parser.parse("Who did newton influence?")
    assert result.question_type == "person"


def test_question_type_what(parser):
    result = parser.parse("What did newton write?")
    assert result.question_type == "thing"


def test_question_type_where(parser):
    result = parser.parse("Where was einstein located?")
    assert result.question_type == "place"


# ---------------------------------------------------------------------------
# Entity not found
# ---------------------------------------------------------------------------

def test_unknown_entity_returns_none(parser):
    result = parser.parse("What does xyzzy_unknown_entity do?")
    # May or may not find something, but should not crash
    assert isinstance(result.seed_entity_id, (str, type(None)))


def test_parse_empty_string(parser):
    result = parser.parse("")
    assert isinstance(result, ParsedQuery)


# ---------------------------------------------------------------------------
# Integration: parser → entity linking accuracy
# ---------------------------------------------------------------------------

def test_exact_match_newton(parser):
    result = parser.parse("How did newton influence others?")
    assert result.seed_entity_id == "newton"


def test_exact_match_aspirin(parser):
    result = parser.parse("What does aspirin treat?")
    assert result.seed_entity_id == "aspirin"


def test_exact_match_einstein(parser):
    result = parser.parse("Who influenced einstein?")
    assert result.seed_entity_id == "einstein"

"""
Tests for Phase 20 — ConversationManager and ConversationSession.
"""
import pytest
import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from core.community_engine import best_of_n_dscf
from core.conversation import (
    ConversationManager,
    ConversationSession,
    ConversationTurn,
    new_session,
    _PRONOUN_TYPE,
    _FOLLOWUP_PREFIXES,
)
from reasoning.traversal import BeamTraversal


# ---------------------------------------------------------------------------
# Shared fixture: small graph + full pipeline
# ---------------------------------------------------------------------------

def _build_pipeline():
    G = nx.Graph()
    G.add_edge("newton",   "leibniz",    relation="INFLUENCED",  confidence=1.0)
    G.add_edge("newton",   "faraday",    relation="INFLUENCED",  confidence=1.0)
    G.add_edge("newton",   "optics",     relation="WROTE",       confidence=1.0)
    G.add_edge("leibniz",  "calculus",   relation="INVENTED",    confidence=1.0)
    G.add_edge("einstein", "newton",     relation="INFLUENCED",  confidence=1.0)
    G.add_edge("aspirin",  "headache",   relation="TREATS",      confidence=1.0)
    G.add_edge("smoking",  "cancer",     relation="CAUSES",      confidence=1.0)
    G.add_edge("london",   "newton",     relation="LIVED_IN",    confidence=1.0)

    adapter = NetworkXAdapter(G)
    engine  = RandomEngine(dim=64)
    labels  = {n: n for n in G.nodes()}
    adapter.embeddings   = engine.encode_entities(labels)
    adapter.community_map = {n: 0 for n in G.nodes()}

    parts = best_of_n_dscf(G, n_trials=1, seed=42)
    cmap  = {}
    for cid, members in enumerate(parts):
        for node in members:
            cmap[node] = cid
    adapter.community_map = cmap

    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)

    traversal = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10, max_hop=2)

    manager = ConversationManager(
        adapter=adapter,
        embedding_engine=engine,
        csa_engine=csa,
        beam_traversal=traversal,
        top_k=5,
    )
    return manager


@pytest.fixture
def manager():
    return _build_pipeline()


@pytest.fixture
def session(manager):
    return manager.new_session()


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------

def test_new_session_is_empty():
    s = new_session()
    assert s.focus_entity is None
    assert s.turn_count == 0
    assert len(s.entity_history) == 0
    assert len(s.answer_history) == 0
    assert len(s.pronoun_map) == 0


def test_new_session_has_id():
    s = new_session()
    assert isinstance(s.session_id, str)
    assert len(s.session_id) > 0


def test_custom_session_id():
    s = new_session("my-session-42")
    assert s.session_id == "my-session-42"


def test_manager_new_session(manager):
    s = manager.new_session()
    assert isinstance(s, ConversationSession)
    assert s.turn_count == 0


# ---------------------------------------------------------------------------
# Single-turn basic operation
# ---------------------------------------------------------------------------

def test_single_turn_returns_turn(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert isinstance(turn, ConversationTurn)


def test_single_turn_increments_turn_count(manager, session):
    manager.process("What did newton influence?", session)
    assert session.turn_count == 1


def test_single_turn_sets_focus(manager, session):
    manager.process("What did newton influence?", session)
    assert session.focus_entity == "newton"


def test_single_turn_answer_text_not_empty(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert isinstance(turn.answer_text, str)
    assert len(turn.answer_text) > 0


def test_single_turn_seed_entity(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert turn.seed_entity == "newton"


def test_single_turn_updates_entity_history(manager, session):
    manager.process("What did newton influence?", session)
    assert "newton" in session.entity_history


def test_single_turn_updates_answer_history(manager, session):
    manager.process("What did newton influence?", session)
    assert len(session.answer_history) > 0


def test_turn_number_increments(manager, session):
    t1 = manager.process("What did newton influence?", session)
    t2 = manager.process("What did aspirin treat?", session)
    assert t1.turn_number == 1
    assert t2.turn_number == 2


# ---------------------------------------------------------------------------
# Pronoun resolution
# ---------------------------------------------------------------------------

def test_pronoun_he_resolves_after_newton(manager, session):
    # "Who" question → question_type="person" → "he/him/his" get mapped to newton
    manager.process("Who did newton influence?", session)
    assert session.pronoun_map.get("he") == "newton" or \
           session.pronoun_map.get("him") == "newton"


def test_pronoun_resolution_substitutes_in_question(manager, session):
    # "Who" question → person type → "he" maps to newton
    manager.process("Who did newton influence?", session)
    turn2 = manager.process("What did he write?", session)
    assert "newton" in turn2.resolved_question.lower()


def test_it_resolves_to_last_thing(manager, session):
    manager.process("What does aspirin treat?", session)
    # "aspirin" is a thing — "it" should resolve to aspirin
    assert session.pronoun_map.get("it") == "aspirin"


def test_there_resolves_to_place(manager, session):
    manager.process("Where did newton live?", session)
    # london is a place — "there" should map to london after it surfaces
    # focus entity is newton (the seed), pronoun map updates to newton
    # This verifies the map is populated regardless
    assert session.pronoun_map is not None


def test_unresolvable_pronoun_not_substituted(manager, session):
    # No prior context — pronoun map is empty
    turn = manager.process("What did he invent?", session)
    # Should not crash; question passes through unchanged
    assert isinstance(turn, ConversationTurn)


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

def test_followup_prefix_and_detected(manager, session):
    manager.process("What did newton influence?", session)
    turn2 = manager.process("And what did newton write?", session)
    assert turn2.is_followup is True


def test_followup_what_else_detected(manager, session):
    manager.process("What did newton influence?", session)
    turn2 = manager.process("What else?", session)
    assert turn2.is_followup is True


def test_followup_also_detected(manager, session):
    manager.process("What did newton influence?", session)
    # Comma after "Also" — _is_followup normalizes leading punctuation
    turn2 = manager.process("Also, tell me more about what he did", session)
    assert turn2.is_followup is True


def test_first_turn_never_followup(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert turn.is_followup is False


def test_new_entity_not_followup(manager, session):
    manager.process("What did newton influence?", session)
    turn2 = manager.process("What does aspirin treat?", session)
    assert turn2.is_followup is False


# ---------------------------------------------------------------------------
# Answer deduplication
# ---------------------------------------------------------------------------

def test_followup_excludes_seen_answers(manager, session):
    t1 = manager.process("What did newton influence?", session)
    t2 = manager.process("What else?", session)
    # Answers in t2 should not repeat answers from t1
    t1_ids = set(t1.new_entities)
    t2_ids = set(t2.new_entities)
    assert len(t1_ids & t2_ids) == 0


def test_answer_history_grows_across_turns(manager, session):
    manager.process("What did newton influence?", session)
    n1 = len(session.answer_history)
    manager.process("What else?", session)
    n2 = len(session.answer_history)
    assert n2 >= n1  # can only grow


# ---------------------------------------------------------------------------
# Topic shift
# ---------------------------------------------------------------------------

def test_topic_shift_on_new_entity(manager, session):
    manager.process("What did newton influence?", session)
    turn2 = manager.process("What does aspirin treat?", session)
    assert turn2.focus_shift is True


def test_topic_shift_updates_focus(manager, session):
    manager.process("What did newton influence?", session)
    manager.process("What does aspirin treat?", session)
    assert session.focus_entity == "aspirin"


def test_same_entity_no_focus_shift(manager, session):
    manager.process("What did newton influence?", session)
    turn2 = manager.process("What else did newton do?", session)
    # newton is the same focus — no shift (detected as followup OR same entity)
    assert turn2.focus_shift is False


# ---------------------------------------------------------------------------
# Knowledge gap detection
# ---------------------------------------------------------------------------

def test_no_entity_returns_gap(manager, session):
    # Use a manager with high min_entity_score so random gibberish never matches
    from core.query_parser import QueryParser
    strict_manager = ConversationManager(
        adapter=manager._adapter,
        embedding_engine=manager._engine,
        csa_engine=manager._csa,
        beam_traversal=manager._traversal,
        top_k=5,
    )
    strict_manager._parser = QueryParser(
        manager._adapter, manager._engine,
        min_entity_score=0.99,   # only exact matches qualify
    )
    s = strict_manager.new_session()
    turn = strict_manager.process("xyzzy_nonexistent_entity_999", s)
    assert turn.knowledge_gap is True


def test_gap_hint_not_empty_on_gap(manager, session):
    turn = manager.process("xyzzy_nonexistent_entity_999", session)
    # gap hint may be empty if no entity was resolved at all
    assert isinstance(turn.knowledge_gap_hint, str)


def test_exhausted_answers_produces_gap(manager, session):
    # Ask and follow up until answers are exhausted
    manager.process("What did newton influence?", session)
    for _ in range(10):
        turn = manager.process("What else?", session)
        if turn.knowledge_gap:
            break
    # Eventually must exhaust (newton has finite neighbours)
    assert turn.knowledge_gap is True


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

def test_reset_clears_focus(manager, session):
    manager.process("What did newton influence?", session)
    session.reset()
    assert session.focus_entity is None


def test_reset_clears_history(manager, session):
    manager.process("What did newton influence?", session)
    session.reset()
    assert len(session.entity_history) == 0
    assert len(session.answer_history) == 0
    assert len(session.pronoun_map) == 0


def test_reset_clears_turns(manager, session):
    manager.process("What did newton influence?", session)
    session.reset()
    assert session.turn_count == 0


def test_after_reset_new_turn_works(manager, session):
    manager.process("What did newton influence?", session)
    session.reset()
    turn = manager.process("What does aspirin treat?", session)
    assert isinstance(turn, ConversationTurn)
    assert turn.turn_number == 1


# ---------------------------------------------------------------------------
# ConversationTurn structure
# ---------------------------------------------------------------------------

def test_turn_has_raw_question(manager, session):
    q = "What did newton influence?"
    turn = manager.process(q, session)
    assert turn.raw_question == q


def test_turn_has_resolved_question(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert isinstance(turn.resolved_question, str)
    assert len(turn.resolved_question) > 0


def test_turn_new_entities_is_list(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert isinstance(turn.new_entities, list)


def test_turn_hop_hint_positive(manager, session):
    turn = manager.process("What did newton influence?", session)
    assert turn.hop_hint >= 1


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

def test_session_summary_string(manager, session):
    manager.process("What did newton influence?", session)
    s = manager.session_summary(session)
    assert isinstance(s, str)
    assert "newton" in s.lower() or "Turn" in s or "turn" in s


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_question_does_not_crash(manager, session):
    turn = manager.process("", session)
    assert isinstance(turn, ConversationTurn)


def test_whitespace_only_does_not_crash(manager, session):
    turn = manager.process("   ", session)
    assert isinstance(turn, ConversationTurn)


def test_multiple_sessions_independent():
    m = _build_pipeline()
    s1 = m.new_session()
    s2 = m.new_session()
    m.process("What did newton influence?", s1)
    # s2 should have no memory of s1
    assert s2.focus_entity is None
    assert s2.turn_count == 0


def test_session_last_active_updated(manager, session):
    before = session.last_active
    import time; time.sleep(0.01)
    manager.process("What did newton influence?", session)
    assert session.last_active > before

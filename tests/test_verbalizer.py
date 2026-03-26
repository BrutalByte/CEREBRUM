"""
Tests for Phase 18a — PathVerbalizer.
"""
import pytest
from unittest.mock import MagicMock

from core.verbalizer import PathVerbalizer, VerbalizationResult
from reasoning.traversal import TraversalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_path(nodes, edge_confidences=None):
    return TraversalPath(
        nodes=nodes,
        seen_entities=set(n for i, n in enumerate(nodes) if i % 2 == 0),
        edge_confidences=edge_confidences or [],
    )


def make_adapter(labels=None, communities=None):
    adapter = MagicMock()
    labels = labels or {}
    communities = communities or {}

    def get_entity(eid):
        e = MagicMock()
        e.label = labels.get(eid, eid)
        return e

    adapter.get_entity.side_effect = get_entity
    adapter.get_community.side_effect = lambda eid: communities.get(eid, -1)
    return adapter


# ---------------------------------------------------------------------------
# VerbalizationResult structure
# ---------------------------------------------------------------------------

def test_verbalize_single_node():
    verb = PathVerbalizer()
    path = make_path(["newton"])
    result = verb.verbalize_path(path)
    assert result.answer_entity == "newton"
    assert result.hop_count == 0


def test_verbalize_one_hop():
    verb = PathVerbalizer()
    path = make_path(["newton", "INFLUENCED", "leibniz"])
    result = verb.verbalize_path(path)
    assert "newton" in result.text.lower()
    assert "leibniz" in result.text.lower()
    assert result.hop_count == 1


def test_verbalize_two_hop():
    verb = PathVerbalizer()
    path = make_path(["newton", "INFLUENCED", "leibniz", "CORRESPONDED_WITH", "huygens"])
    result = verb.verbalize_path(path)
    assert "newton" in result.text.lower()
    assert "huygens" in result.text.lower()
    assert result.hop_count == 2
    assert result.answer_entity == "huygens"


def test_citations_correct_format():
    verb = PathVerbalizer()
    path = make_path(["newton", "INFLUENCED", "leibniz"])
    result = verb.verbalize_path(path)
    assert len(result.citations) == 1
    assert result.citations[0] == "newton-[INFLUENCED]->leibniz"


def test_citations_two_hop():
    verb = PathVerbalizer()
    path = make_path(["a", "CAUSES", "b", "TREATS", "c"])
    result = verb.verbalize_path(path)
    assert len(result.citations) == 2
    assert "a-[CAUSES]->b" in result.citations
    assert "b-[TREATS]->c" in result.citations


def test_answer_entity_is_terminal():
    verb = PathVerbalizer()
    path = make_path(["a", "INFLUENCED", "b", "INFLUENCED", "c"])
    result = verb.verbalize_path(path)
    assert result.answer_entity == "c"


# ---------------------------------------------------------------------------
# Template coverage
# ---------------------------------------------------------------------------

def test_known_relation_active():
    verb = PathVerbalizer()
    path = make_path(["aspirin", "TREATS", "headache"])
    result = verb.verbalize_path(path)
    assert "treats" in result.text.lower()


def test_known_relation_causes():
    verb = PathVerbalizer()
    path = make_path(["smoking", "CAUSES", "cancer"])
    result = verb.verbalize_path(path)
    assert "causes" in result.text.lower()


def test_unknown_relation_fallback():
    verb = PathVerbalizer()
    path = make_path(["a", "SOME_NOVEL_RELATION", "b"])
    result = verb.verbalize_path(path)
    # Should produce something readable, not crash
    assert "a" in result.text.lower()
    assert "b" in result.text.lower()


def test_rem_synthesized_edge():
    verb = PathVerbalizer()
    path = make_path(["gene_a", "REM_SYNTHESIZED", "gene_b"])
    result = verb.verbalize_path(path)
    assert "similar" in result.text.lower()


def test_extra_templates_override():
    verb = PathVerbalizer(extra_templates={
        "ZAPS": ("{src} zapped {dst}", "{dst} was zapped by {src}", "which zapped"),
    })
    path = make_path(["robot", "ZAPS", "target"])
    result = verb.verbalize_path(path)
    assert "zapped" in result.text


# ---------------------------------------------------------------------------
# Adapter integration
# ---------------------------------------------------------------------------

def test_uses_adapter_labels():
    adapter = make_adapter(labels={"n1": "Isaac Newton", "n2": "Gottfried Leibniz"})
    verb = PathVerbalizer()
    path = make_path(["n1", "INFLUENCED", "n2"])
    result = verb.verbalize_path(path, adapter)
    assert "Isaac Newton" in result.text
    assert "Gottfried Leibniz" in result.text


def test_falls_back_to_id_without_adapter():
    verb = PathVerbalizer()
    path = make_path(["entity_001", "INFLUENCED", "entity_002"])
    result = verb.verbalize_path(path, adapter=None)
    assert "entity_001" in result.text


def test_community_note_populated():
    adapter = make_adapter(communities={"leibniz": 7})
    verb = PathVerbalizer()
    path = make_path(["newton", "INFLUENCED", "leibniz"])
    result = verb.verbalize_path(path, adapter)
    assert "7" in result.community_note


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_confidence_qualifier_high():
    verb = PathVerbalizer()
    path = make_path(["a", "INFLUENCED", "b"], edge_confidences=[0.95])
    result = verb.verbalize_path(path)
    assert result.confidence_qualifier == "high confidence"


def test_confidence_qualifier_low():
    verb = PathVerbalizer()
    path = make_path(["a", "INFLUENCED", "b"], edge_confidences=[0.3])
    result = verb.verbalize_path(path)
    assert result.confidence_qualifier == "tentative"


def test_path_confidence_weakest_link():
    verb = PathVerbalizer()
    path = make_path(["a", "CAUSES", "b", "TREATS", "c"],
                     edge_confidences=[0.95, 0.45])
    result = verb.verbalize_path(path)
    assert abs(result.path_confidence - 0.45) < 1e-3


# ---------------------------------------------------------------------------
# verbalize_answer and verbalize_answers
# ---------------------------------------------------------------------------

def test_verbalize_answer_uses_best_path():
    from reasoning.answer_extractor import Answer

    path = make_path(["newton", "INFLUENCED", "leibniz"], edge_confidences=[0.9])
    answer = Answer(entity_id="leibniz", score=0.8, best_path=path)
    verb = PathVerbalizer()
    result = verb.verbalize_answer(answer)
    assert result.answer_entity == "leibniz"
    assert "newton" in result.text.lower()


def test_verbalize_answers_returns_string():
    from reasoning.answer_extractor import Answer

    path = make_path(["a", "CAUSED", "b"])
    answer = Answer(entity_id="b", score=0.7, best_path=path)
    verb = PathVerbalizer()
    text = verb.verbalize_answers([answer], question="What did a cause?")
    assert isinstance(text, str)
    assert "Q:" in text
    assert "a" in text.lower()


def test_verbalize_answers_empty():
    verb = PathVerbalizer()
    text = verb.verbalize_answers([])
    assert "No answers" in text


def test_verbalize_answers_top_k():
    from reasoning.answer_extractor import Answer

    answers = [
        Answer(entity_id=f"e{i}", score=1.0 - i * 0.1,
               best_path=make_path([f"src", f"REL", f"e{i}"]))
        for i in range(5)
    ]
    verb = PathVerbalizer()
    text = verb.verbalize_answers(answers, top_k=2)
    assert "1." in text
    assert "2." in text
    assert "3." not in text

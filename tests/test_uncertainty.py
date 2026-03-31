"""
Tests for Phase 17.2 — Uncertainty Propagation.

Covers:
  - TraversalPath.path_confidence (weakest-link)
  - path_confidence() function in path_scorer
  - Answer.path_confidence populated correctly by extract()
"""
import pytest
import networkx as nx

from reasoning.traversal import TraversalPath
from reasoning.path_scorer import path_confidence
from reasoning.answer_extractor import Answer, extract


# ---------------------------------------------------------------------------
# TraversalPath.path_confidence
# ---------------------------------------------------------------------------

def test_path_confidence_no_edges_returns_one():
    p = TraversalPath(nodes=["A"], seen_entities={"A"})
    assert p.path_confidence == 1.0


def test_path_confidence_all_certain():
    p = TraversalPath(
        nodes=["A", "R", "B"],
        seen_entities={"A", "B"},
        edge_confidences=[1.0, 1.0],
    )
    assert p.path_confidence == 1.0


def test_path_confidence_weakest_link():
    p = TraversalPath(
        nodes=["A", "R1", "B", "R2", "C"],
        seen_entities={"A", "B", "C"},
        edge_confidences=[0.9, 0.3],
    )
    assert p.path_confidence == pytest.approx(0.3)


def test_path_confidence_single_edge():
    p = TraversalPath(
        nodes=["A", "R", "B"],
        seen_entities={"A", "B"},
        edge_confidences=[0.75],
    )
    assert p.path_confidence == pytest.approx(0.75)


def test_path_confidence_zero_edge():
    p = TraversalPath(
        nodes=["A", "R", "B"],
        seen_entities={"A", "B"},
        edge_confidences=[0.0],
    )
    assert p.path_confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# path_confidence() standalone function in path_scorer
# ---------------------------------------------------------------------------

def test_path_scorer_function_no_confidences():
    p = TraversalPath(nodes=["A"])
    assert path_confidence(p) == 1.0


def test_path_scorer_function_weakest_link():
    p = TraversalPath(
        nodes=["A", "R", "B", "S", "C"],
        seen_entities={"A", "B", "C"},
        edge_confidences=[1.0, 0.5, 0.8],
    )
    assert path_confidence(p) == pytest.approx(0.5)


def test_path_scorer_function_on_object_without_attribute():
    """path_confidence() is robust to objects missing edge_confidences."""
    class _Bare:
        pass
    assert path_confidence(_Bare()) == 1.0


# ---------------------------------------------------------------------------
# Answer.path_confidence populated by extract()
# ---------------------------------------------------------------------------

def _make_adapter_with_confidence():
    """
    Graph: A -[high conf]-> B -[low conf]-> C
    A -[certain]-> D (single hop, confidence 1.0)
    """
    from adapters.networkx_adapter import NetworkXAdapter
    G = nx.DiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n, label=n)
    G.add_edge("A", "B", relation="HIGH", weight=1.0, confidence=0.9)
    G.add_edge("B", "C", relation="LOW",  weight=1.0, confidence=0.3)
    G.add_edge("A", "D", relation="CERT", weight=1.0, confidence=1.0)
    adapter = NetworkXAdapter(G)
    adapter.build_communities()
    return adapter


def test_answer_path_confidence_populated():
    from core.attention_engine import CSAEngine
    from reasoning.traversal import BeamTraversal

    adapter = _make_adapter_with_confidence()
    csa = CSAEngine(adapter)
    bt = BeamTraversal(adapter, csa, beam_width=20, max_hop=2)
    paths = bt.traverse(["A"])
    answers = extract(paths, top_k=10)

    assert len(answers) > 0
    for ans in answers:
        assert 0.0 <= ans.path_confidence <= 1.0


def test_answer_d_has_full_confidence():
    """Single hop to D through confidence=1.0 edge → path_confidence=1.0."""
    from core.attention_engine import CSAEngine
    from reasoning.traversal import BeamTraversal

    adapter = _make_adapter_with_confidence()
    csa = CSAEngine(adapter)
    bt = BeamTraversal(adapter, csa, beam_width=20, max_hop=1)
    paths = bt.traverse(["A"])
    answers = extract(paths, top_k=10)

    d_answers = [a for a in answers if a.entity_id == "D"]
    assert d_answers, "D should be in answers"
    assert d_answers[0].path_confidence == pytest.approx(1.0)


def test_answer_c_has_low_confidence():
    """Two-hop path to C goes through confidence=0.3 edge → weakest-link = 0.3."""
    from core.attention_engine import CSAEngine
    from reasoning.traversal import BeamTraversal

    adapter = _make_adapter_with_confidence()
    csa = CSAEngine(adapter)
    bt = BeamTraversal(adapter, csa, beam_width=20, max_hop=2)
    paths = bt.traverse(["A"])
    answers = extract(paths, top_k=10)

    c_answers = [a for a in answers if a.entity_id == "C"]
    assert c_answers, "C should be in answers at 2-hop"
    assert c_answers[0].path_confidence == pytest.approx(0.3)


def test_answer_repr_includes_no_exception():
    """Answer.__repr__ works with path_confidence set."""
    ans = Answer(entity_id="X", score=0.9, best_path=None, path_confidence=0.75)
    r = repr(ans)
    assert "X" in r

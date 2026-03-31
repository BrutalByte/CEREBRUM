"""
Tests for RelationPathPrior and GraphRelationPrior — Phase 27B.

Tests cover:
  - RelationPathPrior: update, freeze, score, score_with_prefix, top_sequences
  - GraphRelationPrior: fit, score, unknown relations
  - score_path() with relation_prior integration
  - extract() with relation_prior integration
"""
import math
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal stub for TraversalPath
# ---------------------------------------------------------------------------

def _make_path(nodes: List[str], attention_weights=None, community_sequence=None,
               embedding=None, edge_confidences=None, score_variance=0.0):
    """Create a minimal TraversalPath-like object for testing."""
    p = SimpleNamespace()
    p.nodes             = nodes
    p.attention_weights = attention_weights or [0.8] * max(len(nodes) // 2, 1)
    p.community_sequence = community_sequence or [0] * ((len(nodes) + 1) // 2)
    p.embedding         = embedding
    p.edge_confidences  = edge_confidences or []
    p.score_variance    = score_variance
    p.hop_depth         = (len(nodes) - 1) // 2
    p.tail              = nodes[-1] if nodes else ""
    return p


# ---------------------------------------------------------------------------
# RelationPathPrior
# ---------------------------------------------------------------------------

class TestRelationPathPrior:
    def _build_prior(self):
        from reasoning.relation_path_prior import RelationPathPrior
        return RelationPathPrior(smoothing=1.0, max_len=3, min_count=1)

    def test_init_defaults(self):
        from reasoning.relation_path_prior import RelationPathPrior
        prior = RelationPathPrior()
        assert prior.smoothing > 0
        assert prior.max_len >= 1

    def test_update_correct_path(self):
        """Paths ending at correct entities should increment success counts."""
        prior = self._build_prior()
        # 1-hop path: A -rel1-> B
        path = _make_path(["A", "rel1", "B"])
        prior.update([path], correct_entities={"B"})
        prior.freeze()
        score = prior.score(path)
        assert score > 0.5, "Correct path should score above 0.5"

    def test_update_wrong_path(self):
        """Relation sequences that lead to correct answers more often score higher.

        The prior tracks success RATE per relation SEQUENCE, not per terminal entity.
        Different relations must be used so the prior can distinguish them.
        """
        prior = self._build_prior()
        # rel_good always leads to a correct answer
        good_paths = [_make_path(["A", "rel_good", "B"]) for _ in range(5)]
        # rel_bad never leads to a correct answer
        bad_paths = [_make_path(["A", "rel_bad", "C"]) for _ in range(5)]
        prior.update(good_paths, correct_entities={"B"})
        prior.update(bad_paths, correct_entities={"B"})  # C is not correct
        prior.freeze()
        assert prior.score(_make_path(["A", "rel_good", "X"])) > \
               prior.score(_make_path(["A", "rel_bad", "X"]))

    def test_score_unknown_sequence_returns_smoothed(self):
        """Unseen relation sequences return smoothing/(smoothing+0) = 1.0 (prior)."""
        prior = self._build_prior()
        prior.freeze()  # empty prior
        path = _make_path(["X", "unseen_rel", "Y"])
        score = prior.score(path)
        assert 0.0 <= score <= 1.0

    def test_score_with_prefix_fallback(self):
        """score_with_prefix falls back to shorter prefix when full seq unseen."""
        from reasoning.relation_path_prior import RelationPathPrior
        prior = RelationPathPrior(smoothing=1.0, max_len=3, min_count=1)
        # Train on 1-hop rel1 being correct
        short_path = _make_path(["A", "rel1", "B"])
        prior.update([short_path], correct_entities={"B"})
        prior.freeze()

        # Query with 2-hop path starting with same rel1
        long_path = _make_path(["A", "rel1", "B", "rel2", "C"])
        score_full = prior.score(long_path)
        score_prefix = prior.score_with_prefix(long_path)
        # Prefix fallback to rel1 should be at least as good as full unknown seq
        assert score_prefix >= score_full

    def test_freeze_prevents_update(self):
        prior = self._build_prior()
        prior.freeze()
        path = _make_path(["A", "rel1", "B"])
        with pytest.raises(RuntimeError):
            prior.update([path], correct_entities={"B"})

    def test_top_sequences(self):
        """top_sequences returns the n highest-scoring relation sequences."""
        prior = self._build_prior()
        paths_correct = [_make_path(["A", "rel1", "B"]) for _ in range(5)]
        paths_wrong   = [_make_path(["A", "rel2", "C"]) for _ in range(2)]
        prior.update(paths_correct, correct_entities={"B"})
        prior.update(paths_wrong, correct_entities={"B"})
        prior.freeze()

        top = prior.top_sequences(n=3)
        assert len(top) <= 3
        assert isinstance(top[0], tuple), "Each entry should be (seq_tuple, score)"

    def test_multi_hop_sequence(self):
        """2-hop paths store their full relation sequence."""
        prior = self._build_prior()
        path = _make_path(["A", "rel1", "B", "rel2", "C"])
        prior.update([path], correct_entities={"C"})
        prior.freeze()
        score = prior.score(path)
        assert score > 0.5

    def test_score_returns_float_in_unit_interval(self):
        prior = self._build_prior()
        for _ in range(10):
            path = _make_path(["A", "r", "B"])
            prior.update([path], correct_entities={"B"})
        prior.freeze()
        path = _make_path(["A", "r", "B"])
        s = prior.score(path)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# GraphRelationPrior
# ---------------------------------------------------------------------------

class TestGraphRelationPrior:
    def _build_prior_from_rels(self, relations: List[str]):
        """Build GraphRelationPrior via fit() on a small NetworkX graph."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from reasoning.relation_path_prior import GraphRelationPrior
        G = nx.DiGraph()
        for i, rel in enumerate(relations):
            G.add_edge(f"n{i}", f"n{i+1}", relation=rel)
        adapter = NetworkXAdapter(G)
        prior = GraphRelationPrior(decay=0.8)
        prior.fit(adapter)
        return prior

    def test_score_known_relation(self):
        """Relations present in graph return non-trivial scores."""
        prior = self._build_prior_from_rels(["rel_common"] * 10 + ["rel_rare"] * 1)
        common_path = _make_path(["A", "rel_common", "B"])
        rare_path   = _make_path(["A", "rel_rare", "B"])
        assert prior.score(common_path) >= prior.score(rare_path)

    def test_score_unknown_relation_returns_default(self):
        """Unknown relations return default fallback score in valid range."""
        prior = self._build_prior_from_rels(["rel1", "rel2"])
        path = _make_path(["A", "unknown_rel", "B"])
        score = prior.score(path)
        assert 0.0 <= score <= 1.0

    def test_score_multi_hop(self):
        """Multi-hop paths produce valid scores."""
        prior = self._build_prior_from_rels(["r1"] * 10 + ["r2"] * 5)
        path = _make_path(["A", "r1", "B", "r2", "C"])
        score = prior.score(path)
        assert 0.0 <= score <= 1.0

    def test_fit_from_adapter(self):
        """fit() builds _rel_score from adapter; frequent relations score higher."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from reasoning.relation_path_prior import GraphRelationPrior

        G = nx.DiGraph()
        G.add_edge("A", "B", relation="friends")
        G.add_edge("B", "C", relation="friends")
        G.add_edge("A", "C", relation="knows")

        adapter = NetworkXAdapter(G)
        prior = GraphRelationPrior(decay=0.8)
        prior.fit(adapter)

        path_friends = _make_path(["A", "friends", "B"])
        path_knows   = _make_path(["A", "knows", "B"])
        assert prior.score(path_friends) >= prior.score(path_knows)

    def test_score_returns_float_in_unit_interval(self):
        prior = self._build_prior_from_rels(["r1", "r2", "r3"])
        for rel in ["r1", "r2", "r3", "unknown"]:
            path = _make_path(["A", rel, "B"])
            s = prior.score(path)
            assert 0.0 <= s <= 1.0, f"score out of [0,1] for {rel}: {s}"

    def test_empty_graph_returns_valid_score(self):
        """Prior built on empty graph should not crash and return valid score."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from reasoning.relation_path_prior import GraphRelationPrior
        G = nx.DiGraph()
        G.add_node("A")
        adapter = NetworkXAdapter(G)
        prior = GraphRelationPrior()
        prior.fit(adapter)
        path = _make_path(["A", "r1", "B"])
        s = prior.score(path)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# score_path integration
# ---------------------------------------------------------------------------

class TestScorePathWithPrior:
    def _make_path_with_emb(self, tail_name="B", rel="r1"):
        path = _make_path(["A", rel, tail_name])
        path.embedding = np.random.randn(4).astype(np.float32)
        return path

    def test_score_path_with_graph_prior(self):
        """score_path() with GraphRelationPrior active differentiates known rels."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from reasoning.path_scorer import score_path
        from reasoning.relation_path_prior import GraphRelationPrior

        # r1 appears 9x more than r2 → prior score(r1) >> prior score(r2)
        G = nx.DiGraph()
        for i in range(9):
            G.add_edge(f"a{i}", f"b{i}", relation="r1")
        G.add_edge("x", "y", relation="r2")
        prior = GraphRelationPrior(decay=0.8)
        prior.fit(NetworkXAdapter(G))

        p_good = _make_path(["A", "r1", "B"])
        p_bad  = _make_path(["A", "r2", "B"])

        s_good = score_path(p_good, relation_prior=prior, weight_prior=0.3)
        s_bad  = score_path(p_bad,  relation_prior=prior, weight_prior=0.3)
        assert s_good > s_bad

    def test_score_path_no_prior_unchanged(self):
        """score_path without prior should behave identically to baseline."""
        from reasoning.path_scorer import score_path
        p = _make_path(["A", "r1", "B"], attention_weights=[0.7])
        s_with    = score_path(p, relation_prior=None, weight_prior=0.15)
        s_without = score_path(p)
        assert abs(s_with - s_without) < 1e-9

    def test_score_normalised_to_unit_interval(self):
        """score_path should always return a value in [0, 1]."""
        from reasoning.path_scorer import score_path
        from reasoning.relation_path_prior import GraphRelationPrior

        prior = GraphRelationPrior(decay=0.8)
        prior._rel_score = {"r1": 1.0}
        prior._decay_dummy = 1.0
        prior._fitted = True

        query_emb = np.random.randn(4).astype(np.float32)
        for _ in range(20):
            p = _make_path(["A", "r1", "B"])
            p.embedding = np.random.randn(4).astype(np.float32)
            s = score_path(p, query_embedding=query_emb,
                           relation_prior=prior, weight_prior=0.2)
            assert 0.0 <= s <= 1.0, f"Out of [0,1]: {s}"


# ---------------------------------------------------------------------------
# extract() integration
# ---------------------------------------------------------------------------

class TestExtractWithPrior:
    def test_extract_with_prior_changes_ranking(self):
        """relation_prior passed to extract() should influence answer ranking."""
        from reasoning.answer_extractor import extract
        from reasoning.relation_path_prior import GraphRelationPrior

        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        G = nx.DiGraph()
        for i in range(9):
            G.add_edge(f"a{i}", f"b{i}", relation="good_rel")
        G.add_edge("x", "y", relation="bad_rel")
        prior = GraphRelationPrior(decay=0.8)
        prior.fit(NetworkXAdapter(G))

        # Two paths with same attention weight but different relations
        # Give p_bad a slightly higher attention weight to make it win without prior
        p_good = _make_path(["seed", "good_rel", "answer_good"],
                             attention_weights=[0.5])
        p_bad  = _make_path(["seed", "bad_rel", "answer_bad"],
                             attention_weights=[0.6])  # higher attention = wins without prior

        # Without prior: p_bad wins (higher attention weight)
        answers_no_prior = extract([p_good, p_bad], top_k=2, relation_prior=None)
        assert answers_no_prior[0].entity_id == "answer_bad"

        # With prior at high weight: prior should overcome attention difference
        answers_with_prior = extract([p_good, p_bad], top_k=2,
                                      relation_prior=prior, weight_prior=0.5)

        ids_with = [a.entity_id for a in answers_with_prior]
        assert "answer_good" in ids_with
        # good_rel (1.0) + lower attention should beat bad_rel (0.0) + higher attention
        assert ids_with[0] == "answer_good"

    def test_extract_no_prior_unchanged(self):
        """extract() without prior returns same results as before."""
        from reasoning.answer_extractor import extract

        paths = [
            _make_path(["seed", "r1", "ans1"], attention_weights=[0.9]),
            _make_path(["seed", "r2", "ans2"], attention_weights=[0.4]),
        ]
        results = extract(paths, top_k=2, relation_prior=None)
        assert len(results) == 2
        assert results[0].entity_id == "ans1"  # Higher attention weight wins

    def test_extract_prior_none_vs_weight_zero(self):
        """relation_prior=None and weight_prior=0.0 should produce same ranking."""
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        from reasoning.answer_extractor import extract
        from reasoning.relation_path_prior import GraphRelationPrior

        G = nx.DiGraph()
        for i in range(9):
            G.add_edge(f"a{i}", f"b{i}", relation="r1")
        G.add_edge("x", "y", relation="r2")
        prior = GraphRelationPrior(decay=0.8)
        prior.fit(NetworkXAdapter(G))

        paths = [
            _make_path(["s", "r1", "a1"], attention_weights=[0.7]),
            _make_path(["s", "r2", "a2"], attention_weights=[0.7]),
        ]

        results_none = extract(paths, top_k=2, relation_prior=None)
        results_zero = extract(paths, top_k=2, relation_prior=prior, weight_prior=0.0)

        ids_none = [a.entity_id for a in results_none]
        ids_zero = [a.entity_id for a in results_zero]
        assert ids_none == ids_zero

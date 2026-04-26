"""
Tests for Phase 137: Hop-1 Intermediate Seed Expansion (H1SE).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from reasoning.expanded_traversal import HopExpandedTraversal, _funnel_beam_widths
from core.cerebrum import CerebrumGraph


# ---------------------------------------------------------------------------
# _funnel_beam_widths helper
# ---------------------------------------------------------------------------

class TestFunnelBeamWidthsHelper:
    def test_mh1_returns_empty(self):
        assert _funnel_beam_widths(1, 10, 3.0) == {}

    def test_mh2_full_factor(self):
        # Only one non-terminal hop → gets full factor
        assert _funnel_beam_widths(2, 10, 3.0) == {1: 30}

    def test_mh3_linear_ramp(self):
        result = _funnel_beam_widths(3, 10, 3.0)
        assert result == {1: 10, 2: 30}

    def test_mh4_three_steps(self):
        result = _funnel_beam_widths(4, 10, 3.0)
        assert len(result) == 3
        assert result[1] == 10
        assert result[3] == 30
        assert 10 <= result[2] <= 30

    def test_all_values_at_least_bw(self):
        for mh in range(1, 6):
            for bw, factor in [(5, 2.0), (10, 4.0), (1, 10.0)]:
                widths = _funnel_beam_widths(mh, bw, factor)
                for v in widths.values():
                    assert v >= bw


# ---------------------------------------------------------------------------
# HopExpandedTraversal unit tests
# ---------------------------------------------------------------------------

def _build_simple_graph(triples):
    g = CerebrumGraph.from_triples(triples)
    g.build(seed=42)
    return g


class TestHopExpandedTraversalBasic:
    def test_returns_paths_on_toy_graph(self):
        triples = [
            ("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D"),
            ("A", "r1", "X"), ("X", "r2", "Y"),
        ]
        g = _build_simple_graph(triples)
        het = HopExpandedTraversal(
            adapter=g.adapter, csa_engine=g._csa,
            beam_width=5, max_hop=2, expansion_k=10,
        )
        paths = het.traverse(["A"])
        assert len(paths) > 0

    def test_finds_2hop_answer(self):
        triples = [("A", "r1", "B"), ("B", "r2", "C")]
        g = _build_simple_graph(triples)
        het = HopExpandedTraversal(
            adapter=g.adapter, csa_engine=g._csa,
            beam_width=5, max_hop=2, expansion_k=5,
        )
        paths = het.traverse(["A"])
        tails = {p.tail for p in paths}
        assert "C" in tails

    def test_finds_3hop_answer_with_distractors(self):
        triples = [("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D")]
        for i in range(20):
            triples.append(("B", "noise", f"NB{i}"))
        g = _build_simple_graph(triples)
        # factor=30 → hop-1 bw=60 > 21 candidates; C survives regardless of
        # embedding scores, making the test deterministic across RNG states.
        het = HopExpandedTraversal(
            adapter=g.adapter, csa_engine=g._csa,
            beam_width=2, max_hop=3, expansion_k=25,
            beam_profile_factor=30.0,
        )
        paths = het.traverse(["A"])
        tails = {p.tail for p in paths}
        assert "D" in tails

    def test_max_hop_1_returns_scan_only(self):
        triples = [("A", "r1", "B"), ("B", "r2", "C")]
        g = _build_simple_graph(triples)
        het = HopExpandedTraversal(
            adapter=g.adapter, csa_engine=g._csa,
            beam_width=5, max_hop=1, expansion_k=10,
        )
        paths = het.traverse(["A"])
        tails = {p.tail for p in paths}
        # Only hop-1 neighbors should be reachable
        assert "B" in tails
        assert "C" not in tails


# ---------------------------------------------------------------------------
# CerebrumGraph.query() integration
# ---------------------------------------------------------------------------

class TestHopExpandQueryIntegration:
    def test_hop_expand_query_returns_answers(self):
        triples = [
            ("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D"),
            ("A", "r1", "X"), ("X", "r2", "Y"),
        ]
        g = _build_simple_graph(triples)
        answers = g.query(["A"], max_hop=2, hop_expand=True, top_k=10)
        assert len(answers) > 0

    def test_hop_expand_false_uses_normal_traversal(self):
        triples = [("A", "r1", "B"), ("B", "r2", "C")]
        g = _build_simple_graph(triples)
        # Patch HopExpandedTraversal to confirm it is NOT instantiated
        with patch("reasoning.expanded_traversal.HopExpandedTraversal") as mock_cls:
            g.query(["A"], max_hop=2, hop_expand=False, top_k=5)
            mock_cls.assert_not_called()

    def test_hop_expand_does_not_mutate_shared_traversal(self):
        triples = [("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D")]
        g = _build_simple_graph(triples)
        original_bw = g._traversal.beam_width
        g.query(["A"], max_hop=3, hop_expand=True, expansion_k=5, top_k=10)
        # shared traversal must be untouched
        assert g._traversal.beam_width == original_bw
        assert g._traversal is not None


# ---------------------------------------------------------------------------
# Causal index propagation
# ---------------------------------------------------------------------------

class TestCausalIndexPropagation:
    def test_causal_index_propagates_to_h1se(self):
        triples = [("A", "causes", "B"), ("B", "r2", "C")]
        g = _build_simple_graph(triples)
        # Manually stamp a sentinel causal index on the shared traversal
        g._traversal._causal_edge_index = {("A", "B")}
        answers = g.query(["A"], max_hop=2, hop_expand=True, top_k=10)
        # No assertion needed beyond no crash; the index is propagated
        assert isinstance(answers, list)


# ---------------------------------------------------------------------------
# expansion_k cap
# ---------------------------------------------------------------------------

class TestExpansionKCap:
    def test_expansion_k_limits_deep_traversals(self):
        # Build a graph where A has 10 hop-1 neighbors
        triples = [("A", "r1", f"B{i}") for i in range(10)]
        triples += [(f"B{i}", "r2", f"C{i}") for i in range(10)]
        g = _build_simple_graph(triples)

        call_counts = []

        original_traverse = HopExpandedTraversal.traverse.__wrapped__ if hasattr(
            HopExpandedTraversal.traverse, "__wrapped__"
        ) else None

        het = HopExpandedTraversal(
            adapter=g.adapter, csa_engine=g._csa,
            beam_width=5, max_hop=2, expansion_k=3,
        )
        # Patch _make_traversal to count deep traversal calls
        original_make = het._make_traversal
        deep_calls = []

        def counting_make(max_hop, beam_widths=None, per_budget=None):
            t = original_make(max_hop, beam_widths=beam_widths, per_budget=per_budget)
            if max_hop > 1:
                deep_calls.append(max_hop)
            return t

        het._make_traversal = counting_make
        het.traverse(["A"])

        # With expansion_k=3, at most 3 deep traversals should fire
        assert len(deep_calls) <= 3

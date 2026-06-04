"""
Unit tests for CerebrumGraph â€” the unified THALAMUS â†’ CORTEX pipeline.

All tests use small hand-crafted graphs (4-6 nodes) and RandomEngine
embeddings to keep execution fast.  No mocking of internal components;
tests go through the public API.
"""
from __future__ import annotations
from typing import Set

import tempfile
import os
from pathlib import Path

import networkx as nx
import pytest

from core.cerebrum import CerebrumGraph
from adapters.networkx_adapter import NetworkXAdapter
from core.graph_completion import InverseRule, CompositionRule
from reasoning.answer_extractor import Answer


# ---------------------------------------------------------------------------
# Shared small graphs
# ---------------------------------------------------------------------------

# Star topology: alice is the hub
STAR_TRIPLES = [
    ("alice", "KNOWS",      "bob"),
    ("alice", "KNOWS",      "carol"),
    ("alice", "KNOWS",      "dave"),
    ("bob",   "WORKS_WITH", "carol"),
    ("carol", "WORKS_WITH", "dave"),
    ("dave",  "KNOWS",      "eve"),
    ("eve",   "MARRIED_TO", "bob"),
    ("alice", "MARRIED_TO", "frank"),
]

# Triangle topology for inverse/composition tests
TRIANGLE_TRIPLES = [
    ("A", "directed_by", "B"),
    ("B", "born_in",     "C"),
    ("C", "KNOWS",       "A"),
    ("A", "KNOWS",       "D"),
]


def build_star_graph(**kwargs) -> CerebrumGraph:
    """Return a fully built CerebrumGraph from STAR_TRIPLES."""
    g = CerebrumGraph.from_triples(STAR_TRIPLES, **kwargs)
    g.build(seed=42)
    return g


# ---------------------------------------------------------------------------
# Factory method tests
# ---------------------------------------------------------------------------

class TestFactoryMethods:

    def test_from_triples_builds_graph(self):
        """from_triples() creates a CerebrumGraph with the right node/edge counts."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        assert g.node_count == 6
        # Not yet built â€” just check the adapter has nodes
        assert g.adapter.node_count() == 6

    def test_from_triples_directed_default(self):
        """from_triples() defaults to directed (DiGraph)."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        G = g.adapter.to_networkx()
        assert G.is_directed()

    def test_from_triples_undirected(self):
        """from_triples(directed=False) creates undirected graph."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES, directed=False)
        G = g.adapter.to_networkx()
        assert not G.is_directed()

    def test_from_triples_is_not_built(self):
        """from_triples() returns an unbuilt graph (is_built == False)."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        assert g.is_built is False

    def test_from_kb_loads_file(self, tmp_path):
        """from_kb() loads triples from a pipe-separated file."""
        kb = tmp_path / "kb.txt"
        lines = [
            "alice|KNOWS|bob",
            "bob|KNOWS|carol",
            "carol|KNOWS|dave",
        ]
        kb.write_text("\n".join(lines), encoding="utf-8")

        g = CerebrumGraph.from_kb(str(kb), sep="|")
        assert g.node_count == 4
        assert g.is_built is False

    def test_from_kb_skips_blank_and_comment_lines(self, tmp_path):
        """from_kb() ignores blank lines and lines starting with #."""
        kb = tmp_path / "kb.txt"
        content = (
            "# This is a comment\n"
            "\n"
            "alice|KNOWS|bob\n"
            "# another comment\n"
            "bob|KNOWS|carol\n"
        )
        kb.write_text(content, encoding="utf-8")
        g = CerebrumGraph.from_kb(str(kb), sep="|")
        assert g.node_count == 3

    def test_from_adapter_wraps_existing_adapter(self):
        """from_adapter() accepts an existing NetworkXAdapter."""
        G = nx.DiGraph()
        G.add_edge("x", "y", relation="RELATED")
        G.add_edge("y", "z", relation="RELATED")
        adapter = NetworkXAdapter(G)

        g = CerebrumGraph.from_adapter(adapter)
        assert g.node_count == 3
        assert g.adapter is adapter

    def test_from_triples_repr_not_built(self):
        """repr() mentions 'not built' before build()."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        assert "not built" in repr(g)


# ---------------------------------------------------------------------------
# complete() method tests
# ---------------------------------------------------------------------------

class TestCompleteMethod:

    def test_complete_applies_inverse_rule(self):
        """complete() with InverseRule adds reverse edges to the adapter graph."""
        g = CerebrumGraph.from_triples(TRIANGLE_TRIPLES)
        G_before = g.adapter._G.number_of_edges()

        g.complete([InverseRule("directed_by", "director_of")])
        G_after = g.adapter._G.number_of_edges()

        assert G_after > G_before, "inverse rule should add at least one edge"
        assert g.adapter._G.has_edge("B", "A"), "Bâ†’A (director_of) should exist"

    def test_complete_invalidates_built_state(self):
        """complete() sets is_built to False even if called on an already-built graph."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        g.build(seed=42)
        assert g.is_built is True

        g.complete([InverseRule("KNOWS")])
        assert g.is_built is False, "complete() must invalidate built state"

    def test_complete_returns_self_for_chaining(self):
        """complete() returns self so calls can be chained."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        result = g.complete([InverseRule("KNOWS")])
        assert result is g

    def test_complete_applies_composition_rule(self):
        """complete() with CompositionRule adds composed edges."""
        g = CerebrumGraph.from_triples(TRIANGLE_TRIPLES)
        g.complete([CompositionRule("directed_by", "born_in", "director_birthplace")])

        assert g.adapter._G.has_edge("A", "C"), "composed Aâ†’C edge should exist"
        data = g.adapter._G.get_edge_data("A", "C")
        assert data["relation"] == "director_birthplace"

    def test_complete_multiple_rules(self):
        """complete() applies a list of rules in sequence."""
        g = CerebrumGraph.from_triples(TRIANGLE_TRIPLES)
        before = g.adapter._G.number_of_edges()
        g.complete([
            InverseRule("directed_by", "director_of"),
            CompositionRule("directed_by", "born_in", "director_birthplace"),
        ])
        after = g.adapter._G.number_of_edges()
        # At minimum: 1 inverse edge + 1 composed edge
        assert after >= before + 2


# ---------------------------------------------------------------------------
# build() method tests
# ---------------------------------------------------------------------------

class TestBuildMethod:

    def test_build_sets_is_built(self):
        """After build(), is_built is True."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        g.build(seed=42)
        assert g.is_built is True

    def test_build_produces_communities(self):
        """After build(), community_count > 0."""
        g = build_star_graph()
        assert g.community_count > 0

    def test_build_node_count_positive(self):
        """After build(), node_count is positive."""
        g = build_star_graph()
        assert g.node_count > 0

    def test_build_edge_count_positive(self):
        """After build(), edge_count is positive."""
        g = build_star_graph()
        assert g.edge_count > 0

    def test_build_returns_self_for_chaining(self):
        """build() returns self so it can be chained."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        result = g.build(seed=42)
        assert result is g

    def test_build_twice_does_not_raise(self):
        """Calling build() twice on the same graph should not raise."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        g.build(seed=42)
        g.build(seed=42)  # second call â€” must not raise
        assert g.is_built is True

    def test_build_empty_graph_raises(self):
        """build() raises ValueError when the graph has no nodes."""
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        g = CerebrumGraph.from_adapter(adapter)
        with pytest.raises(ValueError, match="no nodes"):
            g.build()

    def test_repr_after_build_shows_built(self):
        """repr() mentions 'built' (not 'not built') after build()."""
        g = build_star_graph()
        r = repr(g)
        assert "not built" not in r
        assert "built" in r


# ---------------------------------------------------------------------------
# query() method tests
# ---------------------------------------------------------------------------

class TestQueryMethod:

    def test_query_returns_answer_list(self):
        """query() returns a list of Answer objects."""
        g = build_star_graph()
        answers = g.query(["alice"])
        assert isinstance(answers, list)
        for a in answers:
            assert isinstance(a, Answer)

    def test_query_raises_before_build(self):
        """query() raises RuntimeError if called before build()."""
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        with pytest.raises(RuntimeError, match="build"):
            g.query(["alice"])

    def test_query_top_k_limits_results(self):
        """top_k limits the number of returned answers."""
        g = build_star_graph()
        answers = g.query(["alice"], top_k=2)
        assert len(answers) <= 2

    def test_query_results_sorted_by_score(self):
        """Returned answers are ordered by score descending."""
        g = build_star_graph()
        answers = g.query(["alice"], top_k=10)
        if len(answers) >= 2:
            scores = [a.score for a in answers]
            assert scores == sorted(scores, reverse=True)

    def test_query_unknown_seed_returns_empty(self):
        """query() returns empty list (not an error) for seed not in graph."""
        g = build_star_graph()
        answers = g.query(["entity_that_does_not_exist_xyz123"])
        assert answers == []

    def test_query_max_hop_1_limits_depth(self):
        """max_hop=1 only returns direct neighbours (1-hop)."""
        g = build_star_graph()
        answers_1 = g.query(["alice"], max_hop=1, top_k=20)
        answers_3 = g.query(["alice"], max_hop=3, top_k=20)

        # 1-hop answers must be a subset of 3-hop answers
        ids_1 = {a.entity_id for a in answers_1}
        ids_3 = {a.entity_id for a in answers_3}
        assert ids_1.issubset(ids_3), "1-hop answers should be a subset of 3-hop"

        # The 3-hop set should in general be at least as large (or equal for
        # small graphs where all nodes are reachable in 1 hop)
        assert len(answers_3) >= len(answers_1)

    def test_query_min_hop_2_excludes_depth_1(self):
        """
        min_hop=2 filters out paths of depth 1.

        Build a linear chain: start â†’ mid â†’ end
        With min_hop=2, 'mid' (1-hop from start) should NOT appear in results
        when it is unreachable at depth >= 2 (no back-edges in this chain).
        We verify this by checking that no path with hop_depth < 2 reaches
        an answer entity via the query's best_path.
        """
        # Linear chain â€” end is only reachable in 2 hops from start
        chain = [
            ("start", "step",  "mid"),
            ("mid",   "step",  "end"),
        ]
        g = CerebrumGraph.from_triples(chain).build(seed=42)

        answers = g.query(["start"], min_hop=2, max_hop=3, top_k=10)
        returned_ids = {a.entity_id for a in answers}

        # 'mid' is only reachable in 1 hop; with min_hop=2 it must not appear
        assert "mid" not in returned_ids, (
            "min_hop=2 should exclude nodes reachable only at depth 1"
        )
        # 'end' may appear (reachable at depth 2)
        # (no assertion needed â€” beam may or may not find it on a 2-node chain)

    def test_query_returns_unique_entity_ids(self):
        """Each entity_id appears at most once in the returned list."""
        g = build_star_graph()
        answers = g.query(["alice"], top_k=20)
        ids = [a.entity_id for a in answers]
        assert len(ids) == len(set(ids)), "duplicate entity_id in results"

    def test_query_answer_scores_non_negative(self):
        """All returned answer scores are >= 0."""
        g = build_star_graph()
        for a in g.query(["alice"], top_k=10):
            assert a.score >= 0.0

    def test_query_multiple_seeds(self):
        """query() accepts multiple seed entities."""
        g = build_star_graph()
        answers = g.query(["alice", "bob"], top_k=10)
        assert isinstance(answers, list)

    def test_query_complete_then_build_then_query(self):
        """Full pipeline: from_triples â†’ complete â†’ build â†’ query works end-to-end."""
        g = (
            CerebrumGraph.from_triples(TRIANGLE_TRIPLES)
            .complete([InverseRule("directed_by", "director_of")])
            .build(seed=42)
        )
        assert g.is_built
        answers = g.query(["A"], top_k=5)
        assert isinstance(answers, list)


# ---------------------------------------------------------------------------
# enhance() method tests
# ---------------------------------------------------------------------------

class TestEnhanceMethod:

    def test_enhance_applies_bridge_rule(self):
        """enhance() with GraphBridgeEngine adds similarity edges between components."""
        from core.graph_bridge import GraphBridgeEngine

        # Two disconnected components with same labels
        triples = [
            ("n1", "r", "n2"),
            ("n3", "r", "n4"),
        ]
        g = CerebrumGraph.from_triples(triples, embeddings="random")
        # Set labels so they match
        G = g.adapter.to_networkx()
        G.nodes["n1"]["label"] = "apple"
        G.nodes["n3"]["label"] = "apple"

        before = G.number_of_edges()
        g.enhance([GraphBridgeEngine(min_similarity=0.9)])
        after = G.number_of_edges()

        assert after > before
        # In directed mode, it adds node1->node3 AND node3->node1
        assert G.has_edge("n1", "node3") or G.has_edge("n1", "n3")

    def test_enhance_invalidates_built_state(self):
        """enhance() sets is_built to False."""
        from core.graph_bridge import GraphBridgeEngine
        g = build_star_graph()
        assert g.is_built is True

        g.enhance([GraphBridgeEngine()])
        assert g.is_built is False

    def test_enhance_returns_self(self):
        """enhance() returns self for chaining."""
        from core.graph_bridge import GraphBridgeEngine
        g = CerebrumGraph.from_triples(STAR_TRIPLES)
        assert g.enhance([GraphBridgeEngine()]) is g

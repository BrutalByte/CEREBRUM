"""
Unit tests for provable graph completion rules:
  - InverseRule
  - CompositionRule

Tests operate directly on NetworkXAdapter instances to verify edge-level
semantics without invoking the full CerebrumGraph pipeline.
"""
import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.graph_completion import InverseRule, CompositionRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_directed_adapter(edges):
    """
    Build a directed NetworkXAdapter from a list of
    (src, tgt, relation, confidence) tuples.
    """
    G = nx.DiGraph()
    for src, tgt, rel, conf in edges:
        G.add_edge(src, tgt, relation=rel, confidence=conf, weight=conf)
    return NetworkXAdapter(G)


def get_edge_data(adapter, src, tgt):
    return adapter._G.get_edge_data(src, tgt)


# ---------------------------------------------------------------------------
# InverseRule tests
# ---------------------------------------------------------------------------

class TestInverseRule:

    def _base_adapter(self):
        return make_directed_adapter([
            ("alice", "bob",   "MARRIED_TO", 0.9),
            ("carol", "dave",  "MARRIED_TO", 0.8),
            ("alice", "carol", "directed_by", 1.0),
        ])

    # ------------------------------------------------------------------
    # Basic behaviour
    # ------------------------------------------------------------------

    def test_apply_directed_adds_reverse_edges(self):
        """apply() on directed graph adds correct reverse edges with the inverse relation."""
        adapter = self._base_adapter()
        rule = InverseRule("directed_by", "director_of")
        n = rule.apply(adapter)

        assert n == 1
        assert adapter._G.has_edge("carol", "alice"), "reverse edge missing"
        data = get_edge_data(adapter, "carol", "alice")
        assert data["relation"] == "director_of"

    def test_apply_symmetric_uses_same_relation(self):
        """apply() with no inverse specified uses the same relation name (symmetric)."""
        adapter = self._base_adapter()
        rule = InverseRule("MARRIED_TO")  # no inverse → symmetric
        n = rule.apply(adapter)

        assert n == 2  # two MARRIED_TO forward edges → two reverse edges
        for src, tgt in [("bob", "alice"), ("dave", "carol")]:
            assert adapter._G.has_edge(src, tgt), f"missing {src}→{tgt}"
            data = get_edge_data(adapter, src, tgt)
            assert data["relation"] == "MARRIED_TO"

    def test_apply_skips_existing_inverse(self):
        """apply() does not duplicate an edge whose inverse already exists."""
        G = nx.DiGraph()
        G.add_edge("a", "b", relation="KNOWS", confidence=1.0, weight=1.0)
        G.add_edge("b", "a", relation="KNOWS", confidence=1.0, weight=1.0)
        adapter = NetworkXAdapter(G)

        rule = InverseRule("KNOWS")
        n = rule.apply(adapter)
        assert n == 0, "should skip edge whose inverse already exists"

    def test_added_edge_synthetic_flag(self):
        """All added edges have synthetic=True."""
        adapter = self._base_adapter()
        InverseRule("MARRIED_TO").apply(adapter)

        for src, tgt in [("bob", "alice"), ("dave", "carol")]:
            data = get_edge_data(adapter, src, tgt)
            assert data.get("synthetic") is True

    def test_added_edge_confidence_matches_source(self):
        """Added edges inherit confidence from the source edge."""
        adapter = self._base_adapter()
        InverseRule("MARRIED_TO").apply(adapter)

        # alice→bob has confidence 0.9 → reverse bob→alice should also be 0.9
        data = get_edge_data(adapter, "bob", "alice")
        assert data["confidence"] == pytest.approx(0.9)

        data2 = get_edge_data(adapter, "dave", "carol")
        assert data2["confidence"] == pytest.approx(0.8)

    def test_added_edge_provenance_format(self):
        """Provenance string matches: rule:inverse:{R}→{R_inv}|source:{src}→{tgt}"""
        adapter = self._base_adapter()
        InverseRule("directed_by", "director_of").apply(adapter)

        data = get_edge_data(adapter, "carol", "alice")
        expected = "rule:inverse:directed_by→director_of|source:alice→carol"
        assert data["provenance"] == expected

    def test_apply_returns_count(self):
        """apply() returns the exact count of new edges added."""
        adapter = self._base_adapter()
        n = InverseRule("MARRIED_TO").apply(adapter)
        assert n == 2

    def test_apply_returns_zero_when_no_matching_relation(self):
        """apply() returns 0 when the target relation is absent."""
        adapter = self._base_adapter()
        n = InverseRule("nonexistent_relation").apply(adapter)
        assert n == 0

    def test_describe_returns_string(self):
        """describe() returns a non-empty string for both symmetric and asymmetric cases."""
        r_sym = InverseRule("MARRIED_TO")
        r_asym = InverseRule("directed_by", "director_of")
        assert isinstance(r_sym.describe(), str) and r_sym.describe()
        assert isinstance(r_asym.describe(), str) and r_asym.describe()

    def test_describe_contains_relation_name(self):
        """describe() mentions the relation name."""
        desc = InverseRule("starred_actors").describe()
        assert "starred_actors" in desc

    def test_symmetric_provenance_self_inverse(self):
        """Symmetric rule provenance cites the same relation name on both sides."""
        G = nx.DiGraph()
        G.add_edge("x", "y", relation="SIBLING_OF", confidence=0.7, weight=0.7)
        adapter = NetworkXAdapter(G)
        InverseRule("SIBLING_OF").apply(adapter)

        data = get_edge_data(adapter, "y", "x")
        assert "SIBLING_OF→SIBLING_OF" in data["provenance"]
        assert "source:x→y" in data["provenance"]

    def test_idempotent_second_application(self):
        """Calling apply() twice does not add duplicate edges."""
        adapter = self._base_adapter()
        n1 = InverseRule("MARRIED_TO").apply(adapter)
        n2 = InverseRule("MARRIED_TO").apply(adapter)
        assert n1 == 2
        assert n2 == 0, "second pass should add nothing"


# ---------------------------------------------------------------------------
# CompositionRule tests
# ---------------------------------------------------------------------------

class TestCompositionRule:

    def _chain_adapter(self):
        """
        A  --directed_by-->  B  --born_in-->  C
        Also includes D --directed_by--> E --born_in--> F
        So: A→C and D→F should be composed.
        """
        return make_directed_adapter([
            ("A", "B", "directed_by", 1.0),
            ("B", "C", "born_in",     1.0),
            ("D", "E", "directed_by", 0.9),
            ("E", "F", "born_in",     0.8),
        ])

    # ------------------------------------------------------------------
    # Basic behaviour
    # ------------------------------------------------------------------

    def test_apply_adds_composed_edge(self):
        """apply() creates A→C edge when A→R1→B→R2→C path exists."""
        adapter = self._chain_adapter()
        rule = CompositionRule("directed_by", "born_in", "director_birthplace")
        n = rule.apply(adapter)

        assert n == 2  # A→C and D→F
        assert adapter._G.has_edge("A", "C")
        assert adapter._G.has_edge("D", "F")

    def test_applied_edge_relation_name(self):
        """Composed edge carries the composed_relation name."""
        adapter = self._chain_adapter()
        CompositionRule("directed_by", "born_in", "director_birthplace").apply(adapter)

        data = get_edge_data(adapter, "A", "C")
        assert data["relation"] == "director_birthplace"

    def test_applied_edge_synthetic_flag(self):
        """Composed edges have synthetic=True."""
        adapter = self._chain_adapter()
        CompositionRule("directed_by", "born_in", "director_birthplace").apply(adapter)

        data = get_edge_data(adapter, "A", "C")
        assert data.get("synthetic") is True

    def test_applied_edge_confidence_weakest_link(self):
        """Confidence equals min(conf_AB, conf_BC) — weakest-link."""
        adapter = self._chain_adapter()
        CompositionRule("directed_by", "born_in", "director_birthplace").apply(adapter)

        # A→B: 1.0, B→C: 1.0 → min = 1.0
        data_ac = get_edge_data(adapter, "A", "C")
        assert data_ac["confidence"] == pytest.approx(1.0)

        # D→E: 0.9, E→F: 0.8 → min = 0.8
        data_df = get_edge_data(adapter, "D", "F")
        assert data_df["confidence"] == pytest.approx(0.8)

    def test_applied_edge_provenance_format(self):
        """Provenance matches: rule:compose:{R1}+{R2}→{Rc}|path:{a}→{b}→{c}"""
        adapter = self._chain_adapter()
        CompositionRule("directed_by", "born_in", "director_birthplace").apply(adapter)

        data = get_edge_data(adapter, "A", "C")
        prov = data["provenance"]
        assert prov.startswith("rule:compose:directed_by+born_in→director_birthplace")
        assert "|path:A→B→C" in prov

    def test_skips_trivial_cycles(self):
        """A→B→A should NOT produce an A→A self-edge."""
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="r1", confidence=1.0, weight=1.0)
        G.add_edge("B", "A", relation="r2", confidence=1.0, weight=1.0)
        adapter = NetworkXAdapter(G)

        CompositionRule("r1", "r2", "composed").apply(adapter)
        assert not adapter._G.has_edge("A", "A"), "self-loop should not be added"

    def test_min_occurrences_respected(self):
        """
        min_occurrences=2 means the A→C edge is only added when two distinct
        B intermediates support it.
        """
        G = nx.DiGraph()
        # Only one intermediate B for A→C
        G.add_edge("A", "B1", relation="r1", confidence=1.0, weight=1.0)
        G.add_edge("B1", "C",  relation="r2", confidence=1.0, weight=1.0)
        adapter = NetworkXAdapter(G)

        n = CompositionRule("r1", "r2", "composed", min_occurrences=2).apply(adapter)
        assert n == 0, "should not add edge with only one intermediate"
        assert not adapter._G.has_edge("A", "C")

    def test_min_occurrences_two_intermediates(self):
        """With min_occurrences=2 and two B nodes, edge should be added."""
        G = nx.DiGraph()
        G.add_edge("A", "B1", relation="r1", confidence=1.0, weight=1.0)
        G.add_edge("B1", "C",  relation="r2", confidence=1.0, weight=1.0)
        G.add_edge("A", "B2", relation="r1", confidence=0.9, weight=0.9)
        G.add_edge("B2", "C",  relation="r2", confidence=0.95, weight=0.95)
        adapter = NetworkXAdapter(G)

        n = CompositionRule("r1", "r2", "composed", min_occurrences=2).apply(adapter)
        assert n == 1
        assert adapter._G.has_edge("A", "C")

    def test_max_edges_cap(self):
        """max_edges limits the number of edges added."""
        G = nx.DiGraph()
        # Create 5 distinct (A_i→B_i→C) chains
        for i in range(5):
            G.add_edge(f"A{i}", f"B{i}", relation="r1", confidence=1.0, weight=1.0)
            G.add_edge(f"B{i}", "C",      relation="r2", confidence=1.0, weight=1.0)
        adapter = NetworkXAdapter(G)

        n = CompositionRule("r1", "r2", "composed", max_edges=3).apply(adapter)
        assert n == 3, f"expected 3 (capped), got {n}"

    def test_no_duplicate_if_composed_edge_already_exists(self):
        """Does not add edge when a same-relation edge already exists between (A, C)."""
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="r1", confidence=1.0, weight=1.0)
        G.add_edge("B", "C", relation="r2", confidence=1.0, weight=1.0)
        G.add_edge("A", "C", relation="composed", confidence=0.5, weight=0.5)
        adapter = NetworkXAdapter(G)

        n = CompositionRule("r1", "r2", "composed").apply(adapter)
        assert n == 0, "should not overwrite existing edge with same composed relation"

    def test_apply_returns_count(self):
        """apply() returns the number of edges added."""
        adapter = self._chain_adapter()
        n = CompositionRule("directed_by", "born_in", "director_birthplace").apply(adapter)
        assert n == 2

    def test_apply_returns_zero_for_no_paths(self):
        """apply() returns 0 when no matching two-hop path exists."""
        adapter = self._chain_adapter()
        n = CompositionRule("no_such_rel", "born_in", "composed").apply(adapter)
        assert n == 0

    def test_describe_returns_string(self):
        """describe() returns a non-empty informative string."""
        rule = CompositionRule("directed_by", "born_in", "director_birthplace")
        desc = rule.describe()
        assert isinstance(desc, str)
        assert "directed_by" in desc
        assert "born_in" in desc
        assert "director_birthplace" in desc

    def test_idempotent_second_application(self):
        """Calling apply() twice does not produce duplicate edges."""
        adapter = self._chain_adapter()
        rule = CompositionRule("directed_by", "born_in", "director_birthplace")
        n1 = rule.apply(adapter)
        n2 = rule.apply(adapter)
        assert n1 == 2
        assert n2 == 0, "second pass should add nothing"

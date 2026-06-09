"""Unit tests for CommunityHypothesisGenerator (Phase 233)."""
import networkx as nx
import pytest

from core.community_hypothesis import CommunityHypothesisGenerator


# ---------------------------------------------------------------------------
# Minimal adapter stub
# ---------------------------------------------------------------------------

class _Adapter:
    def __init__(self, G: nx.DiGraph, community_map: dict):
        self._G = G
        self.community_map = community_map

    def to_networkx(self):
        return self._G


def _make_adapter(edges, community_map):
    """edges: list of (u, v, rel_str)"""
    G = nx.DiGraph()
    for u, v, rel in edges:
        G.add_edge(u, v, relation_type=rel)
    return _Adapter(G, community_map)


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

def test_build_empty_graph():
    adp = _make_adapter([], {})
    hg = CommunityHypothesisGenerator().build(adp)
    assert hg._bridge_index == {}
    assert hg._outbound_index == {}


def test_build_single_community():
    adp = _make_adapter(
        [("a", "b", "rel1"), ("b", "c", "rel2")],
        {"a": 0, "b": 0, "c": 0},
    )
    hg = CommunityHypothesisGenerator().build(adp)
    assert hg._bridge_index == {}
    assert hg._outbound_index == {}


def test_build_two_communities_one_relation():
    adp = _make_adapter(
        [("a", "b", "crosses")],
        {"a": 0, "b": 1},
    )
    hg = CommunityHypothesisGenerator().build(adp)
    assert (0, 1) in hg._bridge_index
    assert hg._bridge_index[(0, 1)]["crosses"] == 1
    assert hg._outbound_index[0]["crosses"] == 1


def test_build_multiple_communities_multi_relation():
    edges = [
        ("a", "b", "r1"),
        ("a", "c", "r1"),
        ("a", "d", "r2"),
        ("b", "d", "r3"),
    ]
    cmap = {"a": 0, "b": 1, "c": 1, "d": 2}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    # a→b and a→c both cross 0→1 via r1
    assert hg._bridge_index[(0, 1)]["r1"] == 2
    # a→d crosses 0→2 via r2
    assert hg._bridge_index[(0, 2)]["r2"] == 1
    # b→d crosses 1→2 via r3
    assert hg._bridge_index[(1, 2)]["r3"] == 1
    # outbound from community 0: r1×2, r2×1
    assert hg._outbound_index[0]["r1"] == 2
    assert hg._outbound_index[0]["r2"] == 1


def test_intra_community_edges_excluded():
    edges = [("a", "b", "intra"), ("a", "c", "bridge")]
    cmap = {"a": 0, "b": 0, "c": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    assert "intra" not in hg._outbound_index.get(0, {})
    assert hg._outbound_index[0]["bridge"] == 1


def test_unknown_community_skipped():
    # 'x' has no community entry — edge should be skipped
    adp = _make_adapter([("x", "b", "rel")], {"b": 1})
    hg = CommunityHypothesisGenerator().build(adp)
    assert hg._bridge_index == {}


# ---------------------------------------------------------------------------
# top_bridge_relations
# ---------------------------------------------------------------------------

def test_top_bridge_relations_ordering():
    edges = [("a", "b", "r1"), ("a", "c", "r1"), ("a", "d", "r2")]
    cmap = {"a": 0, "b": 1, "c": 1, "d": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    result = hg.top_bridge_relations(0, 1, top_n=5)
    assert result[0] == ("r1", 2)
    assert result[1] == ("r2", 1)


def test_top_bridge_relations_unknown_pair():
    hg = CommunityHypothesisGenerator()
    result = hg.top_bridge_relations(99, 100)
    assert result == []


# ---------------------------------------------------------------------------
# generate_hop_boosts
# ---------------------------------------------------------------------------

def test_generate_hop_boosts_unknown_entity():
    hg = CommunityHypothesisGenerator()
    assert hg.generate_hop_boosts("no_such_entity") == {}


def test_generate_hop_boosts_no_outbound():
    # entity exists in community map but community has no outbound bridges
    adp = _make_adapter([], {"a": 0})
    hg = CommunityHypothesisGenerator().build(adp)
    assert hg.generate_hop_boosts("a") == {}


def test_generate_hop_boosts_returns_top_n():
    # 6 distinct relations from community 0; top_n=3 should return only 3
    edges = [(f"a{i}", "b", f"r{i}") for i in range(6)]
    cmap = {f"a{i}": 0 for i in range(6)}
    cmap["b"] = 1
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    result = hg.generate_hop_boosts("a0", top_n=3)
    assert len(result) == 3


def test_generate_hop_boosts_max_boost():
    edges = [("a", "b", "r1"), ("a", "c", "r1"), ("a", "d", "r2")]
    cmap = {"a": 0, "b": 1, "c": 1, "d": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    boosts = hg.generate_hop_boosts("a", boost_scale=3.0)
    # r1 appears twice (max), so gets 1.0 + 3.0 = 4.0
    assert boosts["r1"] == pytest.approx(4.0)


def test_generate_hop_boosts_relative_ordering():
    edges = [("a", "b", "r1"), ("a", "c", "r1"), ("a", "d", "r2")]
    cmap = {"a": 0, "b": 1, "c": 1, "d": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    boosts = hg.generate_hop_boosts("a")
    assert boosts["r1"] > boosts["r2"]


def test_generate_hop_boosts_all_values_in_range():
    edges = [("a", "b", "r1"), ("a", "c", "r1"), ("a", "d", "r2")]
    cmap = {"a": 0, "b": 1, "c": 1, "d": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    boosts = hg.generate_hop_boosts("a", boost_scale=2.0)
    for v in boosts.values():
        assert 1.0 <= v <= 3.0


# ---------------------------------------------------------------------------
# community_of / adjacent_community_count
# ---------------------------------------------------------------------------

def test_community_of_known():
    adp = _make_adapter([], {"x": 5})
    hg = CommunityHypothesisGenerator().build(adp)
    assert hg.community_of("x") == 5


def test_community_of_unknown():
    hg = CommunityHypothesisGenerator()
    assert hg.community_of("no_such") == -1


def test_adjacent_community_count():
    edges = [("a", "b", "r1"), ("a", "c", "r2"), ("a", "d", "r3")]
    cmap = {"a": 0, "b": 1, "c": 2, "d": 3}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    assert hg.adjacent_community_count(0) == 3
    assert hg.adjacent_community_count(99) == 0


# ---------------------------------------------------------------------------
# generate_typed_boosts / community_reach_types (Phase 233 typed variant)
# ---------------------------------------------------------------------------

def test_community_reach_types_person():
    # people.person.gender bridges from C0 to C1; suffix "gender" → person type
    edges = [("a", "b", "people.person.gender")]
    cmap = {"a": 0, "b": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    assert "person" in hg.community_reach_types(0)


def test_community_reach_types_place():
    edges = [("a", "b", "location.location.containedby")]
    cmap = {"a": 0, "b": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    assert "place" in hg.community_reach_types(0)


def test_community_reach_types_empty_for_unknown():
    hg = CommunityHypothesisGenerator()
    assert hg.community_reach_types(99) == frozenset()


def test_generate_typed_boosts_empty_answer_type_returns_hop_boosts():
    edges = [("a", "b", "r1"), ("a", "c", "r1")]
    cmap = {"a": 0, "b": 1, "c": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    # answer_type="" should behave same as generate_hop_boosts
    assert hg.generate_typed_boosts("a", answer_type="") == hg.generate_hop_boosts("a")


def test_generate_typed_boosts_person_filters_to_person_rels():
    # Two adjacent communities: C1 has person-suffix rels, C2 has generic rels
    edges = [
        ("a", "b", "people.person.actor"),   # C0 → C1 via person rel
        ("a", "c", "film.film.genre"),        # C0 → C2 via non-person rel
    ]
    cmap = {"a": 0, "b": 1, "c": 2}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    boosts = hg.generate_typed_boosts("a", answer_type="person")
    # Only the person-reaching relation should be boosted
    assert "people.person.actor" in boosts
    assert "film.film.genre" not in boosts


def test_generate_typed_boosts_fallback_when_no_match():
    # No person-reaching community from C0 → should fall back to unfiltered
    edges = [("a", "b", "film.film.genre")]
    cmap = {"a": 0, "b": 1}
    hg = CommunityHypothesisGenerator().build(_make_adapter(edges, cmap))
    boosts_typed = hg.generate_typed_boosts("a", answer_type="person")
    boosts_all = hg.generate_hop_boosts("a")
    assert boosts_typed == boosts_all


def test_generate_typed_boosts_unknown_entity_returns_empty():
    hg = CommunityHypothesisGenerator()
    assert hg.generate_typed_boosts("ghost", answer_type="person") == {}

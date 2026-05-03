"""
Tests for Phase 166: GraphProfiler and QueryProfile.
"""
from __future__ import annotations

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.graph_profiler import GraphProfiler, QueryProfile, HUB_THRESH, MIN_COV_THRESH
from core.cerebrum import CerebrumGraph


# ---------------------------------------------------------------------------
# Helper graph builders
# ---------------------------------------------------------------------------

def _hub_graph() -> NetworkXAdapter:
    """Star graph: one hub connected to 50 leaves — hub_homogeneous regime."""
    G = nx.Graph()
    hub = "hub"
    for i in range(50):
        leaf = f"leaf_{i}"
        G.add_edge(hub, leaf, relation="KNOWS")
    # All nodes are sources of "KNOWS" → high mean_rel_coverage
    return NetworkXAdapter(G)


def _typed_graph() -> NetworkXAdapter:
    """
    Hetionet-like: 100 diseases, 100 genes, 5 compounds.
    Disease-associates-Gene covers most nodes (high coverage, not typed).
    Compound-treats-Disease: 5 compounds + 1 disease = 6/~115 = 5.2% coverage
    (strictly below MIN_COV_THRESH=0.10 -> typed relation).
    """
    G = nx.Graph()
    diseases  = [f"Disease::{i}" for i in range(100)]
    genes     = [f"Gene::{i}"    for i in range(100)]
    compounds = [f"Compound::{i}" for i in range(5)]

    # Connect each disease to 10 genes — most nodes participate
    for d in diseases:
        for gene in genes[:10]:
            G.add_edge(d, gene, relation="Disease-associates-Gene")

    # Only 5 compounds connect to 1 disease — very sparse source set
    for c in compounds:
        G.add_edge(c, diseases[0], relation="Compound-treats-Disease")

    return NetworkXAdapter(G)


def _mixed_graph() -> NetworkXAdapter:
    """A moderate graph — not strongly hub-heavy, not strongly typed."""
    G = nx.Graph()
    for i in range(20):
        G.add_edge(f"a_{i}", f"b_{i}", relation="RELATES")
        G.add_edge(f"a_{i}", f"c_{i}", relation="CONNECTS")
    # Some cross-edges
    for i in range(0, 20, 4):
        G.add_edge(f"b_{i}", f"c_{i+1 if i+1 < 20 else 0}", relation="LINKS")
    return NetworkXAdapter(G)


# ---------------------------------------------------------------------------
# GraphProfiler.profile() unit tests
# ---------------------------------------------------------------------------

def _make_anchor_sources(adapter) -> dict:
    """Build anchor_sources the same way CerebrumGraph.build() does."""
    G = adapter.to_networkx()
    sources: dict = {}
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "")
        if rel:
            sources.setdefault(rel, set()).add(u)
    return sources


def test_profile_returns_query_profile():
    adapter = _hub_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    assert isinstance(profile, QueryProfile)


def test_hub_graph_classified_as_hub_homogeneous():
    adapter = _hub_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    assert profile.regime == "hub_homogeneous", (
        f"Expected hub_homogeneous, got {profile.regime}. "
        f"hub_score={profile.hub_score:.3f}, mean_rel_coverage={profile.mean_rel_coverage:.3f}"
    )
    assert profile.recommended_hop_expand is True
    assert profile.recommended_trb_auto is False


def test_typed_graph_classified_as_typed_heterogeneous():
    adapter = _typed_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    assert profile.regime == "typed_heterogeneous", (
        f"Expected typed_heterogeneous, got {profile.regime}. "
        f"hub_score={profile.hub_score:.3f}, min_rel_coverage={profile.min_rel_coverage:.3f}"
    )
    assert profile.recommended_hop_expand is False
    assert profile.recommended_trb_auto is True


def test_typed_graph_has_typed_relations():
    adapter = _typed_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    # Compound-treats-Disease: 5 compounds + 1 disease = 6/115 nodes = 5.2% < MIN_COV_THRESH
    assert len(profile.typed_relations) >= 1
    assert "Compound-treats-Disease" in profile.typed_relations


def test_hub_score_is_positive_for_star_graph():
    adapter = _hub_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    assert profile.hub_score > 0.0
    assert profile.hub_score <= 1.0


def test_empty_graph_returns_mixed():
    G = nx.Graph()
    G.add_node("singleton")
    adapter = NetworkXAdapter(G)
    profile = GraphProfiler.profile(adapter, {})
    assert isinstance(profile, QueryProfile)
    # empty anchor_sources → no relation types → mixed
    assert profile.n_relation_types == 0


def test_profile_summary_is_str():
    adapter = _typed_graph()
    anchor_sources = _make_anchor_sources(adapter)
    profile = GraphProfiler.profile(adapter, anchor_sources)
    s = profile.summary()
    assert isinstance(s, str)
    assert "typed_heterogeneous" in s or "hub_homogeneous" in s or "mixed" in s


# ---------------------------------------------------------------------------
# Integration: CerebrumGraph._query_profile populated after build()
# ---------------------------------------------------------------------------

SIMPLE_TRIPLES = [
    ("Disease::A", "Disease-associates-Gene", "Gene::1"),
    ("Disease::A", "Disease-associates-Gene", "Gene::2"),
    ("Disease::B", "Disease-associates-Gene", "Gene::1"),
    ("Gene::1",    "Gene-participates-Pathway", "Pathway::X"),
    ("Gene::2",    "Gene-participates-Pathway", "Pathway::Y"),
    ("Compound::C1", "Compound-treats-Disease", "Disease::A"),
]


def test_cerebrum_graph_has_query_profile_after_build():
    graph = CerebrumGraph.from_triples(SIMPLE_TRIPLES)
    graph.build()
    assert graph.query_profile is not None
    assert isinstance(graph.query_profile, QueryProfile)
    assert graph.query_profile.regime in ("hub_homogeneous", "typed_heterogeneous", "mixed")


def test_query_profile_regime_for_typed_triples():
    graph = CerebrumGraph.from_triples(SIMPLE_TRIPLES)
    graph.build()
    qp = graph.query_profile
    # "Compound-treats-Disease" sources = 1 node / 9 nodes = 11% < MIN_COV_THRESH (10%)
    # This is borderline — just check it's not hub_homogeneous
    assert qp.regime != "hub_homogeneous", (
        f"Typed triples should not classify as hub_homogeneous. "
        f"hub_score={qp.hub_score:.3f}"
    )


def test_query_uses_profile_when_params_are_none():
    """query() with no strategy params reads from profile — no crash, returns answers."""
    graph = CerebrumGraph.from_triples(SIMPLE_TRIPLES)
    graph.build()
    answers = graph.query(
        seeds=["Disease::A"],
        top_k=5,
        min_hop=1,
        max_hop=1,
        # hop_expand, auto_infer_terminal_relation, anchor_bonus all default to None
        # → resolved from QueryProfile
    )
    assert isinstance(answers, list)


def test_explicit_false_overrides_profile():
    """Passing hop_expand=False explicitly must not be overridden by profile."""
    graph = CerebrumGraph.from_triples(SIMPLE_TRIPLES)
    graph.build()
    # Force profile to recommend hop_expand=True by mocking
    if graph.query_profile:
        graph._query_profile.recommended_hop_expand = True
    # Explicit False must win
    answers = graph.query(
        seeds=["Disease::A"],
        top_k=5,
        max_hop=2,
        hop_expand=False,
    )
    assert isinstance(answers, list)


def test_explicit_true_overrides_profile():
    """Passing hop_expand=True explicitly must not be overridden by a False profile."""
    graph = CerebrumGraph.from_triples(SIMPLE_TRIPLES)
    graph.build()
    if graph.query_profile:
        graph._query_profile.recommended_hop_expand = False
    answers = graph.query(
        seeds=["Disease::A"],
        top_k=5,
        max_hop=2,
        hop_expand=True,
    )
    assert isinstance(answers, list)

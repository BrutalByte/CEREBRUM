"""
Tests for Phase 19 — TransitiveInferenceEngine.
"""
import time
import pytest
import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.inference_engine import (
    TransitiveInferenceEngine,
    InferenceRule,
    InferenceProposal,
    InferenceReport,
    INFERENCE_RULES,
    _build_rule_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter(edges):
    """Build a NetworkXAdapter from a list of (src, dst, relation, confidence) tuples."""
    G = nx.DiGraph()
    for src, dst, rel, conf in edges:
        G.add_edge(src, dst, relation_type=rel, relation=rel, confidence=conf, weight=conf)
    return NetworkXAdapter(G)


@pytest.fixture
def causal_adapter():
    """aspirin --INHIBITS--> COX2 --PROMOTES--> inflammation"""
    return _make_adapter([
        ("aspirin",     "COX2",         "INHIBITS", 0.9),
        ("COX2",        "inflammation", "PROMOTES", 0.8),
    ])


@pytest.fixture
def academic_adapter():
    """newton --INFLUENCED--> leibniz --INFLUENCED--> euler"""
    return _make_adapter([
        ("newton",  "leibniz", "INFLUENCED", 1.0),
        ("leibniz", "euler",   "INFLUENCED", 1.0),
    ])


@pytest.fixture
def lineage_adapter():
    """gauss --STUDENT_OF--> euler; gauss --STUDENT_OF--> lagrange"""
    return _make_adapter([
        ("riemann",   "gauss",   "STUDENT_OF", 1.0),
        ("gauss",     "euler",   "STUDENT_OF", 1.0),
    ])


@pytest.fixture
def chain_adapter():
    """A multi-hop chain: A -CAUSES-> B -CAUSES-> C -CAUSES-> D"""
    return _make_adapter([
        ("A", "B", "CAUSES", 0.9),
        ("B", "C", "CAUSES", 0.8),
        ("C", "D", "CAUSES", 0.7),
    ])


@pytest.fixture
def clean_adapter():
    """Graph with no composable relations."""
    return _make_adapter([
        ("cat", "dog",  "KNOWS", 0.9),
        ("fish","bird", "KNOWS", 0.9),
    ])


@pytest.fixture
def pre_existing_adapter():
    """Graph where the derived edge already exists."""
    return _make_adapter([
        ("A", "B",  "CAUSES", 0.9),
        ("B", "C",  "CAUSES", 0.8),
        ("A", "C",  "INDIRECTLY_CAUSES", 0.5),   # already present
    ])


# ---------------------------------------------------------------------------
# InferenceRule table
# ---------------------------------------------------------------------------

def test_rule_table_is_nonempty():
    assert len(INFERENCE_RULES) >= 50


def test_rule_table_has_all_domains():
    domains = {r.domain for r in INFERENCE_RULES}
    assert "causal" in domains
    assert "biology" in domains
    assert "academic" in domains
    assert "social" in domains
    assert "general" in domains
    assert "film" in domains


def test_rule_factors_in_range():
    for r in INFERENCE_RULES:
        assert 0.0 < r.factor <= 1.0, f"Rule {r} has out-of-range factor"


def test_rule_index_built_correctly():
    idx = _build_rule_index(INFERENCE_RULES)
    assert ("CAUSES", "CAUSES") in idx
    assert idx[("CAUSES", "CAUSES")].derived == "INDIRECTLY_CAUSES"
    assert ("INHIBITS", "PROMOTES") in idx


# ---------------------------------------------------------------------------
# Basic composition
# ---------------------------------------------------------------------------

def test_causal_chain_discovered(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "INDIRECTLY_REDUCES" in rels


def test_academic_chain_discovered(academic_adapter):
    eng = TransitiveInferenceEngine(academic_adapter)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "INDIRECTLY_INFLUENCED" in rels


def test_lineage_chain_discovered(lineage_adapter):
    eng = TransitiveInferenceEngine(lineage_adapter)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "ACADEMIC_DESCENDANT_OF" in rels


def test_proposal_source_and_target(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    reduced = [p for p in report.proposals if p.derived_relation == "INDIRECTLY_REDUCES"]
    assert len(reduced) == 1
    assert reduced[0].source == "aspirin"
    assert reduced[0].target == "inflammation"
    assert reduced[0].via == "COX2"


def test_proposal_intermediate_node_correct(academic_adapter):
    eng = TransitiveInferenceEngine(academic_adapter)
    report = eng.run(dry_run=True)
    influenced = [p for p in report.proposals if p.derived_relation == "INDIRECTLY_INFLUENCED"]
    assert len(influenced) == 1
    assert influenced[0].via == "leibniz"


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------

def test_confidence_is_product_of_factors(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    p = next(p for p in report.proposals if p.derived_relation == "INDIRECTLY_REDUCES")
    # INHIBITS+PROMOTES factor = 0.70; conf_a=0.9, conf_b=0.8
    expected = 0.9 * 0.8 * 0.70
    assert abs(p.confidence - expected) < 1e-5


def test_confidence_sorted_descending(chain_adapter):
    eng = TransitiveInferenceEngine(chain_adapter)
    report = eng.run(dry_run=True)
    confs = [p.confidence for p in report.proposals]
    assert confs == sorted(confs, reverse=True)


def test_min_confidence_filter():
    adapter = _make_adapter([
        ("A", "B", "INHIBITS", 0.1),
        ("B", "C", "PROMOTES", 0.1),
    ])
    # conf = 0.1 * 0.1 * 0.70 = 0.007 < default min 0.10
    eng = TransitiveInferenceEngine(adapter, min_confidence=0.10)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "INDIRECTLY_REDUCES" not in rels


def test_min_confidence_zero_accepts_all():
    adapter = _make_adapter([
        ("A", "B", "INHIBITS", 0.1),
        ("B", "C", "PROMOTES", 0.1),
    ])
    eng = TransitiveInferenceEngine(adapter, min_confidence=0.0)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "INDIRECTLY_REDUCES" in rels


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

def test_report_is_inference_report(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    assert isinstance(report, InferenceReport)


def test_report_proposal_count(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    assert report.proposal_count == len(report.proposals)


def test_report_dry_run_flag(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    assert eng.run(dry_run=True).dry_run is True
    assert eng.run(dry_run=False).dry_run is False


def test_report_materialized_zero_on_dry_run(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    assert report.materialized == 0


def test_report_rules_applied_dict(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    assert isinstance(report.rules_applied, dict)
    assert all(isinstance(v, int) for v in report.rules_applied.values())


def test_report_duration_positive(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    assert report.duration_seconds >= 0.0


def test_report_summary_string(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    s = report.summary()
    assert "dry-run" in s
    assert "Proposals" in s


# ---------------------------------------------------------------------------
# Skip existing edges
# ---------------------------------------------------------------------------

def test_skips_existing_relation(pre_existing_adapter):
    eng = TransitiveInferenceEngine(pre_existing_adapter)
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "INDIRECTLY_CAUSES" not in rels
    assert report.skipped_existing >= 1


def test_clean_graph_no_proposals(clean_adapter):
    eng = TransitiveInferenceEngine(clean_adapter)
    report = eng.run(dry_run=True)
    # KNOWS+KNOWS → CONNECTED_VIA is a valid rule, but both edges share same
    # domain and no composable pairs at the same intermediate node exist here
    # (cat-KNOWS-dog is separate from fish-KNOWS-bird, no shared node)
    assert report.proposal_count == 0


# ---------------------------------------------------------------------------
# max_proposals cap
# ---------------------------------------------------------------------------

def test_max_proposals_respected():
    # Build a graph with many composable chains
    G = nx.DiGraph()
    hub = "hub"
    for i in range(20):
        G.add_edge(f"src_{i}", hub, relation_type="INHIBITS", confidence=0.9)
        G.add_edge(hub, f"dst_{i}", relation_type="PROMOTES", confidence=0.9)
    adapter = NetworkXAdapter(G)
    eng = TransitiveInferenceEngine(adapter, max_proposals=5)
    report = eng.run(dry_run=True)
    assert report.proposal_count <= 5


def test_max_proposals_takes_highest_confidence():
    G = nx.DiGraph()
    hub = "hub"
    for i in range(10):
        conf = 0.1 + i * 0.08   # 0.1, 0.18, 0.26, ... 0.82
        G.add_edge(f"src_{i}", hub, relation_type="CAUSES", confidence=conf)
        G.add_edge(hub, f"dst_{i}", relation_type="CAUSES", confidence=0.9)
    adapter = NetworkXAdapter(G)
    eng = TransitiveInferenceEngine(adapter, max_proposals=3)
    report = eng.run(dry_run=True)
    # Confidences should be highest 3
    confs = [p.confidence for p in report.proposals]
    assert confs == sorted(confs, reverse=True)
    assert len(confs) == 3


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------

def test_domain_filter_biology_only(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter, enabled_domains={"biology"})
    report = eng.run(dry_run=True)
    for p in report.proposals:
        assert p.rule.domain in {"biology", "general"}


def test_domain_filter_excludes_film():
    adapter = _make_adapter([
        ("actor", "film1", "STARRED_IN", 0.9),
        ("film1", "director", "DIRECTED", 0.9),
    ])
    eng = TransitiveInferenceEngine(adapter, enabled_domains={"biology"})
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "WORKED_UNDER" not in rels


def test_domain_filter_film_enabled():
    adapter = _make_adapter([
        ("actor", "film1", "STARRED_IN", 0.9),
        ("film1", "director", "DIRECTED", 0.9),
    ])
    eng = TransitiveInferenceEngine(adapter, enabled_domains={"film"})
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "WORKED_UNDER" in rels


# ---------------------------------------------------------------------------
# Custom rules
# ---------------------------------------------------------------------------

def test_custom_rule_applied():
    custom = InferenceRule(
        rel_a="ZAPS", rel_b="ZAPS", derived="DOUBLE_ZAPS",
        factor=0.9, domain="general", note="Double zap"
    )
    adapter = _make_adapter([
        ("A", "B", "ZAPS", 0.9),
        ("B", "C", "ZAPS", 0.9),
    ])
    eng = TransitiveInferenceEngine(adapter, custom_rules=[custom])
    report = eng.run(dry_run=True)
    rels = {p.derived_relation for p in report.proposals}
    assert "DOUBLE_ZAPS" in rels


def test_custom_rule_confidence():
    custom = InferenceRule(
        rel_a="ZAPS", rel_b="ZAPS", derived="DOUBLE_ZAPS",
        factor=0.5, domain="general", note="Double zap"
    )
    adapter = _make_adapter([
        ("A", "B", "ZAPS", 1.0),
        ("B", "C", "ZAPS", 1.0),
    ])
    eng = TransitiveInferenceEngine(adapter, custom_rules=[custom])
    report = eng.run(dry_run=True)
    p = next(p for p in report.proposals if p.derived_relation == "DOUBLE_ZAPS")
    assert abs(p.confidence - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Derivation string
# ---------------------------------------------------------------------------

def test_derivation_str_format(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=True)
    p = next(p for p in report.proposals if p.derived_relation == "INDIRECTLY_REDUCES")
    s = p.derivation_str
    assert "aspirin" in s
    assert "INHIBITS" in s
    assert "COX2" in s
    assert "PROMOTES" in s
    assert "inflammation" in s
    assert "INDIRECTLY_REDUCES" in s


# ---------------------------------------------------------------------------
# Materialization and rollback
# ---------------------------------------------------------------------------

def test_dry_run_does_not_mutate(causal_adapter):
    G_before = causal_adapter.to_networkx()
    n_edges_before = G_before.number_of_edges()
    eng = TransitiveInferenceEngine(causal_adapter)
    eng.run(dry_run=True)
    G_after = causal_adapter.to_networkx()
    assert G_after.number_of_edges() == n_edges_before


def test_real_run_adds_edges(causal_adapter):
    G_before = causal_adapter.to_networkx()
    n_before = G_before.number_of_edges()
    eng = TransitiveInferenceEngine(causal_adapter)
    report = eng.run(dry_run=False)
    G_after = causal_adapter.to_networkx()
    assert G_after.number_of_edges() > n_before
    assert report.materialized > 0


def test_materialized_edge_has_provenance(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    eng.run(dry_run=False)
    G = causal_adapter.to_networkx()
    found = False
    for u, v, data in G.edges(data=True):
        if data.get("provenance") == "transitive_inference":
            found = True
            assert data.get("via") is not None
            assert data.get("derivation") is not None
            assert 0.0 < data.get("confidence", 0) <= 1.0
            break
    assert found, "No inferred edge found with provenance=transitive_inference"


def test_rollback_removes_inferred_edges(causal_adapter):
    G = causal_adapter.to_networkx()
    n_before = G.number_of_edges()
    eng = TransitiveInferenceEngine(causal_adapter)
    eng.run(dry_run=False)
    assert causal_adapter.to_networkx().number_of_edges() > n_before
    removed = eng.rollback()
    assert removed > 0
    assert causal_adapter.to_networkx().number_of_edges() == n_before


def test_rollback_raises_without_prior_run():
    adapter = _make_adapter([("A", "B", "CAUSES", 1.0)])
    eng = TransitiveInferenceEngine(adapter)
    with pytest.raises(RuntimeError):
        eng.rollback()


def test_rollback_raises_after_dry_run(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    eng.run(dry_run=True)
    with pytest.raises(RuntimeError):
        eng.rollback()


def test_can_rollback_property(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    assert eng.can_rollback is False
    eng.run(dry_run=False)
    assert eng.can_rollback is True
    eng.rollback()
    assert eng.can_rollback is False


# ---------------------------------------------------------------------------
# last_report and rule_count
# ---------------------------------------------------------------------------

def test_last_report_none_before_run(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    assert eng.last_report is None


def test_last_report_populated_after_run(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    eng.run(dry_run=True)
    assert eng.last_report is not None
    assert isinstance(eng.last_report, InferenceReport)


def test_rule_count_matches_active_rules(causal_adapter):
    eng = TransitiveInferenceEngine(causal_adapter)
    assert eng.rule_count() == len(INFERENCE_RULES)


def test_rule_count_reduced_with_domain_filter(causal_adapter):
    eng_all  = TransitiveInferenceEngine(causal_adapter)
    eng_bio  = TransitiveInferenceEngine(causal_adapter, enabled_domains={"biology"})
    assert eng_bio.rule_count() < eng_all.rule_count()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_graph():
    G = nx.DiGraph()
    adapter = NetworkXAdapter(G)
    eng = TransitiveInferenceEngine(adapter)
    report = eng.run(dry_run=True)
    assert report.proposal_count == 0


def test_single_node_graph():
    G = nx.DiGraph()
    G.add_node("lonely")
    adapter = NetworkXAdapter(G)
    eng = TransitiveInferenceEngine(adapter)
    report = eng.run(dry_run=True)
    assert report.proposal_count == 0


def test_no_composable_rules_in_graph():
    adapter = _make_adapter([
        ("A", "B", "ZZZUNKNOWN", 0.9),
        ("B", "C", "ZZZUNKNOWN", 0.9),
    ])
    eng = TransitiveInferenceEngine(adapter)
    report = eng.run(dry_run=True)
    assert report.proposal_count == 0


def test_self_loop_not_proposed():
    """A --CAUSES--> A --CAUSES--> B should NOT propose A --INDIRECTLY_CAUSES--> A."""
    G = nx.DiGraph()
    G.add_edge("A", "A", relation_type="CAUSES", confidence=0.9)
    G.add_edge("A", "B", relation_type="CAUSES", confidence=0.9)
    adapter = NetworkXAdapter(G)
    eng = TransitiveInferenceEngine(adapter)
    report = eng.run(dry_run=True)
    for p in report.proposals:
        assert p.source != p.target, "Self-loop proposed"

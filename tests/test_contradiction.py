"""
Tests for the CEREBRUM contradiction handling system.

Contradictions are research signals â€” they are surfaced, never suppressed.
A lone contradicting path may be the actual correct answer.

Coverage:
  - CONTRADICTION_PAIRS lookup (bidirectional, case-insensitive)
  - is_valid_at() temporal edge filtering
  - ContradictionRecord structure and auto-computed authority_delta
  - ContradictionEngine.detect_direct() â€” Type 1
  - ContradictionEngine.detect_temporal() â€” Type 3
  - ContradictionEngine.materialize() â€” CONTRADICTS edge creation
  - ContradictionEngine.scan() â€” full index-time pipeline
  - TraversalPath edge_confidences / path_confidence
  - path_scorer.path_confidence()
  - Answer.contradiction_flags populated by extract()
"""
from typing import Type
import random

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from core.community_engine import dscf_communities
from core.contradiction_engine import (
    CONTRADICTS_RELATION,
    ContradictionEngine,
    ContradictionRecord,
    is_valid_at,
    relations_contradict,
)
from core.embedding_engine import RandomEngine
from core.graph_adapter import Edge
from core.structural_encoder import adjacent_community_pairs, build_community_distance_matrix
from reasoning.answer_extractor import (
    Answer,
    ContradictionFlag,
    detect_answer_contradictions,
    extract,
)
from reasoning.path_scorer import path_confidence
from reasoning.traversal import BeamTraversal, TraversalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_directed_contradicting_graph() -> nx.DiGraph:
    """
    A â†’ B via ACTIVATES
    B â†’ A via INHIBITS   (contradicts ACTIVATES)
    A â†’ C via KNOWS      (no contradiction)
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9, provenance="pubmed:1")
    G.add_edge("B", "A", relation="INHIBITS",  confidence=0.7, provenance="pubmed:2")
    G.add_edge("A", "C", relation="KNOWS",     confidence=1.0, provenance="wikidata:Q1")
    return G


def make_temporal_graph() -> nx.DiGraph:
    """
    A â†’ B via ACTIVATES  (valid 2000â€“2010)
    A â†’ D via INHIBITS   (valid 2015â€“2025)   â†� contradicts ACTIVATES but no overlap
    """
    G = nx.DiGraph()
    t2000 = float(946684800)   # 2000-01-01
    t2010 = float(1262304000)  # 2010-01-01
    t2015 = float(1420070400)  # 2015-01-01
    t2025 = float(1735689600)  # 2025-01-01
    G.add_edge("A", "B", relation="ACTIVATES", valid_from=t2000, valid_to=t2010, confidence=0.9)
    G.add_edge("A", "D", relation="INHIBITS",  valid_from=t2015, valid_to=t2025, confidence=0.8)
    return G


def build_simple_traversal(G: nx.Graph) -> BeamTraversal:
    """Build a BeamTraversal over G with RandomEngine embeddings."""
    adapter = NetworkXAdapter(G)
    engine  = RandomEngine(dim=16)
    labels  = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    random.seed(0)
    parts = dscf_communities(G, max_iter=30)
    community_map = {node: cid for cid, members in enumerate(parts) for node in members}

    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)

    return BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5, max_hop=2)


# ---------------------------------------------------------------------------
# CONTRADICTION_PAIRS lookup
# ---------------------------------------------------------------------------

def test_relations_contradict_known_pairs():
    assert relations_contradict("ACTIVATES", "INHIBITS")
    assert relations_contradict("CAUSES", "PREVENTS")
    assert relations_contradict("UPREGULATES", "DOWNREGULATES")
    assert relations_contradict("SUPPORTS", "REFUTES")
    assert relations_contradict("PROVES", "DISPROVES")


def test_relations_contradict_non_contradicting():
    assert not relations_contradict("KNOWS", "WORKS_WITH")
    assert not relations_contradict("ACTIVATES", "ACTIVATES")
    assert not relations_contradict("RELATED_TO", "CAUSED_BY")


def test_relations_contradict_bidirectional():
    # If A contradicts B, B must contradict A
    assert relations_contradict("INHIBITS", "ACTIVATES")
    assert relations_contradict("PREVENTS", "CAUSES")
    assert relations_contradict("DOWNREGULATES", "UPREGULATES")


def test_relations_contradict_case_insensitive():
    assert relations_contradict("activates", "inhibits")
    assert relations_contradict("Activates", "Inhibits")
    assert relations_contradict("ACTIVATES", "inhibits")


# ---------------------------------------------------------------------------
# is_valid_at() â€” temporal filtering
# ---------------------------------------------------------------------------

class _MockEdge:
    """Minimal edge-like object for is_valid_at tests."""
    def __init__(self, valid_from=None, valid_to=None):
        self.valid_from = valid_from
        self.valid_to   = valid_to


def test_is_valid_at_no_query_time():
    """None query_time disables filtering â€” always valid."""
    e = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(e, None)


def test_is_valid_at_no_bounds():
    """Edge with no bounds is always valid."""
    e = _MockEdge()
    assert is_valid_at(e, 9999999.0)


def test_is_valid_at_within_window():
    e = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(e, 1500.0)


def test_is_valid_at_at_boundary():
    e = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert is_valid_at(e, 1000.0)
    assert is_valid_at(e, 2000.0)


def test_is_valid_at_before_window():
    e = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert not is_valid_at(e, 500.0)


def test_is_valid_at_after_window():
    e = _MockEdge(valid_from=1000.0, valid_to=2000.0)
    assert not is_valid_at(e, 3000.0)


def test_is_valid_at_open_ended():
    """No valid_to means still valid."""
    e = _MockEdge(valid_from=1000.0, valid_to=None)
    assert is_valid_at(e, 9999999.0)
    assert not is_valid_at(e, 500.0)


# ---------------------------------------------------------------------------
# ContradictionRecord
# ---------------------------------------------------------------------------

def test_contradiction_record_authority_delta():
    rec = ContradictionRecord(
        node_a="A", node_b="B",
        relation_a="ACTIVATES", relation_b="INHIBITS",
        contradiction_type="direct",
        confidence_a=0.9, confidence_b=0.7,
    )
    assert abs(rec.authority_delta - 0.2) < 1e-6


def test_contradiction_record_defaults():
    rec = ContradictionRecord(
        node_a="X", node_b="Y",
        relation_a="CAUSES", relation_b="PREVENTS",
        contradiction_type="direct",
    )
    assert rec.resolution_status == "unresolved"
    assert not rec.human_reviewed
    assert rec.provenance_a == ""
    assert rec.note == ""


# ---------------------------------------------------------------------------
# ContradictionEngine â€” detect_direct()
# ---------------------------------------------------------------------------

def test_detect_direct_finds_contradiction():
    G = make_directed_contradicting_graph()
    engine = ContradictionEngine()
    records = engine.detect_direct(G)
    assert len(records) == 1
    rec = records[0]
    assert rec.contradiction_type == "direct"
    assert relations_contradict(rec.relation_a, rec.relation_b)


def test_detect_direct_clean_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="KNOWS")
    G.add_edge("B", "C", relation="WORKS_WITH")
    engine = ContradictionEngine()
    records = engine.detect_direct(G)
    assert records == []


def test_detect_direct_preserves_provenance():
    G = make_directed_contradicting_graph()
    engine = ContradictionEngine()
    records = engine.detect_direct(G)
    rec = records[0]
    # One of the provenances should come from the graph edge data
    assert rec.provenance_a != "" or rec.provenance_b != ""


# ---------------------------------------------------------------------------
# ContradictionEngine â€” detect_temporal()
# ---------------------------------------------------------------------------

def test_detect_temporal_finds_non_overlapping():
    G = make_temporal_graph()
    engine = ContradictionEngine()
    records = engine.detect_temporal(G)
    assert len(records) >= 1
    # All temporal records should be resolution_status="temporal"
    temporal_recs = [r for r in records if r.resolution_status == "temporal"]
    assert len(temporal_recs) >= 1


def test_detect_temporal_clean_graph():
    """Graph with no temporal bounds or contradictions â†’ no records."""
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="KNOWS")
    engine = ContradictionEngine()
    records = engine.detect_temporal(G)
    assert records == []


# ---------------------------------------------------------------------------
# ContradictionEngine â€” materialize()
# ---------------------------------------------------------------------------

def test_materialize_adds_contradicts_edge():
    # Use a graph where one direction of (X, Y) is free so CONTRADICTS can be added.
    # Xâ†’Y ACTIVATES exists; Yâ†’X is free â€” materialize adds CONTRADICTS as Yâ†’X.
    G = nx.DiGraph()
    G.add_edge("X", "Y", relation="ACTIVATES", confidence=0.9, provenance="pubmed:1")
    records = [
        ContradictionRecord(
            node_a="X", node_b="Y",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    n_before = G.number_of_edges()
    added = engine.materialize(G, records)
    assert added == 1
    assert G.number_of_edges() == n_before + 1
    all_rels = {data.get("relation") for _, _, data in G.edges(data=True)}
    assert CONTRADICTS_RELATION in all_rels


def test_materialize_contradicts_edge_weight_low():
    G = make_directed_contradicting_graph()
    engine = ContradictionEngine()
    records = engine.detect_direct(G)
    engine.materialize(G, records)

    for u, v, data in G.edges(data=True):
        if data.get("relation") == CONTRADICTS_RELATION:
            assert data["weight"] == 0.1


def test_materialize_contradicts_edge_metadata():
    G = nx.DiGraph()
    G.add_edge("X", "Y", relation="ACTIVATES", confidence=0.9)
    records = [
        ContradictionRecord(
            node_a="X", node_b="Y",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    engine.materialize(G, records)

    found = False
    for u, v, data in G.edges(data=True):
        if data.get("relation") == CONTRADICTS_RELATION:
            assert "contradiction_type" in data
            assert "relation_a" in data
            assert "relation_b" in data
            assert "resolution_status" in data
            found = True
    assert found


def test_materialize_no_duplicates():
    G = nx.DiGraph()
    G.add_edge("X", "Y", relation="ACTIVATES", confidence=0.9)
    records = [
        ContradictionRecord(
            node_a="X", node_b="Y",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    engine.materialize(G, records)
    n_edges = G.number_of_edges()
    # Second call should not add another CONTRADICTS edge
    added2 = engine.materialize(G, records)
    assert added2 == 0
    assert G.number_of_edges() == n_edges


# ---------------------------------------------------------------------------
# ContradictionEngine â€” scan()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ContradictionEngine â€” detect_provenance() (Type 4, multigraph only)
# ---------------------------------------------------------------------------

def test_detect_provenance_finds_conflict():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9, provenance="pubmed:1")
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.5, provenance="arxiv:2")
    engine = ContradictionEngine()
    records = engine.detect_provenance(G)
    assert len(records) == 1
    rec = records[0]
    assert rec.contradiction_type == "provenance"
    assert rec.resolution_status == "source_bias"
    assert abs(rec.confidence_a - rec.confidence_b) > 0.1


def test_detect_provenance_skips_same_source():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9, provenance="pubmed:1")
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.95, provenance="pubmed:1")
    engine = ContradictionEngine()
    records = engine.detect_provenance(G)
    # Same provenance: not a source conflict
    assert records == []


def test_detect_provenance_skips_close_confidence():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9, provenance="pubmed:1")
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.85, provenance="arxiv:2")
    engine = ContradictionEngine()
    records = engine.detect_provenance(G)
    # Spread = 0.05 <= threshold 0.1 â€” not flagged
    assert records == []


def test_detect_provenance_returns_empty_for_simple_graph():
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="ACTIVATES")
    engine = ContradictionEngine()
    records = engine.detect_provenance(G)
    assert records == []


# ---------------------------------------------------------------------------
# ContradictionEngine â€” detect_cross_path() (Type 2, query time)
# ---------------------------------------------------------------------------

def test_detect_cross_path_finds_contradiction():
    path_a = _make_path_with_relation("S", "ACTIVATES", "T")
    path_b = _make_path_with_relation("S", "INHIBITS",  "T")
    engine = ContradictionEngine()
    records = engine.detect_cross_path([path_a], [path_b])
    assert len(records) == 1
    rec = records[0]
    assert rec.contradiction_type == "cross_path"
    assert relations_contradict(rec.relation_a, rec.relation_b)


def test_detect_cross_path_requires_same_terminal():
    path_a = _make_path_with_relation("S", "ACTIVATES", "T1")
    path_b = _make_path_with_relation("S", "INHIBITS",  "T2")
    engine = ContradictionEngine()
    records = engine.detect_cross_path([path_a], [path_b])
    # Different terminals â€” no cross-path contradiction
    assert records == []


def test_detect_cross_path_no_contradiction_on_compatible_relations():
    path_a = _make_path_with_relation("S", "KNOWS",     "T")
    path_b = _make_path_with_relation("S", "WORKS_WITH","T")
    engine = ContradictionEngine()
    records = engine.detect_cross_path([path_a], [path_b])
    assert records == []


# ---------------------------------------------------------------------------
# ContradictionEngine â€” materialize() edge cases
# ---------------------------------------------------------------------------

def test_materialize_multigraph_adds_edge():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9)
    G.add_edge("B", "A", relation="INHIBITS",  confidence=0.7)
    records = [
        ContradictionRecord(
            node_a="A", node_b="B",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    n_before = G.number_of_edges()
    added = engine.materialize(G, records)
    assert added == 1
    assert G.number_of_edges() == n_before + 1


def test_materialize_skips_when_both_directions_occupied():
    # DiGraph with both Aâ†’B and Bâ†’A already filled by non-CONTRADICTS edges.
    # materialize() cannot add without overwriting, so it skips.
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9)
    G.add_edge("B", "A", relation="INHIBITS",  confidence=0.7)
    records = [
        ContradictionRecord(
            node_a="A", node_b="B",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    n_before = G.number_of_edges()
    added = engine.materialize(G, records)
    assert added == 0
    assert G.number_of_edges() == n_before  # no edges added or overwritten


def test_materialize_uses_reverse_when_forward_occupied():
    # Aâ†’B ACTIVATES exists; Bâ†’A is free â€” CONTRADICTS goes Bâ†’A.
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="ACTIVATES", confidence=0.9)
    records = [
        ContradictionRecord(
            node_a="A", node_b="B",
            relation_a="ACTIVATES", relation_b="INHIBITS",
            contradiction_type="direct",
        )
    ]
    engine = ContradictionEngine()
    added = engine.materialize(G, records)
    assert added == 1
    # CONTRADICTS should be on the reverse edge Bâ†’A
    assert G.has_edge("B", "A")
    data = G.get_edge_data("B", "A")
    assert data["relation"] == CONTRADICTS_RELATION


def test_scan_detects_all_types():
    G = make_directed_contradicting_graph()
    engine = ContradictionEngine()
    records = engine.scan(G)
    assert len(records) >= 1  # at least the Type 1 direct contradiction


def test_scan_clean_graph_empty():
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="KNOWS")
    G.add_edge("B", "C", relation="RELATED_TO")
    engine = ContradictionEngine()
    assert engine.scan(G) == []


# ---------------------------------------------------------------------------
# Edge dataclass â€” new fields
# ---------------------------------------------------------------------------

def test_edge_confidence_default():
    e = Edge(source_id="A", target_id="B", relation_type="KNOWS")
    assert e.confidence == 1.0
    assert e.provenance == ""
    assert e.valid_from is None
    assert e.valid_to is None


def test_edge_confidence_set():
    e = Edge(
        source_id="A", target_id="B", relation_type="ACTIVATES",
        confidence=0.75, provenance="pubmed:42",
        valid_from=1000.0, valid_to=2000.0,
    )
    assert e.confidence == 0.75
    assert e.provenance == "pubmed:42"
    assert e.valid_from == 1000.0
    assert e.valid_to == 2000.0


# ---------------------------------------------------------------------------
# TraversalPath â€” edge_confidences / path_confidence
# ---------------------------------------------------------------------------

def test_traversal_path_confidence_default():
    path = TraversalPath(nodes=["A"], seen_entities={"A"}, embedding=np.zeros(4))
    assert path.path_confidence == 1.0


def test_traversal_path_confidence_propagation():
    path = TraversalPath(
        nodes=["A", "ACTIVATES", "B"],
        seen_entities={"A", "B"},
        embedding=np.zeros(4),
        edge_confidences=[0.9],
        edge_provenances=["pubmed:1"],
    )
    assert path.path_confidence == 0.9


def test_traversal_path_confidence_weakest_link():
    path = TraversalPath(
        nodes=["A", "CAUSES", "B", "INCREASES", "C"],
        seen_entities={"A", "B", "C"},
        embedding=np.zeros(4),
        edge_confidences=[0.9, 0.4],
        edge_provenances=["src:1", "src:2"],
    )
    assert path.path_confidence == 0.4


def test_path_scorer_path_confidence_function():
    path = TraversalPath(
        nodes=["A", "R", "B"],
        seen_entities={"A", "B"},
        embedding=np.zeros(4),
        edge_confidences=[0.6, 0.8],
    )
    assert path_confidence(path) == 0.6


def test_path_scorer_path_confidence_no_data():
    path = TraversalPath(nodes=["A"], seen_entities={"A"}, embedding=np.zeros(4))
    assert path_confidence(path) == 1.0


# ---------------------------------------------------------------------------
# Confidence flows through BeamTraversal
# ---------------------------------------------------------------------------

def test_beam_traversal_propagates_confidence():
    """
    NetworkXAdapter edges created from a graph without explicit confidence
    default to Edge.confidence=1.0. Path.edge_confidences should be populated.
    """
    G = nx.Graph()
    G.add_edge("A", "B", relation="KNOWS")
    G.add_edge("B", "C", relation="KNOWS")
    traversal = build_simple_traversal(G)
    paths = traversal.traverse(["A"])
    multi_hop = [p for p in paths if p.hop_depth >= 1]
    assert all(len(p.edge_confidences) == p.hop_depth for p in multi_hop)
    assert all(p.path_confidence == 1.0 for p in multi_hop)


# ---------------------------------------------------------------------------
# Answer contradiction flags
# ---------------------------------------------------------------------------

def _make_path_with_relation(seed: str, rel: str, target: str) -> TraversalPath:
    """Build a minimal TraversalPath with a single hop."""
    emb = np.ones(4, dtype=np.float32) / 2.0
    return TraversalPath(
        nodes=[seed, rel, target],
        seen_entities={seed, target},
        embedding=emb,
        score=0.8,
        attention_weights=[0.8],
        community_sequence=[0, 0],
        edge_confidences=[0.9],
        edge_provenances=["test"],
    )


def _make_answer(entity: str, rel: str, seed: str = "S", score: float = 0.8) -> Answer:
    path = _make_path_with_relation(seed, rel, entity)
    return Answer(
        entity_id=entity,
        score=score,
        best_path=path,
        score_breakdown={},
        community_trace=[0, 0],
    )


def test_detect_answer_contradictions_found():
    ans_a = _make_answer("X", "ACTIVATES")
    ans_b = _make_answer("Y", "INHIBITS")
    detect_answer_contradictions([ans_a, ans_b])
    assert len(ans_a.contradiction_flags) >= 1
    assert len(ans_b.contradiction_flags) >= 1


def test_detect_answer_contradictions_flag_structure():
    ans_a = _make_answer("X", "ACTIVATES", score=0.9)
    ans_b = _make_answer("Y", "INHIBITS",  score=0.7)
    detect_answer_contradictions([ans_a, ans_b])
    flag = ans_a.contradiction_flags[0]
    assert isinstance(flag, ContradictionFlag)
    assert flag.contradiction_type == "cross_path"
    assert flag.conflicting_entity == "Y"
    assert flag.this_confidence == 0.9
    assert flag.conflicting_confidence == 0.9
    ra, rb = flag.contradicting_relations
    assert relations_contradict(ra, rb)


def test_detect_answer_contradictions_symmetric():
    """Both answers should be flagged when a contradiction is found."""
    ans_a = _make_answer("X", "CAUSES")
    ans_b = _make_answer("Y", "PREVENTS")
    detect_answer_contradictions([ans_a, ans_b])
    assert ans_a.contradiction_flags
    assert ans_b.contradiction_flags
    assert ans_b.contradiction_flags[0].conflicting_entity == "X"


def test_no_false_positive_on_clean_answers():
    ans_a = _make_answer("X", "KNOWS")
    ans_b = _make_answer("Y", "WORKS_WITH")
    detect_answer_contradictions([ans_a, ans_b])
    assert ans_a.contradiction_flags == []
    assert ans_b.contradiction_flags == []


def test_extract_attaches_contradiction_flags():
    """
    End-to-end: two paths through different contradicting relations reach
    different terminals. extract() should attach ContradictionFlag to results.
    """
    emb = np.ones(4, dtype=np.float32) / 2.0

    path_a = TraversalPath(
        nodes=["seed", "ACTIVATES", "target_a"],
        seen_entities={"seed", "target_a"},
        embedding=emb,
        score=0.8,
        attention_weights=[0.8],
        community_sequence=[0, 0],
        edge_confidences=[0.9],
    )
    path_b = TraversalPath(
        nodes=["seed", "INHIBITS", "target_b"],
        seen_entities={"seed", "target_b"},
        embedding=emb,
        score=0.7,
        attention_weights=[0.7],
        community_sequence=[0, 1],
        edge_confidences=[0.8],
    )

    answers = extract([path_a, path_b], top_k=10, min_hop=1)
    # Both targets should appear
    entity_ids = {a.entity_id for a in answers}
    assert "target_a" in entity_ids
    assert "target_b" in entity_ids

    # The contradicting answers should be flagged
    flagged = [a for a in answers if a.contradiction_flags]
    assert len(flagged) == 2


def test_extract_contradiction_flags_not_filtered():
    """
    Contradiction flags are research signals â€” they do NOT cause answers
    to be removed from the result set.
    """
    emb = np.ones(4, dtype=np.float32) / 2.0

    paths = [
        TraversalPath(
            nodes=["S", "CAUSES", f"T{i}"],
            seen_entities={"S", f"T{i}"},
            embedding=emb,
            score=0.9 - i * 0.1,
            attention_weights=[0.9 - i * 0.1],
            community_sequence=[0, 0],
            edge_confidences=[1.0],
        )
        for i in range(3)
    ] + [
        TraversalPath(
            nodes=["S", "PREVENTS", "T3"],
            seen_entities={"S", "T3"},
            embedding=emb,
            score=0.5,
            attention_weights=[0.5],
            community_sequence=[0, 1],
            edge_confidences=[0.7],
        )
    ]

    answers = extract(paths, top_k=10, min_hop=1)
    # All 4 distinct terminals must be present
    assert len(answers) == 4

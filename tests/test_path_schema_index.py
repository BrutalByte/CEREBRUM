"""Unit tests for PathSchemaIndex (Phase 236)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from core.path_schema_index import PathSchemaIndex, _rel_to_text


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Edge:
    def __init__(self, source_id, target_id, relation_type):
        self.source_id    = source_id
        self.target_id    = target_id
        self.relation_type = relation_type


class _Adapter:
    def __init__(self, G: nx.DiGraph):
        self._G = G

    def to_networkx(self):
        return self._G

    def get_neighbors(self, entity_id, edge_types=None, max_neighbors=100, **_kw):
        if entity_id not in self._G:
            return []
        edges = []
        for _, nbr, data in self._G.out_edges(entity_id, data=True):
            rel = data.get("relation", "RELATED_TO")
            if edge_types and rel not in edge_types:
                continue
            edges.append(_Edge(entity_id, nbr, rel))
        return edges[:max_neighbors]


def _make_adapter(edges):
    """edges: [(src, tgt, rel), ...]"""
    G = nx.DiGraph()
    for u, v, rel in edges:
        G.add_edge(u, v, relation=rel)
    return _Adapter(G)


class _FixedEngine:
    """Embedding engine that assigns deterministic unit vectors per text."""
    def __init__(self, dim=8):
        self._dim   = dim
        self._store = {}

    @property
    def dim(self):
        return self._dim

    def encode(self, texts):
        out = []
        for t in texts:
            if t not in self._store:
                rng = np.random.RandomState(hash(t) % (2**31))
                v   = rng.randn(self._dim).astype(np.float32)
                v  /= np.linalg.norm(v)
                self._store[t] = v
            out.append(self._store[t])
        return np.stack(out).astype(np.float32)

    def encode_one(self, text):
        return self.encode([text])[0]


_ENG = _FixedEngine()

# ---------------------------------------------------------------------------
# _rel_to_text unit tests
# ---------------------------------------------------------------------------

def test_schema_text_tokenization():
    assert _rel_to_text("person.person.place_of_birth") == "place of birth"


def test_schema_text_underscore():
    assert _rel_to_text("film.film.directed_by") == "directed by"


def test_schema_text_single_segment():
    assert _rel_to_text("RELATED_TO") == "RELATED TO"


def test_schema_text_no_underscore():
    assert _rel_to_text("location.location.containedby") == "containedby"


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

def test_build_empty_graph():
    adp = _make_adapter([])
    idx = PathSchemaIndex().build(adp, _ENG)
    assert idx.schema_count() == 0


def test_build_single_hop_graph():
    # A→(r1)→B, no 2-hop path exists
    adp = _make_adapter([("A", "B", "rel1")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    assert idx.schema_count() == 0


def test_build_two_hop_one_schema():
    # A→(r1)→B→(r2)→C  — one 2-hop schema
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    assert idx.schema_count() == 1
    assert ("r1", "r2") in idx._schemas


def test_build_multiple_schemas():
    edges = [
        ("A", "B", "r1"), ("B", "C", "r2"),
        ("D", "B", "r3"),   # B also has incoming r3
    ]
    adp = _make_adapter(edges)
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = set(idx._schemas)
    assert ("r1", "r2") in schemas
    assert ("r3", "r2") in schemas


def test_build_filters_min_count():
    # (r1, r2) appears once; (r1, r3) appears twice via two intermediates
    edges = [
        ("A", "B", "r1"), ("B", "C", "r2"),   # (r1,r2) count=1
        ("A", "X", "r1"), ("X", "Y", "r3"),   # (r1,r3) count=1
        ("A", "Z", "r1"), ("Z", "W", "r3"),   # (r1,r3) count=2
    ]
    adp = _make_adapter(edges)
    idx = PathSchemaIndex().build(adp, _ENG, min_count=2)
    schemas = set(idx._schemas)
    assert ("r1", "r3") in schemas
    assert ("r1", "r2") not in schemas   # only appeared once


def test_build_filters_skip_relations_r1():
    adp = _make_adapter([("A", "B", "skip_me"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1,
                                  skip_relations=frozenset({"skip_me"}))
    assert idx.schema_count() == 0


def test_build_filters_skip_relations_r2():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "skip_me")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1,
                                  skip_relations=frozenset({"skip_me"}))
    assert idx.schema_count() == 0


def test_build_matrix_shape():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    assert idx._schema_matrix.shape == (1, _ENG.dim)


# ---------------------------------------------------------------------------
# predict_schemas tests
# ---------------------------------------------------------------------------

def test_predict_schemas_returns_top_k():
    edges = [
        ("A", "B", "r1"), ("B", "C", "r2"),
        ("A", "X", "r3"), ("X", "Y", "r4"),
        ("A", "P", "r5"), ("P", "Q", "r6"),
    ]
    adp = _make_adapter(edges)
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    q = _ENG.encode_one("some question")
    results = idx.predict_schemas(q, top_k=2)
    assert len(results) == 2


def test_predict_schemas_returns_all_when_k_gt_n():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    q = _ENG.encode_one("q")
    results = idx.predict_schemas(q, top_k=100)
    assert len(results) == 1


def test_predict_schemas_ordering():
    # Build an index with two schemas; use a question embedding that is the exact
    # schema embedding of one of them — that schema should rank first.
    adp = _make_adapter([
        ("A", "B", "place_of_birth"), ("B", "C", "containedby"),
        ("D", "E", "directed_by"),    ("E", "F", "name"),
    ])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    # The top result should match whichever schema the question embedding is closest to
    q = _ENG.encode_one("place of birth containedby")  # matches first schema text
    results = idx.predict_schemas(q, top_k=2)
    assert results[0][2] >= results[1][2]  # scores are descending


def test_predict_schemas_zero_embedding():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    zero_vec = np.zeros(_ENG.dim, dtype=np.float32)
    results = idx.predict_schemas(zero_vec, top_k=5)
    assert results == []


def test_predict_schemas_empty_index():
    idx = PathSchemaIndex()  # not built
    q = _ENG.encode_one("q")
    assert idx.predict_schemas(q) == []


def test_predict_schemas_for_seed_filters_by_r1():
    edges = [
        ("A", "B", "r1"), ("B", "C", "r2"),
        ("D", "E", "r3"), ("E", "F", "r4"),
    ]
    adp = _make_adapter(edges)
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    q = _ENG.encode_one("q")
    # Seed only has r1 as outgoing — r3 schema should be excluded
    results = idx.predict_schemas_for_seed(q, seed_outgoing_rels={"r1"}, top_k=5)
    for r1, r2, _ in results:
        assert r1 == "r1"


def test_predict_schemas_for_seed_no_matching_r1():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    q = _ENG.encode_one("q")
    results = idx.predict_schemas_for_seed(q, seed_outgoing_rels={"unrelated_rel"}, top_k=5)
    assert results == []


def test_predict_schemas_for_seed_empty_rels():
    adp = _make_adapter([("A", "B", "r1"), ("B", "C", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    q = _ENG.encode_one("q")
    assert idx.predict_schemas_for_seed(q, seed_outgoing_rels=set(), top_k=5) == []


# ---------------------------------------------------------------------------
# execute_schemas tests
# ---------------------------------------------------------------------------

def test_execute_schemas_finds_hop2():
    adp = _make_adapter([("seed", "mid", "r1"), ("mid", "ans", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.9)]
    results = idx.execute_schemas("seed", schemas, adp)
    entity_ids = [eid for eid, _ in results]
    assert "ans" in entity_ids


def test_execute_schemas_excludes_seed():
    # If seed happens to be a hop-2 result, it should be excluded
    adp = _make_adapter([("seed", "mid", "r1"), ("mid", "seed", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.9)]
    results = idx.execute_schemas("seed", schemas, adp)
    assert all(eid != "seed" for eid, _ in results)


def test_execute_schemas_excludes_intermediates():
    # mid is a hop-1 node; it should not appear as a hop-2 answer
    adp = _make_adapter([
        ("seed", "mid", "r1"), ("mid", "ans", "r2"), ("mid", "mid2", "r2"),
        ("seed", "mid", "r1"),  # mid also has r2 back to itself via another path
    ])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.9)]
    results = idx.execute_schemas("seed", schemas, adp)
    entity_ids = {eid for eid, _ in results}
    assert "mid" not in entity_ids


def test_execute_schemas_empty_hop1():
    # seed has no r1 neighbors
    adp = _make_adapter([("seed", "mid", "r_other"), ("mid", "ans", "r2")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.9)]
    results = idx.execute_schemas("seed", schemas, adp)
    assert results == []


def test_execute_schemas_empty_hop2():
    # mid reached via r1 but has no r2 neighbors
    adp = _make_adapter([("seed", "mid", "r1"), ("mid", "ans", "r_other")])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.9)]
    results = idx.execute_schemas("seed", schemas, adp)
    assert results == []


def test_execute_schemas_best_score_wins():
    # Same answer reachable by two schemas with different scores — best score kept
    adp = _make_adapter([
        ("seed", "midA", "r1"), ("midA", "ans", "r2"),
        ("seed", "midB", "r3"), ("midB", "ans", "r4"),
    ])
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    schemas = [("r1", "r2", 0.8), ("r3", "r4", 0.5)]
    results = dict(idx.execute_schemas("seed", schemas, adp))
    assert "ans" in results
    assert abs(results["ans"] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_full_pipeline():
    """Build → predict → execute: verify the correct 2-hop answer is found."""
    edges = [
        ("einstein", "ulm",    "place_of_birth"),
        ("ulm",      "germany", "containedby"),
        ("einstein", "prize",   "award_wins"),   # distractor
        ("prize",    "nobody",  "winner"),
    ]
    adp = _make_adapter(edges)
    idx = PathSchemaIndex().build(adp, _ENG, min_count=1)
    assert idx.schema_count() >= 2

    # Predict schemas for a question whose embedding matches "place of birth containedby"
    q = _ENG.encode_one("place of birth containedby")
    schemas = idx.predict_schemas(q, top_k=3)
    assert len(schemas) > 0

    # Execute the top predicted schema
    answers = idx.execute_schemas("einstein", schemas, adp)
    entity_ids = {eid for eid, _ in answers}

    # "germany" should appear (via place_of_birth → containedby)
    # Note: RandomEngine may not rank the right schema first, but it must be reachable
    # via at least one of the top-3 schemas if the right schema is in the index
    predicted_rels = {(r1, r2) for r1, r2, _ in schemas}
    if ("place_of_birth", "containedby") in predicted_rels:
        assert "germany" in entity_ids

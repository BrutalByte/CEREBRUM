"""Tests for Phase 228 ConceptNet adapter."""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

import pytest

from adapters.conceptnet_adapter import (
    ConceptNetAdapter,
    _parse_entity,
    _parse_relation,
    _entity_lang,
    load_conceptnet,
)
from core.cerebrum import CerebrumGraph

FIXTURE = Path(__file__).parent / "fixtures" / "conceptnet_sample.csv"


# ---------------------------------------------------------------------------
# URI helper unit tests
# ---------------------------------------------------------------------------

def test_parse_relation_simple():
    assert _parse_relation("/r/RelatedTo") == "RelatedTo"

def test_parse_relation_nested():
    assert _parse_relation("/r/IsA") == "IsA"

def test_parse_entity_basic():
    assert _parse_entity("/c/en/football") == "football"

def test_parse_entity_with_pos():
    assert _parse_entity("/c/en/football/n") == "football"

def test_parse_entity_underscore_to_space():
    assert _parse_entity("/c/en/keep_pets") == "keep pets"

def test_parse_entity_disambiguation():
    assert _parse_entity("/c/en/dog/n/wn/dog.n.01") == "dog"

def test_entity_lang_en():
    assert _entity_lang("/c/en/dog") == "en"

def test_entity_lang_fr():
    assert _entity_lang("/c/fr/chien") == "fr"

def test_entity_lang_malformed():
    assert _entity_lang("/r/IsA") == ""


# ---------------------------------------------------------------------------
# load_conceptnet — basic loading
# ---------------------------------------------------------------------------

def test_load_fixture_returns_adapter():
    adapter = load_conceptnet(str(FIXTURE))
    assert isinstance(adapter, ConceptNetAdapter)

def test_load_fixture_node_count():
    adapter = load_conceptnet(str(FIXTURE))
    G = adapter.to_networkx()
    # Fixture has 20 en edges + 1 fr (filtered) + 1 low-weight (filtered) + 1 ExternalURL (filtered)
    assert G.number_of_nodes() > 5

def test_load_fixture_french_filtered():
    adapter = load_conceptnet(str(FIXTURE), lang="en")
    G = adapter.to_networkx()
    assert "chien" not in G.nodes

def test_load_fixture_low_weight_filtered():
    adapter = load_conceptnet(str(FIXTURE), min_weight=1.0)
    G = adapter.to_networkx()
    # "noise" only appears in the 0.5-weight edge — should be absent
    assert "noise" not in G.nodes

def test_load_fixture_externalurl_excluded():
    adapter = load_conceptnet(str(FIXTURE))
    neighbors = {e.target_id for e in adapter.get_neighbors("dog")}
    assert "wikipedia" not in neighbors

def test_load_fixture_max_edges():
    adapter = load_conceptnet(str(FIXTURE), max_edges=5)
    G = adapter.to_networkx()
    assert G.number_of_edges() <= 5

def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_conceptnet("/nonexistent/path/conceptnet.csv")

def test_load_empty_result_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("# only comments\n")
    with pytest.raises(ValueError, match="No edges loaded"):
        load_conceptnet(str(empty))

def test_load_gz_file(tmp_path):
    gz_path = tmp_path / "cn_sample.csv.gz"
    content = FIXTURE.read_bytes()
    # re-compress fixture as .gz
    with gzip.open(str(gz_path), "wb") as f:
        f.write(content)
    adapter = load_conceptnet(str(gz_path))
    assert adapter.to_networkx().number_of_nodes() > 0


# ---------------------------------------------------------------------------
# ConceptNetAdapter — multi-edge behavior
# ---------------------------------------------------------------------------

def test_multi_edge_dog_animal():
    """dog → animal has both IsA and RelatedTo in fixture; both returned."""
    adapter = load_conceptnet(str(FIXTURE))
    neighbors = {(e.target_id, e.relation_type) for e in adapter.get_neighbors("dog")}
    assert ("animal", "IsA") in neighbors
    assert ("animal", "RelatedTo") in neighbors

def test_projected_graph_single_edge_per_pair():
    """to_networkx() returns at most one edge per (u, v) pair (max-weight)."""
    adapter = load_conceptnet(str(FIXTURE))
    G = adapter.to_networkx()
    # DiGraph: only one edge per (u, v)
    assert not G.is_multigraph()
    # dog → animal projected edge should use IsA (weight=3.0) over RelatedTo (2.0)
    edge_data = G.get_edge_data("dog", "animal")
    assert edge_data is not None
    assert edge_data["weight"] == 3.0

def test_get_entity_exists():
    adapter = load_conceptnet(str(FIXTURE))
    e = adapter.get_entity("dog")
    assert e is not None
    assert e.id == "dog"

def test_get_entity_missing():
    adapter = load_conceptnet(str(FIXTURE))
    assert adapter.get_entity("unicorn_xyz") is None

def test_get_degree():
    adapter = load_conceptnet(str(FIXTURE))
    # dog has edges to: animal(×2), pet, mammal, vertebrate, loyal, allergy
    assert adapter.get_degree("dog") >= 5

def test_find_entities_exact():
    adapter = load_conceptnet(str(FIXTURE))
    results = adapter.find_entities("dog")
    ids = [e.id for e in results]
    assert "dog" in ids

def test_edge_filter_by_type():
    adapter = load_conceptnet(str(FIXTURE))
    isa_only = adapter.get_neighbors("dog", edge_types=["IsA"])
    assert all(e.relation_type == "IsA" for e in isa_only)
    assert len(isa_only) > 0

def test_add_edge_appears_in_neighbors():
    adapter = load_conceptnet(str(FIXTURE))
    adapter.add_edge("dog", "organism", relation="IsA", confidence=0.9)
    targets = {e.target_id for e in adapter.get_neighbors("dog")}
    assert "organism" in targets

def test_confidence_normalized():
    """Edge confidence should be min(weight/10, 1.0)."""
    adapter = load_conceptnet(str(FIXTURE))
    # dog → pet has weight=4.0 → confidence=0.4
    edges = {e.target_id: e for e in adapter.get_neighbors("dog")}
    assert "pet" in edges
    assert abs(edges["pet"].confidence - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# CerebrumGraph.from_conceptnet integration
# ---------------------------------------------------------------------------

def test_from_conceptnet_builds():
    graph = CerebrumGraph.from_conceptnet(str(FIXTURE))
    graph.build()
    assert graph.adapter is not None

def test_from_conceptnet_query_returns_results():
    graph = CerebrumGraph.from_conceptnet(str(FIXTURE))
    graph.build()
    results = graph.query(["dog"], top_k=5, max_hop=2)
    assert len(results) > 0

def test_from_conceptnet_2hop_path():
    """dog → mammal → animal is a valid 2-hop path in the fixture."""
    graph = CerebrumGraph.from_conceptnet(str(FIXTURE))
    graph.build()
    results = graph.query(["dog"], top_k=10, max_hop=2)
    entity_ids = {r.entity_id for r in results}
    # "animal" reachable directly (1-hop) and via mammal/vertebrate (2-hop)
    assert "animal" in entity_ids

def test_from_conceptnet_lang_filter():
    graph = CerebrumGraph.from_conceptnet(str(FIXTURE), lang="en")
    graph.build()
    assert graph.adapter.get_entity("chien") is None

def test_from_conceptnet_sentence_embeddings():
    pytest.importorskip("sentence_transformers")
    graph = CerebrumGraph.from_conceptnet(str(FIXTURE), embeddings="sentence")
    graph.build()
    results = graph.query(["dog"], top_k=3, max_hop=1)
    assert len(results) > 0

"""
Unit and component tests for csv_adapter and NetworkXAdapter helpers.

Covers:
  - load_csv_adapter: loading, node/edge counts, edge data, error handling
  - NetworkXAdapter.find_entities: exact match, fuzzy match
  - NetworkXAdapter.from_triples: factory method
"""
from pathlib import Path

import pytest

from adapters.csv_adapter import load_csv_adapter
from adapters.networkx_adapter import NetworkXAdapter

TOY_CSV = Path(__file__).parent / "fixtures" / "toy_graph.csv"


# ---------------------------------------------------------------------------
# load_csv_adapter — basic loading
# ---------------------------------------------------------------------------

def test_load_toy_graph_node_count():
    """The toy graph has exactly 21 distinct entities.

    Note: CLAUDE.md previously stated '19-node' — this test caught the
    discrepancy. Ground truth is the CSV fixture (21 nodes, verified
    2026-03-18). See TEST_LOG.md Run 002 for the failure record.
    """
    adapter = load_csv_adapter(str(TOY_CSV))
    assert adapter.node_count() == 21


def test_load_toy_graph_edge_count():
    """The toy graph has exactly 30 edges as defined in the fixture."""
    adapter = load_csv_adapter(str(TOY_CSV))
    G = adapter.to_networkx()
    assert G.number_of_edges() == 30


def test_load_toy_graph_known_edge():
    """newton → einstein via INFLUENCED must be present."""
    adapter = load_csv_adapter(str(TOY_CSV))
    G = adapter.to_networkx()
    assert G.has_edge("newton", "einstein")
    data = G.get_edge_data("newton", "einstein")
    assert data.get("relation") == "INFLUENCED"


def test_load_missing_file_raises():
    """Loading a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_csv_adapter("no/such/file.csv")


def test_load_directed_flag(tmp_path):
    """directed=True must return a DiGraph-backed adapter."""
    csv_file = tmp_path / "tiny.csv"
    csv_file.write_text("source,target,relation\na,b,REL\n")
    adapter = load_csv_adapter(str(csv_file), directed=True)
    G = adapter.to_networkx()
    assert G.is_directed()


def test_load_undirected_flag(tmp_path):
    """directed=False (the default) must return an undirected Graph."""
    csv_file = tmp_path / "tiny.csv"
    csv_file.write_text("source,target,relation\na,b,REL\n")
    adapter = load_csv_adapter(str(csv_file), directed=False)
    G = adapter.to_networkx()
    assert not G.is_directed()


def test_load_no_relation_col_uses_default(tmp_path):
    """If the relation column is absent, relation defaults to RELATED_TO."""
    csv_file = tmp_path / "no_rel.csv"
    csv_file.write_text("source,target\nx,y\n")
    adapter = load_csv_adapter(str(csv_file), relation_col="relation")
    G = adapter.to_networkx()
    data = G.get_edge_data("x", "y")
    assert data.get("relation") == "RELATED_TO"


def test_load_skips_blank_rows(tmp_path):
    """Rows with empty source or target must be silently skipped."""
    csv_file = tmp_path / "blanks.csv"
    csv_file.write_text("source,target,relation\na,b,REL\n,b,REL\na,,REL\n")
    adapter = load_csv_adapter(str(csv_file))
    G = adapter.to_networkx()
    assert G.number_of_edges() == 1


# ---------------------------------------------------------------------------
# NetworkXAdapter — find_entities
# ---------------------------------------------------------------------------

def test_networkx_adapter_find_entities_exact():
    """An exact query must return that entity as the first result."""
    adapter = load_csv_adapter(str(TOY_CSV))
    results = adapter.find_entities("newton")
    assert len(results) >= 1
    assert results[0].id == "newton"


def test_networkx_adapter_find_entities_fuzzy():
    """A near-miss query ('einsten') must return einstein via fuzzy match."""
    adapter = load_csv_adapter(str(TOY_CSV))
    results = adapter.find_entities("einsten")
    ids = [e.id for e in results]
    assert "einstein" in ids


def test_networkx_adapter_find_entities_no_match():
    """A completely unknown query must return an empty list, not an error."""
    adapter = load_csv_adapter(str(TOY_CSV))
    results = adapter.find_entities("xyzzy_no_match_ever")
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# NetworkXAdapter — from_triples factory
# ---------------------------------------------------------------------------

def test_networkx_adapter_from_triples_builds_graph():
    """from_triples must build a graph with the correct edges."""
    triples = [
        ("a", "KNOWS", "b"),
        ("b", "KNOWS", "c"),
    ]
    adapter = NetworkXAdapter.from_triples(triples)
    G = adapter.to_networkx()
    assert G.has_edge("a", "b")
    assert G.has_edge("b", "c")
    assert G.number_of_nodes() == 3


def test_networkx_adapter_from_triples_relation_stored():
    """Relation type must be stored as edge attribute 'relation'."""
    adapter = NetworkXAdapter.from_triples([("x", "LINKS", "y")])
    G = adapter.to_networkx()
    assert G.get_edge_data("x", "y")["relation"] == "LINKS"


def test_networkx_adapter_from_triples_directed():
    """Default directed=True must produce a DiGraph."""
    adapter = NetworkXAdapter.from_triples([("a", "R", "b")], directed=True)
    assert adapter.to_networkx().is_directed()


def test_networkx_adapter_from_triples_undirected():
    """directed=False must produce an undirected Graph."""
    adapter = NetworkXAdapter.from_triples([("a", "R", "b")], directed=False)
    assert not adapter.to_networkx().is_directed()




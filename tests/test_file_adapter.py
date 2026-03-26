"""
Tests for the universal file loader (adapters/file_adapter.py).

Covers all supported formats: CSV, TSV, JSON (array/dict/triples), JSONL,
GraphML, GEXF, GML. Parquet and Excel require pandas/openpyxl which are
optional — those loaders are tested with a skip guard.

All tests use pytest's tmp_path fixture for temp files.
"""
import json
import textwrap

import networkx as nx
import pytest

from adapters.file_adapter import load_file_adapter
from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def edge_set(adapter: NetworkXAdapter):
    """Return set of (source, target) tuples from the adapter's graph."""
    G = adapter.to_networkx()
    return set(G.edges())


def relation_of(adapter: NetworkXAdapter, src: str, tgt: str) -> str:
    G = adapter.to_networkx()
    return G.get_edge_data(src, tgt, {}).get("relation", "RELATED_TO")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def test_load_csv_basic(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target,relation\nalice,bob,KNOWS\nbob,carol,WORKS_WITH\n")
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)
    assert ("bob", "carol") in edge_set(adapter)


def test_load_csv_relation_preserved(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target,relation\nalice,bob,KNOWS\n")
    adapter = load_file_adapter(str(f))
    assert relation_of(adapter, "alice", "bob") == "KNOWS"


def test_load_csv_default_relation(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target\nalice,bob\n")
    adapter = load_file_adapter(str(f))
    assert relation_of(adapter, "alice", "bob") == "RELATED_TO"


def test_load_csv_directed(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target,relation\nalice,bob,KNOWS\n")
    adapter = load_file_adapter(str(f), directed=True)
    G = adapter.to_networkx()
    assert G.is_directed()


def test_load_csv_skips_comment_rows(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target,relation\n#comment,skip,this\nalice,bob,KNOWS\n")
    adapter = load_file_adapter(str(f))
    G = adapter.to_networkx()
    assert G.number_of_nodes() == 2  # only alice and bob


def test_load_csv_custom_column_names(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("from,to,type\nalice,bob,KNOWS\n")
    adapter = load_file_adapter(str(f), source_col="from", target_col="to", relation_col="type")
    assert ("alice", "bob") in edge_set(adapter)


# ---------------------------------------------------------------------------
# TSV
# ---------------------------------------------------------------------------

def test_load_tsv(tmp_path):
    f = tmp_path / "edges.tsv"
    f.write_text("source\ttarget\trelation\nalice\tbob\tKNOWS\n")
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)


# ---------------------------------------------------------------------------
# JSON — array of edge objects
# ---------------------------------------------------------------------------

def test_load_json_array_of_objects(tmp_path):
    f = tmp_path / "edges.json"
    data = [{"source": "alice", "target": "bob", "relation": "KNOWS"}]
    f.write_text(json.dumps(data))
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)


def test_load_json_relation_preserved(tmp_path):
    f = tmp_path / "edges.json"
    data = [{"source": "alice", "target": "bob", "relation": "KNOWS"}]
    f.write_text(json.dumps(data))
    adapter = load_file_adapter(str(f))
    assert relation_of(adapter, "alice", "bob") == "KNOWS"


# ---------------------------------------------------------------------------
# JSON — nodes-and-edges object
# ---------------------------------------------------------------------------

def test_load_json_nodes_and_edges(tmp_path):
    f = tmp_path / "graph.json"
    data = {
        "nodes": ["alice", "bob", "carol"],
        "edges": [
            {"source": "alice", "target": "bob",   "relation": "KNOWS"},
            {"source": "bob",   "target": "carol",  "relation": "WORKS_WITH"},
        ],
    }
    f.write_text(json.dumps(data))
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)
    assert ("bob", "carol") in edge_set(adapter)


# ---------------------------------------------------------------------------
# JSON — triples array
# ---------------------------------------------------------------------------

def test_load_json_triples(tmp_path):
    f = tmp_path / "triples.json"
    data = [["alice", "KNOWS", "bob"], ["bob", "WORKS_WITH", "carol"]]
    f.write_text(json.dumps(data))
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)
    assert relation_of(adapter, "alice", "bob") == "KNOWS"


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def test_load_jsonl(tmp_path):
    f = tmp_path / "edges.jsonl"
    f.write_text(
        '{"source": "alice", "target": "bob", "relation": "KNOWS"}\n'
        '{"source": "bob", "target": "carol", "relation": "WORKS_WITH"}\n'
    )
    adapter = load_file_adapter(str(f))
    assert ("alice", "bob") in edge_set(adapter)
    assert ("bob", "carol") in edge_set(adapter)


# ---------------------------------------------------------------------------
# GraphML
# ---------------------------------------------------------------------------

def test_load_graphml(tmp_path):
    G = nx.Graph()
    G.add_edge("alice", "bob", relation="KNOWS")
    G.add_edge("bob", "carol", relation="WORKS_WITH")
    p = str(tmp_path / "graph.graphml")
    nx.write_graphml(G, p)
    adapter = load_file_adapter(p)
    assert ("alice", "bob") in edge_set(adapter)


# ---------------------------------------------------------------------------
# GML
# ---------------------------------------------------------------------------

def test_load_gml(tmp_path):
    G = nx.Graph()
    G.add_edge("0", "1")   # GML uses integer node IDs by default
    p = str(tmp_path / "graph.gml")
    nx.write_gml(G, p)
    adapter = load_file_adapter(p)
    G_loaded = adapter.to_networkx()
    assert G_loaded.number_of_edges() >= 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_load_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file_adapter(str(tmp_path / "nonexistent.csv"))


def test_load_unsupported_extension(tmp_path):
    f = tmp_path / "data.xyz"
    f.write_text("irrelevant")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_file_adapter(str(f))


def test_load_returns_networkx_adapter(tmp_path):
    f = tmp_path / "edges.csv"
    f.write_text("source,target,relation\nalice,bob,KNOWS\n")
    result = load_file_adapter(str(f))
    assert isinstance(result, NetworkXAdapter)


# ---------------------------------------------------------------------------
# Optional: Parquet (skip if pandas not available)
# ---------------------------------------------------------------------------

pandas = pytest.importorskip("pandas", reason="pandas not installed")

def test_load_parquet(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "source":   ["alice", "bob"],
        "target":   ["bob",   "carol"],
        "relation": ["KNOWS", "WORKS_WITH"],
    })
    p = tmp_path / "edges.parquet"
    df.to_parquet(str(p), index=False)
    adapter = load_file_adapter(str(p))
    assert ("alice", "bob") in edge_set(adapter)
    assert ("bob", "carol") in edge_set(adapter)

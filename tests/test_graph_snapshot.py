"""
Tests for GraphSnapshot (Phase 81).

Coverage:
  - save(): produces valid JSON with correct node/edge counts
  - save(): edge attributes (confidence, provenance, synthetic, weight) preserved
  - save(): node attributes (label, type, properties) preserved
  - save(): creates parent directories
  - restore(): re-adds edges to a fresh adapter
  - restore(skip_existing=True): skips already-present edges
  - restore(skip_existing=False): adds without checking
  - restore(): returns correct added/skipped/errors counts
  - restore(): edges with missing source/target counted as errors
  - load_raw(): returns raw dict without adapter
  - diff(): identifies added and removed edges between two snapshots
  - diff(): node_delta and edge_delta correct
  - round-trip: save → restore reproduces edge count
  - multigraph: save handles MultiGraph edge iteration
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.persistence import GraphSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(directed=False, multigraph=False):
    if multigraph:
        G = nx.MultiGraph()
    elif directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_node("newton", label="Isaac Newton", type="person")
    G.add_node("einstein", label="Albert Einstein", type="person")
    G.add_node("relativity", label="Relativity", type="concept")

    G.add_edge("newton", "einstein", relation="INFLUENCED",
               confidence=0.95, provenance="research_agent:f-01",
               synthetic=True, weight=1.0)
    G.add_edge("einstein", "relativity", relation="DEVELOPED",
               confidence=1.0, provenance="wikidata:Q42",
               synthetic=False, weight=2.0)
    return NetworkXAdapter(G)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_produces_valid_json(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))
        data = json.loads((tmp_path / "snap.json").read_text())
        assert data["version"] == "1.0"
        assert "saved_at" in data

    def test_node_and_edge_counts(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        result = snap.save(adapter, str(tmp_path / "snap.json"))
        assert result["node_count"] == 3
        assert result["edge_count"] == 2

    def test_edge_attributes_preserved(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))
        data = json.loads((tmp_path / "snap.json").read_text())
        edges_by_key = {(e["source"], e["target"]): e for e in data["edges"]}
        e = edges_by_key[("newton", "einstein")]
        assert e["relation"] == "INFLUENCED"
        assert abs(e["confidence"] - 0.95) < 1e-6
        assert e["provenance"] == "research_agent:f-01"
        assert e["synthetic"] is True
        assert abs(e["weight"] - 1.0) < 1e-6

    def test_node_attributes_preserved(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))
        data = json.loads((tmp_path / "snap.json").read_text())
        nodes_by_id = {n["id"]: n for n in data["nodes"]}
        assert nodes_by_id["newton"]["label"] == "Isaac Newton"
        assert nodes_by_id["newton"]["type"] == "person"

    def test_creates_parent_directories(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        dest = tmp_path / "deep" / "nested" / "snap.json"
        snap.save(adapter, str(dest))
        assert dest.exists()

    def test_returns_stats_dict(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        result = snap.save(adapter, str(tmp_path / "snap.json"))
        assert "node_count" in result
        assert "edge_count" in result
        assert "path" in result

    def test_multigraph_save(self, tmp_path):
        adapter = _make_adapter(multigraph=True)
        snap = GraphSnapshot()
        result = snap.save(adapter, str(tmp_path / "snap.json"))
        assert result["edge_count"] == 2


# ---------------------------------------------------------------------------
# restore()
# ---------------------------------------------------------------------------

class TestRestore:
    def test_roundtrip_edge_count(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))

        G_new = nx.Graph()
        new_adapter = NetworkXAdapter(G_new)
        result = snap.restore(str(tmp_path / "snap.json"), new_adapter)

        assert result["added"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_skip_existing_true(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))

        # Restore into an adapter that already has one of the edges
        G_new = nx.Graph()
        G_new.add_edge("newton", "einstein", relation="INFLUENCED",
                       confidence=0.5, provenance="", synthetic=False)
        new_adapter = NetworkXAdapter(G_new)
        result = snap.restore(str(tmp_path / "snap.json"), new_adapter,
                              skip_existing=True)

        assert result["skipped"] == 1
        assert result["added"] == 1

    def test_skip_existing_false(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))

        # Even if an edge exists, skip_existing=False re-adds without checking
        G_new = nx.Graph()
        G_new.add_edge("newton", "einstein", relation="INFLUENCED",
                       confidence=0.5, provenance="", synthetic=False)
        new_adapter = NetworkXAdapter(G_new)
        result = snap.restore(str(tmp_path / "snap.json"), new_adapter,
                              skip_existing=False)

        assert result["added"] == 2
        assert result["skipped"] == 0

    def test_missing_source_target_counted_as_error(self, tmp_path):
        # Manually craft a snapshot with a bad edge
        snap_data = {
            "version": "1.0", "saved_at": 0.0,
            "node_count": 0, "edge_count": 1,
            "nodes": [],
            "edges": [{"relation": "X"}],  # missing source/target
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(snap_data))

        G_new = nx.Graph()
        new_adapter = NetworkXAdapter(G_new)
        snap = GraphSnapshot()
        result = snap.restore(str(p), new_adapter)
        assert result["errors"] == 1

    def test_edge_attributes_restored(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))

        G_new = nx.Graph()
        new_adapter = NetworkXAdapter(G_new)
        snap.restore(str(tmp_path / "snap.json"), new_adapter)

        edge_data = G_new.get_edge_data("newton", "einstein")
        assert edge_data is not None
        assert edge_data["relation"] == "INFLUENCED"
        assert abs(edge_data["confidence"] - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# load_raw()
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_returns_dict(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))
        raw = GraphSnapshot.load_raw(str(tmp_path / "snap.json"))
        assert isinstance(raw, dict)
        assert "edges" in raw
        assert "nodes" in raw

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GraphSnapshot.load_raw(str(tmp_path / "nonexistent.json"))


# ---------------------------------------------------------------------------
# diff()
# ---------------------------------------------------------------------------

class TestDiff:
    def test_edges_added(self, tmp_path):
        adapter_a = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter_a, str(tmp_path / "a.json"))

        # Add a new edge to create snapshot B
        G_b = adapter_a.to_networkx().copy()
        G_b.add_edge("newton", "relativity", relation="INSPIRED",
                     confidence=0.8, provenance="", synthetic=False, weight=1.0)
        adapter_b = NetworkXAdapter(G_b)
        snap.save(adapter_b, str(tmp_path / "b.json"))

        result = snap.diff(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
        assert len(result["edges_added"]) == 1
        assert result["edges_added"][0]["relation"] == "INSPIRED"
        assert result["edge_delta"] == 1

    def test_edges_removed(self, tmp_path):
        adapter_a = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter_a, str(tmp_path / "a.json"))

        # Remove an edge in B
        G_b = adapter_a.to_networkx().copy()
        G_b.remove_edge("newton", "einstein")
        adapter_b = NetworkXAdapter(G_b)
        snap.save(adapter_b, str(tmp_path / "b.json"))

        result = snap.diff(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
        assert len(result["edges_removed"]) == 1
        assert result["edge_delta"] == -1

    def test_identical_snapshots_zero_diff(self, tmp_path):
        adapter = _make_adapter()
        snap = GraphSnapshot()
        snap.save(adapter, str(tmp_path / "snap.json"))

        result = snap.diff(str(tmp_path / "snap.json"), str(tmp_path / "snap.json"))
        assert result["edges_added"] == []
        assert result["edges_removed"] == []
        assert result["edge_delta"] == 0
        assert result["node_delta"] == 0

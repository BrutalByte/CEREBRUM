"""
Tests for Phase 227: NVMe WAL + MmapConsolidator + MmapAdvisor.

All tests use a temporary directory — no persistent side effects.
No live NVMe required; tmpdir is on whatever drive the OS temp is on.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.graph_wal import GraphWAL
from core.mmap_policy import (
    MmapAdvisor,
    MmapConsolidator,
    MmapPolicy,
    MmapRecommendation,
    resolve_mmap_dir,
)
from core.rem_engine import REMEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "cerebrum_data"
    d.mkdir()
    return d


@pytest.fixture
def simple_adapter():
    G = nx.DiGraph()
    edges = [
        ("einstein", "curie",    "COLLABORATED", 0.9),
        ("curie",    "bohr",     "INFLUENCED",   0.8),
        ("bohr",     "einstein", "INFLUENCED",   0.7),
        ("darwin",   "huxley",   "MENTORED",     0.6),
    ]
    for src, tgt, rel, conf in edges:
        G.add_edge(src, tgt, relation=rel, confidence=conf, weight=conf)
    adapter = NetworkXAdapter(G)
    adapter.community_map = {
        "einstein": 0, "curie": 0, "bohr": 0,
        "darwin": 1, "huxley": 1,
    }
    emb_dim = 8
    rng = np.random.default_rng(42)
    adapter.embeddings = {
        node: rng.random(emb_dim).astype("float32")
        for node in G.nodes()
    }
    return adapter


# ===========================================================================
# GraphWAL tests
# ===========================================================================

class TestGraphWAL:

    def test_append_creates_file(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        assert not wal.exists()
        wal.append("a", "b", "REL")
        assert wal.exists()

    def test_append_records_all_fields(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        wal.append("src", "tgt", "CAUSES", confidence=0.75,
                   provenance="orin", synthetic=True)
        records = list(wal)
        assert len(records) == 1
        r = records[0]
        assert r["op"]   == "add"
        assert r["src"]  == "src"
        assert r["tgt"]  == "tgt"
        assert r["rel"]  == "CAUSES"
        assert abs(r["conf"] - 0.75) < 1e-5
        assert r["prov"] == "orin"
        assert r["syn"]  is True
        assert "ts" in r

    def test_entry_count(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        assert wal.entry_count() == 0
        for i in range(5):
            wal.append(f"n{i}", f"n{i+1}", "REL")
        assert wal.entry_count() == 5

    def test_size_bytes_grows(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        assert wal.size_bytes() == 0
        wal.append("a", "b", "REL")
        assert wal.size_bytes() > 0

    def test_truncate_removes_file(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        wal.append("a", "b", "REL")
        assert wal.exists()
        wal.truncate()
        assert not wal.exists()
        assert wal.entry_count() == 0

    def test_truncate_idempotent(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        wal.truncate()   # should not raise
        wal.truncate()

    def test_replay_into_adapter(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        wal.append("einstein", "darwin", "KNEW", confidence=0.5,
                   provenance="test")
        wal.append("darwin",   "huxley", "MENTORED", confidence=0.9,
                   provenance="test")

        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        count = wal.replay(adapter)

        assert count == 2
        assert adapter._G.has_edge("einstein", "darwin")
        assert adapter._G.has_edge("darwin", "huxley")

    def test_replay_skips_unknown_ops(self, tmp_data_dir):
        wal_path = tmp_data_dir / "edges.wal"
        wal_path.write_text(
            '{"op":"del","ts":1.0,"src":"a","tgt":"b","rel":"R","conf":1.0,"prov":"","syn":false}\n'
            '{"op":"add","ts":2.0,"src":"x","tgt":"y","rel":"R","conf":1.0,"prov":"","syn":false}\n',
            encoding="utf-8",
        )
        wal = GraphWAL(tmp_data_dir)
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        count = wal.replay(adapter)
        assert count == 1
        assert adapter._G.has_edge("x", "y")

    def test_replay_skips_malformed_lines(self, tmp_data_dir):
        wal_path = tmp_data_dir / "edges.wal"
        wal_path.write_text(
            'NOT_JSON\n'
            '{"op":"add","ts":1.0,"src":"a","tgt":"b","rel":"R","conf":1.0,"prov":"","syn":false}\n',
            encoding="utf-8",
        )
        wal = GraphWAL(tmp_data_dir)
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        count = wal.replay(adapter)
        assert count == 1

    def test_thread_safety(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        errors = []
        def _write():
            try:
                for i in range(20):
                    wal.append(f"a{i}", f"b{i}", "REL")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert wal.entry_count() == 100  # 5 threads × 20 appends

    def test_iter_empty_wal(self, tmp_data_dir):
        wal = GraphWAL(tmp_data_dir)
        assert list(wal) == []


# ===========================================================================
# MmapAdvisor tests
# ===========================================================================

class TestMmapAdvisor:

    def test_tiny_graph_recommends_ram(self):
        rec = MmapAdvisor().evaluate(100, 500, 64, MmapPolicy.AUTO)
        assert isinstance(rec, MmapRecommendation)
        assert rec.use_mmap is False
        assert rec.pressure_pct < 20.0

    def test_always_policy_overrides_ram_recommendation(self):
        rec = MmapAdvisor().evaluate(100, 500, 64, MmapPolicy.ALWAYS)
        assert rec.use_mmap is True
        assert rec.override_used is True
        assert rec.policy == MmapPolicy.ALWAYS

    def test_never_policy_overrides_mmap_recommendation(self):
        # Large graph — advisor would recommend mmap
        rec = MmapAdvisor().evaluate(10_000_000, 50_000_000, 768, MmapPolicy.NEVER)
        assert rec.use_mmap is False
        assert rec.policy == MmapPolicy.NEVER

    def test_returns_correct_mb_estimates(self):
        n_nodes, n_edges, dim = 1000, 5000, 128
        rec = MmapAdvisor().evaluate(n_nodes, n_edges, dim, MmapPolicy.AUTO)
        expected_graph_mb = (n_nodes * 32 + n_edges * 12) / 1024 ** 2
        expected_emb_mb   = (n_nodes * dim * 4) / 1024 ** 2
        assert abs(rec.graph_mb     - expected_graph_mb) < 0.001
        assert abs(rec.embedding_mb - expected_emb_mb)   < 0.001

    def test_auto_no_override_flag(self):
        rec = MmapAdvisor().evaluate(100, 500, 64, MmapPolicy.AUTO)
        assert rec.override_used is False

    def test_recommendation_has_reason(self):
        rec = MmapAdvisor().evaluate(100, 500, 64, MmapPolicy.AUTO)
        assert isinstance(rec.reason, str) and len(rec.reason) > 0


# ===========================================================================
# MmapConsolidator tests
# ===========================================================================

class TestMmapConsolidator:

    def test_flush_creates_all_files(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, simple_adapter.embeddings)

        assert report.success
        assert (tmp_data_dir / "graph.a").exists()
        assert (tmp_data_dir / "graph.e").exists()
        assert (tmp_data_dir / "nodes.map").exists()
        assert (tmp_data_dir / "relations.idx").exists()
        assert (tmp_data_dir / "embeddings.e").exists()
        assert (tmp_data_dir / "graph.meta").exists()

    def test_flush_writes_correct_node_count(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, simple_adapter.embeddings)
        assert report.node_count == simple_adapter._G.number_of_nodes()

    def test_flush_writes_correct_edge_count(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, simple_adapter.embeddings)
        assert report.edge_count == simple_adapter._G.number_of_edges()

    def test_graph_meta_contains_expected_fields(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        consolidator.flush(simple_adapter, simple_adapter.embeddings,
                           build_id="test-build-123")
        meta = json.loads((tmp_data_dir / "graph.meta").read_text())
        assert meta["version"]       == 1
        assert meta["build_id"]      == "test-build-123"
        assert meta["node_count"]    == simple_adapter._G.number_of_nodes()
        assert meta["edge_count"]    == simple_adapter._G.number_of_edges()
        assert meta["embedding_dim"] == 8
        assert "written_at" in meta

    def test_relations_idx_contains_all_relations(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        consolidator.flush(simple_adapter, simple_adapter.embeddings)
        stored = (tmp_data_dir / "relations.idx").read_text().splitlines()
        graph_rels = {
            d.get("relation", "LINKED")
            for _, _, d in simple_adapter._G.edges(data=True)
        }
        assert set(stored) == graph_rels

    def test_flush_truncates_wal(self, tmp_data_dir, simple_adapter):
        wal = GraphWAL(tmp_data_dir)
        wal.append("a", "b", "REL")
        assert wal.exists()

        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, simple_adapter.embeddings,
                                    wal=wal)
        assert report.wal_truncated is True
        assert not wal.exists()

    def test_flush_is_atomic_on_success(self, tmp_data_dir, simple_adapter):
        """No _tmp_flush directory should remain after a successful flush."""
        consolidator = MmapConsolidator(tmp_data_dir)
        consolidator.flush(simple_adapter, simple_adapter.embeddings)
        assert not (tmp_data_dir / "_tmp_flush").exists()

    def test_load_meta_returns_none_when_missing(self, tmp_data_dir):
        consolidator = MmapConsolidator(tmp_data_dir)
        assert consolidator.load_meta() is None

    def test_load_meta_returns_dict_after_flush(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        consolidator.flush(simple_adapter, simple_adapter.embeddings)
        meta = consolidator.load_meta()
        assert isinstance(meta, dict)
        assert "build_id" in meta

    def test_flush_no_embeddings(self, tmp_data_dir, simple_adapter):
        """Flush without embeddings should succeed and skip embeddings.e."""
        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, {})
        assert report.success
        assert not (tmp_data_dir / "embeddings.e").exists()
        meta = consolidator.load_meta()
        assert meta["embedding_dim"] == 0

    def test_flush_report_duration(self, tmp_data_dir, simple_adapter):
        consolidator = MmapConsolidator(tmp_data_dir)
        report = consolidator.flush(simple_adapter, simple_adapter.embeddings)
        assert report.duration_s >= 0.0

    def test_embeddings_matrix_shape(self, tmp_data_dir, simple_adapter):
        """Stored embedding matrix must be [N, dim] float32."""
        consolidator = MmapConsolidator(tmp_data_dir)
        consolidator.flush(simple_adapter, simple_adapter.embeddings)
        n = simple_adapter._G.number_of_nodes()
        emb = np.memmap(tmp_data_dir / "embeddings.e", dtype="float32",
                        mode="r", shape=(n, 8))
        assert emb.shape == (n, 8)


# ===========================================================================
# End-to-end: REM → flush → WAL truncate
# ===========================================================================

class TestREMFlushIntegration:

    def test_rem_on_complete_fires_after_real_run(self, tmp_data_dir,
                                                   simple_adapter):
        fired = threading.Event()
        reports = []

        def on_complete(report):
            reports.append(report)
            fired.set()

        rem = REMEngine(simple_adapter, on_complete=on_complete,
                        prune_confidence_threshold=0.0)
        rem.run(dry_run=False)

        fired.wait(timeout=5.0)
        assert fired.is_set(), "on_complete was not called within 5 s"
        assert len(reports) == 1

    def test_rem_on_complete_does_not_fire_on_dry_run(self, simple_adapter):
        called = []

        rem = REMEngine(simple_adapter, on_complete=lambda r: called.append(r),
                        prune_confidence_threshold=0.0)
        rem.run(dry_run=True)
        time.sleep(0.1)   # give background thread a chance to fire (it shouldn't)
        assert called == []

    def test_full_pipeline_rem_flush_wal(self, tmp_data_dir, simple_adapter):
        """
        Complete round-trip:
          1. Append edges to WAL (simulating Orin perception writes).
          2. Run REM — triggers on_complete in background.
          3. on_complete flushes graph to NVMe and truncates WAL.
          4. Verify NVMe files exist and WAL is gone.
        """
        wal          = GraphWAL(tmp_data_dir)
        consolidator = MmapConsolidator(tmp_data_dir)
        flush_done   = threading.Event()

        def on_complete(report):
            consolidator.flush(simple_adapter, simple_adapter.embeddings,
                               wal=wal)
            flush_done.set()

        # Simulate perception writes between REMs
        wal.append("bohr", "darwin",   "KNEW",     0.5, "orin")
        wal.append("huxley", "curie",  "ADMIRED",  0.4, "orin")
        assert wal.entry_count() == 2

        rem = REMEngine(simple_adapter, on_complete=on_complete,
                        prune_confidence_threshold=0.0)
        rem.run(dry_run=False)

        flush_done.wait(timeout=10.0)
        assert flush_done.is_set(), "NVMe flush did not complete"

        # NVMe store should be populated
        assert (tmp_data_dir / "graph.a").exists()
        assert (tmp_data_dir / "graph.meta").exists()

        # WAL should be gone — baked into the mmap store
        assert not wal.exists()

    def test_wal_survives_without_rem(self, tmp_data_dir):
        """
        Edges written to WAL before a crash persist on disk.
        Simulated by writing to WAL without triggering a flush.
        """
        wal = GraphWAL(tmp_data_dir)
        wal.append("a", "b", "REL", 0.9, "test")
        wal.append("b", "c", "REL", 0.8, "test")

        # Simulate restart: create new WAL pointing at same dir and replay
        wal2 = GraphWAL(tmp_data_dir)
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        replayed = wal2.replay(adapter)

        assert replayed == 2
        assert adapter._G.has_edge("a", "b")
        assert adapter._G.has_edge("b", "c")


# ===========================================================================
# resolve_mmap_dir
# ===========================================================================

class TestResolveMmapDir:

    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("CEREBRUM_MMAP_DIR", raising=False)
        p = resolve_mmap_dir("data/cerebrum")
        assert p == Path("data/cerebrum") / "mmap"

    def test_env_var_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CEREBRUM_MMAP_DIR", str(tmp_path / "nvme"))
        p = resolve_mmap_dir("data/cerebrum")
        assert p == tmp_path / "nvme"

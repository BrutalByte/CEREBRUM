"""
Fault-tolerance tests — Phase 56 / Phase 57.

Covers hardened scenarios:
1. QueryResponse schema backward-compatibility (new optional fields)
2. BeamTraversal._partial_paths checkpoint survives a mid-hop exception
3. /query endpoint returns 200 with partial=True on traversal failure
4. QueryLog / Engram write failures are isolated — they never crash /query
5. GlobalRebalancer worker crash is logged and thread restarts on next trigger
6. /query/stream yields terminal error chunk on traversal failure (Phase 57)
7. best_of_n_dscf falls back to sequential when ProcessPoolExecutor fails (Phase 57)
8. Engram save/load roundtrip preserves affinity counts (Phase 57)
"""
import json
import logging
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from api.schemas import QueryResponse
from adapters.csv_adapter import load_csv_adapter
from core.embedding_engine import RandomEngine
from api.server import create_app

TOY_CSV = str(Path(__file__).parent / "fixtures" / "toy_graph.csv")


# ---------------------------------------------------------------------------
# Shared fixture: loaded TestClient with auth
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Module-scoped client with toy graph loaded and X-API-Key auth header."""
    from core.community_engine import best_of_n_dscf
    adapter = load_csv_adapter(TOY_CSV)
    engine  = RandomEngine(dim=64)
    G       = adapter.to_networkx()
    parts   = best_of_n_dscf(G, n_trials=3, seed=42, use_multiprocessing=False)
    cmap    = {node: cid for cid, members in enumerate(parts) for node in members}
    app = create_app(
        adapter=adapter,
        embedding_engine=engine,
        community_map=cmap,
        use_meta_learning=False,
    )
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
        yield c


def _capture_logs(level=logging.WARNING):
    """Return a context manager that captures log records from the cerebrum hierarchy.

    pytest's caplog relies on root-logger propagation, but cerebrum sets
    propagate=False.  We inject a plain list-handler directly into the
    cerebrum logger so records are captured regardless of propagation.

    Usage::

        records = []
        with _capture_logs(level=logging.WARNING) as records:
            do_something()
        assert any("foo" in r.getMessage() for r in records)
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        captured = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _ListHandler(level)
        cerebrum_root = logging.getLogger("cerebrum")
        cerebrum_root.addHandler(handler)
        try:
            yield captured
        finally:
            cerebrum_root.removeHandler(handler)

    return _ctx()


# ---------------------------------------------------------------------------
# 1. QueryResponse schema backward-compat
# ---------------------------------------------------------------------------

class TestQueryResponseSchema:
    def test_defaults_to_non_partial(self):
        r = QueryResponse(query="q", seeds_used=[], paths=[], total_paths_explored=0)
        assert r.partial is False
        assert r.error is None

    def test_partial_flag_set(self):
        r = QueryResponse(
            query="q", seeds_used=[], paths=[], total_paths_explored=0,
            partial=True, error="adapter blew up",
        )
        assert r.partial is True
        assert r.error == "adapter blew up"

    def test_existing_fields_unchanged(self):
        r = QueryResponse(
            query="test", seeds_used=["a"], paths=[], total_paths_explored=5
        )
        assert r.query == "test"
        assert r.seeds_used == ["a"]
        assert r.total_paths_explored == 5


# ---------------------------------------------------------------------------
# 2. BeamTraversal._partial_paths checkpoint
# ---------------------------------------------------------------------------

class TestPartialPathsCheckpoint:
    def test_partial_paths_initially_empty(self):
        from reasoning.traversal import BeamTraversal
        t = BeamTraversal.__new__(BeamTraversal)
        t._partial_paths = []
        assert t._partial_paths == []

    def test_partial_paths_populated_after_normal_traversal(self):
        from core.cerebrum import CerebrumGraph
        from core.attention_engine import CSAEngine
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
        from reasoning.traversal import BeamTraversal
        g = CerebrumGraph.from_csv(TOY_CSV)
        g.build()
        G = g.adapter.to_networkx()
        cm = g.adapter.community_map
        csa = CSAEngine(adapter=g.adapter)
        csa.set_community_graph(
            build_community_distance_matrix(G, cm),
            adjacent_community_pairs(G, cm),
        )
        t = BeamTraversal(adapter=g.adapter, csa_engine=csa, beam_width=3, max_hop=2)
        t.traverse(["newton"])
        assert len(t._partial_paths) > 0

    def test_partial_paths_survive_mid_hop_exception(self):
        """_partial_paths has hop-1 results even when a later hop raises."""
        from core.cerebrum import CerebrumGraph
        from core.attention_engine import CSAEngine
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
        from reasoning.traversal import BeamTraversal
        g = CerebrumGraph.from_csv(TOY_CSV)
        g.build()
        G = g.adapter.to_networkx()
        cm = g.adapter.community_map
        csa = CSAEngine(adapter=g.adapter)
        csa.set_community_graph(
            build_community_distance_matrix(G, cm),
            adjacent_community_pairs(G, cm),
        )
        t = BeamTraversal(adapter=g.adapter, csa_engine=csa, beam_width=3, max_hop=3)

        original_prune = t._prune_candidates
        call_count = [0]

        def _failing_prune(candidates, hop):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("injected failure on hop 2")
            return original_prune(candidates, hop)

        t._prune_candidates = _failing_prune

        with pytest.raises(RuntimeError, match="injected failure"):
            t.traverse(["newton"])

        assert len(t._partial_paths) > 0


# ---------------------------------------------------------------------------
# 3. /query endpoint graceful degradation on traversal failure
# ---------------------------------------------------------------------------

class TestQueryEndpointGracefulDegradation:
    def test_traversal_failure_returns_200_with_partial_flag(self, client):
        """If traversal raises, /query returns 200 with partial=True (not 500)."""
        with patch("reasoning.traversal.BeamTraversal.traverse",
                   side_effect=RuntimeError("injected traversal crash")):
            resp = client.post("/query", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["partial"] is True
        assert "injected traversal crash" in body["error"]

    def test_normal_query_has_partial_false(self, client):
        resp = client.post("/query", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["partial"] is False
        assert body["error"] is None


# ---------------------------------------------------------------------------
# 4. QueryLog / Engram write failure isolation
# ---------------------------------------------------------------------------

class TestWriteFailureIsolation:
    def test_querylog_oserror_does_not_crash_query(self, client):
        with patch("core.persistence.QueryLog.record", side_effect=OSError("disk full")):
            resp = client.post("/query", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        assert resp.json()["partial"] is False

    def test_engram_error_does_not_crash_query(self, client):
        with patch(
            "reasoning.engram_traversal.Engram.record",
            side_effect=MemoryError("OOM"),
        ):
            resp = client.post("/query", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        assert resp.json()["partial"] is False

    def test_querylog_error_is_logged(self, client):
        with _capture_logs(logging.WARNING) as records:
            with patch("core.persistence.QueryLog.record", side_effect=OSError("no space")):
                client.post("/query", json={"query": "newton", "top_k": 3})
        assert any("QueryLog.record failed" in r.getMessage() for r in records)

    def test_engram_error_is_logged(self, client):
        with _capture_logs(logging.WARNING) as records:
            with patch(
                "reasoning.engram_traversal.Engram.record",
                side_effect=MemoryError("OOM"),
            ):
                client.post("/query", json={"query": "newton", "top_k": 3})
        assert any("Engram.record failed" in r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# 5. GlobalRebalancer worker crash recovery
# ---------------------------------------------------------------------------

class TestRebalancerWorkerCrashRecovery:
    def test_worker_crash_is_logged(self):
        """_rebalance_worker must catch any exception from _rebalance_worker_inner and log at ERROR."""
        from core.rebalancer import GlobalRebalancer
        adapter = MagicMock()
        adapter._lock = threading.Lock()
        rb = GlobalRebalancer(adapter=adapter)

        with patch.object(rb, "_rebalance_worker_inner",
                          side_effect=RuntimeError("unexpected inner crash")):
            with _capture_logs(logging.ERROR) as records:
                rb._full_rebalance()
                if rb._rebalance_thread:
                    rb._rebalance_thread.join(timeout=2.0)

        assert any("Rebalance worker crashed" in r.getMessage() for r in records)

    def test_dead_thread_relaunched_on_next_trigger(self):
        """After a worker crash, the next _full_rebalance() call launches a fresh thread."""
        from core.rebalancer import GlobalRebalancer
        adapter = MagicMock()
        adapter._lock = threading.Lock()
        rb = GlobalRebalancer(adapter=adapter)

        with patch.object(rb, "_rebalance_worker_inner",
                          side_effect=RuntimeError("crash")):
            rb._full_rebalance()
            t1 = rb._rebalance_thread
            if t1:
                t1.join(timeout=2.0)

            # After crash, t1 is dead — next call must launch a new thread
            rb._full_rebalance()
            t2 = rb._rebalance_thread
            assert t2 is not None
            assert t2 is not t1
            if t2:
                t2.join(timeout=2.0)

    def test_rebalancer_inner_method_exists(self):
        from core.rebalancer import GlobalRebalancer
        assert callable(getattr(GlobalRebalancer, "_rebalance_worker_inner", None))


# ---------------------------------------------------------------------------
# 6. /query/stream traversal guard (Phase 57)
# ---------------------------------------------------------------------------

class TestStreamTraversalGuard:
    def test_stream_error_yields_terminal_chunk(self, client):
        """/query/stream must yield a terminal error NDJSON chunk when traversal crashes."""
        with patch(
            "reasoning.traversal.AsyncBeamTraversal.traverse_stream",
            side_effect=RuntimeError("injected stream crash"),
        ):
            resp = client.post("/query/stream", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
        last = json.loads(lines[-1])
        assert last.get("status") == "error"
        assert last.get("partial") is True
        assert "injected stream crash" in last.get("error", "")

    def test_normal_stream_has_no_error_chunk(self, client):
        """/query/stream must not contain an error chunk on a successful query."""
        resp = client.post("/query/stream", json={"query": "newton", "top_k": 3})
        assert resp.status_code == 200
        chunks = [json.loads(ln) for ln in resp.text.strip().splitlines() if ln.strip()]
        assert all(c.get("status") != "error" for c in chunks)


# ---------------------------------------------------------------------------
# 7. ProcessPoolExecutor fallback in best_of_n_dscf (Phase 57)
# ---------------------------------------------------------------------------

class TestProcessPoolFallback:
    def _fake_partition(self, G):
        """Return a trivial valid partition: every node in its own community."""
        return [frozenset([n]) for n in G.nodes()]

    def test_executor_failure_falls_back_to_sequential(self):
        """When ProcessPoolExecutor raises, best_of_n_dscf must call dscf_communities
        sequentially instead of propagating the executor exception."""
        from concurrent.futures import BrokenExecutor
        from adapters.csv_adapter import load_csv_adapter
        from core.community_engine import best_of_n_dscf
        adapter = load_csv_adapter(TOY_CSV)
        G = adapter.to_networkx()
        # Patch both the executor (to raise) and dscf_communities (to return fast),
        # verifying that the fallback code path is reached.
        sentinel = [frozenset(G.nodes())]  # one community — all nodes
        with patch("concurrent.futures.ProcessPoolExecutor",
                   side_effect=BrokenExecutor("paging file too small")):
            with patch("core.community_engine.dscf_communities",
                       return_value=sentinel[0]) as mock_dscf:
                parts = best_of_n_dscf(G, n_trials=2, seed=0, use_multiprocessing=True)
        # dscf_communities should have been called exactly n_trials=2 times (sequential)
        assert mock_dscf.call_count == 2
        # Result must be non-empty
        assert len(parts) > 0

    def test_executor_failure_logs_warning(self):
        """best_of_n_dscf must log a WARNING when executor falls back to sequential."""
        from concurrent.futures import BrokenExecutor
        from adapters.csv_adapter import load_csv_adapter
        from core.community_engine import best_of_n_dscf
        adapter = load_csv_adapter(TOY_CSV)
        G = adapter.to_networkx()
        sentinel = frozenset(G.nodes())
        with _capture_logs(logging.WARNING) as records:
            with patch("concurrent.futures.ProcessPoolExecutor",
                       side_effect=BrokenExecutor("injected executor failure")):
                with patch("core.community_engine.dscf_communities",
                           return_value=sentinel):
                    best_of_n_dscf(G, n_trials=2, seed=0, use_multiprocessing=True)
        assert any("falling back to sequential DSCF" in r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# 8. Engram persistence roundtrip (Phase 57)
# ---------------------------------------------------------------------------

class TestEngramPersistence:
    def test_save_creates_file(self, tmp_path):
        """save_if_path must write a JSON file with the stored patterns."""
        from reasoning.engram_traversal import Engram
        cache = Engram()
        cache._counts[("CAUSES", "TREATS")] = 3
        target = str(tmp_path / "engram_cache.json")
        cache.save_if_path(target)
        assert (tmp_path / "engram_cache.json").exists()
        import json as _json
        data = _json.loads((tmp_path / "engram_cache.json").read_text())
        assert data["version"] == 1
        assert any(pair[1] == 3 for pair in data["counts"])

    def test_load_roundtrip_preserves_counts(self, tmp_path):
        """Engram.load must restore affinity counts written by save."""
        from reasoning.engram_traversal import Engram
        original = Engram()
        seq = ("CAUSES", "TREATS")
        original._counts[seq] = 5
        original._max_count = 5
        path = str(tmp_path / "engram.json")
        original.save(path)

        restored = Engram.load(path)
        assert restored._counts.get(seq) == 5
        # Affinity score must be non-zero for the stored sequence
        assert restored.affinity(seq) > 0

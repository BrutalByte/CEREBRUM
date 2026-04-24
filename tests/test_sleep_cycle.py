"""Tests for Phase 119 — Sleep Cycle Orchestrator."""
import time
import pytest
from unittest.mock import MagicMock, patch
from core.sleep_cycle import SleepCycleOrchestrator, SleepReport


def _make_orchestrator(**kwargs):
    adapter = MagicMock()
    return SleepCycleOrchestrator(adapter=adapter, **kwargs)


class TestSleepReport:
    def test_to_dict(self):
        r = SleepReport(started_at=1.0, duration_seconds=0.5, engrams_promoted=3,
                        wm_entries_replayed=2, edges_strengthened=5,
                        rem_shortcuts_added=1, edges_decayed=10, dmn_insights=2)
        d = r.to_dict()
        assert d["engrams_promoted"] == 3
        assert d["edges_decayed"] == 10
        assert d["dry_run"] is False
        assert d["error"] is None


class TestSleepCycleOrchestrator:
    def test_run_all_phases_called(self):
        engram_con = MagicMock()
        engram_con.consolidate.return_value = 4
        engram_con.cache._counts = {}
        engram_con.canonical_patterns = set()
        engram_con.min_success_threshold = 5

        cons_eng = MagicMock()
        cons_result = MagicMock()
        cons_result.entries_replayed = 3
        cons_result.edges_strengthened = 7
        cons_eng.consolidate.return_value = cons_result
        cons_eng.min_score = 0.6

        decay_eng = MagicMock()
        decay_result = MagicMock()
        decay_result.edges_decayed = 12
        decay_eng.decay.return_value = decay_result

        dmn_eng = MagicMock()
        dmn_eng.idle_scan.return_value = [MagicMock(), MagicMock()]

        wm = MagicMock()
        wm.recent.return_value = []

        orc = SleepCycleOrchestrator(
            adapter=MagicMock(),
            engram_consolidator=engram_con,
            consolidation_engine=cons_eng,
            synaptic_decay_engine=decay_eng,
            default_mode_engine=dmn_eng,
            working_memory=wm,
        )

        report = orc.run(dry_run=False)

        engram_con.consolidate.assert_called_once()
        cons_eng.consolidate.assert_called_once_with(wm)
        decay_eng.decay.assert_called_once_with(wm)
        dmn_eng.idle_scan.assert_called_once()

        assert report.wm_entries_replayed == 3
        assert report.edges_strengthened == 7
        assert report.edges_decayed == 12
        assert report.dmn_insights == 2
        assert report.dry_run is False
        assert report.error is None

    def test_dry_run_skips_mutations(self):
        decay_eng = MagicMock()
        orc = SleepCycleOrchestrator(
            adapter=MagicMock(),
            synaptic_decay_engine=decay_eng,
        )
        report = orc.run(dry_run=True)
        decay_eng.decay.assert_not_called()
        assert report.dry_run is True
        assert report.edges_decayed == 0

    def test_missing_engines_handled_gracefully(self):
        orc = _make_orchestrator()
        report = orc.run()
        assert report.engrams_promoted == 0
        assert report.edges_decayed == 0
        assert report.error is None

    def test_report_duration_positive(self):
        orc = _make_orchestrator()
        report = orc.run()
        assert report.duration_seconds >= 0.0

    def test_concurrent_run_skipped(self):
        orc = _make_orchestrator()
        orc._sleeping = True
        report = orc.run()
        assert report.duration_seconds == 0.0

    def test_notify_activity_updates_timestamp(self):
        orc = _make_orchestrator()
        t_before = orc._last_activity_at
        time.sleep(0.01)
        orc.notify_activity()
        assert orc._last_activity_at > t_before

    def test_cancel_stops_timer(self):
        orc = _make_orchestrator()
        orc.schedule(idle_threshold_seconds=3600)
        assert orc._timer is not None
        orc.cancel()
        assert orc._timer is None

    def test_last_report_stored(self):
        orc = _make_orchestrator()
        assert orc.last_report is None
        orc.run()
        assert orc.last_report is not None

    def test_is_sleeping_false_after_run(self):
        orc = _make_orchestrator()
        orc.run()
        assert orc.is_sleeping is False


class TestSleepCycleAPI:
    def test_api_sleep_run_endpoint(self):
        import networkx as nx
        from fastapi.testclient import TestClient
        from adapters.networkx_adapter import NetworkXAdapter
        from api.server import create_app, _state
        import numpy as np

        g = nx.Graph()
        g.add_node("a")
        adapter = NetworkXAdapter(g)
        adapter.community_map = {"a": 0}
        adapter.embeddings = {"a": np.zeros(64)}

        _state["adapter"] = adapter
        _state["community_map"] = adapter.community_map
        _state["embeddings"] = adapter.embeddings
        _state["csa_metadata"] = {"distances": {}, "adjacent_pairs": set()}
        _state["hologram"] = []
        _state["sleep_orchestrator"] = None

        class MockEmbedding:
            def encode_entities(self, labels):
                return {k: np.zeros(64) for k in labels}

        app = create_app(adapter=adapter, embedding_engine=MockEmbedding())
        client = TestClient(app)

        r = client.post("/v1/sleep/run", json={"dry_run": True})
        assert r.status_code == 200
        data = r.json()
        assert "duration_seconds" in data
        assert data["dry_run"] is True

    def test_api_sleep_status_endpoint(self):
        import networkx as nx
        from fastapi.testclient import TestClient
        from adapters.networkx_adapter import NetworkXAdapter
        from api.server import create_app, _state
        import numpy as np

        g = nx.Graph()
        g.add_node("a")
        adapter = NetworkXAdapter(g)
        adapter.community_map = {"a": 0}
        adapter.embeddings = {"a": np.zeros(64)}

        _state["adapter"] = adapter
        _state["community_map"] = adapter.community_map
        _state["embeddings"] = adapter.embeddings
        _state["csa_metadata"] = {"distances": {}, "adjacent_pairs": set()}
        _state["hologram"] = []
        _state["sleep_orchestrator"] = None

        class MockEmbedding:
            def encode_entities(self, labels):
                return {k: np.zeros(64) for k in labels}

        app = create_app(adapter=adapter, embedding_engine=MockEmbedding())
        client = TestClient(app)

        r = client.get("/v1/sleep/status")
        assert r.status_code == 200
        data = r.json()
        assert "is_sleeping" in data
        assert data["is_sleeping"] is False

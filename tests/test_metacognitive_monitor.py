"""Tests for Phase 121 — Metacognitive Monitor."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.metacognitive_monitor import (
    MetacognitiveMonitor, EpistemicState,
    _DISSONANCE_THRESHOLD, _ENTROPY_THRESHOLD, _GROUNDED_THRESHOLD,
)


def _make_answer(score=0.8, path_confidence=0.9, path_score=0.8, consensus_score=0.8):
    ans = MagicMock()
    ans.score = score
    ans.path_confidence = path_confidence
    ans.path_score = path_score
    ans.consensus_score = consensus_score
    return ans


def _make_path(weights=None):
    path = MagicMock()
    path.attention_weights = weights or [0.6, 0.3, 0.1]
    return path


def _make_trace(pe=0.2, soliton=0.7):
    trace = MagicMock()
    trace.prediction_error = pe
    trace.soliton_index = soliton
    return trace


class TestEpistemicState:
    def test_to_dict_keys(self):
        state = EpistemicState()
        d = state.to_dict()
        assert "epistemic_uncertainty" in d
        assert "confidence_in_uncertainty" in d
        assert "is_dissonant" in d
        assert "historical_pe_mean" in d

    def test_flags_default(self):
        state = EpistemicState()
        assert state.is_dissonant is False
        assert state.is_ambiguous is False
        assert state.is_grounded is True

    def test_to_dict_rounding(self):
        state = EpistemicState(epistemic_uncertainty=0.123456)
        d = state.to_dict()
        assert d["epistemic_uncertainty"] == 0.1235


class TestMetacognitiveMonitorAssess:
    def test_assess_no_engines(self):
        monitor = MetacognitiveMonitor()
        state = monitor.assess(paths=[], answers=[])
        assert isinstance(state, EpistemicState)
        assert 0.0 <= state.epistemic_uncertainty <= 1.0
        assert 0.0 <= state.confidence_in_uncertainty <= 1.0

    def test_assess_uses_trace_pe(self):
        monitor = MetacognitiveMonitor()
        trace = _make_trace(pe=0.9, soliton=0.1)
        state = monitor.assess(paths=[], answers=[], trace=trace)
        assert state.prediction_error == pytest.approx(0.9)
        assert state.soliton_index == pytest.approx(0.1)

    def test_assess_uses_path_confidence(self):
        monitor = MetacognitiveMonitor()
        answers = [_make_answer(path_confidence=0.4)]
        state = monitor.assess(paths=[], answers=answers)
        assert state.path_confidence == pytest.approx(0.4)
        assert state.is_grounded is False  # 0.4 < 0.70

    def test_is_dissonant_flag(self):
        monitor = MetacognitiveMonitor()
        # dissonance = path_score - consensus_score = 0.9 - 0.5 = 0.4 >= 0.35
        answers = [_make_answer(path_score=0.9, consensus_score=0.5)]
        state = monitor.assess(paths=[], answers=answers)
        assert state.dissonance == pytest.approx(0.4)
        assert state.is_dissonant is True

    def test_is_not_dissonant_when_gap_small(self):
        monitor = MetacognitiveMonitor()
        answers = [_make_answer(path_score=0.8, consensus_score=0.75)]
        state = monitor.assess(paths=[], answers=answers)
        assert state.is_dissonant is False

    def test_epistemic_uncertainty_formula(self):
        monitor = MetacognitiveMonitor()
        trace = _make_trace(pe=0.4, soliton=0.6)
        answers = [_make_answer(path_confidence=0.8, path_score=0.8, consensus_score=0.8)]
        state = monitor.assess(paths=[], answers=answers, trace=trace)
        # EU = 0.40*0.4 + 0.25*0.5 + 0.20*0.0 + 0.15*(1-0.8) = 0.16+0.125+0+0.03 = 0.315
        # hop_entropy defaults to 0.5 when no calibration engine
        expected_eu = 0.40 * 0.4 + 0.25 * 0.5 + 0.20 * 0.0 + 0.15 * (1.0 - 0.8)
        assert state.epistemic_uncertainty == pytest.approx(expected_eu, abs=0.01)

    def test_confidence_in_uncertainty_formula(self):
        monitor = MetacognitiveMonitor()
        trace = _make_trace(pe=0.0, soliton=0.8)
        answers = [_make_answer(path_confidence=0.9, path_score=0.9, consensus_score=0.9)]
        state = monitor.assess(paths=[], answers=answers, trace=trace)
        # CIU = 0.40*0.8 + 0.35*(1-0.0) + 0.25*0.9 = 0.32 + 0.35 + 0.225 = 0.895
        expected_ciu = 0.40 * 0.8 + 0.35 * (1.0 - 0.0) + 0.25 * 0.9
        assert state.confidence_in_uncertainty == pytest.approx(expected_ciu, abs=0.01)

    def test_hop_entropy_from_calibration_engine(self):
        from core.calibration_engine import CalibrationEngine
        ce = CalibrationEngine()
        monitor = MetacognitiveMonitor(calibration_engine=ce)
        paths = [_make_path(weights=[0.5, 0.5])]  # uniform = max entropy
        state = monitor.assess(paths=paths, answers=[])
        assert state.hop_entropy > 0.8  # uniform weights → high entropy

    def test_soliton_from_predictive_coder(self):
        pc = MagicMock()
        pc.soliton_stability_map = {"newton": 0.75}
        monitor = MetacognitiveMonitor(predictive_coder=pc)
        state = monitor.assess(paths=[], answers=[], trace=None, seed_ids=["newton"])
        assert state.soliton_index == pytest.approx(0.75)

    def test_calibration_drift_from_wm(self):
        from core.working_memory import WorkingMemoryBuffer, MemoryEntry
        import time
        wm = WorkingMemoryBuffer(maxlen=20)
        for _ in range(5):
            wm.record(MemoryEntry(
                timestamp=time.time(), seeds=["a"], answers=["b"],
                top_score=0.8, soliton_index=0.5, prediction_error=0.3,
                source="query"
            ))

        monitor = MetacognitiveMonitor(working_memory=wm)
        trace = _make_trace(pe=0.7)  # drift = |0.7 - 0.3| = 0.4
        state = monitor.assess(paths=[], answers=[], trace=trace)
        assert state.historical_pe_mean == pytest.approx(0.3, abs=0.01)
        assert state.calibration_drift == pytest.approx(0.4, abs=0.01)

    def test_no_wm_drift_returns_none(self):
        monitor = MetacognitiveMonitor()
        state = monitor.assess(paths=[], answers=[])
        assert state.historical_pe_mean is None
        assert state.calibration_drift is None


class TestRecordOutcome:
    def test_record_outcome_stored(self):
        monitor = MetacognitiveMonitor()
        state = EpistemicState(epistemic_uncertainty=0.3)
        monitor.record_outcome(state, was_correct=True)
        monitor.record_outcome(state, was_correct=False)
        assert len(monitor._history) == 2

    def test_history_window_bounded(self):
        monitor = MetacognitiveMonitor(history_window=3)
        state = EpistemicState(epistemic_uncertainty=0.5)
        for _ in range(10):
            monitor.record_outcome(state, was_correct=True)
        assert len(monitor._history) == 3


class TestAPIIntegration:
    def test_query_response_has_epistemic_state_field(self):
        """QueryResponse schema accepts epistemic_state field."""
        from api.schemas import QueryResponse, EpistemicStateSchema
        es = EpistemicStateSchema(
            prediction_error=0.2, soliton_index=0.7,
            hop_entropy=0.4, path_confidence=0.9,
            dissonance=0.1, consensus_score=0.9,
            epistemic_uncertainty=0.25, confidence_in_uncertainty=0.75,
            is_dissonant=False, is_ambiguous=False, is_grounded=True,
        )
        resp = QueryResponse(
            query="test", seeds_used=["test"],
            paths=[], total_paths_explored=0,
            epistemic_state=es,
        )
        assert resp.epistemic_state is not None
        assert resp.epistemic_state.epistemic_uncertainty == pytest.approx(0.25)

    def test_epistemic_state_schema_none_by_default(self):
        from api.schemas import QueryResponse
        resp = QueryResponse(
            query="test", seeds_used=["test"],
            paths=[], total_paths_explored=0,
        )
        assert resp.epistemic_state is None

    def test_api_query_with_monitor_attached(self):
        import networkx as nx
        import numpy as np
        from fastapi.testclient import TestClient
        from adapters.networkx_adapter import NetworkXAdapter
        from api.server import create_app, _state
        from core.metacognitive_monitor import MetacognitiveMonitor

        g = nx.Graph()
        g.add_node("newton")
        adapter = NetworkXAdapter(g)
        adapter.community_map = {"newton": 0}
        adapter.embeddings = {"newton": np.zeros(64)}

        _state["adapter"] = adapter
        _state["community_map"] = adapter.community_map
        _state["embeddings"] = adapter.embeddings
        _state["csa_metadata"] = {"distances": {}, "adjacent_pairs": set()}
        _state["hologram"] = []
        _state["metacognitive_monitor"] = MetacognitiveMonitor()

        class MockEmbedding:
            def encode_entities(self, labels):
                return {k: np.zeros(64) for k in labels}

        app = create_app(adapter=adapter, embedding_engine=MockEmbedding())
        client = TestClient(app)
        r = client.post("/v1/query", json={"query": "newton"})
        assert r.status_code == 200
        data = r.json()
        # epistemic_state is present (monitor is attached)
        assert "epistemic_state" in data
        if data["epistemic_state"] is not None:
            assert "epistemic_uncertainty" in data["epistemic_state"]

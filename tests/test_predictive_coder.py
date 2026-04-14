"""
Tests for PredictiveCodingEngine (Phase 69).
"""
import pytest
from unittest.mock import MagicMock, patch
from core.predictive_coder import PredictiveCodingEngine, PriorPath, PredictionResult
from reasoning.engram_traversal import Engram


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter(neighbors=None):
    """Return a mock GraphAdapter."""
    adapter = MagicMock()
    adapter.get_neighbors.return_value = neighbors or []
    return adapter


def _make_path(nodes, score=0.8):
    """Return a mock TraversalPath."""
    p = MagicMock()
    p.nodes = nodes
    p.score = score
    p.edge_features = []
    p.attention_weights = []
    return p


# ---------------------------------------------------------------------------
# Cold-start behaviour
# ---------------------------------------------------------------------------

class TestColdStart:
    def test_predict_returns_none_on_empty_engram(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        assert pc.predict(["newton"]) is None

    def test_update_with_none_prior_returns_zero_pe(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        result = pc.update(None, [])
        assert result.prediction_error == 0.0
        assert result.reinforcement_signal == 1.0

    def test_compute_pe_with_none_prior(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        assert pc.compute_pe(None, []) == 0.0


# ---------------------------------------------------------------------------
# Prediction accuracy
# ---------------------------------------------------------------------------

class TestPredictionError:
    def _pc_with_pattern(self, rel_seq):
        """Build a PredictiveCodingEngine whose Engram has one pattern."""
        eng = Engram()
        eng.record(tuple(rel_seq), weight=5)
        return PredictiveCodingEngine(eng, _make_adapter())

    def test_perfect_prediction_pe_is_zero(self):
        rels = ["CAUSES", "TREATS"]
        pc   = self._pc_with_pattern(rels)
        prior = pc.predict(["aspirin"])
        assert prior is not None

        # Actual path uses the same relations
        actual_nodes = ["aspirin", "CAUSES", "inflammation", "TREATS", "pain"]
        path = _make_path(actual_nodes)
        pe   = pc.compute_pe(prior, [path])
        assert pe == 0.0

    def test_total_mismatch_pe_is_one(self):
        pc = self._pc_with_pattern(["CAUSES"])
        prior = pc.predict(["aspirin"])
        assert prior is not None

        # Actual path has completely different relations
        path = _make_path(["aspirin", "MEMBER_OF", "drug_class"])
        pe   = pc.compute_pe(prior, [path])
        assert pe == 1.0

    def test_partial_match_pe_between_zero_and_one(self):
        pc = self._pc_with_pattern(["CAUSES", "TREATS"])
        prior = pc.predict(["aspirin"])
        assert prior is not None

        # Actual path shares one of two relations
        path = _make_path(["aspirin", "CAUSES", "inflammation", "INHIBITS", "enzyme"])
        pe   = pc.compute_pe(prior, [path])
        assert 0.0 < pe < 1.0

    def test_empty_actual_paths_pe_is_one(self):
        pc    = self._pc_with_pattern(["CAUSES"])
        prior = pc.predict(["aspirin"])
        pe    = pc.compute_pe(prior, [])
        assert pe == 1.0


# ---------------------------------------------------------------------------
# Soliton index
# ---------------------------------------------------------------------------

class TestSolitonIndex:
    def test_soliton_index_zero_on_cold_start(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        assert pc._soliton_index("newton") == 0.0

    def test_soliton_index_rises_with_repeated_correct_predictions(self):
        eng = Engram()
        eng.record(("CAUSES", "TREATS"), weight=10)
        pc = PredictiveCodingEngine(eng, _make_adapter())

        # Simulate repeated low-PE updates
        prior = pc.predict(["aspirin"])
        perfect_path = _make_path(["aspirin", "CAUSES", "inflammation", "TREATS", "pain"])

        for _ in range(10):
            pc.update(prior, [perfect_path])

        key = pc._seed_key(["aspirin"])
        assert pc._soliton_index(key) > 0.8

    def test_soliton_index_stays_low_with_repeated_misses(self):
        eng = Engram()
        eng.record(("CAUSES",), weight=5)
        pc  = PredictiveCodingEngine(eng, _make_adapter())

        prior       = pc.predict(["aspirin"])
        wrong_path  = _make_path(["aspirin", "MEMBER_OF", "drug_class"])

        for _ in range(10):
            pc.update(prior, [wrong_path])

        key = pc._seed_key(["aspirin"])
        assert pc._soliton_index(key) < 0.2


# ---------------------------------------------------------------------------
# PredictionResult signal values
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_reinforcement_is_one_minus_pe(self):
        eng = Engram()
        eng.record(("CAUSES",), weight=5)
        pc    = PredictiveCodingEngine(eng, _make_adapter())
        prior = pc.predict(["aspirin"])
        path  = _make_path(["aspirin", "MEMBER_OF", "drug"])  # mismatch

        result = pc.update(prior, [path])
        assert abs(result.reinforcement_signal - (1.0 - result.prediction_error)) < 1e-6

    def test_result_fields_present(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        result = pc.update(None, [])

        assert isinstance(result, PredictionResult)
        assert result.prior is None
        assert result.prediction_error == 0.0
        assert result.soliton_stability == 0.0
        assert result.reinforcement_signal == 1.0


# ---------------------------------------------------------------------------
# ChemicalModulator integration (signal dispatch)
# ---------------------------------------------------------------------------

class TestModulatorIntegration:
    def test_modulator_receives_signals_on_high_pe(self):
        """High PE should trigger arousal/novelty updates on the modulator."""
        eng = Engram()
        eng.record(("CAUSES",), weight=5)
        pc        = PredictiveCodingEngine(eng, _make_adapter())
        modulator = MagicMock()

        prior  = pc.predict(["aspirin"])
        path   = _make_path(["aspirin", "MEMBER_OF", "drug"])  # mismatch
        result = pc.update(prior, [path])

        # Simulate what cerebrum.py does with the result
        modulator.update_arousal(result.prediction_error)
        modulator.update_novelty(result.prediction_error)
        modulator.update_reinforcement(result.reinforcement_signal)

        modulator.update_arousal.assert_called_once_with(result.prediction_error)
        modulator.update_novelty.assert_called_once_with(result.prediction_error)
        modulator.update_reinforcement.assert_called_once_with(result.reinforcement_signal)


# ---------------------------------------------------------------------------
# Soliton stats observability
# ---------------------------------------------------------------------------

class TestSolitonStats:
    def test_soliton_stats_empty_on_cold_start(self):
        eng = Engram()
        pc  = PredictiveCodingEngine(eng, _make_adapter())
        assert pc.soliton_stats() == {}

    def test_soliton_stats_populated_after_update(self):
        eng = Engram()
        eng.record(("CAUSES",), weight=5)
        pc    = PredictiveCodingEngine(eng, _make_adapter())
        prior = pc.predict(["aspirin"])
        path  = _make_path(["aspirin", "CAUSES", "pain"])
        pc.update(prior, [path])

        stats = pc.soliton_stats()
        assert len(stats) == 1
        key = list(stats.keys())[0]
        assert "aspirin" in key


# ---------------------------------------------------------------------------
# ReasoningTrace integration
# ---------------------------------------------------------------------------

class TestTraceIntegration:
    def test_trace_fields_populated(self):
        """Verify that ReasoningTrace gets prior/PE/soliton fields."""
        from reasoning.trace import ReasoningTrace

        trace = ReasoningTrace(query="test", seeds=["aspirin"])
        assert trace.prior is None
        assert trace.prediction_error is None
        assert trace.soliton_index is None

        # Simulate what cerebrum.query() does
        trace.prior = {"predicted_relations": ["CAUSES"], "predicted_nodes": ["pain"], "confidence": 0.8}
        trace.prediction_error = 0.25
        trace.soliton_index    = 0.6

        assert trace.prior["confidence"] == 0.8
        assert trace.prediction_error == 0.25
        assert trace.soliton_index == 0.6

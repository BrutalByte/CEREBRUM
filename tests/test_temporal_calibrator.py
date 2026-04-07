"""
Tests for TemporalCalibrator (Phase 55).

Verifies grid-search calibration of eta (temporal decay) and iota (node
recency) parameters, recall measurement, parameter application, and the
calibrator's interaction with CSAEngine.
"""
from pathlib import Path

import pytest

from core.cerebrum import CerebrumGraph
from core.temporal_calibrator import (
    TemporalCalibrator,
    CalibrationExample,
    CalibrationResult,
)

TOY_CSV = str(Path(__file__).parent / "fixtures" / "toy_graph.csv")


@pytest.fixture(scope="module")
def built_graph():
    g = CerebrumGraph.from_csv(TOY_CSV)
    g.build()
    return g


@pytest.fixture
def calibrator(built_graph):
    return TemporalCalibrator(built_graph, recall_at=5, grid_steps=3)


# ---------------------------------------------------------------------------
# CalibrationExample
# ---------------------------------------------------------------------------

class TestCalibrationExample:
    def test_defaults(self):
        ex = CalibrationExample(["newton"], ["gravity"])
        assert ex.top_k == 5
        assert ex.max_hop == 3
        assert ex.beam_width == 10


# ---------------------------------------------------------------------------
# TemporalCalibrator — setup
# ---------------------------------------------------------------------------

class TestTemporalCalibratorSetup:
    def test_initial_state(self, calibrator):
        assert calibrator.n_examples() == 0

    def test_add_example(self, calibrator):
        calibrator.add_example(["newton"], ["gravity"])
        assert calibrator.n_examples() >= 1
        calibrator.clear_examples()
        assert calibrator.n_examples() == 0

    def test_add_examples_batch(self, calibrator):
        exs = [
            CalibrationExample(["newton"], ["gravity"]),
            CalibrationExample(["einstein"], ["relativity"]),
        ]
        calibrator.add_examples(exs)
        assert calibrator.n_examples() == 2
        calibrator.clear_examples()

    def test_chaining(self, built_graph):
        cal = TemporalCalibrator(built_graph, grid_steps=2)
        cal.add_example(["newton"], ["gravity"]).add_example(["einstein"], ["relativity"])
        assert cal.n_examples() == 2

    def test_calibrate_no_examples_raises(self, built_graph):
        cal = TemporalCalibrator(built_graph, grid_steps=2)
        with pytest.raises(ValueError, match="No calibration examples"):
            cal.calibrate()

    def test_calibrate_unbuilt_graph_raises(self):
        g = CerebrumGraph.from_csv(TOY_CSV)  # not built
        cal = TemporalCalibrator(g, grid_steps=2)
        cal.add_example(["newton"], ["gravity"])
        with pytest.raises(RuntimeError, match="not been built"):
            cal.calibrate()


# ---------------------------------------------------------------------------
# TemporalCalibrator — calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def _seed(self, built_graph):
        """Return a valid seed entity ID from the toy graph."""
        return next(iter(built_graph.adapter.embeddings))

    def test_calibrate_returns_result(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5, grid_steps=2)
        cal.add_example([seed], [seed])  # trivial: seed always appears
        result = cal.calibrate()
        assert isinstance(result, CalibrationResult)

    def test_result_fields_populated(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5, grid_steps=2)
        cal.add_example([seed], [seed])
        result = cal.calibrate()
        assert 0.0 <= result.best_eta  <= 1.0
        assert 0.0 <= result.best_iota <= 1.0
        assert 0.0 <= result.recall    <= 1.0
        assert result.n_examples == 1
        assert result.n_correct  == int(round(result.recall * result.n_examples))

    def test_grid_scores_count(self, built_graph):
        """grid_steps=3 → 3×3=9 grid points evaluated."""
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5, grid_steps=3)
        cal.add_example([seed], [seed])
        result = cal.calibrate()
        assert len(result.grid_scores) == 9

    def test_grid_scores_format(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5, grid_steps=2)
        cal.add_example([seed], [seed])
        result = cal.calibrate()
        for eta, iota, recall in result.grid_scores:
            assert 0.0 <= eta   <= 1.0
            assert 0.0 <= iota  <= 1.0
            assert 0.0 <= recall <= 1.0

    def test_csa_params_restored_after_calibration(self, built_graph):
        """eta and iota must be restored to original values after calibrate()."""
        csa = built_graph._csa
        orig_eta  = csa.eta
        orig_iota = csa.iota
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, grid_steps=2)
        cal.add_example([seed], [seed])
        cal.calibrate()
        assert abs(csa.eta  - orig_eta)  < 1e-9
        assert abs(csa.iota - orig_iota) < 1e-9

    def test_apply_writes_params(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5, grid_steps=2)
        cal.add_example([seed], [seed])
        result = cal.calibrate(
            eta_range=(0.3, 0.7),
            iota_range=(0.3, 0.7),
        )
        cal.apply(result)
        assert abs(built_graph._csa.eta  - result.best_eta)  < 1e-9
        assert abs(built_graph._csa.iota - result.best_iota) < 1e-9
        # Restore
        built_graph._csa.eta  = 0.1
        built_graph._csa.iota = 0.05

    def test_custom_ranges(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, grid_steps=2)
        cal.add_example([seed], [seed])
        result = cal.calibrate(eta_range=(0.2, 0.4), iota_range=(0.0, 0.2))
        assert 0.2 <= result.best_eta  <= 0.4
        assert 0.0 <= result.best_iota <= 0.2


# ---------------------------------------------------------------------------
# measure_recall
# ---------------------------------------------------------------------------

class TestMeasureRecall:
    def _seed(self, built_graph):
        return next(iter(built_graph.adapter.embeddings))

    def test_measure_recall_returns_float(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5)
        cal.add_example([seed], [seed])
        r = cal.measure_recall()
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

    def test_measure_recall_with_override(self, built_graph):
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5)
        cal.add_example([seed], [seed])
        r = cal.measure_recall(eta=0.5, iota=0.2)
        assert 0.0 <= r <= 1.0

    def test_measure_recall_does_not_mutate_csa(self, built_graph):
        csa = built_graph._csa
        orig_eta  = csa.eta
        orig_iota = csa.iota
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5)
        cal.add_example([seed], [seed])
        cal.measure_recall(eta=0.99, iota=0.99)
        assert abs(csa.eta  - orig_eta)  < 1e-9
        assert abs(csa.iota - orig_iota) < 1e-9

    def test_trivial_recall_wrong_answer_zero(self, built_graph):
        """An impossible correct answer → R@K = 0.0."""
        seed = self._seed(built_graph)
        cal = TemporalCalibrator(built_graph, recall_at=5)
        cal.add_example([seed], ["__nonexistent_entity_xyz__"])
        r = cal.measure_recall()
        assert r == 0.0

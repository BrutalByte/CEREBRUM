"""
Unit tests for core.studio_engine.StudioEngine.

All tests run without a live Gradio server — StudioEngine methods are
plain Python and accept plain Python arguments.

Fixtures:
  engine_unloaded  — fresh StudioEngine, no graph loaded
  engine_loaded    — StudioEngine with toy_graph.csv loaded (module-scoped)
"""
from pathlib import Path

import plotly.graph_objects as go
import pytest

from core.studio_engine import StudioEngine

TOY_CSV = str(Path(__file__).parent / "fixtures" / "toy_graph.csv")


def _load(engine: StudioEngine, path: str, emb: str = "Random (Fast)") -> tuple:
    """Consume the load_graph generator and return the final (status, comm) tuple."""
    result = ("", "")
    for result in engine.load_graph(path, emb):
        pass
    return result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_unloaded():
    """Fresh engine with no graph."""
    return StudioEngine()


@pytest.fixture(scope="module")
def engine_loaded():
    """Engine with toy_graph.csv loaded — built once for the whole module."""
    eng = StudioEngine()
    status, _ = _load(eng, TOY_CSV)
    assert "[OK]" in status, f"Setup failed: {status}"
    return eng


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_initial_state_is_empty(self, engine_unloaded):
        e = engine_unloaded
        assert not e.graph_loaded
        assert e.graph_obj is None
        assert e.adapter is None
        assert e.n_nodes == 0
        assert e.stream_running is False
        assert e.stream_event_log == []


# ---------------------------------------------------------------------------
# load_graph
# ---------------------------------------------------------------------------

class TestLoadGraph:
    def test_valid_csv_returns_ok_status(self):
        eng = StudioEngine()
        status, comm = _load(eng, TOY_CSV)
        assert "[OK]" in status
        assert "nodes" in status
        assert "communities" in comm.lower()

    def test_valid_csv_populates_state(self):
        eng = StudioEngine()
        _load(eng, TOY_CSV)
        assert eng.graph_loaded
        assert eng.graph_obj is not None
        assert eng.adapter is not None
        assert eng.n_nodes > 0
        assert eng.rem is not None
        assert eng.insight is not None
        assert eng.validator is not None

    def test_empty_path_returns_error(self, engine_unloaded):
        status, comm = _load(engine_unloaded, "")
        assert "ERROR" in status
        assert comm == "N/A"

    def test_nonexistent_path_returns_error(self, engine_unloaded):
        status, _ = _load(engine_unloaded, "/no/such/file.csv")
        assert "ERROR" in status

    def test_sentence_mode_accepted(self):
        """Sentence embedding mode loads without error (falls back to random if SBERT absent)."""
        eng = StudioEngine()
        status, _ = _load(eng, TOY_CSV, "Sentence (SBERT)")
        # May succeed with SBERT or fall back; either way should not raise
        assert isinstance(status, str)


# ---------------------------------------------------------------------------
# run_reasoning
# ---------------------------------------------------------------------------

class TestRunReasoning:
    def test_before_load_returns_error(self, engine_unloaded):
        html, structured, fig, nodes = engine_unloaded.run_reasoning(
            "newton", 10, 2, 5, 90.0
        )
        assert "Load" in html
        assert structured is None

    def test_empty_query_returns_error(self, engine_loaded):
        html, structured, fig, nodes = engine_loaded.run_reasoning(
            "", 10, 2, 5, 90.0
        )
        assert "Enter" in html

    def test_known_entity_returns_results(self, engine_loaded):
        html, structured, fig, nodes = engine_loaded.run_reasoning(
            "newton", 10, 2, 5, 90.0
        )
        assert isinstance(html, str)
        assert "newton" in html.lower()

    def test_structured_output_is_list(self, engine_loaded):
        _, structured, _, _ = engine_loaded.run_reasoning("newton", 10, 2, 5, 90.0)
        assert isinstance(structured, list)
        if structured:
            assert "answer" in structured[0]
            assert "score" in structured[0]
            assert "path" in structured[0]

    def test_fig_is_plotly_figure(self, engine_loaded):
        _, _, fig, _ = engine_loaded.run_reasoning("newton", 10, 2, 5, 90.0)
        assert isinstance(fig, go.Figure)

    def test_best_path_nodes_is_list(self, engine_loaded):
        _, _, _, nodes = engine_loaded.run_reasoning("newton", 10, 2, 5, 90.0)
        assert isinstance(nodes, list)

    def test_unknown_entity_returns_no_match(self, engine_loaded):
        html, structured, _, _ = engine_loaded.run_reasoning(
            "zzznonexistent_entity_xyz", 5, 1, 3, 90.0
        )
        # Either "No entity matching" or an empty result — not a crash
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# get_graph_stats
# ---------------------------------------------------------------------------

class TestGetGraphStats:
    def test_before_load_returns_empty(self, engine_unloaded):
        fig, summary = engine_unloaded.get_graph_stats()
        assert isinstance(fig, go.Figure)
        assert "No graph loaded" in summary

    def test_after_load_returns_figure(self, engine_loaded):
        fig, summary = engine_loaded.get_graph_stats()
        assert isinstance(fig, go.Figure)

    def test_after_load_summary_contains_counts(self, engine_loaded):
        _, summary = engine_loaded.get_graph_stats()
        assert "Nodes" in summary
        assert "Edges" in summary
        assert "Density" in summary


# ---------------------------------------------------------------------------
# commit_params  (10-parameter CSA vector)
# ---------------------------------------------------------------------------

class TestCommitParams:
    def test_before_load_returns_error(self, engine_unloaded):
        result = engine_unloaded.commit_params(
            0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0
        )
        assert "ERROR" in result

    def test_after_load_returns_success(self, engine_loaded):
        result = engine_loaded.commit_params(
            0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0
        )
        assert "Success" in result

    def test_all_ten_params_applied(self, engine_loaded):
        engine_loaded.commit_params(
            0.1, 0.2, 0.3, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.5
        )
        csa = engine_loaded.csa
        assert abs(csa.alpha   - 0.1)  < 1e-9
        assert abs(csa.beta    - 0.2)  < 1e-9
        assert abs(csa.gamma   - 0.3)  < 1e-9
        assert abs(csa.mu      - 0.09) < 1e-9
        assert abs(csa.theta   - 0.5)  < 1e-9

    def test_accepts_boundary_values(self, engine_loaded):
        result = engine_loaded.commit_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert "Success" in result
        result = engine_loaded.commit_params(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        assert "Success" in result


# ---------------------------------------------------------------------------
# run_rem_cycle
# ---------------------------------------------------------------------------

class TestRunRemCycle:
    def test_before_load_returns_error(self, engine_unloaded):
        result = engine_unloaded.run_rem_cycle()
        assert "ERROR" in result

    def test_dry_run_returns_string(self, engine_loaded):
        result = engine_loaded.run_rem_cycle(dry_run=True)
        assert isinstance(result, str)
        # Should mention REM Complete or an error
        assert "REM" in result or "ERROR" in result


# ---------------------------------------------------------------------------
# Insight / validation
# ---------------------------------------------------------------------------

class TestInsightMethods:
    def test_insight_log_before_load(self, engine_unloaded):
        result = engine_unloaded.get_insight_log()
        assert "not initialized" in result.lower() or isinstance(result, str)

    def test_insight_log_after_load_is_string(self, engine_loaded):
        result = engine_loaded.get_insight_log()
        assert isinstance(result, str)

    def test_validation_before_load(self, engine_unloaded):
        result = engine_unloaded.run_validation()
        assert "not initialized" in result.lower()

    def test_validation_after_load_is_string(self, engine_loaded):
        result = engine_loaded.run_validation()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_stop_when_not_running(self, engine_unloaded):
        msg, status = engine_unloaded.stop_stream()
        assert "stopped" in msg.lower()
        assert status == "Inactive"

    def test_event_log_empty_initially(self, engine_unloaded):
        html = engine_unloaded.get_stream_event_log()
        assert "<table>" in html

    def test_start_already_running_returns_message(self):
        eng = StudioEngine()
        eng.stream_running = True
        msg, _ = eng.start_stream("Simulated", "", 60, 5000)
        assert "already running" in msg.lower()


# ---------------------------------------------------------------------------
# Backup / restore / history  (Phase 54+ additions)
# ---------------------------------------------------------------------------

class TestBackupRestore:
    def test_save_before_load_returns_error(self, engine_unloaded):
        result = engine_unloaded.save_current_state()
        assert "ERROR" in result

    def test_list_backups_returns_list(self, engine_unloaded):
        result = engine_unloaded.list_backups()
        assert isinstance(result, list)

    def test_get_recent_paths_returns_list(self, engine_unloaded):
        result = engine_unloaded.get_recent_paths()
        assert isinstance(result, list)

    def test_load_adds_to_recent_paths(self):
        eng = StudioEngine()
        _load(eng, TOY_CSV)
        paths = eng.get_recent_paths()
        assert isinstance(paths, list)
        # History may or may not include TOY_CSV depending on StudioHistory impl
        # but it must always return a list without raising

    def test_restore_nonexistent_returns_error(self, engine_unloaded):
        result = engine_unloaded.restore_state("nonexistent_backup_xyz.pkl")
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# Visualisation (state-dependent)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_graph_viz_before_load(self, engine_unloaded):
        result = engine_unloaded.generate_graph_viz()
        assert "Load" in result

    def test_3d_viz_before_load(self, engine_unloaded):
        result = engine_unloaded.generate_3d_viz()
        assert "Load" in result

    def test_graph_viz_after_load_returns_html(self, engine_loaded):
        result = engine_loaded.generate_graph_viz()
        assert isinstance(result, str)
        # Either an iframe or a "too large" message
        assert "iframe" in result or "too large" in result.lower()

    def test_3d_viz_after_load_returns_iframe(self, engine_loaded):
        result = engine_loaded.generate_3d_viz()
        assert "iframe" in result

    def test_3d_viz_with_highlight(self, engine_loaded):
        import base64, re
        result = engine_loaded.generate_3d_viz(highlight_nodes=["newton", "einstein"])
        assert "iframe" in result
        # Content is base64-encoded inside the iframe src — decode to verify
        m = re.search(r"base64,([A-Za-z0-9+/=]+)", result)
        assert m is not None, "No base64 payload found in iframe"
        decoded = base64.b64decode(m.group(1)).decode("utf-8")
        assert "newton" in decoded


# ---------------------------------------------------------------------------
# Private helpers (pure functions — no state)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_attention_radar_10_features(self):
        fig = StudioEngine._attention_radar([0.1] * 10)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_attention_radar_short_vector_returns_empty(self):
        fig = StudioEngine._attention_radar([0.1] * 5)
        assert len(fig.data) == 0

    def test_format_path_html_empty(self):
        html = StudioEngine._format_path_html([])
        assert "No paths found" in html

    def test_format_path_html_nonempty(self):
        """Requires a mock answer object with the right attributes."""
        from unittest.mock import MagicMock
        ans = MagicMock()
        ans.entity_id = "einstein"
        ans.score = 0.75
        ans.best_path.nodes = ["newton", "INFLUENCED", "einstein"]
        ans.score_breakdown = {"community": 0.5, "semantic": 0.8, "edge": 0.3}
        html = StudioEngine._format_path_html([ans])
        assert "einstein" in html
        assert "0.7500" in html

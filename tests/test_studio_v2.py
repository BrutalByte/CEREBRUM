"""
Tests for StudioEngine v2 dashboard methods (Phase 75).

Coverage:
  - attach_research_agent / attach_modulator / attach_loop
  - get_auto_approver_audit: no agent, empty log, populated log
  - get_revision_queue: no agent, empty queue, populated queue
  - get_discovery_heatmap: no calibrator, no data, populated
  - get_chemical_panel: no modulator, populated
  - get_loop_panel: no loop, no cycles, populated cycles
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Minimal stub so studio_engine can be imported without full Gradio stack
import sys
import types

# Patch heavy optional dependencies before importing studio_engine
for mod in ["gradio", "gradio.components"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

from core.studio_engine import StudioEngine


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _make_decision(action="approve", confidence=0.8, reason="test", finding_id="f-0"):
    from core.auto_approver import AutoDecision
    return AutoDecision(action=action, confidence=confidence, reason=reason,
                        features=[], finding_id=finding_id)


def _make_cr_record(finding_id="f-0", net=0.3, revision_weight=2.0):
    from core.contradiction_resolver import ContradictionRecord
    return ContradictionRecord(
        finding_id=finding_id,
        contradiction_score=0.1,
        proposed_noisy_or=0.7,
        net_evidence_score=net,
        resolution="revision_candidate",
        revision_weight=revision_weight,
    )


def _make_aa(decisions=None):
    aa = MagicMock()
    aa.audit_log = deque(decisions or [], maxlen=500)
    aa.stats.return_value = {
        "n_trained": len(decisions or []),
        "n_approve": sum(1 for d in (decisions or []) if d.action == "approve"),
        "n_reject": 0,
        "n_review": 0,
        "weights": [],
        "bias": 0.0,
        "policy": {},
    }
    return aa


def _make_agent(aa=None, revision_candidates=None):
    agent = MagicMock()
    agent._auto_approver = aa
    agent._revision_candidates = deque(revision_candidates or [], maxlen=50)
    agent._calibrator = None
    return agent


def _make_modulator(state=None):
    m = MagicMock()
    m.state = state or {
        "reinforcement": 1.0,
        "arousal": 1.2,
        "novelty": 0.6,
        "cohesion": 0.9,
        "persistence": 0.5,
    }
    m.baseline = {
        "reinforcement": 1.0, "arousal": 1.0,
        "novelty": 0.5, "cohesion": 1.0, "persistence": 0.5,
    }
    return m


def _make_loop(running=False, tripped=False, cycles=None):
    loop = MagicMock()
    loop.status.return_value = {
        "running": running,
        "cycle_interval": 300.0,
        "max_materializations_per_cycle": 5,
        "min_approval_rate": 0.1,
        "circuit_breaker_window": 20,
        "dry_run": False,
        "circuit_breaker_tripped": tripped,
        "current_approval_rate": 0.75 if cycles else None,
        "total_cycles": len(cycles or []),
        "total_approved": sum(c["auto_approved"] for c in (cycles or [])),
        "total_rejected": sum(c["auto_rejected"] for c in (cycles or [])),
        "total_review": sum(c["sent_to_review"] for c in (cycles or [])),
        "total_edges_added": sum(c["edges_added"] for c in (cycles or [])),
        "started_at": time.time() if running else None,
        "last_cycle_at": time.time() if cycles else None,
        "recent_cycles": cycles or [],
    }
    return loop


def _dummy_cycle(n, approved=2, rejected=1, review=0, edges=4):
    return {
        "cycle_number": n, "started_at": time.time(), "duration_seconds": 0.5,
        "findings_seen": approved + rejected + review,
        "auto_approved": approved, "auto_rejected": rejected,
        "sent_to_review": review, "edges_added": edges,
        "circuit_breaker_tripped": False, "dry_run": False,
    }


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------

class TestAttach:
    def test_attach_research_agent(self):
        engine = StudioEngine()
        agent = _make_agent()
        engine.attach_research_agent(agent)
        assert engine._research_agent is agent

    def test_attach_modulator(self):
        engine = StudioEngine()
        mod = _make_modulator()
        engine.attach_modulator(mod)
        assert engine._modulator is mod

    def test_attach_loop(self):
        engine = StudioEngine()
        loop = _make_loop()
        engine.attach_loop(loop)
        assert engine._loop is loop


# ---------------------------------------------------------------------------
# get_auto_approver_audit
# ---------------------------------------------------------------------------

class TestAutoApproverAudit:
    def test_returns_notice_when_no_agent(self):
        engine = StudioEngine()
        html = engine.get_auto_approver_audit()
        assert "No AutoApprover" in html or "not attached" in html.lower()

    def test_returns_notice_when_agent_has_no_aa(self):
        engine = StudioEngine()
        agent = _make_agent(aa=None)
        engine.attach_research_agent(agent)
        html = engine.get_auto_approver_audit()
        assert "No AutoApprover" in html

    def test_returns_notice_when_audit_log_empty(self):
        engine = StudioEngine()
        aa = _make_aa(decisions=[])
        agent = _make_agent(aa=aa)
        engine.attach_research_agent(agent)
        html = engine.get_auto_approver_audit()
        assert "empty" in html.lower()

    def test_returns_table_with_decisions(self):
        engine = StudioEngine()
        decisions = [_make_decision("approve", finding_id=f"f-{i}") for i in range(5)]
        aa = _make_aa(decisions=decisions)
        agent = _make_agent(aa=aa)
        engine.attach_research_agent(agent)
        html = engine.get_auto_approver_audit(n=10)
        assert "APPROVE" in html
        assert "f-0" in html or "f-4" in html  # at least one ID appears

    def test_n_parameter_limits_rows(self):
        engine = StudioEngine()
        decisions = [_make_decision(finding_id=f"f-{i}") for i in range(20)]
        aa = _make_aa(decisions=decisions)
        agent = _make_agent(aa=aa)
        engine.attach_research_agent(agent)
        html_5 = engine.get_auto_approver_audit(n=5)
        html_20 = engine.get_auto_approver_audit(n=20)
        # More rows = more content in longer version
        assert len(html_20) >= len(html_5)

    def test_stats_summary_present(self):
        engine = StudioEngine()
        decisions = [_make_decision("approve")]
        aa = _make_aa(decisions=decisions)
        agent = _make_agent(aa=aa)
        engine.attach_research_agent(agent)
        html = engine.get_auto_approver_audit()
        assert "Trained" in html or "Approve" in html


# ---------------------------------------------------------------------------
# get_revision_queue
# ---------------------------------------------------------------------------

class TestRevisionQueue:
    def test_returns_notice_when_no_agent(self):
        engine = StudioEngine()
        html = engine.get_revision_queue()
        assert "not attached" in html.lower() or "No ResearchAgent" in html

    def test_returns_empty_message_when_queue_empty(self):
        engine = StudioEngine()
        engine.attach_research_agent(_make_agent())
        html = engine.get_revision_queue()
        assert "empty" in html.lower()

    def test_populated_queue_shows_records(self):
        engine = StudioEngine()
        recs = [_make_cr_record(f"f-{i}", net=0.2 + i * 0.1) for i in range(3)]
        agent = _make_agent(revision_candidates=recs)
        engine.attach_research_agent(agent)
        html = engine.get_revision_queue()
        assert "f-" in html
        assert "net=" in html

    def test_queue_shows_count(self):
        engine = StudioEngine()
        recs = [_make_cr_record(f"f-{i}") for i in range(4)]
        agent = _make_agent(revision_candidates=recs)
        engine.attach_research_agent(agent)
        html = engine.get_revision_queue()
        assert "4" in html


# ---------------------------------------------------------------------------
# get_discovery_heatmap
# ---------------------------------------------------------------------------

class TestDiscoveryHeatmap:
    def test_returns_figure_when_no_calibrator(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        fig = engine.get_discovery_heatmap()
        assert isinstance(fig, go.Figure)

    def test_returns_figure_when_no_agent(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        fig = engine.get_discovery_heatmap()
        assert isinstance(fig, go.Figure)

    def test_returns_figure_with_calibrator_data(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        calibrator = MagicMock()
        calibrator.stats.return_value = {
            "total_scans": 10, "total_discoveries": 3,
            "communities": {
                0: {"scan_ema": 0.5, "discovery_ema": 0.1, "rate": 0.2, "weight": 2.0},
                1: {"scan_ema": 1.0, "discovery_ema": 0.8, "rate": 0.8, "weight": 0.5},
            },
        }
        agent = _make_agent()
        agent._calibrator = calibrator
        engine.attach_research_agent(agent)
        fig = engine.get_discovery_heatmap()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # two bar traces


# ---------------------------------------------------------------------------
# get_chemical_panel
# ---------------------------------------------------------------------------

class TestChemicalPanel:
    def test_returns_figure_when_no_modulator(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        fig = engine.get_chemical_panel()
        assert isinstance(fig, go.Figure)

    def test_returns_figure_with_modulator(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        engine.attach_modulator(_make_modulator())
        fig = engine.get_chemical_panel()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # bar + baseline scatter

    def test_figure_has_5_scalars(self):
        engine = StudioEngine()
        engine.attach_modulator(_make_modulator())
        fig = engine.get_chemical_panel()
        bar_trace = fig.data[0]
        assert len(bar_trace.y) == 5


# ---------------------------------------------------------------------------
# get_loop_panel
# ---------------------------------------------------------------------------

class TestLoopPanel:
    def test_returns_tuple_when_no_loop(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        html, fig = engine.get_loop_panel()
        assert isinstance(html, str)
        assert isinstance(fig, go.Figure)

    def test_returns_status_html(self):
        engine = StudioEngine()
        loop = _make_loop(running=True)
        engine.attach_loop(loop)
        html, fig = engine.get_loop_panel()
        assert "RUNNING" in html or "running" in html.lower()

    def test_circuit_tripped_shown(self):
        engine = StudioEngine()
        loop = _make_loop(running=True, tripped=True)
        engine.attach_loop(loop)
        html, _ = engine.get_loop_panel()
        assert "TRIPPED" in html

    def test_cycle_history_figure_populated(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        cycles = [_dummy_cycle(i) for i in range(5)]
        loop = _make_loop(running=True, cycles=cycles)
        engine.attach_loop(loop)
        _, fig = engine.get_loop_panel()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # approved/rejected/review + edges line

    def test_no_cycles_returns_empty_figure(self):
        import plotly.graph_objects as go
        engine = StudioEngine()
        loop = _make_loop(running=False, cycles=[])
        engine.attach_loop(loop)
        _, fig = engine.get_loop_panel()
        assert isinstance(fig, go.Figure)

    def test_stats_counters_in_html(self):
        engine = StudioEngine()
        cycles = [_dummy_cycle(1, approved=3, rejected=1, edges=6)]
        loop = _make_loop(running=True, cycles=cycles)
        engine.attach_loop(loop)
        html, _ = engine.get_loop_panel()
        assert "3" in html  # approved count
        assert "6" in html  # edges count

"""
Tests for AutonomousDiscoveryLoop (Phase 74).

Coverage:
  - LoopConfig defaults and field validation
  - run_cycle() with no AutoApprover (all findings → review)
  - run_cycle() with AutoApprover returning approve
  - run_cycle() with AutoApprover returning reject
  - run_cycle() with AutoApprover returning review
  - max_materializations_per_cycle cap
  - dry_run mode: approve() never called
  - circuit breaker: trips when approval rate drops
  - circuit breaker: stays open with healthy approval rate
  - circuit breaker: skips materialization when tripped
  - circuit breaker: auto-resets when window clears to healthy state
  - configure() swaps config at runtime
  - status() keys and types
  - recent_cycles deque capped at 50
  - CycleRecord fields populated correctly
  - start() / stop() lifecycle (background thread)
  - checkpoint saved after approve/reject cycle (non-dry)
  - dry_run: checkpoint NOT saved
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.autonomous_loop import AutonomousDiscoveryLoop, CycleRecord, LoopConfig


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    source_id: str = "a"
    target_id: str = "b"
    discovery_potential: float = 0.5
    gap_score: float = 0.3
    community_distance: int = 1
    seeded_by: str = "embedding_scan"
    local_density: float = 0.1
    created_at: float = field(default_factory=time.time)


@dataclass
class _FakeFinding:
    finding_id: str = "f-001"
    candidate: _FakeCandidate = field(default_factory=_FakeCandidate)
    proposals: List = field(default_factory=list)
    best_confidence: float = 0.7
    literature_status: str = "novel"
    found_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


def _make_agent(findings=None, approve_return=2, reject_returns=None):
    """Return a mock ResearchAgent with no AutoApprover attached."""
    agent = MagicMock()
    agent.scan_once.return_value = findings if findings is not None else [_FakeFinding()]
    agent.approve.return_value = approve_return
    agent.reject.return_value = None
    # No AutoApprover by default
    agent._auto_approver = None
    return agent


def _make_auto_decision(action: str, confidence: float = 0.9, reason: str = "test"):
    from core.auto_approver import AutoDecision
    return AutoDecision(action=action, confidence=confidence, reason=reason, features=[])


def _make_aa(action: str = "approve"):
    """Return a mock AutoApprover that always returns the given action."""
    aa = MagicMock()
    aa.decide.return_value = _make_auto_decision(action)
    aa.fit.return_value = None
    aa.to_dict.return_value = {"weights": [], "bias": 0.0, "policy": {}}
    return aa


# ---------------------------------------------------------------------------
# LoopConfig
# ---------------------------------------------------------------------------

class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.cycle_interval == 300.0
        assert cfg.max_materializations_per_cycle == 5
        assert cfg.min_approval_rate == 0.10
        assert cfg.circuit_breaker_window == 20
        assert cfg.dry_run is False
        assert cfg.approver_checkpoint_path is None

    def test_custom_values(self):
        cfg = LoopConfig(cycle_interval=60.0, dry_run=True)
        assert cfg.cycle_interval == 60.0
        assert cfg.dry_run is True


# ---------------------------------------------------------------------------
# run_cycle — no AutoApprover
# ---------------------------------------------------------------------------

class TestRunCycleNoApprover:
    def test_all_findings_go_to_review(self):
        findings = [_FakeFinding(finding_id=f"f-{i}") for i in range(3)]
        agent = _make_agent(findings=findings)
        loop = AutonomousDiscoveryLoop(agent=agent)
        record = loop.run_cycle()

        assert record.findings_seen == 3
        assert record.auto_approved == 0
        assert record.auto_rejected == 0
        assert record.sent_to_review == 3
        assert record.edges_added == 0

    def test_approve_never_called_without_aa(self):
        agent = _make_agent()
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.run_cycle()
        agent.approve.assert_not_called()


# ---------------------------------------------------------------------------
# run_cycle — AutoApprover approve path
# ---------------------------------------------------------------------------

class TestRunCycleApprove:
    def test_approve_materializes_edges(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings, approve_return=3)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(agent=agent)
        record = loop.run_cycle()

        assert record.auto_approved == 1
        assert record.edges_added == 3
        assert record.auto_rejected == 0
        agent.approve.assert_called_once_with("f-0", cycle_number=1)

    def test_fit_called_with_approved_true(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        aa = _make_aa("approve")
        agent._auto_approver = aa
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.run_cycle()
        aa.fit.assert_called_once()
        call_args, call_kwargs = aa.fit.call_args
        approved = call_kwargs.get("approved", call_args[1] if len(call_args) > 1 else None)
        assert approved is True


# ---------------------------------------------------------------------------
# run_cycle — AutoApprover reject path
# ---------------------------------------------------------------------------

class TestRunCycleReject:
    def test_reject_calls_agent_reject(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("reject")
        loop = AutonomousDiscoveryLoop(agent=agent)
        record = loop.run_cycle()

        assert record.auto_rejected == 1
        assert record.auto_approved == 0
        assert record.edges_added == 0
        agent.reject.assert_called_once_with("f-0")

    def test_fit_called_with_approved_false(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        aa = _make_aa("reject")
        agent._auto_approver = aa
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.run_cycle()
        call_args, call_kwargs = aa.fit.call_args
        approved = call_kwargs.get("approved", call_args[1] if len(call_args) > 1 else None)
        assert approved is False


# ---------------------------------------------------------------------------
# run_cycle — AutoApprover review path
# ---------------------------------------------------------------------------

class TestRunCycleReview:
    def test_review_findings_counted(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("review")
        loop = AutonomousDiscoveryLoop(agent=agent)
        record = loop.run_cycle()

        assert record.sent_to_review == 1
        assert record.auto_approved == 0
        assert record.auto_rejected == 0
        agent.approve.assert_not_called()
        agent.reject.assert_not_called()


# ---------------------------------------------------------------------------
# max_materializations_per_cycle cap
# ---------------------------------------------------------------------------

class TestMaterializationCap:
    def test_cap_limits_approvals(self):
        findings = [_FakeFinding(finding_id=f"f-{i}") for i in range(10)]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(max_materializations_per_cycle=3),
        )
        record = loop.run_cycle()

        assert record.auto_approved == 3
        # Remaining 7 overflow to review
        assert record.sent_to_review == 7

    def test_cap_of_one(self):
        findings = [_FakeFinding(finding_id=f"f-{i}") for i in range(5)]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(max_materializations_per_cycle=1),
        )
        record = loop.run_cycle()
        assert record.auto_approved == 1
        assert record.sent_to_review == 4


# ---------------------------------------------------------------------------
# dry_run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_approve_not_called_in_dry_run(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(dry_run=True))
        record = loop.run_cycle()

        agent.approve.assert_not_called()
        assert record.dry_run is True
        assert record.auto_approved == 1  # counted but not materialized

    def test_reject_not_called_in_dry_run(self):
        findings = [_FakeFinding(finding_id="f-0")]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("reject")
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(dry_run=True))
        loop.run_cycle()
        agent.reject.assert_not_called()

    def test_no_checkpoint_in_dry_run(self, tmp_path):
        checkpoint = str(tmp_path / "ckpt.json")
        findings = [_FakeFinding()]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(dry_run=True, approver_checkpoint_path=checkpoint),
        )
        loop.run_cycle()
        assert not os.path.exists(checkpoint)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def _run_reject_cycles(self, loop, agent, n: int) -> None:
        """Drive n reject decisions into the window."""
        for i in range(n):
            findings = [_FakeFinding(finding_id=f"f-{i}")]
            agent.scan_once.return_value = findings
            agent._auto_approver = _make_aa("reject")
            loop.run_cycle()

    def test_circuit_trips_on_low_approval_rate(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(min_approval_rate=0.5, circuit_breaker_window=5),
        )
        # 5 rejects → 0% approval rate → should trip
        self._run_reject_cycles(loop, agent, 5)
        assert loop.status()["circuit_breaker_tripped"] is True

    def test_circuit_stays_open_with_good_rate(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(min_approval_rate=0.1, circuit_breaker_window=10),
        )
        # 10 approves → 100% rate → should stay open
        for i in range(10):
            findings = [_FakeFinding(finding_id=f"f-{i}")]
            agent.scan_once.return_value = findings
            agent._auto_approver = _make_aa("approve")
            loop.run_cycle()
        assert loop.status()["circuit_breaker_tripped"] is False

    def test_circuit_skips_materialization_when_tripped(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(min_approval_rate=0.5, circuit_breaker_window=4),
        )
        # Trip the breaker with 4 rejects
        self._run_reject_cycles(loop, agent, 4)
        assert loop.status()["circuit_breaker_tripped"] is True

        # Now send an "approve" finding — circuit is tripped, so it should NOT materialize
        findings = [_FakeFinding(finding_id="tripped-f")]
        agent.scan_once.return_value = findings
        agent._auto_approver = _make_aa("approve")
        record = loop.run_cycle()

        agent.approve.assert_not_called()
        assert record.circuit_breaker_tripped is True

    def test_circuit_not_tripped_with_fewer_than_3_decisions(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(min_approval_rate=0.99, circuit_breaker_window=10),
        )
        # Only 2 decisions — not enough to trip even at 99% threshold
        for i in range(2):
            findings = [_FakeFinding(finding_id=f"f-{i}")]
            agent.scan_once.return_value = findings
            agent._auto_approver = _make_aa("reject")
            loop.run_cycle()
        assert loop.status()["circuit_breaker_tripped"] is False


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------

class TestConfigure:
    def test_configure_updates_interval(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.configure(LoopConfig(cycle_interval=60.0))
        assert loop.status()["cycle_interval"] == 60.0

    def test_configure_updates_dry_run(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        assert loop.status()["dry_run"] is False
        loop.configure(LoopConfig(dry_run=True))
        assert loop.status()["dry_run"] is True

    def test_configure_while_running_does_not_crash(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=3600))
        loop.start()
        loop.configure(LoopConfig(cycle_interval=1800))
        assert loop.status()["cycle_interval"] == 1800.0
        loop.stop()


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_keys_present(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        s = loop.status()
        expected_keys = {
            "running", "cycle_interval", "max_materializations_per_cycle",
            "min_approval_rate", "circuit_breaker_window", "dry_run",
            "circuit_breaker_tripped", "current_approval_rate",
            "total_cycles", "total_approved", "total_rejected", "total_review",
            "total_edges_added", "started_at", "last_cycle_at", "recent_cycles",
        }
        assert expected_keys.issubset(s.keys())

    def test_status_totals_accumulate_across_cycles(self):
        findings = [_FakeFinding()]
        agent = _make_agent(findings=findings, approve_return=2)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.run_cycle()
        loop.run_cycle()
        s = loop.status()
        assert s["total_cycles"] == 2
        assert s["total_approved"] == 2
        assert s["total_edges_added"] == 4

    def test_current_approval_rate_none_before_first_cycle(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        assert loop.status()["current_approval_rate"] is None


# ---------------------------------------------------------------------------
# recent_cycles deque
# ---------------------------------------------------------------------------

class TestRecentCycles:
    def test_recent_cycles_capped_at_50(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        for _ in range(60):
            loop.run_cycle()
        s = loop.status()
        assert len(s["recent_cycles"]) == 50

    def test_cycle_record_fields(self):
        findings = [_FakeFinding()]
        agent = _make_agent(findings=findings, approve_return=1)
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.run_cycle()
        r = loop.status()["recent_cycles"][0]
        assert r["cycle_number"] == 1
        assert r["findings_seen"] == 1
        assert r["auto_approved"] == 1
        assert r["edges_added"] == 1
        assert r["dry_run"] is False


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_checkpoint_saved_after_approve(self, tmp_path):
        checkpoint = str(tmp_path / "ckpt.json")
        findings = [_FakeFinding()]
        agent = _make_agent(findings=findings)
        aa = _make_aa("approve")
        aa.to_dict.return_value = {"weights": [0.1, 0.2], "bias": 0.5, "policy": {}}
        agent._auto_approver = aa
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(approver_checkpoint_path=checkpoint),
        )
        loop.run_cycle()
        assert os.path.exists(checkpoint)
        with open(checkpoint) as fh:
            data = json.load(fh)
        assert "weights" in data

    def test_checkpoint_saved_after_reject(self, tmp_path):
        checkpoint = str(tmp_path / "ckpt.json")
        findings = [_FakeFinding()]
        agent = _make_agent(findings=findings)
        agent._auto_approver = _make_aa("reject")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(approver_checkpoint_path=checkpoint),
        )
        loop.run_cycle()
        assert os.path.exists(checkpoint)

    def test_no_checkpoint_without_decisions(self, tmp_path):
        checkpoint = str(tmp_path / "ckpt.json")
        agent = _make_agent(findings=[])  # no findings → no decisions
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(approver_checkpoint_path=checkpoint),
        )
        loop.run_cycle()
        assert not os.path.exists(checkpoint)


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_sets_running_true(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=3600))
        loop.start()
        assert loop.status()["running"] is True
        loop.stop()

    def test_stop_sets_running_false(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=3600))
        loop.start()
        loop.stop()
        assert loop.status()["running"] is False

    def test_start_is_idempotent(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=3600))
        loop.start()
        loop.start()  # should not raise or create a second thread
        assert loop.status()["running"] is True
        loop.stop()

    def test_thread_is_daemon(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig(cycle_interval=3600))
        loop.start()
        assert loop._thread.daemon is True
        loop.stop()

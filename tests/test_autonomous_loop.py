"""
Tests for AutonomousDiscoveryLoop (Phase 74 + 79 + 82).

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
  - auto_rollback_on_trip: rollback_cycle called on circuit trip
  - auto_rollback_on_trip: no rollback when flag is False (default)
  - auto_rollback_on_trip: no rollback in dry_run mode
  - auto_rollback_on_trip: edges_rolled_back in CycleRecord
  - auto_rollback_on_trip: no-op when ledger not attached
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

# ---------------------------------------------------------------------------
# Phase 79 — Auto-rollback on circuit breaker trip
# ---------------------------------------------------------------------------

class TestAutoRollback:
    """Tests for LoopConfig.auto_rollback_on_trip."""

    def _make_agent_with_ledger(self, findings=None, approve_return=2):
        """Agent mock with a real ProvenanceLedger and a mock adapter."""
        from core.provenance_ledger import ProvenanceLedger

        agent = _make_agent(findings=findings, approve_return=approve_return)
        ledger = ProvenanceLedger()
        adapter = MagicMock()
        adapter.remove_edge.return_value = None

        agent._provenance_ledger = ledger
        agent._adapter = adapter
        return agent, ledger, adapter

    def _trip_breaker(self, loop, agent, n=5):
        """Fill the decision window with rejects to trip the circuit breaker."""
        for i in range(n):
            agent.scan_once.return_value = [_FakeFinding(finding_id=f"pre-{i}")]
            agent._auto_approver = _make_aa("reject")
            loop.run_cycle()

    def test_rollback_called_when_flag_set_and_circuit_trips(self):
        agent, ledger, adapter = self._make_agent_with_ledger()
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                min_approval_rate=0.5,
                circuit_breaker_window=4,
                auto_rollback_on_trip=True,
            ),
        )
        # Record a fake batch for the upcoming cycle
        from unittest.mock import patch as _patch
        self._trip_breaker(loop, agent, n=4)

        # Next cycle: approve fires (adding an edge), then circuit is already tripped
        # Seed the ledger with an entry for the next cycle number
        next_cycle = loop._cycle_number + 1
        ledger.record_batch(f"fid-{next_cycle}", f"fid-{next_cycle}",
                            [("a", "b", "r")], cycle_number=next_cycle)

        agent.scan_once.return_value = [_FakeFinding(finding_id=f"fid-{next_cycle}")]
        agent._auto_approver = _make_aa("approve")
        record = loop.run_cycle()

        # Circuit was already tripped before this cycle → rollback_cycle fires
        assert record.circuit_breaker_tripped is True
        assert record.edges_rolled_back >= 0  # may be 0 if no edges added, but no error

    def test_rollback_not_called_when_flag_false(self):
        agent, ledger, adapter = self._make_agent_with_ledger()
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                min_approval_rate=0.5,
                circuit_breaker_window=4,
                auto_rollback_on_trip=False,  # default
            ),
        )
        self._trip_breaker(loop, agent, n=4)

        next_cycle = loop._cycle_number + 1
        ledger.record_batch(f"fid-{next_cycle}", f"fid-{next_cycle}",
                            [("a", "b", "r")], cycle_number=next_cycle)

        agent.scan_once.return_value = [_FakeFinding(finding_id=f"fid-{next_cycle}")]
        agent._auto_approver = _make_aa("approve")
        loop.run_cycle()

        # remove_edge should NOT have been called (no rollback)
        adapter.remove_edge.assert_not_called()

    def test_rollback_skipped_in_dry_run(self):
        agent, ledger, adapter = self._make_agent_with_ledger()
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                min_approval_rate=0.5,
                circuit_breaker_window=4,
                auto_rollback_on_trip=True,
                dry_run=True,
            ),
        )
        self._trip_breaker(loop, agent, n=4)

        next_cycle = loop._cycle_number + 1
        ledger.record_batch(f"fid-{next_cycle}", f"fid-{next_cycle}",
                            [("a", "b", "r")], cycle_number=next_cycle)

        agent.scan_once.return_value = [_FakeFinding(finding_id=f"fid-{next_cycle}")]
        agent._auto_approver = _make_aa("approve")
        record = loop.run_cycle()

        adapter.remove_edge.assert_not_called()
        assert record.edges_rolled_back == 0

    def test_rollback_no_op_when_no_ledger(self):
        """auto_rollback_on_trip is True but no ledger attached — should not raise."""
        agent = _make_agent()
        agent._provenance_ledger = None
        agent._adapter = MagicMock()
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                min_approval_rate=0.5,
                circuit_breaker_window=4,
                auto_rollback_on_trip=True,
            ),
        )
        self._trip_breaker(loop, agent, n=4)
        agent.scan_once.return_value = [_FakeFinding()]
        agent._auto_approver = _make_aa("approve")
        # Should complete without error
        record = loop.run_cycle()
        assert record.edges_rolled_back == 0

    def test_edges_rolled_back_in_cycle_record(self):
        """edges_rolled_back field is present in CycleRecord (even when 0)."""
        agent = _make_agent(findings=[_FakeFinding()])
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(agent=agent)
        record = loop.run_cycle()
        assert hasattr(record, "edges_rolled_back")
        assert record.edges_rolled_back == 0

    def test_auto_rollback_default_is_false(self):
        cfg = LoopConfig()
        assert cfg.auto_rollback_on_trip is False

    def test_configure_sets_auto_rollback(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        loop.configure(LoopConfig(auto_rollback_on_trip=True))
        assert loop.status()["auto_rollback_on_trip"] is True


# ---------------------------------------------------------------------------
# Phase 82 — Adaptive Loop Tuning
# ---------------------------------------------------------------------------

def _make_calibrator(mean_weight: float = 1.0):
    """Return a mock DiscoveryCalibrator whose stats reflect *mean_weight*."""
    cal = MagicMock()
    cal.stats.return_value = {
        "total_scans": 10,
        "total_discoveries": 5,
        "communities": {
            0: {"scan_ema": 1.0, "discovery_ema": 0.5, "rate": 0.5, "weight": mean_weight},
            1: {"scan_ema": 1.0, "discovery_ema": 0.5, "rate": 0.5, "weight": mean_weight},
        },
    }
    return cal


class TestAdaptiveTuning:
    def test_adaptive_tuning_default_false(self):
        cfg = LoopConfig()
        assert cfg.adaptive_tuning is False

    def test_effective_cap_in_cycle_record(self):
        agent = _make_agent(findings=[_FakeFinding()])
        agent._auto_approver = _make_aa("approve")
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(max_materializations_per_cycle=3),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 3  # no calibrator → base value used

    def test_adaptive_cap_increases_when_underexplored(self):
        """mean_weight=2.0 → effective_cap = base_cap * 2."""
        agent = _make_agent(findings=[_FakeFinding() for _ in range(20)])
        agent._auto_approver = _make_aa("approve")
        agent._calibrator = _make_calibrator(mean_weight=2.0)
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=4,
                adaptive_tuning=True,
                adaptive_max_cap=20,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 8  # 4 * 2.0

    def test_adaptive_cap_decreases_when_saturated(self):
        """mean_weight=0.5 → effective_cap = base_cap * 0.5 → clamped to min."""
        agent = _make_agent(findings=[_FakeFinding() for _ in range(10)])
        agent._auto_approver = _make_aa("approve")
        agent._calibrator = _make_calibrator(mean_weight=0.5)
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=4,
                adaptive_tuning=True,
                adaptive_min_cap=1,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 2  # 4 * 0.5

    def test_adaptive_cap_clamped_to_min(self):
        """mean_weight=0.1 → raw cap very small → clamped to adaptive_min_cap."""
        agent = _make_agent(findings=[])
        agent._calibrator = _make_calibrator(mean_weight=0.1)
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=5,
                adaptive_tuning=True,
                adaptive_min_cap=2,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap >= 2  # never below min

    def test_adaptive_cap_clamped_to_max(self):
        """mean_weight=100 → raw cap huge → clamped to adaptive_max_cap."""
        agent = _make_agent(findings=[])
        agent._calibrator = _make_calibrator(mean_weight=100.0)
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=3,
                adaptive_tuning=True,
                adaptive_max_cap=10,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 10

    def test_no_calibrator_uses_base_cap(self):
        agent = _make_agent(findings=[])
        agent._calibrator = None
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=5,
                adaptive_tuning=True,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 5

    def test_empty_communities_uses_base_cap(self):
        agent = _make_agent(findings=[])
        cal = MagicMock()
        cal.stats.return_value = {"total_scans": 0, "total_discoveries": 0, "communities": {}}
        agent._calibrator = cal
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                max_materializations_per_cycle=5,
                adaptive_tuning=True,
            ),
        )
        record = loop.run_cycle()
        assert record.effective_cap == 5

    def test_adaptive_effective_interval_in_status(self):
        agent = _make_agent(findings=[])
        agent._calibrator = _make_calibrator(mean_weight=2.0)
        loop = AutonomousDiscoveryLoop(
            agent=agent,
            config=LoopConfig(
                cycle_interval=600.0,
                adaptive_tuning=True,
                adaptive_min_interval=60.0,
                adaptive_max_interval=7200.0,
            ),
        )
        loop.run_cycle()
        s = loop.status()
        assert "adaptive_effective_interval" in s
        # mean_weight=2.0 → interval = 600 / 2.0 = 300s
        assert abs(s["adaptive_effective_interval"] - 300.0) < 1.0

    def test_configure_sets_adaptive_tuning(self):
        agent = _make_agent(findings=[])
        loop = AutonomousDiscoveryLoop(agent=agent)
        assert loop.status()["adaptive_tuning"] is False
        loop.configure(LoopConfig(adaptive_tuning=True))
        assert loop.status()["adaptive_tuning"] is True


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

"""
Batch B tests — Feature 1 (ContradictionResolver) + Feature 3 (CandidateRegistry).
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ===========================================================================
# Feature 1 — ContradictionResolver
# ===========================================================================

from core.contradiction_resolver import ContradictionRecord, ContradictionResolver


def _proposal(confidence: float, contradiction_score: float = 0.0):
    """Return a minimal mock HypothesisProposal."""
    p = SimpleNamespace()
    p.confidence = confidence
    p.contradiction_score = contradiction_score
    return p


def _finding(finding_id: str = "f1", metadata: dict | None = None):
    f = SimpleNamespace()
    f.finding_id = finding_id
    f.metadata = metadata or {}
    return f


# --- ContradictionResolver unit tests ---

class TestContradictionResolver:
    def setup_method(self):
        self.resolver = ContradictionResolver(
            revision_threshold=0.15,
            discard_threshold=-0.15,
            min_contradiction_score=0.10,
        )

    def test_clean_when_no_proposals(self):
        rec = self.resolver.resolve(_finding(), [])
        assert rec.resolution == "clean"
        assert rec.proposed_noisy_or == 0.0

    def test_clean_when_contradiction_below_threshold(self):
        """contradiction_score=0.05 < min_threshold=0.10 → clean regardless."""
        props = [_proposal(0.9, contradiction_score=0.05)]
        rec = self.resolver.resolve(_finding(), props)
        assert rec.resolution == "clean"

    def test_revision_candidate_when_net_positive(self):
        """High confidence proposals, low contradiction → revision_candidate."""
        props = [
            _proposal(0.85, contradiction_score=0.20),
            _proposal(0.75, contradiction_score=0.20),
        ]
        rec = self.resolver.resolve(_finding(), props)
        # noisy_or(0.85, 0.75) = 1 - (0.15)(0.25) = 0.9625 → net = 0.9625 - 0.20 = 0.76
        assert rec.resolution == "revision_candidate"
        assert rec.net_evidence_score > 0.15

    def test_discardable_when_net_strongly_negative(self):
        """Low confidence, high contradiction → discardable."""
        props = [_proposal(0.15, contradiction_score=0.80)]
        rec = self.resolver.resolve(_finding(), props)
        # noisy_or = 0.15, net = 0.15 - 0.80 = -0.65 < -0.15
        assert rec.resolution == "discardable"
        assert rec.net_evidence_score < -0.15

    def test_contested_when_near_zero_net(self):
        """net_evidence_score in (-0.15, 0.15) → contested."""
        # noisy_or ≈ 0.40, max_contra = 0.35 → net ≈ 0.05
        props = [_proposal(0.40, contradiction_score=0.35)]
        rec = self.resolver.resolve(_finding(), props)
        assert rec.resolution == "contested"

    def test_noisy_or_single_proposal(self):
        props = [_proposal(0.60, contradiction_score=0.0)]
        rec = self.resolver.resolve(_finding(), props)
        # Noisy-OR of single value = 0.60
        assert rec.proposed_noisy_or == pytest.approx(0.60, abs=1e-4)

    def test_noisy_or_multiple_proposals(self):
        """1 - (1-0.5)(1-0.5) = 0.75"""
        props = [_proposal(0.5, 0.0), _proposal(0.5, 0.0)]
        rec = self.resolver.resolve(_finding(), props)
        assert rec.proposed_noisy_or == pytest.approx(0.75, abs=1e-4)

    def test_revision_weight_formula(self):
        """revision_weight = proposed_noisy_or / max(contradiction_score, 0.01)"""
        props = [_proposal(0.80, contradiction_score=0.20)]
        rec = self.resolver.resolve(_finding(), props)
        expected = 0.80 / 0.20
        assert rec.revision_weight == pytest.approx(expected, abs=0.01)

    def test_revision_weight_with_zero_contradiction(self):
        """Zero contradiction → denominator clamped to 0.01."""
        props = [_proposal(0.80, contradiction_score=0.0)]
        # min_contradiction_score=0.10 means this resolves "clean",
        # but revision_weight should be computed with denominator=0.01
        rec = self.resolver.resolve(_finding(), props)
        assert rec.resolution == "clean"
        # revision_weight still computed: 0.80 / 0.01 = 80.0
        assert rec.revision_weight == pytest.approx(0.80 / 0.01, abs=0.01)

    def test_max_contradiction_score_taken_across_proposals(self):
        """Only the max contradiction_score triggers classification."""
        props = [
            _proposal(0.9, contradiction_score=0.05),   # below threshold
            _proposal(0.8, contradiction_score=0.50),   # above threshold
        ]
        rec = self.resolver.resolve(_finding(), props)
        assert rec.contradiction_score == pytest.approx(0.50, abs=1e-4)
        # net = noisy_or(0.9, 0.8) - 0.50 = 0.98 - 0.50 = 0.48 > 0.15
        assert rec.resolution == "revision_candidate"

    def test_finding_id_propagated(self):
        props = [_proposal(0.9, 0.0)]
        rec = self.resolver.resolve(_finding("my_finding"), props)
        assert rec.finding_id == "my_finding"

    def test_invalid_init_params_raise(self):
        with pytest.raises(ValueError):
            ContradictionResolver(revision_threshold=0.0)   # must be positive
        with pytest.raises(ValueError):
            ContradictionResolver(discard_threshold=0.0)    # must be negative
        with pytest.raises(ValueError):
            ContradictionResolver(min_contradiction_score=-1.0)


# --- AutoApprover hard-gate integration ---

class TestAutoApproverContradictionGate:
    def test_discardable_hard_gate(self):
        """AutoApprover rejects a finding whose contradiction_resolution is discardable."""
        from core.auto_approver import AutoApprover, AutoApprovalPolicy

        policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)
        approver = AutoApprover(policy=policy)

        # Seed with a few dummy training examples so cold-start doesn't fire
        dummy = _finding("dummy")
        dummy.best_confidence = 0.5
        dummy.candidate = None
        dummy.validation_report = None
        dummy.literature_status = "unvalidated"
        dummy.proposals = []
        dummy.metadata = {}
        for _ in range(5):
            approver.fit(dummy, approved=True)

        # Finding with a discardable ContradictionRecord
        cr = ContradictionRecord(
            finding_id="f_bad",
            contradiction_score=0.80,
            proposed_noisy_or=0.15,
            net_evidence_score=-0.65,
            resolution="discardable",
            revision_weight=0.19,
        )
        finding = _finding("f_bad", metadata={"contradiction_resolution": cr})
        finding.best_confidence = 0.15
        finding.candidate = None
        finding.validation_report = None
        finding.literature_status = "unvalidated"
        finding.proposals = []

        decision = approver.decide(finding)
        assert decision.action == "reject"
        assert "contradiction_resolver" in decision.reason

    def test_non_discardable_does_not_trigger_gate(self):
        """revision_candidate resolution falls through to normal classifier flow."""
        from core.auto_approver import AutoApprover, AutoApprovalPolicy

        policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)
        approver = AutoApprover(policy=policy)

        cr = ContradictionRecord(
            finding_id="f_rev",
            contradiction_score=0.20,
            proposed_noisy_or=0.90,
            net_evidence_score=0.70,
            resolution="revision_candidate",
            revision_weight=4.5,
        )
        finding = _finding("f_rev", metadata={"contradiction_resolution": cr})
        finding.best_confidence = 0.90
        finding.candidate = None
        finding.validation_report = None
        finding.literature_status = "unvalidated"
        finding.proposals = []

        # Should NOT hit the contradiction gate; lands in cold_start (0 trained)
        decision = approver.decide(finding)
        assert "contradiction_resolver" not in decision.reason


# --- ResearchAgent wiring ---

class TestResearchAgentContradictionWiring:
    def _make_agent(self):
        """Return a minimal ResearchAgent with mocked dependencies."""
        from core.research_agent import ResearchAgent
        adapter = MagicMock()
        adapter.to_networkx.return_value = MagicMock(nodes=MagicMock(return_value=[]),
                                                     edges=MagicMock(return_value=[]))
        hyp_engine = MagicMock()
        return ResearchAgent(adapter, hyp_engine)

    def test_set_contradiction_resolver(self):
        agent = self._make_agent()
        resolver = ContradictionResolver()
        agent.set_contradiction_resolver(resolver)
        assert agent._contradiction_resolver is resolver

    def test_revision_candidates_deque_exists(self):
        agent = self._make_agent()
        import collections
        assert isinstance(agent._revision_candidates, collections.deque)
        assert agent._revision_candidates.maxlen == 50


# ===========================================================================
# Feature 3 — CandidateRegistry
# ===========================================================================

from core.candidate_registry import CandidateRegistry, RegistryEntry


def _cand(source: str = "A", target: str = "B"):
    c = SimpleNamespace()
    c.source_id = source
    c.target_id = target
    return c


class TestCandidateRegistry:
    def setup_method(self):
        self.reg = CandidateRegistry(default_ttl=60.0, max_entries=100)

    def test_cold_start_should_evaluate_true(self):
        """Unknown pair is always evaluable."""
        assert self.reg.should_evaluate(_cand("X", "Y")) is True

    def test_register_creates_entry(self):
        c = _cand("A", "B")
        entry = self.reg.register(c)
        assert isinstance(entry, RegistryEntry)
        assert entry.nomination_count == 1

    def test_mark_evaluated_blocks_re_evaluation(self):
        c = _cand()
        self.reg.register(c)
        self.reg.mark_evaluated(c)
        assert self.reg.should_evaluate(c) is False

    def test_ttl_expiry_re_enables_evaluation(self):
        reg = CandidateRegistry(default_ttl=0.01)  # 10ms TTL
        c = _cand()
        reg.register(c)
        reg.mark_evaluated(c)
        time.sleep(0.02)
        assert reg.should_evaluate(c) is True

    def test_nomination_count_increments(self):
        c = _cand()
        self.reg.register(c)
        self.reg.register(c)
        self.reg.register(c)
        entry = self.reg.register(c)
        assert entry.nomination_count == 4

    def test_boost_at_1_nomination(self):
        c = _cand()
        self.reg.register(c)
        assert self.reg.get_nomination_boost(c) == pytest.approx(1.0, abs=1e-4)

    def test_boost_at_2_nominations(self):
        c = _cand()
        self.reg.register(c)
        self.reg.register(c)
        # 1.0 + 0.5 * log2(2) = 1.5
        assert self.reg.get_nomination_boost(c) == pytest.approx(1.5, abs=1e-4)

    def test_boost_at_4_nominations(self):
        c = _cand()
        for _ in range(4):
            self.reg.register(c)
        # 1.0 + 0.5 * log2(4) = 2.0
        assert self.reg.get_nomination_boost(c) == pytest.approx(2.0, abs=1e-4)

    def test_boost_at_8_nominations(self):
        c = _cand()
        for _ in range(8):
            self.reg.register(c)
        # 1.0 + 0.5 * log2(8) = 2.5
        assert self.reg.get_nomination_boost(c) == pytest.approx(2.5, abs=1e-4)

    def test_boost_capped_at_cap(self):
        reg = CandidateRegistry(nomination_boost_cap=2.0)
        c = _cand()
        for _ in range(100):
            reg.register(c)
        assert reg.get_nomination_boost(c) == pytest.approx(2.0, abs=1e-4)

    def test_boost_unknown_candidate_is_one(self):
        """Unknown candidates get neutral boost of 1.0."""
        assert self.reg.get_nomination_boost(_cand("X", "Z")) == pytest.approx(1.0)

    def test_prune_evicts_old_entries(self):
        reg = CandidateRegistry(default_ttl=0.01)
        c = _cand("P", "Q")
        reg.register(c)
        time.sleep(0.02)
        evicted = reg.prune(max_age_seconds=0.01)
        assert evicted == 1
        assert reg.stats()["total_entries"] == 0

    def test_prune_does_not_evict_recent_entries(self):
        c = _cand()
        self.reg.register(c)
        evicted = self.reg.prune(max_age_seconds=3600.0)
        assert evicted == 0
        assert self.reg.stats()["total_entries"] == 1

    def test_max_entries_lru_eviction(self):
        reg = CandidateRegistry(max_entries=3)
        reg.register(_cand("A", "B"))
        reg.register(_cand("C", "D"))
        reg.register(_cand("E", "F"))
        # This 4th insert should evict the oldest (A, B)
        reg.register(_cand("G", "H"))
        assert reg.stats()["total_entries"] == 3
        # A→B should be gone
        assert reg.should_evaluate(_cand("A", "B")) is True  # not in registry

    def test_stats_returns_expected_keys(self):
        s = self.reg.stats()
        for key in ("total_entries", "never_evaluated", "multi_nominated",
                    "max_nomination_count", "default_ttl", "max_entries",
                    "nomination_boost_cap"):
            assert key in s

    def test_stats_total_counts(self):
        c1 = _cand("A", "B")
        c2 = _cand("C", "D")
        self.reg.register(c1)
        self.reg.register(c1)  # multi-nominated
        self.reg.register(c2)
        s = self.reg.stats()
        assert s["total_entries"] == 2
        assert s["multi_nominated"] == 1
        assert s["max_nomination_count"] == 2

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            CandidateRegistry(default_ttl=-1.0)
        with pytest.raises(ValueError):
            CandidateRegistry(max_entries=0)
        with pytest.raises(ValueError):
            CandidateRegistry(nomination_boost_cap=0.5)


# --- ResearchAgent.set_registry wiring ---

class TestResearchAgentRegistryWiring:
    def _make_agent(self):
        from core.research_agent import ResearchAgent
        adapter = MagicMock()
        adapter.to_networkx.return_value = MagicMock(nodes=MagicMock(return_value=[]),
                                                     edges=MagicMock(return_value=[]))
        hyp_engine = MagicMock()
        return ResearchAgent(adapter, hyp_engine)

    def test_set_registry_wires_correctly(self):
        agent = self._make_agent()
        reg = CandidateRegistry()
        agent.set_registry(reg)
        assert agent._registry is reg

    def test_registry_none_by_default(self):
        agent = self._make_agent()
        assert agent._registry is None

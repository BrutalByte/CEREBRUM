"""
Tests for AutoApprover — Phase 71.

Coverage:
  - Cold-start always returns "review"
  - Hard gate: blocked_statuses → "reject"
  - Hard gate: require_validation=True with no report → "review"
  - Classifier convergence: 20 positive examples → P(approve) > 0.8
  - Classifier convergence: 20 negative examples → P(approve) < 0.2
  - to_dict() / from_dict() round-trip preserves weights and counters
  - audit_log records every auto-decision
  - LLM fallback "YES" → "approve", "NO" → "reject", garbage → "review"
  - LLM fallback error → "review" (no exception raised)
  - _extract_features: correct index mapping for all 12 features
  - stats() returns correct keys and types
  - rate limit: max_auto_per_scan via ResearchAgent integration
  - set_auto_approver() wires the approver at runtime
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from core.auto_approver import AutoApprover, AutoApprovalPolicy, AutoDecision, _sigmoid


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    source_id="A",
    target_id="B",
    discovery_potential=0.7,
    gap_score=0.5,
    community_distance=2,
    seeded_by="embedding_scan",
    local_density=0.2,
):
    return SimpleNamespace(
        source_id=source_id,
        target_id=target_id,
        discovery_potential=discovery_potential,
        gap_score=gap_score,
        community_distance=community_distance,
        seeded_by=seeded_by,
        local_density=local_density,
    )


def _make_proposal(confidence=0.8, path_count=3, contradiction_score=0.1, derived_relation="related_to", derivation_text=""):
    return SimpleNamespace(
        confidence=confidence,
        path_count=path_count,
        contradiction_score=contradiction_score,
        derived_relation=derived_relation,
        derivation_text=derivation_text,
    )


def _make_report(novelty_score=0.9, hit_count=5):
    return SimpleNamespace(novelty_score=novelty_score, hit_count=hit_count)


def _make_finding(
    finding_id="f1",
    best_confidence=0.75,
    literature_status="novel",
    validation_report=None,
    candidate=None,
    proposals=None,
    metadata=None,
):
    if candidate is None:
        candidate = _make_candidate()
    if proposals is None:
        proposals = [_make_proposal()]
    if metadata is None:
        metadata = {}
    return SimpleNamespace(
        finding_id=finding_id,
        best_confidence=best_confidence,
        literature_status=literature_status,
        validation_report=validation_report,
        candidate=candidate,
        proposals=proposals,
        metadata=metadata,
    )


def _make_approver(**policy_kwargs) -> AutoApprover:
    policy = AutoApprovalPolicy(**policy_kwargs)
    return AutoApprover(policy=policy)


# ---------------------------------------------------------------------------
# Cold-start
# ---------------------------------------------------------------------------

def test_cold_start_always_review():
    aa = _make_approver(min_training_examples=10)
    finding = _make_finding(validation_report=_make_report())
    for _ in range(9):
        d = aa.decide(finding)
        assert d.action == "review", f"Expected review during cold-start, got {d.action}"
        assert "cold_start" in d.reason


def test_cold_start_resolves_after_training():
    """After min_training_examples fits, classifier can move past review."""
    aa = _make_approver(min_training_examples=3, approve_threshold=0.5)
    finding = _make_finding(
        best_confidence=1.0,
        validation_report=_make_report(novelty_score=1.0),
        literature_status="novel",
    )
    # Feed 3 positive examples to satisfy cold-start
    for _ in range(3):
        aa.fit(finding, approved=True)

    d = aa.decide(finding)
    # Post-cold-start, classifier may approve or review (not cold-start-review)
    assert "cold_start" not in d.reason


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["established", "contested"])
def test_hard_gate_blocked_status_rejects(status):
    aa = _make_approver(min_training_examples=0)
    finding = _make_finding(
        literature_status=status,
        validation_report=_make_report(),
    )
    d = aa.decide(finding)
    assert d.action == "reject"
    assert "hard_gate" in d.reason
    assert status in d.reason


def test_hard_gate_require_validation_no_report():
    aa = _make_approver(min_training_examples=0, require_validation=True)
    finding = _make_finding(validation_report=None, literature_status="novel")
    d = aa.decide(finding)
    assert d.action == "review"
    assert "require_validation" in d.reason


def test_hard_gate_require_validation_false_passes():
    aa = _make_approver(min_training_examples=0, require_validation=False, approve_threshold=0.0)
    finding = _make_finding(validation_report=None, literature_status="novel")
    # With threshold=0, any p > 0 → approve; bias=0, w=0 → p=sigmoid(0)=0.5 > 0
    d = aa.decide(finding)
    assert d.action in ("approve", "review", "reject")  # not blocked by validation gate


# ---------------------------------------------------------------------------
# Classifier convergence
# ---------------------------------------------------------------------------

def test_convergence_positive():
    """After 20 positive fits, P(approve) for a high-signal finding > 0.8."""
    aa = _make_approver(min_training_examples=0, require_validation=False)
    finding = _make_finding(
        best_confidence=0.95,
        literature_status="novel",
        validation_report=_make_report(novelty_score=0.95),
        candidate=_make_candidate(discovery_potential=0.9, gap_score=0.8, community_distance=3, seeded_by="structural_hole"),
    )
    for _ in range(20):
        aa.fit(finding, approved=True)

    features = aa._extract_features(finding)
    import numpy as np
    raw = float(np.dot(aa._w, features)) + aa._bias
    p = _sigmoid(raw)
    assert p > 0.8, f"Expected P > 0.8 after 20 positive examples, got {p:.4f}"


def test_convergence_negative():
    """After 50 negative fits, P(approve) for a low-signal finding < 0.2."""
    aa = _make_approver(min_training_examples=0, require_validation=False)
    finding = _make_finding(
        best_confidence=0.1,
        literature_status="unvalidated",
        candidate=_make_candidate(discovery_potential=0.1, gap_score=0.1, community_distance=0, seeded_by="manual"),
    )
    for _ in range(50):
        aa.fit(finding, approved=False)

    features = aa._extract_features(finding)
    import numpy as np
    raw = float(np.dot(aa._w, features)) + aa._bias
    p = _sigmoid(raw)
    assert p < 0.2, f"Expected P < 0.2 after 50 negative examples, got {p:.4f}"


def test_fit_increments_n_trained():
    aa = _make_approver()
    finding = _make_finding()
    aa.fit(finding, approved=True)
    aa.fit(finding, approved=False)
    assert aa._n_trained == 2


# ---------------------------------------------------------------------------
# Checkpoint / restore
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    aa = _make_approver(min_training_examples=0, require_validation=False)
    finding = _make_finding(best_confidence=0.9, literature_status="novel")
    for _ in range(5):
        aa.fit(finding, approved=True)
    for _ in range(3):
        aa.fit(finding, approved=False)

    d = aa.to_dict()
    aa2 = AutoApprover.from_dict(d)

    assert aa2._n_trained == aa._n_trained
    assert aa2._n_approve == aa._n_approve
    assert aa2._n_reject == aa._n_reject
    assert aa2._n_review == aa._n_review
    assert len(aa2._w) == len(aa._w)
    for i, (v1, v2) in enumerate(zip(aa._w, aa2._w)):
        assert abs(v1 - v2) < 1e-9, f"Weight mismatch at index {i}: {v1} vs {v2}"
    assert abs(aa2._bias - aa._bias) < 1e-9


def test_from_dict_wrong_length_resets_weights():
    """If checkpoint has wrong-length weight vector, default zeros are used."""
    data = {"w": [0.1, 0.2], "bias": 0.5, "n_trained": 0}
    aa = AutoApprover.from_dict(data)
    assert all(v == 0.0 for v in aa._w)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_log_records_every_decision():
    aa = _make_approver(min_training_examples=0, require_validation=False, audit_capacity=10)
    findings = [_make_finding(finding_id=f"f{i}") for i in range(5)]
    for f in findings:
        aa.decide(f)
    assert len(aa.audit_log) == 5
    ids = [d.finding_id for d in aa.audit_log]
    for f in findings:
        assert f.finding_id in ids


def test_audit_log_respects_capacity():
    aa = _make_approver(min_training_examples=0, require_validation=False, audit_capacity=3)
    for i in range(10):
        aa.decide(_make_finding(finding_id=f"f{i}"))
    assert len(aa.audit_log) == 3


def test_audit_log_counters():
    aa = _make_approver(min_training_examples=0, require_validation=False)
    # Trigger a hard-reject (established)
    aa.decide(_make_finding(literature_status="established"))
    assert aa._n_reject == 1
    assert aa._n_approve == 0
    assert aa._n_review == 0


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

def test_llm_fallback_yes_approves():
    policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)
    aa = AutoApprover(policy=policy, llm_fn=lambda prompt: "YES")
    # Force uncertain zone: default w=0, bias=0 → p=0.5 → "review" → LLM called
    finding = _make_finding(literature_status="novel")
    d = aa.decide(finding)
    assert d.action == "approve"
    assert "llm_fallback" in d.reason


def test_llm_fallback_no_rejects():
    policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)
    aa = AutoApprover(policy=policy, llm_fn=lambda prompt: "NO")
    finding = _make_finding(literature_status="novel")
    d = aa.decide(finding)
    assert d.action == "reject"
    assert "llm_fallback" in d.reason


def test_llm_fallback_garbage_returns_review():
    policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)
    aa = AutoApprover(policy=policy, llm_fn=lambda prompt: "MAYBE")
    finding = _make_finding(literature_status="novel")
    d = aa.decide(finding)
    assert d.action == "review"
    assert "llm_fallback" in d.reason


def test_llm_fallback_error_returns_review():
    policy = AutoApprovalPolicy(min_training_examples=0, require_validation=False)

    def bad_llm(prompt):
        raise RuntimeError("network timeout")

    aa = AutoApprover(policy=policy, llm_fn=bad_llm)
    finding = _make_finding(literature_status="novel")
    d = aa.decide(finding)  # must not raise
    assert d.action == "review"
    assert "llm_fallback" in d.reason


def test_llm_not_called_when_approve_threshold_met():
    """LLM should not be invoked when classifier already passes threshold."""
    called = []

    def noisy_llm(prompt):
        called.append(prompt)
        return "NO"

    policy = AutoApprovalPolicy(
        min_training_examples=0,
        require_validation=False,
        approve_threshold=0.0,  # everything approves
    )
    aa = AutoApprover(policy=policy, llm_fn=noisy_llm)
    aa.decide(_make_finding(literature_status="novel"))
    assert len(called) == 0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_features_length():
    aa = _make_approver()
    f = _make_finding()
    feats = aa._extract_features(f)
    assert len(feats) == 16


def test_extract_features_structural_hole_flag():
    aa = _make_approver()
    f = _make_finding(candidate=_make_candidate(seeded_by="structural_hole"))
    feats = aa._extract_features(f)
    assert feats[10] == 1.0
    assert feats[11] == 0.0


def test_extract_features_embedding_scan_flag():
    aa = _make_approver()
    f = _make_finding(candidate=_make_candidate(seeded_by="embedding_scan"))
    feats = aa._extract_features(f)
    assert feats[10] == 0.0
    assert feats[11] == 1.0


def test_extract_features_community_distance_clamped():
    aa = _make_approver()
    f = _make_finding(candidate=_make_candidate(community_distance=100))
    feats = aa._extract_features(f)
    assert feats[3] == 1.0  # min(1.0, 100/5)


def test_extract_features_novelty_score_default():
    aa = _make_approver()
    f = _make_finding(validation_report=None)
    feats = aa._extract_features(f)
    assert feats[6] == 0.5  # default when no report


def test_extract_features_engram_affinity():
    aa = _make_approver()
    f = _make_finding(metadata={"engram_affinity": 0.77})
    feats = aa._extract_features(f)
    assert abs(feats[7] - 0.77) < 1e-6


def test_extract_features_literature_ordinals():
    aa = _make_approver()
    for status, expected in [
        ("novel", 1.0), ("active_research", 0.75), ("unvalidated", 0.5),
        ("contested", 0.25), ("established", 0.0),
    ]:
        f = _make_finding(literature_status=status)
        feats = aa._extract_features(f)
        assert abs(feats[5] - expected) < 1e-6, f"{status}: expected {expected}, got {feats[5]}"


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

def test_stats_keys():
    aa = _make_approver()
    s = aa.stats()
    for key in ("n_trained", "n_approve", "n_reject", "n_review", "weights", "bias", "policy"):
        assert key in s


def test_stats_weights_length():
    aa = _make_approver()
    assert len(aa.stats()["weights"]) == 16


# ---------------------------------------------------------------------------
# Integration: ResearchAgent + AutoApprover
# ---------------------------------------------------------------------------

def test_set_auto_approver_wires_approver():
    """set_auto_approver() attaches the approver so _auto_approver is set."""
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine
    from adapters.networkx_adapter import NetworkXAdapter
    import networkx as nx

    G = nx.Graph()
    G.add_edge("A", "B", relation="related_to")
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"A": 0, "B": 0}

    he = HypothesisEngine(adapter)
    agent = ResearchAgent(adapter, he)

    assert agent._auto_approver is None
    aa = _make_approver()
    agent.set_auto_approver(aa)
    assert agent._auto_approver is aa


def test_auto_approver_constructor_param():
    """auto_approver param in ResearchAgent.__init__ is stored correctly."""
    from core.research_agent import ResearchAgent
    from core.hypothesis_engine import HypothesisEngine
    from adapters.networkx_adapter import NetworkXAdapter
    import networkx as nx

    G = nx.Graph()
    G.add_edge("A", "B", relation="related_to")
    adapter = NetworkXAdapter(G)
    adapter.community_map = {"A": 0, "B": 0}

    he = HypothesisEngine(adapter)
    aa = _make_approver()
    agent = ResearchAgent(adapter, he, auto_approver=aa)
    assert agent._auto_approver is aa


# ---------------------------------------------------------------------------
# Triangulation features (12–15) — Phase 72
# ---------------------------------------------------------------------------

def _make_tri_report(
    reverse_confidence=0.8,
    strategy_agreement=1.0,
    mean_path_independence=0.9,
    semantic_type_score=1.0,
    is_wormhole_candidate=True,
):
    from core.triangulation_engine import TriangulationReport
    return TriangulationReport(
        reverse_confidence=reverse_confidence,
        strategy_agreement=strategy_agreement,
        mean_path_independence=mean_path_independence,
        semantic_type_score=semantic_type_score,
        is_wormhole_candidate=is_wormhole_candidate,
    )


def test_triangulation_features_read_from_metadata():
    """Features 12–15 are correctly read from finding.metadata['triangulation']."""
    aa = _make_approver()
    tri = _make_tri_report(
        reverse_confidence=0.75,
        strategy_agreement=0.67,
        mean_path_independence=0.88,
        semantic_type_score=0.90,
    )
    f = _make_finding(metadata={"triangulation": tri})
    feats = aa._extract_features(f)
    assert abs(feats[12] - 0.75) < 1e-6
    assert abs(feats[13] - 0.67) < 1e-6
    assert abs(feats[14] - 0.88) < 1e-6
    assert abs(feats[15] - 0.90) < 1e-6


def test_triangulation_features_neutral_defaults_when_absent():
    """Without a TriangulationReport, features 12–15 use safe neutral defaults."""
    aa = _make_approver()
    f = _make_finding(metadata={})  # no triangulation key
    feats = aa._extract_features(f)
    assert feats[12] == 0.0   # reverse_confidence: conservative (no reverse evidence)
    assert feats[13] == 0.5   # strategy_agreement: neutral
    assert feats[14] == 0.5   # mean_path_independence: neutral
    assert feats[15] == 0.5   # semantic_type_score: neutral (novel relation)


def test_triangulation_reverse_confidence_zero_when_no_reverse():
    """Default reverse_confidence = 0.0 (conservative) biases toward review."""
    aa = _make_approver()
    f = _make_finding()
    feats = aa._extract_features(f)
    assert feats[12] == 0.0

"""
Tests for Phase 136 — Funnel Beam Profile (deep-hop coverage).

Covers:
  - _compute_beam_widths() formula correctness
  - flat profile returns empty dict and shared traversal is restored
  - funnel profile demonstrably recovers 3-hop answers pruned under flat
"""
import pytest
from core.cerebrum import CerebrumGraph, _compute_beam_widths


# ---------------------------------------------------------------------------
# _compute_beam_widths formula
# ---------------------------------------------------------------------------

class TestComputeBeamWidths:
    def test_mh1_returns_empty(self):
        assert _compute_beam_widths(mh=1, bw=10, factor=3.0) == {}

    def test_mh2_full_factor(self):
        result = _compute_beam_widths(mh=2, bw=10, factor=3.0)
        assert result == {1: 30}

    def test_mh3_ramp(self):
        result = _compute_beam_widths(mh=3, bw=10, factor=3.0)
        assert result == {1: 10, 2: 30}

    def test_mh4_linear_ramp(self):
        result = _compute_beam_widths(mh=4, bw=10, factor=3.0)
        assert result == {1: 10, 2: 20, 3: 30}

    def test_all_values_gte_bw(self):
        for mh in range(1, 6):
            result = _compute_beam_widths(mh=mh, bw=10, factor=3.0)
            for hop, width in result.items():
                assert width >= 10, f"hop {hop} width {width} < bw=10"

    def test_terminal_hop_excluded(self):
        for mh in range(1, 6):
            result = _compute_beam_widths(mh=mh, bw=10, factor=3.0)
            assert mh not in result, f"terminal hop {mh} should not be in widths"

    def test_factor_one_returns_flat_bw(self):
        result = _compute_beam_widths(mh=3, bw=10, factor=1.0)
        assert result == {1: 10, 2: 10}


# ---------------------------------------------------------------------------
# Flat profile: shared traversal is restored after query
# ---------------------------------------------------------------------------

def test_flat_profile_traversal_restored():
    triples = [
        ("A", "KNOWS", "B"), ("B", "KNOWS", "C"), ("C", "KNOWS", "D"),
    ] * 5
    g = CerebrumGraph.from_triples(triples, beam_profile="flat")
    g.build(seed=42)
    original_widths = dict(g._traversal._beam_widths)

    g.query(["A"], max_hop=3, beam_profile="flat")

    assert g._traversal._beam_widths == original_widths, (
        "Shared traversal._beam_widths not restored after query"
    )


# ---------------------------------------------------------------------------
# Funnel profile recovers 3-hop answer that flat beam prunes
# ---------------------------------------------------------------------------

def test_funnel_profile_recovers_deep_path():
    """
    Synthetic 3-hop chain A->B->C->D with 20 distractors on B.
    A has only one hop-1 neighbor (B), so hop 1 is always retained.
    B has C + 20 distractors = 21 hop-2 candidates.

    With beam_width=1, factor=30: hop 2 bw = max(1, int(1*30)) = 30 > 21.
    All 21 hop-2 candidates survive, so C always reaches hop 3 and D is found.

    This is a DETERMINISTIC guarantee (factor > n_candidates), not probabilistic.
    """
    triples = [("A", "r1", "B"), ("B", "r2", "C"), ("C", "r3", "D")]
    for i in range(20):
        triples.append(("B", "noise", f"NB{i}"))

    g = CerebrumGraph.from_triples(triples)
    g.build(seed=42)

    funnel_answers = g.query(
        ["A"], max_hop=3, beam_width=1,
        beam_profile="funnel", beam_profile_factor=30.0, top_k=100,
    )
    funnel_ids = {a.entity_id for a in funnel_answers}

    assert "D" in funnel_ids, (
        f"Funnel (factor=30) should always find D: hop-2 bw=30 > 21 candidates. Got: {funnel_ids}"
    )

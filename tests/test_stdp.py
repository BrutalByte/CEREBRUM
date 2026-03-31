"""
Phase 13 — STDPDiscretizer test suite.

Tests cover:
  - LTP: pre-before-post potentiation
  - LTD: anti-causal direction depression
  - Exponential decay with Δt
  - Threshold emission (w_threshold, n_min)
  - Weight decay across calls
  - Repeated firings accumulate weight
  - reset() clears all state
  - weight() and count() accessors
  - No self-pairing
  - Out-of-window spikes are ignored
  - CAUSES edge fields (metadata, ttl, relation)
  - Custom relation label
  - Re-emission on every call once threshold crossed
  - Asymmetric A_plus / A_minus defaults (A_minus > A_plus prevents runaway)
"""
from __future__ import annotations

import math


from core.discretizer import STDPDiscretizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fire_sequence(disc: STDPDiscretizer, sources: list, gap: float = 0.05):
    """Fire sources one after another with `gap` seconds between each."""
    t = 1000.0  # fixed base timestamp so tests are deterministic
    events = []
    for src in sources:
        evts = disc.process(src, timestamp=t)
        events.extend(evts)
        t += gap
    return events


# ---------------------------------------------------------------------------
# TestLTP — pre fires before post → weight[(pre, post)] grows
# ---------------------------------------------------------------------------

class TestLTP:
    def test_weight_increases_after_pre_then_post(self):
        disc = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)
        w = disc.weight("A", "B")
        expected = 0.1 * math.exp(-0.1 / 0.2)
        assert abs(w - expected) < 1e-9

    def test_weight_grows_with_repeated_pre_post(self):
        disc = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        for i in range(5):
            disc.process("A", timestamp=float(i * 2))
            disc.process("B", timestamp=float(i * 2) + 0.1)
        w = disc.weight("A", "B")
        assert w > 0.1  # accumulated over 5 pairings

    def test_ltp_decays_with_larger_delta_t(self):
        disc = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.01)  # very close → large potentiation
        w_close = disc.weight("A", "B")

        disc2 = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        disc2.process("A", timestamp=0.0)
        disc2.process("B", timestamp=0.8)  # far → small potentiation
        w_far = disc2.weight("A", "B")

        assert w_close > w_far

    def test_no_weight_when_post_fires_first(self):
        disc = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        disc.process("B", timestamp=0.0)
        disc.process("A", timestamp=0.1)
        # A→B direction should have no LTP (A fired after B)
        assert disc.weight("A", "B") == 0.0

    def test_count_increments_per_pairing(self):
        disc = STDPDiscretizer(window_seconds=1.0, weight_decay=1.0)
        for i in range(3):
            disc.process("A", timestamp=float(i * 2))
            disc.process("B", timestamp=float(i * 2) + 0.05)
        assert disc.count("A", "B") == 3


# ---------------------------------------------------------------------------
# TestLTD — anti-causal direction is depressed
# ---------------------------------------------------------------------------

class TestLTD:
    def test_reverse_direction_depressed(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.1, A_minus=0.105,
            tau_plus=0.2, tau_minus=0.2, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)
        # B→A direction should have been depressed (A fired before B,
        # so B→A is anti-causal)
        w_reverse = disc.weight("B", "A")
        # LTD decrement: 0.105 * exp(-0.1 / 0.2) ≈ 0.0637, clamped to 0
        assert w_reverse == 0.0  # clamped — started at 0, can't go negative

    def test_ltd_reduces_pre_built_reverse_weight(self):
        """
        Build B→A weight first (B fires before A), then reverse order.
        The second pass should reduce the B→A weight via LTD.
        """
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.1, A_minus=0.105,
            tau_plus=0.2, tau_minus=0.2, weight_decay=1.0, n_min=1,
        )
        # Round 1: B → A  (builds B→A via LTP)
        disc.process("B", timestamp=0.0)
        disc.process("A", timestamp=0.1)
        w_before = disc.weight("B", "A")
        assert w_before > 0.0

        # Round 2: A → B  (builds A→B via LTP, also applies LTD to B→A)
        disc.process("A", timestamp=2.0)
        disc.process("B", timestamp=2.1)
        w_after = disc.weight("B", "A")
        assert w_after < w_before  # LTD reduced it


# ---------------------------------------------------------------------------
# TestThresholdEmission — CAUSES edges only when weight ≥ w_threshold AND count ≥ n_min
# ---------------------------------------------------------------------------

class TestThresholdEmission:
    def test_no_emission_below_count_threshold(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.5, tau_plus=0.05,
            w_threshold=0.01, n_min=5, weight_decay=1.0,
        )
        # Only 3 pairings — below n_min=5
        for i in range(3):
            evts = []
            evts += disc.process("A", timestamp=float(i * 2))
            evts += disc.process("B", timestamp=float(i * 2) + 0.02)
        assert disc.count("A", "B") == 3
        # Last call shouldn't have emitted
        last = disc.process("X", timestamp=100.0)  # unrelated spike
        assert all(e.source != "A" or e.target != "B" for e in last)

    def test_emission_after_sufficient_pairings(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.2, tau_plus=0.05,
            w_threshold=0.1, n_min=3, weight_decay=1.0,
        )
        all_events = []
        for i in range(5):
            all_events += disc.process("A", timestamp=float(i * 2))
            all_events += disc.process("B", timestamp=float(i * 2) + 0.02)
        causes = [e for e in all_events if e.source == "A" and e.target == "B"]
        assert len(causes) > 0

    def test_emission_uses_correct_relation(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.5, tau_plus=0.05,
            w_threshold=0.01, n_min=1, weight_decay=1.0,
            relation="DRIVES",
        )
        disc.process("A", timestamp=0.0)
        evts = disc.process("B", timestamp=0.02)
        causes = [e for e in evts if e.source == "A" and e.target == "B"]
        assert all(e.relation == "DRIVES" for e in causes)

    def test_emission_metadata_contains_weight_and_count(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.5, tau_plus=0.05,
            w_threshold=0.01, n_min=1, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        evts = disc.process("B", timestamp=0.02)
        causes = [e for e in evts if e.source == "A" and e.target == "B"]
        assert len(causes) == 1
        assert "causal_weight" in causes[0].metadata
        assert "event_count" in causes[0].metadata
        assert causes[0].metadata["event_count"] >= 1

    def test_ttl_propagated_to_events(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.5, tau_plus=0.05,
            w_threshold=0.01, n_min=1, weight_decay=1.0, ttl=30.0,
        )
        disc.process("A", timestamp=0.0)
        evts = disc.process("B", timestamp=0.02)
        causes = [e for e in evts if e.source == "A" and e.target == "B"]
        assert all(e.ttl == 30.0 for e in causes)


# ---------------------------------------------------------------------------
# TestWeightDecay — multiplicative forgetting
# ---------------------------------------------------------------------------

class TestWeightDecay:
    def test_weight_decays_each_call(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=0.5,
        )
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)  # weight set, then decayed once on next B call
        w_after_pair = disc.weight("A", "B")

        # Fire an unrelated spike — decay applied again
        disc.process("C", timestamp=5.0)
        w_after_decay = disc.weight("A", "B")
        assert w_after_decay < w_after_pair

    def test_no_decay_when_weight_decay_is_1(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)
        w = disc.weight("A", "B")
        disc.process("C", timestamp=5.0)
        assert disc.weight("A", "B") == w  # unchanged


# ---------------------------------------------------------------------------
# TestOutOfWindow — spikes outside window_seconds are ignored
# ---------------------------------------------------------------------------

class TestOutOfWindow:
    def test_spike_outside_window_not_paired(self):
        disc = STDPDiscretizer(
            window_seconds=0.5, A_plus=0.1, tau_plus=0.2, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=1.0)  # 1.0 > window_seconds=0.5
        assert disc.weight("A", "B") == 0.0
        assert disc.count("A", "B") == 0

    def test_spike_within_window_is_paired(self):
        disc = STDPDiscretizer(
            window_seconds=0.5, A_plus=0.1, tau_plus=0.2, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.4)  # 0.4 < 0.5 → within window
        assert disc.weight("A", "B") > 0.0


# ---------------------------------------------------------------------------
# TestSelfPairing — a source should never pair with itself
# ---------------------------------------------------------------------------

class TestSelfPairing:
    def test_no_self_pair(self):
        disc = STDPDiscretizer(window_seconds=1.0, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.process("A", timestamp=0.1)
        assert disc.weight("A", "A") == 0.0
        assert disc.count("A", "A") == 0


# ---------------------------------------------------------------------------
# TestReset — reset() clears all state
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_weights(self):
        disc = STDPDiscretizer(window_seconds=1.0, A_plus=0.1, tau_plus=0.2, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)
        assert disc.weight("A", "B") > 0.0
        disc.reset()
        assert disc.weight("A", "B") == 0.0

    def test_reset_clears_counts(self):
        disc = STDPDiscretizer(window_seconds=1.0, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.1)
        disc.reset()
        assert disc.count("A", "B") == 0

    def test_reset_clears_fire_times_so_no_pairing(self):
        disc = STDPDiscretizer(window_seconds=1.0, weight_decay=1.0)
        disc.process("A", timestamp=0.0)
        disc.reset()
        disc.process("B", timestamp=0.1)  # A's fire time was cleared
        assert disc.weight("A", "B") == 0.0


# ---------------------------------------------------------------------------
# TestCausalDirectionality — A→B and B→A are independent
# ---------------------------------------------------------------------------

class TestCausalDirectionality:
    def test_ab_and_ba_independent(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.2, tau_plus=0.1, weight_decay=1.0,
        )
        # A then B → potentiate A→B
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.05)
        w_ab = disc.weight("A", "B")
        w_ba = disc.weight("B", "A")
        assert w_ab > 0.0
        assert w_ba == 0.0  # B→A not potentiated (B came after A)

    def test_bidirectional_pairings_tracked_separately(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.2, tau_plus=0.1, weight_decay=1.0,
        )
        # A before B
        disc.process("A", timestamp=0.0)
        disc.process("B", timestamp=0.05)
        # B before A
        disc.process("B", timestamp=2.0)
        disc.process("A", timestamp=2.05)

        # Both directions should have weight
        assert disc.weight("A", "B") > 0.0
        assert disc.weight("B", "A") > 0.0
        # A→B should be higher (one clean LTP pairing)
        # but B→A got an LTP pairing too in round 2
        assert disc.count("A", "B") >= 1
        assert disc.count("B", "A") >= 1


# ---------------------------------------------------------------------------
# TestCustomMetadata — metadata merged into events
# ---------------------------------------------------------------------------

class TestCustomMetadata:
    def test_metadata_merged(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.5, tau_plus=0.05,
            w_threshold=0.01, n_min=1, weight_decay=1.0,
        )
        disc.process("A", timestamp=0.0)
        evts = disc.process("B", timestamp=0.02, metadata={"sensor": "cam1"})
        causes = [e for e in evts if e.source == "A" and e.target == "B"]
        assert len(causes) == 1
        assert causes[0].metadata.get("sensor") == "cam1"
        assert "causal_weight" in causes[0].metadata


# ---------------------------------------------------------------------------
# TestAccumulation — many pairings build substantial weight
# ---------------------------------------------------------------------------

class TestAccumulation:
    def test_many_pairings_exceed_threshold(self):
        disc = STDPDiscretizer(
            window_seconds=1.0, A_plus=0.05, tau_plus=0.2,
            w_threshold=0.3, n_min=10, weight_decay=1.0,
        )
        all_events = []
        for i in range(15):
            all_events += disc.process("A", timestamp=float(i * 2))
            all_events += disc.process("B", timestamp=float(i * 2) + 0.1)
        causes = [e for e in all_events if e.source == "A" and e.target == "B"]
        assert len(causes) > 0
        assert disc.count("A", "B") == 15
        assert disc.weight("A", "B") > 0.3

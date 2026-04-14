"""
DiscoveryCalibrator — Population-Level Discovery Calibration (Phase 73, Batch A).

Tracks the per-community discovery rate over scan cycles and reweights candidate
scoring so underrepresented communities receive proportionally more attention.

Problem:  ANN scan and structural-hole detection find candidates proportionally
to graph density.  Dense, well-connected communities produce many candidates
while sparse peripheral communities — which are sparse precisely because they
are understudied — receive almost none.  The most valuable novel connections
tend to be in the regions the agent visits least.

Solution: maintain a simple sliding-window discovery rate per community and
apply an inverse-rate multiplier to ``_score_discovery_potential()``.  A
community that has yielded zero findings in N scans gets a high weight; one
that consistently produces findings gets a lower weight, steering exploration
toward genuine gaps.

Multiplier formula
------------------
  rate[c]   = discoveries[c] / max(1, scans[c])
  weight[c] = 1 / (rate[c] + epsilon)              # inverse rate
  normalised = weight[c] / mean(all weights)        # mean-centred at 1.0
  clamped    = clip(normalised, min_weight, max_weight)

A community that has never produced a finding gets the maximum weight boost
(max_weight, default 5.0).  A community saturated with findings gets
min_weight (default 0.2) — it is still explored, just at lower priority.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Set

logger = logging.getLogger("cerebrum.discovery_calibrator")


class DiscoveryCalibrator:
    """
    Online per-community discovery rate tracker and candidate reweighter.

    Parameters
    ----------
    min_weight
        Floor on the community sampling multiplier (prevents starvation of
        communities that have high discovery rates — still visited, just less).
    max_weight
        Ceiling on the multiplier (prevents runaway oversampling of completely
        dark communities on the first few scans).
    epsilon
        Smoothing constant added to the discovery rate before inversion — prevents
        division by zero for communities with exactly zero discoveries.
    window
        Number of recent scan cycles to consider.  Older counts decay by a factor
        of (1 - 1/window) each scan cycle (exponential moving average).
    """

    def __init__(
        self,
        min_weight: float = 0.2,
        max_weight: float = 5.0,
        epsilon: float = 0.05,
        window: int = 20,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.epsilon = epsilon
        self.window = window

        # EMA-smoothed counts per community
        self._discovery_ema: Dict[int, float] = {}  # community_id → smoothed discoveries
        self._scan_ema: Dict[int, float] = {}        # community_id → smoothed scans

        self._total_scans: int = 0
        self._total_discoveries: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording API (called by ResearchAgent)
    # ------------------------------------------------------------------

    def record_scan(self, community_ids: Set[int]) -> None:
        """
        Called once per scan cycle with the set of community IDs that were
        represented among the candidates evaluated.  Updates EMA scan counts.
        """
        decay = 1.0 - 1.0 / max(1, self.window)
        with self._lock:
            self._total_scans += 1
            # Decay all existing entries
            for cid in list(self._scan_ema):
                self._scan_ema[cid] *= decay
            # Increment for communities seen this scan
            for cid in community_ids:
                self._scan_ema[cid] = self._scan_ema.get(cid, 0.0) * decay + 1.0

    def record_discovery(self, community_id: int) -> None:
        """
        Called when a finding is successfully created for a candidate whose
        source belongs to ``community_id``.
        """
        decay = 1.0 - 1.0 / max(1, self.window)
        with self._lock:
            self._total_discoveries += 1
            prev = self._discovery_ema.get(community_id, 0.0)
            self._discovery_ema[community_id] = prev * decay + 1.0

    # ------------------------------------------------------------------
    # Weighting API
    # ------------------------------------------------------------------

    def get_weight(self, community_id: int) -> float:
        """
        Return the sampling multiplier for ``community_id`` in
        [``min_weight``, ``max_weight``].

        Formula: ``global_rate / community_rate``

        - ``global_rate`` = total_discoveries_ema / total_scans_ema
        - ``community_rate`` = community_discoveries_ema / community_scans_ema

        A community whose discovery rate is below the global average gets a
        multiplier > 1.0 (boost toward it).  One above average gets < 1.0.

        Cold-start rules:
          - Community never scanned → ``max_weight`` (completely unexplored)
          - No global data yet → 1.0 (neutral)
        """
        with self._lock:
            scans = self._scan_ema.get(community_id, 0.0)
            if scans < 0.5:
                return self.max_weight  # never scanned → max boost

            # Global rate across all communities
            total_scans = sum(self._scan_ema.values())
            total_disc = sum(self._discovery_ema.values())

            if total_scans < 0.5:
                return 1.0  # no scan data at all

            if total_disc < 0.5:
                # No recent discoveries anywhere — treat whole graph as unexplored
                return self.max_weight

            global_rate = total_disc / total_scans

            # Community rate
            discoveries = self._discovery_ema.get(community_id, 0.0)
            community_rate = discoveries / scans

            # Ratio: how much below (or above) the global rate is this community?
            weight = global_rate / (community_rate + self.epsilon)

            return float(min(self.max_weight, max(self.min_weight, weight)))

    def stats(self) -> dict:
        """Return a human-readable summary for monitoring."""
        with self._lock:
            community_stats = {}
            all_cids = set(self._scan_ema) | set(self._discovery_ema)
            for cid in sorted(all_cids):
                scans = self._scan_ema.get(cid, 0.0)
                discoveries = self._discovery_ema.get(cid, 0.0)
                rate = discoveries / max(0.01, scans)
                community_stats[cid] = {
                    "scan_ema": round(scans, 3),
                    "discovery_ema": round(discoveries, 3),
                    "rate": round(rate, 4),
                    "weight": self.get_weight(cid),
                }
            return {
                "total_scans": self._total_scans,
                "total_discoveries": self._total_discoveries,
                "communities": community_stats,
            }


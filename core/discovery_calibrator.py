"""
DiscoveryCalibrator â€” Population-Level Discovery Calibration (Phase 73, Batch A).

Tracks the per-community discovery rate over scan cycles and reweights candidate
scoring so underrepresented communities receive proportionally more attention.

Problem:  ANN scan and structural-hole detection find candidates proportionally
to graph density.  Dense, well-connected communities produce many candidates
while sparse peripheral communities â€” which are sparse precisely because they
are understudied â€” receive almost none.  The most valuable novel connections
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
min_weight (default 0.2) â€” it is still explored, just at lower priority.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import Counter
from typing import Counter, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter

logger = logging.getLogger("cerebrum.discovery_calibrator")


class DiscoveryCalibrator:
    """
    Online per-community discovery rate tracker and candidate reweighter.

    Parameters
    ----------
    min_weight
        Floor on the community sampling multiplier (prevents starvation of
        communities that have high discovery rates â€” still visited, just less).
    max_weight
        Ceiling on the multiplier (prevents runaway oversampling of completely
        dark communities on the first few scans).
    epsilon
        Smoothing constant added to the discovery rate before inversion â€” prevents
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
        curiosity_alpha: float = 0.3,
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.epsilon = epsilon
        self.window = window
        # Phase 215-D: blend ratio for information-theoretic curiosity signal [0,1].
        # 0.0 = pure inverse-rate (original behaviour), 1.0 = pure entropy curiosity.
        self.curiosity_alpha = curiosity_alpha

        # EMA-smoothed counts per community
        self._discovery_ema: Dict[int, float] = {}  # community_id â†’ smoothed discoveries
        self._scan_ema: Dict[int, float] = {}        # community_id â†’ smoothed scans

        self._total_scans: int = 0
        self._total_discoveries: int = 0
        self._lock = threading.Lock()
        # Phase 215-D: cached entropy-based curiosity scores per community
        self._curiosity_cache: Dict[int, float] = {}

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
          - Community never scanned â†’ ``max_weight`` (completely unexplored)
          - No global data yet â†’ 1.0 (neutral)
        """
        with self._lock:
            return self._get_weight_locked(community_id)

    def _get_weight_locked(self, community_id: int) -> float:
        """Internal thread-unsafe implementation of get_weight."""
        scans = self._scan_ema.get(community_id, 0.0)
        if scans < 0.5:
            return self.max_weight  # never scanned â†’ max boost

        # Global rate across all communities
        total_scans = sum(self._scan_ema.values())
        total_disc = sum(self._discovery_ema.values())

        if total_scans < 0.5:
            return 1.0  # no scan data at all

        if total_disc < 0.5:
            # No recent discoveries anywhere â€” treat whole graph as unexplored
            return self.max_weight

        global_rate = total_disc / total_scans

        # Community rate
        discoveries = self._discovery_ema.get(community_id, 0.0)
        community_rate = discoveries / scans

        # Ratio: how much below (or above) the global rate is this community?
        inv_rate_weight = global_rate / (community_rate + self.epsilon)

        # Phase 215-D: blend with information-theoretic curiosity (entropy signal)
        alpha = self.curiosity_alpha
        if alpha > 0.0 and community_id in self._curiosity_cache:
            curiosity = self._curiosity_cache[community_id]
            weight = (1.0 - alpha) * inv_rate_weight + alpha * curiosity * self.max_weight
        else:
            weight = inv_rate_weight

        return float(min(self.max_weight, max(self.min_weight, weight)))

    # ------------------------------------------------------------------
    # Phase 215-D: Information-Gain Curiosity API
    # ------------------------------------------------------------------

    def compute_curiosity(self, community_id: int, adapter: "GraphAdapter") -> float:
        """
        Compute an information-theoretic curiosity score for a community [0, 1].

        Estimated as the normalised Shannon entropy of the relation-type distribution
        within the community.  A uniform distribution (max entropy) means many
        different relation types are present â€” the community is structurally rich and
        worth exploring.  A peaked distribution (low entropy) means one relation type
        dominates â€” less exploratory value.

        High entropy â†’ high curiosity (more to discover).
        Low entropy  â†’ low curiosity  (already well characterised).
        """
        try:
            # Collect edges from community members via adapter's community_map
            cmap = getattr(adapter, "community_map", None)
            if cmap is None:
                return 0.5  # neutral if no community map
            members = [eid for eid, cid in cmap.items() if cid == community_id]
            if not members:
                return 0.5
            # Sample up to 200 members to keep this O(1) amortised
            sample = members[:200]
            rel_counts: Counter = Counter()
            for eid in sample:
                for edge in adapter.get_neighbors(eid, max_neighbors=20):
                    rel_counts[edge.relation_type] += 1
            if not rel_counts:
                return 0.5
            total = sum(rel_counts.values())
            probs = [c / total for c in rel_counts.values()]
            H = -sum(p * math.log2(p + 1e-12) for p in probs)
            H_max = math.log2(len(rel_counts) + 1)
            return float(H / (H_max + 1e-9))
        except Exception:
            return 0.5

    def set_curiosity_alpha(self, alpha: float) -> None:
        """Phase 223-C: Dynamically adjust curiosity blend ratio."""
        self.curiosity_alpha = max(0.0, min(1.0, alpha))

    def refresh_curiosity(self, adapter: "GraphAdapter", community_ids: Set[int]) -> None:
        """
        Recompute curiosity scores for the given communities and cache them.
        Called from AutonomousDiscoveryLoop between scan cycles (non-blocking intent).
        """
        for cid in community_ids:
            score = self.compute_curiosity(cid, adapter)
            with self._lock:
                self._curiosity_cache[cid] = score

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
                    "weight": self._get_weight_locked(cid),
                }
            return {
                "total_scans": self._total_scans,
                "total_discoveries": self._total_discoveries,
                "communities": community_stats,
            }

    def save(self, path: str) -> None:
        """Phase 218-B: Persist EMA state to JSON for cross-session continuity."""
        import json
        with self._lock:
            data = {
                "discovery_ema": {str(k): v for k, v in self._discovery_ema.items()},
                "scan_ema": {str(k): v for k, v in self._scan_ema.items()},
                "total_scans": self._total_scans,
                "total_discoveries": self._total_discoveries,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        decay_factor: float = 0.8,
        **kwargs,
    ) -> "DiscoveryCalibrator":
        """Phase 218-B: Load persisted EMA state with regularization decay.

        Older sessions are trusted less: all EMA values are multiplied by
        ``decay_factor`` so long-running communities don't lock in stale priors.
        """
        import json
        obj = cls(**kwargs)
        try:
            with open(path) as f:
                data = json.load(f)
            with obj._lock:
                obj._discovery_ema = {int(k): v * decay_factor
                                      for k, v in data.get("discovery_ema", {}).items()}
                obj._scan_ema = {int(k): v * decay_factor
                                 for k, v in data.get("scan_ema", {}).items()}
                obj._total_scans = data.get("total_scans", 0)
                obj._total_discoveries = data.get("total_discoveries", 0)
        except (FileNotFoundError, KeyError, ValueError):
            pass
        return obj


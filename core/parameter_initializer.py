"""
Phase 205: ParameterInitializer — principled hyperparameter defaults from graph statistics.

Replaces black-box tuning as the *required* starting point.  Given a loaded KB,
the class analytically derives parameter values that work out-of-the-box for any
graph structure.  The tuner (cerebrum_tuner.py) becomes an optional refinement step.

All formulas are grounded in data-science TTPs (see docs/PERFORMANCE_TUNING.md):
  trb_factor   — log-normalized category-count scaling (TFIDF analogue)
  gamma        — Zipf normalization of fan_out distribution
  beta         — power-law null hypothesis; regime-specific constant
  r2_boost     — path corroboration scaled by path diversity
  fhrb_factor  — first-hop ambiguity scaled by mean degree
  idf_weight   — IDF hub penalty (standard formula, CV-scaled)
  vote_weight  — community vote reliability via modularity Q
  branch_bonus — naïve-Bayes independent-evidence combination
  beam_width   — fixed at 12; diminishing returns plateau proven by fANOVA

Calibration basis: Phase 204 tuner validation on MetaQA 3-hop (60.36% H@1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph_profiler import QueryProfile
    from core.relation_boost_deriver import RelationBoostDeriver

# ---------------------------------------------------------------------------
# Universal constants (graph-invariant — validated across all MetaQA runs)
# ---------------------------------------------------------------------------

BEAM_WIDTH: int = 12          # fANOVA importance ~0.1%; plateau at 8–12

IDF_SCALE_C: float = 0.0102   # degree_cv × IDF_SCALE_C → idf_weight
                               # MetaQA: cv=5.68, idf=0.058 → C=0.0102

VOTE_BASE: float   = 0.72     # vote_weight lower bound (Q=0 → community uninformative)
VOTE_Q_SCALE: float = 0.15    # sensitivity to modularity: 0.72 at Q=0, 0.87 at Q=1

# ---------------------------------------------------------------------------
# Regime-specific constants
# ---------------------------------------------------------------------------
# Hub-homogeneous calibrated from MetaQA Phase 204.
# Typed-heterogeneous and mixed are analytical estimates; refine when
# multi-KB validation data becomes available.

_TRB_C = {
    "hub_homogeneous":    9.33,   # 21.486 / log(9+1) — Phase 204 MetaQA
    "typed_heterogeneous": 6.00,  # lower: typed KBs have fewer competing relations
    "mixed":               7.50,
}

_GAMMA_C = {
    "hub_homogeneous":    3.73,   # 8.732 / (max_fo/harmonic_fo=2.342) — Phase 204
    "typed_heterogeneous": 6.00,  # higher: typed KBs have wider fan_out spread
    "mixed":               4.50,
}

_BETA = {
    "hub_homogeneous":    2.0,    # Phase 204 best: 2.0846; amplifies hub relations
    "typed_heterogeneous": 1.0,   # linear — typed KBs have uniform fan_out
    "mixed":               1.5,
}

_BRANCH_BONUS = {
    "hub_homogeneous":    0.48,   # Phase 204 best: 0.482
    "typed_heterogeneous": 0.17,  # naïve-Bayes log-odds of 2 independent paths agreeing
    "mixed":               0.30,
}

_R2_C = {
    "hub_homogeneous":    13.50,  # (8.185 - 1.5) / log(5.768/9+1) — Phase 204
    "typed_heterogeneous":  8.00,
    "mixed":               10.00,
}

_FHRB_C = {
    "hub_homogeneous":    1.18,   # (3.260 - 1.0) / log(5.768+1) — Phase 204
    "typed_heterogeneous": 0.70,
    "mixed":               0.90,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InitialParams:
    """Analytically-derived starting point for all CEREBRUM hyperparameters."""
    trb_factor:   float
    gamma:        float
    beta:         float
    r2_boost:     float
    fhrb_factor:  float
    idf_weight:   float
    vote_weight:  float
    branch_bonus: float
    beam_width:   int
    effective_regime: str = "unknown"   # regime used for constant selection

    def as_dict(self) -> dict:
        return {
            "trb_factor":   self.trb_factor,
            "gamma":        self.gamma,
            "beta":         self.beta,
            "r2_boost":     self.r2_boost,
            "fhrb_factor":  self.fhrb_factor,
            "idf_weight":   self.idf_weight,
            "vote_weight":  self.vote_weight,
            "branch_bonus": self.branch_bonus,
            "beam_width":   self.beam_width,
        }

    def summary(self) -> str:
        lines = [f"ParameterInitializer defaults (regime={self.effective_regime}):"]
        for k, v in self.as_dict().items():
            lines.append(f"  {k:14s} = {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ParameterInitializer
# ---------------------------------------------------------------------------

class ParameterInitializer:
    """
    Derives principled hyperparameter defaults from graph statistics.

    Usage::

        from core.parameter_initializer import ParameterInitializer
        params = ParameterInitializer.compute(profile, deriver)
        print(params.summary())
    """

    @staticmethod
    def compute(
        profile: "QueryProfile",
        deriver: "RelationBoostDeriver",
        modularity_Q: float = 0.5,
    ) -> InitialParams:
        """
        Compute principled defaults.

        Parameters
        ----------
        profile      : QueryProfile from GraphProfiler.profile()
        deriver      : RelationBoostDeriver (built from KB file or triples)
        modularity_Q : Newman-Girvan modularity of the community partition.
                       If unknown, 0.5 gives a safe mid-range vote_weight.
        """
        # GraphProfiler's regime is primarily for routing decisions (hop_expand,
        # trb_auto).  For parameter scaling, degree_cv is the dominant signal for
        # hub behavior: if CV > 3.0, the degree distribution is heavy-tailed enough
        # that hub_homogeneous constants apply even when min_rel_coverage flags a few
        # rare metadata relations as "typed" (e.g. MetaQA has_imdb_rating, in_language).
        regime = profile.regime
        if profile.degree_cv > 3.0 and regime == "typed_heterogeneous":
            regime = "hub_homogeneous"

        n_rel  = max(profile.n_relation_types, 1)
        n_nodes = max(profile.n_nodes, 1)
        n_edges = profile.n_edges

        # Undirected mean degree: sum_degrees = 2*n_edges for undirected graph.
        mean_degree = (2.0 * n_edges / n_nodes) if n_edges > 0 else 1.0

        max_fo, _mean_fo, harmonic_fo, n_rel_deriver = deriver.fan_out_stats()
        if n_rel_deriver > 0:
            n_rel = n_rel_deriver  # prefer deriver count (more precise)

        # ------------------------------------------------------------------
        # trb_factor  = C × log(n_rel + 1)
        # ------------------------------------------------------------------
        trb_factor = _TRB_C[regime] * math.log(n_rel + 1)

        # ------------------------------------------------------------------
        # gamma  = C × (max_fan_out / harmonic_mean_fan_out)
        # ------------------------------------------------------------------
        gamma = _GAMMA_C[regime] * (max_fo / harmonic_fo)

        # ------------------------------------------------------------------
        # beta — regime constant
        # ------------------------------------------------------------------
        beta = _BETA[regime]

        # ------------------------------------------------------------------
        # r2_boost  = 1.5 + C × log(mean_deg / n_rel + 1)
        # ------------------------------------------------------------------
        r2_boost = 1.5 + _R2_C[regime] * math.log(mean_degree / n_rel + 1)
        r2_boost = max(1.5, r2_boost)

        # ------------------------------------------------------------------
        # fhrb_factor  = 1.0 + C × log(mean_deg + 1)
        # ------------------------------------------------------------------
        fhrb_factor = 1.0 + _FHRB_C[regime] * math.log(mean_degree + 1)
        fhrb_factor = max(1.0, fhrb_factor)

        # ------------------------------------------------------------------
        # idf_weight  = max(0.01, IDF_SCALE_C × degree_cv)
        # ------------------------------------------------------------------
        idf_weight = max(0.01, IDF_SCALE_C * profile.degree_cv)

        # ------------------------------------------------------------------
        # vote_weight  = VOTE_BASE + VOTE_Q_SCALE × Q
        # ------------------------------------------------------------------
        q_clamped   = max(0.0, min(1.0, modularity_Q))
        vote_weight = VOTE_BASE + VOTE_Q_SCALE * q_clamped

        # ------------------------------------------------------------------
        # branch_bonus — regime constant
        # ------------------------------------------------------------------
        branch_bonus = _BRANCH_BONUS[regime]

        return InitialParams(
            trb_factor       = round(trb_factor,  4),
            gamma            = round(gamma,       4),
            beta             = round(beta,        4),
            r2_boost         = round(r2_boost,    4),
            fhrb_factor      = round(fhrb_factor, 4),
            idf_weight       = round(idf_weight,  4),
            vote_weight      = round(vote_weight, 4),
            branch_bonus     = round(branch_bonus, 4),
            beam_width       = BEAM_WIDTH,
            effective_regime = regime,
        )

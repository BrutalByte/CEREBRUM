"""
Phase 205: ParameterInitializer — principled hyperparameter defaults from graph statistics.

Replaces black-box tuning as the *required* starting point.  Given a loaded KB,
the class analytically derives parameter values that work out-of-the-box for any
graph structure.  The tuner (cerebrum_tuner.py) becomes an optional refinement step.

All formulas are grounded in data-science TTPs (see docs/PERFORMANCE_TUNING.md):
  trb_factor   — log-normalized category-count scaling (TFIDF analogue)
  gamma        — universal BOOST_SCALE formula: score*(1+boost) targets constant
                 mean amplification across all KBs (see BOOST_SCALE constant below)
  beta         — power-law shape; regime-specific constant
  r2_boost     — path corroboration scaled by path diversity
  fhrb_factor  — first-hop ambiguity scaled by mean degree
  idf_weight   — IDF hub penalty (standard formula, CV-scaled)
  vote_weight  — community vote reliability via modularity Q
  branch_bonus — naïve-Bayes independent-evidence combination
  beam_width   — fixed at 12; diminishing returns plateau proven by fANOVA

Calibration basis: Phase 204 tuner validation on MetaQA 3-hop (60.36% H@1).
Multi-KB survey (Phase 205): MetaQA, Hetionet, synthetic typed, synthetic hub.
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

# gamma — regime-specific BOOST_SCALE formula
# SDRB scoring: score *= (1 + boost), where boost(r) = gamma * fan_out(r)^beta.
# Formula: gamma = BOOST_SCALE[regime] / mean_fan_out^beta
#
# BOOST_SCALE is NOT universal — it varies significantly by regime because
# mean_fo^beta grows super-linearly with fan_out as beta increases:
#
#   hub_homogeneous (MetaQA Phase 204):
#     gamma=8.73, mean_fo=1.55, beta=2.085 → BOOST_SCALE=21.79
#
#   typed_heterogeneous (Hetionet Phase 206 tuner validation):
#     gamma=5.9183, mean_fo=9.46, beta=1.8778
#     → BOOST_SCALE = 5.9183 * 9.46^1.8778 = 5.9183 * 68.2 = 403.6
#     Note: with beta=1.88 on Hetionet's large fan_out, mean_fo^beta=68.2
#     so BOOST_SCALE must be much larger to produce a usable gamma.
#
BOOST_SCALE: float = 21.79    # hub_homogeneous — Phase 204 MetaQA validation
GAMMA_MIN:   float = 0.5      # floor: prevents degenerate near-zero gamma

# Regime-specific BOOST_SCALE overrides (derived from per-KB calibration).
# Used in place of the global BOOST_SCALE constant for non-hub regimes.
_BOOST_SCALE = {
    "hub_homogeneous":    21.79,   # Phase 204 MetaQA validated
    "typed_heterogeneous": 403.6,  # Phase 206 Hetionet validated (gamma=5.92, beta=1.88)
    "mixed":              100.0,   # geometric mean estimate — refine when mixed KB available
}

# ---------------------------------------------------------------------------
# Regime-specific constants
# ---------------------------------------------------------------------------
# Hub-homogeneous calibrated from MetaQA Phase 204.
# Typed-heterogeneous and mixed are analytical estimates; refine when
# multi-KB validation data becomes available.

_TRB_C = {
    "hub_homogeneous":    9.33,   # 21.486 / log(9+1) — Phase 204 MetaQA
    "typed_heterogeneous": 6.94,  # 22.350 / log(24+1) — Phase 206 Hetionet
    "mixed":               7.50,
}

_BETA = {
    "hub_homogeneous":    2.0,    # Phase 204 best: 2.0846; amplifies hub relations
    "typed_heterogeneous": 1.88,  # Phase 206 Hetionet tuner: 1.8778 (not linear as estimated)
    "mixed":               1.7,
}

_BRANCH_BONUS = {
    "hub_homogeneous":    0.48,   # Phase 204 best: 0.482
    "typed_heterogeneous": 0.45,  # Phase 206 Hetionet tuner: 0.451
    "mixed":               0.40,
}

_R2_C = {
    "hub_homogeneous":    13.50,  # (8.185 - 1.5) / log(5.768/9+1) — Phase 204
    "typed_heterogeneous":  1.68, # (4.201 - 1.5) / log(95.7/24+1) — Phase 206 Hetionet
    "mixed":               4.00,
}

_FHRB_C = {
    "hub_homogeneous":    1.18,   # (3.260 - 1.0) / log(5.768+1) — Phase 204
    "typed_heterogeneous": 0.44,  # (3.013 - 1.0) / log(95.7+1) — Phase 206 Hetionet
    "mixed":               0.70,
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
        # trb_auto).  For parameter scaling we apply a hub override when:
        #   - degree_cv > 3.0  (heavy-tailed distribution = hub bottleneck present)
        #   - mean_rel_coverage > 0.20  (most nodes are sources of most relations
        #                                = low real typing, peripheral metadata only)
        # This captures MetaQA (cv=5.68, cov=0.261) — whose rare metadata relations
        # (has_imdb_rating, in_language) mis-trigger "typed_heterogeneous" in
        # GraphProfiler — without incorrectly overriding genuinely typed graphs like
        # Hetionet (cv=4.17, cov=0.166) where typing carries semantic meaning.
        regime = profile.regime
        if (regime == "typed_heterogeneous"
                and profile.degree_cv > 3.0
                and profile.mean_rel_coverage > 0.20):
            regime = "hub_homogeneous"

        n_rel  = max(profile.n_relation_types, 1)
        n_nodes = max(profile.n_nodes, 1)
        n_edges = profile.n_edges

        # Undirected mean degree: sum_degrees = 2*n_edges for undirected graph.
        mean_degree = (2.0 * n_edges / n_nodes) if n_edges > 0 else 1.0

        max_fo, mean_fo, harmonic_fo, n_rel_deriver = deriver.fan_out_stats()
        if n_rel_deriver > 0:
            n_rel = n_rel_deriver  # prefer deriver count (more precise)
        mean_fo = max(mean_fo, 1e-6)

        # ------------------------------------------------------------------
        # trb_factor  = C × log(n_rel + 1)
        # ------------------------------------------------------------------
        trb_factor = _TRB_C[regime] * math.log(n_rel + 1)

        # ------------------------------------------------------------------
        # beta — regime constant (must be set before gamma)
        # ------------------------------------------------------------------
        beta = _BETA[regime]

        # ------------------------------------------------------------------
        # gamma  = _BOOST_SCALE[regime] / mean_fan_out^beta
        #
        # SDRB: score *= (1 + gamma * fan_out(r)^beta)
        # Mean boost across all relations ≈ gamma * mean_fo^beta = BOOST_SCALE
        # → gamma = BOOST_SCALE / mean_fo^beta
        #
        # BOOST_SCALE is regime-specific (not universal):
        #   hub_homogeneous:    21.79  (MetaQA Phase 204: 21.79/1.55^2.0 = 9.06, Δ=3.8%)
        #   typed_heterogeneous: 403.6 (Hetionet Phase 206: 403.6/9.46^1.88 = 5.92, validated)
        # ------------------------------------------------------------------
        _bs = _BOOST_SCALE.get(regime, BOOST_SCALE)
        gamma = max(GAMMA_MIN, _bs / (mean_fo ** beta))

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

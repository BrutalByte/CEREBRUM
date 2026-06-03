"""
Phase 205/207/208/213: ParameterInitializer — principled hyperparameter defaults from graph statistics.

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

Constants exist in a 2D space: (regime × embedding_method).
Sentence-transformers shifts the scoring landscape vs random embeddings.
Cross-regime pattern: beta converges (~0.96 both regimes × sentence); trb and fhrb
dominate for hub graphs (fANOVA #1/#2), vote dominates for typed graphs.

Calibration basis (2D constant table — all cells filled for known KBs):
  hub_homogeneous   × random:    Phase 204 MetaQA 3-hop 60.36% H@1 (gamma=8.73,  beta=2.09)
  typed_heterogeneous × random:  Phase 207 Hetionet     61.00% H@1 (hop_expand fixed)
  typed_heterogeneous × sentence: Phase 209 Hetionet    81.1%  H@1 (2-hop, multi-hop tuner)
  hub_homogeneous   × sentence:  Phase 213 MetaQA 3-hop 66.8%  H@1 (gamma=10.14, beta=0.96)
  mixed × random/sentence:       pending Phase 214 (ConceptNet calibration)

Phase 213 fANOVA (hub_homogeneous × sentence): trb=0.22 (#1), fhrb=0.19 (#2),
  gamma=0.15, beta=0.12, vote=0.12, r2=0.09, branch=0.09, idf=0.02
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.graph_profiler import QueryProfile
    from core.relation_boost_deriver import RelationBoostDeriver

# ---------------------------------------------------------------------------
# Universal constants (graph-invariant — validated across all MetaQA runs)
# ---------------------------------------------------------------------------

BEAM_WIDTH: int = 12          # fANOVA importance ~0.1%; plateau at 8–12

IDF_SCALE_C: float = 0.0102   # degree_cv × IDF_SCALE_C → idf_weight
                               # MetaQA random: cv=5.68, idf=0.058 → C=0.0102
                               # For sentence-transformers see _SENTENCE_OVERRIDES["idf_scale_c"]

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
#
# IMPORTANT: BOOST_SCALE changes dramatically when beta changes because
# mean_fo^beta is super-sensitive to beta.
#   hub_homogeneous (MetaQA):   gamma=8.73, mean_fo=1.55, beta=2.09 → BS=21.79
#   typed_heterogeneous (Hetionet Phase 207):
#     gamma=4.97, mean_fo=9.46, beta=0.777 → 9.46^0.777=5.73 → BS=28.48
#   Phase 206 value of 403.6 was tuned on broken 2-hop data (hop_expand bug);
#   Phase 207 (hop_expand fixed) beta dropped from 1.88 → 0.777, BS from 403.6 → 28.48
_BOOST_SCALE = {
    "hub_homogeneous":    21.79,  # Phase 204 MetaQA validated
    "typed_heterogeneous": 28.48, # Phase 207 Hetionet validated (gamma=4.97, beta=0.777)
    "mixed":              22.00,  # estimate — geometric mean; refine when mixed KB available
}

# ---------------------------------------------------------------------------
# Regime-specific constants
# ---------------------------------------------------------------------------
# Hub-homogeneous calibrated from MetaQA Phase 204.
# Typed-heterogeneous and mixed are analytical estimates; refine when
# multi-KB validation data becomes available.

_TRB_C = {
    "hub_homogeneous":    9.33,  # 21.486 / log(9+1) — Phase 204 MetaQA
    "typed_heterogeneous": 6.14, # 19.769 / log(24+1) — Phase 207 Hetionet (hop_expand fixed)
    "mixed":               7.50,
}

_BETA = {
    "hub_homogeneous":    2.0,   # Phase 204 best: 2.0846; amplifies hub relations
    "typed_heterogeneous": 0.777, # Phase 207 Hetionet (hop_expand fixed); was 1.88 on broken 2-hop
    "mixed":               1.4,
}

_BRANCH_BONUS = {
    "hub_homogeneous":    0.48,  # Phase 204 best: 0.482
    "typed_heterogeneous": 0.308, # Phase 207 Hetionet (hop_expand fixed)
    "mixed":               0.40,
}

_R2_C = {
    "hub_homogeneous":    13.50, # (8.185 - 1.5) / log(5.768/9+1) — Phase 204
    "typed_heterogeneous": 1.07, # (3.224 - 1.5) / log(95.7/24+1) — Phase 207 Hetionet
    "mixed":               4.00,
}

_FHRB_C = {
    "hub_homogeneous":    1.18,  # (3.260 - 1.0) / log(5.768+1) — Phase 204
    "typed_heterogeneous": 0.80, # (4.658 - 1.0) / log(95.7+1) — Phase 207 Hetionet
    "mixed":               0.70,
}

# vote_weight lower bound is regime-specific: typed graphs have more edge types,
# making community boundaries noisier and community votes less reliable.
_VOTE_BASE = {
    "hub_homogeneous":    0.72,  # Phase 204 MetaQA validated
    "typed_heterogeneous": 0.55, # Phase 207 Hetionet: optimal=0.6047; lower base for typed graphs
    "mixed":               0.64,
}

# ---------------------------------------------------------------------------
# Sentence-transformers overrides (regime × "sentence" embedding_method)
#
# With real semantic embeddings the scoring landscape shifts substantially:
#   - idf_weight dominates (fANOVA=0.43) vs beta-dominant with random
#   - beta rises (more gradient needed across larger fan-out)
#   - BOOST_SCALE rises correspondingly (beta=1.88 on Hetionet mean_fo=9.46
#     raises mean_fo^beta from 5.73 to 68.0, so BS scales accordingly)
#   - idf_scale_c drops: sentence embeddings already encode hub-ness
#     semantically, so the IDF penalty coefficient is smaller
#
# Source: Phase 208 Hetionet tuner (sentence, 55-trial Sobol+CMA-ES, 20q)
# typed_heterogeneous: gamma=5.9183, beta=1.8778, trb=22.350, r2=4.201,
#   vote=0.6460, idf=0.032, branch=0.451, fhrb=3.013
# ---------------------------------------------------------------------------
_SENTENCE_OVERRIDES: dict[str, dict] = {
    "typed_heterogeneous": {
        # Phase 209 multi-hop calibration (--min-eval-hop 2, hop_expand=False tuning).
        # Phase 208 all-hop tuning was dominated by near-ceiling 1-hop templates
        # → idf_weight over-weighted (0.43 fANOVA). Phase 209 targets 2-hop + 3-hop
        # where vote_weight is the dominant parameter (0.28 fANOVA).
        #
        # Source: gamma=1.0147, beta=0.9545, trb=13.871, r2=1.322, vote=0.6400,
        #   idf=0.018, branch=0.032, fhrb=1.726  (Hetionet Phase 209, 200-trial S+CMA-ES)
        "boost_scale":  8.67,    # 1.0147 × 9.46^0.9545 = 1.0147 × 8.54 = 8.67
        "beta":          0.9545, # Phase 209; lower than Phase 208 (1.878 was all-hop)
        "trb_c":         4.31,   # 13.871 / log(25) = 4.31
        "branch_bonus":  0.032,  # Phase 209: much lower than random (0.308) or Ph208 (0.451)
        "r2_c":          -0.111,  # (1.322−1.5) / log(4.988) = -0.111  (negative for typed)
        "r2_floor":       1.0,   # clamp floor: sentence typed can go below 1.5
        "fhrb_c":        0.159,  # (1.726−1.0) / log(96.7) = 0.726/4.572 = 0.159
        "idf_scale_c":   0.00432,# 0.018 / 4.17 = 0.00432 (Hetionet degree_cv=4.17)
        "vote_base":     0.565,  # 0.6400 − 0.15×0.5 = 0.565 (vote is now #1 fANOVA driver)
    },
    "hub_homogeneous": {
        # Phase 213 MetaQA sentence-transformers calibration (60 Sobol + CMA-ES, 500q/trial).
        # Source: gamma=10.1379, beta=0.9616, trb=31.134, r2=3.977, vote=0.7527,
        #   idf=0.03934, branch=0.3851, fhrb=7.572  → H@1=66.8% (3-hop, 500-sample)
        # fANOVA: trb=0.22 (#1), fhrb=0.19 (#2) — contrast with typed where vote leads.
        # Key shift vs random: beta drops 2.09→0.96 (consistent with typed × sentence=0.9545).
        "boost_scale":  15.4514, # 10.1379 × 1.55^0.9616
        "beta":          0.9616,
        "trb_c":        13.5211, # 31.134 / log(10)
        "branch_bonus":  0.3851,
        "r2_c":          5.0018, # (3.977 − 1.5) / log(5.768/9 + 1)
        "r2_floor":      1.5,
        "fhrb_c":        3.4374, # (7.572 − 1.0) / log(6.768)
        "idf_scale_c":   0.00693,# 0.03934 / 5.68
        "vote_base":     0.6777, # 0.7527 − 0.15×0.5
    },
    # mixed × sentence: pending Phase 215 ConceptNet tuner run
}


# ---------------------------------------------------------------------------
# Phase 218-C: Known-KB profile vectors for cross-KB parameter blending
#
# 5-vector: [degree_cv, mean_degree, mean_fan_out, n_relation_types, mean_rel_coverage]
# Normalised to unit-sphere so cosine similarity == dot product.
# ---------------------------------------------------------------------------
_KB_PROFILE_VECTORS: dict[str, np.ndarray] = {
    # MetaQA: hub graph — high CV (5.68), low rel count (9), high coverage (0.261)
    "metaqa": np.array([5.68, 6.77, 1.55, 9.0, 0.261], dtype=float),
    # Hetionet: typed heterogeneous — moderate CV (4.17), 24 rels, low coverage (0.166)
    "hetionet": np.array([4.17, 9.46, 9.46, 24.0, 0.166], dtype=float),
}
# Normalise once at import time
for _kb in _KB_PROFILE_VECTORS:
    _v = _KB_PROFILE_VECTORS[_kb]
    _norm = float(np.linalg.norm(_v))
    if _norm > 0:
        _KB_PROFILE_VECTORS[_kb] = _v / _norm

# Regime → best-matching known KB (used as primary blend source)
_REGIME_TO_KB: dict[str, str] = {
    "hub_homogeneous":     "metaqa",
    "typed_heterogeneous": "hetionet",
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
    effective_regime: str = "unknown"    # regime used for constant selection
    embedding_method: str = "random"     # embedding method used for constant selection

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
        emb_tag = self.embedding_method if (self.embedding_method == "sentence") else "random"
        lines = [f"ParameterInitializer defaults (regime={self.effective_regime}/{emb_tag}):"]
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
        embedding_method: str = "random",
    ) -> InitialParams:
        """
        Compute principled defaults.

        Parameters
        ----------
        profile          : QueryProfile from GraphProfiler.profile()
        deriver          : RelationBoostDeriver (built from KB file or triples)
        modularity_Q     : Newman-Girvan modularity of the community partition.
                           If unknown, 0.5 gives a safe mid-range vote_weight.
        embedding_method : "random" or "sentence".  Sentence-transformers shifts
                           the scoring landscape (idf dominates, beta rises).
                           Phase 208 constants available for typed_heterogeneous.
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

        # Phase 218-C: for mixed regime, blend from known-KB calibrations
        if regime == "mixed":
            return ParameterInitializer._blend_params_mixed(profile, modularity_Q, embedding_method)

        # Sentence-transformers overrides: apply when embedding_method="sentence"
        # and regime has calibrated sentence constants. Falls back to random
        # constants for regimes not yet validated with sentence embeddings.
        use_sentence = embedding_method == "sentence"
        _sovr: dict = _SENTENCE_OVERRIDES.get(regime, {}) if use_sentence else {}

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
        trb_c      = _sovr.get("trb_c", _TRB_C[regime])
        trb_factor = trb_c * math.log(n_rel + 1)

        # ------------------------------------------------------------------
        # beta — regime constant (must be set before gamma)
        # ------------------------------------------------------------------
        beta = _sovr.get("beta", _BETA[regime])

        # ------------------------------------------------------------------
        # gamma  = BOOST_SCALE / mean_fan_out^beta
        #
        # SDRB: score *= (1 + gamma * fan_out(r)^beta)
        # Mean boost across all relations ≈ gamma * mean_fo^beta = BOOST_SCALE
        # → gamma = BOOST_SCALE / mean_fo^beta
        #
        # BOOST_SCALE is regime × embedding specific:
        #   hub_homogeneous   × random:    21.79 (MetaQA Phase 204)
        #   typed_heterogeneous × random:  28.48 (Hetionet Phase 207, beta=0.777)
        #   typed_heterogeneous × sentence: 8.67 (Hetionet Phase 209, beta=0.9545)
        #   Phase 208 (402.8/beta=1.878) was all-hop tuning; Phase 209 multi-hop
        #   calibration yields a much lower BOOST_SCALE and beta.
        # ------------------------------------------------------------------
        _bs = _sovr.get("boost_scale", _BOOST_SCALE.get(regime, BOOST_SCALE))
        gamma = max(GAMMA_MIN, _bs / (mean_fo ** beta))

        # ------------------------------------------------------------------
        # r2_boost  = r2_base + C × log(mean_deg / n_rel + 1)
        # r2_base defaults to 1.5; sentence-transformers typed uses 1.0 (lower
        # is better when beam already selects semantically-consistent paths).
        # ------------------------------------------------------------------
        r2_c     = _sovr.get("r2_c",    _R2_C[regime])
        r2_floor = _sovr.get("r2_floor", 1.5)
        r2_boost = 1.5 + r2_c * math.log(mean_degree / n_rel + 1)
        r2_boost = max(r2_floor, r2_boost)

        # ------------------------------------------------------------------
        # fhrb_factor  = 1.0 + C × log(mean_deg + 1)
        # ------------------------------------------------------------------
        fhrb_c      = _sovr.get("fhrb_c", _FHRB_C[regime])
        fhrb_factor = 1.0 + fhrb_c * math.log(mean_degree + 1)
        fhrb_factor = max(1.0, fhrb_factor)

        # ------------------------------------------------------------------
        # idf_weight  = max(0.01, idf_scale_c × degree_cv)
        # Sentence-transformers embeddings already encode hub-ness semantically,
        # so the IDF penalty coefficient is smaller than with random embeddings.
        # ------------------------------------------------------------------
        idf_c      = _sovr.get("idf_scale_c", IDF_SCALE_C)
        idf_weight = max(0.01, idf_c * profile.degree_cv)

        # ------------------------------------------------------------------
        # vote_weight  = vote_base + VOTE_Q_SCALE × Q
        # ------------------------------------------------------------------
        q_clamped   = max(0.0, min(1.0, modularity_Q))
        vote_base   = _sovr.get("vote_base", _VOTE_BASE.get(regime, VOTE_BASE))
        vote_weight = vote_base + VOTE_Q_SCALE * q_clamped

        # ------------------------------------------------------------------
        # branch_bonus — regime constant (or sentence override)
        # ------------------------------------------------------------------
        branch_bonus = _sovr.get("branch_bonus", _BRANCH_BONUS[regime])

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
            embedding_method = embedding_method,
        )

    @staticmethod
    def _blend_params_mixed(
        profile: "QueryProfile",
        modularity_Q: float,
        embedding_method: str,
    ) -> InitialParams:
        """Phase 218-C: Blend known-KB calibrations for unseen 'mixed' regime.

        Computes cosine similarity between the new graph's 5-vector profile and
        known KB profile vectors (MetaQA, Hetionet), then soft-mixes their
        per-constant dicts via softmax weights instead of using hardcoded mixed defaults.
        """
        n_rel = max(profile.n_relation_types, 1)
        n_nodes = max(profile.n_nodes, 1)
        mean_degree = (2.0 * profile.n_edges / n_nodes) if profile.n_edges > 0 else 1.0
        mean_fo = max(mean_degree / n_rel, 1e-6)
        mean_rel_coverage = getattr(profile, "mean_rel_coverage", 0.0)

        query_vec = np.array([
            profile.degree_cv, mean_degree, mean_fo, float(n_rel), mean_rel_coverage,
        ], dtype=float)
        q_norm = float(np.linalg.norm(query_vec))
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Cosine similarities against known KB vectors
        sims = {kb: float(np.dot(query_vec, vec)) for kb, vec in _KB_PROFILE_VECTORS.items()}

        # Softmax to blend weights
        max_sim = max(sims.values())
        exp_sims = {kb: math.exp(s - max_sim) for kb, s in sims.items()}
        total = sum(exp_sims.values())
        weights = {kb: v / total for kb, v in exp_sims.items()}

        # Blend per-constant tables using weights
        regime_map = {"metaqa": "hub_homogeneous", "hetionet": "typed_heterogeneous"}
        use_sentence = embedding_method == "sentence"

        def _blend_const(table: dict) -> float:
            return sum(weights[kb] * table[regime_map[kb]] for kb in weights)

        trb_factor   = _blend_const(_TRB_C) * math.log(n_rel + 1)
        beta         = _blend_const(_BETA)
        _bs          = _blend_const(_BOOST_SCALE)
        gamma        = max(GAMMA_MIN, _bs / (mean_fo ** beta))
        r2_boost     = max(1.5, 1.5 + _blend_const(_R2_C) * math.log(mean_degree / n_rel + 1))
        fhrb_factor  = max(1.0, 1.0 + _blend_const(_FHRB_C) * math.log(mean_degree + 1))
        idf_weight   = max(0.01, IDF_SCALE_C * profile.degree_cv)
        vote_base    = _blend_const(_VOTE_BASE)
        q_clamped    = max(0.0, min(1.0, modularity_Q))
        vote_weight  = vote_base + VOTE_Q_SCALE * q_clamped
        branch_bonus = _blend_const(_BRANCH_BONUS)

        return InitialParams(
            trb_factor       = round(trb_factor,   4),
            gamma            = round(gamma,        4),
            beta             = round(beta,         4),
            r2_boost         = round(r2_boost,     4),
            fhrb_factor      = round(fhrb_factor,  4),
            idf_weight       = round(idf_weight,   4),
            vote_weight      = round(vote_weight,  4),
            branch_bonus     = round(branch_bonus, 4),
            beam_width       = BEAM_WIDTH,
            effective_regime = "mixed",
            embedding_method = embedding_method,
        )

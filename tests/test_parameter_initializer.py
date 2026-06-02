"""
Tests for Phase 205: ParameterInitializer.

Validation target: MetaQA hub_homogeneous regime.
Phase 204 best params: trb=21.486 r2=8.185 vote=0.764 idf=0.058
                       fhrb=3.260 gamma=8.7319 beta=2.0846 branch=0.482 bw=12
"""
import math
import pytest
from unittest.mock import MagicMock

from core.parameter_initializer import ParameterInitializer, BEAM_WIDTH, InitialParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_profile(
    regime="hub_homogeneous",
    n_nodes=43234,
    n_edges=124680,
    n_relation_types=9,
    degree_cv=5.6821,
    hub_score=0.45,
    mean_rel_coverage=0.261,
):
    p = MagicMock()
    p.regime = regime
    p.n_nodes = n_nodes
    p.n_edges = n_edges
    p.n_relation_types = n_relation_types
    p.degree_cv = degree_cv
    p.hub_score = hub_score
    p.mean_rel_coverage = mean_rel_coverage
    return p


def _mock_deriver(
    max_fo=3.1103,
    mean_fo=1.5506,
    harmonic_fo=1.3280,
    n_rel=9,
):
    d = MagicMock()
    d.fan_out_stats.return_value = (max_fo, mean_fo, harmonic_fo, n_rel)
    d.is_built = True
    return d


# ---------------------------------------------------------------------------
# MetaQA calibration test — each param within 10% of Phase 204 validated best
# ---------------------------------------------------------------------------

PHASE204_BEST = {
    "trb_factor":   21.486,
    "gamma":        8.7319,
    "beta":         2.0846,
    "r2_boost":     8.185,
    "fhrb_factor":  3.260,
    "idf_weight":   0.058,
    "vote_weight":  0.764,
    "branch_bonus": 0.482,
    "beam_width":   12,
}

@pytest.mark.parametrize("param,expected,tol", [
    ("trb_factor",   PHASE204_BEST["trb_factor"],   0.10),
    ("gamma",        PHASE204_BEST["gamma"],         0.10),
    ("beta",         PHASE204_BEST["beta"],          0.10),
    ("r2_boost",     PHASE204_BEST["r2_boost"],      0.10),
    ("fhrb_factor",  PHASE204_BEST["fhrb_factor"],   0.10),
    ("idf_weight",   PHASE204_BEST["idf_weight"],    0.10),
    ("vote_weight",  PHASE204_BEST["vote_weight"],   0.05),
    ("branch_bonus", PHASE204_BEST["branch_bonus"],  0.10),
])
def test_metaqa_calibration(param, expected, tol):
    """Each param must be within tol fraction of the Phase 204 validated value."""
    profile = _mock_profile()
    deriver = _mock_deriver()
    # vote_weight requires Q≈0.293 (back-calc from 0.764)
    Q = (PHASE204_BEST["vote_weight"] - 0.72) / 0.15
    params = ParameterInitializer.compute(profile, deriver, modularity_Q=Q)
    actual = getattr(params, param)
    assert abs(actual - expected) / expected <= tol, (
        f"{param}: got {actual:.4f}, expected {expected:.4f} (±{tol*100:.0f}%)"
    )


def test_beam_width_fixed():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver())
    assert params.beam_width == BEAM_WIDTH == 12


def test_returns_initial_params():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver())
    assert isinstance(params, InitialParams)


def test_as_dict_has_all_keys():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver())
    d = params.as_dict()
    expected_keys = {
        "trb_factor", "gamma", "beta", "r2_boost", "fhrb_factor",
        "idf_weight", "vote_weight", "branch_bonus", "beam_width",
    }
    assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Regime sensitivity
# ---------------------------------------------------------------------------

def test_typed_heterogeneous_lower_trb():
    hub_p = _mock_profile(regime="hub_homogeneous", degree_cv=5.0)
    # Low mean_rel_coverage prevents the hub_homogeneous override
    typ_p = _mock_profile(regime="typed_heterogeneous", degree_cv=0.8,
                          mean_rel_coverage=0.10)
    d = _mock_deriver()
    assert ParameterInitializer.compute(typ_p, d).trb_factor < \
           ParameterInitializer.compute(hub_p, d).trb_factor


def test_typed_heterogeneous_beta():
    # Phase 207: typed_heterogeneous random beta calibrated to 0.777
    p = _mock_profile(regime="typed_heterogeneous", degree_cv=0.8,
                      mean_rel_coverage=0.10)
    params = ParameterInitializer.compute(p, _mock_deriver())
    assert params.beta == 0.777
    assert params.effective_regime == "typed_heterogeneous"


def test_hub_override_metaqa_like():
    """MetaQA-like: typed_heterogeneous + high CV + high mean_rel_coverage → hub_homogeneous."""
    p = _mock_profile(regime="typed_heterogeneous", degree_cv=5.68,
                      n_nodes=43234, n_edges=124680)
    # mean_rel_coverage defaults to 0.261 in _mock_profile helper — above 0.20
    p.mean_rel_coverage = 0.261
    params = ParameterInitializer.compute(p, _mock_deriver())
    assert params.effective_regime == "hub_homogeneous"
    assert params.beta == 2.0  # hub constant, not typed constant


def test_no_override_hetionet_like():
    """Hetionet-like: typed_heterogeneous + low mean_rel_coverage → stays typed."""
    p = _mock_profile(regime="typed_heterogeneous", degree_cv=4.17)
    p.mean_rel_coverage = 0.166  # below 0.20 — real typed graph
    params = ParameterInitializer.compute(p, _mock_deriver())
    assert params.effective_regime == "typed_heterogeneous"
    assert params.beta == 0.777  # Phase 207 calibrated value


def test_sentence_transformers_overrides():
    """Phase 209: sentence-transformers typed_heterogeneous uses multi-hop calibrated constants."""
    p = _mock_profile(regime="typed_heterogeneous", degree_cv=4.17)
    p.mean_rel_coverage = 0.166
    d = _mock_deriver()
    p_rand = ParameterInitializer.compute(p, d, embedding_method="random")
    p_sent = ParameterInitializer.compute(p, d, embedding_method="sentence")
    # Phase 209: beta=0.9545 (multi-hop calibration vs random 0.777)
    assert p_sent.beta == 0.9545
    assert p_rand.beta == 0.777
    # idf_weight drops with sentence embeddings (semantic hub encoding)
    assert p_sent.idf_weight < p_rand.idf_weight
    # branch_bonus drops dramatically on sentence multi-hop
    assert p_sent.branch_bonus < p_rand.branch_bonus
    # effective_regime unchanged — it's the structural routing regime, not embedding
    assert p_sent.effective_regime == "typed_heterogeneous"
    assert p_sent.embedding_method == "sentence"


def test_sentence_hub_homogeneous_overrides():
    """Phase 213: sentence hub_homogeneous uses vote_base=0.689 (lower than random 0.72)."""
    p = _mock_profile(regime="hub_homogeneous", degree_cv=5.68)
    p.mean_rel_coverage = 0.261  # MetaQA-like: triggers hub override from typed
    d = _mock_deriver()
    p_rand = ParameterInitializer.compute(p, d, embedding_method="random")
    p_sent = ParameterInitializer.compute(p, d, embedding_method="sentence", modularity_Q=0.5)
    # vote_weight is the primary difference: sentence vote_base=0.689 vs random 0.72
    assert p_sent.vote_weight < p_rand.vote_weight
    assert p_sent.vote_weight == pytest.approx(0.689 + 0.15 * 0.5, abs=1e-4)
    assert p_sent.effective_regime == "hub_homogeneous"
    assert p_sent.embedding_method == "sentence"


def test_mixed_regime_mid_range():
    hub_p   = _mock_profile(regime="hub_homogeneous", degree_cv=5.0)
    mixed_p = _mock_profile(regime="mixed", degree_cv=2.0, mean_rel_coverage=0.15)
    typ_p   = _mock_profile(regime="typed_heterogeneous", degree_cv=0.8,
                            mean_rel_coverage=0.10)
    d = _mock_deriver()
    hub_trb   = ParameterInitializer.compute(hub_p, d).trb_factor
    mixed_trb = ParameterInitializer.compute(mixed_p, d).trb_factor
    typ_trb   = ParameterInitializer.compute(typ_p, d).trb_factor
    assert typ_trb <= mixed_trb <= hub_trb


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_vote_weight_clamped_q_zero():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver(), modularity_Q=0.0)
    assert params.vote_weight == pytest.approx(0.72, abs=1e-6)


def test_vote_weight_clamped_q_one():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver(), modularity_Q=1.0)
    assert params.vote_weight == pytest.approx(0.87, abs=1e-6)


def test_vote_weight_out_of_range_clamped():
    p1 = ParameterInitializer.compute(_mock_profile(), _mock_deriver(), modularity_Q=-5.0)
    p2 = ParameterInitializer.compute(_mock_profile(), _mock_deriver(), modularity_Q=99.0)
    assert p1.vote_weight == pytest.approx(0.72, abs=1e-6)
    assert p2.vote_weight == pytest.approx(0.87, abs=1e-6)


def test_idf_weight_minimum():
    # Very low degree_cv should still return at least 0.01
    p = _mock_profile(degree_cv=0.0001)
    params = ParameterInitializer.compute(p, _mock_deriver())
    assert params.idf_weight >= 0.01


def test_single_node_graph_no_crash():
    p = _mock_profile(n_nodes=1, n_edges=0, n_relation_types=1, degree_cv=0.0)
    d = _mock_deriver(max_fo=1.0, mean_fo=1.0, harmonic_fo=1.0, n_rel=1)
    params = ParameterInitializer.compute(p, d)
    assert params.trb_factor > 0
    assert params.gamma > 0


def test_large_graph_scaling():
    """Params must scale gracefully — no infinities or negatives on large KBs."""
    p = _mock_profile(n_nodes=1_000_000, n_edges=50_000_000, n_relation_types=500, degree_cv=3.0)
    d = _mock_deriver(max_fo=1000.0, mean_fo=50.0, harmonic_fo=5.0, n_rel=500)
    params = ParameterInitializer.compute(p, d)
    for k, v in params.as_dict().items():
        assert math.isfinite(float(v)), f"{k} is not finite: {v}"
        if k != "beam_width":
            assert float(v) > 0, f"{k} is non-positive: {v}"


def test_summary_contains_all_params():
    params = ParameterInitializer.compute(_mock_profile(), _mock_deriver())
    s = params.summary()
    for key in params.as_dict():
        assert key in s

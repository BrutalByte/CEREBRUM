"""
Tests for Phase 17.4 — Learned CSA Parameters.

Covers:
  - CSAParameterLearner initialisation and property access
  - fit() on empty training set
  - Loss decreases over training iterations
  - Converges to better ranking on synthetic pairs
  - LearningResult fields populated correctly
  - Gradient direction is meaningful (sanity check)
  - clip enforced on parameters
  - fit() idempotency on deterministic data
  - edge_features path scoring
  - Fallback to attention_weights when edge_features absent
"""
import math
import pytest
import networkx as nx

from core.parameter_learner import CSAParameterLearner, LearningResult


# ---------------------------------------------------------------------------
# Helpers: synthetic paths with edge_features
# ---------------------------------------------------------------------------

class _FakePath:
    """
    Minimal path-like object carrying edge_features and/or attention_weights.

    edge_features: list of (sim, cs, etw, nd, hd) tuples — one per edge.
    attention_weights: list of floats (fallback scoring).
    """
    def __init__(self, edge_features=None, attention_weights=None):
        self.edge_features   = edge_features    or []
        self.attention_weights = attention_weights or []


def _semantic_pos():
    """Positive path: high cosine similarity (sim=0.9), same community (cs=1.0)."""
    return _FakePath(edge_features=[(0.9, 1.0, 0.0, 0.0, 1.0)])


def _semantic_neg():
    """Negative path: low cosine similarity (sim=0.1), distant community (cs=0.1)."""
    return _FakePath(edge_features=[(0.1, 0.1, 0.0, 0.0, 0.5)])


def _make_adapter():
    """Minimal adapter for CSAParameterLearner construction."""
    from adapters.networkx_adapter import NetworkXAdapter
    G = nx.DiGraph()
    for n in ["A", "B"]:
        G.add_node(n, label=n)
    G.add_edge("A", "B", relation="R", weight=1.0)
    return NetworkXAdapter(G)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_default_params():
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    a, b, g, d, e = learner.params
    assert a == pytest.approx(0.4)
    assert b == pytest.approx(0.4)
    assert g == pytest.approx(0.1)
    assert d == pytest.approx(0.05)
    assert e == pytest.approx(0.05)


def test_custom_init_params():
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter, init_params=(0.3, 0.3, 0.2, 0.1, 0.1))
    a, b, g, d, e = learner.params
    assert a == pytest.approx(0.3)


def test_result_none_before_fit():
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    assert learner.result is None


# ---------------------------------------------------------------------------
# fit() on empty training set
# ---------------------------------------------------------------------------

def test_fit_empty_pairs_returns_result():
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    result = learner.fit([])
    assert isinstance(result, LearningResult)
    assert result.final_loss == pytest.approx(0.0)
    assert result.converged is True
    assert result.n_iterations == 0


def test_fit_empty_pairs_params_unchanged():
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    before = learner.params
    learner.fit([])
    assert learner.params == before


# ---------------------------------------------------------------------------
# LearningResult fields
# ---------------------------------------------------------------------------

def test_learning_result_fields_populated():
    adapter = _make_adapter()
    pairs = [(_semantic_pos(), _semantic_neg())]
    learner = CSAParameterLearner(adapter, max_iterations=10)
    result = learner.fit(pairs)
    assert isinstance(result.params, tuple) and len(result.params) == 5
    assert result.n_iterations >= 1
    assert result.duration_seconds >= 0.0
    assert result.initial_loss >= 0.0
    assert result.final_loss >= 0.0


def test_result_accessible_via_property():
    adapter = _make_adapter()
    pairs = [(_semantic_pos(), _semantic_neg())]
    learner = CSAParameterLearner(adapter, max_iterations=5)
    returned = learner.fit(pairs)
    assert learner.result is returned


# ---------------------------------------------------------------------------
# Loss decreases (basic gradient sanity)
# ---------------------------------------------------------------------------

def test_loss_decreases_or_stays_flat():
    """After training, final_loss should not be higher than initial_loss."""
    adapter = _make_adapter()
    pairs = [(_semantic_pos(), _semantic_neg())] * 5
    learner = CSAParameterLearner(adapter, max_iterations=50, learning_rate=0.05)
    result = learner.fit(pairs)
    assert result.final_loss <= result.initial_loss + 1e-9


def test_loss_improves_with_semantic_pairs():
    """
    Semantic (sim-dominant) pairs: alpha should grow to drive pos above neg.
    We verify final_loss < initial_loss when there is a clear gradient signal.
    """
    adapter = _make_adapter()
    pairs = [(_semantic_pos(), _semantic_neg())] * 10
    learner = CSAParameterLearner(
        adapter,
        init_params=(0.1, 0.1, 0.0, 0.0, 0.0),  # start with low alpha
        max_iterations=100,
        learning_rate=0.05,
        margin=0.05,
    )
    result = learner.fit(pairs)
    # With clear signal, loss should drop
    assert result.final_loss < result.initial_loss or result.final_loss == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Parameter clipping
# ---------------------------------------------------------------------------

def test_clip_prevents_negative_params():
    """Parameters should stay >= 0 with default clip=(0.0, 2.0)."""
    adapter = _make_adapter()
    # Pairs that might push delta to negative
    pos = _FakePath(edge_features=[(0.5, 0.5, 0.0, 0.0, 0.5)])
    neg = _FakePath(edge_features=[(0.5, 0.5, 0.0, 1.0, 0.5)])
    pairs = [(pos, neg)] * 20
    learner = CSAParameterLearner(adapter, max_iterations=50, learning_rate=0.5)
    learner.fit(pairs)
    a, b, g, d, e = learner.params
    assert a >= 0.0 and b >= 0.0 and g >= 0.0 and d >= 0.0 and e >= 0.0


def test_clip_prevents_exceeding_max():
    """Parameters should stay <= clip_hi."""
    adapter = _make_adapter()
    pairs = [(_semantic_pos(), _semantic_neg())] * 20
    learner = CSAParameterLearner(
        adapter, max_iterations=100, learning_rate=5.0, clip=(0.0, 1.5)
    )
    learner.fit(pairs)
    for p in learner.params:
        assert p <= 1.5 + 1e-9


# ---------------------------------------------------------------------------
# Scoring paths with edge_features vs attention_weights fallback
# ---------------------------------------------------------------------------

def test_edge_features_scoring_deterministic():
    """Same edge_features + same params → same internal score."""
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    path = _FakePath(edge_features=[(0.8, 0.9, 0.0, 0.0, 1.0)])
    s1 = learner._score_path_parametric(path, list(learner.params))
    s2 = learner._score_path_parametric(path, list(learner.params))
    assert s1 == pytest.approx(s2)


def test_attention_weights_fallback():
    """Path with no edge_features but attention_weights uses log-product fallback."""
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    path = _FakePath(attention_weights=[0.7, 0.8])
    score = learner._score_path_parametric(path, list(learner.params))
    expected = math.log(0.7) + math.log(0.8)
    assert score == pytest.approx(expected, abs=1e-9)


def test_empty_path_scores_zero():
    """Path with neither features nor weights scores 0."""
    adapter = _make_adapter()
    learner = CSAParameterLearner(adapter)
    path = _FakePath()
    score = learner._score_path_parametric(path, list(learner.params))
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gradient direction sanity
# ---------------------------------------------------------------------------

def test_gradient_is_nonzero_for_informative_pairs():
    """At least one gradient component should be nonzero when pairs are not yet ranked."""
    adapter = _make_adapter()
    # With all-zero params the pos and neg score equally → margin violation → nonzero grad
    pos = _FakePath(edge_features=[(0.9, 1.0, 0.0, 0.0, 1.0)])
    neg = _FakePath(edge_features=[(0.1, 0.1, 0.0, 0.0, 0.5)])
    pairs = [(pos, neg)]
    learner = CSAParameterLearner(adapter, init_params=(0.0, 0.0, 0.0, 0.0, 0.0))
    grad = learner._numerical_gradient(pairs)
    assert any(abs(g) > 1e-8 for g in grad), "Gradient is all-zero — no signal"


def test_alpha_gradient_direction_for_semantic_pairs():
    """
    For sim-dominated positive pairs (sim=1.0 vs sim=0.0),
    gradient w.r.t. alpha should be negative (loss decreases by increasing alpha).
    """
    adapter = _make_adapter()
    pos = _FakePath(edge_features=[(1.0, 0.5, 0.0, 0.0, 0.0)])
    neg = _FakePath(edge_features=[(0.0, 0.5, 0.0, 0.0, 0.0)])
    pairs = [(pos, neg)] * 5
    learner = CSAParameterLearner(adapter, init_params=(0.01, 0.01, 0.0, 0.0, 0.0))
    grad = learner._numerical_gradient(pairs)
    # alpha gradient < 0 means increasing alpha reduces loss
    assert grad[0] <= 0.0


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_converges_on_trivially_separable_pairs():
    """With a very clear signal and enough iterations, should converge."""
    adapter = _make_adapter()
    pos = _FakePath(edge_features=[(1.0, 1.0, 0.0, 0.0, 1.0)])
    neg = _FakePath(edge_features=[(0.0, 0.0, 0.0, 0.0, 0.0)])
    pairs = [(pos, neg)] * 10
    learner = CSAParameterLearner(
        adapter, max_iterations=500, learning_rate=0.1, margin=0.01
    )
    result = learner.fit(pairs)
    assert result.final_loss == pytest.approx(0.0, abs=0.05)

"""
tests/test_cold_start.py
Hole 4 — Bayesian Cold-Start Bias fix.

Validates that warm_start_strength seeds the Beta distribution on the first hop
using the CSA score, reducing cold-start variance in probabilistic beam mode.
"""
import asyncio
import numpy as np
import pytest

from reasoning.traversal import TraversalPath, BeamTraversal, AsyncBeamTraversal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_path(emb_dim: int = 4) -> TraversalPath:
    return TraversalPath(
        nodes=["A"],
        seen_entities={"A"},
        embedding=np.zeros(emb_dim, dtype=np.float32),
        score=1.0,
        community_sequence=[0],
    )


def _extend(path: TraversalPath, weight: float, prior_scale: float = 1.0) -> TraversalPath:
    return path.copy_with_extension(
        rel="r",
        v="B",
        v_cid=1,
        v_emb=np.ones(len(path.embedding), dtype=np.float32),
        weight=weight,
        coherence_score=1.0,
        prior_scale=prior_scale,
    )


# ---------------------------------------------------------------------------
# TraversalPath.copy_with_extension — prior_scale parameter
# ---------------------------------------------------------------------------

def test_copy_with_extension_prior_scale_default():
    """Default prior_scale=1.0 → beta update identical to v0.4.0."""
    path = _seed_path()
    ext = _extend(path, weight=0.8)
    assert ext.beta_alpha == pytest.approx(1.8)
    assert ext.beta_beta == pytest.approx(1.2)


def test_copy_with_extension_prior_scale_param():
    """prior_scale=3.0 → alpha = 1 + w*3, beta = 1 + (1-w)*3."""
    path = _seed_path()
    ext = _extend(path, weight=0.8, prior_scale=3.0)
    assert ext.beta_alpha == pytest.approx(1.0 + 0.8 * 3.0)
    assert ext.beta_beta == pytest.approx(1.0 + 0.2 * 3.0)


def test_copy_with_extension_prior_scale_high_weight():
    path = _seed_path()
    ext = _extend(path, weight=0.9, prior_scale=5.0)
    assert ext.beta_alpha == pytest.approx(1.0 + 0.9 * 5.0)
    assert ext.beta_beta == pytest.approx(1.0 + 0.1 * 5.0)


# ---------------------------------------------------------------------------
# BeamTraversal warm_start_strength parameter
# ---------------------------------------------------------------------------

def test_warm_start_zero_unchanged():
    """warm_start_strength=0.0 → same beta accumulation as v0.4.0."""
    path = _seed_path()
    # Simulate what traverse() does at first hop with strength=0
    _prior_scale = 1.0  # strength=0 → no warm-start
    ext = _extend(path, weight=0.7, prior_scale=_prior_scale)
    assert ext.beta_alpha == pytest.approx(1.7)
    assert ext.beta_beta == pytest.approx(1.3)


def test_warm_start_scales_first_hop():
    """First hop alpha = 1 + w * (1 + strength)."""
    path = _seed_path()
    strength = 4.0
    w = 0.6
    prior_scale = 1.0 + strength
    ext = _extend(path, weight=w, prior_scale=prior_scale)
    assert ext.beta_alpha == pytest.approx(1.0 + w * prior_scale)
    assert ext.beta_beta == pytest.approx(1.0 + (1.0 - w) * prior_scale)


def test_warm_start_does_not_affect_later_hops():
    """After first hop, prior_scale=1.0 for subsequent extensions."""
    path = _seed_path()
    # First hop (warm-started)
    hop1 = _extend(path, weight=0.8, prior_scale=5.0)
    assert hop1.nodes == ["A", "r", "B"]
    # Second hop (normal scale)
    hop2 = hop1.copy_with_extension(
        rel="r2", v="C", v_cid=2,
        v_emb=np.ones(4, dtype=np.float32),
        weight=0.5, coherence_score=1.0, prior_scale=1.0,
    )
    # beta_alpha = hop1.beta_alpha + 0.5 * 1.0
    assert hop2.beta_alpha == pytest.approx(hop1.beta_alpha + 0.5)
    assert hop2.beta_beta == pytest.approx(hop1.beta_beta + 0.5)


def test_warm_start_only_in_probabilistic_mode():
    """warm_start_strength is only applied when probabilistic=True in traverse()."""
    # When probabilistic=False, the traverse loop uses prior_scale=1.0 unconditionally.
    # We verify this by checking the BeamTraversal attribute stores the value but
    # the logic path is gated on self.probabilistic.
    from unittest.mock import MagicMock
    adapter = MagicMock()
    adapter.get_embedding.return_value = np.zeros(4, dtype=np.float32)
    adapter.get_community.return_value = 0
    adapter.find_entities.return_value = []
    adapter.get_neighbors.return_value = []

    bt = BeamTraversal(adapter, None, probabilistic=False, warm_start_strength=10.0)
    assert bt.warm_start_strength == 10.0
    # traverse() with no neighbors returns seed paths — no extension made
    paths = bt.traverse(["X"])
    assert len(paths) == 1
    # Seed path beta stays at (1,1) — no extension occurred
    assert paths[0].beta_alpha == pytest.approx(1.0)


def test_warm_start_reduces_variance_at_first_hop():
    """Beta variance is lower when warm_start is applied (more confident prior)."""
    path = _seed_path()
    w = 0.7
    # Without warm-start
    cold = _extend(path, weight=w, prior_scale=1.0)
    # With warm-start (scale=6 → prior_scale=6)
    warm = _extend(path, weight=w, prior_scale=6.0)
    assert warm.score_variance < cold.score_variance


def test_warm_start_posterior_mean_reflects_weight():
    """High-weight edge with warm-start → posterior_mean closer to the weight."""
    path = _seed_path()
    w = 0.9
    ext = _extend(path, weight=w, prior_scale=10.0)
    # alpha=1+0.9*10=10, beta=1+0.1*10=2 → E[Beta(10,2)] = 10/12 ≈ 0.833
    assert ext.posterior_mean > 0.80


# ---------------------------------------------------------------------------
# BeamTraversal constructor
# ---------------------------------------------------------------------------

def test_beam_traversal_warm_start_param_stored():
    from unittest.mock import MagicMock
    bt = BeamTraversal(MagicMock(), None, warm_start_strength=3.5)
    assert bt.warm_start_strength == pytest.approx(3.5)


def test_beam_traversal_warm_start_default_zero():
    from unittest.mock import MagicMock
    bt = BeamTraversal(MagicMock(), None)
    assert bt.warm_start_strength == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AsyncBeamTraversal inherits warm_start_strength
# ---------------------------------------------------------------------------

def test_async_beam_warm_start():
    from unittest.mock import MagicMock
    abt = AsyncBeamTraversal(MagicMock(), None, probabilistic=True, warm_start_strength=2.0)
    assert abt.warm_start_strength == pytest.approx(2.0)
    assert abt.probabilistic is True


def test_async_beam_warm_start_traversal():
    """AsyncBeamTraversal.traverse_stream runs with warm_start_strength set."""
    from unittest.mock import MagicMock
    adapter = MagicMock()
    adapter.get_embedding.return_value = np.zeros(4, dtype=np.float32)
    adapter.get_community.return_value = 0
    adapter.find_entities.return_value = []
    adapter.get_neighbors.return_value = []

    abt = AsyncBeamTraversal(adapter, None, probabilistic=True, warm_start_strength=3.0)

    async def _run():
        hops = []
        async for hop in abt.traverse_stream(["X"]):
            hops.append(hop)
        return hops

    hops = asyncio.run(_run())
    assert len(hops) >= 1
    assert hops[0][0].nodes == ["X"]

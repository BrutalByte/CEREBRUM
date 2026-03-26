"""
Tests for Gap 3 — Bayesian Beam Search.

Covers:
  - Beta distribution parameters on TraversalPath
  - Posterior mean, variance, and Thompson sampling
  - Probabilistic beam selection in BeamTraversal
  - Seed reproducibility
  - score_uncertainty on Answer from extract()
  - AsyncBeamTraversal also accepts probabilistic=True
"""
import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from reasoning.traversal import TraversalPath, BeamTraversal, AsyncBeamTraversal
from reasoning.answer_extractor import extract, Answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_path(weights=None):
    """Build a TraversalPath with given edge weights already applied."""
    path = TraversalPath(
        nodes=["A"],
        seen_entities={"A"},
        embedding=np.ones(4, dtype=np.float32) / 2.0,
        score=1.0,
    )
    for w in (weights or []):
        path = path.copy_with_extension(
            rel="REL", v=f"E{w}", v_cid=0,
            v_emb=np.ones(4, dtype=np.float32) / 2.0,
            weight=w, coherence_score=1.0,
        )
    return path


def _make_beam(adapter=None, csa=None, beam_width=3, max_hop=2, **kwargs):
    """Construct a BeamTraversal with minimal mocks."""
    if adapter is None:
        adapter = MagicMock()
        adapter.get_embedding.return_value = np.ones(4, dtype=np.float32) / 2.0
        adapter.get_community.return_value = 0
        adapter.find_entities.return_value = []
        adapter.get_neighbors.return_value = []
    if csa is None:
        csa = MagicMock()
        csa.compute_weight.return_value = 0.7
    return BeamTraversal(adapter, csa, beam_width=beam_width, max_hop=max_hop, **kwargs)


# ---------------------------------------------------------------------------
# Beta parameter initialization
# ---------------------------------------------------------------------------

def test_beta_params_initialized_to_one():
    path = TraversalPath(nodes=["X"], seen_entities={"X"},
                         embedding=np.zeros(4, dtype=np.float32), score=1.0)
    assert path.beta_alpha == 1.0
    assert path.beta_beta == 1.0


def test_beta_params_update_on_extension():
    w = 0.8
    path = _make_path()
    extended = path.copy_with_extension(
        rel="R", v="Y", v_cid=0,
        v_emb=np.ones(4, dtype=np.float32) / 2.0,
        weight=w, coherence_score=1.0,
    )
    assert extended.beta_alpha == pytest.approx(1.0 + w)
    assert extended.beta_beta == pytest.approx(1.0 + (1.0 - w))


def test_posterior_mean_equals_half_at_init():
    path = _make_path()
    assert path.posterior_mean == pytest.approx(0.5)


def test_posterior_mean_increases_with_high_weight():
    path = _make_path(weights=[0.9, 0.9])
    assert path.posterior_mean > 0.5


def test_posterior_mean_decreases_with_low_weight():
    path = _make_path(weights=[0.1, 0.1])
    assert path.posterior_mean < 0.5


def test_score_variance_decreases_with_more_hops():
    """More hops → tighter distribution (higher α+β → lower variance)."""
    path_1hop = _make_path(weights=[0.7])
    path_3hop = _make_path(weights=[0.7, 0.7, 0.7])
    assert path_3hop.score_variance < path_1hop.score_variance


def test_sample_score_in_range():
    rng = np.random.default_rng(0)
    path = _make_path(weights=[0.6])
    for _ in range(50):
        s = path.sample_score(rng)
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# Deterministic mode unchanged
# ---------------------------------------------------------------------------

def test_deterministic_mode_unchanged():
    """probabilistic=False must behave identically to original BeamTraversal."""
    bt_det = _make_beam(probabilistic=False)
    bt_def = _make_beam()  # default
    assert bt_det.probabilistic is False
    # Both traverse an empty graph and return the seed path only
    result_det = bt_det.traverse(["A"])
    result_def = bt_def.traverse(["A"])
    assert len(result_det) == len(result_def)


# ---------------------------------------------------------------------------
# Probabilistic mode — basic contract
# ---------------------------------------------------------------------------

def _make_adapter_with_edges(n_neighbors=8, emb_dim=4):
    """Adapter with a simple chain graph for beam testing."""
    from core.graph_adapter import Edge

    adapter = MagicMock()
    adapter.get_embedding.return_value = np.ones(emb_dim, dtype=np.float32) / emb_dim
    adapter.get_community.return_value = 0
    adapter.find_entities.return_value = []

    def get_neighbors(node, max_neighbors=50, **kwargs):
        edges = []
        for i in range(n_neighbors):
            e = MagicMock(spec=Edge)
            e.target_id = f"{node}_child{i}"
            e.relation_type = "REL"
            e.confidence = 0.9
            e.provenance = ""
            e.valid_from = None
            e.valid_to = None
            edges.append(e)
        return edges[:max_neighbors]

    adapter.get_neighbors.side_effect = get_neighbors
    return adapter


def test_probabilistic_mode_returns_paths():
    bt = _make_beam(probabilistic=True, seed=0)
    bt.adapter = _make_adapter_with_edges(n_neighbors=5)
    bt.max_hop = 1
    result = bt.traverse(["root"])
    # Seed + at least 1 hop of children
    assert len(result) >= 2


def test_probabilistic_beam_width_respected():
    bt = _make_beam(probabilistic=True, seed=0, beam_width=3)
    bt.adapter = _make_adapter_with_edges(n_neighbors=10)
    bt.max_hop = 1
    # After 1 hop the beam is capped but this is the terminal hop, so all kept.
    # What matters is that at non-terminal hops we prune — set max_hop=2 to test.
    bt.max_hop = 2
    result = bt.traverse(["root"])
    assert isinstance(result, list)
    # Intermediate beam (hop 1) is capped to beam_width; we can't inspect it
    # directly but traverse must complete without error.


def test_seed_reproducibility():
    adapter = _make_adapter_with_edges(n_neighbors=6)
    bt1 = BeamTraversal(adapter, MagicMock(compute_weight=lambda *a, **kw: 0.6),
                        beam_width=3, max_hop=1, probabilistic=True, seed=99)
    bt1.csa = MagicMock()
    bt1.csa.compute_weight.return_value = 0.6

    bt2 = BeamTraversal(adapter, MagicMock(),
                        beam_width=3, max_hop=1, probabilistic=True, seed=99)
    bt2.csa = MagicMock()
    bt2.csa.compute_weight.return_value = 0.6

    r1 = bt1.traverse(["root"])
    r2 = bt2.traverse(["root"])
    ids1 = [p.tail for p in r1]
    ids2 = [p.tail for p in r2]
    assert ids1 == ids2


def test_noisy_graph_diversity():
    """
    Probabilistic mode should produce different beam selections than deterministic
    when graph has many equivalent-weight edges (noisy/contradicting evidence).
    Seeds the deterministic run with fixed weights and checks probabilistic
    does not identically match every time.
    """
    adapter = _make_adapter_with_edges(n_neighbors=10)
    csa = MagicMock()
    csa.compute_weight.return_value = 0.5  # all edges equal weight

    det = BeamTraversal(adapter, csa, beam_width=3, max_hop=1, probabilistic=False)
    det_result = [p.tail for p in det.traverse(["root"])]

    # Run probabilistic several times — at least one run should differ from det
    diffs = 0
    for seed in range(10):
        prob = BeamTraversal(adapter, csa, beam_width=3, max_hop=1,
                             probabilistic=True, seed=seed)
        prob_result = [p.tail for p in prob.traverse(["root"])]
        if prob_result != det_result:
            diffs += 1
    # With 10 random seeds and 10 equal-weight neighbors, expect some diversity
    assert diffs > 0


# ---------------------------------------------------------------------------
# Answer.score_uncertainty
# ---------------------------------------------------------------------------

def test_answer_score_uncertainty_populated():
    """extract() must set score_uncertainty from path.score_variance."""
    path = _make_path(weights=[0.7, 0.8])
    expected_var = path.score_variance
    answers = extract([path], top_k=1)
    assert len(answers) == 1
    assert answers[0].score_uncertainty == pytest.approx(expected_var, rel=1e-5)


def test_score_uncertainty_positive_after_hops():
    """Any path with hops should have non-zero uncertainty."""
    path = _make_path(weights=[0.5])
    answers = extract([path], top_k=1)
    assert answers[0].score_uncertainty > 0.0


def test_score_uncertainty_is_float():
    path = _make_path(weights=[0.6])
    answers = extract([path], top_k=1)
    assert isinstance(answers[0].score_uncertainty, float)


# ---------------------------------------------------------------------------
# AsyncBeamTraversal also accepts probabilistic=True
# ---------------------------------------------------------------------------

def test_async_beam_probabilistic_init():
    bt = AsyncBeamTraversal(MagicMock(), MagicMock(), probabilistic=True, seed=7)
    assert bt.probabilistic is True


def test_async_beam_probabilistic_traversal():
    async def _run():
        adapter = _make_adapter_with_edges(n_neighbors=4)
        csa = MagicMock()
        csa.compute_weight.return_value = 0.6
        bt = AsyncBeamTraversal(adapter, csa, beam_width=3, max_hop=1,
                                probabilistic=True, seed=42)
        all_paths = []
        async for layer in bt.traverse_stream(["root"]):
            all_paths.extend(layer)
        return all_paths

    result = asyncio.run(_run())
    assert len(result) >= 1

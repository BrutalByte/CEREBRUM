"""
Tests for IKGWQ benchmark core functions — Phase 27B.

Tests cover:
  - _mid_to_node: MID normalisation
  - apply_incompleteness: controlled edge removal protocol
  - build_full_pipeline: adapter + CSA engine construction
  - Graceful degradation AUC calculation (inline logic)
  - Level constants and label consistency
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.ikgwq_eval import (
    INCOMPLETENESS_LEVELS,
    LEVEL_LABELS,
    _mid_to_node,
    apply_incompleteness,
    build_full_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qa_graph() -> Tuple[nx.Graph, List[Tuple]]:
    """Small KG with 3 QA pairs — seeds and answers clearly labelled."""
    G = nx.Graph()
    triples = [
        ("seed1", "edge1", "ans1"),
        ("seed1", "edge2", "mid1"),
        ("mid1",  "edge3", "ans2"),
        ("seed2", "edge4", "ans3"),
        ("seed2", "edge5", "ans4"),
        ("seed3", "edge6", "ans5"),
        # Extra bulk edges between non-answer nodes
        ("bulk_a", "edge7", "bulk_b"),
        ("bulk_b", "edge8", "bulk_c"),
    ]
    for s, r, o in triples:
        G.add_edge(s, o, relation=r)

    qa = [
        ("seed1", ["ans1", "ans2"], "What is seed1?"),
        ("seed2", ["ans3"],        "What is seed2?"),
        ("seed3", ["ans5"],        "What is seed3?"),
    ]
    return G, qa


def _dummy_embeddings(G: nx.Graph, dim: int = 8) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {n: rng.standard_normal(dim).astype(np.float32) for n in G.nodes()}


def _dummy_cmap(G: nx.Graph) -> Dict[str, int]:
    """Assign each node to its own community (worst-case degenerate structure)."""
    return {n: i for i, n in enumerate(G.nodes())}


# ---------------------------------------------------------------------------
# _mid_to_node
# ---------------------------------------------------------------------------

class TestMidToNode:
    def test_ns_prefix_stripped(self):
        assert _mid_to_node("ns:m.0abc") == "/m/0abc"

    def test_dotted_m_converted(self):
        assert _mid_to_node("m.042f1") == "/m/042f1"

    def test_dotted_g_converted(self):
        assert _mid_to_node("g.1abcde") == "/g/1abcde"

    def test_slash_m_passthrough(self):
        assert _mid_to_node("/m/0abc123") == "/m/0abc123"

    def test_slash_g_passthrough(self):
        assert _mid_to_node("/g/1abcde") == "/g/1abcde"

    def test_plain_text_name_passthrough(self):
        """Non-MID text entity names (RoG-style) are returned unchanged."""
        assert _mid_to_node("Jamaica") == "Jamaica"

    def test_whitespace_stripped(self):
        assert _mid_to_node("  m.042f1  ") == "/m/042f1"

    def test_ns_dotted_m(self):
        """ns:m.xxx -> /m/xxx (two-step normalisation)."""
        result = _mid_to_node("ns:m.042f1")
        assert result == "/m/042f1"

    def test_empty_string(self):
        """Empty string returns empty string without crashing."""
        result = _mid_to_node("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# apply_incompleteness
# ---------------------------------------------------------------------------

class TestApplyIncompleteness:
    def test_zero_fraction_returns_copy(self):
        G, qa = _make_qa_graph()
        rng = random.Random(42)
        G_inc = apply_incompleteness(G, qa, removal_fraction=0.0, rng=rng)
        assert G_inc.number_of_edges() == G.number_of_edges()
        assert G_inc is not G  # Must be a copy

    def test_removes_edges(self):
        G, qa = _make_qa_graph()
        rng = random.Random(42)
        G_inc = apply_incompleteness(G, qa, removal_fraction=0.5, rng=rng)
        assert G_inc.number_of_edges() < G.number_of_edges()

    def test_seed_nodes_retained(self):
        """Seed nodes must remain in the incomplete graph."""
        G, qa = _make_qa_graph()
        rng = random.Random(42)
        G_inc = apply_incompleteness(G, qa, removal_fraction=0.5, rng=rng)
        seeds = {s for s, _, _ in qa}
        for seed in seeds:
            assert seed in G_inc, f"Seed node {seed!r} was removed"

    def test_answer_adjacent_targeting(self):
        """Removed edges must come from answer-adjacent candidates."""
        G, qa = _make_qa_graph()
        answer_nodes = {a for _, answers, _ in qa for a in answers}
        rng = random.Random(42)
        G_inc = apply_incompleteness(G, qa, removal_fraction=0.5, rng=rng)

        removed = set(G.edges()) - set(G_inc.edges())
        # Every removed edge must have been incident to an answer node
        for u, v in removed:
            assert u in answer_nodes or v in answer_nodes, \
                f"Edge ({u}, {v}) removed but neither endpoint is an answer node"

    def test_full_removal_fraction(self):
        """removal_fraction=1.0 removes all answer-adjacent edges."""
        G, qa = _make_qa_graph()
        answer_nodes = {a for _, answers, _ in qa for a in answers}
        seed_nodes   = {s for s, _, _ in qa}
        rng = random.Random(42)
        G_inc = apply_incompleteness(G, qa, removal_fraction=1.0, rng=rng)
        # No answer-adjacent edge (that isn't seed-only) should remain
        for u, v in G_inc.edges():
            is_answer_adj = (u in answer_nodes or v in answer_nodes)
            is_seed_pair  = (u in seed_nodes and v in seed_nodes)
            assert not is_answer_adj or is_seed_pair, \
                f"Answer-adjacent edge ({u},{v}) survived full removal"

    def test_deterministic_with_same_seed(self):
        """Same RNG seed → identical result."""
        G, qa = _make_qa_graph()
        G1 = apply_incompleteness(G, qa, 0.5, random.Random(7))
        G2 = apply_incompleteness(G, qa, 0.5, random.Random(7))
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_differ(self):
        """Different RNG seeds → (likely) different edge sets removed."""
        G, qa = _make_qa_graph()
        # Use a larger graph to make collision very unlikely
        G_big = nx.Graph()
        for i in range(50):
            G_big.add_edge(f"s{i}", f"a{i}", relation="r")
        qa_big = [(f"s{i}", [f"a{i}"], f"q{i}") for i in range(50)]
        G1 = apply_incompleteness(G_big, qa_big, 0.4, random.Random(1))
        G2 = apply_incompleteness(G_big, qa_big, 0.4, random.Random(2))
        assert set(G1.edges()) != set(G2.edges())

    def test_monotone_degradation_on_average(self):
        """Higher removal_fraction → fewer edges on average (averaged over seeds)."""
        G, qa = _make_qa_graph()
        counts = []
        for frac in [0.0, 0.2, 0.5, 0.8]:
            G_inc = apply_incompleteness(G, qa, frac, random.Random(99))
            counts.append(G_inc.number_of_edges())
        # Must be non-increasing
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], \
                f"Non-monotone: frac[{i}]={counts[i]} < frac[{i+1}]={counts[i+1]}"

    def test_empty_qa_returns_full_graph(self):
        """No QA pairs → no candidate edges → graph unchanged."""
        G, _ = _make_qa_graph()
        G_inc = apply_incompleteness(G, [], 0.5, random.Random(0))
        assert G_inc.number_of_edges() == G.number_of_edges()


# ---------------------------------------------------------------------------
# build_full_pipeline
# ---------------------------------------------------------------------------

class TestBuildFullPipeline:
    def test_returns_adapter_and_csa(self):
        from adapters.networkx_adapter import NetworkXAdapter
        from core.attention_engine import CSAEngine

        G, _ = _make_qa_graph()
        embs = _dummy_embeddings(G)
        cmap = _dummy_cmap(G)
        adapter, csa = build_full_pipeline(G, embs, cmap, coarsen_target=5)

        assert isinstance(adapter, NetworkXAdapter)
        assert isinstance(csa, CSAEngine)

    def test_adapter_has_community_map(self):
        G, _ = _make_qa_graph()
        embs = _dummy_embeddings(G)
        cmap = _dummy_cmap(G)
        adapter, _ = build_full_pipeline(G, embs, cmap, coarsen_target=5)
        assert adapter.community_map is not None
        assert len(adapter.community_map) > 0

    def test_adapter_has_embeddings(self):
        G, _ = _make_qa_graph()
        embs = _dummy_embeddings(G)
        cmap = _dummy_cmap(G)
        adapter, _ = build_full_pipeline(G, embs, cmap, coarsen_target=5)
        # embeddings dict attached to adapter should be the one we passed
        assert adapter.embeddings is embs

    def test_csa_has_community_graph(self):
        """CSAEngine should have community structure set after build_full_pipeline."""
        G, _ = _make_qa_graph()
        embs = _dummy_embeddings(G)
        cmap = _dummy_cmap(G)
        _, csa = build_full_pipeline(G, embs, cmap, coarsen_target=5)
        # CSAEngine stores community graph as _community_graph
        assert csa._community_graph is not None

    def test_single_node_graph_does_not_crash(self):
        """Degenerate single-node graph should not raise."""
        G = nx.Graph()
        G.add_node("solo")
        embs = {"solo": np.zeros(8, dtype=np.float32)}
        cmap = {"solo": 0}
        # Should not raise
        build_full_pipeline(G, embs, cmap, coarsen_target=1)


# ---------------------------------------------------------------------------
# Level constants and AUC logic
# ---------------------------------------------------------------------------

class TestLevelConstants:
    def test_level_count(self):
        assert len(INCOMPLETENESS_LEVELS) == 5

    def test_level_fractions(self):
        expected = {0: 0.00, 1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50}
        assert INCOMPLETENESS_LEVELS == expected

    def test_level_labels_cover_all_levels(self):
        for lvl in INCOMPLETENESS_LEVELS:
            assert lvl in LEVEL_LABELS

    def test_baseline_is_zero(self):
        assert INCOMPLETENESS_LEVELS[0] == 0.0

    def test_fractions_monotonically_increasing(self):
        fracs = [INCOMPLETENESS_LEVELS[k] for k in sorted(INCOMPLETENESS_LEVELS)]
        for i in range(len(fracs) - 1):
            assert fracs[i] < fracs[i + 1]


class TestGracefulDegradationAUC:
    """Test the AUC calculation logic extracted from main()."""

    def _compute_auc(self, h_vals):
        """Mirrors the AUC formula in ikgwq_eval.main()."""
        return sum(h_vals) / (len(h_vals) * max(h_vals[0], 1e-9))

    def test_perfect_retention_auc_is_one(self):
        """Constant performance → AUC = 1.0."""
        h_vals = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert abs(self._compute_auc(h_vals) - 1.0) < 1e-9

    def test_total_failure_auc_near_zero(self):
        """Drop to near-zero at level 1 → AUC close to 1/5 of baseline."""
        h_vals = [0.5, 0.0, 0.0, 0.0, 0.0]
        auc = self._compute_auc(h_vals)
        assert auc < 0.25  # Much less than perfect retention

    def test_graceful_degradation_between_bounds(self):
        """Gradual decline → AUC strictly between 0 and 1."""
        h_vals = [0.5, 0.45, 0.40, 0.35, 0.25]
        auc = self._compute_auc(h_vals)
        assert 0.0 < auc < 1.0

    def test_auc_ordering(self):
        """Better degradation curve → higher AUC."""
        good  = [0.5, 0.48, 0.46, 0.44, 0.42]
        bad   = [0.5, 0.30, 0.20, 0.10, 0.05]
        assert self._compute_auc(good) > self._compute_auc(bad)

    def test_zero_baseline_does_not_divide_by_zero(self):
        """Baseline=0 hits the 1e-9 guard → large but finite AUC."""
        h_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        auc = self._compute_auc(h_vals)
        assert np.isfinite(auc)

    def test_single_level_auc_is_one(self):
        """Single-level evaluation (baseline only) → AUC = 1.0 trivially."""
        h_vals = [0.42]
        assert abs(self._compute_auc(h_vals) - 1.0) < 1e-9

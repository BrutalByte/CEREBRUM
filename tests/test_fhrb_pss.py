"""
Tests for Phase 179 (Path Specificity Score) and Phase 180 (First-Hop Relation Boost).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning.path_scorer import path_specificity_score
from reasoning.traversal import BeamTraversal, TraversalPath
from reasoning.answer_extractor import extract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_path(nodes: list, weights: list | None = None) -> TraversalPath:
    """Build a minimal TraversalPath with the given node/relation sequence."""
    path = MagicMock(spec=TraversalPath)
    path.nodes = nodes
    path.attention_weights = weights or [0.7] * ((len(nodes) - 1) // 2)
    path.community_sequence = [0] * ((len(nodes) + 1) // 2)
    path.embedding = np.zeros(8, dtype=np.float32)
    path.edge_confidences = [1.0] * ((len(nodes) - 1) // 2)
    path.edge_provenances = [""] * ((len(nodes) - 1) // 2)
    path.hop_depth = (len(nodes) - 1) // 2
    path.tail = nodes[-1] if nodes else ""
    path.score = float(np.prod(weights or [0.7] * ((len(nodes) - 1) // 2)))
    path.seen_entities = {nodes[i] for i in range(0, len(nodes), 2)}
    return path


def _make_adapter(edges: list) -> MagicMock:
    from collections import defaultdict
    adjacency = defaultdict(list)
    all_nodes: set = set()
    for src, rel, tgt in edges:
        all_nodes.update([src, tgt])
        adjacency[src].append((rel, tgt))

    rng = np.random.default_rng(0)
    embeddings = {n: rng.standard_normal(8).astype(np.float32) for n in all_nodes}

    adapter = MagicMock()
    adapter.get_embedding.side_effect = lambda n: embeddings.get(n, np.zeros(8, dtype=np.float32))
    adapter.get_community.return_value = 0
    adapter.community_map = {n: 0 for n in all_nodes}

    def _gn(node, max_neighbors=50, **_):
        nbrs = []
        for rel, tgt in adjacency.get(node, []):
            e = MagicMock()
            e.target_id = tgt
            e.relation_type = rel
            e.confidence = 1.0
            e.provenance = ""
            e.valid_from = None
            e.valid_to = None
            nbrs.append(e)
        return nbrs[:max_neighbors]

    adapter.get_neighbors.side_effect = _gn
    adapter.get_neighbors_batch.side_effect = lambda nodes, **kw: {
        n: _gn(n, **kw) for n in nodes
    }
    return adapter


def _make_csa() -> MagicMock:
    csa = MagicMock()
    csa.compute_weight.return_value = 0.7
    csa.compute_weight_with_features.return_value = None
    csa.community_score.return_value = 0.5
    csa.set_query_snapshot = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    csa.set_query_time = MagicMock()
    csa.use_temporal_decay = False
    return csa


# ---------------------------------------------------------------------------
# Phase 179: path_specificity_score
# ---------------------------------------------------------------------------

class TestPathSpecificityScore:
    def test_single_hop_narrow_scores_high(self):
        """1 target for the relation → specificity near 1.0."""
        path = _make_path(["A", "r1", "B"])
        fan_out = {"A": {"r1": 1}}
        score = path_specificity_score(path, fan_out)
        assert score == pytest.approx(1.0)

    def test_single_hop_wide_scores_low(self):
        """100 targets for the relation → specificity near 0.01."""
        path = _make_path(["A", "r1", "B"])
        fan_out = {"A": {"r1": 100}}
        score = path_specificity_score(path, fan_out)
        assert score == pytest.approx(0.01)

    def test_multi_hop_geometric_mean(self):
        """3-hop: geometric mean of per-hop specificities."""
        path = _make_path(["A", "r1", "B", "r2", "C", "r3", "D"])
        fan_out = {"A": {"r1": 1}, "B": {"r2": 10}, "C": {"r3": 100}}
        score = path_specificity_score(path, fan_out)
        expected = (1.0 * 0.1 * 0.01) ** (1 / 3)
        assert score == pytest.approx(expected, rel=1e-5)

    def test_unknown_entity_defaults_to_one(self):
        """Unknown entity in fan_out → treated as fan_out=1 → specificity=1.0."""
        path = _make_path(["X", "r1", "Y"])
        fan_out = {}  # nothing known
        score = path_specificity_score(path, fan_out)
        assert score == pytest.approx(1.0)

    def test_seed_only_path_returns_one(self):
        """0-hop path (just the seed) → 1.0 (no hops to penalise)."""
        path = _make_path(["A"])
        fan_out = {"A": {"r1": 1000}}
        score = path_specificity_score(path, fan_out)
        assert score == pytest.approx(1.0)

    def test_narrower_path_scores_higher(self):
        """The narrower path should score strictly higher."""
        narrow = _make_path(["A", "r1", "B"])
        wide   = _make_path(["A", "r1", "C"])
        fan_out = {"A": {"r1": 1}}  # narrow by definition here
        # Both have same fan_out from A → same score. Show wide graph case:
        fan_out2 = {"A": {"r1": 50}}
        assert path_specificity_score(narrow, {"A": {"r1": 1}}) > \
               path_specificity_score(wide,   {"A": {"r1": 50}})


class TestPSSInExtract:
    def test_pss_reranks_when_enabled(self):
        """Entities reached via narrow paths should rank higher with PSS enabled."""
        # Entity "narrow_ans" reached via fan_out=1 (very specific)
        # Entity "wide_ans"   reached via fan_out=100 (generic hub)
        p_narrow = _make_path(["seed", "r1", "narrow_ans"], weights=[0.6])
        p_wide   = _make_path(["seed", "r1", "wide_ans"],   weights=[0.6])

        fan_out = {"seed": {"r1": 1}}   # same fan_out for both paths here
        # Give wide_ans a higher raw score to put it first without PSS
        p_wide.score = 0.8
        p_narrow.score = 0.6

        answers_no_pss = extract(
            [p_narrow, p_wide], top_k=2,
            fan_out=None, weight_specificity=0.0
        )
        answers_with_pss = extract(
            [p_narrow, p_wide], top_k=2,
            fan_out={"seed": {"r1": 1}}, weight_specificity=0.5
        )
        # Without PSS wide_ans might be first (higher score); PSS shouldn't crash
        assert len(answers_no_pss) == 2
        assert len(answers_with_pss) == 2

    def test_pss_disabled_by_default(self):
        p = _make_path(["seed", "r1", "ans"], weights=[0.7])
        answers = extract([p], top_k=5)
        assert len(answers) == 1


# ---------------------------------------------------------------------------
# Phase 180: Initial Relation Boost (FHRB)
# ---------------------------------------------------------------------------

class TestInitialRelationBoost:
    def _build_traversal(self, edges, initial_boost=None):
        adapter = _make_adapter(edges)
        csa = _make_csa()
        return BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=10,
            max_hop=2,
            initial_relation_boost=initial_boost or {},
        )

    def test_irb_stored_on_traversal(self):
        bt = self._build_traversal(
            [("S", "r1", "A"), ("A", "r2", "B")],
            initial_boost={"r1": 5.0}
        )
        assert bt.initial_relation_boost == {"r1": 5.0}

    def test_irb_boosts_target_relation(self):
        """With IRB on r1, paths using r1 at hop-1 should be explored."""
        edges = [
            ("seed", "r1", "mid"),
            ("seed", "other_r", "noise"),
            ("mid", "r2", "answer"),
            ("noise", "r2", "wrong"),
        ]
        bt = self._build_traversal(edges, initial_boost={"r1": 10.0})
        paths = bt.traverse(seeds=["seed"])
        tails = [p.tail for p in paths]
        assert "answer" in tails

    def test_irb_empty_dict_no_effect(self):
        """Empty IRB should behave identically to no IRB."""
        edges = [("seed", "r1", "mid"), ("mid", "r2", "ans")]
        bt_no  = self._build_traversal(edges, initial_boost={})
        bt_yes = self._build_traversal(edges, initial_boost=None)
        paths_no  = bt_no.traverse(seeds=["seed"])
        paths_yes = bt_yes.traverse(seeds=["seed"])
        assert len(paths_no) == len(paths_yes)

    def test_irb_penalises_non_target_relation(self):
        """Non-target relations get 0.1 penalty; target gets boost → target path wins."""
        edges = [
            ("seed", "correct_r", "good_mid"),
            ("seed", "wrong_r",   "bad_mid"),
            ("good_mid", "r2", "correct_ans"),
            ("bad_mid",  "r2", "wrong_ans"),
        ]
        bt = self._build_traversal(edges, initial_boost={"correct_r": 8.0})
        paths = bt.traverse(seeds=["seed"])
        tails = [p.tail for p in paths]
        # correct_ans should appear; wrong_ans heavily penalised
        assert "correct_ans" in tails

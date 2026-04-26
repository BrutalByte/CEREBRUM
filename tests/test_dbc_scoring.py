"""
Phase 144: Distinct-Branch Convergence (DBC) Scoring tests.

Tests that branch_bonus_weight correctly rewards answers reached via
multiple distinct first-intermediate-node (hop-1 branch) paths.

Note: score_path() recomputes scores from attention_weights (prod),
NOT from path.score. Use the attn= parameter to control which answers
rank higher in the score_path output.
"""
import math
import numpy as np
import pytest

from reasoning.answer_extractor import extract
from reasoning.traversal import TraversalPath


def _make_path(nodes, attn=0.5, hop_depth=None):
    """Build a minimal TraversalPath for testing.

    attn: per-edge attention weight. score_path uses prod(attn_weights),
    so higher attn → higher extracted score. Default 0.5.
    """
    if hop_depth is None:
        hop_depth = (len(nodes) - 1) // 2
    n_edges = max(hop_depth, 1)
    aw = [attn] * n_edges
    return TraversalPath(
        nodes=nodes,
        seen_entities=set(nodes[::2]),
        embedding=np.zeros(8, dtype=np.float32),
        score=attn,
        attention_weights=aw,
        community_sequence=[0] * n_edges,
        edge_confidences=[1.0] * n_edges,
        edge_provenances=["test"] * n_edges,
        edge_features=[{}] * n_edges,
        beta_alpha=1.0,
        beta_beta=1.0,
    )


class TestSingleBranchNoBonus:
    """5 paths all via the same nodes[2] → branch_count=1 → no branch multiplier."""

    def test_single_branch_no_bonus(self):
        paths = [
            _make_path(["seed", "r1", "HOP1", "r2", "C"])
            for _ in range(5)
        ]
        answers = extract(paths, top_k=5, min_hop=2, branch_bonus_weight=0.25)
        assert len(answers) == 1
        ans = answers[0]
        assert ans.entity_id == "C"
        assert ans.branch_count == 1
        # With branch_count=1, no bonus is applied — score equals bonus-disabled result
        answers_no_bonus = extract(paths, top_k=5, min_hop=2, branch_bonus_weight=0.0)
        assert abs(answers[0].score - answers_no_bonus[0].score) < 1e-9


class TestMultiBranchBonusApplied:
    """3 paths via different nodes[2] → branch_count=3 → log-scale bonus applied."""

    def test_multi_branch_bonus_applied(self):
        paths = [
            _make_path(["seed", "r1", "HOP_A", "r2", "C"]),
            _make_path(["seed", "r1", "HOP_B", "r2", "C"]),
            _make_path(["seed", "r1", "HOP_C", "r2", "C"]),
        ]
        answers = extract(paths, top_k=5, min_hop=2, branch_bonus_weight=0.25, vote_weight=0.0)
        assert len(answers) == 1
        ans = answers[0]
        assert ans.branch_count == 3
        # branch_factor = 1 + 0.25 * log1p(2) ≈ 1.275 — score must exceed path_score by >10%
        assert ans.score > ans.path_score * 1.1


class TestBranchBonusWeightZeroDisabled:
    """branch_bonus_weight=0.0 → D (higher attn score) beats C (multi-branch)."""

    def test_branch_bonus_weight_zero_disabled(self):
        # D has higher attention weight → higher score_path result
        paths_c = [
            _make_path(["seed", "r1", "HOP_A", "r2", "C"], attn=0.6),
            _make_path(["seed", "r1", "HOP_B", "r2", "C"], attn=0.6),
        ]
        paths_d = [
            _make_path(["seed", "r1", "HOP_X", "r2", "D"], attn=0.8),
        ]
        all_paths = paths_c + paths_d

        # With both weights at 0: D wins (higher attn score)
        answers_no_bonus = extract(all_paths, top_k=5, min_hop=2,
                                   branch_bonus_weight=0.0, vote_weight=0.0)
        assert answers_no_bonus[0].entity_id == "D"

        # With branch bonus: branch_count tracked correctly regardless of ranking
        answers_with_bonus = extract(all_paths, top_k=5, min_hop=2,
                                     branch_bonus_weight=0.25, vote_weight=0.0)
        c_ans = next(a for a in answers_with_bonus if a.entity_id == "C")
        d_ans = next(a for a in answers_with_bonus if a.entity_id == "D")
        assert c_ans.branch_count == 2
        assert d_ans.branch_count == 1


class TestH1SEIntegrationBoostsConvergentAnswer:
    """C reached via 2 independent hop-1 branches; D via 1 branch but higher attn.

    With branch_bonus_weight=0.25, the branch bonus on C must overcome D's
    higher per-path attention score.
    """

    def test_h1se_integration_boosts_convergent_answer(self):
        # D has attn=0.75 → score_path ≈ 0.75^2 at 2-hop = 0.5625
        # C has attn=0.70 → score_path ≈ 0.70^2 = 0.49, BUT 2 branches
        # branch_factor for C = 1 + 0.25 * log1p(1) ≈ 1.173
        # C effective ≈ 0.49 * 1.173 ≈ 0.575 > D's 0.5625 → C should win
        paths = [
            _make_path(["seed", "r1", "B1", "r2", "C"], attn=0.70),
            _make_path(["seed", "r1", "B2", "r2", "C"], attn=0.70),
            _make_path(["seed", "r1", "B3", "r2", "D"], attn=0.75),
        ]

        answers = extract(paths, top_k=5, min_hop=2, branch_bonus_weight=0.25, vote_weight=0.0)
        assert answers[0].entity_id == "C", f"Expected C first, got {[a.entity_id for a in answers]}"
        assert answers[0].branch_count == 2
        assert answers[1].entity_id == "D"
        assert answers[1].branch_count == 1

        # Without branch bonus: D should win (higher attn score)
        answers_no_bonus = extract(paths, top_k=5, min_hop=2,
                                   branch_bonus_weight=0.0, vote_weight=0.0)
        assert answers_no_bonus[0].entity_id == "D"

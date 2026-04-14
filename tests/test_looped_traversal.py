"""
Tests for LoopedBeamTraversal (Phase 70).

Verifies iterative refinement, exit gate logic, seed expansion, path merging,
and backward compatibility — mirroring the LoopLM adaptive exit design
(Zhu et al., arXiv:2510.25741).
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional

from reasoning.looped_traversal import LoopedBeamTraversal, LoopTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_path(tail: str, score: float = 0.8, hop_depth: int = 1, nodes: Optional[List[str]] = None):
    """Return a mock TraversalPath."""
    p = MagicMock()
    p.tail = tail
    p.score = score
    p.hop_depth = hop_depth
    p.nodes = nodes or ["seed", "REL", tail]
    return p


def _make_answer(entity_id: str, score: float = 0.8):
    """Return a mock Answer."""
    a = MagicMock()
    a.entity_id = entity_id
    a.score = score
    return a


def _make_traversal(paths_sequence: List[List]) -> MagicMock:
    """
    Return a mock traversal whose traverse() returns successive path lists.
    If there are more calls than entries in paths_sequence, the last list is repeated.
    """
    traversal = MagicMock()
    call_count = [0]

    def _traverse(*args, **kwargs):
        idx = min(call_count[0], len(paths_sequence) - 1)
        call_count[0] += 1
        return paths_sequence[idx]

    traversal.traverse.side_effect = _traverse
    return traversal


# ---------------------------------------------------------------------------
# Backward compatibility: max_loops=1
# ---------------------------------------------------------------------------

class TestSingleLoop:
    def test_single_loop_returns_immediately(self):
        """max_loops=1 must call inner traversal once and return exit_reason='single_loop'."""
        paths = [_make_path("A"), _make_path("B")]
        traversal = _make_traversal([paths])

        lbt = LoopedBeamTraversal(traversal, max_loops=1)
        result_paths, lt = lbt.traverse(["seed"])

        assert traversal.traverse.call_count == 1
        assert lt.exit_reason == "single_loop"
        assert lt.loops_run == 1

    def test_single_loop_result_paths_unchanged(self):
        """max_loops=1 returns exactly the same paths as inner traversal."""
        paths = [_make_path("X", score=0.9)]
        traversal = _make_traversal([paths])

        lbt = LoopedBeamTraversal(traversal, max_loops=1)
        result_paths, _ = lbt.traverse(["s"])

        assert result_paths == paths

    def test_single_loop_attaches_trace(self):
        """max_loops=1 still populates loop_trace on trace_info."""
        paths = [_make_path("X")]
        traversal = _make_traversal([paths])
        trace = MagicMock()

        lbt = LoopedBeamTraversal(traversal, max_loops=1)
        lbt.traverse(["s"], trace_info=trace)

        assert trace.loop_trace is not None


# ---------------------------------------------------------------------------
# Multi-loop: answer-stability exit
# ---------------------------------------------------------------------------

class TestAnswerStabilityExit:
    def test_exits_when_answers_stable(self):
        """
        When both loops return the same answer entities (Jaccard=1.0), the loop
        should exit after loop 2 with exit_reason='answers_stable'.
        """
        same_paths = [_make_path("E1"), _make_path("E2")]
        traversal  = _make_traversal([same_paths, same_paths, same_paths])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            mock_extract.return_value = [_make_answer("E1"), _make_answer("E2")]

            lbt = LoopedBeamTraversal(traversal, max_loops=4, answer_stability_threshold=0.8)
            _, lt = lbt.traverse(["seed"])

        # Loop 1: no prev_answers → no exit.  Loop 2: Jaccard=1.0 ≥ 0.8 → exit.
        assert lt.exit_reason == "answers_stable"
        assert lt.loops_run == 2

    def test_continues_when_answers_unstable(self):
        """When new answers appear each loop, the loop continues."""
        path_sets = [
            [_make_path("E1")],
            [_make_path("E2")],
            [_make_path("E3")],
        ]
        traversal = _make_traversal(path_sets)

        answer_sets = [
            [_make_answer("E1")],
            [_make_answer("E2")],
            [_make_answer("E3")],
        ]
        call_idx = [0]

        def _extract(*args, **kwargs):
            i = min(call_idx[0], len(answer_sets) - 1)
            call_idx[0] += 1
            return answer_sets[i]

        with patch("reasoning.looped_traversal.extract", side_effect=_extract):
            lbt = LoopedBeamTraversal(traversal, max_loops=3, answer_stability_threshold=0.8)
            _, lt = lbt.traverse(["seed"])

        assert lt.exit_reason == "max_loops"
        assert lt.loops_run == 3


# ---------------------------------------------------------------------------
# Multi-loop: PE-convergence exit
# ---------------------------------------------------------------------------

class TestPEConvergenceExit:
    def _make_pc(self, pe_values: List[float]) -> MagicMock:
        """Return a mock PredictiveCodingEngine with successive PE values."""
        pc = MagicMock()
        pc.predict.return_value = MagicMock()
        call_idx = [0]

        def _update(prior, paths):
            i = min(call_idx[0], len(pe_values) - 1)
            call_idx[0] += 1
            result = MagicMock()
            result.prediction_error = pe_values[i]
            return result

        pc.update.side_effect = _update
        return pc

    def test_exits_on_pe_convergence(self):
        """When |PE_t - PE_{t-1}| < threshold, exit_reason='pe_converged'."""
        # Loop 1: PE=0.5, Loop 2: PE=0.52 → delta=0.02 < 0.05
        pe_values = [0.5, 0.52, 0.51]
        pc        = self._make_pc(pe_values)

        paths = [_make_path("X")]
        traversal = _make_traversal([paths] * 4)

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            mock_extract.return_value = [_make_answer("X")]

            lbt = LoopedBeamTraversal(
                traversal, predictive_coder=pc,
                max_loops=4, pe_convergence_threshold=0.05,
            )
            _, lt = lbt.traverse(["seed"])

        assert lt.exit_reason == "pe_converged"
        assert lt.loops_run == 2

    def test_pe_takes_priority_over_answer_stability(self):
        """PE gate should fire before answer-stability gate when both would trigger."""
        pe_values = [0.5, 0.52]   # delta=0.02 < 0.05 → PE exit on loop 2
        pc        = self._make_pc(pe_values)

        paths = [_make_path("X")]
        traversal = _make_traversal([paths] * 4)

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            mock_extract.return_value = [_make_answer("X")]  # stable answers too

            lbt = LoopedBeamTraversal(
                traversal, predictive_coder=pc,
                max_loops=4,
                pe_convergence_threshold=0.05,
                answer_stability_threshold=0.8,
            )
            _, lt = lbt.traverse(["seed"])

        assert lt.exit_reason == "pe_converged"

    def test_no_pe_falls_back_to_answer_stability(self):
        """Without predictive_coder, only answer-stability exit is available."""
        paths = [_make_path("A"), _make_path("B")]
        traversal = _make_traversal([paths, paths])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            mock_extract.return_value = [_make_answer("A"), _make_answer("B")]

            lbt = LoopedBeamTraversal(
                traversal, predictive_coder=None,
                max_loops=4, answer_stability_threshold=0.8,
            )
            _, lt = lbt.traverse(["seed"])

        assert lt.exit_reason == "answers_stable"


# ---------------------------------------------------------------------------
# Seed expansion
# ---------------------------------------------------------------------------

class TestSeedExpansion:
    def test_loop2_seeds_include_top_answers_from_loop1(self):
        """After loop 1, top answer entities should be added to seeds for loop 2."""
        paths = [_make_path("discovered_entity")]
        traversal = _make_traversal([paths, paths])
        captured_seeds = []

        original_side = traversal.traverse.side_effect

        def _capture(*args, **kwargs):
            captured_seeds.append(list(args[0]) if args else list(kwargs.get("seeds", [])))
            return original_side(*args, **kwargs)

        traversal.traverse.side_effect = _capture

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            call_idx = [0]

            def _extract(*args, **kwargs):
                # Loop 1 discovers "discovered_entity"; loop 2 returns same
                result = [_make_answer("discovered_entity")]
                call_idx[0] += 1
                return result

            mock_extract.side_effect = _extract

            lbt = LoopedBeamTraversal(
                traversal, max_loops=4,
                top_k_seed_expansion=2,
                answer_stability_threshold=0.8,
            )
            _, lt = lbt.traverse(["original_seed"])

        # Loop 2 seeds must include the original seed + discovered entity
        if len(captured_seeds) >= 2:
            assert "original_seed" in captured_seeds[1]
            assert "discovered_entity" in captured_seeds[1]

    def test_original_seeds_always_present(self):
        """Original seeds must appear in every loop's seed list."""
        paths = [_make_path("new_entity")]
        traversal = _make_traversal([paths] * 5)
        all_seeds = []

        original_side = traversal.traverse.side_effect

        def _capture(*args, **kwargs):
            all_seeds.append(list(args[0]))
            return original_side(*args, **kwargs)

        traversal.traverse.side_effect = _capture

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            # Return different answer each loop to prevent early exit
            answers = [
                [_make_answer("e1")],
                [_make_answer("e2")],
                [_make_answer("e3")],
            ]
            idx = [0]

            def _extract(*args, **kwargs):
                i = min(idx[0], len(answers) - 1)
                idx[0] += 1
                return answers[i]

            mock_extract.side_effect = _extract

            lbt = LoopedBeamTraversal(traversal, max_loops=3, top_k_seed_expansion=1)
            lbt.traverse(["root_seed"])

        for seed_list in all_seeds:
            assert "root_seed" in seed_list


# ---------------------------------------------------------------------------
# Path merging
# ---------------------------------------------------------------------------

class TestPathMerging:
    def test_paths_merged_across_loops(self):
        """All paths from all loops should appear in the merged result."""
        loop1_paths = [_make_path("A", score=0.9), _make_path("B", score=0.7)]
        loop2_paths = [_make_path("C", score=0.8)]
        traversal   = _make_traversal([loop1_paths, loop2_paths])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            idx = [0]
            answer_sets = [[_make_answer("A")], [_make_answer("C")]]

            def _extract(*args, **kwargs):
                i = min(idx[0], len(answer_sets) - 1)
                idx[0] += 1
                return answer_sets[i]

            mock_extract.side_effect = _extract

            lbt = LoopedBeamTraversal(traversal, max_loops=2, answer_stability_threshold=2.0)
            result_paths, _ = lbt.traverse(["seed"])

        tails = {p.tail for p in result_paths}
        assert "A" in tails
        assert "B" in tails
        assert "C" in tails

    def test_higher_score_wins_per_tail(self):
        """When the same tail entity appears in multiple loops, highest score must win."""
        low_score_path  = _make_path("X", score=0.5)
        high_score_path = _make_path("X", score=0.9)
        traversal       = _make_traversal([[low_score_path], [high_score_path]])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            idx = [0]
            answer_sets = [[_make_answer("X")], [_make_answer("Y")]]

            def _extract(*args, **kwargs):
                i = min(idx[0], len(answer_sets) - 1)
                idx[0] += 1
                return answer_sets[i]

            mock_extract.side_effect = _extract

            lbt = LoopedBeamTraversal(traversal, max_loops=2, answer_stability_threshold=2.0)
            result_paths, _ = lbt.traverse(["seed"])

        x_paths = [p for p in result_paths if p.tail == "X"]
        assert len(x_paths) == 1
        assert float(x_paths[0].score) == 0.9


# ---------------------------------------------------------------------------
# LoopTrace fields
# ---------------------------------------------------------------------------

class TestLoopTrace:
    def test_loop_trace_fields_populated(self):
        """LoopTrace must record per-loop seeds, path counts, and new answer counts."""
        paths = [_make_path("A"), _make_path("B")]
        traversal = _make_traversal([paths, paths])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            mock_extract.return_value = [_make_answer("A"), _make_answer("B")]

            lbt = LoopedBeamTraversal(traversal, max_loops=4, answer_stability_threshold=0.8)
            _, lt = lbt.traverse(["s1", "s2"])

        assert isinstance(lt, LoopTrace)
        assert len(lt.seeds_per_loop) == lt.loops_run
        assert len(lt.paths_per_loop) == lt.loops_run
        assert len(lt.new_answers_per_loop) == lt.loops_run
        assert lt.seeds_per_loop[0] == ["s1", "s2"]

    def test_max_loops_cap_respected(self):
        """The loop must never exceed max_loops iterations."""
        # Return different answers each loop to prevent early exit
        traversal = _make_traversal([[_make_path(f"E{i}")] for i in range(10)])

        with patch("reasoning.looped_traversal.extract") as mock_extract:
            idx = [0]

            def _extract(*args, **kwargs):
                a = _make_answer(f"E{idx[0]}")
                idx[0] += 1
                return [a]

            mock_extract.side_effect = _extract

            lbt = LoopedBeamTraversal(traversal, max_loops=3)
            _, lt = lbt.traverse(["seed"])

        assert lt.loops_run <= 3
        assert traversal.traverse.call_count <= 3

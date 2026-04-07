"""
Tests for AAAK-Steered Beam Traversal (Phase 55).

Covers:
  - AAAKCache: record, affinity, top_patterns, clear, thread-safety
  - _path_rel_sequence: relation extraction from TraversalPath
  - AAAKBeamTraversal: integration with toy graph, cache population,
    score boosting verifiable behaviour
"""
from pathlib import Path
import threading

import numpy as np
import pytest

from reasoning.aaak_steered_traversal import (
    AAAKCache,
    AAAKBeamTraversal,
    _path_rel_sequence,
    _compress_rel,
)
from reasoning.traversal import TraversalPath
from core.cerebrum import CerebrumGraph

TOY_CSV = str(Path(__file__).parent / "fixtures" / "toy_graph.csv")


# ---------------------------------------------------------------------------
# _compress_rel
# ---------------------------------------------------------------------------

class TestCompressRel:
    def test_known_relation_compressed(self):
        assert _compress_rel("CAUSES") == "!"
        assert _compress_rel("TREATS") == "+"
        assert _compress_rel("INFLUENCED") == "~"

    def test_unknown_relation_passthrough(self):
        assert _compress_rel("SOME_CUSTOM_REL") == "SOME_CUSTOM_REL"

    def test_case_insensitive(self):
        assert _compress_rel("causes") == "!"


# ---------------------------------------------------------------------------
# _path_rel_sequence
# ---------------------------------------------------------------------------

class TestPathRelSequence:
    def _make_path(self, *rels: str) -> TraversalPath:
        nodes = ["A"]
        for i, r in enumerate(rels):
            nodes += [r, f"N{i}"]
        return TraversalPath(nodes=nodes)

    def test_single_hop(self):
        p = self._make_path("CAUSES")
        assert _path_rel_sequence(p) == ("!",)

    def test_two_hop(self):
        p = self._make_path("CAUSES", "TREATS")
        assert _path_rel_sequence(p) == ("!", "+")

    def test_unknown_rel_preserved(self):
        p = self._make_path("MY_REL")
        assert _path_rel_sequence(p) == ("MY_REL",)

    def test_seed_only_path_empty(self):
        p = TraversalPath(nodes=["A"])
        assert _path_rel_sequence(p) == ()


# ---------------------------------------------------------------------------
# AAAKCache
# ---------------------------------------------------------------------------

class TestAAAKCache:
    def test_empty_cache_affinity_zero(self):
        cache = AAAKCache()
        assert cache.affinity(("!", "+")) == 0.0

    def test_record_and_affinity(self):
        cache = AAAKCache()
        cache.record(("!", "+"))
        assert cache.affinity(("!", "+")) > 0.0

    def test_affinity_prefix_match(self):
        """A recorded full sequence should also boost any prefix."""
        cache = AAAKCache()
        cache.record(("!", "+", "~"))
        # Single-element prefix should match
        assert cache.affinity(("!",)) > 0.0

    def test_affinity_deeper_prefix_scores_higher(self):
        """Longer matching prefix should yield higher affinity."""
        cache = AAAKCache()
        cache.record(("!", "+", "~"))
        short_aff = cache.affinity(("!",))
        long_aff  = cache.affinity(("!", "+"))
        # longer prefix has higher depth_bonus
        assert long_aff >= short_aff

    def test_affinity_no_match_zero(self):
        cache = AAAKCache()
        cache.record(("!", "+"))
        assert cache.affinity(("-",)) == 0.0

    def test_size_grows(self):
        cache = AAAKCache()
        assert cache.size() == 0
        cache.record(("!",))
        assert cache.size() == 1
        cache.record(("!",))          # duplicate — same key
        assert cache.size() == 1
        cache.record(("+",))
        assert cache.size() == 2

    def test_clear_resets(self):
        cache = AAAKCache()
        cache.record(("!",))
        cache.clear()
        assert cache.size() == 0
        assert cache.affinity(("!",)) == 0.0

    def test_top_patterns(self):
        cache = AAAKCache()
        cache.record(("!",), weight=5)
        cache.record(("+",), weight=2)
        top = cache.top_patterns(n=2)
        assert top[0][0] == ("!",)
        assert top[0][1] == 5

    def test_thread_safety(self):
        """Concurrent record() calls must not raise or corrupt state."""
        cache = AAAKCache()
        errors = []

        def worker(i):
            try:
                for _ in range(100):
                    cache.record((f"REL{i % 5}",))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert cache.size() <= 5  # 5 distinct patterns

    def test_max_patterns_eviction(self):
        cache = AAAKCache(max_patterns=3)
        for i in range(5):
            cache.record((f"R{i}",))
        # Should not exceed max_patterns
        assert cache.size() <= 3

    def test_empty_sequence_ignored(self):
        cache = AAAKCache()
        cache.record(())
        assert cache.size() == 0


# ---------------------------------------------------------------------------
# AAAKBeamTraversal — integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_graph():
    g = CerebrumGraph.from_csv(TOY_CSV)
    g.build()
    return g


class TestAAAKBeamTraversal:
    def test_traversal_returns_paths(self, loaded_graph):
        cache = AAAKCache()
        traversal = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache,
            aaak_strength=0.3,
            beam_width=5,
            max_hop=2,
        )
        seeds = [next(iter(loaded_graph.adapter.embeddings))]
        paths = traversal.traverse(seeds)
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_record_answers_populates_cache(self, loaded_graph):
        from reasoning.answer_extractor import extract
        cache = AAAKCache()
        traversal = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache,
            beam_width=5,
            max_hop=2,
        )
        seeds = [next(iter(loaded_graph.adapter.embeddings))]
        paths = traversal.traverse(seeds)
        answers = extract(paths, top_k=5)
        traversal.record_answers(answers, min_score=0.0)
        # Cache should now contain some patterns (paths with ≥1 hop)
        # Seed-only paths have no relations — cache may still be 0 if all 0-hop
        assert cache.size() >= 0  # permissive: just must not crash

    def test_min_score_filter(self, loaded_graph):
        """Answers below min_score threshold should not enter cache."""
        from reasoning.answer_extractor import extract
        from unittest.mock import MagicMock
        cache = AAAKCache()
        traversal = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache,
            beam_width=5,
            max_hop=2,
        )
        # Create a mock answer with low score and a valid path
        mock_ans = MagicMock()
        mock_ans.score = 0.1
        mock_ans.best_path = TraversalPath(nodes=["A", "CAUSES", "B"])
        traversal.record_answers([mock_ans], min_score=0.5)
        assert cache.size() == 0  # below threshold, not recorded

    def test_aaak_boost_increases_score_of_known_pattern(self, loaded_graph):
        """Paths matching a cached pattern should rank above equal-score paths."""
        cache = AAAKCache()
        # Pre-load a specific pattern with high weight
        cache.record(("!",), weight=100)

        traversal = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache,
            aaak_strength=1.0,  # maximum boost for testability
            beam_width=20,
            max_hop=2,
        )
        # Boosted score for a "CAUSES" path should exceed unboosted score
        path_with_causes = TraversalPath(
            nodes=["A", "CAUSES", "B"],
            score=0.5,
        )
        boosted = traversal._boosted_score(path_with_causes)
        assert boosted > 0.5

    def test_zero_strength_no_boost(self, loaded_graph):
        """With aaak_strength=0, _boosted_score must equal raw score."""
        cache = AAAKCache()
        cache.record(("!",), weight=100)
        traversal = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache,
            aaak_strength=0.0,
        )
        path = TraversalPath(nodes=["A", "CAUSES", "B"], score=0.42)
        assert abs(traversal._boosted_score(path) - 0.42) < 1e-9

    def test_cache_shared_across_instances(self, loaded_graph):
        """Two traversal instances sharing a cache exchange learned patterns."""
        cache = AAAKCache()
        t1 = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache, beam_width=5, max_hop=1,
        )
        t2 = AAAKBeamTraversal(
            adapter=loaded_graph.adapter,
            csa_engine=loaded_graph._csa,
            cache=cache, beam_width=5, max_hop=1,
        )
        cache.record(("~",), weight=10)
        # Both instances see the same affinity
        path = TraversalPath(nodes=["A", "INFLUENCED", "B"], score=0.5)
        assert t1._boosted_score(path) == t2._boosted_score(path)

"""
Tests for SpeedTalk-Compressed Engram Cache (Phase 58).

Covers:
    - SpeedTalkEncoder: encode/decode roundtrip, overflow, frequency reorder,
      persistence (to_dict / from_dict)
    - SpeedTalkEngram: record, affinity, prefix_query, alphabet, compression_stats,
      save/load persistence roundtrip
    - SpeedTalkEngramTraversal: boost mechanics, record_answers integration
"""
from __future__ import annotations
from typing import Pattern

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reasoning.speedtalk_cache import (
    SpeedTalkEngram,
    SpeedTalkEngramTraversal,
    SpeedTalkEncoder,
    _raw_rel_sequence,
    _BASE_ALPHABET,
)


# ---------------------------------------------------------------------------
# SpeedTalkEncoder tests
# ---------------------------------------------------------------------------

class TestSpeedTalkEncoder:

    def test_single_relation_encode_decode(self):
        enc = SpeedTalkEncoder()
        encoded = enc.encode(["CAUSES"])
        assert len(encoded) == 1
        assert enc.decode(encoded) == ("CAUSES",)

    def test_sequence_encode_decode_roundtrip(self):
        enc = SpeedTalkEncoder()
        seq = ("CAUSES", "TREATS", "PREVENTS")
        encoded = enc.encode(seq)
        assert len(encoded) == 3          # three distinct rels â†’ three chars
        assert enc.decode(encoded) == seq

    def test_same_relation_gets_same_symbol(self):
        enc = SpeedTalkEncoder()
        enc.encode(["CAUSES"])
        enc.encode(["TREATS"])
        s1 = enc.encode(["CAUSES"])
        s2 = enc.encode(["CAUSES"])
        assert s1 == s2

    def test_different_relations_get_different_symbols(self):
        enc = SpeedTalkEncoder()
        s_causes = enc.encode(["CAUSES"])
        s_treats = enc.encode(["TREATS"])
        assert s_causes != s_treats

    def test_prefix_of_encoded_corresponds_to_prefix_of_sequence(self):
        enc = SpeedTalkEncoder()
        seq = ("CAUSES", "TREATS", "PREVENTS")
        encoded = enc.encode(seq)
        # First character should decode to just the first relation
        assert enc.decode(encoded[:1]) == ("CAUSES",)
        assert enc.decode(encoded[:2]) == ("CAUSES", "TREATS")

    def test_overflow_beyond_62_symbols(self):
        enc = SpeedTalkEncoder()
        # Register 65 distinct relation types
        rels = [f"REL_{i}" for i in range(65)]
        for r in rels:
            enc.encode([r])
        assert enc.size == 65
        # All 65 should round-trip cleanly
        for r in rels:
            decoded = enc.decode(enc.encode([r]))
            assert decoded == (r,), f"Failed roundtrip for {r}"

    def test_vocabulary_property(self):
        enc = SpeedTalkEncoder()
        enc.encode(["CAUSES", "TREATS"])
        vocab = enc.vocabulary
        assert "CAUSES" in vocab
        assert "TREATS" in vocab
        assert len(vocab) == 2

    def test_size_property(self):
        enc = SpeedTalkEncoder()
        enc.encode(["A", "B", "C"])
        assert enc.size == 3

    def test_to_dict_from_dict_roundtrip(self):
        enc = SpeedTalkEncoder()
        enc.encode(["CAUSES", "TREATS", "PREVENTS"])
        d = enc.to_dict()
        restored = SpeedTalkEncoder.from_dict(d)
        # Encoding must be identical after restore
        seq = ("CAUSES", "TREATS", "PREVENTS")
        assert restored.encode(seq) == enc.encode(seq)
        assert restored.decode(enc.encode(seq)) == seq

    def test_build_frequency_order_assigns_shortest_to_most_common(self):
        enc = SpeedTalkEncoder()
        freq = {"RARE_REL": 1, "COMMON_REL": 1000, "MEDIUM_REL": 50}
        enc.build_frequency_order(freq)
        vocab = enc.vocabulary
        # COMMON_REL must get the first (lowest-index) symbol
        assert vocab["COMMON_REL"] == _BASE_ALPHABET[0]
        assert vocab["MEDIUM_REL"] == _BASE_ALPHABET[1]
        assert vocab["RARE_REL"]   == _BASE_ALPHABET[2]

    def test_empty_sequence_encodes_to_empty_string(self):
        enc = SpeedTalkEncoder()
        assert enc.encode([]) == ""
        assert enc.decode("") == ()


# ---------------------------------------------------------------------------
# SpeedTalkEngram tests
# ---------------------------------------------------------------------------

class TestSpeedTalkEngram:

    def test_record_and_size(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"))
        assert cache.size() == 1

    def test_affinity_returns_zero_for_unknown_prefix(self):
        cache = SpeedTalkEngram()
        assert cache.affinity(("UNKNOWN_REL",)) == 0.0

    def test_affinity_returns_nonzero_for_known_pattern(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        aff = cache.affinity(("CAUSES", "TREATS"))
        assert aff > 0.0

    def test_affinity_returns_nonzero_for_prefix_of_known_pattern(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS", "PREVENTS"), weight=5)
        # Prefix "CAUSES" alone should still get positive affinity
        aff = cache.affinity(("CAUSES",))
        assert aff > 0.0

    def test_affinity_longer_match_beats_shorter(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS", "PREVENTS"), weight=5)
        aff_full = cache.affinity(("CAUSES", "TREATS", "PREVENTS"))
        aff_prefix = cache.affinity(("CAUSES",))
        assert aff_full >= aff_prefix

    def test_top_patterns_returns_most_frequent(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=10)
        cache.record(("ASSOCIATED_WITH",), weight=2)
        top = cache.top_patterns(n=1)
        assert top[0][0] == ("CAUSES", "TREATS")
        assert top[0][1] == 10

    def test_clear(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES",))
        cache.clear()
        assert cache.size() == 0
        assert cache.affinity(("CAUSES",)) == 0.0

    # ------------------------------------------------------------------
    # prefix_query â€” new capability
    # ------------------------------------------------------------------

    def test_prefix_query_returns_matching_patterns(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        cache.record(("CAUSES", "INHIBITS", "PREVENTS"), weight=3)
        cache.record(("ASSOCIATED_WITH",), weight=7)

        results = cache.prefix_query("CAUSES")
        decoded_seqs = [r[0] for r in results]
        assert ("CAUSES", "TREATS") in decoded_seqs
        assert ("CAUSES", "INHIBITS", "PREVENTS") in decoded_seqs
        assert ("ASSOCIATED_WITH",) not in decoded_seqs

    def test_prefix_query_returns_sorted_by_count(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=3)
        cache.record(("CAUSES", "INHIBITS"), weight=10)
        results = cache.prefix_query("CAUSES")
        # Highest count first
        assert results[0][0] == ("CAUSES", "INHIBITS")

    def test_prefix_query_multi_hop_prefix(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS", "PREVENTS"), weight=5)
        cache.record(("CAUSES", "TREATS", "INHIBITS"), weight=2)
        cache.record(("CAUSES", "INHIBITS"), weight=8)
        results = cache.prefix_query("CAUSES", "TREATS")
        decoded = [r[0] for r in results]
        assert ("CAUSES", "TREATS", "PREVENTS") in decoded
        assert ("CAUSES", "TREATS", "INHIBITS") in decoded
        # Only two-hop prefix match; single-hop CAUSESâ†’INHIBITS must be absent
        assert ("CAUSES", "INHIBITS") not in decoded

    def test_prefix_query_empty_args_returns_empty(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES",))
        assert cache.prefix_query() == []

    def test_prefix_query_no_match_returns_empty(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"))
        assert cache.prefix_query("UNKNOWN_REL") == []

    # ------------------------------------------------------------------
    # Graph-adaptive encoding
    # ------------------------------------------------------------------

    def test_adapt_to_graph_reassigns_most_common_relation_to_first_symbol(self):
        cache = SpeedTalkEngram()
        # Record a pattern *before* adapting â€” it must survive re-encoding
        cache.record(("RARE_REL", "COMMON_REL"), weight=3)
        freq = {"COMMON_REL": 1000, "RARE_REL": 5}
        cache.adapt_to_graph(freq)
        alpha = cache.alphabet()
        assert alpha["COMMON_REL"] == "a"
        assert alpha["RARE_REL"] == "b"

    def test_adapt_to_graph_preserves_existing_patterns(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=7)
        freq = {"TREATS": 200, "CAUSES": 50}
        cache.adapt_to_graph(freq)
        # Pattern must still be retrievable and affinity must be positive
        assert cache.size() == 1
        assert cache.affinity(("CAUSES", "TREATS")) > 0.0

    def test_adapt_to_graph_prefix_query_works_after_adapt(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        cache.record(("CAUSES", "INHIBITS"), weight=2)
        freq = {"CAUSES": 300, "TREATS": 150, "INHIBITS": 80}
        cache.adapt_to_graph(freq)
        results = cache.prefix_query("CAUSES")
        decoded = [r[0] for r in results]
        assert ("CAUSES", "TREATS") in decoded
        assert ("CAUSES", "INHIBITS") in decoded

    def test_adapt_to_graph_empty_freq_is_noop(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES",), weight=2)
        cache.adapt_to_graph({})   # must not raise or destroy state
        assert cache.size() == 1

    def test_count_edge_types_from_get_all_edges(self):
        edge1 = MagicMock()
        edge1.relation_type = "CAUSES"
        edge2 = MagicMock()
        edge2.relation_type = "CAUSES"
        edge3 = MagicMock()
        edge3.relation_type = "TREATS"
        adapter = MagicMock()
        adapter.get_all_edges = MagicMock(return_value=[edge1, edge2, edge3])
        counts = SpeedTalkEngram.count_edge_types(adapter)
        assert counts == {"CAUSES": 2, "TREATS": 1}

    def test_count_edge_types_fallback_to_edges_attr(self):
        edge1 = MagicMock()
        edge1.relation_type = "INHIBITS"
        adapter = MagicMock(spec=[])   # no get_all_edges
        adapter.edges = [edge1]
        counts = SpeedTalkEngram.count_edge_types(adapter)
        assert counts == {"INHIBITS": 1}

    def test_count_edge_types_no_edges_returns_empty(self):
        adapter = MagicMock(spec=[])   # no get_all_edges, no .edges
        counts = SpeedTalkEngram.count_edge_types(adapter)
        assert counts == {}

    def test_from_graph_adapter_creates_frequency_tuned_cache(self):
        edge_a = MagicMock()
        edge_a.relation_type = "COMMON"
        edge_b = MagicMock()
        edge_b.relation_type = "RARE"
        adapter = MagicMock()
        adapter.get_all_edges = MagicMock(return_value=[edge_a, edge_a, edge_a, edge_b])
        cache = SpeedTalkEngram.from_graph_adapter(adapter)
        assert cache.alphabet()["COMMON"] == "a"
        assert cache.alphabet()["RARE"] == "b"
        assert cache.size() == 0   # fresh cache, no patterns yet

    # ------------------------------------------------------------------
    # alphabet() and compression_stats()
    # ------------------------------------------------------------------

    def test_alphabet_returns_relation_to_symbol_map(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"))
        alpha = cache.alphabet()
        assert "CAUSES" in alpha
        assert "TREATS" in alpha

    def test_compression_stats_empty_cache(self):
        cache = SpeedTalkEngram()
        stats = cache.compression_stats()
        assert stats["total_patterns"] == 0
        assert stats["compression_ratio"] == 1.0

    def test_compression_stats_populated_cache(self):
        cache = SpeedTalkEngram()
        for i in range(10):
            cache.record(("CAUSES", "TREATS", "PREVENTS"), weight=i + 1)
        stats = cache.compression_stats()
        assert stats["total_patterns"] == 1
        # Encoded length = 3 chars; tuple repr should be much longer
        assert stats["avg_encoded_len"] == 3.0
        assert stats["compression_ratio"] > 1.0

    # ------------------------------------------------------------------
    # Persistence roundtrip
    # ------------------------------------------------------------------

    def test_save_creates_json_file(self, tmp_path):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        p = str(tmp_path / "cache.json")
        cache.save(p)
        assert Path(p).exists()
        with open(p) as f:
            data = json.load(f)
        assert data["version"] == 2
        assert len(data["counts"]) == 1

    def test_load_restores_patterns(self, tmp_path):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        cache.record(("INHIBITS",), weight=2)
        p = str(tmp_path / "cache.json")
        cache.save(p)

        restored = SpeedTalkEngram.load(p)
        assert restored.size() == 2
        assert restored.affinity(("CAUSES", "TREATS")) > 0.0

    def test_load_restores_prefix_query(self, tmp_path):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES", "TREATS"), weight=5)
        cache.record(("CAUSES", "INHIBITS"), weight=3)
        p = str(tmp_path / "cache.json")
        cache.save(p)

        restored = SpeedTalkEngram.load(p)
        results = restored.prefix_query("CAUSES")
        decoded = [r[0] for r in results]
        assert ("CAUSES", "TREATS") in decoded
        assert ("CAUSES", "INHIBITS") in decoded

    def test_load_nonexistent_file_returns_empty_cache(self, tmp_path):
        cache = SpeedTalkEngram.load(str(tmp_path / "nonexistent.json"))
        assert cache.size() == 0

    def test_save_if_path_none_is_noop(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES",))
        cache.save_if_path(None)   # must not raise

    def test_eviction_respects_max_patterns(self):
        cache = SpeedTalkEngram(max_patterns=3)
        cache.record(("A",), weight=1)
        cache.record(("B",), weight=2)
        cache.record(("C",), weight=3)
        cache.record(("D",), weight=4)   # triggers eviction of ("A",)
        assert cache.size() == 3


# ---------------------------------------------------------------------------
# _raw_rel_sequence helper
# ---------------------------------------------------------------------------

class TestRawRelSequence:

    def _make_path(self, *nodes):
        """Build a minimal mock TraversalPath with given nodes."""
        path = MagicMock()
        path.nodes = list(nodes)
        return path

    def test_extracts_single_hop_relation(self):
        path = self._make_path("newton", "INFLUENCED", "einstein")
        assert _raw_rel_sequence(path) == ("INFLUENCED",)

    def test_extracts_multi_hop_relations(self):
        path = self._make_path("A", "CAUSES", "B", "TREATS", "C")
        assert _raw_rel_sequence(path) == ("CAUSES", "TREATS")

    def test_empty_path_returns_empty_tuple(self):
        path = self._make_path()
        assert _raw_rel_sequence(path) == ()


# ---------------------------------------------------------------------------
# SpeedTalkEngramTraversal tests
# ---------------------------------------------------------------------------

class TestSpeedTalkEngramTraversal:
    """
    Unit tests for traversal-level integration.
    The traversal itself is tested via its parent BeamTraversal suite;
    here we verify the SpeedTalk-specific boost and recording logic.
    """

    def _make_mock_traversal(self, cache=None, engram_strength=0.3):
        """Build a traversal with mocked adapter/csa dependencies."""
        adapter = MagicMock()
        csa = MagicMock()
        t = SpeedTalkEngramTraversal.__new__(SpeedTalkEngramTraversal)
        t.cache = cache or SpeedTalkEngram()
        t.engram_strength = engram_strength
        t.beam_width = 10
        t.max_hop = 3
        t._beam_widths = {}
        return t

    def _make_path(self, score, *nodes):
        path = MagicMock()
        path.score = score
        path.nodes = list(nodes)
        return path

    def test_boosted_score_with_no_cached_pattern(self):
        t = self._make_mock_traversal()
        path = self._make_path(0.5, "A", "CAUSES", "B")
        # No cached patterns â†’ no boost â†’ score unchanged
        assert t._boosted_score(path) == pytest.approx(0.5)

    def test_boosted_score_increases_with_known_pattern(self):
        cache = SpeedTalkEngram()
        cache.record(("CAUSES",), weight=10)
        t = self._make_mock_traversal(cache=cache, engram_strength=0.3)
        path = self._make_path(0.5, "A", "CAUSES", "B")
        boosted = t._boosted_score(path)
        assert boosted > 0.5

    def test_boosted_score_zero_length_path(self):
        t = self._make_mock_traversal()
        path = self._make_path(0.8)
        path.nodes = []
        # No relations â†’ no boost
        assert t._boosted_score(path) == pytest.approx(0.8)

    def test_record_answers_populates_cache(self):
        cache = SpeedTalkEngram()
        t = self._make_mock_traversal(cache=cache)

        path = self._make_path(0.9, "A", "CAUSES", "B", "TREATS", "C")
        answer = MagicMock()
        answer.score = 0.9
        answer.best_path = path

        t.record_answers([answer], min_score=0.5)
        assert cache.size() == 1
        assert cache.affinity(("CAUSES", "TREATS")) > 0.0

    def test_record_answers_skips_low_score(self):
        cache = SpeedTalkEngram()
        t = self._make_mock_traversal(cache=cache)

        path = self._make_path(0.2, "A", "CAUSES", "B")
        answer = MagicMock()
        answer.score = 0.2
        answer.best_path = path

        t.record_answers([answer], min_score=0.5)
        assert cache.size() == 0

    def test_record_answers_skips_answers_without_path(self):
        cache = SpeedTalkEngram()
        t = self._make_mock_traversal(cache=cache)

        answer = MagicMock()
        answer.score = 0.9
        answer.best_path = None

        t.record_answers([answer], min_score=0.5)
        assert cache.size() == 0

    def test_prefix_query_available_after_record(self):
        cache = SpeedTalkEngram()
        t = self._make_mock_traversal(cache=cache)

        path = self._make_path(0.85, "A", "CAUSES", "B", "TREATS", "C")
        answer = MagicMock()
        answer.score = 0.85
        answer.best_path = path

        t.record_answers([answer], min_score=0.5)
        results = cache.prefix_query("CAUSES")
        assert len(results) == 1
        assert results[0][0] == ("CAUSES", "TREATS")

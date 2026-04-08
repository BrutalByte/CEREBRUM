"""
Tests for QueryLog (core/persistence.py) and Engram persistence.

QueryLog is the durable, append-only query history (core/persistence.py).
Its sole job is to record completed reasoning
queries as NDJSON and warm up Engram on restart.
"""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.persistence import QueryLog
from reasoning.engram_traversal import Engram
from reasoning.traversal import TraversalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_answer(entity_id: str, score: float, path_nodes=None):
    ans = MagicMock()
    ans.entity_id = entity_id
    ans.score = score
    if path_nodes is not None:
        ans.best_path = TraversalPath(nodes=path_nodes)
    else:
        ans.best_path = None
    return ans


# ---------------------------------------------------------------------------
# QueryLog — basic I/O
# ---------------------------------------------------------------------------

class TestQueryLogBasic:
    def test_creates_file_on_first_write(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        ans = _make_answer("newton", 0.8, ["seed", "CAUSES", "newton"])
        log.record(["seed"], [ans])
        assert (tmp_path / "q.ndjson").exists()

    def test_count_zero_before_any_write(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        assert log.count() == 0

    def test_count_increases_per_record(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["seed"], [_make_answer("a", 0.5, ["seed", "REL", "a"])])
        log.record(["seed"], [_make_answer("b", 0.6, ["seed", "REL", "b"])])
        assert log.count() == 2

    def test_no_entry_for_empty_answers(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["seed"], [])
        assert log.count() == 0

    def test_min_score_filters_low_answers(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        low = _make_answer("low", 0.1, ["s", "R", "low"])
        log.record(["s"], [low], min_score=0.5)
        assert log.count() == 0

    def test_answers_above_min_score_recorded(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        high = _make_answer("high", 0.9, ["s", "R", "high"])
        log.record(["s"], [high], min_score=0.5)
        assert log.count() == 1

    def test_record_contains_timestamp(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        t0 = time.time()
        log.record(["s"], [_make_answer("e", 0.8, ["s", "R", "e"])])
        rec = log.read_recent(1)[0]
        assert rec["ts"] >= t0

    def test_record_contains_seeds(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["newton", "faraday"], [_make_answer("e", 0.8, ["newton", "R", "e"])])
        rec = log.read_recent(1)[0]
        assert "newton" in rec["seeds"]

    def test_record_contains_rel_sequence(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["s"], [_make_answer("e", 0.8, ["s", "CAUSES", "e"])])
        rec = log.read_recent(1)[0]
        assert rec["answers"][0]["rels"] == ["CAUSES"]

    def test_no_path_answer_stores_empty_rels(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        ans = _make_answer("e", 0.8, None)   # no best_path
        log.record(["s"], [ans])
        rec = log.read_recent(1)[0]
        assert rec["answers"][0]["rels"] == []

    def test_read_recent_limit(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        for i in range(10):
            log.record([f"s{i}"], [_make_answer(f"e{i}", 0.5, [f"s{i}", "R", f"e{i}"])])
        recs = log.read_recent(n=3)
        assert len(recs) == 3

    def test_clear_resets_count(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["s"], [_make_answer("e", 0.8, ["s", "R", "e"])])
        log.clear()
        assert log.count() == 0

    def test_nonexistent_read_returns_empty(self, tmp_path):
        log = QueryLog(str(tmp_path / "no_file.ndjson"))
        assert log.read_recent() == []

    def test_multi_answer_per_record(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        answers = [
            _make_answer("a", 0.9, ["s", "R1", "a"]),
            _make_answer("b", 0.7, ["s", "R2", "b"]),
        ]
        log.record(["s"], answers)
        rec = log.read_recent(1)[0]
        assert len(rec["answers"]) == 2

    def test_file_is_valid_ndjson(self, tmp_path):
        """Each line of the log must be parseable as JSON."""
        log = QueryLog(str(tmp_path / "q.ndjson"))
        for i in range(3):
            log.record([f"s{i}"], [_make_answer(f"e{i}", 0.5, [f"s{i}", "R", f"e{i}"])])
        lines = (tmp_path / "q.ndjson").read_text().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "ts" in obj and "seeds" in obj and "answers" in obj


# ---------------------------------------------------------------------------
# QueryLog → Engram replay
# ---------------------------------------------------------------------------

class TestQueryLogReplay:
    def test_replay_populates_cache(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["s"], [_make_answer("e", 0.9, ["s", "CAUSES", "e"])])
        cache = Engram()
        replayed = log.replay_into_cache(cache)
        assert replayed == 1
        assert cache.size() == 1

    def test_replay_min_score_filter(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        low  = _make_answer("low", 0.1, ["s", "R1", "low"])
        high = _make_answer("hi",  0.9, ["s", "R2", "hi"])
        log.record(["s"], [low, high])
        cache = Engram()
        replayed = log.replay_into_cache(cache, min_score=0.5)
        # Only "hi" passes the filter
        assert replayed == 1

    def test_replay_empty_log(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        cache = Engram()
        assert log.replay_into_cache(cache) == 0
        assert cache.size() == 0

    def test_replay_no_path_skipped(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["s"], [_make_answer("e", 0.8, None)])
        cache = Engram()
        replayed = log.replay_into_cache(cache)
        assert replayed == 0

    def test_replay_cache_affinity_set(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        log.record(["s"], [_make_answer("e", 0.9, ["s", "CAUSES", "e"])])
        cache = Engram()
        log.replay_into_cache(cache)
        assert cache.affinity(("CAUSES",)) > 0.0

    def test_replay_multiple_records(self, tmp_path):
        log = QueryLog(str(tmp_path / "q.ndjson"))
        for i in range(5):
            log.record([f"s{i}"], [_make_answer(f"e{i}", 0.8, [f"s{i}", "CAUSES", f"e{i}"])])
        cache = Engram()
        replayed = log.replay_into_cache(cache)
        assert replayed == 5
        # All used "CAUSES" → prefix count should be high
        assert cache.affinity(("CAUSES",)) > 0.5


# ---------------------------------------------------------------------------
# Engram persistence (save / load)
# ---------------------------------------------------------------------------

class TestEngramPersistence:
    def test_save_creates_file(self, tmp_path):
        cache = Engram()
        cache.record(("!",))
        cache.save(str(tmp_path / "engram.json"))
        assert (tmp_path / "engram.json").exists()

    def test_save_is_valid_json(self, tmp_path):
        cache = Engram()
        cache.record(("!",), weight=3)
        p = str(tmp_path / "engram.json")
        cache.save(p)
        with open(p) as f:
            data = json.load(f)
        assert "counts" in data
        assert data["version"] == 1

    def test_load_empty_file_returns_empty_cache(self, tmp_path):
        cache = Engram.load(str(tmp_path / "nonexistent.json"))
        assert cache.size() == 0

    def test_roundtrip_preserves_counts(self, tmp_path):
        cache = Engram()
        cache.record(("!",), weight=5)
        cache.record(("+",), weight=3)
        p = str(tmp_path / "engram.json")
        cache.save(p)

        loaded = Engram.load(p)
        assert loaded.size() == 2
        assert loaded.affinity(("!",)) > 0.0
        assert loaded.affinity(("+",)) > 0.0

    def test_roundtrip_preserves_affinity_ordering(self, tmp_path):
        cache = Engram()
        cache.record(("!",), weight=10)  # higher
        cache.record(("+",), weight=1)   # lower
        p = str(tmp_path / "engram.json")
        cache.save(p)

        loaded = Engram.load(p)
        assert loaded.affinity(("!",)) > loaded.affinity(("+",))

    def test_roundtrip_max_count_restored(self, tmp_path):
        cache = Engram()
        cache.record(("!",), weight=7)
        p = str(tmp_path / "engram.json")
        cache.save(p)
        loaded = Engram.load(p)
        assert loaded._max_count == 7

    def test_save_if_path_none_no_error(self):
        cache = Engram()
        cache.record(("!",))
        cache.save_if_path(None)   # must not raise

    def test_save_if_path_writes(self, tmp_path):
        cache = Engram()
        cache.record(("!",))
        p = str(tmp_path / "sub" / "engram.json")
        cache.save_if_path(p)
        assert Path(p).exists()

    def test_prefix_index_rebuilt_after_load(self, tmp_path):
        cache = Engram()
        cache.record(("!", "+", "~"), weight=4)
        p = str(tmp_path / "engram.json")
        cache.save(p)
        loaded = Engram.load(p)
        # Prefix ("!",) should have affinity derived from full sequence
        assert loaded.affinity(("!",)) > 0.0
        assert loaded.affinity(("!", "+")) > 0.0

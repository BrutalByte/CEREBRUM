"""Tests for core/working_memory.py (Phase 95)."""
import threading
import time

import pytest

from core.working_memory import MemoryEntry, WorkingMemoryBuffer


def _entry(seeds=None, answers=None, top_score=0.5, soliton=None, pe=None, source="query", ts=None):
    return MemoryEntry(
        timestamp=ts if ts is not None else time.time(),
        seeds=seeds or [],
        answers=answers or [],
        top_score=top_score,
        soliton_index=soliton,
        prediction_error=pe,
        source=source,
    )


def test_record_and_retrieve():
    wm = WorkingMemoryBuffer(maxlen=10)
    e1 = _entry(seeds=["a"])
    e2 = _entry(seeds=["b"])
    wm.record(e1)
    wm.record(e2)
    recent = wm.recent(10)
    assert len(recent) == 2
    # newest first
    assert recent[0].seeds == ["b"]
    assert recent[1].seeds == ["a"]


def test_most_relevant_jaccard():
    wm = WorkingMemoryBuffer(maxlen=10)
    now = time.time()
    e_match = _entry(seeds=["newton", "gravity"], answers=["apple"], ts=now - 1)
    e_unrelated = _entry(seeds=["beethoven", "symphony"], answers=["music"], ts=now - 2)
    wm.record(e_unrelated)
    wm.record(e_match)

    results = wm.most_relevant(["newton", "apple"], k=2)
    assert len(results) >= 1
    assert results[0].seeds == ["newton", "gravity"]


def test_recency_decay():
    wm = WorkingMemoryBuffer(maxlen=10)
    now = time.time()
    old = _entry(seeds=["x"], answers=["y"], ts=now - 10000)  # very old
    fresh = _entry(seeds=["x"], answers=["y"], ts=now - 1)    # fresh, same seeds
    wm.record(old)
    wm.record(fresh)

    results = wm.most_relevant(["x"], k=2)
    # fresh should rank higher
    assert results[0].timestamp > results[1].timestamp


def test_maxlen_eviction():
    wm = WorkingMemoryBuffer(maxlen=3)
    for i in range(5):
        wm.record(_entry(seeds=[str(i)]))
    assert wm.stats()["count"] == 3
    recent = wm.recent(10)
    # oldest two evicted; remaining are entries 2, 3, 4
    seed_vals = [e.seeds[0] for e in recent]
    assert "0" not in seed_vals
    assert "1" not in seed_vals
    assert "4" in seed_vals


def test_thread_safety():
    wm = WorkingMemoryBuffer(maxlen=200)
    errors = []

    def writer():
        try:
            for _ in range(50):
                wm.record(_entry(seeds=["a"]))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert wm.stats()["count"] == 200  # maxlen cap


def test_clear():
    wm = WorkingMemoryBuffer(maxlen=10)
    wm.record(_entry())
    wm.record(_entry())
    wm.clear()
    assert wm.stats()["count"] == 0
    assert wm.recent(5) == []


def test_stats_empty():
    wm = WorkingMemoryBuffer(maxlen=10)
    s = wm.stats()
    assert s["count"] == 0
    assert s["oldest_ts"] is None
    assert s["newest_ts"] is None


def test_most_relevant_empty():
    wm = WorkingMemoryBuffer(maxlen=10)
    assert wm.most_relevant(["a", "b"]) == []


def test_most_relevant_no_query_seeds():
    wm = WorkingMemoryBuffer(maxlen=10)
    now = time.time()
    wm.record(_entry(seeds=["a"], ts=now - 10))
    wm.record(_entry(seeds=["b"], ts=now - 1))
    # with no query seeds, jaccard=0 for all; recency breaks ties
    results = wm.most_relevant([], k=2)
    assert len(results) == 2
    # Both have score 0 (jaccard=0), so order is implementation-defined but no crash

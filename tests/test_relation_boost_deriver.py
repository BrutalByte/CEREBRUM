"""Unit tests for RelationBoostDeriver (Phase 202)."""
from __future__ import annotations

import tempfile
import os
import pytest

from core.relation_boost_deriver import RelationBoostDeriver


_TRIPLES = [
    ("m1", "directed_by", "d1"),
    ("m1", "directed_by", "d2"),
    ("m2", "directed_by", "d1"),
    ("m1", "starred_actors", "a1"),
    ("m1", "starred_actors", "a2"),
    ("m1", "starred_actors", "a3"),
    ("m2", "starred_actors", "a1"),
    ("m2", "starred_actors", "a4"),
    ("m1", "release_year", "1999"),
    ("m2", "release_year", "2001"),
]

# directed_by:    3 triples, 2 unique heads → fan_out = 1.5
# starred_actors: 5 triples, 2 unique heads → fan_out = 2.5
# release_year:   2 triples, 2 unique heads → fan_out = 1.0


@pytest.fixture(scope="module")
def deriver():
    d = RelationBoostDeriver()
    d.build_from_triples(_TRIPLES)
    return d


def test_not_built_before_build():
    assert not RelationBoostDeriver().is_built


def test_built_after_triples(deriver):
    assert deriver.is_built


def test_fan_out_values(deriver):
    assert abs(deriver.fan_out("directed_by")    - 1.5) < 1e-9
    assert abs(deriver.fan_out("starred_actors") - 2.5) < 1e-9
    assert abs(deriver.fan_out("release_year")   - 1.0) < 1e-9


def test_fan_out_unknown_returns_one(deriver):
    assert deriver.fan_out("nonexistent_relation") == 1.0


def test_boost_map_scale(deriver):
    bmap = deriver.boost_map(gamma=2.0)
    assert bmap is not None
    assert abs(bmap["directed_by"]    - 3.0) < 1e-9
    assert abs(bmap["starred_actors"] - 5.0) < 1e-9
    assert abs(bmap["release_year"]   - 2.0) < 1e-9


def test_boost_map_ordering(deriver):
    bmap = deriver.boost_map(gamma=1.0)
    assert bmap["starred_actors"] > bmap["directed_by"] > bmap["release_year"]


def test_boost_map_zero_gamma(deriver):
    assert deriver.boost_map(gamma=0.0) is None


def test_boost_map_negative_gamma(deriver):
    assert deriver.boost_map(gamma=-1.0) is None


def test_boost_map_contains_all_relations(deriver):
    bmap = deriver.boost_map(gamma=1.0)
    assert set(bmap.keys()) == {"directed_by", "starred_actors", "release_year"}


def test_boost_map_beta_amplifies_high_fanout(deriver):
    """beta > 1 should widen the gap between high and low fan_out relations."""
    bmap1 = deriver.boost_map(gamma=1.0, beta=1.0)
    bmap2 = deriver.boost_map(gamma=1.0, beta=2.0)
    ratio1 = bmap1["starred_actors"] / bmap1["directed_by"]
    ratio2 = bmap2["starred_actors"] / bmap2["directed_by"]
    assert ratio2 > ratio1


def test_boost_map_beta_one_matches_linear(deriver):
    """beta=1.0 must produce identical results to the default."""
    bmap_default = deriver.boost_map(gamma=3.0)
    bmap_beta1   = deriver.boost_map(gamma=3.0, beta=1.0)
    assert bmap_default == bmap_beta1


def test_boost_map_beta_power_law(deriver):
    """boost(r) = gamma * fan_out(r)^beta — verify exact values."""
    bmap = deriver.boost_map(gamma=2.0, beta=2.0)
    # directed_by: fan_out=1.5, beta=2 → 2.0 * 1.5^2 = 4.5
    assert abs(bmap["directed_by"] - 4.5) < 1e-9
    # starred_actors: fan_out=2.5, beta=2 → 2.0 * 2.5^2 = 12.5
    assert abs(bmap["starred_actors"] - 12.5) < 1e-9


def test_build_from_file():
    lines = "\n".join(f"{h}|{r}|{t}" for h, r, t in _TRIPLES)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(lines)
        path = f.name
    try:
        d = RelationBoostDeriver()
        d.build_from_file(path)
        assert d.is_built
        assert abs(d.fan_out("directed_by") - 1.5) < 1e-9
    finally:
        os.unlink(path)


def test_build_from_file_skips_malformed():
    lines = "m1|directed_by|d1\nbadline\nm2|directed_by|d1\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write(lines)
        path = f.name
    try:
        d = RelationBoostDeriver()
        d.build_from_file(path)
        assert d.is_built
        assert abs(d.fan_out("directed_by") - 1.0) < 1e-9  # 2 triples, 2 unique heads
    finally:
        os.unlink(path)


def test_relation_stats(deriver):
    stats = deriver.relation_stats()
    assert "directed_by" in stats
    s = stats["directed_by"]
    assert s["triple_count"] == 3
    assert s["unique_heads"] == 2
    assert s["unique_tails"] == 2
    assert abs(s["fan_out"] - 1.5) < 1e-9


def test_summary_string(deriver):
    s = deriver.summary()
    assert "starred_actors" in s
    assert "RelationBoostDeriver" in s


def test_rebuild_overwrites(deriver):
    d = RelationBoostDeriver()
    d.build_from_triples(_TRIPLES)
    original = d.fan_out("directed_by")
    # Rebuild with different data
    d.build_from_triples([("m1", "directed_by", "d1")])
    assert abs(d.fan_out("directed_by") - 1.0) < 1e-9
    assert d.fan_out("starred_actors") == 1.0  # unknown after rebuild

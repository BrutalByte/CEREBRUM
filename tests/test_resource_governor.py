"""
Tests for ResourceGovernor — memory-aware expansion budget management.

All tests use real psutil calls (no mocking) because the governor's whole
purpose is to read real system state. Tests are written to be robust on any
machine regardless of current memory pressure.
"""

from core.resource_governor import ResourceGovernor


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_default_construction():
    gov = ResourceGovernor()
    assert gov.threshold == 85.0
    assert gov.buffer_bytes == 500 * 1024 * 1024


def test_custom_threshold():
    gov = ResourceGovernor(memory_threshold_pct=90.0, safety_buffer_mb=256)
    assert gov.threshold == 90.0
    assert gov.buffer_bytes == 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# get_current_stats
# ---------------------------------------------------------------------------

def test_stats_keys_present():
    gov = ResourceGovernor()
    stats = gov.get_current_stats()
    assert "system_ram_pct" in stats
    assert "system_ram_free_mb" in stats
    assert "process_rss_mb" in stats


def test_stats_system_ram_pct_range():
    gov = ResourceGovernor()
    stats = gov.get_current_stats()
    assert 0.0 <= stats["system_ram_pct"] <= 100.0


def test_stats_free_mb_non_negative():
    gov = ResourceGovernor()
    stats = gov.get_current_stats()
    assert stats["system_ram_free_mb"] >= 0


def test_stats_process_rss_positive():
    gov = ResourceGovernor()
    stats = gov.get_current_stats()
    # The current Python process must be using at least 1 MB
    assert stats["process_rss_mb"] >= 1


# ---------------------------------------------------------------------------
# can_expand — budget cap
# ---------------------------------------------------------------------------

def test_can_expand_within_budget():
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1)
    # With a near-100% threshold and 1MB buffer, budget cap is the only limiter
    assert gov.can_expand(0, 100) is True


def test_can_expand_at_budget_limit():
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1)
    assert gov.can_expand(100, 100) is False


def test_can_expand_exceeds_budget():
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1)
    assert gov.can_expand(150, 100) is False


def test_can_expand_zero_budget():
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1)
    assert gov.can_expand(0, 0) is False


def test_can_expand_memory_pressure_cap():
    # Set threshold to 0% — any memory usage exceeds it — should always return False
    gov = ResourceGovernor(memory_threshold_pct=0.0, safety_buffer_mb=1)
    result = gov.can_expand(0, 10000)
    assert result is False


def test_can_expand_extreme_buffer():
    # Set safety buffer to 1TB — no machine has this free — should always return False
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1_000_000)
    result = gov.can_expand(0, 10000)
    assert result is False


# ---------------------------------------------------------------------------
# estimate_path_capacity
# ---------------------------------------------------------------------------

def test_estimate_path_capacity_returns_int():
    gov = ResourceGovernor()
    cap = gov.estimate_path_capacity()
    assert isinstance(cap, int)


def test_estimate_path_capacity_non_negative():
    gov = ResourceGovernor()
    cap = gov.estimate_path_capacity()
    assert cap >= 0


def test_estimate_path_capacity_zero_when_no_memory():
    gov = ResourceGovernor(safety_buffer_mb=1_000_000)
    cap = gov.estimate_path_capacity()
    assert cap == 0


def test_estimate_path_capacity_larger_bytes_fewer_paths():
    gov = ResourceGovernor(memory_threshold_pct=99.0, safety_buffer_mb=1)
    small = gov.estimate_path_capacity(avg_path_bytes=1)
    large = gov.estimate_path_capacity(avg_path_bytes=1_000_000)
    assert small >= large

"""
pytest configuration for the CEREBRUM test suite.

Patches ResourceGovernor to use a 99% memory threshold so tests pass
regardless of host system memory pressure (the governor is a production
safety feature, not a unit-test concern).

Tests in test_resource_governor.py are exempted so they can verify the
governor's own logic with real thresholds.
"""
import pytest
from core.resource_governor import ResourceGovernor

_original_rg_init = ResourceGovernor.__init__


@pytest.fixture(autouse=True)
def _relax_resource_governor(request, monkeypatch):
    """Raise the governor's memory threshold to 99% for all tests except
    those that explicitly test the ResourceGovernor itself."""
    if "test_resource_governor" in request.fspath.basename:
        return  # let governor tests run with real defaults

    def _patched_init(self, *args, **kwargs):
        # Always inject the relaxed test defaults
        kwargs.setdefault("memory_threshold_pct", 99.0)
        kwargs.setdefault("safety_buffer_mb", 50)
        _original_rg_init(self, *args, **kwargs)

    monkeypatch.setattr(ResourceGovernor, "__init__", _patched_init)

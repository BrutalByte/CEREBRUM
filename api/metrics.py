"""
CEREBRUM Prometheus metrics registry.

Uses prometheus_client when available (pip install prometheus-client).
Falls back to no-op stubs if not installed, so the rest of the server
never needs to branch on availability.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

logger = logging.getLogger("cerebrum.metrics")

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, CollectorRegistry,
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.debug("prometheus_client not installed — metrics disabled")


# ---------------------------------------------------------------------------
# No-op stubs (used when prometheus_client not installed)
# ---------------------------------------------------------------------------

class _NoOpCounter:
    def inc(self, amount=1): pass
    def labels(self, **kw): return self

class _NoOpHistogram:
    def observe(self, v): pass
    def labels(self, **kw): return self
    def time(self): return _NoOpCtx()

class _NoOpGauge:
    def set(self, v): pass
    def inc(self, amount=1): pass
    def dec(self, amount=1): pass
    def labels(self, **kw): return self

class _NoOpCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if _AVAILABLE:
    QUERIES_TOTAL = Counter(
        "cerebrum_queries_total",
        "Total query requests",
        ["status"],           # "ok" | "error" | "partial"
    )
    QUERY_LATENCY = Histogram(
        "cerebrum_query_latency_seconds",
        "End-to-end query latency",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    PATHS_EXPLORED = Counter(
        "cerebrum_paths_explored_total",
        "Beam paths explored across all queries",
    )
    HOP_DEPTH = Histogram(
        "cerebrum_hop_depth",
        "Hop depth of the top-1 answer path",
        buckets=[1, 2, 3, 4, 5, 6],
    )
    ANSWER_CONFIDENCE = Histogram(
        "cerebrum_answer_confidence",
        "Confidence score of the top-1 answer",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    ACTIVE_TENANTS = Gauge(
        "cerebrum_active_tenants",
        "Number of registered tenant knowledge graphs",
    )
    GRAPH_NODES = Gauge(
        "cerebrum_graph_nodes",
        "Entity count of the loaded graph",
    )
    GRAPH_EDGES = Gauge(
        "cerebrum_graph_edges",
        "Edge count of the loaded graph",
    )
    AUDIT_RECORDS = Gauge(
        "cerebrum_audit_records_total",
        "Total compliance audit records in the in-memory buffer",
    )
    API_KEYS_ACTIVE = Gauge(
        "cerebrum_api_keys_active",
        "Number of active dynamic API keys",
    )
else:
    QUERIES_TOTAL     = _NoOpCounter()
    QUERY_LATENCY     = _NoOpHistogram()
    PATHS_EXPLORED    = _NoOpCounter()
    HOP_DEPTH         = _NoOpHistogram()
    ANSWER_CONFIDENCE = _NoOpHistogram()
    ACTIVE_TENANTS    = _NoOpGauge()
    GRAPH_NODES       = _NoOpGauge()
    GRAPH_EDGES       = _NoOpGauge()
    AUDIT_RECORDS     = _NoOpGauge()
    API_KEYS_ACTIVE   = _NoOpGauge()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_available() -> bool:
    return _AVAILABLE


def prometheus_response() -> tuple[bytes, str]:
    """Return (body_bytes, content_type) for the /metrics endpoint."""
    if not _AVAILABLE:
        raise RuntimeError("prometheus_client not installed")
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


class QueryTimer:
    """Context manager that records query latency and increments counters."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.monotonic() - self._start) * 1000
        QUERY_LATENCY.observe(self.elapsed_ms / 1000.0)
        return False  # never suppress exceptions

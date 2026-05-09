# [Buchorn, 2026]: The Streaming Engine
## Real-Time Continuous Ingest, Incremental DSCF, and SSE Push

**Status**: v2.51.0 (Phase 167 (Sleep-Phase Consolidation) COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Field**: Streaming Systems / Real-Time Reasoning / Event Processing
**Modules**: `adapters/stream_adapter.py`, `core/discretizer.py`, `core/rebalancer.py`, `api/server.py`

---

### 1. Introduction
Static batch ingest is insufficient for real-world deployments where edges arrive continuously from sensors, event logs, financial feeds, or clinical monitors. The **Streaming Engine** provides continuous ingest, incremental community re-optimization, and real-time push delivery of insights to connected clients — all without interrupting active query processing.

### 2. StreamAdapter

The `StreamAdapter` wraps any `GraphAdapter` with a thread-safe event queue and sliding-window buffer.

**Unlocked Preprocessing (Hole 10 Fix)**:
To ensure high-velocity ingestion does not block reasoning queries, v2.51.0 introduces a two-stage ingest pipeline:
1.  **Stage 1 (Concurrent)**: Raw events are processed by the `IngestionPipeline` (Thalamus) **outside the graph lock**. This includes CPU-bound string normalization, deduplication, and metadata enrichment.
2.  **Stage 2 (Atomic)**: Fully prepared triples are committed to the adjacency list and sliding-window buffer under a single lock acquisition.

This refactoring enables ingestion throughput to scale linearly with available CPU cores while maintaining zero-block reasoning for active beams.

**Constructor parameters:**
```python
StreamAdapter(
    time_window_seconds: float = 60.0,
    max_edges: int = 10_000,
    neighborhood_radius: int = 2,
    min_events_before_update: int = 10,
    directed: bool = True,
    pipeline: Optional[IngestionPipeline] = None,
)
```

**Thread safety model:**
- Producers call `adapter.ingest(event)`
- Preprocessing (Stage 1) is thread-safe and non-blocking
- Graph mutation (Stage 2) is protected by `self._lock` (RLock)
- Reads never block: `BeamTraversal` snapshots the community map at query start [Buchorn, 2026]

**Sliding-window buffer:**
- Maintains the most recent `window_size` events in a `deque`
- Used by temporal discretizers to compute inter-event intervals
- Not used by the query path — purely for discretizer context

### 3. The Five Discretizers

All discretizers implement the `BaseDiscretizer` interface:

```python
class BaseDiscretizer:
    def process(self, event: StreamEvent) -> Optional[EdgeCandidate]:
        ...
```

#### 3.1 ThresholdDiscretizer
Emits an edge when a continuous float stream crosses a configured threshold.

```python
ThresholdDiscretizer(
    entity_a: str,
    entity_b: str,
    threshold: float,
    direction: str = "rising",  # "rising" | "falling" | "both"
    relation: str = "EXCEEDS",
)
```

#### 3.2 STDPDiscretizer
Materializes directional `CAUSES` edges from spike timing using Hebbian-inspired weight accumulation. See [Buchorn, 2026] for STDP algorithm details.

Additional streaming parameters:
```python
STDPDiscretizer(
    w_threshold: float = 0.5,
    n_min: int = 5,
    min_causal_span: float = 0.0,   # Phase 19: minimum wall-clock span (seconds)
    use_chi_squared: bool = False,   # Phase 19: uniformity guard
)
```

#### 3.3 DeltaDiscretizer
Emits an edge when the rate-of-change of a signal exceeds a threshold.

```python
DeltaDiscretizer(
    entity_a: str,
    entity_b: str,
    delta_threshold: float,          # |Δx/Δt| threshold
    window_s: float = 1.0,           # Measurement window in seconds
    relation: str = "CHANGES_WITH",
)
```

#### 3.4 WindowedFrequencyDiscretizer
Emits an edge when two entities co-occur more than `min_freq` times within a sliding time window.

```python
WindowedFrequencyDiscretizer(
    window_s: float = 60.0,
    min_freq: int = 10,
    relation: str = "CO_OCCURS",
)
```

#### 3.5 PatternDiscretizer
Emits an edge when a symbolic event sequence matches a configurable pattern.

```python
PatternDiscretizer(
    pattern: List[str],              # Ordered event-type sequence
    entity_a: str,                   # Source entity for emitted edge
    entity_b: str,                   # Target entity for emitted edge
    tolerance: float = 0.8,          # Pattern match probability threshold
    relation: str = "PATTERN_MATCH",
)
```

### 4. GlobalRebalancer

The `GlobalRebalancer` monitors community structure stability and triggers background DSCF re-runs when modularity drift exceeds a threshold.

**Constructor:**
```python
GlobalRebalancer(
    adapter: GraphAdapter,
    q_drift_threshold: float = 0.05,   # Trigger re-run when ΔQ exceeds this
    min_interval_s: float = 30.0,      # Minimum seconds between re-runs
    bridge_engine: Optional[BridgeTwinEngine] = None,  # Phase 19 hook
)
```

**Drift detection:**
After each batch commit, the rebalancer samples local modularity contributions for modified nodes. When cumulative drift $\Delta Q_{cum} \geq \theta_Q$, a background DSCF task is queued.

**Atomic community-map swap:**
```python
# 1. Compute fresh partition on a snapshot graph (no lock held)
new_map = _run_dscf(graph_snapshot)
# 2. Atomic swap under adapter lock
with adapter._lock:
    adapter.community_map = new_map
# 3. Fire post-rebalance hooks
if bridge_engine:
    bridge_engine.on_rebalance(new_map)
```

### 5. SSE Push Endpoints

The FastAPI server exposes two Server-Sent Event streams:

**`GET /stream/events`** — Raw event stream
```json
{"event": "edge_added", "data": {"u": "A", "v": "B", "relation": "CAUSES", "weight": 0.72}, "timestamp": 1743000000.0}
```

**`GET /stream/insights`** — Materialized insight notifications
```json
{"event": "community_rebalanced", "data": {"communities": 7, "q": 0.43}, "timestamp": 1743000012.0}
{"event": "bridge_formed", "data": {"twin_id": "bridge_42", "src": "A", "dst": "B"}, "timestamp": 1743000015.0}
```

SSE connections are managed by an async multiplexer; each subscriber receives events from a shared broadcast queue. The server maintains a maximum of 500 concurrent SSE connections.

### 6. Implementation Notes (v2.51.0)

- **Back-pressure**: If the event queue exceeds `2 × window_size`, the `ingest()` call blocks until the consumer drains below `window_size`. This prevents unbounded memory growth under burst load.
- **Thalamic Scalability**: By unlocking Stage 1 preprocessing, `StreamAdapter` can handle >10,000 events/sec on modern multi-core hardware without degrading query performance.
- **Discretizer isolation**: Discretizer exceptions are caught and logged; they never propagate to the commit path or abort a batch.
- **Integration with Query Snapshot Isolation [Buchorn, 2026]**: The GlobalRebalancer's atomic swap triggers a broadcast to all active query contexts, which then upgrade to the new snapshot on their next hop. In-flight queries complete against the old snapshot.
- **Persistence**: The `Persistence` module [Buchorn, 2026] checkpoints the sliding-window buffer periodically for crash recovery.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.51.0

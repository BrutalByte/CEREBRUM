# CEREBRUM Performance Tuning Guide

**Version**: v2.21.0

This guide covers the key parameters affecting query latency, throughput, reasoning quality, and memory usage — and how to tune them for your specific workload.

---

## The Performance Levers (Quick Reference)

| Parameter | Default | Affects | Tune When |
|---|---|---|---|
| `beam_width` | 10 | Latency, recall | Graph is large or queries are slow |
| `max_hop` | 3 | Latency, recall depth | Questions require deeper reasoning |
| `n_trials` (DSCF) | 5 | Community quality, startup time | Startup is slow or Q is low |
| `alpha` (DSCF) | 0.5 | Modularity weight | Community boundaries are fuzzy |
| `beta` (DSCF) | 0.5 | LPA weight | Graph has strong local structure |
| `gamma` (TSC) | 0.0 | Centrality weight | Triadic closure is critical |
| `q_drift_threshold` | 0.05 | Rebalance frequency | Graph updates are frequent |
| `warm_start_strength` | 0.0 | First-hop variance | Probabilistic mode on sparse graphs |
| `window_size` (stream) | 10,000 | Memory, temporal context | High-volume ingest |
| `batch_size` (stream) | 50 | Ingest throughput, latency | High-frequency event streams |
| `CEREBRUM_WORKERS` | 4 | Concurrent query throughput | Multi-user deployments |
| `max_loops` | 1 | Iterative refinement depth | Complex multi-hop queries |
| `adaptive_tuning` | False | Auto-scale cap + interval | Long-running production loops |
| `adaptive_min_cap` / `adaptive_max_cap` | 1 / 20 | Materialization burst control | Saturated or sparse graph regions |

---

## 1. GPU Acceleration (DSCF / TSC)

### GPUDSCFEngine
The `GPUDSCFEngine` (Phase 22) provides a 10–100× speedup for community detection on large graphs ($N > 10,000$).

| Graph Size (N) | CPU Time (s) | GPU Time (s) | Speedup |
|---|---|---|---|
| 1,000 | 0.4s | 0.1s | 4× |
| 10,000 | 12.5s | 0.8s | 15× |
| 100,000 | 480s | 9.2s | 52× |
| 1,000,000 | 5,400s | 84s | 64× |

**Configuration:**
```python
from core.dscf_gpu import GPUDSCFEngine, GPUDSCFConfig
config = GPUDSCFConfig(
    device="auto",          # Best available (CUDA -> MPS -> HPU -> XLA -> CPU)
    alpha=0.5, beta=0.5,    # Standard DSCF
    temp_start=1.0,         # High noise for better global optima
    cooling=0.92,           # Fast annealing
)
engine = GPUDSCFEngine(config)
partitions = engine.detect(G)
```

**Hardware-Specific Tuning:**
- **NVIDIA CUDA**: Supports `dtype="float64"` for precision. Uses best-card selection by free VRAM.
- **Apple MPS**: Requires `dtype="float32"`. 10-20% slower than entry-level NVIDIA but zero-copy with system RAM.
- **Intel Gaudi (HPU)**: Ideal for million-node clusters; requires `habana-torch-plugin`.
- **Google TPU / AWS Trainium (XLA)**: Uses `mark_step()` barriers; extremely fast for massive batches.

**VRAM Pre-flight**: The engine estimates peak VRAM ($O(N \cdot \sqrt{N})$) and falls back to CPU automatically if the card is too small, preventing OOM crashes.

---

## 2. Query Latency

### beam_width
The single most impactful parameter for latency. Beam width controls how many candidates are evaluated at each hop.

| beam_width | Latency (21-node) | Latency (10K-node) | H@10 (MetaQA-3hop) |
|---|---|---|---|
| 3 | 0.9ms | 2.1ms | 0.271 |
| 5 | 1.2ms | 3.4ms | 0.294 |
| **10** | **1.8ms** | **6.3ms** | **0.248** |
| 20 | 3.1ms | 14.8ms | 0.331 |
| 50 | 7.4ms | 48.2ms | 0.339 |

**Rule of thumb**: `beam_width=10` is the quality/latency sweet spot for most graphs. Increase to 20–50 only for high-stakes queries where latency budget is relaxed (>50ms acceptable).

### max_hop
Each additional hop multiplies the search space by approximately `beam_width × avg_degree`.

```
Total candidates ≈ beam_width ^ max_hop
```

For a graph with avg_degree=5 and beam_width=10:
- max_hop=2: ~100 candidates
- max_hop=3: ~1,000 candidates
- max_hop=4: ~10,000 candidates (consider increasing beam_width instead)

**Recommendation**: Start at `max_hop=3`. Increase to 4 only if 3-hop recall is inadequate on your specific queries.

---

## 2. Community Detection Quality vs. Speed

### n_trials (DSCF)
DSCF is stochastic; running multiple trials and selecting the highest-modularity result improves Q.

| n_trials | Avg Q (toy_graph) | Startup time |
|---|---|---|
| 1 | 0.38 | Fast |
| 3 | 0.41 | Moderate |
| **5** | **0.43** | **Recommended** |
| 10 | 0.44 | Slow |

For production, compute communities once at startup or cache them. Communities only need recomputation when the graph changes substantially (triggered by `GlobalRebalancer`).

### Algorithm selection
| Algorithm | Q | Speed | Use when |
|---|---|---|---|
| `dscf` | Highest | Moderate | Default — best reasoning quality |
| `leiden` | High | Moderate | Large graphs where DSCF is slow |
| `lpa` | Lower | Fast | Very large graphs (>1M nodes) where startup time matters |

---

## 3. Streaming Ingest Throughput

### batch_size and flush_interval_ms
Events are batched before commit to amortize graph write cost.

```python
StreamAdapter(
    base_adapter=adapter,
    batch_size=50,              # events per commit — increase for high-volume ingest
    flush_interval_ms=100,      # max wait before commit — decrease for low-latency
    window_size=10_000,         # recency buffer size — reduce to save memory
)
```

| Scenario | batch_size | flush_interval_ms | Throughput |
|---|---|---|---|
| Low-latency IoT | 10 | 20 | ~500 events/s |
| General streaming | 50 | 100 | ~2,000 events/s |
| High-throughput ingest | 200 | 500 | ~8,000 events/s |

**Memory**: Each event in the sliding window uses ~200 bytes. `window_size=10,000` → ~2MB.

### GlobalRebalancer tuning
```python
GlobalRebalancer(
    adapter,
    q_drift_threshold=0.05,   # lower = more frequent rebalancing (higher quality, more CPU)
    min_interval_s=30.0,      # minimum seconds between rebalances
)
```

Rebalancing is expensive ($O(N \log N)$). For high-ingest deployments, set `min_interval_s=120` to prevent rebalance storms during burst traffic.

---

## 4. Probabilistic Mode (Bayesian Beam Search)

Probabilistic mode adds Thompson sampling overhead (~15% latency increase) but improves quality on sparse graphs and provides confidence intervals.

```python
BeamTraversal(
    probabilistic=True,
    warm_start_strength=5.0,    # 0.0 = uniform prior (cold), 5.0 = CSA-seeded (warm)
)
```

**warm_start_strength** guidance:
- `0.0` — standard uniform Beta(1,1) prior (v0.4.0 behavior)
- `2.0` — mild warm-start (reduces first-hop variance ~40%)
- `5.0` — strong warm-start (reduces first-hop variance ~85%, recommended for sparse graphs)
- `10.0` — aggressive seeding (can reduce diversity; use only on very sparse graphs)

---

## 5. Embedding Choice

| Engine | Semantic quality | Speed | Use when |
|---|---|---|---|
| `RandomEngine(dim=64)` | None (structural only) | Fastest | Graph topology dominates; no text labels |
| `RandomEngine(dim=128)` | None | Fast | Larger graphs needing more embedding capacity |
| `SentenceEngine` | High | Slow (GPU-accelerated) | Entity names carry semantic meaning |
| `RotatEEngine` (trained) | Relational | Medium (after training) | Rich relation diversity |

For pure structural reasoning (graphs where entity names are opaque IDs), `RandomEngine` achieves the same H@10 as `SentenceEngine` because the community consensus term ($\beta$) dominates over semantic similarity ($\alpha$).

---

## 6. ResourceGovernor Budgets

```python
from core.hardware import ResourceGovernor

governor = ResourceGovernor(
    max_cpu_percent=80,         # cap CPU usage across all queries
    max_memory_mb=2048,         # hard memory limit
    query_timeout_ms=5000,      # abort queries exceeding this
    rem_cpu_budget=0.15,        # fraction of CPU reserved for REM cycle
)
```

For latency-critical deployments, set `rem_cpu_budget=0.05` to give more CPU to query processing and reduce REM cycle priority.

---

## 7. Memory Optimization

| Component | Memory | Reduction strategy |
|---|---|---|
| Embedding matrix | `N × dim × 4 bytes` | Reduce `dim` (64→32 halves memory) |
| Community map | `N × 8 bytes` | Minimal — unavoidable |
| Sliding window buffer | `W × 200 bytes` | Reduce `window_size` |
| Materialized path store | Variable | Call `adapter.clear_path_cache()` |
| InsightEvent graph | Capped at 10K nodes | Reduce `meta_insight_engine.max_events` |

For a 100K-node graph with `dim=64`: embedding matrix ≈ 25MB. For 1M nodes: ≈ 250MB.

---

## 8. Benchmark Your Deployment

```bash
# Run the synthetic benchmark
python benchmarks/synthetic_eval.py --nodes 1000 --edges 5000 --queries 100

# Run baseline comparison
python benchmarks/baseline_comparison.py --algorithm dscf --beam-width 10

# Feature Impact Benchmark (Phase 77): measures Hits@1, Hits@5, MRR
# across baseline / +engram / +looped / +full configurations
python benchmarks/feature_impact_benchmark.py

# Profile a specific configuration
python -m cProfile -s cumulative benchmarks/synthetic_eval.py 2>&1 | head -30
```

See `benchmarks/README.md` for full benchmark documentation.

---

## 9. Looped Beam Traversal (Phase 70)

`LoopedBeamTraversal` applies beam traversal T times with seed expansion between loops, guided by PredictiveCodingEngine and Engram channels. This increases answer quality but multiplies latency.

```python
# Opt into iterative refinement (Phase 70)
answers = graph.query("entity", max_hops=3, beam_width=10, max_loops=3)
```

| max_loops | Latency multiplier | Quality gain (MRR) |
|---|---|---|
| 1 (default) | 1× baseline | baseline |
| 2 | ~1.9× | +12–18% |
| 3 | ~2.8× | +18–25% |
| 5 | ~4.6× | +22–28% |

**Adaptive exit gate**: loops terminate early when `|ΔPE| < γ` (prediction error stabilises) or answer-set Jaccard ≥ θ (answers stop changing). In practice, 3-loop queries often exit after 2 iterations.

**Latency budget guidance:**
- Real-time queries (<50ms): `max_loops=1` (default, no change)
- Interactive queries (50–500ms): `max_loops=2`
- Batch / offline: `max_loops=3–5`

---

## 10. Autonomous Discovery Loop Tuning (Phase 74 / 82)

The `AutonomousDiscoveryLoop` runs in the background. Its parameters affect CPU and graph saturation, not query latency.

### Per-cycle cap (`max_materializations_per_cycle`)
Controls how many new edges can be added per cycle. Start conservative; increase once you observe acceptable approval rates.

| Deployment type | Recommended cap |
|---|---|
| Exploratory / dev | 5–10 |
| Production (initial) | 3–5 |
| Production (established, high approval rate) | 10–20 |

### Cycle interval (`cycle_interval`)
Shorter intervals = more frequent discoveries, higher background CPU. Set based on how quickly your source data changes.

| Data velocity | Recommended interval |
|---|---|
| Static graph (infrequent ingest) | 600–3600 s |
| Moderate streaming | 300–600 s |
| High-frequency ingest | 120–300 s |

### Adaptive tuning (`adaptive_tuning=True`) — Phase 92
When enabled, `DiscoveryCalibrator` community weights automatically scale both the cap and interval:
- **Underexplored communities** → higher cap + shorter interval (more aggressive)
- **Saturated communities** → lower cap + longer interval (backing off)

Configure bounds to prevent extreme swings:
```python
LoopConfig(
    adaptive_tuning=True,
    adaptive_min_cap=2,
    adaptive_max_cap=15,
    adaptive_min_interval=120.0,
    adaptive_max_interval=3600.0,
)
```

### Circuit breaker
`min_approval_rate=0.6` with `circuit_breaker_window=20` pauses materialization when fewer than 60% of the last 20 decisions were approved. If tripped frequently, reduce the cap or increase `cycle_interval`.

---

## 11. ChemicalModulator Overhead (Phase 68)

`ChemicalModulator` is a lightweight state machine updating 5 float scalars per query. Overhead is negligible (<0.1ms). No tuning is required for latency.

The indirect performance effect is positive: Arousal dynamically scales `beam_width` during high-uncertainty queries, and Reinforcement adjusts CSA weights, reducing wasted beam candidates on known-bad paths over time.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

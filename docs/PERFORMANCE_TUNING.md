# CEREBRUM Performance Tuning Guide

**Version**: v1.1.0

This guide covers the key parameters affecting query latency, throughput, reasoning quality, and memory usage — and how to tune them for your specific workload.

---

## The Performance Levers (Quick Reference)

| Parameter | Default | Affects | Tune When |
|---|---|---|---|
| `beam_width` | 10 | Latency, recall | Graph is large or queries are slow |
| `max_hop` | 3 | Latency, recall depth | Questions require deeper reasoning |
| `n_trials` (DSCF) | 5 | Community quality, startup time | Startup is slow or Q is low |
| `q_drift_threshold` | 0.05 | Rebalance frequency | Graph updates are frequent |
| `warm_start_strength` | 0.0 | First-hop variance | Probabilistic mode on sparse graphs |
| `window_size` (stream) | 10,000 | Memory, temporal context | High-volume ingest |
| `batch_size` (stream) | 50 | Ingest throughput, latency | High-frequency event streams |
| `CEREBRUM_WORKERS` | 4 | Concurrent query throughput | Multi-user deployments |

---

## 1. Query Latency

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

# Profile a specific configuration
python -m cProfile -s cumulative benchmarks/synthetic_eval.py 2>&1 | head -30
```

See `benchmarks/README.md` for full benchmark documentation.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

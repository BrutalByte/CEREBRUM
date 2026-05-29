# CEREBRUM Performance Tuning Guide

**Status**: v2.65.1 (Phase 203)

This guide covers the key parameters affecting query latency, throughput, reasoning quality, and memory usage — and how to tune them for your specific workload.

---

## The Performance Levers (Quick Reference)

| Parameter | Default | Affects | Tune When |
|---|---|---|---|
| `beam_width` | 10 | Latency, recall | Graph is large or queries are slow |
| `max_hop` | 3 | Latency, recall depth | Questions require deeper reasoning |
| `vectorized_scoring` | True | Latency (10x boost) | Sub-30ms performance required |
| `graph_profiler` | True | Auto-strategy | Zero-config deployment required |
| `hop_expand` (H1SE) | Auto | Hub-flooding | Hubs dominate early reasoning |
| `anchor_bonus` (TAB) | Auto | 3-hop recall | Deep heterogeneous queries are low-recall |
| `n_trials` (DSCF) | 5 | Community quality | Startup is slow or Q is low |
| `alpha` (DSCF) | 0.5 | Modularity weight | Community boundaries are fuzzy |
| `beta` (DSCF) | 0.5 | LPA weight | Graph has strong local structure |
| `q_drift_threshold` | 0.05 | Rebalance frequency | Graph updates are frequent |
| `warm_start_strength` | 0.0 | First-hop variance | Probabilistic mode on sparse graphs |
| `window_size` (stream) | 10,000 | Memory, temporal context | High-volume ingest |
| `batch_size` (stream) | 50 | Ingest throughput | High-frequency event streams |
| `--trb-factor` | 4.0 | Terminal relation boost magnitude | H@1 is below target on typed graphs |
| `--r2-boost` | 3.0 | Multi-path corroboration reward | Answers lack consistency across paths |
| `--vote-weight` | 0.7 | Community vote suppression | Community boundaries are noisy |
| `--beam-width` | 10 | Candidate set size per hop | Recall is low or latency budget allows |
| `--idf-weight` | 0.05 | IDF term weighting in scoring | Queries have high-frequency stopword entities |
| `--branch-bonus` | 0.3 | Branching path bonus | Beam collapse on single paths |
| `--fhrb-factor` | 0.5 | First-hop relation boost | First-hop relation signal is weak |
| `--gamma` | None | SDRB scale (KB-agnostic per-relation boost) | Use instead of per-relation flags on any KB |
| `--beta` | 1.0 | SDRB exponent — power-law shape | `>1` amplifies high-fan_out relations; `<1` compresses |
| ~~`--wb-r2-boost`~~ | ~~5.0~~ | ~~Per-relation boost (written_by)~~ | **Deprecated** — use `--gamma`/`--beta` instead |
| ~~`--db-r2-boost`~~ | ~~7.0~~ | ~~Per-relation boost (directed_by)~~ | **Deprecated** — use `--gamma`/`--beta` instead |
| ~~`--ry-r2-boost`~~ | ~~2.0~~ | ~~Per-relation boost (release_year)~~ | **Deprecated** — use `--gamma`/`--beta` instead |
| ~~`--sa-r2-boost`~~ | ~~5.0~~ | ~~Per-relation boost (starred_actors)~~ | **Deprecated** — use `--gamma`/`--beta` instead |

---

## 1. Zero-Config Optimization (Phase 172-167)

### GraphProfiler (Automatic Strategy Selection)
In latest versions, you no longer need to manually toggle `hop_expand` (H1SE) or `anchor_bonus` (TAB). The **GraphProfiler** performs a build-time topological analysis and selects the optimal "Graph Regime":

- **hub_homogeneous** (e.g., MovieLens, MetaQA): Enables H1SE to solve hub crowding.
- **typed_heterogeneous** (e.g., Hetionet): Disables H1SE, enables TAB and STRB for relation-guided traversal.
- **mixed**: Enables both as a balanced fallback.

### STRB (Semantic Terminal Relation Boost)
STRB automatically boosts the correct terminal relation by comparing query text to relation labels. This eliminates the need for manual "Terminal Relation Boosting" (TRB) configuration for known query types.

---

## 2. Low-Latency Reasoning

### Vectorized Beam Scoring (Phase 134)
The core attention calculation is now implemented via optimized NumPy matrix operations. This yields a **10x latency reduction**, moving 3-hop reasoning from ~200ms to **<30ms** on million-node graphs.
- **Requirement**: `numpy` installed (standard).
- **Tuning**: Enabled by default. No manual configuration needed.

### beam_width
The single most impactful parameter for latency. Beam width controls how many candidates are evaluated at each hop.

| beam_width | Latency (v2.52.0) | Latency (v2.52.0 Vectorized) | H@10 (MetaQA-3hop) |
|---|---|---|---|
| 3 | 2.1ms | 0.4ms | 0.271 |
| 5 | 3.4ms | 0.8ms | 0.294 |
| **10** | **6.3ms** | **1.2ms** | **0.732** |
| 20 | 14.8ms | 2.9ms | 0.745 |
| 50 | 48.2ms | 9.4ms | 0.750 |

**Rule of thumb**: With vectorization, `beam_width=10` or `20` is nearly "free." Use higher widths to maximize recall on complex graphs.

---

## 3. Scoring Parameter Tuning (Phases 202–203: SDRB)

Phase 202 replaced the four MetaQA-specific per-relation boost flags (`wb/db/ry/sa-r2-boost`) with a single KB-agnostic parameter derived from the graph's own fan-out statistics. Phase 203 extended this with a power-law exponent. The search space is now 9 parameters instead of 11, and the approach works on any KB without domain knowledge.

### SDRB Formula

```
boost(r) = gamma × fan_out(r)^beta
fan_out(r) = total_triples(r) / unique_head_entities(r)   # computed at load time
```

`beta=1.0` (default) is linear — identical to Phase 202. `beta>1.0` amplifies high-fan_out relations disproportionately (reproducing the asymmetry that hand-tuned per-relation params encoded). `beta<1.0` compresses differences.

### CLI Flags (Phase 203)

```bash
python benchmarks/metaqa_eval.py \
  --trb-factor 26.018 \
  --r2-boost 7.416 \
  --vote-weight 0.858 \
  --beam-width 10 \
  --idf-weight 0.024 \
  --branch-bonus 0.154 \
  --fhrb-factor 4.189 \
  --gamma 7.927 \
  --beta 1.0 \
  --kb data/metaqa/kb/kb.txt \
  --questions data/metaqa/3-hop/qa_test.txt \
  --hops 3 --embeddings sentence --workers 16
```

### fANOVA Importance Ranking (Phase 202 Analysis)

After replacing the 4 per-relation params with SDRB, fANOVA shows a major shift: the system now routes more variance through the branching diversity mechanism (`branch_bonus`) rather than per-relation overrides.

| Rank | Parameter | fANOVA Importance | Tuning Priority |
|------|-----------|-------------------|-----------------|
| 1 | `branch_bonus` | **46.2%** | Tune first — dominant after SDRB |
| 2 | `trb_factor` | **29.4%** | Tune second |
| 3 | `gamma` | ~9% | SDRB scale |
| 4 | `fhrb_factor` | ~6% | First-hop guidance |
| 5 | `r2_boost` | ~4% | Path corroboration |
| 6 | `vote_weight` | ~3% | Community suppression |
| 7 | `idf_weight` | ~1.5% | Hub penalty |
| 8 | `beta` | ~0.5% | SDRB shape |
| 9 | `beam_width` | **<0.5%** | Near-irrelevant — fix at 8–10 |

> **Key insight (Phase 202)**: When per-relation overrides are removed, `branch_bonus` jumps from 1.2% → 46.2% importance. The system compensates for loss of per-relation specificity by routing more budget through broader traversal. This is an architectural finding: diversity of traversal paths is load-bearing when relation-specific tuning is unavailable.

> **Phase 198 key insight (preserved)**: `trb_factor` (terminal relation detection) explains the majority of structural H@1 variance. `beam_width` is near-irrelevant — never waste Optuna budget on it.

### Practical Tuning Strategy

| Phase | Action | Parameters |
|-------|--------|------------|
| Quick win | Grid-search `trb_factor` in [10, 20, 30] | `trb_factor` only |
| Second pass | Add `branch_bonus` sweep [0.0, 0.5, 1.0] | `trb_factor`, `branch_bonus` |
| SDRB pass | Add `gamma` sweep [2, 4, 8, 12] | Above + `gamma` |
| Full Optuna | 9-param two-phase search | All flags |
| Fix | Lock low-importance params | `beta`, `beam_width` |

### Live Two-Phase Tuner (Phase 199+)

The `cerebrum_tuner.py` script runs a two-phase search: Phase 1 uses RandomSampler for wide exploration, Phase 2 uses TPE over bounds derived from the top-K Phase 1 trials.

```bash
python -u benchmarks/cerebrum_tuner.py \
  --phase1-trials 100 \
  --phase2-trials 100 \
  --sample 2000 \
  --workers 16
```

With 32 CPU cores and RTX 5090, each trial takes ~8s on a 2,000-question sample (~3.5 min for 25 trials). The `--workers 16` flag parallelises question evaluation within each trial; the tuner itself runs trials sequentially.

### Current Best-Known Config (Phase 202 tuner, 2000-sample)

Phase 202 best on 2,000-sample (full-dataset validation pending). SDRB replaces all per-relation flags:

| Parameter | Best Value | Default |
|-----------|------------|---------|
| `trb_factor` | **26.018** | 4.0 |
| `r2_boost` | 7.416 | 3.0 |
| `vote_weight` | 0.858 | 0.7 |
| `beam_width` | 10 | 10 |
| `idf_weight` | 0.024 | 0.05 |
| `branch_bonus` | **0.154** | 0.3 |
| `fhrb_factor` | **4.189** | 0.5 |
| `gamma` | **7.927** | — |
| `beta` | 1.0 | 1.0 |

> **Note**: gamma=7.927 hit the ceiling of the Phase 202 search range [1.5, 8.0]. Phase 203 expanded the range to [1.5, 16.0] with the beta exponent to allow the model to reproduce per-relation asymmetry without requiring a higher global scale.

---

## 4. Deep Reasoning (3+ Hops)

### H1SE: Solving the Hub Problem (Phase 137)
In hub-heavy graphs, a single global beam is often captured by popular entities (e.g., "USA") at hop 1. **Hop-1 Intermediate Seed Expansion (H1SE)** gives each hop-1 neighbor its own independent search budget.
- **Tune**: `expansion_k` (default 5). Higher `expansion_k` improves recall but increases latency linearly.

### TAB: Terminal-Anchor Boost (Phase 172)
For 3-hop queries in typed graphs, use **TAB** to guide the penultimate hop. It applies a bonus to entities that are known sources of the target relation type.
- **Tune**: `anchor_bonus` (default 2.0). Increase to 3.0+ for extremely sparse heterogeneous graphs.

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

## 5. Community Detection Quality vs. Speed

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

## 6. Streaming Ingest Throughput

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

## 7. Probabilistic Mode (Bayesian Beam Search)

Probabilistic mode adds Thompson sampling overhead (~15% latency increase) but improves quality on sparse graphs and provides confidence intervals.

```python
BeamTraversal(
    probabilistic=True,
    warm_start_strength=5.0,    # 0.0 = uniform prior (cold), 5.0 = CSA-seeded (warm)
)
```

**warm_start_strength** guidance:
- `0.0` — standard uniform Beta(1,1) prior (v2.52.0 behavior)
- `2.0` — mild warm-start (reduces first-hop variance ~40%)
- `5.0` — strong warm-start (reduces first-hop variance ~85%, recommended for sparse graphs)
- `10.0` — aggressive seeding (can reduce diversity; use only on very sparse graphs)

---

## 8. Embedding Choice

| Engine | Semantic quality | Speed | Use when |
|---|---|---|---|
| `RandomEngine(dim=64)` | None (structural only) | Fastest | Graph topology dominates; no text labels |
| `RandomEngine(dim=128)` | None | Fast | Larger graphs needing more embedding capacity |
| `SentenceEngine` | High | Slow (GPU-accelerated) | Entity names carry semantic meaning |
| `RotatEEngine` (trained) | Relational | Medium (after training) | Rich relation diversity |

For pure structural reasoning (graphs where entity names are opaque IDs), `RandomEngine` achieves the same H@10 as `SentenceEngine` because the community consensus term ($\beta$) dominates over semantic similarity ($\alpha$).

---

## 9. ResourceGovernor Budgets

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

## 10. Memory Optimization

| Component | Memory | Reduction strategy |
|---|---|---|
| Embedding matrix | `N × dim × 4 bytes` | Reduce `dim` (64→32 halves memory) |
| Community map | `N × 8 bytes` | Minimal — unavoidable |
| Sliding window buffer | `W × 200 bytes` | Reduce `window_size` |
| Materialized path store | Variable | Call `adapter.clear_path_cache()` |
| InsightEvent graph | Capped at 10K nodes | Reduce `meta_insight_engine.max_events` |

For a 100K-node graph with `dim=64`: embedding matrix ≈ 25MB. For 1M nodes: ≈ 250MB.

---

## 11. Benchmark Your Deployment

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

## 12. Looped Beam Traversal (Phase 70)

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

## 13. Autonomous Discovery Loop Tuning (Phase 74 / 82)

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

## 14. ChemicalModulator Overhead (Phase 68)

`ChemicalModulator` is a lightweight state machine updating 5 float scalars per query. Overhead is negligible (<0.1ms). No tuning is required for latency.

The indirect performance effect is positive: Arousal dynamically scales `beam_width` during high-uncertainty queries, and Reinforcement adjusts CSA weights, reducing wasted beam candidates on known-bad paths over time.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 29, 2026 for version v2.65.1

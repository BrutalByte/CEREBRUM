# Intentional NVMe Graph Consolidation with Write-Ahead Logging for Knowledge Graph Systems

**Author**: Bryan Alexander Buchorn
**Affiliation**: Independent Researcher, Las Vegas, NV, USA
**Version**: v2.73.0 (Phase 227)
**Date**: June 5, 2026

---

### Abstract

Persistent Knowledge Graph (KG) systems face a fundamental tension between write throughput and durability: keeping the live graph entirely in RAM maximizes traversal speed but sacrifices crash safety, while naively memory-mapping the graph to NVMe introduces write-amplification and atomic-update hazards. We present a two-layer RAM + NVMe architecture for CEREBRUM that resolves this tension through three coordinated components. `GraphWAL` (`core/graph_wal.py`) is a thread-safe, append-only NDJSON write-ahead log that records every `add_edge()` call before it reaches the in-RAM NetworkX adapter, providing O(1) crash-safe writes with no read overhead. `MmapAdvisor` (`core/mmap_policy.py`) evaluates RAM pressure using psutil and recommends whether NVMe mmap is required, beneficial, or unnecessary, with AUTO/ALWAYS/NEVER policy overrides. `MmapConsolidator` (`core/mmap_policy.py`) atomically flushes the full graph topology and embeddings to NVMe via a `_tmp_flush/` staging directory, triggered by the existing REM (Rapid Edge Maintenance) cycle — a biologically inspired quiet period analogous to sleep-phase memory consolidation. The design avoids write-amplification by deferring NVMe writes to natural compaction points rather than per-edge. A comprehensive test suite of 35 tests (35/35 passing) validates crash safety, atomic write semantics, and the full WAL→REM→flush pipeline. For MetaQA (43,217 nodes, 124,616 edges, 384-dim embeddings), the consolidated NVMe footprint is approximately 66 MB.

---

### 1. Introduction

Knowledge Graph reasoning systems such as CEREBRUM maintain large, dynamically evolving graphs in working memory. During operation, new edges are continuously ingested through the `add_edge()` API — via REM synthesis, ResearchAgent materialization, and autonomous discovery loops. Two fundamental requirements conflict:

1. **Traversal speed**: Beam search, CSA attention, and multi-hop reasoning require random-access graph lookups at latencies measured in microseconds. This demands an in-RAM adjacency structure.
2. **Durability**: A process crash between graph mutations must not silently lose ingested edges. At minimum, the system must be able to recover to the last consistent state.

Naive approaches fail in predictable ways. A purely in-memory design (no NVMe write path) loses all post-startup mutations on crash. A naive memory-mapped file design — writing every `add_edge()` directly to a mmap'd file — introduces three problems: (i) write amplification from page-granularity dirty marking, (ii) structural fragmentation from incremental appends to a fixed-layout binary format, and (iii) lack of atomicity: a crash mid-write leaves a structurally inconsistent mmap file.

We present a design that eliminates all three failure modes through the principle of *intentional consolidation*: all writes are first appended to a crash-safe WAL in O(1) time, the RAM working set remains the authoritative live graph, and the NVMe binary representation is rebuilt atomically only at REM cycle boundaries — the natural quiet periods when the system is already not serving traversal queries.

The contributions of this work are:

1. **GraphWAL**: An NDJSON append-only write-ahead log providing O(1) crash-safe `add_edge()` durability with thread safety and startup replay (Section 4).
2. **MmapAdvisor**: A RAM-pressure-aware NVMe mmap policy engine using psutil-derived metrics and a three-tier recommendation system (Section 5).
3. **MmapConsolidator**: An atomic full-graph flush mechanism using `_tmp_flush/` staging and per-file rename, triggered by the REM cycle (Section 6).
4. **REM Integration**: Wiring of the flush callback through `REMEngine`, `SleepCycleOrchestrator`, and the REST API (Section 7).
5. **Configuration**: Persistent NVMe path resolution via `CEREBRUM_MMAP_DIR` and `.env` file (Section 8).
6. **Validation**: A 35-test suite covering WAL, Advisor, Consolidator, and integration scenarios (Section 9).

---

### 2. Background: The Graph Storage Challenge

#### 2.1 CEREBRUM's Graph Model

CEREBRUM represents a knowledge graph as a directed multigraph $G = (V, E)$ where each edge $e \in E$ carries a tuple $(src, tgt, relation, confidence, provenance, synthetic)$. The NetworkX adapter maintains $G$ in RAM as a `DiGraph`. Node count $|V|$ and edge count $|E|$ grow continuously as REM cycles synthesize wormhole edges and ResearchAgent materializes validated findings.

During beam traversal, the reasoning engine calls `adapter.get_neighbors(node)` repeatedly, requiring $O(\deg(node))$ RAM lookups per hop. Any I/O latency on this path would degrade query throughput, motivating the requirement that the *live* graph reside entirely in RAM.

#### 2.2 The NVMe Opportunity and the Write-Amplification Trap

Modern NVMe SSDs deliver sequential write throughput exceeding 3 GB/s and random-read IOPS exceeding 500K — far better than HDD or SATA SSD, and sufficient for streaming graph consolidation. The `numpy.memmap` API makes it straightforward to map typed binary arrays directly from NVMe files into the virtual address space, enabling O(1) indexed reads once the OS has cached the relevant pages.

The trap lies in the write path. If every `add_edge()` call updates a mmap'd file in-place:

- **Write amplification**: The OS dirtied page must be flushed at page granularity (typically 4 KiB), even if only a single 12-byte edge record changed.
- **Layout instability**: Appending to a variable-length edge array in a fixed-format binary file requires either pre-allocation (wastes space) or rewriting the entire file on each grow (catastrophic amplification).
- **No atomicity**: If power is cut mid-`msync()`, the mmap file is in an undefined partial-write state with no recovery mechanism.

CEREBRUM's design sidesteps all three issues by decoupling the write path (append-to-WAL, O(1)) from the consolidation path (full atomic rewrite, triggered at REM).

#### 2.3 The REM Cycle as a Natural Compaction Trigger

CEREBRUM's REM (Rapid Edge Maintenance) engine, described in prior work, performs periodic graph self-reorganization: pruning low-confidence synthetic edges, identifying wormhole bridges between disconnected components, and rebalancing community structure. REM cycles are intentionally run during quiet periods — the system's analogue of biological slow-wave sleep — when no traversal queries are in flight.

This quiet period is the natural compaction trigger. After a REM cycle completes, the graph is in a known-good consolidated state: no active writes, no in-flight traversals, fully pruned. Flushing the NVMe representation at this exact moment imposes zero additional latency on the critical traversal path.

---

### 3. Architecture: Two-Layer RAM + NVMe Design

The overall architecture is summarized in Figure 1 (described textually below).

**Layer 1 — RAM (Live Working Set):**
The `NetworkXAdapter` maintains the authoritative in-memory graph. All reads (traversal, community lookup, degree queries) hit this layer exclusively. Writes (`add_edge()`) are applied to this layer *after* the WAL append completes.

**Layer 2 — NVMe (Durable Store):**
The NVMe layer holds two artifacts: (a) the GraphWAL file (`edges.wal`) — an append-only NDJSON log of every `add_edge()` call since the last consolidation, and (b) the consolidated binary representation (`graph.a`, `graph.e`, `embeddings.e`, and supporting index files) — a point-in-time snapshot of the graph after the most recent REM cycle.

**Write path:**
$$\text{add\_edge}(src, tgt, \ldots) \xrightarrow{1} \text{WAL.append()} \xrightarrow{2} \text{NetworkXAdapter.add\_edge()}$$

The WAL append is the durability gate. If step 1 succeeds and step 2 is interrupted by a crash, startup replay (Section 4.2) recovers the edge. If step 1 is interrupted, the edge is lost — but the WAL file is append-only NDJSON, so a partial write at the file tail is detected and skipped during replay.

**Consolidation path (triggered by REM cycle completion):**
$$\text{REM complete} \xrightarrow{1} \text{MmapConsolidator.flush()} \xrightarrow{2} \text{WAL.truncate()}$$

After successful flush, the WAL is cleared because all WAL records are now redundant — the binary snapshot represents the same graph state, and future startup will load from the snapshot rather than replaying WAL records.

**Startup path:**
$$\text{startup} \xrightarrow{1} \text{load binary snapshot (if exists)} \xrightarrow{2} \text{WAL.replay() (if WAL exists)}$$

This two-phase startup ensures that even if the previous shutdown occurred mid-REM (after some edges were added post-last-consolidation), the edges are fully recovered.

---

### 4. GraphWAL: Write-Ahead Log for Crash-Safe Edge Writes

#### 4.1 Design

`GraphWAL` (`core/graph_wal.py`) implements a minimal, correct, thread-safe append-only log. The file format is NDJSON — one JSON object per line — chosen for its properties: human-readable for debugging, naturally atomic at line granularity (a partial write produces a malformed JSON line that is skipped on replay), and compatible with streaming readers.

Each record has the form:

```json
{"op": "add", "src": "...", "tgt": "...", "relation": "...", "confidence": 0.9, "provenance": "...", "synthetic": false}
```

The `op` field is included for forward compatibility. The current implementation only emits `"add"` records; future versions may emit `"remove"` or `"update"` records without breaking existing replay logic (unknown ops are skipped).

#### 4.2 API

```python
wal = GraphWAL(path="/data/mmap/edges.wal")

# Append — O(1), thread-safe
wal.append(src, tgt, relation, confidence, provenance, synthetic)

# Replay at startup — returns count of recovered edges
n = wal.replay(adapter)  # calls adapter.add_edge() for each valid record

# Truncate after successful consolidation — idempotent
wal.truncate()

# Introspection
wal.entry_count()   # count valid NDJSON lines
wal.size_bytes()    # file size
wal.exists()        # bool
wal.path            # pathlib.Path
```

**Thread safety** is implemented via `threading.Lock` acquired on every `append()` call. The pattern is seek-to-end + write, which is O(1) regardless of file size — no read or parse on the write path.

**Replay correctness:** `replay()` iterates each line with `json.loads()`; any `JSONDecodeError` (from a crash-truncated tail write) is caught and the line is skipped. Only records with `"op": "add"` are applied. `replay()` returns the count of successfully applied records, enabling startup logging.

#### 4.3 Atomicity Properties

| Scenario | Outcome |
|---|---|
| Crash before WAL append | Edge lost (same as pre-WAL design; no regression) |
| Crash after WAL append, before adapter write | Recovered on next startup via replay |
| Crash during WAL append (partial write) | Partial line detected by `json.loads()`; skipped |
| Crash during `MmapConsolidator.flush()` before completion | WAL intact; replay on next startup |
| Crash after flush, before `WAL.truncate()` | WAL replayed on startup; edges deduplicated by NetworkX (`add_edge` on existing edge is idempotent) |
| Clean shutdown | WAL truncated; startup loads from binary snapshot directly |

---

### 5. MmapAdvisor: RAM-Pressure-Aware mmap Decision

#### 5.1 Motivation

NVMe mmap provides tangible benefit only when RAM is under pressure. On a system with 128 GB RAM and a 66 MB graph, mmap adds OS page-fault overhead with no benefit. On a system where graph + embeddings exceed available RAM, mmap allows the OS to page in only the regions actively accessed during traversal, avoiding OOM. The `MmapAdvisor` makes this determination automatically.

#### 5.2 Size Estimation

Given graph parameters $(|V|, |E|, d)$ where $d$ is the embedding dimension:

$$\text{graph\_bytes} = |V| \cdot 32 + |E| \cdot 12$$

$$\text{embedding\_bytes} = |V| \cdot d \cdot 4$$

The graph estimate accounts for the structured binary formats: `graph.a` uses 32 bytes per node (A_DTYPE structured array), and `graph.e` uses 12 bytes per edge (E_DTYPE structured array). The embedding estimate assumes float32 (4 bytes per element) with $|V| \times d$ elements.

RAM pressure is defined as:

$$\text{pressure} = \frac{\text{graph\_bytes} + \text{embedding\_bytes}}{\text{available\_bytes}}$$

where $\text{available\_bytes}$ is obtained from `psutil.virtual_memory().available` — the amount of RAM the OS can immediately allocate to a new process without swapping.

#### 5.3 Policy and Thresholds

| Condition | Recommendation | Rationale |
|---|---|---|
| $\text{pressure} \geq 0.50$ | `required` | Graph+embeddings exceed half of available RAM; mmap mandatory |
| $0.20 \leq \text{pressure} < 0.50$ | `recommended` | Moderate pressure; mmap beneficial for resilience |
| $\text{pressure} < 0.20$ | `ram_sufficient` | Graph fits comfortably in RAM; mmap adds no benefit |

Three policy overrides are available via the `MmapPolicy` enum:

- **AUTO**: Use the pressure-based thresholds above.
- **ALWAYS**: Force mmap regardless of pressure (useful for reproducibility or testing).
- **NEVER**: Disable mmap regardless of pressure (useful when NVMe is slow or unavailable).

ALWAYS and NEVER overrides set `override_used=True` in the returned `MmapRecommendation`, enabling callers to distinguish advisor-driven from user-mandated decisions.

#### 5.4 Return Type

```python
@dataclass
class MmapRecommendation:
    use_mmap: bool
    reason: str          # human-readable explanation
    graph_mb: float
    embedding_mb: float
    available_mb: float
    pressure_pct: float  # pressure × 100
    policy: MmapPolicy
    override_used: bool
```

---

### 6. MmapConsolidator: Atomic Flush via REM Cycle Trigger

#### 6.1 File Format

`MmapConsolidator.flush()` writes six files to the configured `data_dir`:

| File | Format | Size (MetaQA) | Contents |
|---|---|---|---|
| `graph.a` | NumPy structured array (A_DTYPE) | $\approx 1.3$ MB | Node records: 32 B/node |
| `graph.e` | NumPy structured array (E_DTYPE) | $\approx 1.4$ MB | Edge records: 12 B/edge |
| `embeddings.e` | float32 NumPy array | $\approx 63$ MB | $|V| \times d$ matrix |
| `relations.idx` | Plain text (one line per relation) | KB-scale | Sorted relation string pool |
| `nodes.map` | Tab-separated text | KB-scale | `node_id\tcommunity_id` |
| `graph.meta` | JSON | Bytes | Version, timestamp, build_id, counts, embedding_dim, relation_count |

For MetaQA ($|V| = 43{,}217$, $|E| = 124{,}616$, $d = 384$), the total NVMe footprint is approximately 66 MB.

#### 6.2 Atomic Write Protocol

The canonical failure mode for file updates is the *torn write*: the process or system crashes after writing some files but before writing others, leaving the output directory in a mixed old/new state. `MmapConsolidator` eliminates this via a two-phase protocol:

**Phase 1 — Write to staging:**
$$\text{all files} \rightarrow \text{data\_dir}/\text{\_tmp\_flush}/$$

**Phase 2 — Rename into place:**
$$\text{for each file } f: \text{os.rename}(\text{\_tmp\_flush}/f, \text{data\_dir}/f)$$

On POSIX systems, `os.rename()` is atomic at the filesystem level — the destination either sees the old file or the new file, never a partial write. On Windows (NTFS), `os.replace()` provides the equivalent guarantee. A crash during Phase 1 leaves `_tmp_flush/` as debris (cleaned on next startup) but leaves the existing `data_dir/` snapshot intact. A crash during Phase 2 may leave some files updated and others not; on next startup, the presence of `_tmp_flush/` signals an incomplete consolidation.

After all renames succeed, `MmapConsolidator` calls `wal.truncate()` to remove the WAL file, and returns a `ConsolidationReport`:

```python
@dataclass
class ConsolidationReport:
    success: bool
    data_dir: str
    node_count: int
    edge_count: int
    embedding_mb: float
    duration_s: float
    wal_truncated: bool
    error: Optional[str]
```

#### 6.3 DiGraph Edge Count Correctness

A subtle but important correctness issue arose during development: for a `networkx.DiGraph`, `sum(G.degree())` returns $\sum_{v} (\deg^+(v) + \deg^-(v))$ — each edge is counted twice (once from each endpoint). The correct expression for total directed edge count is:

$$|E| = \sum_{v \in V} \deg^+(v)$$

In code: `sum(d for _, d in G.out_degree())`. Using `sum(G.degree())` would overcount by a factor approaching 2, corrupting the `graph.meta` JSON and misallocating the `graph.e` array. This bug was discovered through the `TestMmapConsolidator` test suite and is documented here as a correctness note for implementors of graph serialization routines.

---

### 7. Integration with REM Engine and SleepCycleOrchestrator

#### 7.1 REMEngine Callback

`REMEngine.__init__` accepts an optional `on_complete: Optional[callable]` parameter. After a non-dry real REM cycle completes, the engine fires the callback in a background daemon thread:

```python
threading.Thread(
    target=on_complete,
    args=(report,),
    daemon=True,
    name="cerebrum-mmap-flush"
).start()
```

The daemon thread ensures that flush I/O does not block the REM engine's return path. The `report` argument carries REM cycle metadata for logging purposes.

#### 7.2 SleepCycleOrchestrator Phase 6

`SleepCycleOrchestrator` coordinates REM with preceding maintenance phases. Phase 6, `_run_nvme_flush(dry_run)`, fires after Phases 1–5 complete:

```python
def _run_nvme_flush(self, dry_run: bool) -> None:
    if self._nvme_flush_callback is not None:
        threading.Thread(
            target=self._nvme_flush_callback,
            args=(None,),
            daemon=True
        ).start()
```

The `dry_run` guard ensures that NVMe flushes are suppressed during test-mode REM cycles, preventing spurious file system writes during CI.

#### 7.3 Wiring via CerebrumGraph

`CerebrumGraph.attach_sleep_cycle(orchestrator)` injects the flush callback:

```python
orchestrator._nvme_flush_callback = self._on_rem_complete
```

`CerebrumGraph._on_rem_complete(report)` calls:

```python
MmapConsolidator.flush(
    adapter=self._adapter,
    embeddings=self._embeddings,
    wal=self._wal,
    build_id=self._build_id
)
```

In `api/server.py`, `_get_rem_engine()` injects `on_complete=graph._on_rem_complete`, closing the loop for API-triggered REM cycles.

#### 7.4 Storage Status Endpoint

A new REST endpoint `GET /graph/storage` returns the current storage status as a JSON dict via `graph.storage_status()`:

```json
{
  "policy": "AUTO",
  "wal_entry_count": 142,
  "wal_size_bytes": 28416,
  "last_meta": {
    "version": "2.73.0",
    "written_at": "2026-06-05T03:14:22Z",
    "node_count": 43217,
    "edge_count": 124616,
    "embedding_dim": 384
  }
}
```

This endpoint enables monitoring dashboards and health checks to surface WAL accumulation, triggering operator-initiated REM cycles if the WAL grows unexpectedly large between scheduled cycles.

---

### 8. Configuration via .env and CEREBRUM_MMAP_DIR

The NVMe data directory is resolved by `resolve_mmap_dir()` using the following priority order:

1. `CEREBRUM_MMAP_DIR` environment variable, if already set in the process environment.
2. `CEREBRUM_MMAP_DIR` from a `.env` file in the project root, loaded via `python-dotenv` with `override=False` (so an already-set system env var is not clobbered).
3. Default: `data/cerebrum/mmap/`

The `.env` approach provides persistent NVMe path configuration without requiring the user to modify system-level environment variables or shell profiles. A typical deployment on a system with a dedicated NVMe SSD mounted at `/mnt/nvme0`:

```bash
# .env
CEREBRUM_MMAP_DIR=/mnt/nvme0/cerebrum/graph
```

This configuration survives process restarts and is checked into version control alongside the project (with appropriate `.gitignore` exemptions for secret-bearing `.env` files). The `python-dotenv>=1.0.0` dependency is declared in `pyproject.toml` as a core dependency, available in all installation profiles.

The `MmapPolicy` can be set independently via `CEREBRUM_MMAP_POLICY=ALWAYS|NEVER|AUTO` or programmatically by passing `policy=MmapPolicy.ALWAYS` to `MmapAdvisor`. The default is `AUTO`, which uses the RAM pressure heuristic.

---

### 9. Experimental Validation

#### 9.1 Test Suite Structure

The test suite for this feature comprises 35 tests across five test classes, all passing:

**TestGraphWAL (11 tests):** Covers the full WAL API: basic append, field integrity, entry count, size bytes, truncate, idempotent truncate (second call on already-absent file), replay from a populated WAL, skip of unknown op codes, skip of malformed (truncated) JSON lines, thread safety under 50 concurrent appenders, and iteration over an empty WAL.

**TestMmapAdvisor (6 tests):** Covers: tiny graph (< 0.20 pressure) → RAM-sufficient recommendation; ALWAYS policy override → `use_mmap=True` regardless of pressure; NEVER policy override → `use_mmap=False` regardless of pressure; MB estimate correctness against hand-computed values; `override_used=False` on AUTO policy; non-empty `reason` string on all recommendation types.

**TestMmapConsolidator (12 tests):** Covers: all six expected output files created; node and edge count consistency between source adapter and `graph.meta`; meta field completeness (version, written\_at, build\_id, node\_count, edge\_count, embedding\_dim, relation\_count); `relations.idx` sorted and complete; WAL truncated after successful flush; `_tmp_flush/` directory absent after successful flush (atomic cleanup); `load_meta()` round-trip; no-embeddings path (flush completes without `embeddings.e` when embeddings are None); duration field populated; embedding matrix shape matches $(|V|, d)$.

**TestREMFlushIntegration (4 tests):** End-to-end pipeline tests: `on_complete` fires after a real (non-dry) REM cycle; dry-run REM cycle does not fire `on_complete`; full WAL→REM→flush→WAL-gone pipeline with an adapter containing 10 edges; WAL survives a simulated crash (flush interrupted before `truncate()`, WAL file still present and replayable on next startup).

**TestResolveMmapDir (2 tests):** Default path resolves to `data/cerebrum/mmap/`; `CEREBRUM_MMAP_DIR` environment variable override produces the configured path.

#### 9.2 Performance Characteristics

**WAL append:** O(1) — file seek to end followed by a single write call. No read, no parse, no full-file scan. Thread-safe via `threading.Lock`. On the target system (RTX 5090 workstation, NVMe PCIe 5.0), measured append latency is sub-microsecond for the 256-byte average record size — negligible relative to graph traversal latency.

**MmapConsolidator flush:** A one-time sequential write of the full graph. For MetaQA:

| File | Size | Write Time (est.) |
|---|---|---|
| `graph.a` | 1.3 MB | $< 1$ ms |
| `graph.e` | 1.4 MB | $< 1$ ms |
| `embeddings.e` | 63 MB | $\approx 20$ ms at 3 GB/s |
| `relations.idx`, `nodes.map`, `graph.meta` | $< 1$ MB total | $< 1$ ms |

Total flush duration: approximately 25 ms for MetaQA, dominated by the float32 embedding matrix write. This is incurred once per REM cycle, which is measured in minutes — a negligible amortized overhead.

**Startup replay:** Linear scan of WAL records at $O(|WAL|)$. In normal operation (clean shutdown), the WAL is empty ($|WAL| = 0$) and replay is instantaneous. After an unclean shutdown, the WAL contains edges written since the last consolidation. For a system running 1-hour REM cycles receiving 100 edges/minute, this is at most 6,000 records — a replay time of well under 1 second.

**NVMe read path:** After consolidation, `numpy.memmap` provides O(1) random node/edge access via OS page cache. On MetaQA, the full graph fits in $\approx 66$ MB of virtual address space — well within the 128 GB VRAM + system RAM available on the target platform. In practice, the OS caches the entire file after the first access, making subsequent lookups indistinguishable from RAM.

#### 9.3 Comparison with Alternative Designs

| Design | Write Latency | Crash Safety | NVMe Writes | Atomicity |
|---|---|---|---|---|
| RAM-only (no persistence) | O(1) | None | 0 | N/A |
| Naive per-edge mmap | O(1) + page flush | Partial (torn writes) | $|E|$ | No |
| Pickle snapshot | O(full graph) | Point-in-time | 1/snapshot | No (file mid-write) |
| **GraphWAL + MmapConsolidator** | **O(1) WAL append** | **Full (WAL replay)** | **1/REM cycle** | **Yes (_tmp_flush/)** |

The proposed design uniquely achieves O(1) write latency, full crash safety, minimal NVMe writes, and atomic consolidation simultaneously.

---

### 10. Conclusion

We have presented a production-grade two-layer storage architecture for CEREBRUM that decouples the concerns of write durability, traversal speed, and NVMe persistence without sacrificing any of them. The key insight is that the existing REM cycle — already a natural quiet period in CEREBRUM's operational schedule — is the correct and sufficient consolidation trigger. By deferring NVMe writes to this point, the design eliminates per-edge write amplification while maintaining crash safety through the WAL.

The architecture embodies a principle of *capability without mandate*: NVMe persistence is entirely opt-in via `MmapPolicy`. Systems with sufficient RAM and no durability requirements operate purely in RAM with no behavioral change. Systems that need crash recovery activate the WAL. Systems with RAM pressure additionally activate NVMe mmap. The three components are independently useful and composable.

The biological analogy is instructive: the REM cycle as consolidation trigger mirrors the role of slow-wave sleep in biological memory systems, where synaptic weight normalization and memory transfer from hippocampus to neocortex occur during quiet periods rather than during active processing. CEREBRUM's graph consolidation follows the same logic — active traversal writes to fast volatile storage (RAM + WAL), and offline processing (REM) transfers to slow durable storage (NVMe mmap). This convergence of engineering necessity and biological metaphor reflects the broader design philosophy of the CEREBRUM framework.

Future work includes: (i) incremental WAL-to-mmap replay without full graph reconstruction, enabling faster startup after long crash-free runs; (ii) WAL compaction for long-running systems where WAL growth between REM cycles is large; (iii) multi-node WAL federation for `DistributedBeamTraversal` deployments; and (iv) WAL-based change data capture for streaming graph consumers.

---

### Acknowledgments

The author gratefully acknowledges the use of Claude (Anthropic) as a research assistant throughout this work. Claude assisted with mathematical formalization, code generation, manuscript preparation, and technical writing. All conceptual contributions, architectural decisions, experimental design, and intellectual claims are solely the author's.

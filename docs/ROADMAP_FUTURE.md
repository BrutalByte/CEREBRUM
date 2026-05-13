# ROADMAP_FUTURE.md
## Future Vision: Hybrid-Memory Architecture (Post-Phase 172)

This roadmap defines the transition from a purely RAM-resident reasoning engine to a **Hybrid-Memory Architecture**, allowing CEREBRUM to operate on graphs of arbitrary size while maintaining sub-millisecond reasoning performance for hot reasoning paths.

---

## 1. Hybrid-Memory Implementation Plan

### Phase 168: Engine Cap Resource Controller — COMPLETE (v2.52.0)
-   **Resource Monitoring**: Integrated `MemoryGovernor` into Hardware Abstraction Layer.
-   **CLI Limits**: Added `--max-ram-gb` and `--max-vram-gb` flags to all reasoning commands.
-   **OOM Prevention**: `ResourceGovernor` now enforces absolute RSS limits in real-time.

### Phase 169: Mmap-Backed Edge Layer & Auto-Spill — COMPLETE (v2.52.0)
-   **Binary Topology**: Implemented A-File (Adjacency) and E-Block (Edge) binary formats.
-   **Auto-Offload**: `CerebrumGraph` automatically serializes and spills to disk when RAM thresholds are breached.
-   **Zero-Copy Loading**: Memory-mapped binary blocks providing high-efficiency retrieval.

### Phase 170: Holographic Fragment Storage — COMPLETE (v2.52.0)
-   **Community Fragmentation**: Binary topology partitioned by community ID (`comm_N.a`).
-   **On-Demand Mounting**: Only active reasoning communities are mapped into process memory.
-   **Fragment Cache**: LRU-style management of open community handles.

### Phase 171: NVME-Optimized Vectorized Mmap Architecture — COMPLETE (v2.52.0)
-   **NumPy Zero-Copy**: Replaced `mmap.mmap` with structured `numpy.memmap` for zero-overhead array indexing.
-   **Parallel Fetch**: Implemented `get_neighbors_batch` interface for deep-queue NVME access.
-   **Memory Efficiency**: Reduced overhead of disk-resident reasoning from 200x to <10x.

### Phase 172: Vectorized Traversal Integration — COMPLETE (v2.52.0)
-   **Batch Reasoning**: Refactored `BeamTraversal` to expand the entire beam frontier in a single call.
-   **IO Parallelism**: Exploited NVME high-bandwidth read capabilities via concurrent neighbor block resolution.

---

## 2. Resource-Aware Reasoning (The Tuning Matrix)

| Mode | Memory Budget | Reasoning Strategy | Target Hardware |
|---|---|---|---|
| **PERFORMANCE** | Unlimited | All-RAM (Full Resident) | Enterprise Servers |
| **BALANCED** | Limited (Configurable) | Hybrid (Pinned Hot-Nodes + Mmap) | High-End Workstations |
| **ECONOMY** | Minimal | Mmap-First (Disk-Backed) | Laptops / Edge / Cloud-Free Tier |

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

# ROADMAP_FUTURE.md
## Future Vision: Hybrid-Memory Architecture & Autonomous Operation (Post-Phase 172)

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

## 2. Studio & Federated Reasoning (Phases 173–178)

### Phase 173: Bug Fixes & API Hardening — COMPLETE (v2.52.0)
-   **NameError Fix**: Resolved import-time `NameError` in traversal.
-   **Pydantic Schema Conflicts**: Resolved `PathNode`/`PathResult` schema conflicts in `api/schemas.py`.
-   **API Server Init**: Fixed `FastAPI` app initialization order.

### Phase 174: NVMe SSD Management UI — COMPLETE (v2.53.0)
-   **Hardware Refactor**: `core/hardware.py` streamlined; removed dead code, improved clarity.
-   **Settings Tab**: Studio gains a dedicated NVMe SSD management tab in `ui/studio.py`.
-   **Runtime Configuration**: Configure drive paths and spill thresholds from the UI without restarts.

### Phase 175: Studio Hot-Swap & Adaptive Control — COMPLETE (v2.53.0)
-   **Live Graph Hot-Swap**: Load a new graph into the running server without restarting.
-   **Adaptive Toggle**: Enable/disable H1SE, TAB, and STRB at runtime from the settings panel.

### Phase 176: FederatedGraphRegistry — COMPLETE (v2.53.0)
-   **Cross-Domain Registry**: `core/federated_registry.py` manages multiple independent graph backends.
-   **Alias Resolution**: `resolve_alias()` bridges entity IDs across domain boundaries during beam traversal.
-   **Batch Fallback**: `BeamTraversal` uses `get_neighbors_batch` when available; falls back to per-node `get_neighbors` on all other adapters, preserving backward compatibility.

### Phase 177: Continuous Improvement Trifecta — COMPLETE (v2.53.0)
-   **Autonomous Discovery**: Auto-traversal of federated graphs to surface missing links.
-   **Self-Correction**: `ProvenanceLedger`-backed rollback when traversal confidence drops below threshold.
-   **Evolutionary Tuning**: Adaptive CSA parameter backpropagation over session history.

### Phase 178: DON'T PANIC Emergency Snapshot — COMPLETE (v2.53.0)
-   **Atomic Snapshot**: `StudioEngine.emergency_snapshot()` persists all reasoning state to `panics/snapshot_<ts>/`.
-   **Recoverable State**: Saves Engram JSON, node/edge ID maps, community assignments, and active CSA parameters.
-   **Zero-Loss Recovery**: Enables post-mortem analysis and full state restoration after crashes.

---

## 3. Resource-Aware Reasoning (The Tuning Matrix)

| Mode | Memory Budget | Reasoning Strategy | Target Hardware |
|---|---|---|---|
| **PERFORMANCE** | Unlimited | All-RAM (Full Resident) | Enterprise Servers |
| **BALANCED** | Limited (Configurable) | Hybrid (Pinned Hot-Nodes + Mmap) | High-End Workstations |
| **ECONOMY** | Minimal | Mmap-First (Disk-Backed) | Laptops / Edge / Cloud-Free Tier |

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

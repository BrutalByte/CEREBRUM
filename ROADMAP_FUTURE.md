# ROADMAP_FUTURE.md
## Future Vision: Hybrid-Memory Architecture (Post-Phase 167)

This roadmap defines the transition from a purely RAM-resident reasoning engine to a **Hybrid-Memory Architecture**, allowing CEREBRUM to operate on graphs of arbitrary size while maintaining sub-millisecond reasoning performance for hot reasoning paths.

---

## 1. Hybrid-Memory Implementation Plan

### Phase 168: The 'Engine Cap' Resource Controller
- **Objective**: Implement global resource constraints for RAM and VRAM.
- **Mechanism**:
    - Add `MemoryGovernor` to `core/hardware.py`.
    - Allow users to set global limits in `config.yaml` or via CLI (`--max-ram-gb`, `--max-vram-gb`).
    - The Governor monitors current process utilization and prevents the `Adapter` from loading new graph components if the memory budget is exceeded.

### Phase 169: Mmap-Backed Edge Layer
- **Objective**: Offload raw edge storage to disk.
- **Mechanism**:
    - Implement `MmapAdapter` as a high-performance alternative to `NetworkXAdapter`.
    - Binary-serialize the graph topology into a flat-file structure optimized for page-aligned random access.
    - Transparently map this file into virtual memory, allowing the kernel to handle the RAM/NVMe paging.

### Phase 2: Intelligent Paging & Caching
### Phase 170: Hot-Community Pinning
- **Objective**: Pin high-authority/high-traversal communities to resident RAM.
- **Mechanism**: 
    - The `CommunityEngine` ranks communities by "authority score" (PageRank + query recency).
    - Blocks corresponding to top-performing communities are `mlock`'d or kept in a dedicated RAM-resident cache to ensure the "Attention Flashlight" never hits an I/O wait-state during a reasoning step.

### Phase 171: Transparent Paging Buffer
- **Objective**: Implement a user-configurable disk-spill buffer.
- **Mechanism**:
    - Users define a `DISK_SPILL_ENABLED: true` policy and a storage directory.
    - If the `MemoryGovernor` (Phase 168) detects memory pressure, it triggers the `Adapter` to evict the coldest graph segments (least frequently traversed edges) to the NVMe-backed buffer, replacing them with virtual address pointers.

---

## 2. Resource-Aware Reasoning (The Tuning Matrix)

| Mode | Memory Budget | Reasoning Strategy | Target Hardware |
|---|---|---|---|
| **PERFORMANCE** | Unlimited | All-RAM (Full Resident) | Enterprise Servers |
| **BALANCED** | Limited (Configurable) | Hybrid (Pinned Hot-Nodes + Mmap) | High-End Workstations |
| **ECONOMY** | Minimal | Mmap-First (Disk-Backed) | Laptops / Edge / Cloud-Free Tier |

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

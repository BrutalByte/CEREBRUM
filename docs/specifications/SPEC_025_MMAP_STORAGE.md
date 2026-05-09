# [Buchorn, 2026]: Memory-Mapped (Mmap) Graph Storage
## Scaling to Massive Knowledge Graphs via NVMe

**Status**: PROPOSED (Post-Phase 167)
**Author**: Bryan Alexander Buchorn  
**Field**: Storage / Performance / Memory-Management
**Target Modules**: `adapters/mmap_adapter.py`, `core/hardware.py`

---

### 1. Introduction
The Mmap Storage layer allows CEREBRUM to operate on Knowledge Graphs exceeding available RAM/VRAM by leveraging the OS virtual memory manager (VMM) to transparently swap graph segments between NVMe and memory.

### 2. Architectural Design
- **Adjacency File (A-File)**: Graph topology is stored in a structured, binary-mapped file. Each entry contains a fixed-size header (ID, degree) followed by offsets into a variable-length edge record block.
- **Edge Block (E-Block)**: Stores compact (source, target, relation_id, confidence) tuples.
- **Mmap-Layer**: The `MmapAdapter` uses `mmap.mmap()` to map the A-File and E-Block into the process’s virtual address space.

### 3. Reasoning Latency Management
To preserve reasoning performance:
- **Hot-Community Pinning**: The `CommunityEngine` pins frequently accessed community blocks into resident RAM using `mlock` (POSIX) or equivalent.
- **Page-Alignment**: Structs are aligned to 4KB page boundaries to minimize I/O page faults during traversal.

### 4. Implementation Constraints
- **Immutable Topology**: Mmap reasoning is optimized for read-heavy query/traversal workloads. In-place graph updates (e.g., REM synthesis) are buffered in a sidecar "delta-log" and merged periodically.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

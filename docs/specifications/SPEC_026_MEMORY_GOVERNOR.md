# [Buchorn, 2026]: The MemoryGovernor
## Resource-Aware Constraint Enforcement

**Status**: v2.51.0 (Phase 168 COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Field**: Performance / Hardware Management
**Module**: `core/hardware_governor.py`

---

### 1. Introduction
The **MemoryGovernor** provides a unified interface for enforcing hard RAM/VRAM usage limits on the CEREBRUM reasoning engine. It functions as the foundational gatekeeper for the Hybrid-Memory Architecture, allowing the system to signal when an Mmap spill-over or eviction is required.

### 2. Implementation
- **Resource Constraints**: Users define `max_ram_gb` and `max_vram_gb`.
- **Safety Buffer**: A configurable `safety_buffer_mb` prevents OOM crashes by forcing spills before reaching the absolute hardware limit.
- **Governor API**:
    - `get_stats()`: Returns real-time RAM/VRAM utilization.
    - `is_spill_needed()`: Predicate checked by the `GraphAdapter` and `TraversalEngine` before loading new graph segments.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

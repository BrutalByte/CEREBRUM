# [Buchorn, 2026]: Binary Graph Topology Format (v1.0)
## Structure of A-File and E-Block for Mmap Storage

**Status**: v2.51.0 (Phase 169 DRAFT)
**Author**: Bryan Alexander Buchorn  
**Field**: Storage / Performance
**Module**: `adapters/mmap_adapter.py`

---

### 1. Overview
The binary format optimizes for $O(1)$ random-access node lookups and contiguous edge-list scanning.

### 2. A-File (Adjacency Index)
Fixed-size header for each node, indexed by `node_id`.

| Field | Type | Size | Description |
|---|---|---|---|
| `id` | `uint64` | 8 bytes | Internal node ID |
| `degree` | `uint32` | 4 bytes | Total outgoing edges |
| `offset` | `uint64` | 8 bytes | Start pointer in E-Block |
| `community`| `uint16` | 2 bytes | Community membership ID |
| *Total* | | *22 bytes* | |

### 3. E-Block (Edge Records)
Variable-length block, accessed via `offset` from A-File.

| Field | Type | Size | Description |
|---|---|---|---|
| `target_id`| `uint64` | 8 bytes | Target internal node ID |
| `rel_id` | `uint16` | 2 bytes | Relation type lookup ID |
| `conf` | `float16`| 2 bytes | Edge confidence (0.0–1.0) |
| *Total* | | *12 bytes* | |

### 4. Page Alignment
Each node record in the A-File is 22 bytes. To optimize page alignment for 4KB OS pages, we pad each record to **32 bytes** (adding 10 bytes of reserved/metadata space). This ensures that exactly 128 nodes fit per 4KB memory page, minimizing page faults during neighbor list traversal.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

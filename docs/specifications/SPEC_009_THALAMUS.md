# [Buchorn, 2026]: THALAMUS
## Intelligent Ingestion Preprocessing and Normalization

**Status**: v2.51.0 (Phase 167 (Sleep-Phase Consolidation) COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Field**: Data Engineering / Ontology Mapping / Parallel Systems  
**Module**: `core/thalamus.py`

---

### 1. Introduction
Knowledge Graph ingestion typically suffers from "GIGO" (Garbage In, Garbage Out) problems: duplicate entity IDs, inconsistent relation labels, and "identity collapse" where unrelated domains share identical strings. **THALAMUS** is a composable preprocessing pipeline that acts as an "intelligent filter," normalizing data *at ingest time* to ensure graph integrity and reasoning precision.

### 2. Core Functional Layers

#### 2.1 Entity Normalization & Deduplication
THALAMUS utilizes a bidirectional deduplication map to resolve aliases to canonical IDs. 
-   **Exact Match**: Direct string lookup.
-   **Fuzzy Match**: Optional n-gram or Levenshtein distance for near-identical labels.
-   **Ontology Mapping**: Translates raw labels (e.g., "was_born_in") to canonical relations (e.g., "BORN_IN").

#### 2.2 Namespace Isolation (Hole Fix 1.1.0)
To prevent "Identity Collapse" across modalities, THALAMUS enforces strict namespace prefixing:
-   `text:EntityName` for textual data.
-   `signal:SensorID` for telemetry data.
-   `remote:NodeID` for federated entities.
This ensures that a reasoning path cannot jump from a "Pressure" sensor directly to a "Pressure" project document unless an explicit cross-modal edge exists.

#### 2.3 Provenance Tracking
Every triple processed by THALAMUS is decorated with a `provenance_id` and a `trust_score`. This enables the **CSA Engine** [Buchorn, 2026] to weight edges based on the reliability of the source.

### 3. Enterprise Optimization: Parallel Ingestion (Hole Fix 1.2.0)

In high-velocity environments, normalization is a CPU-bound bottleneck. THALAMUS implements **Un-locked Preprocessing**:
1.  **Stage 1 (Parallel)**: Batches of raw triples are distributed across worker threads. Each worker performs normalization, namespace isolation, and metadata enrichment *outside* the graph mutex.
2.  **Stage 2 (Atomic)**: The fully normalized and enriched triples are committed to the `GraphAdapter` in a single atomic lock acquisition.

This refactoring removes THALAMUS as a serial bottleneck, allowing ingestion rates to scale linearly with available CPU cores.

### 4. Implementation Notes
*   **Pipeline Composition**: Supports custom `normalization_fn` and `provenance_fn` hooks.
*   **Memory Efficiency**: Deduplication maps use an LRU (Least Recently Used) cache to prevent memory bloat on infinite data streams.
*   **Integration**: THALAMUS is the primary gateway for the `StreamAdapter` and `CSVAdapter`, ensuring all data entering the CEREBRUM core is pre-aligned.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.51.0

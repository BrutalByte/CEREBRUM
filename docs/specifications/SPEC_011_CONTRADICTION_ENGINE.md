# SPEC_011: Contradiction Materialization Engine
## Reasoning about Factual Conflict and Uncertainty

**Status**: v2.24.0 (Phase 111 (Active Inference) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Formal Logic / Knowledge Integrity / Evidence-Based Reasoning  
**Module**: `core/contradiction_engine.py`

---

### 1. Introduction
Knowledge Graphs often treat conflicting data as an error to be resolved (De-duplication) or suppressed (Majority Vote). However, in research, intelligence, and scientific domains, contradictions are high-value signals indicating emerging discoveries, unsettled debates, or data-source unreliability. The **Contradiction Materialization Engine** treats conflict as a first-class citizen, identifying logical and structural inconsistencies and materializing them as queryable `CONTRADICTS` edges.

### 2. Typology of Contradictions

The engine identifies five distinct classes of factual conflict:

#### 2.1 Direct Predicate Conflict
Occurs when two triples $(s, r, o_1)$ and $(s, r, o_2)$ exist where $r$ is a functional relation (e.g., `BORN_IN`) and $o_1 \neq o_2$.

#### 2.2 Temporal Anachronism
Occurs when the temporal metadata of a triple violates physical or logical sequence (e.g., a "Died" event occurring before a "Born" event).

#### 2.3 Structural Impossibility
Identifies violations of graph-theoretic constraints, such as a node having a "Parent" relation to its own "Grandparent" (Cycle violation in hierarchies).

#### 2.4 Provenance Dissonance
Occurs when two high-trust sources provide diametrically opposed facts about the same entity relationship.

### 3. Materialization & Delta-Authority

Instead of deleting the "weaker" fact, the engine expands the graph:
1.  **Linkage**: A `CONTRADICTS` edge is created between the conflicting target nodes $o_1$ and $o_2$.
2.  **Delta-Authority**: The edge is decorated with an `authority_delta` score:
    $$\Delta A = |T(source_1) - T(source_2)|$$
    where $T$ is the trust score of the provenance.
3.  **Conflict Context**: The edge stores the `relation_type` and `reasoning_path` that led to the detection.

### 4. Reasoning through Conflict

The **CSA Engine** (SPEC_002) recognizes `CONTRADICTS` edges as "High-Surprise" signals. 
-   **Exploratory Mode**: The reasoning beam can follow a contradiction link to explore alternative hypotheses.
-   **Skeptical Mode**: The presence of a contradiction link reduces the attention weight of both associated facts, forcing the beam to seek a third "triangulating" source.

### 5. Implementation Notes
*   **Background Scanning**: Contradiction detection is scheduled during the **REM Cycle** (SPEC_007) to prevent overhead during active ingestion.
*   **Resolution Hooks**: Provides an API for human-in-the-loop (HITL) resolution, where a human expert can "Verify" one side of a conflict, causing the engine to prune the `CONTRADICTS` edge and its opposing fact.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

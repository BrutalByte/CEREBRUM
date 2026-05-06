# SPEC_024: STRB (Semantic Terminal Relation Boost)
## Zero-Config Intent Matching via Query Embeddings

**Status**: v2.51.0 (Phase 167 COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)
**Field**: NLP / Reasoning / Zero-Shot
**Modules**: `core/structural_relation_inferrer.py`, `core/cerebrum.py`

---

### 1. Introduction
**STRB** (Phase 167) identifies the intended terminal relation of a query using semantic similarity, eliminating the need for manual configuration.

### 2. Methodology
- **Embedding Match**: Encodes query text and compares it to labels of every relation in the graph.
- **Auto-Boost**: Automatically prioritizes answer-type edges (e.g., "treats" for "What compound treats X?").
- **Benefit**: Provides high-accuracy reasoning on typed graphs with zero manual setup.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

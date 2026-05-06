# SPEC_021: H1SE (Hop-1 Intermediate Seed Expansion)
## Solving Hub Crowding in Large-Scale Knowledge Graphs

**Status**: v2.51.0 (Phase 167 COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)
**Field**: Beam Search / Topology / Scalability
**Modules**: `reasoning/traversal.py`

---

### 1. Introduction
**H1SE** (Phase 137) solves the problem of "popular" nodes (hubs) capturing the entire beam width at hop 1, preventing the exploration of more specific but less central reasoning branches.

### 2. Methodology
- **Independent Beams**: Every unique node reached at hop 1 receives its own dedicated search budget (mini-beam).
- **Expansion K**: The number of hop-1 nodes to expand independently.
- **Aggregation**: Results from all independent traversals are merged and ranked at the end.
- **Benefit**: Significantly improves recall in graphs with extreme degree skew (e.g., MovieLens, MetaQA).

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

# Plan: cerebrum-3hop-sota

Objective: Achieve SOTA for 3-hop Hits@1 on MetaQA (Target: >0.228 to beat GraftNet; >0.298 for EmbedKGQA).

## Status
- **Current Hits@1**: 0.136 (Phase 147 baseline)
- **Current Hits@10**: 0.526
- **Status**: [ACTIVE]

## Tasks
- [ ] **Task 1: Bottleneck Audit** (Researcher)
  - Analyze why the 3-hop ranking is failing despite high Hits@10.
  - Check if the correct entities are being pruned at Hop 2.
  - Verify if TRB (Terminal Relation Boost) is correctly identifying the 3rd hop relation.
- [ ] **Task 2: Evidence Accumulation Refinement** (Reviewer/Researcher)
  - Evaluate "Path Multiplicity" sum vs max.
  - Investigate if "Shared Neighbor Bonus" from Phase 140 is firing correctly for 3-hop intersections.
- [ ] **Task 3: MetaQA-Specific Structural Exploitation** (Writer)
  - Propose "Strict Bipartite Mode" (High-Penultimate Boost).
- [ ] **Task 4: Verification & Benchmarking** (Verifier)
  - Run full-hop benchmarks.
  - Generate provenance sidecar.

## Verification State
- `unverified`: Intermediate ranking bottleneck location.
- `unverified`: Effect of Path Multiplicity Sum.

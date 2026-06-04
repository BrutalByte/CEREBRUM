# ROADMAP_FUTURE.md
## Post-Phase 223 Research Directions

**Current state**: v2.73.0, Phase 223 COMPLETE — 2,269 tests passing, 4 skipped.
The self-awareness loop is fully closed: CEREBRUM can now detect uncertainty, attempt recovery, calibrate its own confidence, punish overconfident parameters, and adapt its curiosity level — all without external feedback.

---

## 1. Known Open Problems

### 1.1 Cross-Type Semantic Ceiling
**Root cause**: Sentence-transformer embeddings hurt cross-type 3-hop reasoning (−30 pp H@1 on MetaQA 3-hop) while helping typed 2-hop queries (+10 pp). The beam confuses semantic similarity of query text with entity type compatibility.

**Direction**: A type-aware embedding layer that decouples entity-type compatibility from semantic similarity. Possible approaches: type-conditioned attention masking in STRB, explicit entity-type priors in `CSAEngine`, or a lightweight type-classification head trained from graph schema only (no labeled QA pairs — maintaining the training-free principle).

### 1.2 Ranking Gap vs. Supervised Baselines
CEREBRUM 3-hop H@1 = 60.2% (zero training) vs. NSM 98.0% (supervised). The gap is predominantly a **ranking problem**, not a coverage problem — H@10 = 89.4% means the correct answer is in the beam. Improving H@1 without training labels requires better final-hop disambiguation.

**Direction**: Contrastive path reranking using only graph structure (degree, community membership, relation path frequency) rather than learned weights. `CounterfactualReranker` (Phase 126) already exists; deeper integration at the final-hop level is the next step.

### 1.3 WebQSP Performance
CEREBRUM has not been formally benchmarked against WebQSP (Freebase-backed, CVT mediator nodes, free-text questions). The `CVT passthrough` option exists but has not been calibrated.

**Direction**: Full WebQSP evaluation harness with CVT flattening; compare against EPERM (88.8% supervised) baseline.

---

## 2. Architecture Extensions

### 2.1 Persistent Calibration History
`PlattCalibration` (Phase 222) now tracks ECE and drift. A natural extension is logging ECE history across sessions to visualize calibration stability over the lifetime of a deployed graph.

### 2.2 Cross-KB EngramTransfer (Phase 219) — Full Validation
`EngramTransferRegistry` re-encodes patterns across KB vocabulary changes. A rigorous benchmark measuring transfer efficiency (how many queries benefit, how much accuracy is preserved) has not been done.

### 2.3 GPU-Accelerated Community Detection
`CommunityEngine` runs DSCF/TSC on CPU. For graphs >5M edges, GPU-resident spectral methods (cuGraph) would reduce build time from minutes to seconds.

---

## 3. Publication Pipeline

Six arXiv deliverables are in progress (see `CEREBRUM_PUBLICATION_GAMEPLAN.md`). Priority order:

1. **Technical Report** (`research/papers/00-technical-report/`) — full system spec, v2.73.0 draft ready
2. **Flagship Paper** (`research/papers/01-flagship/`) — training-free KGQA, v2.73.0 draft ready
3. **SDRB/PI Paper** (`research/papers/06-sdrb/`) — Semantic Terminal Relation Boost, v2.73.0 draft ready
4. Papers A–D (community detection, graph plasticity, federated, production) — planned

---

## 4. Resource-Aware Deployment Modes

| Mode | Memory Budget | Reasoning Strategy | Target Hardware |
|---|---|---|---|
| **PERFORMANCE** | Unlimited | All-RAM (Full Resident) | Enterprise Servers |
| **BALANCED** | Limited (Configurable) | Hybrid (Pinned Hot-Nodes + Mmap) | High-End Workstations |
| **ECONOMY** | Minimal | Mmap-First (Disk-Backed) | Laptops / Edge / Cloud-Free Tier |

Hybrid-memory architecture (Phases 168-172) is complete and production-ready. Mode selection is via `--max-ram-gb` / `--max-vram-gb` CLI flags.

---

**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**
**Last updated**: June 4, 2026 for v2.73.0

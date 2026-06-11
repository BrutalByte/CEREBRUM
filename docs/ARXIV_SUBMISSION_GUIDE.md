# arXiv Submission Guide

**CEREBRUM — 40 Papers (PAPER_001 flagship + PAPER_002–037 complete · PAPER_038–039 staged)**

This guide covers the submission process, per-paper metadata, and checklist for submitting all CEREBRUM papers to arXiv.

---

## arXiv Categories

All papers should be submitted to **primary category `cs.IR`** (Information Retrieval) with the following cross-list categories as appropriate:

| Primary | Cross-list | Use for |
|---|---|---|
| `cs.IR` | `cs.AI`, `cs.LG` | All papers (default) |
| `cs.AI` | `cs.IR`, `cs.NE` | Papers 3, 4, 7 (neuromorphic / biological analog) |
| `cs.LG` | `cs.IR`, `cs.AI` | Papers 6, 15 (learning algorithms) |
| `cs.DC` | `cs.IR`, `cs.AI` | Papers 5, 13 (distributed / streaming) |

---

## Submission Order (Recommended)

Submit foundational papers first so cross-references resolve correctly.

| Order | Paper | File | Category |
|---|---|---|---|
| 0 | **CEREBRUM: Training-Free Multi-Hop KGQA (flagship)** | `PAPER_001_CEREBRUM_TRAINING_FREE_KGQA.md` | `cs.IR` + `cs.AI` + `cs.LG` |
| 1 | DSCF/TSC Community Fusion | `PAPER_001_DSCF_TSC.md` | `cs.IR` + `cs.AI` |
| 2 | Community-Structured Attention | `PAPER_002_CSA.md` | `cs.IR` + `cs.LG` |
| 3 | Bridge Twin Nodes | `PAPER_003_BRIDGE_TWINS.md` | `cs.AI` + `cs.NE` |
| 4 | STDP Causal Edge Inference | `PAPER_004_STDP_CAUSAL.md` | `cs.AI` + `cs.NE` |
| 5 | Holographic Federated Index | `PAPER_005_HOLOGRAPHIC_INDEXING.md` | `cs.DC` + `cs.IR` |
| 6 | Bayesian Beam Search | `PAPER_006_BAYESIAN_BEAM.md` | `cs.LG` + `cs.IR` |
| 7 | REM Cycle Maintenance | `PAPER_007_REM_CYCLE.md` | `cs.AI` + `cs.IR` |
| 8 | Cross-Modal Signal Alignment | `PAPER_008_SIGNAL_ENCODER.md` | `cs.IR` + `cs.LG` |
| 9 | THALAMUS Ingestion Pipeline | `PAPER_009_THALAMUS.md` | `cs.IR` + `cs.AI` |
| 10 | Inference Validation | `PAPER_010_INFERENCE_VALIDATION.md` | `cs.IR` + `cs.AI` |
| 11 | Contradiction Materialization | `PAPER_011_CONTRADICTION.md` | `cs.IR` + `cs.AI` |
| 12 | Glass-Box Reasoning Studio | `PAPER_012_REASONING_STUDIO.md` | `cs.HC` + `cs.IR` |
| 13 | Streaming Engine | `PAPER_013_STREAMING_ENGINE.md` | `cs.DC` + `cs.IR` |
| 14 | Metacognitive Verification | `PAPER_014_INSIGHT_ENGINE.md` | `cs.AI` + `cs.IR` |
| 15 | Algorithmic Depth | `PAPER_015_ALGORITHMIC_DEPTH.md` | `cs.LG` + `cs.IR` |
| 16 | Production Hardening | `PAPER_016_PRODUCTION_HARDENING.md` | `cs.SE` + `cs.IR` |
| 17 | GraphSAGE Neighbourhood Smoothing | `PAPER_017_GRAPHSAGE_SMOOTHING.md` | `cs.LG` + `cs.IR` |
| 18 | Engram-Steered Traversal | `PAPER_018_ENGRAM_TRAVERSAL.md` | `cs.IR` + `cs.AI` |
| 19 | Temporal Calibration | `PAPER_019_TEMPORAL_CALIBRATOR.md` | `cs.LG` + `cs.IR` |
| 20 | Five Fault-Tolerance Patterns for Production KG Servers | `PAPER_020_FAULT_TOLERANCE.md` | `cs.SE` + `cs.IR` |
| 21 | SpeedTalk Phonemic Compression | `PAPER_021_SPEEDTALK_COMPRESSION.md` | `cs.IR` + `cs.LG` |
| 22 | Looped Beam Traversal | `PAPER_022_LOOPED_TRAVERSAL.md` | `cs.IR` + `cs.LG` |
| 23 | PredictiveCodingEngine | `PAPER_023_PREDICTIVE_CODING.md` | `cs.AI` + `cs.NE` |
| 24 | AutoApprover | `PAPER_024_AUTOAPPROVER.md` | `cs.AI` + `cs.IR` |
| 25 | TriangulationEngine | `PAPER_025_TRIANGULATION.md` | `cs.AI` + `cs.IR` |
| 26 | DiscoveryCalibrator + ContradictionResolver | `PAPER_026_DISCOVERY_CALIBRATION.md` | `cs.AI` + `cs.IR` |
| 27 | AutonomousDiscoveryLoop | `PAPER_027_AUTONOMOUS_LOOP.md` | `cs.AI` + `cs.IR` |
| 28 | Studio v2 Dashboard + Provenance Panel | `PAPER_028_STUDIO_V2.md` | `cs.HC` + `cs.IR` |
| 29 | ProvenanceLedger | `PAPER_029_PROVENANCE_LEDGER.md` | `cs.SE` + `cs.IR` |
| 30 | Feature Impact Benchmark | `PAPER_030_FEATURE_IMPACT.md` | `cs.IR` + `cs.LG` |
| 31 | Loop-Provenance Recovery | `PAPER_031_LOOP_PROVENANCE_RECOVERY.md` | `cs.SE` + `cs.AI` |
| 32 | GraphAdapter remove_edge Protocol | `PAPER_032_GRAPH_ADAPTER_PROTOCOL.md` | `cs.SE` + `cs.IR` |
| 33 | GraphSnapshot Persistence | `PAPER_033_GRAPH_SNAPSHOT.md` | `cs.SE` + `cs.IR` |
| 34 | Adaptive Loop Tuning | `PAPER_034_ADAPTIVE_LOOP_TUNING.md` | `cs.AI` + `cs.IR` |
| 35 | UE5 Neural Visualization Bridge | `PAPER_035_UE5_NEURAL_VISUALIZATION.md` | `cs.HC` + `cs.IR` |
| 36 | The Cingulate Engine: Meta-Cognitive Error Correction in Graph Attention | `PAPER_036_CINGULATE_ENGINE.md` | `cs.AI` + `cs.IR` |
| 37 | GraphProfiler and STRB: Autonomous Strategy Selection for Zero-Shot KG Reasoning | `PAPER_037_GRAPHPROFILER_STRB.md` | `cs.IR` + `cs.AI` |
| 38 | Hop-Scaled Semantic Attention: Suppressing Cross-Hop Semantic Bias in Multi-Hop KGQA | `PAPER_038_HOP_SCALED_SEMANTIC.md` | `cs.IR` + `cs.LG` |
| 39 | Intentional NVMe Graph Consolidation with Write-Ahead Logging for Knowledge Graph Systems | `PAPER_039_NVME_WAL.md` | `cs.DB` + `cs.IR` |

---

## Per-Paper Metadata Template

Each submission requires the following metadata fields on arXiv:

```
Title:        [From paper header]
Authors:      Bryan Alexander Buchorn
Abstract:     [From paper Abstract section — max 1,920 characters]
Comments:     16 pages. Part of the CEREBRUM framework series (v2.86.0).
              Code: https://github.com/BrutalByte/CEREBRUM
MSC-class:    68T30 (Knowledge representation)
ACM-class:    I.2.4; H.3.3
License:      CC BY 4.0 (arXiv default for non-commercial research)
```

---

## Formatting Requirements

arXiv accepts `.tex` (preferred) and `.pdf`. Our papers are in Markdown; conversion is required.

### Markdown → LaTeX conversion

```bash
pip install pandoc
pandoc docs/arxiv/PAPER_001_DSCF_TSC.md -o arxiv_submission/paper_001.tex \
    --template arxiv_template.tex \
    --bibliography refs.bib
```

A LaTeX template (`arxiv_template.tex`) following arXiv's `article` document class should be prepared once and reused for all papers.

### Required LaTeX elements
- `\usepackage{amsmath}` — for displayed equations
- `\usepackage{booktabs}` — for benchmark tables
- `\usepackage{hyperref}` — for cross-references and URLs
- All figures as `.pdf` or `.png` at 300 DPI minimum

### Figure requirements
Each paper should include at minimum:
1. An architecture diagram (flowchart of the algorithm)
2. A results table (benchmark comparison)

Figures should be prepared in a vector format (PDF) for crisp rendering.

---

## Abstract Length Check

arXiv abstracts are capped at 1,920 characters. Check each abstract:

```bash
for paper in docs/arxiv/PAPER_*.md; do
    abstract=$(awk '/^### Abstract/{found=1; next} found && /^###/{exit} found{print}' "$paper")
    count=$(echo "$abstract" | wc -c)
    echo "$paper: $count chars"
done
```

---

## Pre-Submission Checklist (per paper)

- [ ] Abstract ≤ 1,920 characters
- [ ] All equations render correctly in LaTeX
- [ ] All tables have `\toprule/\midrule/\bottomrule` (booktabs style)
- [ ] All citations in `refs.bib` with correct BibTeX keys
- [ ] Author affiliation: "Independent Researcher, Las Vegas, NV, USA"
- [ ] Prior art section explicitly differentiates from closest prior work
- [ ] Benchmark numbers match `docs/BENCHMARK_CANONICAL.md`
- [ ] GitHub/code link included in Comments field
- [ ] License selected: CC BY 4.0
- [ ] Cross-list categories correct for paper topic

---

## Citation Cross-Referencing

Papers cite each other. Use consistent arXiv IDs once papers 1–4 are submitted. Placeholder format until IDs are assigned:

```bibtex
@article{buchorn2026dscf,
  title   = {Dual-Signal Community Fusion for Knowledge Graph Attention},
  author  = {Buchorn, Bryan Alexander},
  journal = {arXiv preprint arXiv:2026.XXXXX},
  year    = {2026}
}
```

---

## Comprehensive Paper (CEREBRUM_White_Paper_arXiv.md)

The comprehensive umbrella paper (`docs/CEREBRUM_White_Paper_arXiv.md`) covers all subsystems in one document and should be submitted to `cs.IR` as the primary reference after all individual papers are live. Title it:

> **CEREBRUM: Community-Structured Graph Attention for Zero-Shot Multi-Hop Knowledge Graph Reasoning**

This paper should cite all subsystem papers (PAPER_001 through PAPER_039) by their arXiv IDs. Submit after PAPER_039 is live.

`PAPER_100_CONCLUSION.md` (`docs/archive/pre-arXiv/arxiv/PAPER_100_CONCLUSION.md`) is the series conclusion — submit last, after the umbrella paper is live and all arXiv IDs are assigned.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: June 5, 2026 for version v2.73.0 (Phase 227)

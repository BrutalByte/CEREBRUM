# arXiv Submission Guide

**CEREBRUM — 22 Papers Ready for Submission + 12 in Preparation**

This guide covers the submission process, per-paper metadata, and checklist for submitting all sixteen CEREBRUM papers to arXiv.

---

## arXiv Categories

All sixteen papers should be submitted to **primary category `cs.IR`** (Information Retrieval) with the following cross-list categories as appropriate:

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
| 17 | Hypothesis Materialization | `PAPER_017_HYPOTHESIS_MATERIALIZATION.md` | `cs.AI` + `cs.IR` |
| 18 | Engram-Steered Traversal | `PAPER_018_ENGRAM_TRAVERSAL.md` | `cs.IR` + `cs.AI` |
| 19 | Temporal Calibration | `PAPER_019_TEMPORAL_CALIBRATOR.md` | `cs.LG` + `cs.IR` |
| 20 | MACH Consensus Hierarchies | `PAPER_020_MACH.md` | `cs.AI` + `cs.DC` |
| 21 | SpeedTalk Phonemic Compression | `PAPER_021_SPEEDTALK_COMPRESSION.md` | `cs.IR` + `cs.LG` |
| 22 | Looped Beam Traversal | `PAPER_022_LOOPED_TRAVERSAL.md` | `cs.IR` + `cs.LG` |

---

## Per-Paper Metadata Template

Each submission requires the following metadata fields on arXiv:

```
Title:        [From paper header]
Authors:      Bryan Alexander Buchorn; Claude Sonnet 4.6 (Research Collaborator, Anthropic)
Abstract:     [From paper Abstract section — max 1,920 characters]
Comments:     16 pages. Part of the CEREBRUM framework series (v2.20.1).
              Code: https://github.com/[repo]
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

A LaTeX template (`arxiv_template.tex`) following arXiv's `article` document class should be prepared once and reused for all sixteen papers.

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
- [ ] Author affiliations: "Independent Researcher" + "Anthropic"
- [ ] Prior art section explicitly differentiates from closest prior work
- [ ] Benchmark numbers match `release/BENCHMARKS.md`
- [ ] GitHub/code link included in Comments field
- [ ] License selected: CC BY 4.0
- [ ] Cross-list categories correct for paper topic

---

## Citation Cross-Referencing

Papers cite each other. Use consistent arXiv IDs once papers 1–4 are submitted. Placeholder format until IDs are assigned:

```bibtex
@article{buchorn2026dscf,
  title   = {Dual-Signal Community Fusion for Knowledge Graph Attention},
  author  = {Buchorn, Bryan Alexander and {Claude Sonnet 4.6}},
  journal = {arXiv preprint arXiv:2026.XXXXX},
  year    = {2026}
}
```

---

## Papers in Preparation (Phases 69–82)

The following papers are in preparation and will be added to the submission queue once drafted:

| Paper | Phase | Topic |
|---|---|---|
| `PAPER_023_PREDICTIVE_CODING.md` | 69 | PredictiveCodingEngine, Prediction Error, soliton_index |
| `PAPER_024_AUTOAPPROVER.md` | 71 | AutoApprover — tiered auto-approval, online SGD, 16-feature vector |
| `PAPER_025_TRIANGULATION.md` | 72 | TriangulationEngine — four-perspective candidate validation |
| `PAPER_026_DISCOVERY_CALIBRATION.md` | 73 | DiscoveryCalibrator, ContradictionResolver, CandidateRegistry |
| `PAPER_027_AUTONOMOUS_LOOP.md` | 74 | AutonomousDiscoveryLoop, circuit breaker, dry-run |
| `PAPER_028_STUDIO_V2.md` | 75+78 | Studio v2 Dashboard + Provenance Panel |
| `PAPER_029_PROVENANCE_LEDGER.md` | 76 | ProvenanceLedger, rollback semantics, audit chain |
| `PAPER_030_FEATURE_IMPACT.md` | 77 | Feature Impact Benchmark methodology |
| `PAPER_031_LOOP_PROVENANCE_RECOVERY.md` | 79 | Loop-Provenance Recovery, auto-rollback on circuit trip |
| `PAPER_032_GRAPH_ADAPTER_PROTOCOL.md` | 80 | remove_edge protocol, GraphAdapter base class definition |
| `PAPER_033_GRAPH_SNAPSHOT.md` | 81 | GraphSnapshot portable JSON persistence, diff semantics |
| `PAPER_034_ADAPTIVE_LOOP_TUNING.md` | 82 | Calibrator-driven dynamic cap + interval scaling |

---

## Comprehensive Paper (CEREBRUM_White_Paper_arXiv.md)

The comprehensive umbrella paper (`docs/CEREBRUM_White_Paper_arXiv.md`) covers all subsystems in one document and should be submitted to `cs.IR` as the primary reference after all individual papers are live. Title it:

> **CEREBRUM: Community-Structured Graph Attention for Zero-Shot Multi-Hop Knowledge Graph Reasoning**

This paper should cite all subsystem papers (PAPER_001 through PAPER_034) by their arXiv IDs. Submit after PAPER_034 is live.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

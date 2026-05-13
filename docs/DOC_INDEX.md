# CEREBRUM Documentation Index
## v2.52.0 (Phase 172) — May 2026

Single-page navigation for all CEREBRUM documentation. Start here.

---

## Canonical Academic Source

| File | Purpose |
|------|---------|
| [`CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md`](CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md) | **Primary source** — 37-paper complete technical specification. All publication content derives from this file. |
| [`BENCHMARK_CANONICAL.md`](BENCHMARK_CANONICAL.md) | **Locked benchmark reference** — canonical numbers for all publications. Do not use numbers from anywhere else. |
| [`NOVEL_CONTRIBUTIONS.md`](NOVEL_CONTRIBUTIONS.md) | Summary of original contributions per phase — useful for scoping each paper's claims. |
| [`GLOSSARY.md`](GLOSSARY.md) | Term definitions and notation. Resolves the η conflict (η = CSA temporal decay; η_T = TSC temperature decay). |

---

## Publication Deliverables (In Progress)

Target: 6 arXiv submissions per `CEREBRUM_PUBLICATION_GAMEPLAN.md`. Submission order: Technical Report → Flagship → Papers A–D.

| Paper | Directory | Status | arXiv Category |
|-------|-----------|--------|---------------|
| Technical Report (full spec) | `research/papers/00-technical-report/` | Planned — Phase 4 | cs.AI |
| Flagship: Training-Free KGQA | `research/papers/01-flagship/` | Planned — Phase 5 | cs.AI |
| Paper A: TSC Community Detection | `research/papers/02-community-detection/` | Planned — Phase 5 | cs.SI / cs.LG |
| Paper B: Graph Plasticity (Bridge + STDP) | `research/papers/03-graph-plasticity/` | Planned — Phase 5 | cs.AI |
| Paper C: Federated Holographic Indexing | `research/papers/04-federated/` | Planned — Phase 5 | cs.DC |
| Paper D: Production Engineering | `research/papers/05-production/` | Planned — Phase 5 | cs.SE / cs.DB |

See [`CEREBRUM_PUBLICATION_GAMEPLAN.md`](CEREBRUM_PUBLICATION_GAMEPLAN.md) for full submission strategy.  
See [`ARXIV_SUBMISSION_GUIDE.md`](ARXIV_SUBMISSION_GUIDE.md) for submission mechanics.

---

## Draft Paper Files (docs/arxiv/)

38 focused papers mapping to master manuscript sections. All authorship-corrected, acknowledgments added.

| Range | Content |
|-------|---------|
| [`arxiv/PAPER_001–010`](arxiv/) | DSCF/TSC, CSA, Bridge Twins, STDP, Holographic Indexing, Bayesian Beam, REM, Signal Encoder, THALAMUS, Inference Validation |
| [`arxiv/PAPER_011–020`](arxiv/) | Contradiction Resolver, Glass-Box Studio, Streaming Engine, Metacognitive Verification, Algorithmic Depth, Structural Holes, GraphSAGE, Engram, Temporal Calibrator, Fault Tolerance |
| [`arxiv/PAPER_021–030`](arxiv/) | SpeedTalk, Looped Beam, Predictive Coding, AutoApprover, Triangulation, Discovery Calibration, Autonomous Loop, Studio v2, Provenance Ledger, Feature Impact |
| [`arxiv/PAPER_031–037`](arxiv/) | Loop-Provenance Recovery, Graph Adapter Protocol, Graph Snapshot, Adaptive Loop Tuning, UE5 Visualization, Cingulate Engine, GraphProfiler/STRB |
| [`arxiv/PAPER_100`](arxiv/PAPER_100_BENCHMARK_COMPARISON.md) | Benchmark Comparison (MetaQA, WebQSP, Hetionet, IKGWQ) |

---

## Technical Specifications (docs/specifications/)

27 detailed specs (`SPEC_001` – `SPEC_027`) mapping to each subsystem. Internal reference only — not for direct arXiv submission.

---

## Developer Documentation

| File | Purpose |
|------|---------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System architecture, module map, Transformer↔KG analogy, data flow |
| [`API_REFERENCE.md`](API_REFERENCE.md) | All REST endpoints with request/response schemas |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Production deployment, `--kge-model` flag, sentence-transformers requirement |
| [`TESTING.md`](TESTING.md) | Test suite, fixture conventions, 2177 passed / 1 skipped (v2.52.0) |
| [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) | How to integrate CEREBRUM with external systems |
| [`MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) | Version migration notes |
| [`PERFORMANCE_TUNING.md`](PERFORMANCE_TUNING.md) | Latency, beam width, GraphSAGE tradeoffs |
| [`REASONING_STUDIO_GUIDE.md`](REASONING_STUDIO_GUIDE.md) | Glass-Box Studio UI usage |
| [`CLAUDE.md`](CLAUDE.md) | Agent directives and code quality rules for this repo |

---

## Supporting Research References

| File | Purpose |
|------|---------|
| [`CEREBRUM_BENCHMARK_COMPARISON_PAPER.md`](CEREBRUM_BENCHMARK_COMPARISON_PAPER.md) | Full benchmark analysis vs SOTA — detailed version of the comparison table in BENCHMARK_CANONICAL.md |
| [`BENCHMARK_ANALYSIS_V2.51.0.md`](BENCHMARK_ANALYSIS_V2.51.0.md) | Phase 172 benchmark run analysis and ablation notes |
| [`ROADMAP_FUTURE.md`](ROADMAP_FUTURE.md) | Post-v2.51 research directions |
| [`whitepapers/`](whitepapers/) | Topic-specific technical whitepapers |
| [`docs/Research/`](Research/) | Supporting research notes and phase-by-phase findings |

---

## Archive

Superseded document versions — preserved for reference, not for new citations.

| File | Superseded By |
|------|--------------|
| [`archive/CEREBRUM_MASTER_PAPER_v251.md`](archive/CEREBRUM_MASTER_PAPER_v251.md) | `CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md` |
| [`archive/CEREBRUM_MASTER_PAPER_CLEAN_v251.md`](archive/CEREBRUM_MASTER_PAPER_CLEAN_v251.md) | `CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md` |
| [`archive/CEREBRUM_MASTER_WHITEPAPER_v251.md`](archive/CEREBRUM_MASTER_WHITEPAPER_v251.md) | `CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md` |
| [`archive/Cerebrum_White_Paper_v251.md`](archive/Cerebrum_White_Paper_v251.md) | `CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md` |
| [`archive/PAPER_v251.md`](archive/PAPER_v251.md) | `CEREBRUM_MASTER_SCIENTIFIC_MANUSCRIPT.md` |
| [`archive/PARALLAX_v251.md`](archive/PARALLAX_v251.md) | Internal reference only — do not cite |

---

## LaTeX Source

| Path | Purpose |
|------|---------|
| [`latex/compiled/`](latex/compiled/) | Per-paper `.tex` files (Papers 001–017 compiled; 018–037 planned Phase 6) |
| [`latex/cerebrum_master.tex`](latex/cerebrum_master.tex) | Master LaTeX driver (inputs all compiled papers) |
| [`latex/references.bib`](latex/references.bib) | Bibliography (KGQA baselines + internal citations) |
| [`latex/templates/`](latex/templates/) | Shared LaTeX templates |

---

*Last updated: 2026-05-08 | v2.52.0 publication cycle*

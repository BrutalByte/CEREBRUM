# CEREBRUM Publication & IP Strategy
## Game Plan for Claude Code Execution
### Bryan Alexander Buchorn | May 2026

---

## The Problem

You have one 4,800-line master document containing 22 papers (37 claimed, ~22 distinct with reference sections). It's currently a project monograph — great for internal documentation, but not submittable to arXiv as-is.

**Two goals, two different artifacts:**
1. **arXiv submissions** — peer-readable, properly scoped, competitively positioned
2. **IP documentation** — timestamped, comprehensive, establishing prior art

These goals need different outputs. Trying to serve both with one document is why the current version doesn't quite work for either.

---

## Recommended Structure: 1 + 4 + 1

### The "1" — Flagship Systems Paper (arXiv: cs.AI)

**Title:** *CEREBRUM: Training-Free Knowledge Graph Reasoning via Community-Structured Graph Attention*

**Purpose:** This is the anchor paper. It introduces the full system architecture and positions CEREBRUM against SOTA. Everything else references back to this.

**Length:** 12-15 pages (NeurIPS / AAAI format)

**Content pulled from these current papers:**
- DSCF/TSC (community detection — the structural foundation)
- CSA (the attention formula — the core contribution)
- Bayesian Beam Search (the traversal mechanism)
- Bridge Twin Engine (graph plasticity — what makes it novel)

**What makes this paper work:**
- Single coherent narrative: detect communities → score edges → search paths → evolve topology
- Clear SOTA comparison table (MetaQA, WebQSP, Hetionet) with explicit baselines
- The "training-free" angle is your competitive differentiator — lean into it hard
- One benchmark results section, not the same table repeated 10 times

**Critical additions needed:**
- Proper related work section positioning against: GNN-QE, QTO, UniKGQA, NSM, TransferNet, EmbedKGQA
- Honest SOTA comparison: MetaQA SOTA is ~95%+ H@1 (supervised). Your H@1=46.1% (1-hop) needs framing as "training-free baseline" not just raw numbers
- Ablation study: CSA with/without bridge twins, with/without adaptive beam, etc.
- Reproducibility section: hyperparameters, dataset splits, compute requirements

---

### The "4" — Focused Technique Papers

Each one stands alone. Each one references the flagship paper. Submit to arXiv within 1-2 weeks of each other.

#### Paper A: Community Detection (arXiv: cs.SI or cs.LG)

**Title:** *Triple-Signal Consensus: Temperature-Annealed Community Detection for Graph Attention*

**Source:** DSCF/TSC paper (expanded)

**Why it works standalone:** The Q=0.88 vs Leiden Q=0.48 result on caveman graphs is a real contribution to the community detection literature, independent of CEREBRUM. But you need to:
- Benchmark on standard community detection datasets (LFR benchmarks, not just caveman)
- Compare against Leiden, Louvain, Infomap, spectral methods
- Show NMI/ARI scores, not just modularity
- The temperature annealing schedule is the novel part — give it space

**Length:** 8-10 pages

#### Paper B: Graph Plasticity (arXiv: cs.AI)

**Title:** *Experience-Dependent Structural Plasticity in Knowledge Graphs: Bridge Twins and Causal Discovery via STDP*

**Source:** Bridge Twin Engine + STDP Causal Discovery (merged)

**Why merge these:** Both papers use the same biological analogy (Hebbian plasticity → graph evolution). Separately they're thin. Together they tell a story: the graph learns from experience (bridges) AND discovers causal structure (STDP). The Lazy Decay optimization is a nice practical contribution.

**Length:** 10-12 pages

#### Paper C: Federated Reasoning (arXiv: cs.DC or cs.AI)

**Title:** *Holographic Indexing: Privacy-Preserving Discovery in Federated Knowledge Graph Networks*

**Source:** Holographic Indexing paper + federation sections from other papers

**Why standalone:** Federated KG reasoning with privacy guarantees is a distinct research area. The Bloom filter + community centroid dual-tier approach is a clean contribution. The Procrustes alignment for cross-node embedding spaces is a nice addition.

**Needs:**
- Formal privacy analysis (what information leaks through community centroids?)
- Communication complexity analysis vs. full federation approaches
- Experimental evaluation on partitioned benchmark KGs

**Length:** 8-10 pages

#### Paper D: Production Engineering (arXiv: cs.SE or cs.DB)

**Title:** *Production Knowledge Graph Reasoning: Fault Tolerance, Streaming Ingestion, and Metacognitive Maintenance*

**Source:** Five Fault-Tolerance Patterns + Streaming Engine + REM Cycle + THALAMUS + Structural Hole Patching

**Why this grouping:** These are all "making it work in production" papers. Individually they're engineering reports. Together they form a systems paper about operationalizing KG reasoning — which is genuinely underserved in the literature.

**Length:** 10-12 pages

---

### The Other "1" — IP Technical Report (arXiv: cs.AI, category: technical report)

**Title:** *CEREBRUM v2.52.0: Complete Technical Specification for Autonomous Knowledge Graph Reasoning*

**Purpose:** This is your prior art document. It's the full monograph, cleaned up.

**What goes here:** Everything. All 22 papers, lightly edited for consistency, with a unified introduction, shared notation table, and comprehensive bibliography. This is where the UE5 visualization paper lives, the TemporalCalibrator, the Engram traversal, the GraphSAGE smoothing, the Glass-Box Studio, the InsightValidator — all the pieces that don't fit neatly into the 4 focused papers.

**arXiv allows technical reports.** They don't need to be peer-reviewed. They establish a public timestamp. That's your IP play.

---

## Authorship — The Non-Negotiable Fix

**Current:** "Bryan Alexander Buchorn " with "Anthropic" as affiliation.

**Problem:** arXiv requires authors to be human individuals who take legal and intellectual responsibility. Listing an AI model as co-author will get flagged. Listing Anthropic as affiliation without their approval creates a separate legal issue.

**Fix for all papers:**

```
Author: Bryan Alexander Buchorn
Affiliation: Independent Researcher, Las Vegas, NV, USA

Acknowledgments: "Portions of this work were developed in collaboration
with Claude (Anthropic), which served as a research assistant for
formalization, code generation, and manuscript preparation."
```

This is honest, avoids the arXiv policy issue, and is consistent with how other researchers are handling AI-assisted work right now.

---

## Self-Citation Strategy

**Current problem:** References cite internal spec files like "SPEC_003.md" and "PARALLAX.md" — reviewers can't access these.

**Fix:** Once the technical report (the "1" at the bottom) is on arXiv, all focused papers cite it via its arXiv ID. Internal spec references become: "See Section 4.3 of the CEREBRUM Technical Report [Buchorn, 2026]."

**Submission order matters:**
1. Submit the technical report first (it's the easiest to prepare — closest to current state)
2. Submit the flagship systems paper next
3. Submit the 4 focused papers, each referencing both the technical report and the flagship

---

## Benchmark Positioning — Be Honest, Be Strategic

Your MetaQA numbers need context. Here's roughly where things stand:

| Method | Type | 1-hop H@1 | 2-hop H@1 | 3-hop H@1 |
|--------|------|-----------|-----------|-----------|
| EmbedKGQA | Supervised | ~97% | ~94% | ~94% |
| NSM | Supervised | ~97% | ~99% | ~98% |
| UniKGQA | Supervised | ~97% | ~99% | ~99% |
| **CEREBRUM** | **Training-free** | **46.1%** | **30.0%** | **12.5%** |

Raw comparison kills you. But "training-free" is a different category. Frame it like this:

> "CEREBRUM achieves these results with zero task-specific training, no labeled question-answer pairs, and no gradient updates — operating purely from graph structure and pre-trained sentence embeddings. To our knowledge, this represents the first training-free baseline for multi-hop KGQA, establishing a reference point for what structural reasoning alone can achieve."

Then the H@10 numbers (96.6% / 86.3% / 50.3%) become the real story — the system *finds* the right answers in its top-10, it just doesn't rank them first. That's a ranking problem, not a reasoning problem. Say that explicitly.

**You should also verify these SOTA numbers are current before submission.** Search arXiv for "MetaQA" and "KGQA" papers from 2024-2026 to get the latest baselines.

---

## Claude Code Execution Plan

Here's the sequence of tasks to hand to Claude Code. Each is a discrete work unit.

### Phase 1: Setup & Extraction (Day 1)

```
Task 1.1: Create project directory structure

cerebrum-papers/
├── 00-technical-report/        # The full monograph (IP document)
│   ├── sections/               # One .md per original paper
│   ├── figures/
│   ├── tables/
│   ├── cerebrum-v251-report.tex
│   └── references.bib
├── 01-flagship/                # Main systems paper
│   ├── cerebrum-flagship.tex
│   ├── figures/
│   └── references.bib
├── 02-community-detection/     # Paper A: TSC
│   ├── tsc-community.tex
│   └── references.bib
├── 03-graph-plasticity/        # Paper B: Bridge + STDP
│   ├── graph-plasticity.tex
│   └── references.bib
├── 04-federated/               # Paper C: Holographic Indexing
│   ├── holographic-indexing.tex
│   └── references.bib
├── 05-production/              # Paper D: Production engineering
│   ├── production-kg.tex
│   └── references.bib
├── shared/
│   ├── notation.tex            # Shared math notation macros
│   ├── cerebrum-macros.sty     # Shared LaTeX package
│   └── author-block.tex        # Standardized author info
└── GAMEPLAN.md                 # This file
```

```
Task 1.2: Extract and split the master manuscript

Read the master .docx file. Split it into individual markdown files
in 00-technical-report/sections/, one per paper. Name them:

01-dscf-tsc.md
02-csa.md
03-bridge-twin.md
04-stdp-causal.md
05-holographic-indexing.md
06-bayesian-beam.md
07-rem-cycle.md
08-procrustes.md
09-thalamus.md
10-inference-validator.md
11-contradiction.md
12-glass-box.md
13-streaming-engine.md
14-metacognitive-verification.md
15-algorithmic-depth.md
16-structural-holes.md
17-graphsage-smoothing.md
18-engram-traversal.md
19-temporal-calibrator.md
20-fault-tolerance.md
21-neural-viz-bridge.md
22-conclusion.md

Preserve all math notation, tables, and references.
```

```
Task 1.3: Build a unified references.bib

Extract all references from every paper. Deduplicate.
Use consistent BibTeX keys. Flag any references that cite
internal spec files (SPEC_xxx.md, PARALLAX.md) — these need
to be replaced with the technical report arXiv citation later.
```

### Phase 2: Technical Report Assembly (Days 2-3)

```
Task 2.1: Write unified front matter for the technical report

- Title: "CEREBRUM v2.52.0: Complete Technical Specification for
  Autonomous Knowledge Graph Reasoning"
- Author: Bryan Alexander Buchorn (Independent Researcher)
- Single comprehensive abstract covering the full system
- Notation table defining ALL symbols used across all papers
- System architecture diagram description (for a figure)

Do NOT list Claude as author. Add acknowledgment section.
```

```
Task 2.2: Create a shared notation table

Go through every paper and extract every mathematical symbol.
Build a single table:

| Symbol | Definition | First Appears |
|--------|-----------|---------------|
| ℒ(v,C) | Local LPA signal | DSCF/TSC |
| 𝒢(v,C) | Global modularity gain | DSCF/TSC |
| ...    | ...       | ...           |

Ensure notation is consistent across all papers. Flag conflicts
where the same symbol means different things in different papers.
```

```
Task 2.3: Write chapter transitions

The technical report needs 1-2 paragraph transitions between
each paper/chapter explaining how they connect. Example:

"The community partitions produced by DSCF/TSC (Chapter 2)
provide the structural foundation for the attention mechanism
described next. CSA treats each community as a discrete
attention head..."
```

```
Task 2.4: Compile technical report LaTeX

Use a standard arXiv-compatible template (article class or
similar). Number chapters sequentially. Single bibliography
at the end. Include a table of contents.
```

### Phase 3: Flagship Paper (Days 3-5)

```
Task 3.1: Draft flagship paper structure

Sections:
1. Introduction (1.5 pages)
   - KG reasoning is important
   - Existing methods require training
   - We propose a training-free approach
   - Contributions list (4 bullets)

2. Related Work (1.5 pages)
   - Embedding-based KGQA (TransE, RotatE, EmbedKGQA)
   - GNN-based KGQA (NSM, UniKGQA, GNN-QE)
   - Community detection in graphs
   - Neuro-symbolic approaches
   - Training-free / zero-shot approaches

3. The CEREBRUM Framework (4 pages)
   3.1 Community Detection via TSC (condensed from Paper 1)
   3.2 Community-Structured Attention (condensed from Paper 2)
   3.3 Bayesian Beam Traversal (condensed from Paper 6)
   3.4 Experience-Dependent Bridge Synthesis (condensed from Paper 3)

4. Experimental Evaluation (3 pages)
   4.1 Datasets (MetaQA, WebQSP, Hetionet)
   4.2 Baselines and metrics
   4.3 Main results table
   4.4 Ablation study
   4.5 Qualitative analysis / case study

5. Discussion (1 page)
   - Training-free vs supervised gap
   - H@10 vs H@1 — ranking vs retrieval
   - Interpretability advantage
   - Limitations

6. Conclusion (0.5 pages)
```

```
Task 3.2: Write the related work section

This is the most critical section for arXiv credibility.
Search for and cite these papers (verify they exist and are current):

- EmbedKGQA (Saxena et al., 2020)
- NSM (He et al., 2021)
- UniKGQA (Jiang et al., 2023)
- GNN-QE (Zhu et al., 2022)
- QTO (Bai et al., 2023)
- TransferNet (Shi et al., 2021)
- KV-Mem (Miller et al., 2016)
- GraftNet (Sun et al., 2018)
- PullNet (Sun et al., 2019)

Position CEREBRUM explicitly: "Unlike these methods, CEREBRUM
requires no task-specific training data..."
```

```
Task 3.3: Build the comparison table

Create a properly formatted table with columns:
Method | Training Required | MetaQA 1-hop | 2-hop | 3-hop | WebQSP

Include at least 5-6 baselines. Mark CEREBRUM as "None" for
training. Add a footnote explaining the training-free distinction.
```

```
Task 3.4: Design the ablation study table

| Configuration | MetaQA 1-hop H@1 | H@10 | Notes |
|---------------|-------------------|------|-------|
| Full CEREBRUM | 46.1% | 96.6% | All components |
| No bridge twins | ? | ? | Static graph |
| No adaptive beam | ? | ? | Fixed beam width |
| DSCF instead of TSC | ? | ? | No centrality signal |
| No GraphSAGE smoothing | ? | ? | Raw embeddings |

NOTE: If you don't have these ablation numbers, you need to
run them before submission. This is non-negotiable for a
credible systems paper.
```

### Phase 4: Focused Papers (Days 5-8)

```
Task 4.1: Draft Paper A (TSC Community Detection)

Pull from 01-dscf-tsc.md. Expand with:
- LFR benchmark evaluation (or note that it's needed)
- NMI/ARI comparison vs Leiden, Louvain, Infomap
- Scalability analysis (wall-clock time vs graph size)
- The temperature schedule as a contribution (not just a detail)
```

```
Task 4.2: Draft Paper B (Graph Plasticity)

Merge 03-bridge-twin.md and 04-stdp-causal.md. Frame as:
- Section 1: The case for plastic KGs
- Section 2: Bridge Twin potentiation (LTP analog)
- Section 3: STDP causal discovery (temporal learning)
- Section 4: Lazy Decay (shared optimization)
- Section 5: Interaction effects (bridges inform causal paths)
```

```
Task 4.3: Draft Paper C (Federated Reasoning)

Pull from 05-holographic-indexing.md. Add:
- Formal privacy analysis
- Communication cost analysis (bits exchanged vs alternatives)
- Procrustes alignment evaluation
- Multi-node experimental setup description
```

```
Task 4.4: Draft Paper D (Production Engineering)

Merge: fault-tolerance, streaming-engine, REM cycle,
THALAMUS, structural holes. Frame as lessons learned:
- "We deployed CEREBRUM as a production service and
  encountered these N problems. Here's how we solved them."
- This framing is honest and publishable in systems venues.
```

### Phase 5: Polish & Submit (Days 8-10)

```
Task 5.1: Cross-reference audit

For every paper:
- Replace all internal spec citations with technical report refs
- Ensure notation matches the shared notation table
- Verify no benchmark numbers are inconsistent across papers
- Check that the same result isn't claimed as novel in two papers
```

```
Task 5.2: LaTeX compilation and formatting

- Use a clean arXiv-compatible template for each paper
- Compile cleanly with pdflatex/bibtex
- Check page limits
- Generate PDF for manual review
```

```
Task 5.3: Submission order

1. Technical report → arXiv (gets an ID immediately)
2. Wait 24 hours (ID propagates)
3. Flagship paper → arXiv (cites technical report)
4. Papers A-D → arXiv (cite both flagship and report)
```

---

## Things That Still Need Doing (Outside Claude Code)

These are tasks only you can do:

1. **Run ablation experiments.** The flagship paper needs ablation numbers. Without them, reviewers will ask and you'll have no answer.

2. **Verify SOTA baselines.** Search arXiv for the latest MetaQA and WebQSP results. The numbers I referenced may be outdated.

3. **Decide on LaTeX template.** NeurIPS format? AAAI? Plain article class? This affects page counts and formatting.

4. **Proofread for voice consistency.** Claude Code can draft, but the final voice should be yours throughout.

5. **Consider a project page.** A simple GitHub Pages site with links to all papers, the codebase (if you're open-sourcing), and a system diagram. This is increasingly expected for systems papers.

6. **IP filing timeline.** If you're considering patent protection for any specific mechanism (TSC temperature annealing, Lazy Decay, Holographic Indexing), file a provisional before the arXiv submission. arXiv preprints are public disclosure.

---

## Summary

| Deliverable | Pages | Source Papers | arXiv Category | Priority |
|-------------|-------|---------------|----------------|----------|
| Technical Report | 60-80 | All 22 | cs.AI | 1 (submit first) |
| Flagship Paper | 12-15 | 1,2,3,6 | cs.AI | 2 |
| Paper A: TSC | 8-10 | 1 | cs.SI / cs.LG | 3 |
| Paper B: Plasticity | 10-12 | 3,4 | cs.AI | 3 |
| Paper C: Federation | 8-10 | 5 | cs.DC | 3 |
| Paper D: Production | 10-12 | 7,9,13,16,20 | cs.SE / cs.DB | 3 |

Total: 6 artifacts from 1 monograph. Each serves a purpose. No wasted work.

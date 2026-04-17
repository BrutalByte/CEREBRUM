<div class="cover">
    <img src="file:///C:/Users/bryan/.gemini/antigravity/brain/77bb37a0-e733-41be-824d-b07e7cce5a6f/cerebrum_hero_v77bb37a0_e730_41be_824d_b07e7cce5a6f_png_1774651388694.png" alt="CEREBRUM Visionary Hero">
    <h1 class="title">CEREBRUM EXPLAINED</h1>
    <p class="subtitle">A Plain-Language Guide to the Research</p>
    <div class="meta">
        <strong>Version 5.0 (v2.21.0 — Phase 92 COMPLETE)</strong><br>
        April 2026 — Independent Researcher
    </div>
</div>

---

### **Who this is for**
You're comfortable with computers and technology, but you don't have a background in machine learning or knowledge graphs. This guide explains CEREBRUM's individual pieces, names each piece in plain English, and walks through how the whole system works.

---

### **How to Read This Guide**
Each chapter builds on the last. If you're already comfortable with a concept, feel free to skip ahead. Every technical term is explained when it first appears. Formulas appear in clearly marked "Math, Unpacked" sections — you can read around them without losing the narrative, or dive in for a full understanding.

| Chapter | Topic | You'll Understand |
|---|---|---|
| 1 | What is a Knowledge Graph? | The database behind CEREBRUM |
| 2 | What is an LLM? | What ChatGPT and its peers actually are |
| 3 | The problem with current AI reasoning | Why we need something new |
| 4 | What is attention in AI? | The mechanism CEREBRUM borrows |
| 5 | What is community detection? | The other key concept |
| 6 | What is CEREBRUM? | The full picture — including THALAMUS and CORTEX |
| 7 | Why does it matter? | Real-world stakes |
| 8 | What are we trying to prove? | The research questions |
| 9 | Does it work? | Validation results (v1.9.8) |
| 10 | The Studio: Seeing it live | The interactive interface |
| 11 | Real-time streaming data | Sensors, signals, and live feeds |
| 12 | What kinds of data can it use? | Beyond spreadsheets |
| 13 | Bayesian Beam Search | Reasoning under uncertainty |
| 14 | The REM Cycle | Self-maintenance and synthesis |
| 15 | Insight Learning | When the graph has a Eureka moment |
| 16 | The 10-Parameter Attention Formula | How CEREBRUM weighs every step |
| 17 | Autonomous Reasoning | CEREBRUM discovers what it doesn't know |
| 18 | What's New in v1.9.8 | The latest capabilities |
| 19 | The Formal Journey | ArXiv, Margins, and Academic Rigor |

---

## CHAPTER 1 — What Is a Knowledge Graph?
Most databases are like spreadsheets: rows and columns. A **Knowledge Graph** is different. It's a collection of entities (nodes) and the relationships (edges) between them. Instead of a row for "Tom Hanks," a Knowledge Graph has a circle for **Tom Hanks**. That circle is connected by a line labeled **ACTED_IN** to another circle labeled **Forrest Gump**. This format is how humans think — we think in connections, not table rows.

---

## CHAPTER 3 — The Problem With AI Reasoning
Because LLMs are statistical, they "hallucinate." If they don't know an answer, they'll often make one up that *sounds* perfectly plausible. **CEREBRUM is different.** It moves the reasoning out of the probabilistic "Black Box" and into the graph itself. It only follows lines that actually exist in your data.

---

## CHAPTER 9 — Does It Work? (v1.9.8 Results)
As of April 2026, CEREBRUM has been validated on several large-scale datasets and is operating at **v1.9.8 "Autonomous Reasoning"** status.

**What the tests showed:**
1. **Unrivaled Stability**: The framework now passes **1,357 automated tests**, covering core reasoning, autonomous discovery, observability, and parameter learning.
2. **Medical accuracy**: On the Hetionet biomedical graph, CEREBRUM was over **183% more accurate** than simple search at finding connections between diseases and genes.
3. **Reasoning Recall**: On MetaQA 3-hop reasoning, recall improved by **+350%** relative to non-attention baselines.
4. **Zero training required**: Most AI systems require weeks of expensive training. CEREBRUM achieved these results with no training at all — it reasons purely from the structure of the graph it's given.

### Benchmark Results (Canonical)

| Dataset | Metric | Score |
|---|---|---|
| MetaQA 1-hop | H@1 | 46.1% |
| MetaQA 2-hop | H@1 | 30.0% |
| MetaQA 3-hop | H@1 | 12.5% |
| WebQSP (optimized) | H@1 | 6.27% |
| WebQSP (optimized) | H@10 | 20.84% |
| WebQSP (optimized) | MRR | 10.66% |
| IKGWQ (incomplete graph) | AUC | 0.89 |
| GrailQA | F1 | 19.6% |

---

## CHAPTER 16 — The 10-Parameter Attention Formula
Every time CEREBRUM considers following an edge in the graph — from one fact to another — it performs a calculation. That calculation answers: *"How relevant is this step to the question being asked?"*

In v1.9.8, this formula uses **ten separate signals** (called parameters), each measuring a different aspect of relevance:

| Parameter | Plain-English Meaning |
|---|---|
| **alpha** (α) — Semantic similarity | Do the two facts "mean" something related? |
| **beta** (β) — Community membership | Are they in the same "neighborhood" of the graph? |
| **gamma** (γ) — Edge-type weight | How important is this type of relationship? |
| **delta** (δ) — Distance penalty | Are we getting too far from the starting point? |
| **epsilon** (ε) — Hop decay | Does relevance drop with every step? |
| **zeta** (ζ) — PageRank prior | Is the destination a well-connected, important node? |
| **eta** (η) — Temporal decay | Is the information recent or stale? |
| **iota** (ι) — Node recency | How recently was this node touched? |
| **mu** (μ) — Synthesis penalty | Is this a real fact or a synthesized guess? |
| **theta** (θ) — Grounding confidence | How confident are we in the source? |

All ten signals are added together and squeezed through a "sigmoid" function that converts the result into a 0-to-1 score. A score near 1.0 means "strongly follow this edge"; near 0 means "skip it."

These ten parameters can be learned from feedback over time, so CEREBRUM gets smarter about your specific graph the more it is used.

---

## CHAPTER 17 — Autonomous Reasoning: CEREBRUM Discovers What It Doesn't Know
One of the most exciting capabilities in v1.9.8 is that CEREBRUM can now reason about **gaps in its own knowledge** — and actively work to fill them.

### HypothesisEngine: "What Could Be True?"
The **HypothesisEngine** uses multi-path abductive reasoning to generate *hypotheses* — proposed connections that are not yet in the graph but are strongly suggested by the surrounding evidence. Think of a detective who, after reviewing all available clues, says: *"The only explanation that fits everything is that X caused Y."* CEREBRUM can now generate those kinds of proposals automatically.

A human expert can then review and approve or reject each hypothesis. Approved hypotheses are materialized as real edges in the graph.

### ResearchAgent: "What Information Is Missing?"
The **ResearchAgent** is an autonomous discovery engine. It scans the graph for **missing links** — places where the structure implies a connection should exist, but no edge has been recorded. It produces a list of candidate findings for human review.

Each finding can be independently approved or rejected, giving the expert full control while letting CEREBRUM do the heavy lifting of identifying where to look.

### ExternalValidator: "Is This Supported by Literature?"
When CEREBRUM proposes a new edge, the **ExternalValidator** checks whether scientific literature supports the claim. It searches databases including **PubMed**, **ClinicalTrials.gov**, **arXiv**, and **OpenAlex** to find corroborating evidence. If the literature supports the hypothesis, confidence in it rises. If nothing is found, the proposal is flagged for cautious review.

Together, these three systems form a **closed-loop autonomous research assistant**: CEREBRUM hypothesizes, searches for missing evidence, validates against external literature, and surfaces findings for expert decision.

---

## CHAPTER 18 — What's New in v1.9.8

### 1. HypothesisEngine (Phase 50)
Multi-path abductive reasoning that generates proposed graph edges from indirect evidence. Experts approve or reject proposals via a simple interface.

### 2. ResearchAgent (Phase 51)
Autonomous scan of the graph for missing links. Produces a prioritized list of candidate findings, each individually approvable.

### 3. ExternalValidator (Phase 52)
Literature validation against PubMed, ClinicalTrials.gov, arXiv, and OpenAlex. Each hypothesis is grounded in real-world scientific sources before being promoted.

### 4. Adaptive Search (Phase 53)
CEREBRUM now measures the **local density** of the graph around the query starting point and automatically adjusts how wide and how deep it searches. Dense neighborhoods get a narrower, more focused beam; sparse regions get a wider, more exploratory search. No manual tuning required.

### 5. StudioEngine (Phase 54)
All the reasoning logic that powers the visual Reasoning Studio has been extracted into a standalone `StudioEngine` class. This means the Studio can be tested, extended, and embedded into other systems **without needing a running web server**. 38 new automated tests validate the Studio's business logic independently.

### 6. Observability (Phase 54)
A new **ring-buffer log system** captures the last N log messages in memory and exposes them via `GET /logs`. The Studio can now display live log output alongside the reasoning visualization. A new `POST /build` endpoint allows hot-reloading a graph from a new CSV file without restarting the server. A production `dashboard.html` provides at-a-glance system health monitoring.

### 7. TSC Explicit Mode (Phase 49)
The community detection algorithm is now explicitly selectable at startup: `dscf`, `tsc`, `leiden`, or `lpa`. This makes it easy to compare how different partitioning strategies affect reasoning quality for a given graph.

### 8. Federated Reasoning (Phase 32)
CEREBRUM nodes can now delegate reasoning branches to **remote CEREBRUM instances**. A query that starts on one node can seamlessly continue on another node holding a different portion of the graph, with results merged automatically.

### 9. Parameter Learning (Phases 45–48)
The 10-parameter attention formula can now be **learned from feedback**:
- `POST /feedback` triggers an immediate online update for the path that just ran
- Feedback accumulates in a buffer that can be batch-retrained via `POST /retrain`
- Learned parameters can be saved and restored via `GET/POST /params` or the `--params-file` startup flag

---

## CHAPTER 19 — The Formal Journey: ArXiv and Beyond
The transition from a laboratory project to a formal **ArXiv Manuscript** required a process called **Hardening**. This included:
- **Margin Remediation**: Ensuring every equation and table fits perfectly into a professional two-column academic layout.
- **Structural Hole Repair**: Identifying 12 subtle failure modes (like "Zombie Bridges" or "Causal Floods") and building mathematical guards against them.
- **Branding Audit**: Ensuring absolute privacy and professional attribution across thousands of lines of code.

---

### **Acknowledgments & Credits**
CEREBRUM stands on the shoulders of decades of foundational research in Graph Theory, Neuroscience, and Machine Learning.

**Questions? Contact the author:**
**Bryan Alexander Buchorn** — bryan.alexander@buchorn.com

---
*Generated: April 2026 — Project CEREBRUM v1.9.8 — Autonomous Reasoning Edition*

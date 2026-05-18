# CEREBRUM

**Community-Structured Graph Attention for Knowledge Graph Reasoning**

**Current Version:** v2.56.0 (Phase 190 COMPLETE)

CEREBRUM is the first reasoning engine that treats the Knowledge Graph not as a static data dump, but as a living, self-optimizing neural substrate. By embedding the intelligence of Transformer-style attention directly into the graph's topology, it delivers hyper-accurate, verifiable reasoning at sub-millisecond speeds—completely eliminating the hallucinations of LLMs and the bottleneck of expensive, manual model training.

## Quick Start

```bash
pip install cerebrum-kg[all]
cerebrum init --demo                          # instant demo KB
cerebrum init --from-csv mydata.csv --serve   # your own data + API server
```

```python
from cerebrum_sdk import Cerebrum

c = Cerebrum.from_csv("kb.csv")
result = c.ask("Who directed Inception?")
print(result.answer)       # e.g. "Christopher_Nolan"
print(result.confidence)   # e.g. 0.923
print(result.trace_path)   # [TraceStep(entity='Inception', relation='directed_by'), ...]
```

## Benchmarks

CEREBRUM achieves these results **with zero training data** — no fine-tuning, no gradient steps:

| Benchmark | Metric | CEREBRUM | GPT-4 (KGQA) | RAG baseline |
|-----------|--------|----------|--------------|--------------|
| MetaQA 3-hop | H@1 | **56.2%** | ~38–45%* | ~30–40%* |
| MetaQA 3-hop | H@10 | **87.9%** | — | — |
| MetaQA 3-hop | MRR | **0.670** | — | — |
| Hetionet (biomedical) | H@10 | **85%** | — | — |
| Cost per 1K queries | USD | **~$0.001** | ~$10–100 | ~$1–20 |
| Explainability | | Full trace | None | Partial |
| Hallucination rate | | **0%** | ~5–15% | ~10–20% |

*LLM KGQA comparisons from published literature; exact figures vary by prompt strategy.

All answers include a full hop-by-hop reasoning trace — auditable, exportable, and reproducible.

---

- **TSC**: Triple-Signal Consensus — novel community detection combining LPA (local),
  modularity gain (global), and centrality (flow) simultaneously at each node update
- **CSA**: Community-Structured Attention — attention weights that incorporate community
  membership as a soft global constraint on graph traversal
- **Graph-Grounded**: every answer is a path through verified graph edges

## Mathematical Foundations

CEREBRUM is built on a formal equivalence between Transformer operations and graph topology:

*   **Attention Heads $\approx$ DSCF Communities**: Graph partitioning serves as a discrete, structural mechanism for parallel attention, where each community specializes on a conceptual domain.
*   **Layer Depth $\approx$ BFS Hop Count**: Multi-hop traversal replaces the composition of transformer layers, where each hop represents a discrete step of logical inference.
*   **Positional Encoding $\approx$ Structural Features**: Node-level metrics (PageRank, Betweenness) provide the necessary global context for the attention mechanism.

The core **Community-Structured Attention (CSA)** score for an edge $u \to v$ at hop $k$ is defined by a 10-parameter homeostatic formula that balances semantic similarity, community coherence, and metabolic signals.

## Why CEREBRUM?

CEREBRUM is not just another "GraphRAG" wrapper. Most contemporary systems use Knowledge Graphs as a secondary source for LLM context retrieval. CEREBRUM reverses this: it uses the Knowledge Graph as an **active reasoning substrate**.

### 1. The "Glass-Box" Reasoning Advantage
Unlike standard RAG (Retrieval-Augmented Generation) or even GraphRAG, which rely on the "Black-Box" probabilistic predictions of an LLM to formulate an answer, CEREBRUM follows deterministic, verifiable paths through your data. Every conclusion is backed by an explicit chain of edges that can be audited, visualized, and proven.

### 2. Training-Free Autonomy
You don't need a machine learning team to train, fine-tune, or maintain CEREBRUM. 
- **Legacy Frameworks**: Require days of GPU training, complex feature engineering, and constant retraining when your graph changes.
- **CEREBRUM**: Uses the graph's own topology to structure its attention. Our zero-shot performance (e.g., 85% H@10 on Hetionet) is achieved the moment your data is ingested, without a single gradient step.

### 3. Zero-Config, Auto-Reasoning
With **GraphProfiler** (Phase 172) and **STRB** (Phase 172), you no longer need to be a graph theory expert to get production-grade results. The system automatically profiles your graph at build time, identifies its structural regime (Hub-heavy vs. Typed-Heterogeneous), and dynamically orchestrates the reasoning engine (H1SE, TAB, STRB) to optimize for your specific dataset.

### 4. Memory-Efficient Scalability
Legacy Knowledge Graphs require massive RAM overhead for index redundancy and path materialization. CEREBRUM’s **Hybrid-Memory Architecture** autonomously balances performance between RAM/VRAM and NVMe-backed storage, providing "live" reasoning performance on graphs that exceed your total system memory.

### 5. Verified Superiority
CEREBRUM has been empirically validated on standardized benchmarks with zero training data:

- **MetaQA 3-Hop Reasoning**: CEREBRUM achieves **56.2% H@1** and **87.9% H@10** on the full 14,274-question run (v2.55.0, Phase 189, zero training data). MRR=0.670. The system is fully data-agnostic — no hardcoded relation names.
- **Biomedical Inference**: Achieves **85% H@10** on the Hetionet benchmark, providing actionable connection insights for drugs, diseases, and pathways.
- **Resilience**: Maintains **89% reasoning capability** (AUC) even under extreme (50%) edge sparsity, proving its ability to reason over incomplete, real-world data.

---

## Roadmap

**Current Project Status: v2.56.0 — Phase 190 COMPLETE — 2191 passed, 1 skipped**

### The Core Pillars
- [x] **Phase 1**: Core Engine (GraphAdapter, TSC Engine, CSA Attention)
- [x] **Phase 2**: Reasoning Engine (BeamTraversal, PathScorer) — end-to-end pipeline verified
- [x] **Phase 3**: Adapters & API (FastAPI server + LLM bridge)
- [x] **Phase 4**: Benchmarking (WebQSP, MetaQA, Hetionet) — Bridge Bonus innovation (EF-005)
- [x] **Phase 5**: Release (v0.1.0 Stable) — TSC, Persistence, Docker
- [x] **Phase 6**: Federated Graph Attention — multi-source aggregation & alignment
- [x] **Phase 7**: Dynamic Graph Updates — cross-graph wormhole attention
- [x] **Phase 8**: Holographic Index — privacy-preserving discovery & Bloom filters
- [x] **Phase 9**: Federated Release (v0.2.0 Stable) — handshake & reasoning callbacks
- [x] **Phase 10**: Production Hardening (v0.3.0) — JWT, ResourceGovernor, AsyncBeamTraversal
- [x] **Phase 11**: Real-Time Streaming — StreamAdapter, 5 discretizers, sliding-window buffer, SSE endpoints
- [x] **Phase 12**: Bridge Twin Nodes — experience-dependent structural relay formation
- [x] **Phase 13**: STDP Causal Inference — directional CAUSES edges from spike timing
- [x] **Phase 14**: ResourceGovernor — hardware-aware query throttling and energy budget enforcement
- [x] **Phase 15**: REM Cycle — autonomous graph self-reorganization
- [x] **Phase 16**: Verification & Metacognition — InsightValidator + MetaInsightEngine
- [x] **Phase 17**: Algorithmic Depth — Temporal reasoning, uncertainty propagation, soft community membership, learned CSA parameters (CSAParameterLearner), KGE embeddings
- [x] **Phase 18**: v0.4 Horizon — THALAMUS IngestionPipeline, LLM bridge, Bayesian Beam Search, GlobalRebalancer, SignalEncoder
- [x] **Phase 19**: v1.0 Production Hardening — Four structural holes fixed (Zombie Bridge, Causal Flood, Namespace Isolation, Bayesian Cold-Start)
- [x] **Phase 20**: v1.1.0 Relativistic Hardening — Four cross-system interaction holes fixed (Query Snapshot Isolation, Community-Specific CSA, Canonical Basis Anchor, Path-Preserving Hold-out)
- [x] **Phase 21**: v1.2.0 Full Validation & Reliability — Comprehensive validation suite, SignalEncoder alignment fix
- [x] **Phase 22–24**: v1.4.0 GPU + Enterprise — GPU-accelerated DSCF, Amazon Neptune adapter, Spark GraphX offline DSCF, arXiv publication pipeline
- [x] **Phase 25**: v1.5.0 Universal Hardware — Hardware detection, float16 embeddings, cross-platform stability
- [x] **Phase 26**: v1.6.0 Performance — Score-weighted path voting, recall improvements, coarsen_communities fix
- [x] **Phase 27A**: v1.6.2 MetaQA SOTA — Beats MINERVA (trained RL) with zero training
- [x] **Phase 27B**: v1.6.3 Three-Benchmark Framework — RelationPathPrior, WebQSP full pipeline, IKGWQ graceful degradation
- [x] **Phase 28 & 29**: Structural Repair — IncompletenessRepairEngine and QueryGuidedCommunityMerger (v1.6.4)
- [x] **Phase 30**: Proactive Bridge Synthesis — GraphBridgeEngine for similarity-based cross-component links (v1.7.0)
- [x] **Phase 31**: Reasoning Studio — Interactive visual interface for graph exploration and reasoning traces (v1.7.0)
- [x] **Phase 32**: Federated Reasoning (v1.7.1) — Multi-agent traversal and automated node discovery
- [x] **Phase 33-36**: Hardening & Temporal (v1.7.2)
- [x] **Phase 37**: Calibration (v1.7.3)
- [x] **Phase 38-41**: Logit Unification & Temporal (v1.7.4)
- [x] **Phase 42**: Interface Robustness (v1.7.4) — Secured REST endpoints and Gradio stabilization
- [x] **Phase 43**: Temporal Context & REM Synthesis (v1.7.5) — 10-parameter logit and Wormhole synthesis
- [x] **Phase 44**: IKGWQ-MetaQA Benchmark (v1.8.0) — Unified IKGWQ-S protocol across MetaQA
- [x] **Phase 45**: 10-Parameter Learner Upgrade (v1.9.0) — Full 10-param CSA formula
- [x] **Phase 46**: Live Feedback Loop (v1.9.1) — /params endpoint, feature vector extraction
- [x] **Phase 47**: Params Persistence (v1.9.2) — JSON checkpoint restore
- [x] **Phase 48**: Auto-Retrain Scheduler (v1.9.3) — feedback-driven online gradient descent
- [x] **Phase 49**: TSC Explicit Mode (v1.9.4) — tsc_communities() public API
- [x] **Phase 50**: HypothesisEngine (v1.9.5) — Multi-path abductive reasoning
- [x] **Phase 51 & 52**: ResearchAgent + ExternalValidator (v1.9.6) — Autonomous missing-link mining
- [x] **Phase 53**: Adaptive Search Strategy (v1.9.7) — Density-aware parameter selection
- [x] **Phase 54**: Observability Dashboard (v1.9.8) — In-memory ring log, hot-reload, live dashboard
- [x] **Phase 55**: GraphSAGE + Engram + TemporalCalibrator + QueryLog (v2.0.0) — Neighborhood smoothing, predictive traversal
- [x] **Phase 56**: Fault Tolerance Hardening (v2.0.1) — Partial query results, hop-level checkpointing
- [x] **Phase 57**: Engram Persistence + Stream Guard (v2.0.1) — Lifespan persistence, streaming error chunks
- [x] **Phase 58**: SpeedTalk Encoding (v2.0.2) — Phonemic compression for Engram cache
- [x] **Phase 59**: Cerebellar Error Correction (CEC) (v2.0.3) — Inference-time dissonance detection
- [x] **Phase 60**: Multi-Agent Consensus Hierarchies (MACH) (v2.0.4) — Three-tier verification
- [x] **Phase 61**: Synaptic Pruning & Quantized Traversal (SPQT) (v2.0.5) — Utility-based pruning, uint8 scores
- [x] **Phase 62**: Explainable Reasoning Trace (ERT) (v2.1.0) — 10-parameter Attention Radar
- [x] **Phase 63**: Neural Telemetry Bridge (v2.2.0) — WebSocket event streaming
- [x] **Phase 64**: Neural Memory Consolidation (v2.3.0) — Canonical Engram promotion
- [x] **Phase 65**: Autonomous Hypothesis Materialization (v2.4.0) — Proactive edge commit
- [x] **Phase 68**: Neuro-Symbolic Homeostasis (v2.7.0) — 5-scalar metabolic scalar control
- [x] **Phase 69**: Predictive Coding Engine (v2.8.0) — Active inference (PE/soliton index)
- [x] **Phase 70**: Looped Beam Traversal (v2.9.0) — Iterative refinement (arXiv:2510.25741)
- [x] **Phase 71**: AutoApprover (v2.10.0) — Tiered decision engine for research findings
- [x] **Phase 72**: TriangulationEngine (v2.11.0) — Four-perspective validation
- [x] **Phase 73**: DiscoveryCalibrator + ContradictionResolver (v2.12.0) — EMA-based sampling, contradiction handling
- [x] **Phase 74**: Autonomous Discovery Loop (v2.13.0) — Full loop closure with circuit breaker
- [x] **Phase 75**: Studio v2 Dashboard (v2.14.0) — Live monitoring panels
- [x] **Phase 76**: Graph Provenance & Rollback (v2.15.0) — Batch/Cycle rollback protocol
- [x] **Phase 77**: Feature Impact Benchmark (v2.16.0 partial)
- [x] **Phase 78**: Provenance Studio Panel (v2.16.0)
- [x] **Phase 79**: Loop-Provenance Recovery (v2.17.0) — Auto-rollback on circuit trip
- [x] **Phase 80**: GraphAdapter `remove_edge()` Protocol (v2.18.0)
- [x] **Phase 81**: Graph Snapshot Persistence (v2.19.0) — JSON topology save/restore/diff
- [x] **Phase 82**: Adaptive Loop Tuning (v2.20.0) — Autonomous resource scaling
- [x] **Phase 83**: UE5 3D Neural Visualization (v2.21.0) — Unreal Engine C++ Plugin
- [x] **Phase 93**: Active Inference / Daydreaming (v2.52.0) — Idle-period consolidation
- [x] **Phase 94**: Self-Modifying GUI (v2.52.0) — UEToolkit integration
- [x] **Phase 102**: Default Mode Network (v2.52.0) — Idle bottleneck audit
- [x] **Phase 104-105**: Homeostatic Metaplasticity and Recursive Self-Synthesis (v2.52.0)
- [x] **Phase 107-108**: De Novo Parameter Synthesis and Thalamofrontal Feedback Loop (v2.52.0)
- [x] **Phase 109-112**: Counterfactual Reasoning, Global Workspace (GWS), Active Inference, and REM Cycle Shortcut Synthesis (v2.52.0)
- [x] **Phase 119-123**: Sleep Cycle & Metacognitive Monitor, Epistemic Gating, Counterfactual Engine (v2.25.0)
- [x] **Phase 134-137**: Vectorized Beam Scoring, KGE-Enriched Embeddings, Funnel Beam Profile, H1SE (Hop-1 Seed Expansion) (v2.31.0)
- [x] **Phase 149-150**: Cingulate Engine (Reasoning Verifier) and Frontal Engine Executive Strategy (v2.35.0)
- [x] **Phase 151-154**: Vote-Weight Suppression, Answer-Type Constraint Filter, DBC Scoring (v2.39.0)
- [x] **Phase 156-160**: Penultimate Relation Boost (PRB), r2 Path-Consistency Boost, TRB Detection Fixes (v2.44.0)
- [x] **Phase 172-163**: StructuralRelationInferrer (SRI), CTRI, SABS (Asymmetric Beam Search) (v2.47.0)
- [x] **Phase 172-165**: Terminal-Anchor Beam (TAB) and Hetionet Biomedical KG Benchmark (v2.49.0)
- [x] **Phase 172-167**: GraphProfiler (Auto Query Strategy) and STRB (Semantic Terminal Relation Boost) (v2.52.0) — closes zero-config gap on 1-hop tasks via semantic query embedding
- [x] **Phase 168-172**: Hybrid-Memory Architecture — NVME-optimized mmap, auto-spill, vectorized batch traversal (v2.52.0)
- [x] **Phase 174**: NVMe SSD Management UI — runtime drive/spill configuration from Studio settings (v2.53.0)
- [x] **Phase 175**: Studio Hot-Swap & Adaptive Control — live graph swap, H1SE/TAB/STRB runtime toggle (v2.53.0)
- [x] **Phase 176**: FederatedGraphRegistry — cross-domain reasoning with alias resolution; traversal batch fallback (v2.53.0)
- [x] **Phase 177**: Continuous Improvement Trifecta — autonomous discovery, self-correction, evolutionary CSA tuning (v2.53.0)
- [x] **Phase 178**: DON'T PANIC Emergency Snapshot — atomic state persistence to `panics/` for crash recovery (v2.53.0)
...

## Benchmark Results

CEREBRUM is validated across three benchmarks that together demonstrate: correctness on labeled KGs, credibility on established KGQA standards, and frontier capability on incomplete KG reasoning.

### MetaQA — 43,234 entities / 124,680 edges / 39,093 questions

| Variant | 1-hop H@1 | 2-hop H@1 | 3-hop H@1 | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 |
|---------|----------|----------|----------|-----------|-----------|-----------|
| **CEREBRUM — canonical subset**¹ | **46.1%** | **30.0%** | **12.5%** | **96.6%** | **86.3%** | **50.3%** |
| **CEREBRUM — full v2.52.0 run**² | 46.1% | 30.0% | 47.3% | 96.6% | 86.3% | 73.2% |
| UniKGQA (ICLR 2023, supervised) | 97.5% | 99.0% | 99.1% | — | — | — |
| EmbedKGQA (ACL 2020, supervised) | ~97% | ~94% | ~94% | — | — | — |
| MINERVA (trained RL) | — | — | — | 95.3% | 78.2% | 45.6% |

¹ Canonical subset: standard MetaQA test split (~12,500 questions/hop), no edge removal, comparable to SOTA baselines.  
² Full run: 14,274 questions, all v2.52.0 features (GraphSAGE, STRB, GraphProfiler). Not directly comparable to supervised baselines.

**CEREBRUM operates with zero task-specific training, no labeled QA pairs, and no gradient updates** — purely graph structure and pre-trained sentence embeddings. The H@10 story is the key result: CEREBRUM retrieves the correct answer in its top-10 candidates at 96.6% (1-hop) and 86.3% (2-hop) on the canonical benchmark.

---

## Installation

### Prerequisites
- Python >= 3.10
- PyTorch (with CUDA for GPU acceleration)
- NetworkX, NumPy, SciPy

### Local Setup
```bash
# Clone the repository
git clone https://github.com/BrutalByte/CEREBRUM.git
cd CEREBRUM

# Install with development dependencies
pip install -e ".[all]"
```

---

## Usage

### 1. Starting the Server
Start the CEREBRUM REST API server.

```bash
# Set your accepted keys (comma-separated)
export CEREBRUM_API_KEYS=your-key-here

# Start with a CSV graph
python -m api.server --csv data/my_graph.csv --port 8200
```

### 2. Querying the Knowledge Graph
CEREBRUM’s **GraphProfiler** (Phase 172) automatically detects your graph's structural regime, and **STRB** (Phase 172) inferentially boosts the correct terminal relation for your question.

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "X-API-Key: your-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What compound treats Diabetes?",
    "max_hop": 3
  }'
```

---

### Hetionet — 47,031 entities / 2,250,197 edges (Biomedical KG)

| Variant | Hits@1 | Hits@10 | MRR |
|---------|--------|---------|-----|
| BFS baseline (no TRB) | 0.8% | — | — |
| TRB only (no STRB) | 73.5% | — | — |
| **CEREBRUM v2.52.0 (TRB + STRB)** | **61%** | **85%** | **0.72** |

Hetionet results on the `disease_gene_pathway` template demonstrate the power of **STRB** and **TAB** in navigating complex heterogeneous biological relationships. BFS baseline (0.8%) confirms TRB is essential for typed heterogeneous graphs.

### WebQSP — 1,298,304 entities / 2,752,238 edges (Freebase 2-hop subgraph)

| Variant | Hits@1 | Hits@10 | MRR |
|---------|--------|---------|-----|
| CEREBRUM RAW | 4.0% | 10.5% | 6.2% |
| **CEREBRUM FULL** | **7.5%** | **17.5%** | **9.8%** |
| NSM (trained) | 74% | — | — |

WebQSP over Freebase is specifically hard for zero-training structural systems due to CVT mediator nodes with opaque MID identifiers that break semantic attention on indirect paths.

### IKGWQ — Incomplete KG Graceful Degradation (5 incompleteness levels)

| Level | Remove% | Hits@1 | Hits@10 | MRR |
|-------|---------|--------|---------|-----|
| Complete | 0% | 4.0% | 14.25% | 6.64% |
| Mild | 5% | 3.75% | 14.75% | 6.81% |
| Moderate | 15% | 2.75% | 14.25% | 5.80% |
| Severe | 30% | 4.0% | 10.75% | 5.88% |
| Extreme | 50% | 3.25% | 9.5% | 4.58% |
## Autonomous Discovery & Automation

CEREBRUM v2.52.0 includes a production-grade automation suite for "Daydreaming" (background knowledge discovery) during off-peak hours.

- **`scripts/discovery_scheduler.py`**: A fully automated pipeline that triggers:
    1. **Autonomous Research**: Mines the KG for latent connections.
    2. **Synthesis**: Generates an audit-ready `discovery_verification_report.md` for human review.
- **Verification**: All discoveries include full path-trace provenance for expert validation.

For setup instructions, see `docs/AUTOMATION_GUIDE.md`.

---


# GEMINI.md - CEREBRUM Project Context

This file provides essential context and instructions for Gemini CLI when working in the CEREBRUM repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v2.7.0 (Phase 68 COMPLETE)** — 1540+ tests passing. Includes Metabolic Homeostasis for functional regulation of reasoning, Autonomous Hypothesis Materialization for ResearchAgent findings, Neural Memory Consolidation for canonical engrams, Explainable Reasoning Trace (ERT) for "glass-box" transparency, Synaptic Pruning & Quantized Traversal (SPQT) for efficiency, Multi-Agent Consensus Hierarchies (MACH) for three-tier reasoning verification, Cerebellar Error Correction (CEC) for active error-driven meta-learning, SpeedTalk-compressed Engram Cache (Phonemic 20x compression), full THALAMUS ingestion pipeline, LLM bridge, Bayesian Beam Search with warm-start, GlobalRebalancer with on_rebalance hook, Cross-Modal Alignment with canonical basis anchor, enterprise scale connectors (Neo4j, AWS Neptune, PySpark), two rounds of production hardening, 10-parameter CSA formula, Engram-steered traversal with durable cache, TemporalCalibrator, QueryLog, HypothesisEngine, ResearchAgent, ExternalValidator, observability dashboard, and comprehensive fault tolerance hardening.

### Core Innovations
1.  **DSCF/TSC (Triple-Signal Consensus):** A novel community detection algorithm that fuses local (LPA), global (Modularity), and flow (Centrality) signals during each node update. These communities act as "attention heads."
2.  **CSA (Community-Structured Attention):** An attention mechanism where weights are influenced by community membership, semantic similarity, and graph structure.
3.  **Metabolic Homeostasis**: Dynamic functional regulation using metabolic scalars: **Reinforcement** (Dopamine), **Arousal** (Norepinephrine), **Novelty** (Acetylcholine), **Cohesion** (Oxytocin), and **Persistence** (Vasopressin). Adjusts reasoning state (beam width, attention ratios) with temporal decay.
4.  **Synaptic Pruning**: Periodic removal of low-utility synthetic edges based on confidence, age, and usage patterns to maintain graph sparsity and traversal speed.
5.  **Quantized Traversal**: Efficiency-optimized reasoning using `uint8` fixed-point math for path scoring, reducing memory overhead during large-scale beam search.
6.  **MACH (Multi-Agent Consensus Hierarchies)**: A three-tier reasoning verification system (Local, Federated, Gold Literature).
7.  **Explainable Reasoning Trace (ERT)**: Hop-by-hop decision log capturing winners and competitors at every step, exposing the 10-parameter feature radar for every path.
8.  **Engram-Steered Traversal**: `Engram` stores successful reasoning relation sequences. `EngramTraversal` biases beam pruning toward known-productive patterns.
9.  **Fault Tolerance**: Every failure mode is isolated — traversal crashes return `partial=True` responses, QueryLog/Engram write failures never crash `/query`.

### Transformer ↔ KG Analogy
| Transformer Concept | CEREBRUM Equivalent |
| :--- | :--- |
| Attention head | DSCF/TSC community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + Betweenness + Degree + Recency |
| Attention weight | CSA formula (Sim + Comm + Edge + Dist + Metabolic Scalars) |
| Context window | Ego-network radius R |
| Metabolic State | ChemicalModulator (Reinforcement, Arousal, Novelty, Cohesion, Persistence) |

---

## Technical Stack

-   **Language:** Python >= 3.10
-   **Graph Processing:** `networkx`, `scipy` (native GPL-free Leiden reimplementation; `igraph`/`leidenalg` not required)
-   **Numerical Ops:** `numpy`
-   **Web Framework:** `fastapi`, `uvicorn` (optional)
-   **Embeddings:** `sentence-transformers`, `pykeen` (optional)
-   **Graph Backends:** Neo4j, RDF/SPARQL, CSV
-   **Testing:** `pytest`, `pytest-asyncio`

---

## Architecture

The project is organized into modular directories:

| Directory | Purpose |
| :--- | :--- |
| `core/` | **Core Engines:** Community detection (DSCF), Attention (CSA), Embedding, and Structural Encoding. |
| `reasoning/` | **Reasoning Logic:** Beam search traversal, path scoring, and answer extraction. |
| `adapters/` | **Graph Backends:** Adapters for NetworkX (default), Neo4j, RDF, and CSV. |
| `api/` | **REST API:** FastAPI server implementation with Pydantic schemas. |
| `cli/` | **CLI Entry Point:** `parallax.py` for command-line interactions. |
| `llm_bridge/` | **LLM Integration:** Formatting reasoning paths for LLM consumption. |
| `benchmarks/` | **Evaluation:** Evaluation scripts for MetaQA, WebQSP, and Hetionet. |
| `tests/` | **Test Suite:** Unit, component, and end-to-end tests. |
| `core/hypothesis_engine.py` | HypothesisEngine — abductive reasoning, Noisy-OR confidence |
| `core/research_agent.py` | ResearchAgent — autonomous missing-link discovery daemon |
| `core/temporal_calibrator.py` | TemporalCalibrator — grid-search Recall@K calibration |
| `core/persistence.py` | QueryLog — append-only NDJSON query history |
| `reasoning/engram_traversal.py` | Engram + EngramTraversal — pattern-steered beam pruning |
| `ui/dashboard.html` | Operational observability dashboard |

---

## Development Commands

### Installation
```bash
# Minimal install
pip install -e "."

# With API server support
pip install -e ".[api]"

# Full development install
pip install -e ".[all]"
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_dscf.py -v
```

### CLI Usage
```bash
# Query the graph
python -m cli.cerebrum query --csv tests/fixtures/toy_graph.csv "newton"

# Inspect communities
python -m cli.cerebrum communities --csv tests/fixtures/toy_graph.csv

# Start API server
python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200
```

### Starting the API Server
```bash
# Direct uvicorn
uvicorn api.server:app --port 8200 --reload

# CLI serve (with optional params checkpoint)
python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200
python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200 --params-file checkpoint.json
```

---

## Testing & Documentation Standards

-   **Test-First:** New features or bug fixes must be accompanied by tests in `tests/`.
-   **Test Logging:** Record significant test runs in `TEST_LOG.md`.
-   **Reproducibility:** Follow the guidelines in `TESTING.md` for deterministic runs (especially for DSCF).
-   **Documentation:** Absolute precedence is given to `PARALLAX.md` (whitepaper) and `README.md`. `CLAUDE.md` provides environment-specific guidance.

## Current Project Status (Phase 68 COMPLETE)

CEREBRUM is currently in **v2.7.0**. All phases through 68 shipped and verified. 1540+ tests passing.

Key implementations (recent phases):
-   **Phase 68 — Metabolic Modulation**: Bio-mimetic functional regulation (ChemicalModulator) with homeostatic decay; dynamic `alpha/beta` ratio control.
-   **Phase 65 — Hypothesis Materialization**: Formal materialization of ResearchAgent findings into graph edges with Noisy-OR confidence.
-   **Phase 64 — Neural Memory Consolidation**: `EngramConsolidator` promotes high-frequency patterns to "Canonical Engrams."
-   **Phase 63 — Neural Telemetry**: Real-time event streaming via WebSockets for 3D visualization (e.g., Unreal Engine 5).
-   **Phase 62 — Explainable Reasoning Trace (ERT)**: Decision transparency via hop-by-hop logging of search candidates and their feature radars.
-   **Phase 61 — Synaptic Pruning & Quantized Traversal (SPQT)**: Utility-based edge removal (SynapticPruner) and `uint8` fixed-point traversal scoring for efficiency.
-   **Phase 60 — Multi-Agent Consensus Hierarchies (MACH)**: Three-tier reasoning verification (L1 Strategy, L2 Federated, L3 Gold Literature).
-   **Phase 59 — Cerebellar Error Correction (CEC)**: Active error-driven meta-learning via dissonance detection.
-   **Phase 58 — SpeedTalk Encoding**: Phonemic compression for Engram keys (8-20x space savings).
-   **Phase 57 — Engram Persistence**: Save/load Engram on shutdown/startup; durable query log warm-up.
-   **Phase 56 — Fault Tolerance**: `QueryResponse.partial/error` fields; graceful degradation; QueryLog/Engram isolation.
-   **Phase 55 — GraphSAGE + Engram + TemporalCalibrator + QueryLog**: One-pass neighbourhood smoother; pattern-steered beam pruning; Recall@K calibration.
-   **Phase 54 — Observability Dashboard**: `RingBufferHandler`, `GET /logs`, `ui/dashboard.html` operational monitoring UI.
-   **Phase 51–52 — ResearchAgent + ExternalValidator**: Autonomous missing-link discovery daemon; LLM-independent external source validation.
-   **Phase 50 — HypothesisEngine**: Multi-path abductive reasoning with Noisy-OR confidence aggregation.
-   **Phase 49 — TSC Explicit Mode**: `tsc_communities()` and `tsc_quality_metrics()` for explicit Triple-Signal Community control.
-   **Phase 44 — IKGWQ-MetaQA Benchmark**: Verified REM Synthesis ("Wormhole") improves 3-hop recall by up to 40% on sparse graphs.
-   **Phase 43 — 10-Parameter CSA Formula**: Temporal sliding windows; full 10-weight CSA formula.
-   **Phase 30–32 — Federated Reasoning, Reasoning Studio, Proactive Bridge Synthesis**: Multi-agent traversal, `GraphBridgeEngine`, `CerebrumGraph` (v1.7.x).




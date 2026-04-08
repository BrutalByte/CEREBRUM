# GEMINI.md - CEREBRUM Project Context

This file provides essential context and instructions for Gemini CLI when working in the CEREBRUM repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v2.0.1 (Phase 57 COMPLETE)** — 1490+ tests passing. Includes full THALAMUS ingestion pipeline, LLM bridge, Bayesian Beam Search with warm-start, GlobalRebalancer with on_rebalance hook, Cross-Modal Alignment with canonical basis anchor, enterprise scale connectors (Neo4j, AWS Neptune, PySpark), two rounds of production hardening, 10-parameter CSA formula, Engram-steered traversal with durable cache, TemporalCalibrator, QueryLog, HypothesisEngine, ResearchAgent, ExternalValidator, observability dashboard, and comprehensive fault tolerance hardening.

### Core Innovations
1.  **DSCF (Dual-Signal Community Fusion):** A novel community detection algorithm that fuses local (Label Propagation) and global (Modularity) signals during each node update. These communities act as "attention heads."
2.  **CSA (Community-Structured Attention):** An attention mechanism where weights are influenced by community membership, semantic similarity, and graph structure.
3.  **10-Parameter CSA Formula**: The attention weight formula now has 10 learnable parameters covering semantic similarity, community score, edge-type weight, distance penalty, hop decay, PageRank prior, temporal decay, node recency, synthesis-density penalty, and grounding confidence. Online learning via `POST /feedback` (SGD) and batch retraining via `POST /retrain`.
4.  **Engram-Steered Traversal**: `Engram` stores successful reasoning relation sequences. `EngramTraversal` biases beam pruning toward known-productive patterns. Cache persists to disk across restarts.
5.  **Fault Tolerance**: Every failure mode is isolated — traversal crashes return `partial=True` responses, QueryLog/Engram write failures never crash `/query`, GlobalRebalancer has a top-level crash guard.

### Transformer ↔ KG Analogy
| Transformer Concept | CEREBRUM Equivalent |
| :--- | :--- |
| Attention head | DSCF community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + Betweenness + Degree |
| Attention weight | CSA formula (Sim + Comm + Edge + Dist) |
| Context window | Ego-network radius R |

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

## Current Project Status (Phase 57 COMPLETE)

CEREBRUM is currently in **v2.0.1**. All phases through 57 shipped and verified. 1490+ tests passing.

Key implementations (recent phases):
-   **Phase 57 — Engram Persistence**: Save/load Engram on shutdown/startup; `/query/stream` error chunk; `ProcessPoolExecutor` sequential fallback for fault-isolated rebalancing.
-   **Phase 56 — Fault Tolerance**: `QueryResponse.partial/error` fields; `_partial_paths` checkpoint on mid-traversal crash; graceful degradation throughout; `GlobalRebalancer` top-level crash guard.
-   **Phase 55 — GraphSAGE + Engram + TemporalCalibrator + QueryLog**: `smooth_with_graphsage()` one-pass neighbourhood smoother; `Engram` + `EngramTraversal` pattern-steered beam pruning; `TemporalCalibrator` grid-search Recall@K calibration; `QueryLog` append-only NDJSON history.
-   **Phase 54 — Observability Dashboard**: `RingBufferHandler`, `GET /logs`, `ui/dashboard.html` operational monitoring UI.
-   **Phase 53 — Adaptive Search by Graph Density**: Local graph density detection auto-adjusts beam width and candidate pruning strategy.
-   **Phase 51–52 — ResearchAgent + ExternalValidator**: Autonomous missing-link discovery daemon; LLM-independent external source validation; `/research/*` endpoints.
-   **Phase 50 — HypothesisEngine**: Multi-path abductive reasoning with Noisy-OR confidence aggregation; `/hypothesize` endpoint.
-   **Phase 49 — TSC Explicit Mode**: `tsc_communities()` and `tsc_quality_metrics()` for explicit Triple-Signal Community control.
-   **Phase 44 — IKGWQ-MetaQA Benchmark**: Verified REM Synthesis ("Wormhole") improves 3-hop recall by up to 40% on sparse graphs (Level 4, 50% edge removal).
-   **Phase 43 — 10-Parameter CSA Formula**: Temporal sliding windows; `ReasoningLogit` with Synthesis Density (`sd`); full 10-weight CSA formula.
-   **Phase 30–32 — Federated Reasoning, Reasoning Studio, Proactive Bridge Synthesis**: Multi-agent traversal, `GraphBridgeEngine`, `CerebrumGraph` (v1.7.x).




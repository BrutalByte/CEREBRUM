# GEMINI.md - CEREBRUM Project Context

This file provides essential context and instructions for Gemini CLI when working in the CEREBRUM repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v1.4.0 (Phase 24 COMPLETE)** — 1042 tests passing. Includes full THALAMUS ingestion pipeline, LLM bridge, Bayesian Beam Search with warm-start, GlobalRebalancer with on_rebalance hook, Cross-Modal Alignment with canonical basis anchor, enterprise scale connectors (Neo4j, AWS Neptune, PySpark), and two rounds of production hardening.

### Core Innovations
1.  **DSCF (Dual-Signal Community Fusion):** A novel community detection algorithm that fuses local (Label Propagation) and global (Modularity) signals during each node update. These communities act as "attention heads."
2.  **CSA (Community-Structured Attention):** An attention mechanism where weights are influenced by community membership, semantic similarity, and graph structure.

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
-   **Graph Processing:** `networkx`, `igraph`, `leidenalg`, `scipy`
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

---

## Development Commands

### Installation
```bash
# Full development install
pip install -e ".[all]"

# Minimal install
pip install -e "."
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
uvicorn api.server:app --port 8200 --reload
```

---

## Testing & Documentation Standards

-   **Test-First:** New features or bug fixes must be accompanied by tests in `tests/`.
-   **Test Logging:** Record significant test runs in `TEST_LOG.md`.
-   **Reproducibility:** Follow the guidelines in `TESTING.md` for deterministic runs (especially for DSCF).
-   **Documentation:** Absolute precedence is given to `PARALLAX.md` (whitepaper) and `README.md`. `CLAUDE.md` provides environment-specific guidance.

## Current Project Status (Phase 42 COMPLETE)

CEREBRUM is currently in **v1.7.4**. All phases through 42 shipped and verified. This release focused on interface robustness, REST API hardening, and automated headless testing.

Key implementations:
-   **Phase 42 — Interface Robustness & Hardening**: Stabilized Reasoning Studio with `gr.Progress`, secured REST endpoints, and implemented automated headless robustness tests for both UI and API.
-   **Phase 41 — Temporal & REM Synthesis**: Corrected reversed recency bias, integrated Node Recency (9-feature logit), and implemented cross-component "Wormhole" detection in REMEngine.
-   **Phase 38 — Reasoning Hardening**: Unified `ReasoningLogit` signal pipeline.
-   **Phase 39 — Async Bridge Synthesis**: Decoupled `BridgeTwinEngine` and `InsightEngine` via `TaskQueue`.
-   **Phase 40 — IKGWQ Hardening**: Full system validation under 50% edge removal (v1.7.2).
-   **Phase 37 — Calibration**: Entropy-based self-doubt and confidence adjustment.
-   **Phase 34 — Symbolic Guardrails**: Hard-logic integrity constraints for path pruning.
-   **Phase 33 — Temporal Reasoning**: Integration of temporal distance into structural encoding.
-   **Phase 30 — Proactive Bridge Synthesis**: `GraphBridgeEngine` and `CerebrumGraph` (1.7.0).
-   **Phase 31 — Reasoning Studio**: Interactive visual reasoning traces (1.7.0).
-   **Phase 32 — Federated Reasoning**: Multi-agent traversal and automated node discovery (1.7.1).




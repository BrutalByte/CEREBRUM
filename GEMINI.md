# GEMINI.md - Parallax Project Context

This file provides essential context and instructions for Gemini CLI when working in the Parallax repository.

## Project Overview

**Parallax** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v0.2.0 (Federated)** enables distributed reasoning across sensitive datasets using holographic indexing and privacy-preserving search.

### Core Innovations
1.  **DSCF (Dual-Signal Community Fusion):** A novel community detection algorithm that fuses local (Label Propagation) and global (Modularity) signals during each node update. These communities act as "attention heads."
2.  **CSA (Community-Structured Attention):** An attention mechanism where weights are influenced by community membership, semantic similarity, and graph structure.

### Transformer ↔ KG Analogy
| Transformer Concept | Parallax Equivalent |
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
python -m cli.parallax query --csv tests/fixtures/toy_graph.csv "newton"

# Inspect communities
python -m cli.parallax communities --csv tests/fixtures/toy_graph.csv

# Start API server
python -m cli.parallax serve --csv tests/fixtures/toy_graph.csv --port 8200
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

## Current Project Status (Phase 4)

The project has completed **Phase 3 (Adapters & API)**.
Current focus is **Phase 4: Benchmarking**, specifically implementing and running evaluations on:
-   **MetaQA** (3-hop reasoning)
-   **WebQSP** (Freebase reasoning)
-   **Hetionet** (Biomedical KG)

Refer to `TEST_LOG.md` for the latest benchmarking results and engineering findings (EF-001 to EF-004).




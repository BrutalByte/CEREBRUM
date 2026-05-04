# GEMINI.md - CEREBRUM Project Context

This file provides essential context and instructions for Gemini CLI when working in the CEREBRUM repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v2.51.0 (Phase 167 COMPLETE)** — 2175+ tests passing. Includes STRB (Semantic Terminal Relation Boost), GraphProfiler (Auto Query Strategy), Terminal-Anchor Hints (TAB), Vectorized Beam Scoring, and all previous advancements.

### Core Innovations
1.  **STRB (Semantic Terminal Relation Boost - Phase 167)**: Zero-config reasoning using query embeddings to automatically identify intended terminal relations.
2.  **GraphProfiler (Phase 166)**: O(E) build-time topology analysis for automatic per-query strategy selection (Regime-based auto-config).
3.  **TAB (Terminal-Anchor Boost - Phase 164)**: Penultimate-hop biasing for 3+ hop queries in heterogeneous graphs.
4.  **DSCF/TSC (Triple-Signal Consensus):** A novel community detection algorithm that fuses local (LPA), global (Modularity), and flow (Centrality) signals.
5.  **CSA (Community-Structured Attention):** An attention mechanism influenced by community membership and semantic similarity.
6.  **H1SE (Hop-1 Intermediate Seed Expansion - Phase 137):** Eliminates cross-branch beam competition by giving each hop-1 entity its own independent deep traversal.
7.  **Thalamofrontal Feedback Loop (Phase 108):** Dynamic metabolic gating of reasoning paths. Prunes "thermal waste" by tightening the attention gate when search quality is high.
8.  **Sleep Cycle (Phases 119-121):** Self-organizing idle cycles for Engram Consolidation, REM Shortcut Synthesis, and Synaptic Decay.
9.  **Global Workspace (GWS - Phase 110)**: Blackboard-based competitive attention layer for cognitive flexibility.
10. **Metabolic Homeostasis**: Dynamic functional regulation using metabolic scalars: Reinforcement, Arousal, Novelty, Cohesion, and Persistence.

### Transformer ↔ KG Analogy
| Transformer Concept | CEREBRUM Equivalent |
| :--- | :--- |
| Attention head | DSCF/TSC community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + Betweenness + Degree + Recency |
| Attention weight | CSA formula (Sim + Comm + Edge + Dist + Metabolic Scalars) |
| Context window | Ego-network radius R |
| Metabolic State | ChemicalModulator (Reinforcement, Arousal, Novelty, Cohesion, Persistence) |
| Evolution | Metaplastic Recursive Self-Synthesis |

---

## Technical Stack

-   **Language:** Python >= 3.10
-   **Graph Processing:** `networkx`, `scipy`
-   **Numerical Ops:** `numpy`
-   **Web Framework:** `fastapi`
-   **Embeddings:** `sentence-transformers`
-   **Graph Backends:** Neo4j, RDF/SPARQL, CSV, AWS Neptune
-   **Testing:** `pytest`

---

## Architecture

The project is organized into modular directories:

| Directory | Purpose |
| :--- | :--- |
| `core/` | **Core Engines:** Community detection, Attention, Embedding, Structural Encoding, Autonomous Research. |
| `reasoning/` | **Reasoning Logic:** Beam search traversal, path scoring, and answer extraction. |
| `adapters/` | **Graph Backends:** Adapters for NetworkX, Neo4j, RDF, CSV, and Neptune. |
| `api/` | **REST API:** FastAPI server implementation with Pydantic schemas. |
| `cli/` | **CLI Entry Point:** `parallax.py` for command-line interactions. |
| `llm_bridge/` | **LLM Integration:** Formatting reasoning paths for LLM consumption. |
| `benchmarks/` | **Evaluation:** Evaluation scripts for MetaQA, WebQSP, and Hetionet. |
| `tests/` | **Test Suite:** Unit, component, and end-to-end tests. |

---

## Development Commands

### Installation
```bash
pip install -e ".[all]"
```

### Running Tests
```bash
python -m pytest tests/ -v
```

### Self-Optimization (Autoresearch)
```bash
python core/autonomous_researcher.py --continuous --interval 60
```

---

## Testing & Documentation Standards

-   **Test-First:** New features must be accompanied by tests in `tests/`.
-   **Test Logging:** Record significant test runs in `TEST_LOG.md`.
-   **Reproducibility:** Deterministic runs for DSCF/TSC.
-   **Documentation:** Absolute precedence to `PARALLAX.md` and `README.md`.

## Current Project Status (Phase 167 COMPLETE)

CEREBRUM is currently in **v2.51.0**. All phases through 167 shipped and verified.

Key implementations (recent phases):
-   **Phase 167 — STRB**: Semantic Terminal Relation Boost using query embeddings.
-   **Phase 166 — GraphProfiler**: Automatic Query Strategy Selection via topology profiling.
-   **Phase 165 — Hetionet Benchmark**: 85% accuracy on zero-shot biomedical reasoning.
-   **Phase 164 — TAB**: Terminal-Anchor Hints for penultimate-hop biasing.
-   **Phase 141 — Autonomous H1SE Tuning**: Parameter self-optimization against MetaQA.
-   **Phase 140 — Multi-Seed Interaction**: Intersection queries with shared-neighbor bonus.
-   **Phase 137 — H1SE**: Hop-1 Intermediate Seed Expansion.
-   **Phase 136 — Funnel Beam Profile**: Linearly ramped beam widths for deeper hop coverage.
-   **Phase 135 — KGE-Enriched Embeddings**: TransE/RotatE signals integrated into semantic embeddings.
-   **Phase 134 — Vectorized Beam Scoring**: 10x performance boost via NumPy-vectorized scoring loops.
-   **Phases 124-133 — Causal Accuracy Suite**: Comprehensive causal inference benchmarks and weighting.
-   **Phase 123 — Counterfactual Engine**: Simulation of "what-if" KG state changes.
-   **Phase 122 — Epistemic Gating**: Unified uncertainty model for path pruning.
-   **Phases 119-121 — Sleep Cycle & Metacognitive Monitor**: Integrated self-optimization loop.
-   **Phase 112 — REM Cycle Shortcut Synthesis**: Autonomous synthesis of shortcut edges.
-   **Phase 111 — Active Inference**: Proactive reasoning during idle cycles.
-   **Phase 110 — Global Workspace (GWS)**: Centralized blackboard for multi-agent coordination.
-   **Phase 109 — Counterfactual Reasoning (Early)**: Initial implementation of hypothetical traversal.
-   **Phase 108 — Thalamofrontal Feedback Loop**: Dynamic metabolic gating of reasoning.
-   **Phase 107 — De Novo Parameter Synthesis**: Autonomous activation of dormant features.
-   **Phase 105 — Recursive Self-Synthesis**: System architects its own subroutines.
-   **Phase 104 — Homeostatic Metaplasticity**: Metabolic control of evolution.
-   **Phase 102 — Default Mode Network**: Self-referential idle reasoning.
-   **Phase 60 — MACH**: Multi-Agent Consensus Hierarchies.
-   **Phase 43–45 — 10-Parameter CSA Formula & CSAParameterLearner**.

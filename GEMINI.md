# GEMINI.md - CEREBRUM Project Context

This file provides essential context and instructions for Gemini CLI when working in the CEREBRUM repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph (KG) reasoning. It implements a multi-hop traversal mechanism inspired by Transformer architectures, allowing KGs to perform complex reasoning without an LLM or training data.

**v2.23.0 (Phase 108 COMPLETE)** — 1540+ tests passing. Includes Thalamofrontal Feedback Loop for dynamic gating, De Novo Parameter Synthesis for autonomous cold-starts, Recursive Self-Synthesis for autonomous architectural expansion, Homeostatic Metaplasticity for chemical control of evolution, Structural Mutation for logic self-optimization, metabolic regulation of reasoning, Autonomous Hypothesis Materialization, Neural Memory Consolidation, Explainable Reasoning Trace (ERT), Synaptic Pruning & Quantized Traversal (SPQT), Multi-Agent Consensus Hierarchies (MACH), Cerebellar Error Correction (CEC), SpeedTalk-compressed Engram Cache, full THALAMUS ingestion pipeline, LLM bridge, Bayesian Beam Search, GlobalRebalancer, Cross-Modal Alignment, enterprise connectors (Neo4j, AWS Neptune, PySpark), 10-parameter CSA formula, Engram-steered traversal, TemporalCalibrator, QueryLog, HypothesisEngine, ResearchAgent, ExternalValidator, and observability dashboard.

### Core Innovations
1.  **DSCF/TSC (Triple-Signal Consensus):** A novel community detection algorithm that fuses local (LPA), global (Modularity), and flow (Centrality) signals.
2.  **CSA (Community-Structured Attention):** An attention mechanism influenced by community membership and semantic similarity.
3.  **Thalamofrontal Feedback Loop (Phase 108):** Dynamic metabolic gating of reasoning paths. Prunes "thermal waste" by tightening the attention gate when search quality is high. Inspired by **ALARM Theory** (Ruhr University Bochum, 2025) and human thalamofrontal loop research (Zhang et al., 2025).
4.  **De Novo Parameter Synthesis (Phase 107):** Autonomous "Cold-Start" mechanism. The system identifies dormant (`0.0`) parameters and jump-starts them to activate new reasoning pathways without manual intervention.
5.  **Recursive Self-Synthesis (Phase 105)**: Phase 105. System architects its own subroutines (e.g., `StructuralEntropyPruner`) based on DMN bottleneck audits.
6.  **Homeostatic Metaplasticity (Phase 104)**: Phase 104. Chemical control of the self-improvement process via Arousal (mutation rate) and Reinforcement (commit gates).
7.  **Metabolic Homeostasis**: Dynamic functional regulation using metabolic scalars: Reinforcement, Arousal, Novelty, Cohesion, and Persistence.
8.  **Synaptic Pruning**: Periodic removal of low-utility synthetic edges to maintain sparsity.
9.  **Quantized Traversal**: Efficiency-optimized reasoning using `uint8` fixed-point math.
10. **MACH (Multi-Agent Consensus Hierarchies)**: A three-tier reasoning verification system.
11. **Explainable Reasoning Trace (ERT)**: Hop-by-hop decision log exposing the 10-parameter feature radar.
12. **Engram-Steered Traversal**: Memory-based biasing of beam pruning toward successful patterns.

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
| `core/autonomous_researcher.py` | AutonomousResearcher — recursive logic synthesis daemon |
| `core/hypothesis_engine.py` | HypothesisEngine — abductive reasoning, Noisy-OR confidence |
| `core/research_agent.py` | ResearchAgent — autonomous missing-link discovery daemon |
| `core/persistence.py` | QueryLog & GraphSnapshot — durability layer |
| `ui/dashboard.html` | Operational observability dashboard |

---

## Development Commands

### Installation
```bash
pip install -e ".[all]"
```

### Running Tests
```bash
pytest tests/ -v
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

## Current Project Status (Phase 108 COMPLETE)

CEREBRUM is currently in **v2.23.0**. All phases through 108 shipped and verified. 1540+ tests passing.

Key implementations (recent phases):
-   **Phase 108 — Thalamofrontal Feedback Loop**: Dynamic metabolic gating inspired by ALARM theory.
-   **Phase 107 — De Novo Parameter Synthesis**: Autonomous cold-start for dormant parameters.
-   **Phase 105 — Recursive Self-Synthesis**: System architects its own subroutines based on bottleneck audits.
-   **Phase 104 — Homeostatic Metaplasticity**: Chemical control of evolution via arousal and reinforcement signals.
-   **Phase 102 — Default Mode Network**: Self-referential idle reasoning for graph auditing and hypothesis generation.
-   **Phase 83 — UE5 3D Neural Visualization**: Production Unreal Engine 5 C++ plugin for live KG exploration.
-   **Phase 82 — Adaptive Loop Tuning**: Calibration-driven scaling of materialization caps.
-   **Phase 81 — Graph Snapshot Persistence**: Portable JSON serialization of graph topology.
-   **Phase 74 — Autonomous Discovery Loop**: Closes the loop from discovery to materialization.
-   **Phase 71 — AutoApprover**: Three-tier approval stack for discovery findings.
-   **Phase 70 — Looped Beam Traversal**: Iterative refinement of reasoning paths.
-   **Phase 69 — Predictive Coding**: Active inference loop using Engram priors.
-   **Phase 68 — Metabolic Modulation**: Bio-mimetic functional regulation (ChemicalModulator).
-   **Phase 64–65 — Engram Consolidation & Hypothesis Materialization**.
-   **Phase 62 — Explainable Reasoning Trace (ERT)**: Hop-by-hop transparency.
-   **Phase 61 — Synaptic Pruning & Quantized Traversal**.
-   **Phase 60 — MACH**: Multi-Agent Consensus Hierarchies.
-   **Phase 51–52 — ResearchAgent + ExternalValidator**.
-   **Phase 50 — HypothesisEngine**.
-   **Phase 49 — TSC Explicit Mode**.
-   **Phase 43–45 — 10-Parameter CSA Formula & CSAParameterLearner**.




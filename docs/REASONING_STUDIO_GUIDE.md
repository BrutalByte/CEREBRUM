# CEREBRUM Reasoning Studio — User Guide

**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Interface**: Gradio web UI + pyvis graph visualization
**Business Logic Module**: `core/studio_engine.py` (`StudioEngine`)
**UI Entry Point**: `ui/studio.py`

---

## Overview

The Reasoning Studio is CEREBRUM's interactive visual interface. It lets you load a Knowledge Graph, run multi-hop reasoning queries, inspect every step of the CSA attention mechanism, and watch community structure form in real time — all without writing code.

This is the "Glass-Box" in action: you can see exactly what the AI is doing at every reasoning step.

As of **Phase 167**, the Studio now supports **Zero-Config Reasoning**:
- **GraphProfiler Integration**: The Studio automatically analyzes your graph upon loading and sets optimal default reasoning parameters.
- **STRB (Semantic Terminal Relation Boost)**: The system uses its query embedding to automatically boost the most relevant terminal relation based on your question text.
- **Vectorized Performance**: Thanks to NumPy-vectorized scoring, the "Reasoning" button now provides near-instant results (sub-30ms) even for complex 3-hop queries.

---

## Installation

```bash
pip install -r ui/requirements.txt
```

Dependencies installed:
- `gradio` — web UI framework
- `pyvis` — interactive graph visualization
- `networkx` — graph backend (already in core)
- `plotly` — for Attention Radar charts

---

## Starting the Studio

```bash
python ui/studio.py
```

The Studio opens at `http://localhost:7860` in your browser. No API server is required — the Studio embeds the full CEREBRUM engine directly via `StudioEngine`.

### Command-line options

```bash
# Specify a graph file to load on startup
python ui/studio.py --csv path/to/graph.csv

# Use semantic embeddings (requires sentence-transformers)
python ui/studio.py --embeddings sentence

# Change port
python ui/studio.py --port 8080

# Share publicly (Gradio tunnel — for demos)
python ui/studio.py --share
```

---

## Architecture: StudioEngine (Phase 54)

All Studio business logic now lives in `core/studio_engine.py`. This is a deliberate architectural separation:

```
ui/studio.py          ← Gradio UI layer (thin shell)
    └── core/studio_engine.py   ← StudioEngine (all business logic)
            ├── load_graph()
            ├── run_query()
            ├── get_community_map()
            ├── get_logs()
            └── build_from_csv()
```

### Benefits
- **Unit testing without a server**: 44+ tests in `tests/test_studio_engine.py` validate all Studio business logic in isolation.
- **Embeddability**: `StudioEngine` can be imported and used in scripts, notebooks, or other UIs without Gradio.
- **Separation of concerns**: UI layout changes do not affect reasoning logic, and vice versa.

### Programmatic use

```python
from core.studio_engine import StudioEngine
from reasoning.trace import ReasoningTrace

engine = StudioEngine()
engine.load_graph("tests/fixtures/toy_graph.csv")
trace = ReasoningTrace(query="Marie Curie", seeds=["marie_curie"])
results = engine.run_query("Marie Curie", max_hops=3, beam_width=5, trace_info=trace)
logs = engine.get_logs(level="INFO", limit=50)
```

---

## Interface Overview

The Studio has four main panels:

### 1. Graph Panel (left)
An interactive force-directed visualization of the loaded Knowledge Graph powered by pyvis.

- **Nodes** are colored by community (each community gets a distinct color)
- **Edges** show relation types as labels
- **Node size** reflects PageRank centrality — important nodes appear larger

### 2. Control Panel (top right)
Load data, set parameters, and run reasoning.

- **Load CSV**: Path to a triple-store CSV file.
- **Max Hops**: How many steps to traverse.
- **Beam Width**: How many candidates to keep at each step.
- **Reasoning Button**: Trigger the Attention Traversal.

### 3. Results Panel (right)
A ranked list of the best reasoning paths found by CEREBRUM.

- **Ranked Paths**: Each path is shown with its total score and breakdown.
- **Explainable Reasoning Trace (ERT)**: (New in Phase 62) An interactive `<details>` block that exposes the internal decision log of the beam search. For each hop, you can see the winners and the top rejected competitors, along with their 10-parameter Attention Radars.
- **Attention Radar**: A 10-feature radar chart (sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding) for the best path.
- **Engram Trace**: A high-density symbolic shorthand for the relation sequences found.

### 4. Community Panel (bottom)
Displays community metrics, sizes, and membership lists.

---

## Studio v2 — Live Monitoring Dashboard (Phases 75 + 78)

Six additional panels are available when optional engines are attached to `StudioEngine`. All panels degrade gracefully when an engine is not attached.

### Attachment API

```python
engine = StudioEngine()
engine.load_graph("my_graph.csv")

# Attach optional engines
engine.attach_research_agent(research_agent)   # panels 1 + 2
engine.attach_modulator(chemical_modulator)    # panel 4
engine.attach_loop(autonomous_loop)            # panel 5
engine.attach_provenance_ledger(ledger)        # panel 6
```

### Panel 1 — AutoApprover Audit Log (`get_auto_approver_audit`)
HTML table of the last N AutoApprover decisions with color-coded action column (APPROVE / REJECT / REVIEW) and a summary row showing approval rate and total decisions.

### Panel 2 — ContradictionResolver Revision Queue (`get_revision_queue`)
HTML list of findings where proposed evidence outweighs existing contradiction — candidates flagged as `revision_candidate`. Shows net evidence score and contradiction weight.

### Panel 3 — DiscoveryCalibrator Heatmap (`get_discovery_heatmap`)
Plotly dual-bar chart: sampling weight (inverse-rate multiplier) and raw discovery rate per community. Communities with no prior scans show at `max_weight` (5.0).

### Panel 4 — ChemicalModulator Blood Panel (`get_chemical_panel`)
Plotly bar + scatter chart of all 5 metabolic scalars (Reinforcement, Arousal, Novelty, Cohesion, Persistence) plotted against their homeostatic baseline. Bars are color-coded green/amber/red by deviation magnitude.

### Panel 5 — Autonomous Loop Panel (`get_loop_panel`)
Three-card status header (running state, circuit breaker status, current approval rate) plus a stacked bar/line cycle history chart showing per-cycle approved/rejected/review counts and cumulative edges added.

### Panel 6 — Provenance Panel (`get_provenance_panel`)
Four-card summary row (total batches, edges recorded, rollback count, cycles seen) plus two charts:
- **Batch bar chart**: horizontal bars for the N most recent materialization batches; green = active, red = rolled back
- **Cycle timeline**: per-cycle edge count (bars) + cumulative edges (dashed line, secondary y-axis)

---

## New in v2.24.0 (Phase 62)

### Explainable Reasoning Trace (ERT)
The "Glass-Box" is now fully transparent. In the Results panel, each query result now includes an **Explainable Reasoning Trace** block. This trace records:
1. **Hop-by-Hop Decision Log**: Which nodes were selected for the beam and which were pruned.
2. **Pruning Justification**: For every pruned candidate, you can see its CSA score and feature vector, allowing you to audit why the system chose one path over another.
3. **Attention Radar Integration**: Feature vectors are captured for all candidates, not just the winners, providing a complete picture of the "Relational Attention" landscape at each step.
4. **Serialization Hardened**: All trace data is serialized to standard Python primitives, ensuring stability in the Gradio interface and the REST API.

---

## Troubleshooting

### "Graph not built"
Ensure you have clicked "Build Graph" or loaded a CSV that triggers an auto-build. Community detection must run before reasoning can start.

### Memory Issues
For very large graphs, reduce the **Beam Width** or **Max Hops**. The Studio is optimized for mesoscale exploration (~10k-50k nodes). For enterprise-scale graphs, use the REST API with a Neo4j or Neptune backend.

### UI Unresponsive
If the UI hangs during community detection, check the console logs. DSCF can be computationally intensive on CPU for dense graphs. Ensure you have enough RAM available.

---
**Reviewed on**: May 3, 2026 for version v2.51.0

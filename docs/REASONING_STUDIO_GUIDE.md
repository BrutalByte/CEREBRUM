# CEREBRUM Reasoning Studio — User Guide

**Version**: v2.4.0
**Interface**: Gradio web UI + pyvis graph visualization
**Business Logic Module**: `core/studio_engine.py` (`StudioEngine`)
**UI Entry Point**: `ui/studio.py`

---

## Overview

The Reasoning Studio is CEREBRUM's interactive visual interface. It lets you load a Knowledge Graph, run multi-hop reasoning queries, inspect every step of the CSA attention mechanism, and watch community structure form in real time — all without writing code.

This is the "Glass-Box" in action: you can see exactly what the AI is doing at every reasoning step.

As of **Phase 62**, the Studio now includes the **Explainable Reasoning Trace (ERT)**, providing a detailed hop-by-hop breakdown of the decision process, including the winners and pruned competitors at each step of the beam search.

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

## New in v2.4.0 (Phase 62)

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

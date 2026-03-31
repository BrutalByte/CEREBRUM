# CEREBRUM Reasoning Studio — User Guide

**Version**: v1.1.0
**Interface**: Gradio web UI + pyvis graph visualization
**Module**: `ui/studio.py`

---

## Overview

The Reasoning Studio is CEREBRUM's interactive visual interface. It lets you load a Knowledge Graph, run multi-hop reasoning queries, inspect every step of the CSA attention mechanism, and watch community structure form in real time — all without writing code.

This is the "Glass-Box" in action: you can see exactly what the AI is doing at every reasoning step.

---

## Installation

```bash
pip install -r ui/requirements.txt
```

Dependencies installed:
- `gradio` — web UI framework
- `pyvis` — interactive graph visualization
- `networkx` — graph backend (already in core)

---

## Starting the Studio

```bash
python ui/studio.py
```

The Studio opens at `http://localhost:7860` in your browser. No API server is required — the Studio embeds the full CEREBRUM engine directly.

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

## Interface Overview

The Studio has four main panels:

### 1. Graph Panel (left)
An interactive force-directed visualization of the loaded Knowledge Graph powered by pyvis.

- **Nodes** are colored by community (each community gets a distinct color)
- **Edges** show relation types as labels
- **Node size** reflects PageRank centrality — important nodes appear larger
- **Click any node** to set it as the query start entity
- **Hover** over an edge to see its CSA attention weight

### 2. Query Panel (top center)
Controls for running reasoning queries:

| Control | Description |
|---|---|
| **Entity** | Starting node for traversal |
| **Max Hops** | Maximum reasoning depth (1–5) |
| **Beam Width** | Candidates per hop (1–50) |
| **Top-K** | Number of answer paths to return |
| **Probabilistic** | Enable Bayesian beam search |
| **Warm-Start** | Beta prior seeding strength (0–10) |

Click **Run Query** to execute. Results appear in the Answer Panel within milliseconds.

### 3. Answer Panel (center)
Displays ranked answer paths with:
- **Entity**: the terminal node of each path
- **Score**: composite CSA path score (0–1)
- **Path**: the full hop-by-hop trace (`A → rel → B → rel → C`)
- **Confidence interval**: when probabilistic mode is on
- **Community trace**: which communities were traversed at each hop

Click any answer row to **highlight the reasoning path** in the Graph Panel.

### 4. Attention Math Panel (right)
When you click an edge in the Graph Panel or an answer path in the Answer Panel, this panel shows the full CSA formula breakdown for that specific edge:

```
a(Marie Curie → Radium, hop=1) = sigmoid(
  0.4 × cos(e_u, e_v) = 0.4 × 0.71  =  0.284
  0.4 × S_C(u,v)      = 0.4 × 1.00  =  0.400   ← same community
  0.1 × w_rel         = 0.1 × 0.95  =  0.095   ← "discovered" weight
  0.05 × d_norm       = 0.05 × 0.10 = -0.005
  0.05 × φ(k)         = 0.05 × 1.00 =  0.050   ← hop 1
  0.1 × PR(v)         = 0.1 × 0.43  =  0.043
                        ─────────────────────
                        total input           =  0.867
  sigmoid(0.867)                              =  0.704
)
```

This is the "Forensic Math Panel" — you see exactly why this edge received its score.

---

## Loading a Graph

### From a CSV file
Click **Upload CSV** in the Graph Panel toolbar. Format: `subject,predicate,object[,weight]`

### From the toy graph (built-in)
Click **Load Toy Graph** to load the canonical 21-node test graph from `tests/fixtures/toy_graph.csv`. Useful for exploring the interface before loading your own data.

### From a URL (Remote CSV)
Enter a URL in the **Graph URL** field and click **Load**. The Studio fetches and parses the CSV remotely.

---

## Community Visualization

Click the **Communities** tab to switch the Graph Panel to community-focus mode:
- Each community is shown as a convex hull overlay
- Community sizes and modularity Q score are displayed in the sidebar
- Use the **Rebalance** button to trigger a fresh DSCF run and watch communities reorganize

---

## Interactive Attention Tuning (Dialectics)

The **Attention Sliders** section lets you adjust the six CSA weights (α, β, γ, δ, ε, ζ) in real time using sliders. When you move a slider, the last query automatically re-runs with the new weights and the Graph Panel updates to show how the reasoning path changes.

This allows you to answer questions like:
- "What if I trust community structure more than semantic similarity?" (increase β)
- "What if I weight relation types more heavily?" (increase γ)
- "What if I penalize long paths more?" (increase δ)

---

## Live Streaming Mode

Click the **Stream** tab to enable streaming ingest mode:
1. Select a discretizer (Threshold, STDP, Delta, Frequency, Pattern)
2. Configure thresholds
3. Click **Start Stream**
4. Watch edges materialize in the Graph Panel in real time as events are processed
5. Community structure auto-updates when modularity drift triggers a rebalance

---

## Exporting Results

- **Export Path** — saves the selected reasoning trace as a JSON file including the full CSA math breakdown
- **Export Graph** — saves the current graph (with community assignments) as a CSV
- **Screenshot** — captures the current Graph Panel view as a PNG

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Enter` | Run query with current settings |
| `Esc` | Clear selection |
| `Space` | Toggle community overlay |
| `R` | Rebalance communities |
| `→` | Step through reasoning path (hop by hop) |

---

## Troubleshooting

**Studio won't start**: Ensure `gradio` and `pyvis` are installed: `pip install -r ui/requirements.txt`

**Graph not rendering**: Large graphs (>5,000 nodes) may take several seconds for pyvis to lay out. Use the **Limit Display** slider to show only the top-N nodes by PageRank.

**No paths found**: The query entity may not exist in the loaded graph. Try the autocomplete dropdown or check entity names are exact matches (case-sensitive).

**Slow queries**: Reduce `beam_width` or `max_hops` in the Query Panel.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

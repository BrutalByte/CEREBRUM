# CEREBRUM Reasoning Studio — User Guide

**Version**: v1.9.8
**Interface**: Gradio web UI + pyvis graph visualization
**Business Logic Module**: `core/studio_engine.py` (`StudioEngine`)
**UI Entry Point**: `ui/studio.py`

---

## Overview

The Reasoning Studio is CEREBRUM's interactive visual interface. It lets you load a Knowledge Graph, run multi-hop reasoning queries, inspect every step of the CSA attention mechanism, and watch community structure form in real time — all without writing code.

This is the "Glass-Box" in action: you can see exactly what the AI is doing at every reasoning step.

As of **Phase 54**, the Studio's business logic has been fully extracted into `core/studio_engine.py`. The Gradio layer in `ui/studio.py` is now a thin UI shell over `StudioEngine`. This means Studio functionality is fully unit-testable without a running server or browser — see `tests/test_studio_engine.py` (38 tests).

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
- **Unit testing without a server**: 38 new tests in `tests/test_studio_engine.py` validate all Studio business logic in isolation.
- **Embeddability**: `StudioEngine` can be imported and used in scripts, notebooks, or other UIs without Gradio.
- **Separation of concerns**: UI layout changes do not affect reasoning logic, and vice versa.

### Programmatic use

```python
from core.studio_engine import StudioEngine

engine = StudioEngine()
engine.load_graph("tests/fixtures/toy_graph.csv")
results = engine.run_query("Marie Curie", max_hops=3, beam_width=5)
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
When you click an edge in the Graph Panel or an answer path in the Answer Panel, this panel shows the full 10-parameter CSA formula breakdown for that specific edge:

```
a(Marie Curie → Radium, hop=1) = sigmoid(
  alpha=0.4  × sim=0.71   =  0.284   ← semantic similarity
  beta=0.4   × cs=1.00    =  0.400   ← same community
  gamma=0.1  × etw=0.95   =  0.095   ← "discovered" weight
  delta=0.05 × nd=0.10    = -0.005   ← distance penalty
  eps=0.05   × hd=1.00    =  0.050   ← hop 1
  zeta=0.1   × pr=0.43    =  0.043   ← PageRank prior
  eta=0.1    × td=0.95    =  0.095   ← temporal decay
  iota=0.05  × nr=0.80    =  0.040   ← node recency
  mu=0.1     × sd=0.00    = -0.000   ← synthesis penalty (real edge)
  theta=1.0  × gc=0.90    =  0.900   ← grounding confidence
                             ──────
                             total  =  1.902
  sigmoid(1.902)           =  0.870
)
```

This is the "Forensic Math Panel" — you see exactly why this edge received its score.

---

## Interactive Attention Tuning (Dialectics)

The **Attention Sliders** section exposes all **10 CSA weight sliders** (α, β, γ, δ, ε, ζ, η, ι, μ, θ) in real time. When you move a slider, the last query automatically re-runs with the new weights and the Graph Panel updates to show how the reasoning path changes.

| Slider | Parameter | Effect |
|---|---|---|
| Alpha (α) | Semantic similarity | Weight on embedding cosine similarity |
| Beta (β) | Community score | Weight on structural co-membership |
| Gamma (γ) | Edge-type weight | Importance of relationship type |
| Delta (δ) | Distance penalty | Penalizes long paths |
| Epsilon (ε) | Hop decay | Relevance drop-off per hop |
| Zeta (ζ) | PageRank prior | Preference for high-centrality destinations |
| Eta (η) | Temporal decay | Preference for recent evidence |
| Iota (ι) | Node recency | Preference for recently-touched nodes |
| **Mu (μ)** | **Synthesis penalty** | **Penalizes edges synthesized by REM (new in v1.9.8)** |
| Theta (θ) | Grounding confidence | Weight on source reliability |

**Note**: Mu (μ) is new in v1.9.8. It penalizes paths that rely on synthetic (REM-synthesized) edges, giving you control over how much CEREBRUM trusts its own hypotheses versus grounded facts.

This allows you to answer questions like:
- "What if I trust community structure more than semantic similarity?" (increase β)
- "What if I penalize synthetic edges heavily?" (increase μ)
- "What if I prioritize well-grounded sources?" (increase θ)

---

## Loading a Graph

### From a CSV file
Click **Upload CSV** in the Graph Panel toolbar. Format: `subject,predicate,object[,weight]`

### From the toy graph (built-in)
Click **Load Toy Graph** to load the canonical 21-node test graph from `tests/fixtures/toy_graph.csv`.

### From a URL (Remote CSV)
Enter a URL in the **Graph URL** field and click **Load**. The Studio fetches and parses the CSV remotely.

### Hot Reload via /build (new in v1.9.8)
The server exposes `POST /build` — a multipart endpoint that accepts a CSV file and rebuilds the graph in-process without a server restart. The Studio's **Reload Graph** button calls this endpoint. Useful for CI/CD pipelines and rapid iteration.

```bash
curl -X POST http://localhost:8200/build \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@new_graph.csv"
```

---

## Community Visualization

Click the **Communities** tab to switch the Graph Panel to community-focus mode:
- Each community is shown as a convex hull overlay
- Community sizes and modularity Q score are displayed in the sidebar
- Use the **Rebalance** button to trigger a fresh DSCF run and watch communities reorganize
- The algorithm selector (dscf / tsc / leiden / lpa) is configurable in the Studio settings panel

---

## Live Log Monitor (new in v1.9.8)

The Studio now includes a **Logs** tab powered by `GET /logs`. This streams recent server log entries from the `RingBufferHandler` (defined in `core/log_config.py`) directly into the Studio UI.

### Log filter options
| Parameter | Description |
|---|---|
| `level` | Minimum log level: DEBUG, INFO, WARNING, ERROR |
| `limit` | Maximum number of entries to return |
| `since` | ISO timestamp — only entries after this time |
| `search` | Substring filter on log message |

The **Clear Logs** button calls `DELETE /logs` to flush the ring buffer.

### RingBufferHandler
The handler is installed at server startup and captures all `logging`-module records into a fixed-size circular buffer. When the buffer fills, the oldest entries are dropped. This provides low-overhead, zero-persistence observability suitable for development and staging environments.

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

## Production Monitoring: dashboard.html

For production environments, `ui/dashboard.html` provides a lightweight monitoring page that does not require Gradio. It displays:
- Node and edge counts
- Active community count and modularity Q
- Recent query latencies
- Live log stream (polling `GET /logs`)
- Server uptime

Open it in any browser pointed at a running CEREBRUM API server. No installation required.

---

## Unit Testing StudioEngine

Because all business logic is isolated in `core/studio_engine.py`, the Studio can be tested without a browser or Gradio:

```bash
pytest tests/test_studio_engine.py
```

38 tests cover:
- Graph loading and hot-reload
- Query execution and result schema
- CSA weight application
- Log buffer read/write/clear
- Community algorithm selection
- CSV build pipeline

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

**Slow queries**: Reduce `beam_width` or `max_hops` in the Query Panel. With Adaptive Search (Phase 53), these are auto-tuned but can still be overridden manually.

**Mu slider not visible**: Ensure you are running v1.9.8 or later. Older Studio versions had 9 sliders (alpha through theta, without mu).

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

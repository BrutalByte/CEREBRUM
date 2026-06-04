# [Buchorn, 2026]: Glass-Box Reasoning Studio
## Interactive Visualization and Forensic Audit of Graph Attention

**Status**: v2.73.0 (Phase 223 (Sleep-Phase Consolidation) COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Field**: Human-Computer Interaction / Explainable AI (XAI) / Data Visualization  
**Module**: `ui/studio.py`

---

### 1. Introduction
Modern AI reasoning is often opaque, providing answers without explaining the underlying logic. The **Glass-Box Reasoning Studio** is a proprietary visual interface for CEREBRUM designed to transform "Black-Box" graph operations into an interactive, auditable experience. It provides forensic-level visibility into the reasoning beam, allowing users to see exactly how community structures (Attention Heads) and latent semantic signals influence every step of a decision.

### 2. The Reasoning Trace
The Studio's primary feature is the **Reasoning Trace Viewer**. Unlike a static graph plot, the Trace Viewer renders the specific multi-hop path chosen by the **BeamTraversal** engine.

#### 2.1 Dynamic Path Highlighting
-   **Edges**: Scaled by their **CSA Attention Weight** [Buchorn, 2026].
-   **Nodes**: Color-coded by their **DSCF Community** [Buchorn, 2026].
-   **Heatmaps**: Nodes are shaded by their contribution to the final answer's confidence score.

#### 2.2 Forensic Score Breakdown
Selecting any hop in the trace opens a "Forensic Panel" that displays the raw math behind the CSA formula:
-   Semantic Similarity ($\alpha$)
-   Community Guidance ($\beta$)
-   Structural Centrality ($\delta$)
This allows a human analyst to verify that the AI is following a logical path and not hallucinating connections.

### 3. The Live Graph Feed
For streaming environments, the Studio provides a **Live Feed** tab that visualizes graph evolution in real-time.
-   **Spike Animation**: Entities "glow" when they fire in the stream (STDP spikes).
-   **Edge Materialization**: New `CAUSES` or `INSIGHT_LINK` edges appear as shimmering links, indicating their "Speculative" status before REM-cycle verification.
-   **Community Drift**: A global modularity gauge shows the real-time health of the graph's attention heads.

### 4. Interactive Community Exploration
The Studio allows users to "drill down" into the community hierarchy:
-   **Centroid Visualization**: View the "Semantic North Star" of any cluster.
-   **Boundary Scanning**: Manually trigger an `InsightEngine` scan between two specific communities to discover hidden relationships.
-   **Bridge Management**: Visual interface for promoting or pruning **Bridge Twin** relays [Buchorn, 2026].

### 5. Implementation Notes
*   **Architecture**: Built using a reactive Python/Gradio framework with a `vis-network` JavaScript backend for high-performance rendering.
*   **Security**: Fully integrated with JWT authentication; users only see graph partitions they are authorized to access.
*   **Scale**: Utilizes adaptive node clustering to prevent visual clutter, automatically coarsening the view for graphs exceeding 10,000 nodes.

---

## Studio v2 Architecture (Phases 75–78)

Studio v2 introduces a **six-panel live dashboard** driven by optional engine attachments. All panels degrade gracefully if the corresponding engine is not attached — no error is raised, and the panel renders a "not configured" notice.

### 6. Optional Attachments API

```python
studio = StudioEngine(graph)

# Attach any subset; all are optional
studio.attach_research_agent(agent)           # enables panels 1–3
studio.attach_modulator(chemical_modulator)   # enables panel 4
studio.attach_loop(autonomous_loop)           # enables panel 5
studio.attach_provenance_ledger(ledger)       # enables panel 6
```

### 7. Dashboard Panels

#### 7.1 AutoApprover Audit Log (Phase 75)
```python
html: str = studio.get_autoapprover_panel()
```
Renders a scrollable table of the last N AutoApprover decisions: finding ID, entity pair, decision (APPROVE / REJECT / REVIEW), confidence score, tier reached (hard gate / SGD / LLM), and timestamp.

**Graceful degradation**: If `research_agent` not attached → renders `"ResearchAgent not configured"`.

#### 7.2 ContradictionResolver Revision Queue (Phase 75)
```python
html: str = studio.get_revision_queue_panel()
```
Lists findings currently in `ResearchAgent._revision_candidates`. Columns: entity pair, net_evidence_score, resolution class, nomination count, time in queue.

**Graceful degradation**: If `research_agent` not attached → renders `"ResearchAgent not configured"`.

#### 7.3 DiscoveryCalibrator Community Heatmap (Phase 75)
```python
fig: matplotlib.Figure = studio.get_calibrator_heatmap()
```
Renders a bar chart of per-community discovery rates and inverse-rate weights. Cold-start communities (never scanned) shown in a distinct color. Helps identify saturated vs. underexplored graph regions at a glance.

**Graceful degradation**: If calibrator not attached → returns an empty figure with `"DiscoveryCalibrator not configured"` annotation.

#### 7.4 ChemicalModulator Blood Panel (Phase 75)
```python
fig: matplotlib.Figure = studio.get_modulator_panel()
```
Renders a 5-bar horizontal chart showing current scalar levels (Reinforcement, Arousal, Novelty, Cohesion, Persistence) against their homeostatic baselines. Overdriven scalars shown in orange; depleted in blue.

**Graceful degradation**: If `modulator` not attached → returns figure with `"ChemicalModulator not attached"` annotation.

#### 7.5 Autonomous Loop Cycle History (Phase 75)
```python
html: str = studio.get_loop_panel()
```
Renders a table of recent `CycleRecord` objects: cycle number, timestamp, materializations, approvals, rejections, circuit breaker trips, `effective_cap` (Phase 82), `edges_rolled_back` (Phase 79). Includes a status badge (RUNNING / STOPPED / TRIPPED).

**Graceful degradation**: If `loop` not attached → renders `"AutonomousDiscoveryLoop not configured"`.

#### 7.6 Provenance Panel (Phase 78)
```python
stats_html: str, batch_fig: Figure, timeline_fig: Figure = studio.get_provenance_panel(n=20)
```
Three-part panel:
- **4-card summary row** (HTML): total batches, total edges materialized, total edges rolled back, active batch count.
- **Batch bar chart**: horizontal bars per batch (green = active; red = rolled back). Truncated to `n` most recent.
- **Cycle timeline**: dual-series chart — per-cycle materialization bars (left axis) + cumulative materialized edges dashed line (right axis).

**Graceful degradation**: If `provenance_ledger` not attached → all three return empty/placeholder outputs.

### 8. ERT Integration (Phase 62+)

The `POST /query/trace` endpoint returns a `ReasoningTrace` that exposes:
- `trace.prior` — predicted relation sequence from PredictiveCodingEngine (Phase 69)
- `trace.prediction_error` — Jaccard divergence between prior and actual
- `trace.soliton_index` — rolling coherence metric (1 = stable prior)
- `trace.loop_trace` — per-loop iteration summary for LoopedBeamTraversal (Phase 70)

All fields are rendered in the Studio's **ERT viewer** when the trace endpoint is used.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.73.0

# SPEC_012: Glass-Box Reasoning Studio
## Interactive Visualization and Forensic Audit of Graph Attention

**Status**: v1.1.0 (Phase 20 COMPLETE)  
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Human-Computer Interaction / Explainable AI (XAI) / Data Visualization  
**Module**: `ui/studio.py`

---

### 1. Introduction
Modern AI reasoning is often opaque, providing answers without explaining the underlying logic. The **Glass-Box Reasoning Studio** is a proprietary visual interface for CEREBRUM designed to transform "Black-Box" graph operations into an interactive, auditable experience. It provides forensic-level visibility into the reasoning beam, allowing users to see exactly how community structures (Attention Heads) and latent semantic signals influence every step of a decision.

### 2. The Reasoning Trace
The Studio's primary feature is the **Reasoning Trace Viewer**. Unlike a static graph plot, the Trace Viewer renders the specific multi-hop path chosen by the **BeamTraversal** engine.

#### 2.1 Dynamic Path Highlighting
-   **Edges**: Scaled by their **CSA Attention Weight** (SPEC_002).
-   **Nodes**: Color-coded by their **DSCF Community** (SPEC_001).
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
-   **Bridge Management**: Visual interface for promoting or pruning **Bridge Twin** relays (SPEC_003).

### 5. Implementation Notes
*   **Architecture**: Built using a reactive Python/Gradio framework with a `vis-network` JavaScript backend for high-performance rendering.
*   **Security**: Fully integrated with JWT authentication; users only see graph partitions they are authorized to access.
*   **Scale**: Utilizes adaptive node clustering to prevent visual clutter, automatically coarsening the view for graphs exceeding 10,000 nodes.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

# SPEC_007: The REM Cycle
## Metacognitive Maintenance and Insight Synthesis

**Status**: v2.24.0 (Phase 112 (Sleep-Phase Consolidation) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Metacognition / System 2 Reasoning / Autonomous Maintenance  
**Module**: `core/rem_engine.py`

---

### 1. Introduction
Autonomous Knowledge Graphs (KGs) are subject to "Informational Entropy"—the accumulation of noisy edges, stale community mappings, and spurious causal correlations. Without periodic maintenance, the graph's signal-to-noise ratio (SNR) inevitably degrades.

The **REM Cycle** (Rapid Edge Maintenance) is a background metacognitive loop that mimics biological sleep to perform three core functions: (1) **Pruning** of stale or unverified data, (2) **Consolidation** of high-usage reasoning paths, and (3) **Synthesis** of latent global insights.

### 2. Bilateral Verification (Triangulation)

To distinguish between grounded facts and speculative noise, the REM cycle employs **Bilateral Verification**. A target edge $E_{uv}$ (typically an `INSIGHT_LINK` or `STDP_CAUSES` edge) is verified if it meets at least two of the following criteria:

1.  **Topological Density**: The edge exists in a locally dense neighborhood (clustering coefficient $C \geq 0.4$).
2.  **Transitive Support**: The `InferenceEngine` identifies an alternative path $P = \{u \to w_1 \to \dots \to v\}$ with confidence $\geq 0.7$, where $P$ does not include $E_{uv}$.
3.  **Community Consensus**: Both $u$ and $v$ reside in the same community or are bridged by a high-strength `BRIDGE_TWIN` node (SPEC_003).

If an edge fails bilateral verification over two consecutive REM cycles, its confidence is reduced to 0.0 and it is deleted.

### 3. Recursive Hallucination Prevention

A unique danger in self-optimizing graphs is the "Insight Feedback Loop," where the system materializes a false insight and then "learns" from it, making it appear more grounded than it is.

The REM cycle implements **Insight Confidence Decay**:
*   **Grounded Edges**: Decay linearly at rate $\lambda$.
*   **Insight Edges**: Decay exponentially at rate $\lambda \cdot \rho$, where $\rho = 0.8$ is the **Skepticism Factor**.

**Survival Rule**: An `INSIGHT_LINK` only transitions to a "Grounded" state if it is explicitly utilized in a successful user-validated query or matches new data ingested via THALAMUS.

### 4. Background Re-optimization

The REM cycle acts as the scheduler for the **GlobalRebalancer** (SPEC_001). It monitors the graph for:
1.  **Modularity Drift**: Cumulative change in node assignments vs. the original partition.
2.  **Size Inbalance**: One community growing to $> 30\%$ of the total graph.

When triggers are met, REM spawns a background thread to:
1.  Compute a fresh DSCF/TSC partition.
2.  Perform an **Atomic Swap** of the `community_map` under a global lock.
3.  Call `on_rebalance` hooks for all engines (Bridge, STDP, Thalamus).

### 5. Implementation Notes (v2.24.0)

*   **Resource Throttling**: The `ResourceGovernor` limits REM tasks to 15% of available CPU/Memory. If a high-priority user query arrives, REM tasks are paused (the "Arousal Interrupt").
*   **Persistence**: All REM actions (prunes, consolidations, re-balances) are logged to the `METADATA` store for forensic audit.
*   **Scheduling**:
    *   **Hot Path (10 min)**: Pruning of ephemeral TTL edges.
    *   **Cold Path (1 hour)**: Bilateral verification and Insight decay.
    *   **REM Path (Daily/Triggered)**: Full modularity re-optimization.

### 6. Sleep-Phase Consolidation (Phase 112)
CEREBRUM v2.24.0 introduces a unified `ConsolidationEngine` that merges traditional REM maintenance with mnemonic optimization.

1.  **Hebbian Replay (Phase 96)**: The engine replays high-salience (high score, low prediction error) traces from Working Memory. Edges participating in these successful reasoning paths receive a weight boost ($+ \Delta w$), increasing their future attention priority (LTP analog).
2.  **Shortcut Synthesis (Phase 112)**: The engine identifies recurrent multi-hop trajectories in the `QueryLog`. Trajectories exceeding the `threshold` are materialized as direct `REM_SHORTCUT` edges.
    *   **Effect**: Reduces hop-count and computational complexity for frequently asked questions.
    *   **Grounding**: Shortcuts inherit the minimum confidence of their component edges and are subject to the same decay rules as standard insights.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

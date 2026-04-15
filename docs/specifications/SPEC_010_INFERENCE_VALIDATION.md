# SPEC_010: Inference Validator
## Self-Contained Precision/Recall Methodology for Unsupervised Graph Reasoning

**Status**: v2.1.0 (Phase 82 COMPLETE)  
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Model Evaluation / Information Retrieval / Graph Algorithms  
**Module**: `core/inference_validator.py`

---

### 1. Introduction
Evaluating unsupervised reasoning engines is notoriously difficult due to the lack of ground-truth labels. In Knowledge Graph (KG) environments, traditional cross-validation is often impossible. **Inference Validator** provides a self-contained methodology for measuring precision and recall by treating the graph's own topology as ground truth. By utilizing a **Path-Preserving Hold-out** strategy, it quantifies the engine's ability to rediscover hidden knowledge through multi-hop inference.

### 2. The Hold-out Methodology

The validator simulates a "missing knowledge" scenario to test the engine's predictive power.

#### 2.1 Edge Sampling
The system selects a fraction $\theta$ (default 0.2) of existing edges from the graph. For each sampled edge $E_{uv}$ (the target), the engine attempts to "re-derive" the connection.

#### 2.2 Path-Preserving Constraint (Hole Fix 1.1.0)
Random hold-out can "shatter" sparse graphs, making reasoning impossible. The **Path-Preserving Hold-out** rule ensures that an edge $E_{uv}$ is only held out if:
1.  There exists at least one alternative path $P = \{u \to w_1 \dots \to v\}$ in the graph.
2.  The length of $P$ is $\geq 2$ (multi-hop).
3.  $P$ does not contain the edge $E_{uv}$.

This ensures that the "Reasoning Task" is valid—the answer *is* discoverable from the remaining structure.

### 3. Metric Calculation

For each hold-out trial, the reasoning engine is queried for $u \to ?$ or $? \to v$.

#### 3.1 Unsupervised Recall
Recall is the fraction of trials where the held-out target $v$ is present in the top-$K$ answers returned by the **BeamTraversal** (SPEC_006):
$$R@K = \frac{|\text{Rediscovered Edges}|}{|\text{Total Hold-outs}|}$$

#### 3.2 Unsupervised Precision
Precision is measured by the **Confidence Calibration Error**. We compare the system's predicted confidence score $s_{pred}$ for the re-derived edge with the actual presence of the edge in the hold-out set. A well-calibrated system should exhibit high $s_{pred}$ for successfully rediscovered edges and low $s_{pred}$ for noise.

### 4. Implementation: The Validation Harness

The `InferenceValidator` class executes the following workflow:
1.  **Clone**: Create a transient copy of the `GraphAdapter`.
2.  **Prune**: Identify and remove edges satisfying the Path-Preserving Constraint.
3.  **Traverse**: Run the `BeamTraversal` on the pruned graph.
4.  **Audit**: Compare beam results against the "Secret" hold-out set.
5.  **Restore**: Return the metrics and release the transient graph.

### 5. Use Cases
*   **A/B Testing**: Compare different CSA parameters (SPEC_002) to see which configuration yields higher recall.
*   **Confidence Hardening**: Use validation scores to adjust the `warm_start_strength` in **Bayesian Beam Search** (SPEC_006).
*   **Infrastructure Health**: Periodically run validation on a production graph to detect "Reasoning Decay" as data drift occurs.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

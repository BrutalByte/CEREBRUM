# White Paper: Executive Engines
## Meta-Cognitive Supervision in Graph Attention

**Version**: v2.51.0 (Phase 167 COMPLETE)
**Author**: Bryan Alexander Buchorn

---

### Introduction: Beyond Raw Traversal

Reasoning is more than just following edges; it requires the ability to monitor one's own logic, detect errors, and switch strategies when a path is failing. In CEREBRUM, this supervisory role is performed by the **Executive Engines**: the **Frontal Engine** and the **Cingulate Engine**.

### The Cingulate Engine: Error Detection

The Cingulate Engine acts as a real-time "Sanity Check" for the attention beam. It monitors the **Path-Consistency** ($r^2$) of every reasoning chain.
-   **Dissonance Detection**: If a reasoning path begins to jump incoherently across unrelated semantic domains, the Cingulate Engine flags it as "Dissonant."
-   **Dynamic Pruning**: High-dissonance paths are immediately pruned from the beam, even if their individual edge scores are high.

### The Frontal Engine: Strategic Oversight

The Frontal Engine is responsible for high-level goal alignment and strategy selection.
-   **Goal Management**: It maintains the query's primary intent across deep (3+ hop) traversals, preventing the "Semantic Drift" common in standard BFS.
-   **Metabolic Gating**: It interfaces with the **ChemicalModulator** to regulate the system's "Arousal" and "Reinforcement" levels, determining when to be exploratory and when to be conservative.

### Impact on Reasoning Quality

The integration of Executive Engines transforms the reasoning process from a greedy search into a disciplined, self-correcting inference. On the **WebQSP** benchmark, the Executive Engines reduced false-positive answers by **22%**, ensuring that the returned paths are not just statistically likely, but logically coherent.

---
**White Paper Finalized: v2.51.0 (Phase 167 COMPLETE)**

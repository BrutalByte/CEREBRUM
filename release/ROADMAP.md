# Parallax: Research Roadmap

*What has been built, what is next, and why it matters.*

---

## What Parallax Is Today

The current release — DSCF + CSA + BeamTraversal — is a complete, production-hardened reasoning engine. It answers multi-hop questions over knowledge graphs with full interpretability, no training data, and sub-millisecond latency. The core algorithms are stable and validated.

This document describes the active research program that extends the engine into new domains and capabilities. All work described below has been prototyped and tested in the development codebase. These are not speculations — they are completed research phases being prepared for integration into the next public release.

---

## Completed Extensions (Development Branch)

### Real-Time Streaming Reasoning

**Problem**: Knowledge graphs change. Sensor readings, financial transactions, network events, and medical signals produce continuous streams of new facts. Batch-mode graph loading cannot keep pace.

**Solution**: A full streaming pipeline — pluggable data sources (file tail, HTTP, WebSocket, MQTT), five signal discretizer classes that convert raw signals into typed graph edges, a sliding-window buffer with reference-counted edge eviction, and incremental ego-network community updates that re-run DSCF only over nodes affected by new edges.

The result: Parallax can reason over a live, changing knowledge graph without reprocessing the entire graph. Community structure adapts continuously. New facts enter the reasoning engine within milliseconds of being observed.

**Biological analogy**: Online learning — the brain processes new sensory input continuously, updating its internal model without replaying all prior experience.

---

### Bridge Twin Nodes — Experience-Dependent Structural Relay Formation

**Problem**: The CSA community distance penalty correctly discounts speculative cross-domain hops. But some cross-community connections are not speculative — they are verified reasoning steps that recur because they are structurally correct. Repeatedly penalizing them imposes a permanent cost on paths that have already been validated by experience.

**Solution**: Bridge Twins. When a node crosses community boundaries repeatedly (count $\geq n_{min}$) and its embedding fits the destination community's centroid above a threshold:

$$\text{fit}(v, \mathcal{C}_d) = \cos\!\left(\mathbf{e}_v,\; \frac{1}{|\mathcal{C}_d|}\sum_{u \in \mathcal{C}_d} \mathbf{e}_u\right) \geq \theta_{bridge}$$

a twin node $v'$ is materialised in the destination community. The twin carries the same embedding as the original, belongs to the destination community ($c(v') = \mathcal{C}_d$), and is connected to the original by bidirectional `BRIDGE_TWIN` edges. CSA assigns bridge edges a fixed high weight of $\sigma(0.925) \approx 0.716$, eliminating the distance penalty for validated crossings.

Twins diverge over time as they accumulate community-specific edges — the same concept represented through the lens of two different structural contexts. Idle twins are pruned after $\tau_{prune}$ days (a long-term depression analog).

**Biological analogy**: Thalamic relay nuclei. The lateral geniculate nucleus (LGN) holds the same retinotopic map as the retina but is positioned inside the thalamus, close to the visual cortex it serves. Formation requires repeated use; pruning occurs with disuse. Bridge twins implement the same principle in graph space.

---

### STDP: Spike-Timing Dependent Plasticity — Causal Inference from Timing

**Problem**: Co-activation detection (sensors A and B fire together → emit `CO_ACTIVATES` edge) is symmetric. It cannot distinguish cause from effect. Industrial, biomedical, and financial sensor networks produce causal signals that are directional — one event precedes another because it causes it, not merely correlates with it.

**Solution**: A Spike-Timing Dependent Plasticity (STDP) analog. Each sensor event is a "spike." When source A fires before source B within a configurable time window, the directed weight $w(A \to B)$ is potentiated:

$$\Delta w(A \to B) = A_+ \cdot \exp\!\left(-\frac{\Delta t}{\tau_+}\right) \qquad \Delta t = t_B - t_A > 0$$

The anti-causal direction $B \to A$ is simultaneously depressed:

$$\Delta w(B \to A) \leftarrow \max\!\left(0,\; w(B \to A) - A_- \cdot \exp\!\left(-\frac{\Delta t}{\tau_-}\right)\right)$$

Default parameters ($A_+ = 0.1$, $A_- = 0.105$, $\tau_\pm = 0.2\,$s) implement slight LTD dominance, matching biological measurements (Bi & Poo, 1998) and preventing runaway weight accumulation. Per-spike weight decay ($\lambda = 0.99$ by default) provides exponential forgetting.

When $w(A \to B) \geq w_{threshold}$ and co-occurrence count $\geq n_{min}$, a directed `CAUSES(A → B)` edge is emitted into the knowledge graph.

**Result**: From a stream of sensor events with no labels, no training, and no domain configuration, Parallax autonomously discovers directed causal chains. A factory's pressure sensor, pump relay, and relief valve self-organize into a verified causal chain — observable and queryable through the same reasoning engine.

**Biological analogy**: The STDP rule (Bi & Poo, 1998) is one of the primary mechanisms of synaptic learning in the brain. Neurons that fire in a consistent temporal order strengthen their connections; neurons that fire in the reverse order weaken them. The brain learns causation from timing. Parallax does the same.

---

## What Comes Next

The three extensions above — streaming, Bridge Twins, and STDP — form a coherent platform for **continuous, self-organizing, causally-aware** knowledge graph reasoning. The next major milestone targets:

**Distributed beam traversal**: Federated reasoning across multiple Parallax instances, each holding a partial graph. Holographic indexing (Bloom filters + community centroids) enables "blind" discovery of relevant remote nodes without exposing the full graph — a privacy-preserving architecture for enterprise multi-tenant deployments.

**GPU acceleration**: Community detection and embedding computation are parallelizable. The CSA formula is embarrassingly parallel over edges. A GPU-accelerated implementation would support graphs with millions of entities while maintaining sub-millisecond query latency.

**Formal academic publication**: The DSCF + CSA + BeamTraversal algorithms, the bridge twin formation analysis, and the STDP causal inference framework together constitute a novel contribution to the knowledge graph reasoning literature. A formal paper submission is in preparation.

**Adaptive parameter learning**: The five CSA coefficients ($\alpha, \beta, \gamma, \delta, \varepsilon$) are currently fixed. A lightweight meta-learning layer that adapts them per-domain — trained on a small set of example queries — would close the gap between zero-shot performance and supervised methods without requiring full training.

---

## The Larger Vision

Parallax is built on a single observation: the structural principles that make Transformer attention effective over sequences also apply to graphs — and graphs are the natural representation of real-world knowledge.

The algorithms in this release (DSCF + CSA + Beam) establish the foundation. The streaming pipeline, Bridge Twins, and STDP demonstrate that the foundation is extensible to continuous, adaptive, and causally-aware reasoning without modifying the core algorithms.

The goal is a reasoning engine that learns the structure of any domain automatically, reasons over it transparently, and grows more capable with experience — without labeled training data, without opaque model weights, and without the possibility of fabricating facts it was never told.

That is what it means to reason from a glass box.

---

*Active research: Bryan Alexander Buchorn (AMP) · March 2026*
*Contact: bryan.alexander@buchorn.com*

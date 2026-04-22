# SPEC_004: Autonomous Causal Discovery via STDP
## Temporal Graph Plasticity in Knowledge Streams

**Status**: v2.24.0 (Phase 111 (Active Inference) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Field**: Causal Inference / Stream Processing / Neuro-Symbolic AI  
**Module**: `core/discretizer.py`

---

### 1. Introduction
Inferring causal relationships from unstructured event streams traditionally requires massive labeled datasets and supervised training. The **STDPDiscretizer** provides an autonomous, unsupervised alternative by adapting the biological principle of **Spike-Timing-Dependent Plasticity (STDP)** to temporal knowledge streams.

By tracking the relative timing of entity "spikes" (mentions/events in a stream), the engine infers directional `CAUSES` edges. This specification formalizes the causal weighting rule, the significance filters, and the **Lazy Decay** optimization that enables $O(1)$ real-time processing at scale.

### 2. The STDP Causal Rule

When an entity $v$ "spikes" (occurs in the stream) at time $t_{post}$, the causal weight $w_{uv}$ between $v$ and any prior spike $u$ (at time $t_{pre}$) within a sliding window $W$ is updated according to the Bi & Poo (1998) rule.

#### 2.1 Potentiation (LTP)
If $u$ occurs before $v$ ($t_{post} > t_{pre}$), the directional hypothesis $u \rightarrow v$ is strengthened:

$$
\Delta w_{uv} = A_+ \cdot e^{-(t_{post} - t_{pre}) / \tau_+}
$$

*   $A_+$: Maximum potentiation (default $0.1$).
*   $\tau_+$: Potentiation time constant (default $20.0s$).

#### 2.2 Depression (LTD)
If $u$ occurs after $v$ ($t_{pre} > t_{post}$), or if $v$ is a repeat spike without a cause, the hypothesis is weakened:

$$
\Delta w_{uv} = -A_- \cdot e^{-(t_{pre} - t_{post}) / \tau_-}
$$

*   $A_-$: Maximum depression (default $0.05$).
*   $\tau_-$: Depression time constant (default $40.0s$).

### 3. Enterprise Optimization: Lazy Decay

In a graph with $N$ potential causal pairs, applying a global weight decay ($w = w \cdot \lambda$) on every event is $O(N)$, which leads to a "Performance Ceiling" as the graph grows. CEREBRUM implements **Lazy Decay**, turning this into a local $O(1)$ operation.

For each pair $(u, v)$, the system stores the weight $w_{uv}$ and the global step counter $t_{last}$ at which that specific pair was last touched.

Given a current global step $T$ and decay rate $\lambda \in [0, 1]$:
1.  **On Access**: Compute adjusted weight $w'_{uv} = w_{uv} \cdot \lambda^{(T - t_{last})}$.
2.  **Update**: Apply the STDP delta $\Delta w$ to $w'_{uv}$.
3.  **Store**: Save the new weight and update $t_{last} = T$.

This ensures the CPU cost of "forgetting" does not scale with the size of the causal memory.

### 4. Causal Significance Filtering

To prevent "Causal Floods" from high-frequency noise or adversarial jitter, an edge is only materialized as a `CAUSES` triple if it survives a four-stage validation harness.

#### 4.1 Threshold & Count
*   $w_{uv} \geq w_{threshold}$ (default $0.5$).
*   Pairing count $n \geq n_{min}$ (default $5$).

#### 4.2 Temporal Span Filter
The time between the first and last observed pairings must exceed a threshold:
$$
(t_{last\_pair} - t_{first\_pair}) \geq T_{span} \quad (\text{default } 1.0s)
$$
This blocks instantaneous adversarial bursts.

#### 4.3 $\chi^2$ Uniformity Test
The inter-spike intervals between $u$ and $v$ are binned and tested against a uniform distribution. If the distribution is highly clustered (low p-value), it is rejected as a non-causal coincidence or a rhythmic artifact.

### 5. Implementation Notes (v1.1.0)

*   **Directionality**: Emitted edges are strictly directional: `source` $\xrightarrow{CAUSES}$ `target`.
*   **Memory Eviction**: Causal pairs whose weights decay below $0.05$ are automatically evicted from the weight map to prevent memory bloat.
*   **Real-time Integration**: Discretized edges are pushed into the `StreamAdapter` buffer with a TTL (Time-to-Live), allowing the Reasoning Engine to follow "Active Causal Chains."

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

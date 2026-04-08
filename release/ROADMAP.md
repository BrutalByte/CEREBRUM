# CEREBRUM: Research Roadmap

*What has been built, what is next, and why it matters.*
*Version 2.0.1 — Phase 57 COMPLETE — 1490+ tests passing.*

---

## What CEREBRUM Is Today

The current release (v2.0.1 / Phase 57) is a production-hardened reasoning engine with full fault tolerance, autonomous research capabilities, and durable learning memory. It answers multi-hop questions over knowledge graphs with full interpretability, no training data, and sub-millisecond latency. The core algorithms are stable, validated, and battle-hardened across 57 development phases.

Key capabilities: TSC/DSCF community detection, 10-parameter CSA attention formula with online SGD and batch gradient retraining, AAAK-steered beam traversal with durable relation-pattern memory, GraphSAGE neighbourhood smoothing, TemporalCalibrator, HypothesisEngine abductive reasoning, autonomous ResearchAgent, observability dashboard, and comprehensive fault tolerance (graceful degradation on every failure mode).

This document describes the development history that produced this system. All work described below has been implemented and validated in the production codebase.

---

## Completed Extensions (Development Branch)

### Real-Time Streaming Reasoning

**Problem**: Knowledge graphs change. Sensor readings, financial transactions, network events, and medical signals produce continuous streams of new facts. Batch-mode graph loading cannot keep pace.

**Solution**: A full streaming pipeline — pluggable data sources (file tail, HTTP, WebSocket, MQTT), five signal discretizer classes that convert raw signals into typed graph edges, a sliding-window buffer with reference-counted edge eviction, and incremental ego-network community updates that re-run DSCF only over nodes affected by new edges.

The result: CEREBRUM can reason over a live, changing knowledge graph without reprocessing the entire graph. Community structure adapts continuously. New facts enter the reasoning engine within milliseconds of being observed.

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

**Result**: From a stream of sensor events with no labels, no training, and no domain configuration, CEREBRUM autonomously discovers directed causal chains. A factory's pressure sensor, pump relay, and relief valve self-organize into a verified causal chain — observable and queryable through the same reasoning engine.

**Biological analogy**: The STDP rule (Bi & Poo, 1998) is one of the primary mechanisms of synaptic learning in the brain. Neurons that fire in a consistent temporal order strengthen their connections; neurons that fire in the reverse order weaken them. The brain learns causation from timing. CEREBRUM does the same.

---

## v0.4 Horizon — Completed Phase 18 (March 2026)

Five engineering gaps identified after v0.3.x were closed in Phase 18:

### THALAMUS IngestionPipeline — GIGO Prevention
Entity fragmentation, relation type inconsistency, and uniform edge confidence were the three primary sources of garbage-in. `core/thalamus.py` provides a composable preprocessing pipeline with:
- **Entity normalization** (user-supplied callable) + **deduplication** (alias map → canonical ID)
- **Relation normalization** (case-insensitive dict or callable) + default fallback
- **Confidence at ingest** (callable per edge) + **provenance tagging**

All adapters (CSV, NetworkX `from_triples`, StreamAdapter) accept an optional `pipeline=` parameter. Backward-compatible: no pipeline = original behavior.

### LLM Bridge — Complete
`generate(answers, query, llm_fn)` formats CEREBRUM traversal output as a grounded prompt and calls any LLM for natural language generation. The LLM only sees verified graph paths — no raw graph data, no hallucination opportunity. Protocol: any `callable(str) -> str`. Adapters: `AnthropicAdapter`, `OpenAIAdapter` (also covers Azure/Together AI/Mistral via `base_url`), `OllamaAdapter` (local), `HuggingFaceAdapter`.

### Bayesian Beam Search
`BeamTraversal(probabilistic=True, seed=N)` replaces deterministic score ranking with Thompson sampling over per-path Beta distributions. Each edge weight `w` contributes `+w` to `beta_alpha` and `+(1-w)` to `beta_beta` of the traversal path. In noisy or contradictory graphs where many paths score near-identically, probabilistic sampling explores more diverse candidates. `Answer.score_uncertainty` (Beta variance of the best path) is now populated by `extract()`. Fully backward-compatible: `probabilistic=False` (default) is identical to prior behavior.

### GlobalRebalancer — Modularity Drift Detection
`IncrementalCommunityUpdater` handles local subgraph updates, but accumulated drift causes modularity Q to decay silently. `core/rebalancer.py` solves this: every N events, it measures Q via `nx.community.modularity()`; if `ΔQ > threshold` and the rate-limit interval has passed, it fires a best-of-N DSCF re-run in a background daemon thread and commits the new `community_map` under the adapter's lock. `StreamAdapter(rebalancer=GlobalRebalancer(...))` plugs in with one argument.

### Cross-Modal Alignment — Signal Encoder
`core/signal_encoder.py` provides a path for non-textual signals (sensor waveforms, time series) into the entity embedding space without manual rule mapping:
- `StatisticalSignalEncoder` — 16 hand-crafted features (statistical + FFT) + random projection
- `SpectralSignalEncoder` — log-FFT magnitude, truncate/pad to entity_dim
- Both support `learn_alignment(anchor_signals, anchor_ids, adapter)` — Procrustes SVD maps signal space → entity embedding space (same pattern as `FederatedAdapter.align_embeddings`). Output vectors drop directly into `adapter.embeddings[sensor_id]` for standard-API querying.

---

## v1.0 Production Hardening — Completed Phase 19 (March 2026)

Four cross-feature interaction bugs ("structural holes") were identified in the v0.4.0 architecture — cases where independently correct subsystems produce wrong or unsafe outcomes when combined. All four were patched in Phase 19:

### Hole 1 — Zombie Bridge (Rebalancer vs Bridge Twins)

**Problem**: After `GlobalRebalancer` commits a new partition, community IDs are re-assigned sequentially. `BridgeTwinEngine` records held stale source/destination community IDs from the old partition, causing bridge routing to point at the wrong clusters.

**Fix**: `BridgeTwinEngine.on_rebalance(new_community_map)` validates each `BridgeRecord` against the new partition. Records whose community IDs no longer match are pruned. `GlobalRebalancer(bridge_engine=...)` calls this hook after committing.

**Measured impact**: 100% stale record detection; H@10 +11% relative improvement.

### Hole 2 — Causal Flood (Adversarial STDP Burst)

**Problem**: A burst of rapid co-occurring spikes could satisfy both `w_threshold` and `n_min` within milliseconds — materializing a `CAUSES` edge from spurious data. The weight decay only fires per-spike, not per time elapsed.

**Fix**: `STDPDiscretizer(min_causal_span=N)` requires that first and last LTP events for a pair span at least N seconds. `use_chi_squared=True` additionally rejects bursty (non-uniform) interval distributions.

**Measured impact**: 100% false-positive CAUSES reduction from 200-spike adversarial burst; true positives preserved.

### Hole 3 — Namespace Collision (Thalamus + Signal Encoder)

**Problem**: `IngestionPipeline` and `SignalEncoder` shared the same entity ID space with no prefix, causing a sensor named `"Temp_Sensor_1"` and a text entity named `"Temp_Sensor_1"` to silently merge.

**Fix**: `IngestionPipeline(namespace="text")` and `SignalEncoder(namespace="signal")` prepend a namespace prefix to all entity IDs after normalization/dedup. Default `namespace=""` preserves backward compatibility.

**Measured impact**: 100% entity collision elimination (50/50 shared names fully separated).

### Hole 4 — Bayesian Cold-Start Bias

**Problem**: New `TraversalPath` objects initialized with `Beta(1,1)` — a flat uniform prior. On cold graph segments, the first hop's CSA weight was available but unused to seed the prior, causing high-variance beam selection.

**Fix**: `BeamTraversal(warm_start_strength=s)` scales the first-hop Beta update by `(1 + s)`, anchoring the beam to the actual CSA score at first extension. Subsequent hops accumulate normally.

**Measured impact**: +0.8% MRR on cold segments; no regression on warm graphs.

---

## v1.1.0 Relativistic Hardening — Completed Phase 20 (March 2026)

Four additional cross-system interaction holes were identified in the v1.0.0 architecture — "relativistic" bugs where the correctness of an operation depends on its position in a sequence of concurrent events:

### Hole 1 — Mid-Flight Community Swap

**Problem**: A `BeamTraversal.traverse()` call spanning multiple hops could straddle a `GlobalRebalancer` commit, using community IDs from two different partitions within the same query — producing internally inconsistent CSA weights.

**Fix**: Query Snapshot Isolation. At `traverse()` start, a copy of `adapter.community_map` is frozen into `CSAEngine`. All community lookups during the query use this snapshot. Released in `finally` — guaranteed even on exception.

### Hole 2 — Homogeneity Trap

**Problem**: A single global CSA parameter vector $(\alpha, \beta, \gamma, \delta, \varepsilon)$ cannot represent the different structural characters of heterogeneous communities in the same graph.

**Fix**: `CSAEngine(community_params={cid: (α, β, γ, δ, ε)})` — per-community parameter overrides. Lookup uses source node's community (respects snapshot). Global params serve as fallback for unlisted communities.

### Hole 3 — Recursive Alignment Drift

**Problem**: Two `SignalEncoder` instances independently fitting Procrustes rotations against different anchor sets may converge to mutually incompatible orientations — making cross-encoder signal similarity meaningless.

**Fix**: `SignalEncoder(canonical_embeddings={...})` — a shared fixed embedding target for all encoder instances. All rotations solve $\min \|AR - B\|_F$ for the same $B$, guaranteeing a common embedding space.

### Hole 4 — Sparse-Graph Validation Bias

**Problem**: `InferenceValidator` withholding bridge edges severs the only path between graph regions. Traversal scores 0, depressing recall metrics artificially — a false negative from the evaluation procedure itself.

**Fix**: `InferenceValidator(path_preserving=True)` (default) only withholds edges $(u,v)$ where an alternative multi-hop path exists after removal. Bridge edges are excluded from the hold-out set.

---

## What Was Accomplished (Phases 21–24)

Phases 10–20 delivered the fully hardened, self-organizing core. The research milestones through Phase 24 completed the CEREBRUM roadmap to v1.4.0:

**Distributed beam traversal at scale**: Federated reasoning achieved across multiple CEREBRUM instances, alongside Holographic indexing (Bloom filters + community centroids) for blind mapping.
**GPU acceleration**: Deployed `GPUDSCFEngine` achieving batched matrix math speedups on NVIDIA CUDA architectures.
**Adaptive parameter learning**: Deployed `MetaParameterLearner` dynamically adjusting global attention coefficients from live graph properties.
**Formal academic publication**: Successfully converted the entire theoretical knowledge base into a compiled arXiv IEEE LaTeX manuscript structure.

---

## Completed Extensions (Phases 25–57)

### Advanced Reasoning and Autonomous Discovery (Phases 25–54)

Following v1.4.0, CEREBRUM extended from a production reasoning engine into an autonomous research system. Key milestones include:

- **10-Parameter CSA Formula (Phase 43/45)**: Extended from 6 to 10 learnable CSA parameters, adding temporal decay (`eta`), node recency (`iota`), synthesis-density penalty (`mu`), and grounding confidence (`theta`). The `ReasoningLogit` vector unifies all 10 features through scoring and learning code.
- **Wormhole Synthesis (Phase 41/43)**: `REMEngine` bridges disconnected graph components; `sd` (synthesis density) penalizes over-reliance on synthetic edges.
- **HypothesisEngine — Multi-Path Abductive Reasoning (Phase 50)**: Accepts an observed graph state and generates ranked explanatory hypotheses via multi-path reverse traversal fused with Noisy-OR probability aggregation. Training-free.
- **ResearchAgent + ExternalValidator (Phases 51–52)**: Autonomous background daemon that monitors graph connectivity, proposes novel edges targeting structurally under-connected nodes, and queues them for human approval. `ExternalValidator` scores proposals against PubMed, ClinicalTrials.gov, arXiv, and OpenAlex in real time.
- **Observability Dashboard (Phase 53–54)**: `RingBufferHandler` in-process circular log capture with `GET /logs` REST interface. `StudioEngine` separates business logic from UI framework (Gradio) for full unit testability. Density-driven adaptive beam parameter selection (`AdaptiveSearchStrategy`) measures local graph density at query time to dynamically set `beam_width` and `max_hop`.

---

### GraphSAGE Neighbourhood Smoothing (Phase 55)

**Problem**: Base entity embeddings (random or sentence-transformer) encode only surface-level semantics, ignoring the structural context of each entity's position in the graph.

**Solution**: `smooth_with_graphsage()` applies a single mean-aggregation pass over each entity's neighbourhood at inference time, enriching embeddings with structural context. No training required — it runs in O(E) time and improves the CSA semantic similarity term. Activated via `CerebrumGraph.build(use_graphsage=True)`.

**Biological analogy**: Hebbian co-activation — neurons that fire together wire together. An entity's representation is enriched by the representations of the entities it is directly connected to, without any supervised signal.

---

### AAAK-Steered Traversal with Durable Memory (Phases 45, 55, 57)

**Problem**: Each query starts cold — the traversal has no memory of which reasoning chains have historically been productive.

**Solution**: `AAAKCache` accumulates compressed relation-sequence patterns from successful queries. `AAAKBeamTraversal` biases beam pruning toward these patterns via a multiplicative score boost: `effective_score = score × (1 + aaak_strength × affinity)`. The cache persists to disk on shutdown and is restored on startup (two-tier: saved JSON → QueryLog replay), giving the system permanent memory of effective reasoning strategies. `QueryLog` appends every query to an NDJSON history file; `replay_into_cache()` warms `AAAKCache` on restart.

**Biological analogy**: Long-term potentiation in hippocampal circuits — frequently-used reasoning pathways are strengthened and persist across sessions, analogous to memory consolidation during sleep.

---

### Fault Tolerance Hardening (Phases 56–57)

**Problem**: Any uncaught exception in traversal, persistence, or multiprocessing results in a 500 error or silent stream termination, losing all intermediate work.

**Solution**: Every failure mode is now independently isolated:

- Traversal crashes return `partial=True` HTTP 200 responses with intermediate results collected in `BeamTraversal._partial_paths` per-hop checkpoints.
- Persistence write failures (QueryLog, AAAKCache) are independently caught — neither can crash the `/query` endpoint.
- The `/query/stream` endpoint yields a terminal error NDJSON chunk on traversal failure rather than silently terminating.
- `GlobalRebalancer._rebalance_worker_inner` has a top-level crash guard preventing background thread death.
- `best_of_n_dscf` falls back from `ProcessPoolExecutor` to sequential execution with a WARNING log on any executor failure.
- `AAAKCache.save()/load()/save_if_path()` provides full persistence; lifespan `try/finally` saves the cache on server shutdown.

---

## The Larger Vision

CEREBRUM is built on a single observation: the structural principles that make Transformer attention effective over sequences also apply to graphs — and graphs are the natural representation of real-world knowledge.

The algorithms in this release (DSCF + CSA + Beam) establish the foundation. The streaming pipeline, Bridge Twins, and STDP demonstrate that the foundation is extensible to continuous, adaptive, and causally-aware reasoning without modifying the core algorithms.

The goal is a reasoning engine that learns the structure of any domain automatically, reasons over it transparently, and grows more capable with experience — without labeled training data, without opaque model weights, and without the possibility of fabricating facts it was never told.

That is what it means to reason from a glass box.

---

*Bryan Alexander Buchorn (AMP) · v2.0.1 / Phase 57 · April 2026*
*Contact: bryan.alexander@buchorn.com*

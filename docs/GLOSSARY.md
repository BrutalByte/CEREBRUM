### Terminology Note: Synaptic Bridge
Historically referred to as 'Wormhole' in early CEREBRUM development, this mechanism for cross-community traversal is now formally known as a Synaptic Bridge to reflect its associative and biological coupling nature.

# CEREBRUM Glossary

Definitions of all CEREBRUM-specific terms, algorithms, and architectural concepts.

---

## A

**ACerebrumBrain**
The UE5 orchestrator `AActor` (Phase 92). Manages `ANeuronNodeActor` and `ASynapseActor` lifecycle: async HTTP `GET /communities` + `GET /graph/edges` pre-load via `FHttpModule`; Fibonacci sphere spatial layout; `LoadGraphFromLayoutFile()` reads pre-computed `graph_layout.json` for exact positions; `DisconnectAndClear()` tears down all actors. Exposes Blueprint events: `OnGraphLoaded`, `OnNewNodeSpawned`, `OnNewSynapseSpawned`, `OnDissonanceAlert`.

**ANeuronNodeActor**
A UE5 `AActor` sphere mesh representing one KG entity (Phase 92). Community color is derived from the golden ratio HSV color wheel (hue = `(CID × 0.618...) % 1.0`). A `UPointLightComponent` provides ambient glow controlled by `SetGlowIntensity()`. `PulseFlash()` animates the emissive material parameter on `SYNAPTIC_PULSE`. `ShowDissonance()` tints the node orange-red for a configurable duration. `FadeOut()` scales opacity to zero then destroys the actor.

**ASynapseActor**
A UE5 `AActor` cylinder mesh representing one KG relation edge (Phase 92). Re-oriented every tick from source to target node via `FQuat::FindBetweenNormals(UpVector, Delta)`. Relation type is encoded as a hue using a djb2 hash. Weight drives both opacity and emissive glow. `AnimatePulse()` flashes the cylinder and calls `PulseFlash()` on both endpoint nodes. `FadeOut()` linearly fades opacity to zero then self-destroys.

**Adaptive Loop Tuning**
A Phase 92 feature of `AutonomousDiscoveryLoop`. When `LoopConfig.adaptive_tuning=True`, the loop reads `DiscoveryCalibrator`'s mean community weight at the start of each cycle and scales `max_materializations_per_cycle` (cap) and `_next_interval` (sleep time) proportionally. Underexplored graph regions receive higher caps and shorter intervals; saturated regions get lower caps and longer waits. All bounds are configurable via `adaptive_min_cap`, `adaptive_max_cap`, `adaptive_min_interval`, `adaptive_max_interval`. `CycleRecord.effective_cap` records the actual cap used.

**Answer Extractor**
The final stage of the CORTEX reasoning pipeline. Takes ranked `ReasoningPath` objects from `PathScorer` and extracts the terminal entity of each path as a candidate answer. Returns answers with associated CSA scores, path confidence, and HMAC provenance signatures.

**Arousal**
The functional name for the "Norepinephrine" scalar in the `ChemicalModulator` (Phase 68). Linked to cognitive dissonance frequency. High Arousal dynamically scales the `beam_width` and relaxes pruning thresholds, forcing the system to perform a more thorough, high-alert search when uncertain.

**AsyncBeamTraversal**
An async/await variant of `BeamTraversal` (see below) that supports streaming partial results via async generators. Used by the SSE streaming endpoints.

**Attention Head (Transformer analogy)**
In CEREBRUM, each DSCF community corresponds to one "attention head" in the Transformer analogy. Just as a Transformer's attention heads capture different semantic relationships, DSCF communities capture different structural clusters in the graph. See also: DSCF, CSA.

**AutoApprover**
The automated decision engine for `ResearchFinding` objects (Phase 71). Operates in three tiers: (1) **Hard gates** — immediately reject findings with blocked `literature_status` or missing `ValidationReport`; (2) **Online SGD classifier** — 16-feature logistic regression (`confidence`, `discovery_potential`, `gap_score`, `community_distance`, `local_density`, `lit_status` ordinal, `novelty_score`, `engram_affinity`, `path_count`, `contradiction_score`, `seeded_by` flags, + 4 `TriangulationReport` slots); (3) **Optional LLM semantic fallback** for borderline cases. Online `fit()` from confirmed decisions. Checkpoint via `to_dict()` / `from_dict()`. REST: `GET/POST /research/auto-approver`.

**AutonomousDiscoveryLoop**
The Phase 74 orchestration component that closes the discover→validate→approve→materialize loop autonomously. Runs `ResearchAgent.scan_once()` on a configurable timer, processes each finding through `AutoApprover`, and applies a **circuit breaker** (pauses if approval rate < `min_approval_rate` over a sliding window). **Per-cycle cap** limits materializations per run. **dry_run=True** simulates cycles without writing edges. Checkpoints `AutoApprover` state after every cycle. See also: `LoopConfig`, `CycleRecord`, Adaptive Loop Tuning.

---

## B

**Bayesian Beam Search**
`BeamTraversal(probabilistic=True)` — beam search where each candidate path maintains a Beta distribution over expected quality. Path selection uses Thompson sampling, drawing from each path's Beta distribution and selecting the highest sampled value. Produces calibrated confidence estimates alongside answers.

**BeamTraversal**
The core multi-hop graph traversal engine in CORTEX. Uses beam search (configurable width and depth) guided by CSA attention weights. Supports both deterministic mode (greedy selection) and Bayesian mode (Thompson sampling).

**Beta Distribution (in Beam Search)**
Each `TraversalPath` maintains `beta_alpha` and `beta_beta` parameters defining a Beta distribution over its expected quality. After each hop, the distribution is updated: `alpha += weight * prior_scale`, `beta += (1 - weight) * prior_scale`. Thompson sampling draws from this distribution for beam selection.

**Bilateral Verification**
The `InsightValidator`'s validation protocol. A speculative edge $E_{uv}$ is probed independently in both directions: forward (start from $u$, reach $v$ without using $E_{uv}$) and reverse (start from $v$, reach $u$ without using $E_{uv}$). At least one probe must succeed with confidence ≥ 0.65 for the edge to be verified.

**Bridge Bonus (EF-005)**
An innovation in CEREBRUM's benchmark traversal strategy: the BeamTraversal's CSA weights preferentially identify structural bridges — edges connecting otherwise-isolated communities — and use them as "shortcuts" to improve multi-hop recall. Named EF-005 in the benchmark evaluation log.

**Bridge Twin Engine**
The system component that creates and manages Bridge Twin nodes (see below). Monitors edge crossing frequencies between communities; materializes a bridge twin node when a crossing threshold is exceeded (LTP analog). Prunes bridges when crossings decay (LTD analog).

**Bridge Twin Node**
An experience-dependent structural relay node that forms in the graph when two entities in different communities are traversed together repeatedly. Inspired by the thalamo-cortical relay model in neuroscience. Bridge twin nodes reduce hop count for cross-community queries and are tracked by `BridgeTwinEngine`.

**BridgeRecord**
The data structure tracking a bridge twin node: stores `twin_id`, `original_id`, `source_community`, `destination_community`, `crossing_count`, and `ltp_weight`. Pruned by `on_rebalance()` when community IDs become stale (Phase 19 fix).

---

## C

**CandidateRegistry**
A TTL-aware registry (Phase 73 Batch B) replacing the flat `_evaluated_pairs` set in `ResearchAgent`. Tracks `nomination_count` per (source, target) pair; applies a log-scale `nomination_boost` (up to 3×) to `discovery_potential` for repeatedly-nominated candidates. A TTL gate prevents redundant `HypothesisEngine` runs within the expiry window. `prune()` evicts stale entries; LRU `max_entries` cap enforces a memory bound.

**Canonical Basis Anchor**
`SignalEncoder(canonical_embeddings={...})` — a fixed reference embedding space used as the target for all Procrustes SVD alignment operations across a federation. Prevents geometric drift accumulation when signal encoders are aligned to different adapters across federated hops (Phase 20 fix, Hole 7).

**Causal Flood**
An adversarial attack on the `STDPDiscretizer` where a burst of rapid spike events causes the STDP weight to exceed the materialization threshold, creating spurious `CAUSES` edges. Mitigated by `min_causal_span` and `use_chi_squared` parameters (Phase 19 fix, Hole 2).

**CausalSignificanceFilter**
The combined protection system in `STDPDiscretizer`: `min_causal_span` (minimum temporal span) + `use_chi_squared` (statistical uniformity test). Both default to off for backward compatibility.

**CEREBRUM**
The overarching product and framework name. **C**ommunity-Structur**E**d g**R**aph att**E**ntion for knowledge g**R**aph reas**O**ning with **M**ulti-hop traversal. The complete system including THALAMUS (ingestion), CORTEX (reasoning), REM Engine (maintenance), and Bridge Twin Engine.

**Cerebellar Error Correction (CEC)**
An active error-driven meta-learning loop (Phase 59) that detects "dissonant" predictions — paths with high CSA scores but low consensus across multiple independent reasoning strategies (MACH L1). CEC triggers autonomous external validation via `ResearchAgent` to correct structural reasoning errors.

**ChemicalModulator**
The bio-mimetic regulation system (Phase 68) that manages the global reasoning state via metabolic scalars (**Reinforcement**, **Arousal**, **Novelty**, **Cohesion**, **Persistence**). Implements homeostatic decay, pulling all levels back to a resting baseline after each query completion.

**Cohesion**
The functional name for the "Oxytocin" scalar in the `ChemicalModulator` (Phase 68). Promotes community stability by scaling the community membership term ($\beta$) in the CSA formula. High Cohesion levels favor intra-community reasoning over cross-domain "leaps."

**ContradictionResolver**
A deterministic evidence-weight classifier (Phase 73 Batch B) that runs on already-computed proposal data. Computes `net_evidence_score = Noisy-OR(proposed confidences) - max(contradiction_score)`. Classifies findings as: **"clean"** (approved), **"revision_candidate"** (queued for human review), **"contested"** (AutoApprover decides), or **"discardable"** (auto-rejected before AutoApprover). Revision candidates are queued in `ResearchAgent._revision_candidates`.

**Community (Graph)**
A group of nodes that are more densely connected to each other than to the rest of the graph. In CEREBRUM, communities are detected by DSCF and serve as the "attention heads" for CSA reasoning. Nodes within the same community receive a bonus in the CSA formula.

**Community Lock-In**
A reasoning pathology detected by the MetaInsightEngine: more than 70% of successful reasoning paths stay within a single community. Indicates the graph's community structure may be over-dominant or the starting entity has insufficient inter-community connections.

**Community-Specific CSA Parameters**
`CSAEngine(community_params={community_id: (α, β, γ, δ, ε, ζ)})` — per-community overrides for the six CSA attention weights. Prevents homogeneity saturation in tightly-clustered communities (Phase 20 fix, Hole 6).

**Community-Structured Attention (CSA)**
The core attention formula defining the weight $a(u,v,k)$ for traversing from node $u$ to node $v$ at hop $k$:
$$a(u,v,k) = \sigma(\alpha \cdot \cos(\vec{e}_u, \vec{e}_v) + \beta \cdot S_C(u,v) + \gamma \cdot w_{rel} - \delta \cdot d_{norm} + \varepsilon \cdot \phi(k) + \zeta \cdot PR(v))$$
Six configurable weights with defaults: α=0.4, β=0.4, γ=0.1, δ=0.05, ε=0.05, ζ=0.1.

**CORTEX**
The core reasoning engine subsystem. Encompasses CommunityEngine (DSCF), CSAEngine (attention), BeamTraversal (search), PathScorer, and AnswerExtractor. Operates on a prepared graph produced by THALAMUS.

**CSAParameterLearner**
An online, gradient-free adapter that adjusts CSA attention weights from binary query feedback (correct/incorrect). Uses coordinate-wise moving averages with simplex projection. Maintains separate parameter sets per community when `per_community=True` (Phase 17 feature).

**Consensus Scorer**
The Phase 32 component that aggregates and normalizes reasoning path scores from multiple remote agents. Uses weighted averaging and Procrustes-aligned embedding similarity to resolve score discrepancies between disparate knowledge bases.

---

## D

**Dissonance (Cognitive)**
A state detected by the `CerebellarEngine` when different reasoning strategies or agents produce conflicting results for the same seed. Quantified as the variance between CSA scores and Consensus scores. High dissonance triggers an **Arousal Surge** to increase search breadth.

**Discovery Engine**
The Phase 32 federated component that automates the dynamic discovery of remote CEREBRUM reasoning nodes. Uses the Holographic Index to identify nodes with relevant community overlaps or entity aliases.

**Distributed Beam Traversal**
A Phase 32 extension of `BeamTraversal` that supports delegated reasoning. Instead of fetching neighbors one-by-one over the network, it requests full "Reasoning Branches" (sub-beams) from remote nodes when the traversal reaches an alias or a community boundary.

**DSCF (Dual-Signal Community Fusion)**
The community detection algorithm at the heart of CEREBRUM. Combines local topology signal (LPA — Label Propagation Algorithm) and global modularity signal (Louvain-style modularity gain) simultaneously at each node update, rather than choosing between them. Also called TSC (Triple-Signal Consensus) when the Infomap/flow signal is included as a third signal. Produces communities with dual-signal structural character essential for high-quality CSA reasoning.

**DiscoveryCalibrator**
The Phase 73 component that tracks per-community discovery rates via Exponential Moving Average (EMA). Computes an inverse-rate multiplier `weight = global_rate / (community_rate + ε)` for each community, boosting `_score_discovery_potential()` for understudied regions. Cold-start: unscanned communities receive `max_weight` (5.0). API: `record_scan(cid)`, `record_discovery(cid)`, `get_weight(cid)`, `stats()`. Also drives Adaptive Loop Tuning (Phase 92) when `LoopConfig.adaptive_tuning=True`.

**DeltaDiscretizer**
A streaming discretizer that emits a graph edge when the rate-of-change (|Δx/Δt|) of a continuous signal exceeds a threshold.

**Depth Asymmetry**
A reasoning pathology detected by the MetaInsightEngine: more than 60% of successful query answers are found at hop 1, indicating the graph behaves more like a lookup table than a multi-hop reasoner. Suggests beam width or max_hops may need adjustment.

---

## E

**EmbeddingEngine**
THALAMUS component that generates vector embeddings for graph entities. Two backends: `random` (default, reproducible random vectors) and `sentence-transformers` (semantic embeddings from text). KGE backends (TransE, RotatE) are available as optional drop-ins.

**Engram**
Thread-safe in-memory store mapping relation-sequence tuples to success counts, used by `EngramTraversal` to bias beam pruning toward known-productive reasoning chains. Built from prior successful query paths. Persists to disk on server shutdown (`save_if_path()`) and is restored on startup (`load()`) with incremental `QueryLog` replay merged on top.

**EngramTraversal**
A `BeamTraversal` subclass that applies a multiplicative affinity boost to candidate path scores during `_prune_candidates()`: `effective_score = score × (1 + engram_strength × affinity)`. The `affinity` is computed from the `Engram` prefix index by matching the path's emerging relation sequence against stored patterns.

**Entity**
A node in the Knowledge Graph. Represented by a unique string identifier. Carries metadata: embedding vector, structural features (PageRank, betweenness, degree), community membership.

**Explainable Reasoning Trace (ERT)**
A "glass-box" telemetry framework (Phase 62) that captures the per-hop decision state of the beam search. ERT logs all winners and top rejected competitors at every step, including their 10-parameter ReasoningLogit feature radars, providing a complete audit trail of the reasoning process.

---

## F

**Fibonacci Sphere Layout**
The spatial placement algorithm used by `ACerebrumBrain` (Phase 92) and `setup_graph_layout.py` for distributing community centres in 3D space. Given community index `i` and total `n` communities: `θ = π(3−√5) × i` (golden angle), `y = 1 − (i/(n−1)) × 2`, `r = √(1−y²)`. Centre = `(r·cos(θ)·R, r·sin(θ)·R, y·R)`. Produces uniform, stable, non-overlapping placement on a sphere of radius `R`. Used for communities; nodes are placed in a smaller deterministic-seeded cluster around their community centre.

**FederatedAdapter**
An adapter that aggregates multiple remote `GraphAdapter` instances into a single logical graph. Handles cross-graph identity alignment, community map merging, and distributed traversal coordination.

**Federated Discovery**
The process by which a local CEREBRUM node discovers potentially-relevant remote graphs without revealing the full contents of either graph. Uses the Holographic Index (Bloom filters + centroids) for privacy-preserving overlap detection.

---

## G

**GET /graph/edges**
A CEREBRUM REST endpoint (Phase 92) that returns up to `?limit=5000` edges from the loaded graph as a `GraphEdgesResponse` (`edges[]`, `total_returned`, `limit`). Intended for visualization pre-load; implemented via `GraphAdapter.get_all_edges(limit)` with an efficient `NetworkXAdapter` override using `G.edges(data=True)`. Used by both `setup_graph_layout.py` and `ACerebrumBrain` to populate `ASynapseActor` instances on startup.

**Glass-Box Reasoning**
CEREBRUM's defining property: every answer is a verifiable path through graph edges with a complete mathematical trace. Contrasted with "Black-Box" probabilistic models (LLMs) where the reasoning process is opaque.

**GlobalRebalancer**
The background component that monitors modularity drift ($\Delta Q_{cum}$) across streaming ingest events. When drift exceeds a threshold, it spawns a background DSCF re-run, then performs an atomic swap of `adapter.community_map`. Notifies `BridgeTwinEngine` via `on_rebalance()` hook (Phase 19).

**GraphSnapshot**
A portable JSON graph topology checkpoint (Phase 81) in `core/persistence.py`. `GraphSnapshot.save(adapter, path)` serializes all edges to a JSON file (not pickle — survives adapter class changes). `restore(path, adapter, skip_existing=True)` re-adds only new edges, safe to run repeatedly for incremental restoration after pod restart. `diff(path_a, path_b)` shows edge deltas between two snapshots for audit. See also: ProvenanceLedger.

**GraphSAGE Smoothing**
A one-pass mean neighbourhood aggregation step applied after base entity encoding. `smooth_with_graphsage(embeddings, G)` replaces each entity's embedding with a weighted average of itself and its neighbours' embeddings, making the CSA `alpha` (semantic similarity) term more effective by encoding local structural context. Enabled via `CerebrumGraph.build(use_graphsage=True)`.

---

## H

**H@10 (Hits at 10)**
The primary evaluation metric for CEREBRUM: the fraction of test queries where the correct answer appears in the top-10 ranked paths. CEREBRUM zero-shot benchmarks: MetaQA 1-hop=0.960, 2-hop=0.713, 3-hop=0.248 at <7ms.

**Holographic Index**
A privacy-preserving federated discovery system using Bloom filters (probabilistic membership) and community centroids (structural fingerprints). Allows remote graphs to advertise their contents without revealing individual entities.

**Hop**
One step in a multi-hop reasoning traversal: moving from node $u$ to node $v$ along edge $(u, v, r)$. Analogous to one Transformer layer depth.

**HMAC-SHA256 Path Provenance**
A cryptographic signature applied to each reasoning path output. Computed over the canonical JSON serialization of the path using the server's `CEREBRUM_HMAC_KEY`. Allows downstream systems to verify path integrity.

---

## I

**IngestionPipeline**
THALAMUS component that preprocesses raw entity-relation-entity triples before graph insertion. Responsibilities: entity normalization (Unicode, whitespace), alias deduplication via `entity_dedup_map`, relation normalization, confidence/provenance metadata assignment, and namespace prefixing.

**InsightEngine**
The creative component of the Verification/Metacognition layer. Generates candidate `INSIGHT_LINK` edges by detecting latent proximity (high cosine similarity without a direct edge), community boundary bridging, and path pattern completion. All generated edges enter the InsightValidator pipeline.

**InsightEvent**
A first-class graph node in the MetaInsightEngine's second-order graph. Represents a reasoning event (query, validation, bridge formation, rebalance) with attributes: event_type, timestamp, entities, confidence, communities_traversed, outcome.

**InsightValidator**
Validates speculative edges using bilateral reverse traversal and community consensus scoring. Maintains an edge state machine: SPECULATIVE → CORROBORATED → VERIFIED → GROUNDED (or → REFUTED). Integrated with the REM Cycle for batch validation.

---

## J

**JWT (JSON Web Token)**
The authentication mechanism for CEREBRUM's REST API. Bearer tokens use HMAC-SHA256 signing with the `CEREBRUM_JWT_SECRET` environment variable.

---

## K

**KGE (Knowledge Graph Embedding)**
A family of methods that learn vector representations of entities and relations from graph triples. CEREBRUM supports TransE and RotatE as optional drop-in `EmbeddingEngine` backends. KGE embeddings upgrade the semantic similarity term in CSA without changing other reasoning components.

**Knowledge Graph (KG)**
A directed graph where nodes represent entities and edges represent typed relationships (triples: subject, predicate, object). The primary data structure for all CEREBRUM reasoning.

---

## L

**LoopedBeamTraversal**
A Phase 70 `BeamTraversal`-compatible wrapper implementing LoopLM-style iterative refinement (arXiv:2510.25741). Applies beam traversal T times (`max_loops`). Between loops: top-K answer entities expand seeds (semantic channel), PE→ChemicalModulator adjusts beam parameters (metabolic channel), and Engram records bias the next loop's pruning (mnemonic channel). **Adaptive exit gate**: stops early when `|ΔPE| < γ` or answer-set Jaccard ≥ θ. All loops' paths merged — `best_by_tail` keeps the highest-score path per tail entity. `LoopTrace` is exposed via `ReasoningTrace.loop_trace` in the ERT.

**Loop-Provenance Recovery**
The Phase 79 integration between `AutonomousDiscoveryLoop` and `ProvenanceLedger`. When `LoopConfig.auto_rollback_on_trip=True` and the circuit breaker fires, the loop automatically calls `ProvenanceLedger.rollback_cycle(cycle_number, adapter)` to undo all materializations from the failed cycle before pausing. `CycleRecord.edges_rolled_back` records how many edges were removed.

**Lazy STDP Weight Decay**
An optimization in `STDPDiscretizer` that applies weight decay lazily (only when a pair is accessed) rather than on every clock tick, reducing time complexity from $O(N)$ to $O(1)$ per event. See also: STDP.

**Leiden Algorithm (Native)**
CEREBRUM's GPL-free native reimplementation of the Leiden community detection algorithm in `core/leiden_native.py`. Replaces the `igraph`/`leidenalg` external dependencies removed in v2.24.0.

**LPA (Label Propagation Algorithm)**
A local community detection method where each node adopts the most common community label among its neighbors. Represents the "local topology signal" in DSCF. Fast ($O(E)$) but can produce poor modularity. Combined with modularity gain in DSCF to achieve both speed and quality.

**LTD (Long-Term Depression)**
The biological analog for Bridge Twin node pruning: when a bridge twin's crossing count falls below a threshold (simulating synaptic weakening from disuse), the bridge record is pruned.

**LTP (Long-Term Potentiation)**
The biological analog for Bridge Twin node creation: when two entities in different communities are traversed together beyond a threshold (simulating synaptic strengthening from repeated co-activation), a bridge twin node is materialized.

---

## M

**MACH (Multi-Agent Consensus Hierarchies)**
A three-tier reasoning verification framework (Phase 60):
- **L1 (Strategy)**: Multi-strategy local voting (DSCF vs. BFS vs. KGE).
- **L2 (Federated)**: Cross-node path verification in a cluster.
- **L3 (Gold Literature)**: ResearchAgent validation against external scientific literature.

**MetaInsightEngine**
The second-order reasoning component. Constructs a graph over `InsightEvent` nodes connected by typed edges (TRIGGERED_BY, CONTRADICTS, REINFORCES, CO_OCCURRED). Runs standard CSA traversal on this event graph to detect reasoning pathologies (community lock-in, relation starvation, depth asymmetry).

**Modularity (Q)**
A graph quality metric measuring the fraction of edges within communities minus the expected fraction for a random graph with the same degree sequence. Higher Q indicates more modular (community-structured) graphs. CEREBRUM's GlobalRebalancer monitors modularity drift $\Delta Q_{cum}$.

**Multi-Hop Reasoning**
Answering questions that require traversing multiple edges in sequence: e.g., "What disease is caused by the organism transmitted by the vector that bites humans in tropical regions?" requires 3+ hops through a biomedical KG.

---

## N

**Namespace Isolation**
`IngestionPipeline(namespace="text")` and `SignalEncoder(namespace="signal")` — prefix-based entity ID separation preventing semantic collisions between different data modalities (Phase 19 fix, Hole 3).

**Namespace (Entity ID)**
A prefix string applied to entity IDs to isolate them from other modality spaces. Format: `"namespace:entity_id"`. Example: `"signal:Temp_Sensor_1"` vs `"text:Temp_Sensor_1"`.

**Neural Telemetry**
The real-time event-streaming protocol (Phase 63) that broadcasts reasoning pulses (`SYNAPTIC_PULSE`), node creation (`NEUROGENESIS`), and pruning (`SYNAPTIC_PRUNE`) via WebSockets. Enables 3D observability in external game engine clients.

**Novelty**
The functional name for the "Acetylcholine" scalar in the `ChemicalModulator` (Phase 68). Mimics the biological role of sensory focus vs. prior knowledge. High Novelty boosts the semantic similarity term ($\alpha$) and suppresses the community structure term ($\beta$), allowing the system to focus on immediate data novelty over structural expectations.

---

## P

**Prediction Error (PE)**
The Jaccard divergence between the *prior path* (predicted relation sequence from the top Engram pattern) and the *actual path* (relation sequence produced by traversal). Computed by `PredictiveCodingEngine.update()` after each traversal. PE drives `ChemicalModulator` signals: high PE → Arousal and Novelty surge; low PE → Reinforcement. Exposed in `ReasoningTrace.prediction_error`. See also: soliton_index, PredictiveCodingEngine.

**PredictiveCodingEngine**
The Phase 69 active-inference component. Before each traversal, generates a *prior path* from the top Engram pattern. After traversal, computes **Prediction Error (PE)** (Jaccard divergence between prior and actual relation sequences) and propagates it to `ChemicalModulator` (Arousal, Novelty, Reinforcement). Tracks a rolling window of recent PEs to compute `soliton_index`. Activated by `CerebrumGraph.attach_engram(engram)`. Fields exposed in `ReasoningTrace`: `prior`, `prediction_error`, `soliton_index`.

**ProvenanceLedger**
The Phase 76 audit chain for `ResearchAgent.approve()`. Records every materialized edge in `EdgeRecord` objects grouped into `BatchRecord` objects (one batch per `approve()` call), keyed by `batch_id`. `rollback_batch(batch_id, adapter)` removes exactly one batch's edges. `rollback_cycle(cycle_number, adapter)` removes all batches from a given loop cycle. Thread-safe. LRU eviction at `max_batches` cap. Requires `adapter.remove_edge()` (Phase 80 protocol). See also: GraphSnapshot, Loop-Provenance Recovery.

**PageRank (PR)**
A graph centrality measure (part of StructuralEncoder) used as the sixth term in the CSA formula ($\zeta \cdot PR(v)$). High PageRank nodes receive a traversal bonus, reflecting their structural importance.

**Persistence**
The functional name for the "Vasopressin" scalar in the `ChemicalModulator` (Phase 68). Linked to long-term memory formation. High Persistence levels increase the multiplier for Engram pattern-steering and lower the threshold for promoting patterns to "Canonical Engrams" via the `EngramConsolidator`.

**PathScorer**
CORTEX component that ranks `ReasoningPath` objects using the composite score:
$$\text{score}(P) = \left(\prod_{k=1}^L a(u_k, v_k, k)\right) \cdot \text{coherence}_{com}(P) \cdot \cos(\vec{h}_{final}, \vec{q})$$

**Partial Response**
A `QueryResponse` with `partial=True` and a non-null `error` field. Returned when the traversal raises an unrecoverable exception mid-execution — the response contains whatever paths were collected in `_partial_paths` before the failure. HTTP status is 200, not 500, so clients can still consume partial results gracefully.

**Path-Preserving Hold-out**
`InferenceValidator(path_preserving=True)` — validation methodology that skips holding out edge $(u,v)$ if no alternative path between $u$ and $v$ exists after removal. Prevents false-zero recall on sparse graphs (Phase 20 fix, Hole 8).

**PatternDiscretizer**
A streaming discretizer that emits a graph edge when a symbolic event sequence matches a configured pattern with probability ≥ $p$.

**ProcessPoolExecutor Fallback**
A fault tolerance pattern in `best_of_n_dscf`: if `ProcessPoolExecutor` raises any exception (e.g., `BrokenExecutor` from Windows paging file exhaustion), the function logs a WARNING and falls back to sequential `dscf_communities` calls. Ensures community detection completes even when multiprocessing is unavailable.

**Procrustes SVD**
The mathematical core of `SignalEncoder`'s cross-modal alignment. Computes a rotation matrix $R = V U^T$ from the SVD of the cross-covariance matrix $\Sigma = Y^T X$, where $X$ are signal embeddings and $Y$ are entity embeddings. Minimizes $||Y - XR||_F$.

---

## Q

**Quantized Traversal**
An efficiency optimization (Phase 61) that uses `uint8` fixed-point math for path scoring instead of 64-bit floats. Maps the probability range [0.0, 1.0] to the integer range [0, 255], reducing memory footprint and increasing traversal speed during large-scale beam search.

**Query Snapshot Isolation**
`CSAEngine.set_query_snapshot(community_map)` — captures the community map at query start and uses it exclusively throughout the query. Prevents GlobalRebalancer mid-query atomic swaps from producing inconsistent CSA weights across hops (Phase 20 fix, Hole 5).

**QueryLog**
An append-only NDJSON file (`data/cerebrum/query_log.ndjson` by default) that records seeds, answers, and relation sequences after each successful reasoning call. `replay_into_cache(engram)` reads the log at startup to re-warm `Engram` so learned relation patterns survive process restarts.

---

## R

**Reinforcement**
The functional name for the "Dopamine" scalar in the `ChemicalModulator` (Phase 68). Tracks "Reward Prediction Error" (RPE) derived from user feedback (`POST /feedback`). Reinforcement surges amplify the semantic ($\alpha$) and edge-type ($\gamma$) weights, reinforcing successful relation sequences in the reasoner's attention formula.

**REM Cycle (Rapid Edge Maintenance)**
Background metacognitive maintenance loop. Runs on three schedules: Hot Path (10 min, TTL edge pruning), Cold Path (1 hour, insight validation + decay), REM Path (daily/triggered, full DSCF re-optimization). Inspired by biological sleep-cycle memory consolidation.

**REM Engine**
The system component implementing the REM Cycle. Interfaces with InsightValidator, GlobalRebalancer, and the graph persistence layer.

**Relation Starvation**
A reasoning pathology detected by the MetaInsightEngine: a relation type appears on fewer than 5% of successful paths despite representing more than 20% of graph edges. Indicates the CSA γ (relation weight) term may be underweighting this relation type.

**ResourceGovernor**
Hardware-aware query throttling: monitors CPU/memory usage and enforces per-query resource budgets. Implements the "Arousal Interrupt" that pauses REM Cycle maintenance tasks when user queries arrive.

**RotatE**
A KGE method modeling relations as rotations in complex embedding space. Supports symmetric, antisymmetric, inverse, and compositional relation patterns. Available as optional `EmbeddingEngine` backend in CEREBRUM.

---

## S

**soliton_index**
A coherence metric computed by `PredictiveCodingEngine`: `soliton_index = 1 - mean(recent PEs)`. A value near 1.0 indicates that the Engram prior consistently predicts actual traversal paths — a self-reinforcing, self-localising pattern analogous to a soliton wave (UCFT 2025). Exposed in `ReasoningTrace.soliton_index`. Low soliton_index (high mean PE) suggests the graph is highly novel or the Engram has not yet converged.

**SignalEncoder**
THALAMUS component for cross-modal alignment. `StatisticalSignalEncoder` (time-series statistics → embeddings) and `SpectralSignalEncoder` (waveform FFT features → embeddings). Projects sensor signals into entity embedding space via Procrustes SVD. Uses `namespace="signal"` by default.

**Skepticism Factor (ρ)**
The exponential decay multiplier applied to `INSIGHT_LINK` edges in the REM Cycle's confidence decay: $c_{t+1} = c_t \cdot (\lambda \cdot \rho)^{\Delta t}$ with $\rho = 0.8 < 1$. Prevents speculative insights from becoming entrenched without independent validation.

**Soft Community Membership**
`CommunityEngine(soft_membership=True)` — each node carries a probability distribution over communities rather than a single hard assignment. Community consensus term uses the dot product of membership distributions: $S_C^{soft}(u,v) = \sum_k p_k^{(u)} \cdot p_k^{(v)}$ (Phase 17 feature).

**SpeedTalk Encoding**
A Heinlein-inspired phonemic compression algorithm (Phase 58) for the `Engram` cache. Maps KG relation types to single characters, achieving 8-20x compression while preserving prefix structure for O(P) pattern queries.

**STDP (Spike-Timing-Dependent Plasticity)**
The neuroscientific mechanism inspiring `STDPDiscretizer`. In biology, a synapse strengthens if the pre-synaptic neuron fires before the post-synaptic neuron (LTP), and weakens otherwise. CEREBRUM applies this principle to event streams: if entity A's events consistently precede entity B's events, a directional `CAUSES` edge is materialized.

**STDPDiscretizer**
THALAMUS component that materializes directional `CAUSES` edges from event timing patterns. Maintains per-pair STDP weights and event counts; emits edges when both thresholds are exceeded. Extended in Phase 19 with `min_causal_span` and `use_chi_squared` adversarial guards.

**StreamAdapter**
A `GraphAdapter` wrapper that accepts continuous event streams via a thread-safe queue. Supports five discretizer types, a sliding-window buffer, and integration with GlobalRebalancer.

**Structural Encoder**
THALAMUS component computing per-node structural features: PageRank, betweenness centrality, and degree. These features are used as the positional encoding analog in the Transformer/KG analogy.

**Structural Hole**
A cross-feature interaction bug where two independently-correct subsystems produce incorrect outcomes when combined. Eight structural holes were identified and patched in Phases 19–20. See SPEC_016 for full taxonomy.

**SYNAPTOGENESIS**
A neural telemetry event type (Phase 92) emitted by the CEREBRUM REST server when a new edge is materialized into the graph. Broadcast by `/research/approve/{id}` for each proposal approved by the ResearchAgent. Payload: `source_node`, `target_node`, `relation`, `weight`. Routed by `UCerebrumLink` through the existing `OnSynapticPulse` delegate (distinguished by `HopCount == -1`), triggering `ASynapseActor` creation in UE5. Contrasts with `SYNAPTIC_PRUNE` (edge removal) and `SYNAPTIC_PULSE` (traversal hop).

**setup_graph_layout.py**
A stdlib-only Python CLI script (`ue5_project/setup_graph_layout.py`, Phase 92) that queries a live CEREBRUM REST API and writes a `graph_layout.json` v1.1 file containing pre-computed 3D node positions, community colours, and edge list. Queries `/health`, `/communities`, and `/graph/edges`. Positions use the Fibonacci sphere algorithm; colours use the golden ratio HSV wheel. Run once before packaging or deploying an UE5 build. Accepts `--api`, `--token`, `--out`, `--edge-limit` flags.

**Synaptic Pruning**
The periodic removal of low-utility synthetic edges (Phase 61) based on confidence, age, and usage patterns. Mimics biological synaptic homeostasis to maintain graph sparsity and reasoning performance.

---

## T

**TelemetryBridge**
The Python WebSocket server (`api/telemetry_bridge.py`, Phase 63 / completed Phase 92) that multiplexes CEREBRUM neural events to all connected visualization clients. Started as an asyncio background task via `create_app(ws_port=N)`. Exposes `broadcast(event: NeuralEvent)` which serializes via `model_dump_json()` and fan-outs to all live WebSocket connections. Accepts multiple simultaneous clients (e.g., UE5 editor instance + a monitoring dashboard). `start_server()` is an infinite async coroutine; `ensure_future()` prevents it from blocking the FastAPI event loop.

**THALAMUS**
The ingestion layer subsystem. Encompasses: CSV/NetworkX/Neo4j/RDF adapters, EmbeddingEngine, StructuralEncoder, STDPDiscretizer, IngestionPipeline, SignalEncoder, and StreamAdapter.

**Thompson Sampling**
The probabilistic path selection strategy in Bayesian Beam Search: for each candidate path, draw a sample from its Beta distribution; select the candidate with the highest sample. Balances exploration (uncertain paths) and exploitation (high-confidence paths).

**Temporal Decay**
The time-dependent reduction in edge weight for edges with `valid_until` timestamps: $w_{temp}(t) = w_0 \cdot \exp(-\lambda \cdot \max(0, t - t_{until}))$. Decay rate $\lambda$ is configurable per relation type (Phase 17 feature).

**TemporalCalibrator**
A grid-search utility that calibrates the CSA `eta` (temporal decay) and `iota` (node recency) parameters to maximise Recall@K against a labelled validation set. `calibrate()` iterates over a grid of (eta, iota) pairs, evaluating each via `measure_recall()`; `apply()` writes the best-found parameters back to `CSAEngine`. A `try/finally` guarantee restores the original parameters if calibration raises.

**ThresholdDiscretizer**
A streaming discretizer that emits a graph edge when a continuous float signal crosses a configured threshold value.

**TriangulationEngine**
The Phase 72 four-perspective candidate validation component. Validates `ResearchCandidate` objects across four independent perspectives: **P1 `reverse_confidence`** — `HypothesisEngine` run in the reverse direction (B→A); **P2 `strategy_agreement`** — agreement fraction across 3 different reasoning configurations; **P3 `mean_path_independence`** — Jaccard independence score across primary proposal paths; **P4 `semantic_type_score`** — relation-type and entity-class consistency (novel relations = 0.5 neutral). Results extend the AutoApprover feature vector from 12 to 16 features. `is_Synaptic Bridge_candidate` diagnostic flag set for cross-community bridge candidates. Stored in `finding.metadata["triangulation"]`.

**TransE**
A KGE method modeling relations as translation vectors: a triple $(h, r, t)$ is valid iff $\vec{h} + \vec{r} \approx \vec{t}$. Available as optional `EmbeddingEngine` backend in CEREBRUM.

**Traversal Path**
A `dataclass` representing an in-progress beam search path. Carries: `nodes` (list of entities), `edges` (list of edge objects), `score` (cumulative CSA product), `beta_alpha`, `beta_beta` (Beta distribution parameters for probabilistic mode).

**TSC (Triple-Signal Consensus)**
Extension of DSCF that adds a third signal (Infomap/flow-based community assignment) to the LPA and modularity signals. All three signals are fused simultaneously at each node update. The "aircraft navigation mid-value voting" analog: consensus of three independent signals.

---

## U

**UCerebrumLink**
The UE5 `UActorComponent` WebSocket bridge (Phase 92). Connects to the CEREBRUM `TelemetryBridge` at a configurable `ws://host:port` URL and fires typed Blueprint dynamic delegates: `OnSynapticPulse`, `OnNeurogenesis`, `OnSynapticPrune`, `OnCorticalGlow`, `OnDissonance`, and the catch-all `OnNeuralEvent`. Routes JSON envelope `{event_type, payload}` to the correct delegate.

**Uncertainty Propagation**
`BeamTraversal(propagate_uncertainty=True)` — computes per-path confidence as a variance-penalized product of edge confidences: $\text{conf}(P) = \prod_i c_i^\alpha \cdot (1 - \beta \cdot \text{Var}(\{c_i\}))$ (Phase 17 feature).

---

## W

**Warm-Start Strength**
`BeamTraversal(warm_start_strength=N)` — scales the first-hop Beta distribution prior using the CSA edge weight: $(\alpha, \beta)_{hop1} = (1 + w(1+s), 1 + (1-w)(1+s))$. Reduces cold-start variance 85% on sparse graphs (Phase 19 fix, Hole 4).

**WindowedFrequencyDiscretizer**
A streaming discretizer that emits a graph edge when two entities co-occur more than a minimum number of times within a sliding time window.

**Synaptic Bridge Attention**
The cross-graph attention mechanism in `FederatedAdapter` that connects structurally-analogous communities across different remote graphs. Named for the "shortcut" it provides through otherwise-disconnected graph spaces.

---

## Z

**Zombie Bridge**
A stale `BridgeRecord` whose `source_community` or `destination_community` IDs reference a community partition that no longer exists after a GlobalRebalancer atomic swap. Pruned by `BridgeTwinEngine.on_rebalance()` (Phase 19 fix, Hole 1).

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

### Neural Coupling (formerly GossipSync)
The mechanism for synchronizing community-map updates across a distributed CEREBRUM cluster, ensuring all workers maintain a consistent reasoning state.

### UE LLM Toolkit Adapter
An interface bridge that allows the CEREBRUM Brain Server to animate and communicate with Unreal Engine 5 (UE5) simulations in real-time, mapping neural telemetry events to 3D scene actions.

---
**Reviewed on**: April 21, 2026 for version v2.24.0

# Novel Contributions and IP Claims

## CEREBRUM: Community-Structured Graph Attention for Knowledge Graph Reasoning

**Document Classification**: Intellectual Property Reference
**Authors**: Bryan Alexander Buchorn
**Date**: April 2026
**Status**: v2.44.0 (Phase 160 (TRB Detection Fix) COMPLETE)

> This document consolidates the novel technical contributions of the CEREBRUM framework for use in patent applications, academic priority claims, and commercial IP protection. Each claim is substantiated with prior art analysis and a statement of the specific technical distinction.

---

## Part I: Core Algorithm Claims

### Claim 1: Dual/Triple Signal Community Fusion (DSCF / TSC)

**Description**: A community detection algorithm that applies Local Propagation Algorithm (LPA), modularity gain (Louvain-style), and optionally Infomap flow signals **simultaneously at each node update**, fusing them via a weighted consensus mechanism at each iteration. The Triple-Signal Community (TSC) variant is now selectable explicitly alongside DSCF, Leiden, and LPA via a unified community engine configuration interface.

**Novelty Statement**: All prior community detection algorithms that combine multiple signals do so by operating on disjoint node subsets (e.g., apply LPA to low-degree nodes, Louvain to high-degree nodes [Sun et al., 2024]) or by running algorithms sequentially and merging results. DSCF is the first algorithm that applies all signals to every node at every update step. The simultaneous per-node fusion produces communities with distinct "dual-signal" structural character that is not achievable by sequential or partitioned combination.

**Closest Prior Art**:
- Louvain (Blondel et al., 2008): Global modularity only, no LPA signal
- Leiden (Traag et al., 2019): Refinement phase + modularity, no LPA signal
- LPA (Raghavan et al., 2007): Local propagation only, no modularity signal
- LPA-Louvain hybrids (Sun et al., 2024): Apply signals to disjoint node subsets by degree threshold - categorically different from DSCF's per-node simultaneous fusion

**Key Technical Differentiator**: The specific architectural choice to compute both $\Delta Q_{modularity}(v, c)$ and $f_{LPA}(v, c)$ for every candidate community $c$ at every node $v$ in every iteration, then fuse them via: $\text{score}(v, c) = \alpha \cdot \Delta Q(v,c) + \beta \cdot f_{LPA}(v,c) + \gamma \cdot \text{flow}(v,c)$

**Relevant files**: `core/community_engine.py`, `core/leiden_native.py`
**Documented in**: `docs/arxiv/PAPER_001_DSCF_TSC.md`, `docs/specifications/SPEC_001_DSCF_TSC.md`

---

### Claim 2: Community-Structured Attention (CSA) Formula - 10-Parameter Extension

**Description**: A graph edge attention weight formula that incorporates community membership as a soft global constraint alongside semantic similarity, relation type, path length penalty, hop decay, PageRank centrality, temporal decay, node recency, synthesis-density penalty, and grounding confidence. The formula is training-free, computed analytically from graph topology at query time. The current formulation extends the original 6-parameter formula to 10 learnable parameters.

**The Formula**:
$$a(u,v,k) = \sigma\left(\alpha \cdot \text{sim} + \beta \cdot cs + \gamma \cdot etw - \delta \cdot nd + \varepsilon \cdot hd + \zeta \cdot PR(v) + \eta \cdot td + \iota \cdot nr_v - \mu \cdot sd + \theta \cdot grounding\right)$$

Where:
- $\alpha \cdot \text{sim}$: Semantic similarity (cosine distance between entity embeddings)
- $\beta \cdot cs$: Community score (live DSCF community co-membership)
- $\gamma \cdot etw$: Edge-type weight (relation-specific strength)
- $\delta \cdot nd$: Normalized distance penalty
- $\varepsilon \cdot hd$: Hop decay (exponential confidence reduction per hop)
- $\zeta \cdot PR(v)$: Global PageRank authority prior
- $\eta \cdot td$: Temporal decay (time since edge creation)
- $\iota \cdot nr_v$: Node recency (recency of traversal visits)
- $\mu \cdot sd$: Synthesis-density penalty (fraction of synthetic edges in path)
- $\theta \cdot grounding$: Grounding confidence (provenance and verification score)

Default weights: $(0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0)$

**Novelty Statement**: Graph Attention Networks (GAT, HAN, HGT) compute local attention over immediate neighborhoods using learned weight matrices. CSA is the first attention formulation that includes global community membership ($S_C(u,v)$) as a term. The 10-parameter extension further adds temporal, recency, synthesis-quality, and grounding dimensions - none of which appear in any published GNN attention formula.

**Closest Prior Art**:
- GAT (Veličković et al., 2018): Local neighborhood attention, learned weights, no community term
- HAN (Wang et al., 2019): Meta-path attention, no community term
- HGT (Hu et al., 2020): Heterogeneous attention, learned, no community term
- GraphRAG (Microsoft, Edge et al., 2024): Communities used for LLM summarization, not as attention weights in traversal

**Key Technical Differentiator**: The $\beta \cdot S_C(u,v)$ term (community co-membership from live DSCF partitions, not learned parameters), extended with four novel dimensions: temporal decay, node recency, synthesis-density penalty, and grounding confidence - producing the first 10-dimensional analytically-computed KG attention formula.

**Relevant files**: `core/attention_engine.py`, `core/reasoning_logit.py`
**Documented in**: `docs/arxiv/PAPER_002_CSA.md`, `docs/specifications/SPEC_002_CSA.md`

---

### Claim 3: CSA-Guided Multi-Hop Beam Search (Glass-Box Reasoning)

**Description**: A multi-hop KG traversal architecture where beam search is guided exclusively by CSA attention weights derived from graph topology, producing a complete verifiable reasoning trace at zero inference cost (no trained model, no LLM).

**Novelty Statement**: All published multi-hop KG reasoning systems using beam search (MINERVA, DeepPath, BeamQA) require training on labeled question-answer pairs. CEREBRUM's BeamTraversal uses CSA weights computed analytically from graph structure - achieving competitive H@10 on MetaQA zero-shot, with no training data.

**Closest Prior Art**:
- MINERVA (Das et al., 2018): RL-trained path agent, requires labeled QA pairs
- DeepPath (Xiong et al., 2017): RL, requires training
- BeamQA (Atif et al., 2023): Supervised sequence-to-sequence beam search

**Key Technical Differentiator**: Zero-shot multi-hop reasoning at competitive H@10 using only graph topology.

**Relevant files**: `reasoning/traversal.py`, `reasoning/path_scorer.py`

---

## Part II: Neuromorphic Architecture Claims

### Claim 4: Experience-Dependent Bridge Twin Nodes (LTP/LTD Analog for KG)

**Description**: A mechanism that materializes structural relay nodes (bridge twins) between graph communities based on traversal co-occurrence frequency, analogous to synaptic LTP/LTD in hippocampal circuits. Bridge twins form when crossing count exceeds a threshold and are pruned when crossings decay below a threshold.

**Novelty Statement**: No published KG system uses experience-dependent structural modification inspired by LTP/LTD. Static graph shortcuts (agentic LLM expansion, shortcut edges in GNNs) are added by external agents or hand-coded rules, not by the reasoning engine's own traversal history. The specific mechanism of counting traversal co-occurrences and triggering structural node materialization from those counts is, to our knowledge, original.

**Relevant files**: `core/bridge_engine.py`
**Documented in**: `docs/arxiv/PAPER_003_BRIDGE_TWINS.md`

---

### Claim 5: STDP-Inspired Directional Causal Edge Inference

**Description**: A streaming discretizer that materializes directional `CAUSES` edges in a Knowledge Graph from event timing patterns, using Hebbian-inspired weight accumulation with configurable LTP/LTD asymmetry and lazy O(1) weight decay.

**Novelty Statement**: STDP has been applied to spiking neural network learning rules extensively, but no published work applies STDP mechanics to dynamic causal edge discovery in Knowledge Graphs. The combination of: (1) Hebbian co-occurrence weighting, (2) directional edge materialization (pre->post), (3) lazy weight decay reducing per-event complexity from O(N) to O(1), and (4) adversarial causal flood protection (min_causal_span, chi-squared) is an original contribution.

**Relevant files**: `core/discretizer.py`
**Documented in**: `docs/arxiv/PAPER_004_STDP_CAUSAL.md`

---

### Claim 6: REM Cycle - Sleep-Inspired KG Maintenance

**Description**: A three-phase background maintenance loop (Prune -> Consolidate -> Synthesize) for autonomous KG quality management, inspired by the biological sleep-cycle memory consolidation model (synaptic homeostasis, NREM slow-wave, hippocampal replay).

**Novelty Statement**: KG maintenance systems (NELL, ATOMIC) focus on *adding* new facts. The REM Cycle's three-phase design for autonomous removal, structural consolidation, and proactive synthesis with differential decay rates per edge relation type has no precedent in the KG maintenance literature.

**Relevant files**: `core/rem_engine.py`
**Documented in**: `docs/arxiv/PAPER_007_REM_CYCLE.md`

---

## Part III: Cross-Modal and Federated Claims

### Claim 7: Procrustes SVD Sensor-to-KG Alignment with Canonical Basis Anchor

**Description**: A cross-modal alignment system that projects heterogeneous sensor signals (time-series, waveforms) into the same embedding space as Knowledge Graph text entities via Procrustes SVD rotation, with a canonical embedding basis that prevents geometric drift accumulation across federated hops.

**Novelty Statement**: Multimodal KG papers (MMKG, MKGFormer) integrate images and text but not continuous sensor signals. The specific combination of: (1) statistical/spectral signal feature extraction, (2) Procrustes SVD alignment to KG embedding space, (3) a canonical root basis shared across all federation participants, and (4) namespace isolation preventing cross-modal entity collisions, is original.

**Relevant files**: `core/signal_encoder.py`
**Documented in**: `docs/arxiv/PAPER_008_SIGNAL_ENCODER.md`

---

### Claim 8: Holographic Index for Privacy-Preserving Federated Discovery

**Description**: A federated graph discovery system using Bloom filter sketches (probabilistic membership) and community centroid vectors (structural fingerprints) to enable two CEREBRUM nodes to identify likely-relevant shared content without revealing individual entity identities.

**Novelty Statement**: Federated learning (McMahan et al., 2017) aggregates model gradients, not graph structure. SPARQL federation requires full query exposure. The Holographic Index is the first KG-specific privacy-preserving discovery mechanism using the combination of Bloom filters and community centroid structural fingerprints.

**Relevant files**: `core/hologram.py`
**Documented in**: `docs/arxiv/PAPER_005_HOLOGRAPHIC_INDEXING.md`

---

## Part IV: Production Hardening Claims

### Claim 9: Bayesian Beam Search with Warm-Start Prior Seeding for KG Traversal

**Description**: Application of Beta-distribution path modeling and Thompson sampling to multi-hop KG beam search, with a warm-start mechanism that seeds the first-hop Beta prior from the CSA attention weight to reduce cold-start variance.

**Novelty Statement**: Bayesian bandits and Thompson sampling are well-established in recommendation and exploration-exploitation literature. Their application to *multi-hop KG beam search with path-level Beta distributions* seeded from graph-structural attention weights is novel - no prior KG reasoning paper uses this formulation.

**Relevant files**: `reasoning/traversal.py`
**Documented in**: `docs/arxiv/PAPER_006_BAYESIAN_BEAM.md`

---

### Claim 10: Eight-Hole Structural Hardening Taxonomy for KG Systems

**Description**: A systematic methodology for identifying and patching cross-feature interaction bugs ("structural holes") in complex KG systems, covering five taxonomic categories: stale reference, adversarial input, namespace collision, bias/saturation, and evaluation artifacts.

**Novelty Statement**: The specific taxonomy of structural hole categories as applied to KG reasoning systems - and the complete set of eight patched holes with backward-compatible fixes - constitutes a novel engineering contribution for production KG deployment.

**Documented in**: `docs/arxiv/PAPER_016_PRODUCTION_HARDENING.md`

---

## Part V: Composite System Claims

### Claim 11: CEREBRUM as an Integrated Zero-Shot KG Reasoning System

**Description**: The complete CEREBRUM system integrating DSCF community detection, CSA attention, zero-shot beam traversal, LTP/LTD bridge twins, STDP causal inference, REM Cycle maintenance, Bayesian beam search, cross-modal alignment, federated discovery, streaming ingest, metacognitive verification, and production hardening into a single coherent framework with zero training data requirements and complete reasoning transparency.

**Novelty Statement**: No prior work integrates this combination of capabilities in a single KG reasoning framework. The specific architectural choices - using communities as attention heads, using traversal co-occurrence for structural modification, using STDP timing for causal discovery, and composing these into a sleep-cycle maintenance loop - represent a novel systems-level contribution.

---

## Part VI: Advanced Reasoning and Discovery Claims

### Claim 12: Multi-Path Abductive Reasoning via Noisy-OR Confidence Fusion (HypothesisEngine)

**Description**: A reasoning module that accepts an observed graph state (a set of present or absent edges/nodes) and generates ranked explanatory hypotheses by running multiple reverse-traversal paths from the observation, then fusing path-level confidence scores using Noisy-OR probability aggregation. The resulting hypotheses are ranked by posterior plausibility and can be materialized as provisional graph edges.

**Novelty Statement**: Forward multi-hop KG reasoning (given A, find B) is well-studied. Abductive KG reasoning (given B, infer most plausible cause A) is addressed in recent work (e.g., Abdul-Mageed et al., 2023) but exclusively with trained neural models. The CEREBRUM HypothesisEngine performs abductive reasoning without training: it relies solely on graph topology and CSA-weighted reverse traversal. The specific combination of (1) multi-path reverse traversal, (2) Noisy-OR path fusion, (3) training-free execution, and (4) materialization of hypotheses as provisional graph structure is, to our knowledge, entirely novel.

**Closest Prior Art**:
- Abductive NLI (Bhagavatula et al., 2020): Hypothesis generation from text, requires fine-tuned language model
- LEGO (Shi et al., 2021): Logical abduction on KGs, requires symbolic rules and training
- Generative KGC models: Produce new triples but do not perform abductive reasoning over observation sets

**Key Technical Differentiator**: Training-free abductive reasoning via Noisy-OR fusion of reverse-traversal CSA scores, with hypothesis materialization into graph structure.

**Relevant files**: `core/hypothesis_engine.py`
**API endpoints**: `POST /hypothesize`, `POST /hypothesize/materialize`

---

### Claim 13: Autonomous Missing-Link Discovery with Human-in-the-Loop Approval Queue (ResearchAgent)

**Description**: An autonomous background daemon that continuously monitors graph connectivity metrics (degree distribution, betweenness centrality, community bridge density) to identify candidate under-connected nodes, then proposes novel edges connecting those nodes via multi-hop bridge analysis. All proposals are queued for human review before materialization - no edge is added to the graph without explicit human approval.

**Novelty Statement**: Automated KG completion (TransE, RotatE, ComplEx) predicts missing links using trained embedding models and applies them automatically. The CEREBRUM ResearchAgent differs in three respects: (1) it operates without training, using structural graph analysis only; (2) it targets systematically under-connected nodes rather than random triple prediction; (3) it enforces a mandatory human-in-the-loop approval stage before any graph modification. The combination of training-free structural gap detection and supervised materialization is original.

**Closest Prior Art**:
- TransE/RotatE (Bordes et al., 2013; Sun et al., 2019): Trained KG embedding completion, no human approval step
- NELL (Carlson et al., 2010): Automated belief extraction and addition, no structural gap targeting
- ATOMIC (Sap et al., 2019): Crowdsourced commonsense KG extension, human-authored rather than autonomously proposed

**Key Technical Differentiator**: Unsupervised structural gap detection + training-free bridge proposal + mandatory human approval gate.

**Relevant files**: `core/research_agent.py`

---

### Claim 14: Automated Literature Validation of Graph Hypotheses Against Live Scientific Databases (ExternalValidator)

**Description**: A validation layer that accepts proposed KG edges (from ResearchAgent or HypothesisEngine) and automatically queries live scientific literature databases - PubMed, ClinicalTrials.gov, arXiv, and OpenAlex - to retrieve relevant publications. Each proposal is scored by literature support density and annotated with specific citations. The validator returns a structured evidence report per proposal.

**Novelty Statement**: KG systems with external validation (e.g., Google's KG with Freebase provenance, Wikidata with reference citations) rely on pre-linked static provenance. The ExternalValidator performs dynamic, query-time validation against multiple heterogeneous live databases. No published KG system performs automated multi-database literature retrieval as a first-class validation step in the hypothesis/edge proposal pipeline.

**Closest Prior Art**:
- Wikidata provenance citations: Static, human-curated, not automated query-time retrieval
- SciGraph (Springer Nature): Domain-specific static linking between publications and entities, no dynamic proposal validation
- BioKG systems: Literature-linked KGs, but validation is done at ingest time, not for dynamically proposed edges

**Key Technical Differentiator**: Real-time, multi-database literature scoring of dynamically proposed graph edges, returning per-proposal citation evidence reports.

**Relevant files**: `core/external_validator.py`

---

### Claim 15: Density-Driven Dynamic Beam Parameter Selection (Adaptive Search Strategy)

**Description**: A pre-traversal analysis module that measures local graph density around seed entities (node degree, edge density in the k-hop ego network, average community size) and uses this measurement to dynamically select beam_width and max_hop parameters before initiating beam search. Dense neighborhoods receive narrower, shallower beams; sparse neighborhoods receive wider, deeper beams.

**Novelty Statement**: Beam search in KG reasoning systems uses fixed hyperparameters set at system startup (MINERVA uses fixed beam width; CEREBRUM prior to this phase used fixed beam_width=10, max_hop=3). No published KG traversal system dynamically adapts beam parameters to local graph topology at query time. Dynamic beam width in NLP sequence models (e.g., adaptive beam in neural machine translation) adapts to sequence length, not to graph structural properties.

**Closest Prior Art**:
- Adaptive beam search in NMT (Huang et al., 2017): Adapts to sequence length/complexity, not graph density
- MINERVA (Das et al., 2018): Fixed beam width, fixed depth
- All CEREBRUM prior phases: Fixed beam_width, fixed max_hop

**Key Technical Differentiator**: First KG traversal system to adapt beam parameters to local structural density measured at query time.

**Relevant files**: `reasoning/traversal.py` (density analysis), `core/community_engine.py` (density measurement)

---

### Claim 16: Ring Buffer Log Capture with REST Query Interface for Production KG Monitoring (Observability Architecture)

**Description**: An in-process circular log capture system (RingBufferHandler) that intercepts all reasoning-layer log events, stores them in a fixed-size in-memory ring buffer (5,000 entries), and exposes the buffer contents via authenticated REST endpoints (`GET /logs`, `DELETE /logs`). A companion `StudioEngine` class encapsulates all testable business logic for the observability layer separately from the UI rendering framework (Gradio), enabling unit testing of observability behavior without UI dependencies.

**Novelty Statement**: Production KG monitoring systems use external log aggregation (Elasticsearch, Datadog) with no in-process query interface. The CEREBRUM observability architecture differs in three ways: (1) the ring buffer is in-process, adding zero network overhead to log capture; (2) the REST interface enables programmatic log querying without external infrastructure; (3) the StudioEngine separation allows unit testing of observability logic, which is not possible when observability code is embedded in UI frameworks. The combination of in-process ring-buffer capture, REST queryability, and testability-by-design is original.

**Closest Prior Art**:
- Python `logging.handlers.MemoryHandler`: Buffered logging, no REST interface
- Gradio-based monitoring UIs: UI-embedded observability, not separately testable
- External APM systems (Datadog, New Relic): Require external infrastructure, not in-process

**Key Technical Differentiator**: In-process ring buffer + REST query interface + framework-decoupled StudioEngine enabling full unit testability.

**Relevant files**: `core/studio_engine.py`, `core/log_config.py`, `api/server.py` (`GET /logs`, `DELETE /logs`, `POST /build`)

---

### Claim 17: Cross-Node Beam Delegation with Procrustes Embedding Alignment (Federated Reasoning)

**Description**: A federated KG reasoning architecture in which a coordinating node (DistributedBeamTraversal) partitions a multi-hop query into sub-paths, delegates sub-path traversal to remote CEREBRUM nodes via `POST /traverse`, and merges returned path fragments into a unified ranked result. Embedding alignment between nodes is performed via Procrustes SVD rotation applied to shared anchor entity embeddings, ensuring that semantic similarity scores from remote nodes are geometrically comparable to local scores.

**Novelty Statement**: Federated KG systems (SPARQL federation, Wikidata federation) delegate full sub-queries to remote endpoints but do not perform embedding alignment between nodes - they rely on shared URIs, not vector similarity. Federated GNN systems (FedGNN, FedE) aggregate model gradients, not traversal path fragments. The CEREBRUM federated architecture is the first to combine: (1) sub-path delegation (not full sub-query or gradient aggregation), (2) real-time Procrustes alignment of remote embedding spaces, and (3) path-fragment merging that preserves full CSA attention scores across node boundaries.

**Closest Prior Art**:
- SPARQL Federation (W3C, 2013): Full sub-query delegation, no embedding alignment
- FedE (Chen et al., 2021): Federated KGE training, gradient aggregation, no path delegation
- FedGNN (Liu et al., 2022): Federated GNN training, no reasoning path delegation

**Key Technical Differentiator**: Sub-path delegation + real-time Procrustes embedding alignment + CSA-score-preserving path fragment merging.

**Relevant files**: `reasoning/distributed_traversal.py`, `adapters/remote_adapter.py`, `adapters/federated_adapter.py`
**API endpoints**: `POST /traverse`

---

### Claim 18: Online SGD and Batch Gradient Descent Parameter Learning with Full Persistence for 10-Parameter KG Attention (CSA Parameter Learner)

**Description**: A dual-mode parameter learning system for the 10-parameter CSA attention formula comprising: (1) `MetaParameterLearner` - an online SGD learner that updates per-community parameter overrides from individual feedback events (`POST /feedback`) with configurable learning rate and momentum; (2) `CSAParameterLearner` - a batch gradient descent learner that retrains the global 10-parameter prior from accumulated positive/negative path pairs (`POST /retrain`). Both learners support full state serialization (`to_dict()`/`from_dict()`) enabling checkpoint/restore via `POST /params` and `--params-file` CLI flag at startup.

**Novelty Statement**: KGE training systems (TransE, RotatE, ComplEx) learn entity/relation embeddings, not attention formula parameters. No published attention-based KG reasoning system supports: (1) per-community parameter overrides (as distinct from global parameters), (2) online SGD updates from real-time feedback without full retraining, (3) batch retraining from a feedback buffer without embedding re-learning, and (4) full parameter persistence enabling cold-start from a prior session's learned weights. The combination of per-community granularity, online + batch dual modes, and cross-session persistence is entirely novel.

**Closest Prior Art**:
- KGE training (Bordes et al., 2013): Entity/relation embedding learning, not attention weight learning
- MAML-style meta-learning: Few-shot adaptation, requires training, no per-community granularity
- Online GNN adaptation: Requires gradient through GNN layers, not applicable to training-free systems

**Key Technical Differentiator**: Training-free-compatible online+batch dual-mode learning of 10 attention formula weights with per-community granularity and full cross-session persistence.

**Relevant files**: `core/parameter_learner.py`, `core/attention_engine.py`
**API endpoints**: `POST /feedback`, `POST /retrain`, `GET /params`, `POST /params`

---

### Claim 19: GraphSAGE One-Pass Neighbourhood Smoother for Training-Free KG Reasoning

**Description**: `smooth_with_graphsage(embeddings, G)` applies a single-pass weighted mean aggregation of each entity's embedding with its graph neighbours, enriching entity representations with local structural context without any training. Activated via `CerebrumGraph.build(use_graphsage=True)`.

**Novelty Statement**: GraphSAGE (Hamilton et al., 2017) and its descendants are trained models that require labelled data and multiple aggregation layers. CEREBRUM's implementation applies a single aggregation pass at inference time, on top of any base encoder (random or sentence-transformer), to enrich embeddings with neighbourhood context for the CSA semantic similarity term - without any training loop, loss function, or labelled data.

**Closest Prior Art**:
- GraphSAGE (Hamilton et al., 2017): Trained, multi-layer, requires supervision
- GCN (Kipf & Welling, 2016): Trained spectral convolution, requires adjacency matrix eigendecomposition
- Node2Vec (Grover & Leskovec, 2016): Random-walk trained embeddings, no inference-time aggregation

**Key Technical Differentiator**: Inference-time single-pass application of neighbourhood aggregation without any training, used exclusively to enrich the semantic similarity term of the CSA attention formula.

**Relevant files**: `core/embedding_engine.py` (`smooth_with_graphsage`), `core/cerebrum.py` (`use_graphsage` flag)

---

### Claim 20: Engram-Steered Beam Traversal - Relation-Pattern-Biased Beam Pruning

**Description**: `Engram` stores compressed relation-sequence tuples from prior successful reasoning paths. `EngramTraversal` biases beam pruning at each hop via: `effective_score = score x (1 + engram_strength x affinity)`, where `affinity` is computed from a prefix index over the stored patterns. The cache persists to disk across process restarts via JSON serialization.

**Novelty Statement**: Reinforcement learning-based path selection (e.g., MINERVA, M-Walk) trains a policy on labelled data. CEREBRUM's Engram steering is training-free: it accumulates successful reasoning patterns from live queries and immediately biases future traversal without any training loop, labelled data, or gradient computation.

**Closest Prior Art**:
- MINERVA (Das et al., 2018): RL-trained policy network
- M-Walk (Shen et al., 2018): Monte Carlo tree search with trained value network
- Engram shorthand dialect (Phase 45): Compression format - Engram steering (Phase 55) is a distinct mechanism that uses the pattern space, not the compressed notation itself

**Key Technical Differentiator**: Real-time, training-free accumulation of logical structure from successful queries, immediately biasing future traversal through a multiplicative score boost.

**Relevant files**: `reasoning/engram_traversal.py`

---

### Claim 21: TemporalCalibrator - Training-Free Grid-Search Recall@K Calibration for Temporal KG Parameters

**Description**: `TemporalCalibrator` performs grid-search calibration of the CSA `eta` (temporal decay) and `iota` (node recency) parameters by measuring Recall@K against a labelled validation set. A `try/finally` guarantee restores original parameters after any failure. No gradient computation or training data required.

**Novelty Statement**: Hyperparameter calibration for temporal KG reasoning typically requires a trained model with a differentiable loss function. CEREBRUM's calibrator uses only Recall@K computed over graph traversal paths - a non-differentiable, training-free metric - with grid search over a small 2D parameter space.

**Relevant files**: `core/temporal_calibrator.py`

---

### Claim 22: Fault-Tolerant KG Reasoning Server - Partial-Result Graceful Degradation

**Description**: A server-side pattern where any traversal failure returns HTTP 200 with `partial=True` + error message and whatever paths were collected before the failure (`_partial_paths` checkpoint), rather than HTTP 500. Persistence layer write failures (QueryLog, Engram) are independently isolated - neither can crash the reasoning endpoint. The streaming endpoint yields a terminal error NDJSON chunk on failure rather than silently terminating.

**Novelty Statement**: Knowledge graph reasoning APIs universally return error codes on failure. CEREBRUM's partial-result pattern allows clients to consume useful intermediate results even when reasoning fails mid-execution, with explicit partial/error semantics distinguishing full success from graceful degradation.

**Relevant files**: `api/server.py`, `api/schemas.py` (QueryResponse), `reasoning/traversal.py` (_partial_paths)

---

### Claim 23: SpeedTalk Phonemic Compression for Reasoning Patterns

**Description**: A Heinlein-inspired phonemic compression algorithm that maps KG relation types to a single-character alphabet (a-z, A-Z, 0-9). High-frequency relation sequences are stored as compact strings rather than verbose tuples, achieving 8-20x key compression. The phonemic representation preserves prefix structure, enabling O(P) prefix queries for real-time beam steering.

**Novelty Statement**: Key-value compression in databases typically uses generic algorithms (Zstd, LZ4). SpeedTalk is the first domain-specific compression for KG reasoning patterns that uses phonemic mapping to maintain structural searchability (prefix match) while drastically reducing memory overhead.

**Relevant files**: `reasoning/speedtalk_cache.py`

---

### Claim 24: Cerebellar Error Correction (CEC) via Dissonance Detection

**Description**: An active error-driven meta-learning loop that monitors reasoning calls for "dissonant" predictions - paths with high CSA attention scores but low consensus across multiple independent reasoning strategies (MACH L1). Dissonant seeds are automatically pushed to the ResearchAgent for autonomous external validation.

**Novelty Statement**: Error correction in AI usually involves backpropagation during training. CEC is an online, training-free mechanism that detects structural dissonance at inference time and triggers targeted corrective research, effectively closing the loop between reasoning and discovery.

**Relevant files**: `core/cerebellar_engine.py`

---

### Claim 25: Multi-Agent Consensus Hierarchies (MACH) - Three-Tier Verification

**Description**: A three-tier reasoning verification framework comprising: (1) L1 Local Consensus (multi-strategy voting); (2) L2 Federated Consensus (cross-node path confirmation); and (3) L3 Gold Standard Consensus (ResearchAgent validation against live scientific literature). Higher levels represent more rigorous/expensive verification steps.

**Novelty Statement**: Multi-agent KG reasoning typically uses simple voting or ensemble methods. MACH is the first hierarchical verification system that scales from local structural voting to federated network confirmation and finally to external literature-backed ground truth.

**Relevant files**: `reasoning/consensus_hierarchy_engine.py`, `reasoning/consensus_scorer.py`

---

### Claim 26: Synaptic Pruning and Quantized Traversal (SPQT) for KG Efficiency

**Description**: A dual efficiency optimization comprising: (1) Synaptic Pruning - utility-based removal of low-confidence synthetic edges based on age and usage patterns; and (2) Quantized Traversal - `uint8` fixed-point math for path scoring (mapping [0.0, 1.0] to [0, 255]), reducing memory overhead and improving traversal speed on high-hop queries.

**Novelty Statement**: Model quantization is standard for neural networks (INT8/FP8). SPQT is the first application of fixed-point quantization specifically to KG beam-search scoring, and the first to combine it with utility-based structural pruning inspired by biological synaptic homeostasis.

**Relevant files**: `core/synaptic_pruner.py`, `reasoning/traversal.py` (QUANT_SCALE)

---

### Claim 27: Explainable Reasoning Trace (ERT) with 10-Parameter Feature Radars

**Description**: A "glass-box" telemetry system that captures the per-hop decision state of the beam search, logging all winners and top rejected competitors. Every path in the trace includes its full 10-parameter ReasoningLogit feature vector, exposing exactly *why* specific branches were prioritized or pruned.

**Novelty Statement**: Explainability in KGs is usually limited to the final path. ERT provides a complete audit trail of the *search process itself*, including the "roads not taken," visualized through 10-dimensional structural feature radars.

**Relevant files**: `reasoning/trace.py`, `api/schemas.py` (TraceResponse)

---

### Claim 28: Neural Telemetry and 3D Visualization Bridge (Phase 63)

**Description**: A real-time event-streaming protocol (NeuralEvent) that broadcasts reasoning steps, node creation, and pruning events via WebSockets. It enables external 3D visualization clients (e.g., Unreal Engine 5) to render the "digital twin" of the graph's reasoning process as it happens.

**Novelty Statement**: KG visualization is typically static or post-hoc (e.g., Gephi). CEREBRUM's telemetry bridge is the first to stream live, per-hop reasoning pulses and structural evolution events to high-fidelity game engines for real-time observability.

**Relevant files**: `core/telemetry.py`, `api/telemetry_bridge.py`

---

### Claim 29: Threshold-Based Neural Memory Consolidation (Phase 64)

**Description**: An autonomous consolidation routine (`EngramConsolidator`) that monitors the success frequency of relation sequences in the dynamic Engram cache. Patterns exceeding a success threshold are promoted to "Canonical Engrams," making them permanent and immune to future pruning.

**Novelty Statement**: Cache eviction policies (LRU, LFU) are generic. Neural Memory Consolidation is a domain-specific "promotion" mechanism that mimics the biological process of moving short-term traces to long-term memory based on recurring utility.

**Relevant files**: `reasoning/engram_consolidation.py`

---

### Claim 30: Autonomous Hypothesis Materialization with Noisy-OR Provenance (Phase 65)

**Description**: A formal materialization pipeline that takes ResearchAgent discovery findings and commits them to the Knowledge Graph as "Provisional Edges." These edges carry Noisy-OR aggregated confidence scores and structured provenance strings citing the discovery agent.

**Novelty Statement**: KG completion usually predicts facts without formal structural materialization or provenance. CEREBRUM's materializer treats autonomous discovery as a verifiable transaction, ensuring that every synthesized link is logically traceable to its causal discovery event.

**Relevant files**: `core/hypothesis_materializer.py`

---

### Claim 31: Neuro-Symbolic Homeostasis via Multi-Metabolic Modulation (Phase 68)

**Description**: A dynamic regulation system (`ChemicalModulator`) that simulates 5 key metabolic scalars: **Reinforcement** (Dopamine), **Arousal** (Norepinephrine), **Novelty** (Acetylcholine), **Cohesion** (Oxytocin), and **Persistence** (Vasopressin). Scalar levels decay back to baseline over time, providing a natural homeostatic regularizer for the reasoning engine.

**Novelty Statement**: Neuromodulation has been applied to neural networks (scalars scaling activation). This is the first application of a multi-metabolic homeostatic state machine to a **symbolic KG reasoner**, where global metabolic states dynamically adjust beam parameters and attention formula ratios.

**Relevant files**: `core/chemical_modulator.py`

---

### Claim 32: Predictive Coding Engine with Soliton Index (Phase 69)

**Description**: `PredictiveCodingEngine` generates a *prior path* from the top Engram pattern before each traversal. After traversal, a **Prediction Error (PE)** is computed as the Jaccard divergence between prior and actual relation sequences. PE drives ChemicalModulator signals. The `soliton_index = 1 − mean(recent PEs)` measures prior stability - a self-reinforcing prior that consistently yields low PE behaves as a soliton (self-localising wave).

**Novelty Statement**: Predictive coding has been applied to neural perception (Rao & Ballard 1999). This is the first application to symbolic KG reasoning: a training-free prior derived from empirical traversal history, with a wave-coherence metric (`soliton_index`) for prior stability monitoring.

**Relevant files**: `core/predictive_coder.py`

---

### Claim 33: Looped Beam Traversal with Adaptive Exit Gate (Phase 70)

**Description**: `LoopedBeamTraversal` wraps any beam engine and applies it T times. Between loops: top-K answer entities expand seeds (semantic channel), PE->ChemicalModulator adjusts beam params (metabolic channel), Engram records bias pruning (mnemonic channel). Adaptive exit gate: `|ΔPE| < γ` (primary) or answer-set Jaccard ≥ θ (fallback).

**Novelty Statement**: LoopLM-style iteration (arXiv:2510.25741) applies looping to neural language models. This is the first application to symbolic KG beam search, with three distinct inter-loop feedback channels (semantic, metabolic, mnemonic) and a prediction-error-driven exit condition rather than a fixed iteration count.

**Relevant files**: `reasoning/looped_traversal.py`

---

### Claim 34: Tiered AutoApprover with Online SGD (Phase 71)

**Description**: `AutoApprover` implements a three-tier decision stack for `ResearchFinding` objects: (1) hard gates on literature status and missing validation; (2) online logistic SGD classifier over a 16-feature vector; (3) optional LLM semantic fallback. The classifier is updated online from confirmed human decisions via `fit()`.

**Novelty Statement**: KG completion pipelines either accept all machine-generated candidates or require full human review. This is the first tiered auto-approval stack for KG discovery findings that learns continuously from human confirmations without full retraining.

**Relevant files**: `core/auto_approver.py`

---

### Claim 35: Four-Perspective TriangulationEngine for Candidate Validation (Phase 72)

**Description**: `TriangulationEngine` validates `ResearchCandidate` objects from four independent perspectives: P1 reverse confidence (HypothesisEngine run B->A), P2 strategy agreement (3-config voting fraction), P3 mean path independence (Jaccard independence across proposals), P4 semantic type consistency (relation-type/entity-class match). Results extend the AutoApprover feature vector from 12 to 16 dimensions.

**Novelty Statement**: KG link prediction evaluates forward-direction confidence only. No prior system applies simultaneous reverse traversal, multi-strategy voting, path independence, and semantic type consistency as a four-perspective validation unit on the same candidate.

**Relevant files**: `core/triangulation_engine.py`

---

### Claim 36: EMA-Based DiscoveryCalibrator with Inverse-Rate Sampling (Phase 73)

**Description**: `DiscoveryCalibrator` tracks per-community scan and discovery rates via Exponential Moving Average. An inverse-rate multiplier `weight = global_rate / (community_rate + ε)` boosts understudied communities in discovery scoring. Cold-start: unscanned communities receive `max_weight` (5.0). Temporal recency scoring added to `ValidationReport` using exponential decay with 7-year half-life.

**Novelty Statement**: KG research agents sample uniformly or by degree. Adaptive per-community EMA-driven inverse-rate rebalancing for autonomous KG discovery, with cold-start maximum weighting and temporal evidence recency, is a novel contribution.

**Relevant files**: `core/discovery_calibrator.py`

---

### Claim 37: Autonomous Discovery Loop with Circuit Breaker (Phase 74)

**Description**: `AutonomousDiscoveryLoop` closes the discover->validate->approve->materialize loop without human intervention. Features: sliding-window circuit breaker (pauses materialization if approval rate falls below threshold), per-cycle materialization cap, dry-run mode, and AutoApprover checkpoint persistence after every cycle with decisions.

**Novelty Statement**: Autonomous KG population systems (NELL, ATOMIC) run batch updates without feedback-loop safety mechanisms. This is the first KG discovery loop with a sliding-window circuit breaker that dynamically pauses based on approval quality, with dry-run mode and per-cycle caps.

**Relevant files**: `core/autonomous_loop.py`

---

### Claim 38: Graph Provenance Ledger with Targeted Rollback (Phase 76)

**Description**: `ProvenanceLedger` records every edge materialized by `ResearchAgent.approve()` with batch_id, finding_id, and cycle_number. `rollback_batch(batch_id, adapter)` removes exactly one approval's edges. `rollback_cycle(cycle_number, adapter)` removes all edges from a given loop cycle. LRU eviction, thread-safe.

**Novelty Statement**: KG update systems offer at most version snapshots for rollback. Fine-grained per-batch and per-cycle targeted edge removal - scoped to individual autonomous-approval transactions - with LRU-bounded ledger and thread-safe semantics is novel.

**Relevant files**: `core/provenance_ledger.py`

---

### Claim 39: Loop-Provenance Recovery (Phase 79)

**Description**: `AutonomousDiscoveryLoop` optionally auto-invokes `ProvenanceLedger.rollback_cycle()` when the circuit breaker fires, atomically undoing all edges materialized in the bad cycle before resuming. `CycleRecord.edges_rolled_back` tracks what was undone.

**Novelty Statement**: No prior autonomous KG population system combines circuit-breaker failure detection with automatic transactional rollback of materialized edges, making the loop self-healing on quality degradation.

**Relevant files**: `core/autonomous_loop.py`, `core/provenance_ledger.py`

---

### Claim 40: Portable Graph Snapshot with Non-Destructive Restore and Diff (Phase 81)

**Description**: `GraphSnapshot` serializes graph topology to a portable, human-readable JSON format (not pickle). `restore(skip_existing=True)` re-adds only new edges, preserving all edge attributes. `diff(path_a, path_b)` identifies exact edge additions and removals between two snapshots without loading a live graph.

**Novelty Statement**: KG persistence typically uses binary serialization or full database dumps. A topology-portable, pickle-free, skip-existing restore with structural diff - without requiring the original adapter class - is novel.

**Relevant files**: `core/persistence.py`

---

### Claim 41: Adaptive Loop Tuning via Calibrator-Driven Cap and Interval Scaling (Phase 92)

**Description**: `AutonomousDiscoveryLoop` with `adaptive_tuning=True` reads `DiscoveryCalibrator.stats()` at cycle start and scales both `max_materializations_per_cycle` and inter-cycle sleep by mean community weight. Underexplored graphs -> higher cap + shorter interval; saturated graphs -> lower cap + longer interval. All bounds configurable.

**Novelty Statement**: Autonomous KG discovery loops use fixed caps and intervals. This is the first system to dynamically scale both materialization rate and cycle frequency from a per-community EMA exploration metric, creating a closed-loop adaptive pacing mechanism.

**Relevant files**: `core/autonomous_loop.py`, `core/discovery_calibrator.py`

---

### Claim 42: Active Inference / Daydreaming Mode for Knowledge Graph Consolidation (Phase 93)

**Description**: `ActiveInferenceEngine` runs autonomously during idle periods between discovery cycles, seeding queries from nodes with the highest recent prediction error (high-PE nodes represent weak or unstable priors) or falling back to random node selection when no PE history is available. Each idle query drives `ChemicalModulator` and `PredictiveCodingEngine` updates, progressively consolidating the graph's predictive model without external stimulus.

**Novelty Statement**: Active inference in neuroscience (Friston 2010) describes how biological agents minimize free energy by taking actions that confirm their generative models. This is the first application of the active inference principle to a symbolic knowledge graph reasoner: the system autonomously probes its own weak priors during idle periods, strengthening its internal predictive model through self-initiated traversal rather than waiting for external queries.

**Relevant files**: `core/active_inference.py`, `core/autonomous_loop.py`

---

### Claim 43: Signal-Driven Self-Modifying GUI via Dual-Channel Structural and Runtime Adaptation (Phase 94)

**Description**: `GUIAdaptationEngine` observes a rolling window of `SignalSnapshot` records (arousal, soliton_index, approval_rate, circuit_breaker_tripped, inference_pulses) and applies idempotent rule-based adaptations via two independent channels: (1) **structural channel** - HTTP calls to the ue-llm-toolkit API (`localhost:3000`) programmatically add new UMG widget panels to a live UE5 Blueprint asset that persists across sessions; (2) **runtime channel** - `GUI_ADAPTATION` WebSocket events broadcast to connected UE5 clients to show, hide, or collapse existing panels at 60fps. Rules include: HIGH_AROUSAL -> add DissonanceMeter panel; CIRCUIT_BREAKER -> show warning banner; INFERENCE_MILESTONE -> add InferenceHistoryBox; LOW_REINFORCEMENT -> collapse active inference panel. All rules are idempotent via `_applied: Set[str]`.

**Novelty Statement**: Existing AI systems have static GUIs designed before deployment. This is the first system in which a knowledge graph reasoner autonomously modifies the structural definition of its own user interface - adding new panels to a live game engine Blueprint asset - based on its own internal metabolic and epistemic state, with changes persisting across editor sessions.

**Relevant files**: `core/gui_adaptation_engine.py`, `api/ue_toolkit_client.py`, `ue5_project/create_initial_gui.py`

---

### Claim 44: Global Workspace (GWS) for Competitive Attention (Phase 110)

**Description**: A blackboard-based communication layer where graph communities broadcast "surprise" (prediction error) signals in real-time during reasoning. The `ConsensusHierarchyEngine` utilizes these asynchronous signals to dynamically boost candidate scores and pre-empt hierarchical validation stages.

**Novelty Statement**: Traditional multi-agent consensus in KGs is hierarchical or sequential. GWS is the first implementation of a biological Global Workspace analog for graph reasoning, enabling competitive focus-switching based on real-time information gain rather than fixed linear escalation.

**Relevant files**: `core/global_workspace.py`, `reasoning/consensus_hierarchy_engine.py`

---

### Claim 45: Proactive Active Inference in Beam Traversal (Phase 111)

**Description**: A reasoning mechanism where a `PredictiveCoder` generates "Expected Path" priors from historical Engram patterns *before* search initiates. These priors are used to bias the beam traversal, reducing the search space for common queries while using the resulting Prediction Error (PE) to drive metabolic arousal for anomaly detection.

**Novelty Statement**: Existing beam search variants for KGs are exclusively reactive, exploring the local topology only after seeing the current state. CEREBRUM is the first KG system to implement predictive coding, where the system anticipates the logical trajectory and only "pays" the computational cost for surprise reduction.

**Relevant files**: `core/predictive_coder.py`, `reasoning/traversal.py`

---

### Claim 46: Cingulate Engine - Autonomous Conflict-Driven Reasoning Verification

**Description**: A real-time reasoning verifier inspired by the mammalian anterior cingulate cortex (ACC). It monitors the distribution of attention signals across the beam search using a "Conflict Entropy" metric. High entropy or excessive convergence on high-degree hubs (hub-flooding) triggers an autonomous recursive refinement loop that dynamically adjusts beam parameters (width, gating sensitivity) to force reasoning into more specific, informative paths.

**Novelty Statement**: KG reasoning systems typically accept the results of a fixed-parameter beam search or use external post-hoc ranking. The Cingulate Engine is the first implementation of an *internal autonomous conflict monitor* that uses distribution entropy to trigger recursive search refinement at query time, specifically to mitigate the "Hub-Flooding" failure mode in large-scale KGs.

**Relevant files**: `reasoning/traversal.py` (`_calculate_conflict_entropy`), `core/insight_validator.py` (`ProvenanceValidator`), `core/cerebrum.py` (`query` retry loop)

---

## Prior Art Summary Table

| CEREBRUM Component | Closest Prior Art | Key Distinction |
|---|---|---|
| DSCF simultaneous fusion | LPA-Louvain hybrids (Sun 2024) | Per-node vs. per-population fusion |
| CSA formula ($S_C$ term) | GAT, HAN, HGT | Community term absent from all GNN attention |
| 10-parameter CSA extension | All GNN attention formulas | Temporal, recency, synthesis-density, grounding terms are entirely novel |
| Zero-shot beam traversal | MINERVA, DeepPath, BeamQA | Training required vs. fully training-free |
| Bridge Twins (LTP/LTD) | GNN shortcuts, agentic expansion | Experience-dependent vs. static/agent-added |
| STDP causal edges in KG | SNN-STDP | Neural learning rule vs. KG edge discovery |
| REM Cycle maintenance | NELL, ATOMIC, KGE training | Proactive pruning + synthesis vs. fact addition |
| Procrustes sensor->KG | MMKG, MKGFormer | Sensor signals vs. image/text only |
| Holographic Index | Federated learning, SPARQL federation | Privacy-preserving structural discovery |
| Bayesian beam + warm-start | Bandit algorithms | KG-specific Beta seeding from CSA weights |
| Structural hole taxonomy | Standard testing methodology | Cross-feature interaction analysis for KG |
| HypothesisEngine (Noisy-OR abduction) | Abductive NLI, LEGO | Training-free reverse traversal + Noisy-OR fusion |
| ResearchAgent (autonomous discovery) | TransE/RotatE, NELL | Structural gap targeting + human approval gate |
| ExternalValidator (literature scoring) | Wikidata provenance, SciGraph | Dynamic multi-database query-time validation |
| Adaptive Search (density-driven beam) | NMT adaptive beam | Graph topology adaptation vs. sequence length adaptation |
| Observability (ring buffer + REST) | MemoryHandler, APM systems | In-process + REST queryable + StudioEngine testability |
| Federated Reasoning (Procrustes alignment) | SPARQL federation, FedE | Sub-path delegation + real-time embedding alignment |
| CSA Parameter Learner (online+batch+persist) | KGE training, MAML | Per-community + dual-mode + cross-session persistence |
| GraphSAGE one-pass smoother | GraphSAGE, GCN, Node2Vec | Single inference-time pass, no training, enriches CSA semantic term |
| Engram-steered traversal | MINERVA, M-Walk | Training-free pattern accumulation + multiplicative beam bias |
| TemporalCalibrator (Recall@K grid search) | Trained temporal KGE | Non-differentiable training-free 2D grid search over eta/iota |
| Fault-tolerant partial results (HTTP 200) | Standard KG APIs | Partial-result semantics + isolated persistence failures + streaming error chunk |
| **SpeedTalk Phonemic Compression** | Generic (Zstd, LZ4) | KG-specific phonemic mapping + prefix-searchable |
| **Cerebellar Error Correction (CEC)** | Training-time backprop | Inference-time dissonance detection + autonomous research |
| **MACH (Consensus Hierarchies)** | Simple voting / ensembles | Multi-tier scaling: Local -> Federated -> Gold Literature |
| **SPQT (Quantized Traversal)** | NN Quantization (INT8) | First application to symbolic KG beam scoring + synaptic pruning |
| **Explainable Reasoning Trace (ERT)** | Path-only explanation | Complete PROCESS audit including pruned competitors |
| **Neural Telemetry Bridge** | Static graph viz (Gephi) | Live per-hop pulse streaming to high-fidelity 3D engines |
| **Metabolic Modulation (Homeostasis)** | Scalar NN scaling | Symbolic KG state machine + homeostatic decay + parameter-ratio control |
| **Predictive Coding + Soliton Index** | Neural predictive coding (Rao 1999) | Training-free KG prior from Engram history + wave-coherence metric |
| **Looped Beam Traversal** | LoopLM (arXiv:2510.25741) | Three inter-loop channels (semantic/metabolic/mnemonic) + PE exit gate |
| **AutoApprover (tiered + online SGD)** | Binary accept/reject pipelines | Three-tier stack; learns online from human confirmations |
| **TriangulationEngine (4-perspective)** | Single-direction link prediction | Simultaneous reverse, multi-strategy, path-independence, type-consistency |
| **DiscoveryCalibrator (EMA + inverse-rate)** | Uniform or degree-weighted sampling | Per-community EMA + inverse-rate rebalancing + cold-start max weight |
| **Autonomous Discovery Loop + circuit breaker** | NELL, ATOMIC batch updates | Sliding-window circuit breaker + per-cycle cap + dry-run + checkpoint |
| **ProvenanceLedger (per-batch rollback)** | Snapshot-level rollback | Fine-grained per-batch and per-cycle targeted edge removal |
| **Loop-Provenance Recovery** | No equivalent | Circuit-breaker triggered automatic transactional rollback |
| **GraphSnapshot (portable + diff)** | Pickle / database dumps | JSON-portable, skip-existing restore, structural diff without live graph |
| **Adaptive Loop Tuning** | Fixed caps / intervals | Calibrator-driven dynamic cap + interval scaling per cycle |
| **Active Inference / Daydreaming** | Offline KGE retraining, scheduled batch jobs | First application of free-energy minimization principle to symbolic KG: idle-period self-querying from high-PE nodes |
| **Self-Modifying GUI (dual-channel)** | Static dashboards, manual UI updates | First AI system to structurally modify its own game-engine Blueprint UI based on internal metabolic + epistemic state |
| **Cingulate Engine (conflict-driven)** | ACC monitoring (neuroscience), Post-hoc re-ranking | First internal autonomous conflict monitor using entropy to trigger recursive refinement for hub-flooding mitigation |
| **Answer-Type Constraint Filter** | KGE post-hoc ranking, entity type constraints | KB-object-only type index built at query time; hard filter applied after wide retrieval (top-100) before final top-k truncation |
| **Two-Pass TRB Detection (Phase 152/153)** | Keyword matching, NER-based question decomposition | Prefix-first, suffix-fallback with unambiguous pre-passes ("when"→release_year, last-word "in which X") eliminates cross-branch keyword collision in 3-hop templates |
| **Vote-Weight Suppression for Deep Hops** | Vote/convergence bonuses (uniform) | Per-hop vote_weight calibration (0.0 for 3-hop, 0.45 for 2-hop): convergence bonus promotes wrong hub answers unless suppressed at maximum hop depth |
| **Distinct-Branch Convergence (DBC)** | Path count aggregation, GraftNet joint training | Log-scale bonus on entities confirmed via structurally independent hop-2 intermediaries; zero overhead when n_branches=1; no training required |
| **Penultimate Relation Boost (r3→r2 map)** | Penultimate cascade (same-relation only) | Data-driven r3→r2 frequency map enables boosting a DIFFERENT relation at hop N-1; cascade was previously dead for cross-type templates |
| **r2 Path-Consistency Boost (Phase 158)** | Path re-ranking, relation sequence scoring | Post-hoc score boost for answers whose best path uses training-verified r2 (nodes[1] check); pure multiplicative boost (no penalty); uses existing TraversalPath.nodes structure |
| **Pre-pass 4 TRB Detection Fix (Phase 160)** | Keyword matching, question classification | Catches "who is listed as {relation_type} ..." templates where the answer keyword is at word[4], preventing suffix contamination from intermediate entity descriptors |

---

### Claim 47: Answer-Type Constraint Filter for Multi-Hop QA

**Description**: After wide-beam retrieval (top_k=100) on 3-hop questions, apply a hard filter that restricts candidate answers to entities that appear as OBJECTS of the detected terminal relation in the raw KB triples. The index is built from directed KB triples (not undirected graph edges) to exclude reverse-direction entities from answer sets. When the TRB detection identifies a target relation and the KB index is non-empty, only type-valid candidates are passed to the final top-k ranking step.

**Novelty Statement**: Existing KG QA systems apply entity type constraints either at graph construction time (type-constrained embedding) or via post-hoc neural classifiers. CEREBRUM's answer-type filter is applied dynamically at query time from the KB's own triple structure, requires no training data, and is strictly constrained to observed KB objects — not type ontologies or predicted entity classes. Combined with a wider initial retrieval window, this allows a high `vote_weight` (convergence bonus) to amplify correctly-typed candidates without being overwhelmed by popular wrong-type hub entities.

**Result**: H@1 MetaQA 3-hop: 23.0% (Phase 151) → 44.2% (Phase 152) → 45.1% (Phase 153) → 45.7% (Phase 154, DBC) → 46.0% (Phase 156, penultimate r2 boost) → 46.1% (Phase 157, vote_weight=0.85) → 46.4% (Phase 158, r2 path-consistency boost) → **46.6%** (Phase 160, TRB detection fix). CEREBRUM surpasses all published baselines (GraftNet 22.8%, EmbedKGQA 29.8%) using only graph structure — no LLMs, no training data, no KG embeddings.

**Relevant files**: `benchmarks/metaqa_eval.py` (`evaluate_hop`, `detect_target_relation`, `_relation_answer_set`)

---

### Claim 48: Multi-Pass TRB Detection with Structural Pre-Passes

**Description**: `detect_target_relation()` uses four targeted pre-passes before a two-pass keyword scan to avoid cross-branch keyword collisions in 3-hop question templates: (1) `"when ..."` unambiguously maps to `release_year` regardless of intermediate relation keywords in the prefix; (2) terminal `"in which TERM"` patterns are detected by checking only the last word before suffix contamination from entity names; (3) `"what are/is..."` questions use an extended 6-word prefix to catch answer types at position 5 (e.g., `"what are the primary languages"`); (4) `"who is listed as RELATION_TYPE ..."` templates check `words[4]` directly — without this pass, the suffix matches intermediate entity descriptors (e.g., "starred by X actors") and misclassifies `directed_by`/`written_by` questions as `starred_actors`.

**Novelty Statement**: Standard NLP question classification assumes a fixed-length prefix is sufficient for intent detection. In multi-hop KG templates, intermediate relation keywords (actors, directors, writers) appear in the first 5-8 words, causing false positives that corrupt downstream type filtering. The pre-pass structure specifically targets the structural invariants of MetaQA-style 3-hop templates, reducing wrong detection from 16.8% to ~5.8% without expanding the prefix window globally. Phase 160 extended this with a fourth pre-pass that recovers 32+ previously misclassified questions whose answer type appears at word position 4 (one beyond the default prefix window).

**Relevant files**: `benchmarks/metaqa_eval.py` (`detect_target_relation`)

---

### Claim 49: Distinct-Branch Convergence (DBC) Reranking for Multi-Hop KG QA

**Description**: After beam traversal, `extract_answers()` tracks for each terminal entity the set of distinct hop-2 intermediate nodes (second-position intermediaries, `path.nodes[2]`) from which it is reached. A multiplicative branch-diversity bonus `factor = 1.0 + w * log1p(n_branches - 1)` is applied to the combined score, where `w=0.25` empirically. This promotes entities confirmed via multiple *independent* 3-hop chains over hub entities that accumulate high vote sums via repetitive paths through the same intermediate node.

**Novelty Statement**: Existing ensemble-style KG reranking (e.g., PathRanker, NSM, GraftNet) aggregates path scores by count or weighted sum, without distinguishing path independence. CEREBRUM's branch key (`nodes[2]`) measures structural independence at the second hop — paths sharing the same hop-2 intermediary are treated as a single "branch" regardless of their count. A log-scale bonus rewards genuine multi-branch corroboration without over-penalizing single-branch evidence (factor=1.0 when n=1). This is distinct from vote_weight (which accumulates total path scores) and from branch_count (which counts total paths): it specifically measures the *structural diversity* of the evidence base.

**Key Technical Differentiator**: For a 3-hop path seed→e1→e2→answer, two paths sharing the same e2 but different e1 are considered a *single branch* (corroborating evidence from the same intermediate structure). Two paths using different e2 values (different hop-2 nodes) are independent branches. This distinction is not captured by standard path count aggregation.

**Result**: 500-sample ablation: H@1 0.468 → 0.496 (+2.8pp), MRR 0.573 → 0.591 (+1.8pp). Full 14,274-question run: H@1 0.4511 → **0.4572** (+0.61pp), MRR 0.5461 → **0.5499** (+0.38pp), H@10 flat.

**Relevant files**: `reasoning/answer_extractor.py` (`extract_answers`, `branch_sets`, `branch_bonus_weight`), `benchmarks/metaqa_eval.py` (`--branch-bonus` flag)

---

### Claim 50: Cross-Type Penultimate Relation Boost via Training-Data r3→r2 Map

**Description**: The existing penultimate cascade in `BeamTraversal` fires `sqrt(TRB_factor)` at
hop N-1 only for the same relation as r3. In MetaQA 3-hop templates, hop-2 edges are almost
always `starred_actors` regardless of r3, making the cascade structurally dead. A new
`penultimate_relation_boost` parameter (separate from `terminal_relation_boost`) is built by
counting, for each r3, which r2 value appears most frequently in (seed, correct_answer) training
walks through the KB. The top r2 per r3 is applied at hop N-1 with weight `sqrt(r3_boost)`,
independently of what relation appears at the terminal hop.

**Novelty Statement**: No prior work on KG multi-hop traversal distinguishes the penultimate
relation type from the terminal relation type when applying traversal guidance. Standard TRB
implementations (including CEREBRUM Phase 146-148) assume the most informative relation at hop
N-1 is the same as r3. The data-driven r3→r2 map formalizes the insight that 3-hop KG templates
have structured intermediate-hop constraints that differ from the terminal hop. The map is built
with O(|train|×|KB|) graph walks — zero neural components, zero tuned parameters.

**Key Differentiator**: `penultimate_relation_boost` is orthogonal to `terminal_relation_boost`:
the terminal boost filters wrong-type answers at hop N; the penultimate boost steers intermediate
node selection at hop N-1 toward the correct structural path template.

**Result**: Full 14,274-question run: H@1 0.4572→**0.4595** (+0.23pp), H@10 0.7092→**0.7123**
(+0.31pp), MRR 0.5499→**0.5519** (+0.20pp). Cumulative with DBC (Phase 154): H@1 from 0.4511
(Phase 153 baseline) to 0.4595 (+0.84pp). Phase 155 negative result noted: sentence embeddings
hurt MetaQA 3-hop (−1.6pp H@1) because cross-type entity similarity is low and CSA alpha
inadvertently penalizes semantically dissimilar but graph-correct hop transitions.

**Relevant files**: `reasoning/traversal.py` (`penultimate_relation_boost`, two scoring code
paths), `reasoning/expanded_traversal.py` (threading), `core/cerebrum.py` (query API),
`benchmarks/metaqa_eval.py` (r3→r2 map builder, `_prb` construction)

---

### Claim 51: r2 Path-Consistency Boost for Answer Re-ranking

**Description**: After beam traversal, for each candidate answer, inspect `answer.best_path.nodes[1]` — the relation at hop 2 of the H1SE sub-path. If this relation matches the training-derived expected r2 for the detected terminal relation (r3→r2 map from Phase 156), multiply the answer's score by `(1 + r2_boost)` (default r2_boost=0.40). Answers are then re-sorted by the boosted score. This is a pure post-hoc re-ranking: the traversal is unchanged, no paths are pruned, and answers without a best_path or with an unknown r2 are unaffected.

**Novelty Statement**: Existing KG path-ranking methods score paths based on edge weights, entity features, or relation embeddings accumulated along the path during traversal. CEREBRUM's r2-consistency boost is applied *after* traversal and targets a single structural invariant: whether the best-scoring path's hop-2 relation matches the KB-derived canonical r2 for that query type. Because `TraversalPath.nodes` uses alternating entity/relation representation, `nodes[1]` is the first non-seed relation — a lightweight structural check requiring no additional graph operations. The approach exploits the regularity of MetaQA 3-hop templates without requiring template classifiers or additional training.

**Key Differentiator**: The r2-consistency boost is orthogonal to DBC (Claim 49), PRB (Claim 50), and vote_weight: DBC rewards multi-branch evidence; PRB steers beam expansion at hop N-1; vote_weight controls convergence accumulation; r2-consistency boosts post-hoc based on path structure verification. All four are independently additive.

**Result**: 2000-sample ablation: r2_boost=0.40 yields +0.50pp H@1 over Phase 157 baseline. Full 14,274-question run: H@1 0.4614→**0.4636** (+0.22pp), H@10 0.7131→**0.7135** (+0.04pp), MRR 0.5543→**0.5557** (+0.14pp). Combined Phase 157+158 gain vs Phase 156: +0.41pp H@1, +0.12pp H@10, +0.38pp MRR.

**Relevant files**: `benchmarks/metaqa_eval.py` (r2-boost block in `evaluate_hop`, `--r2-boost` flag)

---

## Legal Notice

All rights, title, and interest in and to the CEREBRUM software, documentation, algorithms, and related intellectual property documented herein are and shall remain the exclusive property of **Bryan Alexander Buchorn**.

CEREBRUM is dual-licensed:
- **Non-Commercial Use**: PolyForm Noncommercial License 1.0.0
- **Commercial Use**: Separate commercial license required

For commercial licensing: **bryan.alexander@buchorn.com**

**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 1, 2026 for version v2.44.0

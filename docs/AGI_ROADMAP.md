# CEREBRUM — A Path to General Knowledge-Grounded Reasoning

**Internal Research Vision Document**
Version 1.0 — June 2026
Bryan Buchorn / CEREBRUM Project

---

## 1. Where We Are

CEREBRUM v2.92.0 is a training-free knowledge graph reasoning framework. It takes a natural language question, decomposes it into graph traversal operations, executes multi-hop beam search over a knowledge graph, and returns ranked answer candidates — each one a verifiable, citable path through structured knowledge. There are no gradient steps, no training data, and no learned weights of any kind. Every answer is traceable to a sequence of explicit (entity, relation, entity) triples.

The system's reasoning core — CORTEX — implements Community-Structured Attention (CSA), a 10-parameter formula that uses graph topology as discrete attention heads. Communities detected via the Louvain algorithm function analogously to cortical columns: nodes within the same community reinforce one another's activation scores, while cross-community edges carry a configurable penalty that models the attentional cost of integrating distant concepts. The Schema-Derived Relation Boost (SDRB) further refines traversal by promoting high-fan-out relations that are structurally central to the knowledge base, derived analytically from the graph itself rather than from held-out training signal.

Supporting CORTEX are several additional mechanisms that have measurably improved benchmark performance: PathSchemaIndex predicts likely 2-hop relation schemas from the question embedding before traversal begins, pruning the beam to structurally plausible paths. Guaranteed 1-hop Pass (G1P) exhaustively injects 1-hop neighbors to prevent beam collapse on sparse subgraphs. ParameterInitializer derives all hyperparameters analytically from graph statistics — degree distribution, community structure, hub centrality — so the system adapts to any knowledge graph without manual tuning.

On structured benchmark tasks, the system is competitive. MetaQA 3-hop reaches H@1 of 60.6% across 14,274 questions with no training. Hetionet 1-hop reaches H@1 of 95.3%. WebQSP on Freebase is harder — H@1 of 11.92% — which is an honest signal about where the system struggles: large, heterogeneous, real-world knowledge graphs with ambiguous entity linking and sparse coverage.

What CEREBRUM does not do is equally important to state clearly. It does not perceive the world. It does not learn continuously from unstructured experience. It does not plan sequences of actions toward goals. It does not build its own knowledge graph from raw input. It does not engage in causal interventional reasoning. It does not modify its own reasoning strategy in response to failure. These are not minor gaps — they are the gaps between a sophisticated retrieval system and a general intelligence.

---

## 2. What AGI Actually Requires

The term AGI is used loosely in the field, often to mean "better than current LLMs on some benchmark." A more disciplined framing is needed before claiming any roadmap leads there.

Legg and Hutter's formal definition (2007) characterizes general intelligence as the ability to achieve goals across a wide range of environments — formally, the expectation of reward summed over all computable environments weighted by their Kolmogorov complexity. This definition is useful not because it is implementable, but because it makes the generality requirement explicit: a system that excels in one environment (MetaQA) but fails systematically in another (Freebase QA) is not general.

Chollet's ARC-AGI framing (2019) tightens this further by emphasizing sample efficiency and novel abstraction. A system that achieves good performance by exhaustive training on domain-specific data is not exhibiting general intelligence — it is exhibiting memorization. The benchmark is deliberately constructed to test whether a system can apply abstract reasoning to patterns it has never seen, using only a small number of demonstrations. CEREBRUM's training-free design is philosophically aligned with this criterion, but the current system handles only structured graph queries, not the visual-spatial abstract reasoning ARC targets.

Taking both framings seriously, a credible AGI roadmap must address the following capabilities: grounded perception (the system must be able to build internal representations from raw sensory input), open-ended learning (the system must extend its knowledge and revise its beliefs from experience), causal modeling (the system must distinguish correlation from causation and reason about interventions), goal-directed planning (the system must generate and evaluate action sequences toward objectives), continual memory consolidation (knowledge must be integrated without catastrophic interference), linguistic competence (the system must handle the full pragmatic range of natural language), and self-modification (the system must revise its own reasoning strategies in response to observed failures).

CEREBRUM addresses none of these fully. Some it addresses partially. The question is whether its architecture provides a credible foundation for addressing them.

---

## 3. The CEREBRUM Thesis

The thesis underlying CEREBRUM's design is that AGI requires an explicit, verifiable, extensible world model — and that current approaches fail not because they lack parameters but because they lack structure.

Large language models have become extraordinarily capable at pattern-matching over text. But the world model implicit in transformer weights is unstructured: facts are distributed across billions of parameters with no explicit representation of the entities, relations, or logical dependencies between them. When an LLM hallucinates, it is not making a random error — it is generating text that is locally plausible given its training distribution, but that fails to correspond to facts in the world, because the model has no reliable mechanism for checking correspondence. It cannot look up the fact. It cannot verify the path. It cannot distinguish "I learned this" from "I inferred this" from "I generated this because it sounded right."

CEREBRUM externalizes the world model as a graph. Every claim is a path. Every path is checkable. The knowledge base can be corrected, extended, and versioned. This is not merely an engineering preference — it is a claim about the architecture of cognition. Biological brains do not store knowledge as diffuse statistical patterns across synapses alone; they organize knowledge into structured schemas, semantic categories, causal models, and episodic traces that can be retrieved, combined, and reasoned over explicitly. CEREBRUM's architecture reflects this hypothesis.

The cognitive architecture names are not cosmetic. THALAMUS, as the ingestion layer, mirrors the thalamus's role as a sensory gating and relay station — it decides what enters the reasoning system and in what form. CORTEX's CSA mechanism models the columnar organization of cortical processing, where community structure in the graph plays the role of cortical area specialization. The REM Engine's graph self-reorganization during low-activity periods mirrors the role of sleep in memory consolidation, replaying and restructuring stored experiences. Bridge Twin Engine's STDP-based edge formation mirrors Hebbian synaptic potentiation: connections that co-activate repeatedly are strengthened, implementing a form of experience-dependent structural plasticity. ChemicalModulator implements dopaminergic reward signaling, noradrenergic arousal, and cholinergic attention modulation as explicit parameters that shift the system's exploration-exploitation balance and attention allocation. PredictiveCodingEngine implements active inference with explicit prediction error signals, consistent with Friston's free energy principle as a unifying account of brain function.

The claim is not that these modules are biologically accurate simulations. The claim is that they map to known functional requirements of general intelligence and that implementing them in a graph-structured system creates a more principled foundation for scaling toward AGI than scaling parameters alone.

---

## 4. The Seven Gaps and How to Close Them

### 4a. The Perception Gap

CEREBRUM reasons over pre-structured knowledge graphs. The graph must be constructed before reasoning can begin. This means someone or something has already done the work of perceiving the world, extracting entities and relations, and formalizing them into triples. The system's SignalEncoder module exists but is early-stage. It does not ground visual or physical input into graph structure.

AGI requires the ability to build a world model from raw perception — images, sensor streams, continuous text. The path forward is a perceptual grounding module: a vision-language model (VLM) that takes raw input and extracts (entity, relation, entity) propositions, which are then injected into the graph via THALAMUS. The critical design constraint is that this module must produce typed, schema-aware triples — not unstructured text — so that CORTEX can reason over them immediately upon ingestion. This is not a solved problem. Entity resolution across perceptual modalities, relation type assignment, and confidence-weighted edge insertion all require careful engineering.

### 4b. The Open-Ended Learning Gap

The AutonomousDiscoveryLoop, ResearchAgent, and ProvenanceLedger provide a foundation for continuous knowledge extension. ResearchAgent can identify missing links in the graph and issue queries to fill them. But these mechanisms operate within a fixed schema: they add new nodes and edges of types already defined. They cannot infer that a new relation type is needed.

AGI requires schema-free knowledge extension: the ability to observe that two entities co-occur in a context not captured by any existing relation type and to hypothesize a new relation. The path is meta-schema inference — statistical co-occurrence analysis over incoming data that identifies candidate new relation types, proposes them to the schema registry, and begins populating them once sufficient evidence accumulates. This is analogous to the biological process of concept formation: the brain does not merely fill in an existing ontology; it creates new categories when existing ones fail to account for observed patterns.

### 4c. The Causal Modeling Gap

CEREBRUM has causal edge weighting through CausalEngine and STDP-based plasticity. These represent associative causal priors — edges that have been traversed frequently in contexts where the answer was correct accumulate higher weight. This is not causal reasoning in the interventional sense.

Pearl's do-calculus distinguishes observational queries ("what is the probability of Y given X?") from interventional queries ("what is the probability of Y if I force X to take value x?") from counterfactual queries ("what would Y have been if X had been x, given that we observed Y'?"). CEREBRUM can answer observational queries over the graph. It cannot answer interventional or counterfactual queries without a causal graph model that distinguishes the direction of causal influence and represents the effect of graph surgery — removing incoming edges to a node to simulate intervention.

The path is a do-calculus layer over the knowledge graph: explicit causal edge direction, identification of backdoor paths, and a query mode that executes graph surgery for interventional reasoning. This requires annotating the graph with causal direction metadata — non-trivial for knowledge graphs built from text, but feasible for domain-specific graphs (biomedical, physical systems) where causal direction is known.

### 4d. The Planning Gap

BeamTraversal retrieves answers to questions. It does not generate sequences of actions. The distinction matters: planning requires not just finding a path between a start state and a goal state in a knowledge graph, but modeling the consequences of actions taken in the world and selecting a sequence that achieves an objective.

The extension is to treat actions as a special class of relations in the graph, where (state, action, next\_state) triples encode a world model for planning. BeamTraversal then becomes a forward search over this action graph, and PathSchemaIndex can be extended to predict likely action chain schemas from a goal description. This is the standard formalization of goal-directed planning as graph search over a state-action space. The novel contribution would be CEREBRUM's ability to verify that each state transition is consistent with the broader knowledge graph — constraining the planner to physically and logically coherent action sequences.

### 4e. The Continual Memory Gap

REM Engine, ProvenanceLedger, and the Engram cache implement a form of memory management. ProvenanceLedger maintains a versioned record of all edge additions and deletions with timestamps and source attribution. REM Engine reorganizes community structure during low-activity periods.

The gap is the graph-theoretic analogue of catastrophic forgetting: adding new knowledge can corrupt existing community structure, invalidating CSA attention weights that were derived from the previous graph topology. Adding a large batch of new nodes can shift community boundaries, causing previously correct reasoning paths to score lower. The solution is selective graph plasticity with community stability metrics: before materializing new edges, compute the expected delta to community structure, and apply a stability threshold that rejects changes that would catastrophically reorganize existing communities. ProvenanceLedger already provides rollback support for exactly this case — the architectural foundation exists; the stability metric and gating mechanism need to be implemented.

### 4f. The Linguistic Competence Gap

RelationNameIndex and QuestionDecomposer handle structured queries and translate natural language question patterns into graph traversal operations. This works well for question types seen in benchmark datasets — "what did X do to Y?", "which X is related to Y through Z?" — but it does not generalize to open-ended natural language with full pragmatic range.

The path here is hybrid: use an LLM as a front-end that grounds natural language to graph entity and relation references, with CEREBRUM as the reasoning backend that verifies and extends the LLM's claims. The LLM is not asked to produce an answer; it is asked to produce a structured query — entity mentions, candidate relation types, expected path schema — that CEREBRUM then executes and verifies against the ground-truth graph. Where the LLM's grounding fails (entity not found, relation type not in schema), CEREBRUM returns a structured error that the LLM can use to revise its query. This bidirectional interface is the core of the hybrid architecture thesis developed in Section 5.

### 4g. The Self-Modification Gap

CEREBRUM's parameters are currently tuned by an external Optuna-based tuner: CMA-ES and TPE search over the CSA formula weights, SDRB coefficients, beam width, and community resolution. The system does not modify its own reasoning strategy.

ParameterInitializer already derives default parameters analytically from graph statistics — this is a meaningful step toward self-calibration. The extension is a MetaParameterLearner that continuously re-derives parameters as the graph evolves: as new nodes and edges are added, as community structure shifts, and as performance metrics on held-out validation queries change, the system updates its own parameter estimates without external intervention. This is not gradient descent — it is principled analytical re-derivation triggered by structural change events. The foundation is already in place; the continuous update loop and the performance-feedback signal need to be added.

---

## 5. The Hybrid Architecture Thesis

The central architectural claim is that AGI is not achievable through pure neural or pure symbolic approaches, and that the distinction is not merely pragmatic but principled.

Pure neural systems — transformers at scale — achieve remarkable generality in pattern matching because they can approximate any smooth function over their training distribution. But they cannot verify claims, cannot trace reasoning to explicit evidence, cannot detect when they have moved outside their training distribution, and cannot reliably distinguish valid inference from plausible confabulation. These are not engineering failures to be solved by scaling; they are structural properties of implicit distributed representations.

Pure symbolic systems — classical knowledge bases, logic programming, SPARQL — are verifiable and explicit but brittle. They cannot generalize beyond explicitly represented facts, cannot handle noise or ambiguity in input, and require hand-engineered schemas that cannot adapt to new domains without manual ontology engineering.

CEREBRUM's architecture is hybrid at the core in a specific and principled way. CSA attention is neural-inspired: it computes continuous-valued attention scores over graph structure, tolerating the absence of any one path and integrating evidence from multiple paths with weighted summation. The graph structure itself is symbolic: every node is an explicit entity, every edge is an explicit relation, and every answer is a citable path. PathSchemaIndex bridges both: it uses a learned embedding space to predict likely path schemas before traversal, combining the generalization capacity of learned representations with the verifiability of structured path execution.

The path to AGI goes through this hybrid in a specific direction: LLMs handle perception and language (the things they are genuinely good at — pattern recognition over high-dimensional unstructured input), while CEREBRUM handles structured reasoning and verification (the things symbolic systems are genuinely good at — logical consistency, explicit evidence chains, knowledge correction). The bidirectional interface between these two components — where LLMs ground language to graph structure and CEREBRUM verifies and corrects LLM-generated claims — is the architectural bet at the center of this roadmap.

---

## 6. Concrete Milestones Toward AGI

**Phase 300-series: Perceptual Grounding.** Integrate a vision-language model front-end that populates graph entities and relations from image and text streams via THALAMUS. The deliverable is a system that can watch a video, extract entity-relation-entity triples, and immediately reason over them with CORTEX. Success metric: knowledge graph auto-population from unstructured text with measurable precision/recall against a held-out annotation.

**Phase 400-series: Meta-Schema Inference.** Implement statistical co-occurrence analysis over incoming data to identify candidate new relation types. Build a schema proposal pipeline that accumulates evidence for new relation types, gates them through a statistical significance threshold, and registers them in the schema when confirmed. Success metric: discovery of valid novel relation types in a domain-specific dataset that were absent from the initial schema.

**Phase 500-series: Do-Calculus Interventional Reasoning.** Add causal edge direction metadata to the graph. Implement interventional query mode (do-calculus graph surgery). Implement counterfactual query mode (evidence + intervention). Success metric: correct answers on a causal inference benchmark (e.g., CausalQA or a custom biomedical intervention dataset where ground-truth causal directions are known).

**Phase 600-series: Action Graph Planning.** Extend BeamTraversal to search over (state, action, next\_state) triples toward a goal state. Implement PathSchemaIndex extension for action chain schema prediction. Build a goal decomposition module that breaks high-level objectives into subgoal sequences. Success metric: correct multi-step plan generation on a simple gridworld or logistics planning task, with all action steps verifiable against the knowledge graph.

**Phase 700-series: LLM-CEREBRUM Bidirectional Interface.** Build the full bidirectional API: LLM grounds NL to graph entity/relation references; CEREBRUM executes and returns a verification result; LLM revises grounding on failure; CEREBRUM confirms on success. Implement claim verification mode where CEREBRUM assesses whether an LLM-generated statement is consistent with the knowledge graph. Success metric: measurable reduction in LLM hallucination rate on knowledge-intensive QA tasks when CEREBRUM verification is in the loop.

**Phase 800-series: Self-Modifying ParameterInitializer.** Extend ParameterInitializer to a MetaParameterLearner that continuously re-derives CSA and SDRB parameters as the graph evolves, triggered by structural change events (new batch ingestion, community reorganization, performance delta on validation queries). Success metric: performance on a streaming knowledge graph benchmark where the graph changes over time, without any external tuning intervention.

**Phase 900-series: Multi-Agent CEREBRUM Networks.** Federate multiple CEREBRUM instances over complementary knowledge graphs. Implement cross-instance query routing, knowledge sharing with provenance, and conflict resolution when instances disagree. Success metric: a federated system that achieves higher performance on a heterogeneous multi-domain QA task than any single instance can achieve alone.

**Phase 1000: Knowledge-Grounded General Reasoner.** A system that builds, extends, verifies, and reasons over its own world model in real-time from raw perceptual streams — perceiving new information, grounding it to graph structure, integrating it with existing knowledge, reasoning over the integrated model, planning actions, and revising its own reasoning strategies as performance feedback accumulates. This is the operational definition of a knowledge-grounded general reasoner. It is not AGI as conventionally imagined, but it is a system that addresses all seven gaps described in Section 4 with verifiable, traceable mechanisms.

---

## 7. Why This Architecture, Not Transformers Alone

This requires a direct and honest comparative assessment, not a dismissal of LLMs.

Transformers trained at scale are the most capable general-purpose learning systems ever built. They exhibit emergent capabilities that their architects did not predict, generalize across tasks with minimal fine-tuning, and handle the ambiguity and pragmatic richness of natural language better than any prior system. These are real achievements and the hybrid architecture in Section 5 depends on them.

The specific failure mode of transformers on knowledge-intensive tasks is well-documented: they hallucinate facts, fail to distinguish what they know from what they have generated, and cannot update their knowledge without retraining. These failures are not primarily a function of scale — they have persisted and in some cases worsened as models have grown. They are a structural consequence of storing world knowledge as implicit statistical patterns in weights rather than as explicit verifiable structures.

CEREBRUM's approach does not fix this by replacing transformers. It fixes it by giving transformers a structured external memory that they can read, write, and verify against. Every claim a CEREBRUM-integrated LLM makes can be checked. Every answer is a path. Every edge has a source and a timestamp. This is the distinguishing property of the architecture: not that it is more capable than a transformer in isolation, but that its outputs are verifiable in a way that transformer outputs alone cannot be.

---

## 8. The Hard Problems That Remain

Intellectual honesty requires naming the problems this roadmap does not solve.

The binding problem is the hardest: how does a symbolic graph node correspond to a perceptual entity in the world? When CEREBRUM's graph contains the node "aspirin" and a perception system observes a white tablet, what warrants the inference that the tablet is an instance of the node? This is the grounding problem in philosophy of language, and it has no generally accepted computational solution. The perceptual grounding modules in the Phase 300-series roadmap address this partially through embedding-based entity resolution, but the deep question of what makes a symbol mean something — rather than merely co-occur with perceptual patterns — remains open.

The frame problem is equally serious: knowledge graphs go stale. The world changes faster than any knowledge graph can be updated. A fact that was true when an edge was added may be false now. ProvenanceLedger's timestamp metadata and the REM Engine's reorganization provide partial solutions — stale edges can be flagged and downweighted — but determining when a fact has changed requires either continuous monitoring of all relevant external sources or an explicit model of which facts are time-indexed and which are stable. This is an unsolved problem in knowledge representation.

The compositionality problem is specific to the graph-structured approach: human concepts are compositionally structured in ways that knowledge graphs capture poorly. The concept "the cause of the war that ended the empire that was founded by the general who defeated Caesar" requires compositional recursive reference that flat graph traversal handles awkwardly. CEREBRUM's PathSchemaIndex and QuestionDecomposer handle some compositional queries, but deeply nested compositional reference remains a challenge.

These problems are not unique to CEREBRUM. They are shared by every approach to AGI. The claim of this document is not that CEREBRUM solves them — it does not — but that its architecture provides a more principled foundation for eventually addressing them than architectures that lack explicit, verifiable world models.

---

## Conclusion

CEREBRUM is, at present, a high-performance training-free knowledge graph reasoning system. It is not a general intelligence. The distance between those two descriptions is the substance of this document.

The roadmap above is not a guarantee of arrival at AGI. It is a sequence of concrete, measurable engineering milestones grounded in the genuine gaps identified in Section 4. Each phase closes one gap partially and creates the infrastructure needed for the next. The architectural thesis — that structured, verifiable, graph-based world models are a necessary component of any credible path to general intelligence — is a substantive claim that may be wrong, and should be revised if evidence demands it.

What the current benchmarks establish is that training-free, graph-structured reasoning is competitive with learned approaches on structured tasks, scales to real-world knowledge graphs, and produces fully verifiable outputs. That is a foundation worth building on.

---

*Document status: Internal research vision, v1.0. Subject to revision as architecture evolves.*
*CEREBRUM v2.92.0 | Phase 237+ | June 2026*

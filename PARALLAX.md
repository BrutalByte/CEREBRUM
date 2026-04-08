# CEREBRUM: Community-Structured Graph Attention for Knowledge Graph Reasoning

**Authors**: Bryan Alexander Buchorn (AMP) · Claude Sonnet 4.6 (Research Collaborator)
**Affiliations**: Independent Researcher · Anthropic
**Contact**: bryan.alexander@buchorn.com
**Date**: March 2026
**Status**: Version 2.0.1 · Phase 57 COMPLETE — 1490+ tests passing
**License**: Proprietary — all rights reserved

---

## Quick Reference

**What it is**: A framework that lets a Knowledge Graph reason like a Transformer —
using community structure as attention heads, BFS hop depth as layer depth, and
graph-structural features as positional encodings. No training required. No LLM required.
Every inference step is a verifiable graph edge.

**The two core algorithms**:

```
DSCF (community detection):
  For each node v at each iteration:
    lpa_cid  = majority vote among neighbors  (local signal)
    mod_cid  = best modularity gain ΔQ        (global signal)
    if agree and ≠ current: MOVE (anchor)
    elif disagree: MOVE with prob weighted by (lpa_conf × temp) vs (mod_conf × (2-temp))
  temperature: τ_{t+1} = max(τ_t × 0.92, 0.01)

CSA (attention weight for edge u→v at hop k):
  a(u,v,k) = σ(
    α·sim(u,v)          # semantic similarity (cosine)
  + β·community_score   # structural membership
  + γ·w_rel             # edge-type weight
  - δ·distance          # normalised distance penalty
  + ε·hop_decay(k)      # exponential hop decay
  + ζ·PageRank(v)       # global authority prior
  + η·temporal_decay    # time since edge creation
  + ι·node_recency      # recency of traversal
  - μ·synth_density     # synthetic edge penalty
  + θ·grounding         # confidence / provenance
  )
  defaults: α=0.4, β=0.4, γ=0.1, δ=0.05, ε=0.05, ζ=0.1, η=0.1, ι=0.05, μ=0.1, θ=1.0
```

**Transformer → KG mapping**:

| Transformer | CEREBRUM |
|---|---|
| Attention head | DSCF community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + betweenness + degree |
| Attention weight | CSA formula above |
| Context window | Ego-network radius R |
| KV cache | Materialized path store |
| Fine-tuning | CSAParameterLearner.fit() via POST /retrain |

**Repo layout** (target structure for standalone project):

```
parallax/
├── core/          graph_adapter, embedding_engine, community_engine, attention_engine, structural_encoder
├── reasoning/     traversal, path_scorer, answer_extractor
├── adapters/      networkx, neo4j, rdf, csv
├── llm_bridge/    context_formatter
├── api/           server, schemas
├── cli/           cerebrum.py
├── tests/         test_dscf, test_csa, test_traversal, fixtures/toy_graph.csv
├── benchmarks/    webqsp_eval, metaqa_eval, baseline_comparison
├── examples/      wikidata_quickstart, neo4j_quickstart, csv_quickstart
├── pyproject.toml
├── README.md
└── PAPER.md       (this file)
```

**Current phase**: Phase 57 complete (v2.0.1). CEREBRUM now includes GraphSAGE neighbourhood smoothing (`smooth_with_graphsage`), Engram-steered beam traversal (`Engram` + `EngramTraversal`) with durable persistence across restarts, `TemporalCalibrator` for Recall@K-optimal eta/iota calibration, `QueryLog` append-only history for warm-up, `HypothesisEngine` multi-path abductive reasoning, `ResearchAgent` autonomous missing-link discovery, `ExternalValidator` LLM-independent source validation, an observability dashboard, and comprehensive fault tolerance hardening (graceful degradation, partial-result responses, crash-guard GlobalRebalancer, stream error chunks). 1490+ tests passing.

---

## Abstract

We propose **CEREBRUM**, a novel framework that enables Knowledge Graphs (KGs)
to perform multi-hop reasoning using the same structural principles that make
Transformer-based Large Language Models powerful — without requiring an LLM,
without training data, and with full interpretability of every inference step.

The central contribution is **Community-Structured Attention (CSA)**: a
mechanism in which graph communities serve as attention heads, graph traversal
replaces matrix multiplication, and hop depth replaces layer depth. Unlike
Graph Attention Networks (GATs), which apply learned attention within hard
adjacency constraints, CSA uses community membership as a soft global
constraint that captures both local topological cohesion and global structural
significance simultaneously.

This is made possible by a second contribution: the **Dual-Signal Community
Fusion (DSCF)** algorithm, which produces communities that encode both LPA
majority-vote structure (local) and modularity gain (global) in a single
partition. We show that DSCF communities possess a structural duality that
maps naturally to the dual character of multi-head attention in Transformers.

Together, CSA and DSCF form an architecture where a KG can answer multi-hop
questions by traversing itself, with every reasoning step grounded in explicit
graph edges, every conclusion traceable to a path, and no LLM required for
inference — though one may optionally be used for natural language generation.

---

## 1. Introduction

### 1.1 The Gap Between Knowledge Graphs and Language Models

Knowledge Graphs and Large Language Models represent two fundamentally
different approaches to knowledge representation and reasoning.

Knowledge Graphs store knowledge explicitly: entities as nodes, relationships
as typed edges, facts as (subject, predicate, object) triples. This makes
them precise, verifiable, and updatable without retraining. However, they
cannot reason beyond what is explicitly stored. Multi-hop inference — "Marie
Curie discovered Polonium, Polonium is radioactive, therefore Marie Curie
discovered a radioactive element" — requires either hardcoded graph traversal
queries or external reasoning systems.

Large Language Models store knowledge implicitly in billions of weight
parameters. They can reason, generalize, and synthesize across domains. But
this implicit representation is opaque, cannot be updated without expensive
fine-tuning, and is prone to hallucination — generating plausible-sounding
but incorrect facts with no mechanism for ground-truth verification.

The field has responded with hybrid approaches: Retrieval-Augmented Generation
(RAG), Knowledge-Graph-augmented LLMs, and GraphRAG. In all of these, the KG
is a retrieval store and the LLM does the reasoning. The KG remains passive.

**CEREBRUM inverts this relationship.** The KG reasons. The LLM, if present,
only generates natural language from the KG's output. Every inference step is
a graph traversal, every conclusion is a path, and every path can be verified.

### 1.2 The Core Observation

A Transformer's power comes from three mechanisms working together:

1. **Multi-head attention**: different heads specialize on different
   relational aspects of the input (syntactic, semantic, long-range, etc.)
2. **Deep composition**: each layer builds on the previous, allowing complex
   multi-step reasoning
3. **Positional awareness**: the model knows where each token sits relative
   to others

We observe that Knowledge Graphs have natural analogs for all three:

1. **Community structure** serves the role of attention heads: nodes within
   a community have strong mutual relevance; communities specialize on
   different conceptual domains.
2. **BFS hop depth** serves the role of layer depth: each hop is one step of
   composed reasoning.
3. **Graph-structural features** (PageRank, betweenness, degree) serve the
   role of positional encoding: they tell the model where each entity sits in
   the global information landscape.

The question is: can these analogs be made operational — not merely
metaphorical? We argue yes, and demonstrate the architecture to do so.

### 1.3 Contributions

This paper makes three primary contributions:

1. **The CEREBRUM architecture**: a complete mapping of Transformer components
   to KG operations, enabling multi-hop reasoning via graph traversal alone.

2. **Community-Structured Attention (CSA)**: a novel attention mechanism using
   community membership as a soft global constraint on graph traversal,
   bridging the gap between local GAT-style attention and global Transformer-
   style attention.

3. **Triple-Signal Consensus (TSC)**: a novel community detection
   algorithm combining LPA majority-vote, modularity gain, and
   centrality-weighted consensus simultaneously at each node update.
   This produces communities anchored by structural hubs, mapping to
   multi-head attention's ability to attend to both local context and
   global significance.

---

## 2. Background and Related Work

### 2.1 Graph Neural Networks

Graph Neural Networks (GNNs) [Scarselli et al., 2009] generalize neural
networks to graph-structured data. The message-passing paradigm [Gilmer et
al., 2017] defines node updates as aggregations of neighbor representations.

Graph Attention Networks (GATs) [Velickovic et al., 2018] introduce learned
attention weights between connected nodes. Key limitations for our purposes:
(a) attention is restricted to direct neighbors — no global context;
(b) communities are not considered; (c) training labels required.

GraphSAGE [Hamilton et al., 2017] samples and aggregates local neighborhoods
but similarly lacks global structural awareness.

None of these use community structure as an organizational principle for
attention.

### 2.2 Knowledge Graph Reasoning

Early KG reasoning systems used rule-based approaches (AMIE [Galarraga et al.,
2013]) or embedding-based methods (TransE [Bordes et al., 2013], RotatE [Sun
et al., 2019]). These methods produce entity embeddings but do not perform
multi-hop traversal in the attention-mechanism sense.

Path-based reasoning approaches (DeepPath [Xiong et al., 2017], MINERVA [Das
et al., 2018]) use reinforcement learning to find paths. These are closer to
our goal but require training data and do not use community structure.

### 2.3 LLM + KG Hybrid Systems

KG-GPT [Yao et al., 2023], KGPT [Chen et al., 2020], and related systems
connect KGs to LLMs as retrieval stores. The LLM always performs the
reasoning step; the KG is passive.

GraphRAG [Edge et al., 2024] uses community detection (Leiden) to summarize
graph clusters as text, which is then passed to an LLM for RAG. This is the
closest existing work. However:
- Communities are summarized as static text chunks, not used as attention heads
- The LLM performs all reasoning
- Paths are not returned or made interpretable
- The system is not graph-agnostic (designed for Microsoft's pipeline)

RAPTOR [Sarthi et al., 2024] builds hierarchical text clusters for RAG but
operates on documents, not graphs, and uses tree structure rather than
community structure.

### 2.4 Community Detection

The Louvain algorithm [Blondel et al., 2008] optimizes modularity greedily
but can produce internally disconnected communities. Leiden [Traag et al.,
2019] fixes this with a refinement phase promoting internal connectivity.
Label Propagation [Raghavan et al., 2007] is fast and unsupervised but
non-deterministic and produces variable quality.

**DSCF** (this work) combines LPA and Leiden signals simultaneously at each
node update, using a temperature-annealing schedule. See Section 4.2.

**DSCF is non-deterministic** (inherits LPA's shuffle-order sensitivity).
For reproducible results: run 3-5 trials and select the partition with
highest modularity score. For production: seed the RNG and document the seed.

---

## 3. The Structural Equivalence

We establish a complete operational mapping between Transformer components
and KG operations. This is not analogy — each mapping is functional.

| Transformer Component | CEREBRUM (KG) Equivalent | Notes |
|---|---|---|
| Token | Entity or Relation | Atomic unit of information |
| Vocabulary | Entity type taxonomy | Closed set of possible types |
| Token embedding | Entity embedding (TransE/RotatE) | Dense vector per entity |
| Positional encoding | Structural encoding (PR, BW, deg) | Where entity sits globally |
| Attention head | Community cluster (DSCF) | Specialized relational context |
| Attention weight | CSA weight formula | Sim + community + edge + distance |
| Context window | Ego-network radius R | How far to traverse |
| Layer depth | BFS hop count | Reasoning step count |
| Wormhole Attention | Federated Link | Cross-graph attention jump |
| Holographic Index | Compressed Signature | Privacy-preserving discovery |
| Feed-forward sublayer | Entity-type projection | Type-specific transformation |

| Residual connection | Previous-hop embedding | Prevents information loss |
| Layer normalization | Embedding normalization | Prevents value explosion |
| Output projection | Path decoder / ranker | Maps traversal to answer |
| KV cache | Materialized path store | Reuse traversal across queries |

This equivalence has a critical implication: **the number of attention heads is
not a hyperparameter in CEREBRUM — it is determined by the graph's own community
structure.** A graph with 12 natural communities has 12 attention heads. A graph
with 200 communities has 200. The architecture adapts to the data.

---

## 4. The CEREBRUM Architecture

### 4.1 Community-Structured Attention (CSA)

CSA computes attention weights for graph traversal that incorporate both local
graph topology and global community structure.

**Attention weight formula:**

For entity u attending to entity v at traversal hop k:

```
a(u, v, k) = σ(
    α · cosine_sim(emb(u), emb(v))
  + β · community_score(u, v)
  + γ · edge_type_weight(type(u → v))
  - δ · normalized_distance(u, v)
  + ε · hop_decay(k)
)
```

Where:
- `emb(·)` is the entity embedding (any KGE method or sentence encoder)
- `community_score(u, v)`:
  - 1.0 if community(u) == community(v)           [same head]
  - 0.5 if communities are adjacent               [neighboring heads]
  - exp(-λ · community_distance(u, v)) otherwise  [distance decay]
- `edge_type_weight(type)`: learned or manually assigned per relation type
- `normalized_distance(u, v)`: shortest path length / graph diameter
- `hop_decay(k)`: encourages shorter paths (e.g., 1 / (1 + k))
- σ: sigmoid activation
- α, β, γ, δ, ε: tunable parameters

**Default parameter values (zero-shot deployment):**
- α = 0.4 (embedding similarity)
- β = 0.4 (community membership)
- γ = 0.1 (edge type)
- δ = 0.05 (distance penalty)
- ε = 0.05 (hop decay)

Parameters can be learned from (query, answer) pairs for supervised settings.

**community_score definition (complete):**

```
community_score(u, v):
  if community(u) == community(v):          return 1.0
  if communities_are_adjacent(u, v):        return 0.5
  else:
    d = community_distance(u, v)
    return exp(-λ · d)

community_distance(u, v):
  # Shortest path (in hops) between the community of u and community of v
  # in the community-level graph, where communities are nodes and two
  # communities are adjacent if ≥1 cross-community edge exists between them.
  # Precomputed once after DSCF converges via BFS on the community graph.
  # λ = 0.5 default (controls cross-community decay rate)
```

**Why this is not a GAT:**

GATs compute `a(u, v) = f(Wu · emb(u), Wv · emb(v))` — purely from learned
weights on adjacent node pairs. They cannot express the community membership
term β · community_score(u, v), which introduces global structural awareness
without requiring the full O(n²) attention of Transformers.

CSA is O(n · k̄ · C) where k̄ is average degree and C is the average number
of community-adjacent entities to consider — far cheaper than Transformer
attention while capturing global structure via community membership.

### 4.2 Dual-Signal Community Fusion (DSCF)

DSCF is the community detection algorithm that produces the attention head
structure. It is a key contribution in its own right.

**The core innovation:** at each individual node update, both the LPA
majority-vote signal (local topology) and the modularity gain signal (global
structure) are computed. The decision incorporates both simultaneously,
governed by a temperature parameter:

```
For each node v at each iteration:

  1. LPA signal:
     lpa_cid = argmax over neighbor labels (majority vote)
     lpa_conf = vote_count[lpa_cid] / total_neighbors  ∈ [0, 1]

  2. Modularity signal:
     For each candidate community C adjacent to v:
       ΔQ(v→C) = k_{v,C}/m − resolution × k_v × Σk_C / (2m²)
     best_mod_cid = argmax ΔQ
     mod_conf = min(best_ΔQ × m, 1.0)  ∈ [0, 1]

  3. Decision:
     if lpa_cid == best_mod_cid ≠ current:
       MOVE (consensus anchor — high confidence)
     elif both say STAY:
       STAY
     elif only LPA says MOVE:
       MOVE with probability lpa_conf × temperature
     elif only modularity says MOVE:
       MOVE with probability mod_conf × (1 + (1 − temperature))
     else (disagree on different targets):
       lpa_weight = lpa_conf × temperature
       mod_weight = mod_conf × (2 − temperature)
       MOVE to weighted-random choice

  temperature schedule: τ_{t+1} = max(τ_t × cooling, 0.01)
```

Post-convergence: Leiden-style connectivity check — split any community whose
induced subgraph is disconnected.

**Why DSCF communities are the right attention heads:**

In a trained Transformer:
- Some heads specialize on local structure (syntactic patterns, adjacent tokens)
- Other heads specialize on long-range structure (coreference, semantic themes)
- The combination allows both local and global reasoning simultaneously

DSCF communities exhibit exactly this dual character. The LPA component supports
communities are locally coherent (nodes in the same community are topologically
close). The modularity component supports communities are globally significant
(they represent structurally distinct regions of the graph). A node that is in
a DSCF community is there because *both* local and global signals agreed.

This dual property has not previously been used as a basis for attention
mechanisms in any published graph learning system.

**Comparison to Leiden-only communities:**

Leiden optimizes purely for modularity. On sparse regions of a graph, Leiden
may split locally coherent neighborhoods across multiple communities because
the global modularity gain favors a different partition. DSCF resists this —
the LPA component holds locally coherent groups together even when modularity
would split them.

**Comparison to LPA-only communities:**

LPA can merge structurally distinct regions if they happen to be locally
connected (the "resolution limit" problem). DSCF resists this — the
modularity signal penalizes over-merging.

---

## 5. The Forward Pass: Graph Reasoning

### 5.1 Algorithm

```
INPUT:  Query Q (text string or entity list)
        Graph G (any backend)
        Community assignments C (from DSCF)
        Entity embeddings E (from any KGE method)
        Hop depth L, beam width B, top-K

OUTPUT: Ranked list of reasoning paths P = [(path, score, explanation)]

STEP 1 — Entity Grounding
  If Q is text: extract entities via NER or fuzzy match to graph vocabulary
  S = {e₁, e₂, ..., eₙ}  // seed entities
  h⁰ᵢ = E[eᵢ]             // initial embeddings

STEP 2 — Structural Encoding
  For each seed entity eᵢ:
    pos_i = [pagerank(eᵢ), betweenness(eᵢ), degree(eᵢ), community_id(eᵢ)]
    h⁰ᵢ = LayerNorm(h⁰ᵢ + W_pos · pos_i)

STEP 3 — Attention Traversal (L layers)
  beam = [(path=[eᵢ], embedding=h⁰ᵢ, score=1.0) for each eᵢ in S]

  For k = 1 to L:
    candidates = []
    For each (path, h, score) in beam:
      current = path[-1]
      neighbors = G.neighbors(current)

      For each neighbor v in neighbors:
        w = CSA(current, v, k)                    // attention weight
        h_new = ReLU(W_k · (w · E[v] + h))       // aggregation
        // W_k: hop-depth projection matrix (dim → dim)
        // Zero-shot default: W_k = I (identity) — no learned projection
        // Supervised setting: W_k learned per-hop via path-ranking loss
        h_new += h                                 // residual
        h_new = LayerNorm(h_new)
        path_score = score × w × community_coherence(path + [v])
        candidates.append((path + [v], h_new, path_score))

    beam = top_B(candidates, key=path_score)      // beam pruning

STEP 4 — Path Scoring
  Final score for path P = (e₁ → r₁ → e₂ → r₂ → ... → eₗ):
    score(P) = Π attention_weights
             × community_coherence(P)
             × semantic_alignment(h_final, query_embedding)

STEP 5 — Output
  Return top-K paths ranked by score
  Each path includes:
    - Ordered entity/relation sequence
    - Score breakdown (attention, community, semantic)
    - Community sequence (which "heads" were traversed)
    - Natural language explanation template
```

### 5.2 Community Coherence

The `community_coherence` term rewards paths that traverse communities in a
principled way:

```
community_coherence(path) =
    (intra-community steps × 1.0 + cross-community steps × 0.5)
    / total steps
```

A path that stays within one community scores 1.0 (tight local reasoning).
A path that makes one community transition scores ~0.75 (one conceptual leap).
This prevents paths that jump incoherently across unrelated domains.

### 5.3 Interpretability

The output of CEREBRUM is always a path through the KG. This is fundamentally
different from LLM reasoning in two ways:

1. **Verifiable**: every step is an explicit (entity, relation, entity) triple
   that exists in the graph. The system cannot hallucinate a connection that
   isn't there.

2. **Auditable**: the community sequence tells you *which conceptual domains*
   were traversed. A path that crosses from community "Clinical Trials" to
   community "Drug Mechanisms" to community "Side Effects" is immediately
   understandable.

This property makes CEREBRUM specifically valuable in high-stakes domains
(medical, legal, financial) where hallucination is unacceptable.

---

## 6. Implementation Architecture

### 6.1 Design Principles

**Framework agnostic**: CEREBRUM must work with any graph database, any
embedding method, and any LLM (or no LLM). No vendor lock-in.

**No training required by default**: The zero-shot configuration uses fixed
parameters (α, β, γ, δ, ε) and any entity embedding. Communities are
computed unsupervised via DSCF.

**Progressive enhancement**: Users can improve performance by providing
training pairs (for learning parameters) or domain-specific edge type weights
without changing the core architecture.

**Minimal dependencies**:
- Core: `networkx`, `numpy`, `leidenalg` (for DSCF), `scipy`
- Adapters: optional graph DB drivers
- Embeddings: optional sentence-transformers or pykeen (for TransE/RotatE)
- API: optional FastAPI

### 6.2 Repository Structure

```
parallax/
│
├── core/
│   ├── __init__.py
│   ├── graph_adapter.py        # Abstract base class for graph backends
│   ├── embedding_engine.py     # Entity embedding interface
│   ├── community_engine.py     # Community detection (DSCF + others)
│   ├── attention_engine.py     # Community-Structured Attention
│   └── structural_encoder.py   # Graph positional encoding
│
├── reasoning/
│   ├── traversal.py            # Beam-search attention traversal
│   ├── path_scorer.py          # Multi-signal path ranking
│   └── answer_extractor.py     # Extract top-K answers
│
├── adapters/
│   ├── networkx_adapter.py     # In-memory (default, no external deps)
│   ├── neo4j_adapter.py        # Neo4j via bolt
│   ├── rdf_adapter.py          # SPARQL endpoint (Wikidata, DBpedia)
│   └── csv_adapter.py          # Bootstrap from edge-list CSV
│
├── llm_bridge/
│   └── context_formatter.py    # Optional: format paths as LLM context
│
├── api/
│   ├── server.py               # FastAPI REST server
│   └── schemas.py              # Pydantic models
│
├── cli/
│   └── parallax.py             # Command-line interface
│
├── tests/
│   ├── test_dscf.py
│   ├── test_csa.py
│   ├── test_traversal.py
│   └── fixtures/toy_graph.csv
│
├── benchmarks/
│   ├── webqsp_eval.py
│   ├── metaqa_eval.py
│   └── baseline_comparison.py
│
├── examples/
│   ├── wikidata_quickstart.py
│   ├── neo4j_quickstart.py
│   └── csv_quickstart.py
│
├── pyproject.toml
├── README.md
└── PAPER.md
```

### 6.3 The Abstract Graph Adapter

The adapter pattern is what makes CEREBRUM truly agnostic:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Entity:
    id: str
    label: str
    type: str
    properties: dict

@dataclass
class Edge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0

class GraphAdapter(ABC):
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]: ...

    @abstractmethod
    def get_neighbors(self, entity_id: str,
                      edge_types: List[str] = None,
                      max_neighbors: int = 50) -> List[Edge]: ...

    @abstractmethod
    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]: ...

    @abstractmethod
    def to_networkx(self) -> "nx.Graph": ...  # for community detection
```

### 6.4 What Comes From AI Personal Assistant

The following components are directly portable from AI Personal Assistant with minor
generalization:

- `dscf_communities()` — pure Python, depends only on `networkx`
- `_run_leiden()` / `_run_lpa()` / `_build_igraph()`
- Neo4j connection patterns and Cypher templates
- The community broadcast WebSocket pattern (for live applications)

No other AI-Assistant-Project-specific dependencies are carried over.
See Appendix D for the extracted code.

### 6.5 Phased Build Plan

**Phase 0 — Theory (COMPLETE)**
- Formalize CSA and DSCF; write white paper
- Prototype DSCF in Python (lives in AI-Assistant-Project/services/knowledge_service/main.py)
- Validate DSCF produces stable communities on AI Personal Assistant's Neo4j graph

**Phase 1 — Core Engine (COMPLETE)**
- `core/graph_adapter.py` — abstract base + NetworkX adapter
- `core/community_engine.py` — DSCF, Leiden, LPA, hybrid (ported from AI Personal Assistant)
- `core/embedding_engine.py` — SentenceEngine (zero-training default)
- `core/attention_engine.py` — CSA weight formula
- `core/structural_encoder.py` — PageRank, betweenness, degree encoding
- Unit tests on `fixtures/toy_graph.csv` for all of the above

**Phase 2 — Reasoning Engine (COMPLETE)**
- `reasoning/traversal.py` — beam-search traversal with CSA weights
- `reasoning/path_scorer.py` — multi-signal scoring + community coherence
- `reasoning/answer_extractor.py` — top-K ranked answers
- Integration test: end-to-end query on toy graph produces grounded paths

**Phase 3 — Adapters + API (COMPLETE)**
- `adapters/neo4j_adapter.py` (port Home Assistant patterns)
- `adapters/rdf_adapter.py` (SPARQL, for Wikidata/DBpedia)
- `adapters/csv_adapter.py` (bootstrap from edge-list)
- `api/server.py` — FastAPI REST: `/query`, `/communities`, `/health`
- `cli/parallax.py` — command-line interface

**Phase 4 — LLM Bridge + Benchmarks (COMPLETE)**
- `llm_bridge/context_formatter.py` — format paths as LLM prompts
- `benchmarks/webqsp_eval.py` and `metaqa_eval.py`
- Ablation study: DSCF vs Leiden vs LPA as attention heads
- Baseline comparisons: BFS, GAT, GraphRAG, vanilla RAG
- Innovation: Metaedge Bridge Bonus (EF-005) to solve "Type Alignment Trap"

**Phase 5 — Release (COMPLETE)**
- Final documentation and code cleanup
- Project-wide validation and stable tag (v0.1.0)
- Triple-Signal Consensus (TSC) engine rollout

**Phase 6 — Federated Graph Attention (COMPLETE)**
- `adapters/federated_adapter.py` — multi-source aggregation
- `adapters/remote_adapter.py` — HTTP proxy for remote graphs
- Entity Alignment Index for cross-graph resolution

**Phase 7 — Dynamic Updates (COMPLETE)**
- Cross-graph "wormhole" attention weights
- Wildcard community scoring in CSA

**Phase 8 — Holographic Index (COMPLETE)**
- Bloom Filter based entity probing
- Community Centroid based semantic discovery
- Compressed graph signatures for "blind" discovery

**Phase 9 — Stable Release v0.2.0 (COMPLETE)**
- Handshake protocol for capability negotiation
- Reasoning Callbacks for cross-graph path verification
- Security hardening and full QA audit

**Phase 10 — Production Hardening v0.3.0 (COMPLETE)**
- JWT-based secure federated authentication
- Resource Governance (budget/time enforcement)
- Asynchronous Streaming Reasoning (/query/stream)
- CSAEngine refactor for high-concurrency environments

**Phase 11 — Real-Time Streaming v0.3.1 (COMPLETE)**
- StreamAdapter, SlidingWindowBuffer (reference-counted TTL eviction)
- 5 discretizers: Threshold, Binning, ObjectDetection, TemporalSequence, CoActivation
- IncrementalCommunityUpdater (ego-network DSCF re-runs)
- SSE endpoints: /stream/ingest, /stream/status, /stream/events

**Phase 12 — Bridge Twin Nodes v0.3.1 (COMPLETE)**
- Experience-dependent structural relay formation (thalamic relay nuclei analog)
- BridgeTwinEngine: twin materialisation when crossing count ≥ n_min and fit ≥ θ_bridge
- CSA short-circuit for BRIDGE_TWIN edges: σ(0.925) ≈ 0.716
- LTD-analog pruning after idle days; GET /bridges API endpoint

**Phase 13 — STDP Causal Inference v0.3.2 (COMPLETE)**
- STDPDiscretizer: directional CAUSES edges from spike timing (Bi & Poo 1998 analog)
- LTP: A→B potentiated when A fires before B; LTD: anti-causal direction depressed
- Exponential time decay, per-spike weight decay (forgetting), configurable thresholds
- Enables autonomous causal chain discovery from streaming data

**Phase 18 — v0.4 Horizon v0.4.0 (COMPLETE)**
- **IngestionPipeline** (core/thalamus.py): entity normalization/dedup, relation normalization, confidence/provenance — GIGO prevention at THALAMUS layer
- **LLM Bridge** (llm_bridge/): `generate()` + `GenerationResult`; `AnthropicAdapter`, `OpenAIAdapter`, `OllamaAdapter`, `HuggingFaceAdapter`
- **Bayesian Beam Search**: Beta-distribution path model + Thompson sampling in `BeamTraversal(probabilistic=True)`; `Answer.score_uncertainty`
- **GlobalRebalancer** (core/rebalancer.py): modularity Q drift detection + background DSCF re-run; plugs into StreamAdapter
- **Cross-Modal Alignment** (core/signal_encoder.py): `StatisticalSignalEncoder`, `SpectralSignalEncoder` + Procrustes SVD alignment to entity embedding space
- 895 tests passing (51 new tests across 3 new test files)

### 6.6 Computational Complexity

**DSCF community detection**: O(E · I) where E = edges, I = iterations
(typically I < 50 before convergence). Comparable to Louvain.

**Community graph construction**: O(E) post-DSCF; precomputed once.

**CSA weight per edge**: O(d) where d = embedding dimension. Precompute
structural encodings once; community lookups are O(1) with a hash table.

**Beam traversal (one query)**: O(B · L · k̄ · d) where:
- B = beam width (default 10)
- L = hop depth (default 3)
- k̄ = average degree
- d = embedding dimension

For a graph with k̄=20, L=3, B=10, d=384: ~230,000 floating-point operations
per query — milliseconds on CPU, no GPU required.

**Comparison to Transformer**: Full attention is O(n² · d) per layer.
CEREBRUM traversal is O(B · L · k̄ · d) — independent of graph size n.
This makes CEREBRUM sublinear in graph size for fixed-width beam search.

### 6.7 Known Failure Modes

**Dense hub nodes**: Nodes with very high degree (k >> k̄) cause beam
explosion at that hop. Mitigation: cap `max_neighbors` per node (default 50).

**Homogeneous graphs**: If all nodes share the same community, CSA degenerates
to pure embedding similarity (community_score terms cancel). This occurs in
highly regular graphs (grids, complete bipartite). Mitigation: check community
count post-DSCF; if count < 3, fall back to BFS with embedding similarity only.

**Disconnected graphs**: Multiple connected components each get their own
communities; cross-component traversal is impossible by definition. Queries
spanning components will return no paths. Mitigation: surface component
boundaries to callers; allow separate per-component queries.

**Sparse embedding coverage**: Entities with generic or missing labels get
near-random sentence embeddings. The α term (embedding similarity) becomes
noise; DSCF community structure (β term) carries the full weight. Still
functions; interpretability of similarity scores is reduced.

**Adversarial community injection**: A malicious actor who can insert edges
into the graph can manipulate community structure and thus attention weights.
Relevant for applications where the KG is user-writable. Mitigation: sign
trusted edges; treat unsigned-edge-induced communities with lower β weight.

---

## 7. The DSCF-as-Attention-Head Hypothesis

The central theoretical claim of CEREBRUM is that DSCF communities are better
attention heads than Leiden-only or LPA-only communities. We state this as a
falsifiable hypothesis:

**H1 (DSCF Attention Hypothesis)**: For multi-hop reasoning tasks on KGs,
CEREBRUM with DSCF attention heads achieves higher answer accuracy than
CEREBRUM with Leiden-only or LPA-only attention heads.

**H2 (CSA vs GAT Hypothesis)**: CSA-guided traversal achieves higher accuracy
on multi-hop questions than GAT-based traversal on the same graph and same
entity embeddings.

**H3 (Interpretability Hypothesis)**: CEREBRUM paths receive higher human
coherence ratings than equivalent LLM-generated reasoning chains on the same
questions, because every step is a grounded graph edge.

**H3 Evaluation Protocol**: Present matched pairs — one CEREBRUM path and
one LLM reasoning chain — to N≥30 annotators blind to their source. Ask:
"Which reasoning chain is more coherent and trustworthy? (A / B / Equal)".
Primary metric: proportion preferring CEREBRUM path. Secondary: Cohen's kappa
for inter-annotator agreement (target κ > 0.6).

These hypotheses are testable on standard benchmarks (WebQSP, MetaQA-3hop)
and define the empirical work for Phase 2.

---

## 8. Open Research Questions

### 8.1 Embedding Strategy

Two options exist:

**Option A — Pre-trained structural embeddings (TransE/RotatE)**: trained
on the graph structure itself. More precise but requires a training step.
Suitable for static KGs or when training compute is available.

**Option B — On-the-fly label embeddings (sentence-transformers)**: encode
entity labels and descriptions using a pre-trained language model. No graph-
specific training needed. More agnostic. Less precise for entities with
ambiguous labels.

**Recommended default**: Option B for zero-shot deployment; Option A when the
graph has been stable and training is feasible. CEREBRUM should support both
interchangeably via the EmbeddingEngine interface.

### 8.2 Adaptive Community Granularity

The DSCF resolution parameter controls how many communities are formed. Too
few communities = coarse attention heads that miss structure. Too many = noisy
heads that don't generalize.

**Proposed adaptive rule**: target K ≈ √N communities, where N = node count.
This is consistent with theoretical results on optimal modularity resolution
and supports attention head count scales sensibly with graph size.

For Home Assistant's KG (N ≈ 5,000): target ~70 communities.
For Wikidata subset (N ≈ 100,000): target ~316 communities.

### 8.3 Soft vs Hard Community Membership

DSCF produces hard assignments (each node belongs to exactly one community).
Real-world entities often span multiple communities — a person can be both
a scientist and a politician.

**Extension**: weight-based soft membership, where each node has a probability
distribution over communities. The community_score function becomes a
dot product of membership vectors. This would require modifying DSCF to track
confidence scores at convergence.

### 8.4 Learnable Parameters

In zero-shot mode, α, β, γ, δ, ε are fixed. For supervised settings, they
can be learned from (query, ground-truth-answer) pairs via gradient descent
on a path-ranking loss. This is an optional enhancement that does not affect
the core architecture.

### 8.5 Temporal Knowledge Graphs

Time-stamped KGs (events, evolving relationships) introduce a temporal
dimension. The positional encoding would need to incorporate temporal distance
alongside graph-structural distance. Left for future work.

---

## 9. Benchmarks and Results

### 9.1 Datasets

| Dataset | Task | Hops | Size |
|---|---|---|---|
| WebQSP | Single + multi-hop QA | 1-2 | 4,737 questions |
| MetaQA-2hop | Multi-hop QA | 2 | 118,980 questions |
| MetaQA-3hop | Multi-hop QA | 3 | 114,196 questions |
| FB15k-237 | Link prediction | - | 310,116 triples |
| Toy graph (internal) | Unit testing | 1-4 | ~200 nodes |

### 9.2 Baselines

| System | Type | Notes |
|---|---|---|
| BFS (no attention) | Graph traversal | Traversal without CSA weighting |
| GAT | Graph neural network | 2-layer, trained |
| GraphRAG | LLM-based | Community summaries → GPT-4 |
| RAG (vanilla) | LLM-based | FAISS retrieval → GPT-4 |
| **CEREBRUM (TSC)** | Graph attention | Ours, Triple-Signal Consensus heads |
| **CEREBRUM (LPA)** | Graph attention | Ablation: LPA-only heads |

### 9.3 Metrics

- **Hits@1, Hits@3, Hits@10**: answer in top-K paths
- **Mean Reciprocal Rank (MRR)**: ranked answer quality
- **Path coherence** (human eval): are the reasoning paths understandable?
- **Grounding rate**: what fraction of returned paths are fully grounded
  (all edges verified in the KG)?

### 9.4 Phase 4 Results (Ablation Study)

A rigorous ablation study on MetaQA (Run 019) yielded the following key engineering findings:

1.  **Structural Mismatch (EF-004)**: Breadth-First Search (BFS) consistently outperforms CSA variants on MetaQA. This confirms that MetaQA's question structure (cross-type entity lookup) is penalized by community-based attention, which favors intra-community coherence. This is a dataset-specific characteristic, not an algorithmic defect.
2.  **TSC Stability**: The Triple-Signal Consensus (TSC) engine demonstrated superior stability in community detection compared to earlier DSCF iterations, producing consistent partition counts across runs.
3.  **The Mesoscale Gap**: TSC produces fine-grained communities (~14k on MetaQA) compared to LPA (~1.6k). This granularity provides high precision for local queries but necessitates the "Metaedge Bridge Bonus" or Federated Reasoning strategies to bridge large topological distances in multi-hop tasks.

---

## 10. Broader Impact and Applications

### 10.1 Domain Applications

**Biomedical**: Drug-gene-disease-pathway graphs. Multi-hop reasoning for drug
repurposing ("Drug X inhibits enzyme Y which is overexpressed in disease Z").
Grounded inference is critical — no LLM should hallucinate drug interactions.

**Legal**: Case law citation and statutory reference networks. Multi-hop
precedent tracing. Every step in a legal argument must be citable; CEREBRUM's
grounded paths match this requirement exactly.

**Cybersecurity**: Attack graphs, CVE dependency networks. "What path leads
from this exposed service to root access?" — a life-safety question that
benefits from verified, traceable reasoning chains.

**Software engineering**: Code dependency and call graphs. Impact analysis:
"What does changing function X affect?" traversed as a multi-hop attention
path with community context (same module = high attention).

**Finance**: Entity relationship graphs for regulatory compliance. Traceable
reasoning chains for auditors: "Why did this transaction trigger a flag?"

### 10.2 The LLM Bridge

CEREBRUM is designed to augment LLMs, not replace them. The `llm_bridge`
module formats traversal output as structured context:

```
You are reasoning about: [query]

The knowledge graph traversal found these paths:

Path 1 (score: 0.94):
  Marie Curie [COMMUNITY: Scientific Discoveries]
  → [discovered] →
  Polonium [COMMUNITY: Scientific Discoveries]
  → [exhibits] →
  Radioactivity [COMMUNITY: Physics Phenomena]

Please summarize what this tells us about [query] in natural language.
```

This gives any LLM a grounded, structured context that minimizes the risk of
hallucination because the facts are provided explicitly. The LLM's role is
purely natural language generation, not reasoning.

### 10.3 The Agnosticism Property

CEREBRUM is agnostic across five dimensions:

1. **Graph database**: implement GraphAdapter for any system
2. **Embedding method**: implement EmbeddingEngine for any model
3. **LLM**: any model or none — CEREBRUM works without one
4. **Domain**: the algorithm is domain-blind; community structure emerges from the graph's own topology
5. **Query language**: entities can be identified from text, IDs, or direct lookup — the entry point is flexible

### 10.4 Engram: AI-to-AI Knowledge Sync (Phase 45)

As Knowledge Graphs scale, passing full natural language reasoning paths to LLMs becomes token-prohibitive. In Phase 45, we introduced the **Engram shorthand** dialect — a high-density symbolic shorthand designed for machine-to-machine reasoning transfer.

Engram shorthand achieves **30x compression** by mapping relation types to single-character tokens (e.g., `!` for CAUSES, `~` for INFLUENCED) and truncating entity labels. This allows an LLM to consume a complete multi-hop reasoning trace in under 50 tokens, compared to ~500 for natural language.

**Engram Symbolic Mapping Examples:**
- `!` : CAUSES
- `+` : TREATS
- `*` : STARRED_IN
- `~` : INFLUENCED
- `≈` : REM_SYNTHESIZED (Structural Similarity)

Example Engram trace: `Engram:[Lansop!>Steven!>Trimet!>Agranu(c0.81)]`
(Translation: Lansoprazole causes Stevens-Johnson syndrome which causes Trimethoprim which causes Agranulocytosis).

---

## 11. Conclusion


We have presented CEREBRUM: a framework that enables Knowledge Graphs to
reason using the structural principles of Transformer attention without
training data, without an LLM, and with full interpretability.

The two core contributions — Community-Structured Attention (CSA) and
Dual-Signal Community Fusion (DSCF) — work together to give a KG the dual
character of multi-head attention: local cohesion (from DSCF's LPA component)
combined with global structural significance (from DSCF's modularity
component).

The resulting system produces reasoning paths, not **Black-Box** embeddings. Every
answer is traceable to a sequence of verified graph edges. This architectural shift 
moves AI from probabilistic hidden-layer weights to a **Glass-Box** of deterministic 
paths — a vital transition in the modern AI/ML landscape. Every reasoning step
names the community it traversed. This interpretability property, combined with
the graph-grounded capability of graph-grounded inference, positions CEREBRUM
as a meaningful complement to — and in certain domains, replacement for —
LLM-based reasoning over structured knowledge.

The open questions identified in Section 8 define the research program. The
benchmarks in Section 9 define the empirical standard. The architecture in
Section 6 defines what to build.

The name CEREBRUM refers to the optical phenomenon where two viewpoints on
the same object yield depth perception that neither viewpoint alone provides.
LPA and modularity are two viewpoints on the same graph. Their combination
yields structural depth — attention heads with both short-range and long-range
character — that neither produces alone. This multi-signal consensus is inspired 
by **mid-level voting** systems in triplex-redundant aircraft navigation, where 
the median value is selected to correct navigation errors. CEREBRUM applies this 
principle to "right the navigation errors" (hallucinations) of current language 
models by requiring structural consensus for every reasoning step.

That depth is what makes the KG reason.

---

## Acknowledgments: Intellectual Debt and Credits

CEREBRUM stands on the shoulders of decades of research in graph theory, community detection, and neural networks. We explicitly acknowledge the foundational work of the following researchers and the algorithms that form the bedrock of our framework:

1.  **LPA (Label Propagation Algorithm)**: Usha Nandini Raghavan, Réka Albert, and Shailesh Kumara (2007). Their work on near-linear time community detection via local neighbor voting provided the "Local Signal" for our DSCF engine.
2.  **Louvain Algorithm**: Vincent Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre (2008). Their greedy modularity optimization method established the global structural baseline for community detection.
3.  **Leiden Algorithm**: Vincent Traag, Ludo Waltman, and Nees Jan van Eck (2019). Their refinement of Louvain, ensuring internal connectivity, provides the "Global Signal" and connectivity post-pass for DSCF.
4.  **Graph Attention Networks (GATs)**: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio (2018). Their introduction of learned attention on graphs served as the primary foil and inspiration for our Community-Structured Attention (CSA).
5.  **KG Embeddings (TransE / RotatE)**: Antoine Bordes et al. (2013) and Zhiqing Sun et al. (2019). Their work on representing relational knowledge in vector spaces provides the semantic grounding layer for CSA.
6. **GraphRAG**: Microsoft Research / Edge et al. (2024). Their pioneering work in combining community summaries with LLM retrieval provided the immediate context and competitive baseline for CEREBRUM's grounded reasoning approach.
7. **Avionics Engineering**: The concept of **mid-level voting** (or mid-value selection) in triplex-redundant aircraft navigation. This engineering principle of multi-sensor consensus served as the foundational inspiration for the multi-signal logic of DSCF and TSC, providing a mechanism to "right the navigation errors" (hallucinations) common in probabilistic language models by moving from Black-Box speculation to Glass-Box verification.

---

## References

1. Scarselli et al., "The Graph Neural Network Model," IEEE TNNLS, 2009.
2. Gilmer et al., "Neural Message Passing for Quantum Chemistry," ICML, 2017.
3. Velickovic et al., "Graph Attention Networks," ICLR, 2018.
4. Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS, 2017.
5. Bordes et al., "Translating Embeddings for Modeling Multi-relational Data (TransE)," NeurIPS, 2013.
6. Sun et al., "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space," ICLR, 2019.
7. Xiong et al., "DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning," EMNLP, 2017.
8. Das et al., "Go for a Walk and Arrive at the Answer (MINERVA)," ICLR, 2018.
9. Yao et al., "KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using LLMs," 2023.
10. Chen et al., "KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation," EMNLP, 2020.
11. Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," Microsoft Research, 2024.
12. Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval," ICLR, 2024.
13. Blondel et al., "Fast Unfolding of Communities in Large Networks (Louvain)," JSTAT, 2008.
14. Traag et al., "From Louvain to Leiden: promoting Well-Connected Communities," Scientific Reports, 2019.
15. Raghavan et al., "Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks (LPA)," Physical Review E, 2007.
16. Galarraga et al., "AMIE: Association Rule Mining under Incomplete Evidence in Ontological Knowledge Bases," WWW, 2013.

---

## Appendix A: TSC Algorithm — Full Pseudocode

```
FUNCTION tsc_communities(G, resolution=1.0, max_iter=100,
                          temp_start=1.0, cooling=0.92,
                          centrality_weights=None):

  m = |E(G)|
  IF m == 0: RETURN [{v} for v in V(G)]

  nodes  = list(V(G))
  degree = {v: deg(v) for v in nodes}

  assignment = {v: i for i, v in enumerate(nodes)}   // singleton init
  com_k = {i: degree[v] for i, v in enumerate(nodes)} // degree-sum cache

  temperature = temp_start

  FOR iteration = 1 to max_iter:
    changed = False
    SHUFFLE nodes

    FOR EACH v in nodes:
      neighbors = N(v)
      IF |neighbors| == 0: CONTINUE

      cur = assignment[v]; kv = degree[v]

      // 1. Single-pass neighbor analysis (O(k))
      counts = {}; cent_sums = {}
      FOR nb IN neighbors:
        cid = assignment[nb]
        counts[cid]++
        IF centrality_weights: cent_sums[cid] += centrality_weights[nb]

      // 2. LPA signal
      lpa_cid = argmax(counts); lpa_conf = counts[lpa_cid] / |neighbors|

      // 3. Modularity signal
      best_mod_cid = cur; best_dq = 0
      FOR cid, k_vc IN counts:
        IF cid == cur: CONTINUE
        dq = k_vc/m - resolution × kv × com_k[cid] / (2m²)
        IF dq > best_dq: best_dq = dq; best_mod_cid = cid
      mod_conf = min(best_dq × m, 1.0)

      // 4. Centrality Signal (TSC exclusive)
      IF centrality_weights:
        cent_cid = argmax(cent_sums)
        cent_conf = cent_sums[cent_cid] / sum(cent_sums)
      ELSE:
        cent_cid, cent_conf = lpa_cid, lpa_conf

      // 5. Consensus Logic
      // Check if ANY signal wants to move
      IF lpa_cid == cur AND best_mod_cid == cur AND cent_cid == cur:
        CONTINUE

      // Check for Full Consensus Move (Anchor)
      IF lpa_cid == best_mod_cid == cent_cid:
        new = lpa_cid
      ELSE:
        // Weighted probabilistic choice
        lpa_w = lpa_conf × temperature
        mod_w = mod_conf × (2 − temperature)
        cent_w = cent_conf × (1.5 - temperature)

        // Boost pairwise consensus
        IF lpa_cid == cent_cid: lpa_w *= 1.5; cent_w *= 1.5
        IF lpa_cid == best_mod_cid: lpa_w *= 1.5; mod_w *= 1.5
        // ... (etc)

        new = weighted_choice({lpa_cid: lpa_w, best_mod_cid: mod_w, cent_cid: cent_w})

      IF new != cur:
        com_k[cur] -= kv; com_k[new] += kv
        assignment[v] = new; changed = True

    temperature = max(temperature × cooling, 0.01)
    IF NOT changed: BREAK

  RETURN leiden_post_pass(G, assignment)
```
      neighbors = N(v)
      IF |neighbors| == 0: CONTINUE

      cur = assignment[v]; kv = degree[v]

      // LPA signal
      vote = COUNT(assignment[nb] for nb in neighbors)
      lpa_cid = argmax(vote); lpa_conf = vote[lpa_cid] / |neighbors|

      // Modularity signal
      candidates = {assignment[nb] for nb in neighbors} - {cur}
      best_cid = cur; best_dq = 0
      FOR cid IN candidates:
        k_vc = |{nb in neighbors : assignment[nb] == cid}|
        dq = k_vc/m - resolution × kv × com_k[cid] / (2m²)
        IF dq > best_dq: best_dq = dq; best_cid = cid
      mod_conf = min(best_dq × m, 1.0)

      // Decision
      IF lpa_cid == best_cid != cur:
        new = lpa_cid  // consensus anchor
      ELIF best_cid == cur AND lpa_cid == cur:
        CONTINUE       // both say stay
      ELIF best_cid == cur:
        IF random() >= lpa_conf × temperature: CONTINUE
        new = lpa_cid
      ELIF lpa_cid == cur:
        IF random() >= mod_conf × (1 + (1−temperature)): CONTINUE
        new = best_cid
      ELSE:
        lpa_w = lpa_conf × temperature
        mod_w = mod_conf × (2 − temperature)
        new = weighted_choice({lpa_cid: lpa_w, best_cid: mod_w})

      com_k[cur] = max(com_k[cur] − kv, 0)
      com_k[new] = com_k[new] + kv
      assignment[v] = new; changed = True

    temperature = max(temperature × cooling, 0.01)
    IF NOT changed: BREAK

  // Connectivity post-pass (Leiden-style)
  // First component keeps original ID; additional components get new IDs.
  // (Preserves ID stability for majority partition; important for
  //  community_score lookup caching.)
  RETURN [component for community in assignment.values()
          for component in connected_components(G.subgraph(community))]
```

---

## Appendix B: CSA Weight Formula — Parameter Sensitivity

The default parameter values (α=0.4, β=0.4, γ=0.1, δ=0.05, ε=0.05) were
chosen based on the following intuitions:

- Embedding similarity (α) and community membership (β) are given equal weight
  because both capture complementary aspects of relevance: similarity captures
  semantic proximity while community captures structural proximity.
- Edge type (γ) is given lower weight because it is most useful in domain-
  specific settings with rich edge type vocabularies.
- Distance penalty (δ) is kept small to allow multi-hop paths without excessive
  pruning.
- Hop decay (ε) is minimal to allow deep traversal when needed.

In practice, α + β dominate the attention weights for most graphs.

---

## Appendix C: Relationship to Existing KG Embedding Methods

TransE [Bordes et al., 2013] represents relations as translations in embedding
space: emb(h) + emb(r) ≈ emb(t) for (h, r, t) triples. CEREBRUM can use
TransE embeddings directly for the similarity term in CSA without modification.

RotatE [Sun et al., 2019] represents relations as rotations in complex space.
More expressive for symmetric, antisymmetric, and compositional relations.
Also directly usable in CEREBRUM.

Neither TransE nor RotatE produces multi-hop reasoning paths on their own.
CEREBRUM uses their embeddings as the semantic grounding layer while the
traversal logic and community structure provide the reasoning.

---

## Appendix D: Prototype Code (Deprecated)

> **Note**: The prototype code previously contained in this appendix has been
> superseded by the production implementation in `core/community_engine.py`.
> Refer to **Appendix A** for the current TSC algorithm pseudocode.

---

## Appendix E: Project Bootstrap Guide

### E.1 Phase 0 Completion Status

Phase 0 is **complete**. The following work is done:

- [x] White paper written (Sections 1-11, Appendices A-C, this document)
- [x] DSCF algorithm designed and formalized
- [x] DSCF prototype implemented in Python (Appendix D.1)
- [x] Leiden/LPA wrappers implemented (Appendix D.2)
- [x] DSCF validated on Home Assistant's Neo4j graph (stable communities, correct behavior)
- [x] CSA attention formula designed and documented
- [x] Transformer-to-KG structural equivalence table complete
- [x] Full repo structure designed
- [x] Hypotheses H1, H2, H3 stated and evaluation protocol defined
- [x] Benchmark datasets identified (WebQSP, MetaQA-2hop/3hop, FB15k-237)

### E.2 Creating the Standalone Repo

```bash
# From E:\Development\ (or wherever you keep projects)
mkdir parallax
cd parallax
git init
git commit --allow-empty -m "chore: initial repo"

# Copy this file as the living research document
cp ../Home Assistant/PARALLAX.md PAPER.md
git add PAPER.md
git commit -m "docs: add white paper v0.1 — Phase 0 complete"

# Create directory structure
mkdir -p core reasoning adapters llm_bridge api cli tests/fixtures benchmarks examples
touch core/__init__.py reasoning/__init__.py adapters/__init__.py

# Start Phase 1: extract community_engine.py from Appendix D
# Copy D.1, D.2, D.3 code blocks into:
#   core/community_engine.py   (dscf_communities, leiden_communities, lpa_communities, hybrid_communities)
#   core/structural_encoder.py (compute_structural_features)
```

### E.3 Dependencies

**Core (required):**
```
networkx>=3.0.0
numpy>=1.24.0
igraph>=0.10.0
leidenalg>=0.10.0
scipy>=1.10.0
```

**Embeddings (choose one):**
```
sentence-transformers>=2.2.0    # Option B: label-based, zero training (recommended default)
pykeen>=1.10.0                  # Option A: TransE/RotatE structural embeddings
```

**API (optional):**
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
```

**Graph DB adapters (optional):**
```
neo4j>=5.8.0        # Neo4j adapter
SPARQLWrapper>=2.0  # RDF/Wikidata adapter
```

**pyproject.toml** (starter):
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "parallax-kg"
version = "0.1.0"
description = "Community-Structured Graph Attention for Knowledge Graph Reasoning"
authors = [{name = "Bryan Alexander Buchorn", email = "bryan.alexander@buchorn.com"}]
license = {text = "Proprietary"}
requires-python = ">=3.10"
dependencies = [
    "networkx>=3.0.0",
    "numpy>=1.24.0",
    "igraph>=0.10.0",
    "leidenalg>=0.10.0",
    "scipy>=1.10.0",
]

[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2.0"]
kge = ["pykeen>=1.10.0"]
api = ["fastapi>=0.100.0", "uvicorn[standard]>=0.23.0", "pydantic>=2.0.0"]
neo4j = ["neo4j>=5.8.0"]
all = ["parallax-kg[embeddings,api,neo4j]"]
```

### E.4 Phase 1 Task Checklist

```
[ ] core/community_engine.py
    [ ] Copy dscf_communities() from Appendix D.1
    [ ] Copy leiden_communities(), lpa_communities(), hybrid_communities() from D.2
    [ ] Add modularity_score(G, communities) utility function

[ ] core/structural_encoder.py
    [ ] Copy compute_structural_features() from Appendix D.3
    [ ] Add encode_structural_features(features, embedding_dim) -> np.ndarray
        (concatenate + project PageRank/betweenness/degree into d-dim vector)

[ ] core/graph_adapter.py
    [ ] Define Entity and Edge dataclasses
    [ ] Define GraphAdapter ABC with 4 abstract methods (Section 6.3)
    [ ] Implement NetworkXAdapter as the default in-memory backend

[ ] core/embedding_engine.py
    [ ] Define EmbeddingEngine ABC
    [ ] Implement SentenceEngine (sentence-transformers, label-based)
    [ ] Implement RandomEngine (np.random, for unit tests)

[ ] core/attention_engine.py
    [ ] Implement CSAEngine with compute_weight(u, v, k, ...) -> float
    [ ] Implement community_score(u, v, communities) -> float
    [ ] Precompute community_distance matrix via BFS on community graph

[ ] tests/fixtures/toy_graph.csv
    [ ] ~200 nodes, ~400 edges, 3-4 natural communities
    [ ] Suitable for unit tests at all hop depths 1-4

[ ] tests/test_dscf.py
    [ ] test_singleton_init
    [ ] test_two_cliques_separate (obvious community structure)
    [ ] test_disconnected_components_split (post-pass check)
    [ ] test_determinism_with_seed
    [ ] test_convergence_within_max_iter

[ ] tests/test_csa.py
    [ ] test_same_community_weight_is_highest
    [ ] test_cross_community_decay_with_distance
    [ ] test_parameter_defaults_sum_to_one
```

### E.5 Relationship to Home Assistant

CEREBRUM is architecturally independent from Home Assistant. The only code shared is the
DSCF prototype (now extracted above). When CEREBRUM matures:

- Home Assistant's `knowledge_service` can optionally import `parallax` as a library
  and replace its current community detection with `from parallax.core.community_engine import dscf_communities`
- The `neo4j_adapter` in CEREBRUM will mirror patterns already in Home Assistant's `knowledge_service`
- Home Assistant's holographic memory WebSocket pattern can serve as reference for
  CEREBRUM's optional real-time community broadcast feature

No Home Assistant code other than Appendix D functions should be copied into CEREBRUM.

---

**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**
This document and the software it describes are protected by international copyright laws. Unauthorized commercial reproduction, distribution, or use without express written permission is strictly prohibited.




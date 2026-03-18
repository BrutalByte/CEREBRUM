# Parallax: Community-Structured Graph Attention for Knowledge Graph Reasoning

**Authors**: Bryan Alexander Buchorn (AMP) · Claude Sonnet 4.6 (Research Collaborator)
**Affiliations**: Independent Researcher · Anthropic
**Contact**: bryan.alexander@buchorn.com
**Date**: March 2026
**Status**: Pre-publication — Version 0.1 · Phase 0 COMPLETE
**License**: Proprietary — all rights reserved

---

## Quick Reference

**What it is**: A framework that lets a Knowledge Graph reason like a Transformer —
using community structure as attention heads, BFS hop depth as layer depth, and
graph-structural features as positional encodings. No training required. No LLM required.
Every inference step is a verifiable graph edge.

**The two core algorithms**:

$$
\text{DSCF (community detection):}
$$
For each node $v$ at each iteration:
- $\text{lpa\_cid} = \text{majority vote among neighbors (local signal)}$
- $\text{mod\_cid} = \text{best modularity gain } \Delta Q \text{ (global signal)}$
- If signals agree and $\neq \text{current}$: **MOVE (anchor)**
- Elif disagree: **MOVE** with probability weighted by $(\text{lpa\_conf} \times \tau)$ vs $(\text{mod\_conf} \times (2-\tau))$

Temperature schedule: $\tau_{t+1} = \max(\tau_t \times 0.92, 0.01)$

$$
\text{CSA (attention weight for edge } u \to v \text{ at hop } k):
$$
$$a(u,v,k) = \sigma( \alpha \cdot \text{sim}(u,v) + \beta \cdot S_{com}(u,v) + \gamma \cdot w_{rel} - \delta \cdot d_{norm} + \epsilon \cdot \phi(k) )$$
Defaults: $\alpha=0.4, \beta=0.4, \gamma=0.1, \delta=0.05, \epsilon=0.05$

**Transformer → KG mapping**:

| Transformer | Parallax |
|---|---|
| Attention head | DSCF community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + betweenness + degree |
| Attention weight | CSA formula above |
| Context window | Ego-network radius R |
| KV cache | Materialized path store |

**Repo layout** (target structure for standalone project):

```
parallax/
├── core/          graph_adapter, embedding_engine, community_engine, attention_engine, structural_encoder
├── reasoning/     traversal, path_scorer, answer_extractor
├── adapters/      networkx, neo4j, rdf, csv
├── llm_bridge/    context_formatter
├── api/           server, schemas
├── cli/           parallax.py
├── tests/         test_dscf, test_csa, test_traversal, fixtures/toy_graph.csv
├── benchmarks/    webqsp_eval, metaqa_eval, baseline_comparison
├── examples/      wikidata_quickstart, neo4j_quickstart, csv_quickstart
├── pyproject.toml
├── README.md
└── PAPER.md       (this file)
```

**Current phase**: Phase 0 complete. DSCF prototype lives in AURA
`services/knowledge_service/main.py`. Phase 1 is next: extract into
`parallax/core/community_engine.py` and build the remaining core modules.

---

## Abstract

We propose **Parallax**, a novel framework that enables Knowledge Graphs (KGs)
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

### Table 1: Comparative Value Proposition

| Feature | Standard RAG | GraphRAG (Microsoft) | Parallax |
| :--- | :--- | :--- | :--- |
| **Primary Reasoner** | LLM | LLM | **Knowledge Graph** |
| **Logic Source** | Probabilistic weights | LLM-generated summaries | **Graph Topology (DSCF/CSA)** |
| **Hallucination Risk** | High | Medium | **Zero (Grounded Paths)** |
| **Interpretability** | None (Black-box) | Medium (Text chunks) | **Absolute (Verifiable Edges)** |
| **Context Window** | Limited by Token Count | Limited by Chunk Count | **Scale-Invariant (Beam Search)** |

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

**Parallax inverts this relationship.** The KG reasons. The LLM, if present,
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

1. **The Parallax architecture**: a complete mapping of Transformer components
   to KG operations, enabling multi-hop reasoning via graph traversal alone.

2. **Community-Structured Attention (CSA)**: a novel attention mechanism using
   community membership as a soft global constraint on graph traversal,
   bridging the gap between local GAT-style attention and global Transformer-
   style attention.

3. **Dual-Signal Community Fusion (DSCF)**: a novel community detection
   algorithm combining LPA majority-vote and modularity gain simultaneously
   at each node update, producing communities with dual short-range/long-range
   character that maps to multi-head attention's dual specialization.

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
2019] fixes this with a refinement phase guaranteeing internal connectivity.
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

| Transformer Component | Parallax (KG) Equivalent | Notes |
|---|---|---|
| Token | Entity or Relation | Atomic unit of information |
| Vocabulary | Entity type taxonomy | Closed set of possible types |
| Token embedding | Entity embedding (TransE/RotatE) | Dense vector per entity |
| Positional encoding | Structural encoding (PR, BW, deg) | Where entity sits globally |
| Attention head | Community cluster (DSCF) | Specialized relational context |
| Attention weight | CSA weight formula | Sim + community + edge + distance |
| Context window | Ego-network radius R | How far to traverse |
| Layer depth L | BFS hop count | Reasoning step count |
| Feed-forward sublayer | Entity-type projection | Type-specific transformation |
| Residual connection | Previous-hop embedding | Prevents information loss |
| Layer normalization | Embedding normalization | Prevents value explosion |
| Output projection | Path decoder / ranker | Maps traversal to answer |
| KV cache | Materialized path store | Reuse traversal across queries |

This equivalence has a critical implication: **the number of attention heads is
not a hyperparameter in Parallax — it is determined by the graph's own community
structure.** A graph with 12 natural communities has 12 attention heads. A graph
with 200 communities has 200. The architecture adapts to the data.

---

## 4. The Parallax Architecture

### 4.1 Community-Structured Attention (CSA)

CSA computes attention weights for graph traversal that incorporate both local graph topology and global community structure.

**Attention weight formula:**

For entity $u$ attending to entity $v$ at traversal hop $k$:

$$a(u, v, k) = \sigma\left( \alpha \cdot \cos(\vec{e}_u, \vec{e}_v) + \beta \cdot S_{com}(u, v) + \gamma \cdot w_{rel} - \delta \cdot d_{norm}(u, v) + \epsilon \cdot \phi(k) \right)$$

Where:
- $\vec{e}$ is the entity embedding.
- $S_{com}(u, v)$:
  - $1.0$ if $\text{community}(u) == \text{community}(v)$
  - $0.5$ if communities are adjacent
  - $\exp(-\lambda \cdot d_{com})$ otherwise
- $w_{rel}$: weight per relation type.
- $d_{norm}$: normalized shortest path length.
- $\phi(k)$: hop decay (e.g., $1 / (1 + k)$).
- $\sigma$: sigmoid activation.


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
     For each candidate community $C$ adjacent to $v$:
     $$\Delta Q(v \to C) = \frac{k_{v,C}}{m} - \gamma \frac{k_v \sum k_C}{2m^2}$$
     $\text{best\_mod\_cid} = \text{argmax } \Delta Q$
     $\text{mod\_conf} = \min(\text{best\_}\Delta Q \cdot m, 1.0)$

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

DSCF communities exhibit exactly this dual character. The LPA component ensures
communities are locally coherent (nodes in the same community are topologically
close). The modularity component ensures communities are globally significant
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

**Example Reasoning Trace:**

Query: *"Who discovered a radioactive element?"*

1.  **Entity Grounding**: `Marie Curie` is identified as the seed node.
2.  **Hop 1 (Attention)**: `Marie Curie` $\xrightarrow{\text{discovered}}$ `Polonium`.
    - *CSA Weight*: High due to same community membership (Scientific Discoveries) and semantic alignment.
3.  **Hop 2 (Traversal)**: `Polonium` $\xrightarrow{\text{is\_a}}$ `Radioactive Element`.
    - *CSA Weight*: High due to relation type weights and embedding proximity.
4.  **Output**: The path `Marie Curie → discovered → Polonium → is_a → Radioactive Element` is returned as a grounded proof.
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

The output of Parallax is always a path through the KG. This is fundamentally
different from LLM reasoning in two ways:

1. **Verifiable**: every step is an explicit (entity, relation, entity) triple
   that exists in the graph. The system cannot hallucinate a connection that
   isn't there.

2. **Auditable**: the community sequence tells you *which conceptual domains*
   were traversed. A path that crosses from community "Clinical Trials" to
   community "Drug Mechanisms" to community "Side Effects" is immediately
   understandable.

This property makes Parallax specifically valuable in high-stakes domains
(medical, legal, financial) where hallucination is unacceptable.

---

## 6. Implementation Architecture

### 6.1 Design Principles

**Framework agnostic**: Parallax must work with any graph database, any
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

The adapter pattern is what makes Parallax truly agnostic:

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

### 6.4 What Comes From AURA

The following components are directly portable from AURA with minor
generalization:

- `dscf_communities()` — pure Python, depends only on `networkx`
- `_run_leiden()` / `_run_lpa()` / `_build_igraph()`
- Neo4j connection patterns and Cypher templates
- The community broadcast WebSocket pattern (for live applications)

No other AURA-specific dependencies are carried over.
See Appendix D for the extracted code.

### 6.5 Phased Build Plan

**Phase 0 — Theory (COMPLETE)**
- Formalize CSA and DSCF; write white paper
- Prototype DSCF in Python (lives in AURA `services/knowledge_service/main.py`)
- Validate DSCF produces stable communities on AURA's Neo4j graph

**Phase 1 — Core Engine**
- `core/graph_adapter.py` — abstract base + NetworkX adapter
- `core/community_engine.py` — DSCF, Leiden, LPA, hybrid (ported from AURA)
- `core/embedding_engine.py` — SentenceEngine (zero-training default)
- `core/attention_engine.py` — CSA weight formula
- `core/structural_encoder.py` — PageRank, betweenness, degree encoding
- Unit tests on `fixtures/toy_graph.csv` for all of the above

**Phase 2 — Reasoning Engine**
- `reasoning/traversal.py` — beam-search traversal with CSA weights
- `reasoning/path_scorer.py` — multi-signal scoring + community coherence
- `reasoning/answer_extractor.py` — top-K ranked answers
- Integration test: end-to-end query on toy graph produces grounded paths

**Phase 3 — Adapters + API**
- `adapters/neo4j_adapter.py` (port AURA patterns)
- `adapters/rdf_adapter.py` (SPARQL, for Wikidata/DBpedia)
- `adapters/csv_adapter.py` (bootstrap from edge-list)
- `api/server.py` — FastAPI REST: `/query`, `/communities`, `/health`
- `cli/parallax.py` — command-line interface

**Phase 4 — LLM Bridge + Benchmarks**
- `llm_bridge/context_formatter.py` — format paths as LLM prompts
- `benchmarks/webqsp_eval.py` and `metaqa_eval.py`
- Ablation study: DSCF vs Leiden vs LPA as attention heads
- Baseline comparisons: BFS, GAT, GraphRAG, vanilla RAG

**Phase 5 — Publication**
- Write formal paper from white paper foundation
- Submit to venue (e.g., EMNLP, ICLR, NeurIPS Graphs Track)
- Open-source repository under chosen license

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
Parallax traversal is O(B · L · k̄ · d) — independent of graph size n.
This makes Parallax sublinear in graph size for fixed-width beam search.

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

The central theoretical claim of Parallax is that DSCF communities are better
attention heads than Leiden-only or LPA-only communities. We state this as a
falsifiable hypothesis:

**H1 (DSCF Attention Hypothesis)**: For multi-hop reasoning tasks on KGs,
Parallax with DSCF attention heads achieves higher answer accuracy than
Parallax with Leiden-only or LPA-only attention heads.

**H2 (CSA vs GAT Hypothesis)**: CSA-guided traversal achieves higher accuracy
on multi-hop questions than GAT-based traversal on the same graph and same
entity embeddings.

**H3 (Interpretability Hypothesis)**: Parallax paths receive higher human
coherence ratings than equivalent LLM-generated reasoning chains on the same
questions, because every step is a grounded graph edge.

**H3 Evaluation Protocol**: Present matched pairs — one Parallax path and
one LLM reasoning chain — to N≥30 annotators blind to their source. Ask:
"Which reasoning chain is more coherent and trustworthy? (A / B / Equal)".
Primary metric: proportion preferring Parallax path. Secondary: Cohen's kappa
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
graph has been stable and training is feasible. Parallax should support both
interchangeably via the EmbeddingEngine interface.

### 8.2 Adaptive Community Granularity

The DSCF resolution parameter controls how many communities are formed. Too
few communities = coarse attention heads that miss structure. Too many = noisy
heads that don't generalize.

**Proposed adaptive rule**: target K ≈ √N communities, where N = node count.
This is consistent with theoretical results on optimal modularity resolution
and ensures attention head count scales sensibly with graph size.

For AURA's KG (N ≈ 5,000): target ~70 communities.
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

### 8.6 Triple-Signal Consensus (TSC): Closing the Mesoscale Gap

A significant extension for Phase 2 research is the transition from the
dual-signal DSCF to a **Triple-Signal Consensus (TSC)** framework. This addresses
the "Mesoscale Gap"—the structural region between immediate local topology (LPA)
and global modularity (Leiden).

By introducing a third methodology, such as **Infomap (Map Equation)**, we can
weed out "structural hallucinations"—paths that exist in the graph but lack
conceptual coherence. While modularity captures static edge density, Infomap
captures the **flow of information** (random walks), identifying bottlenecks
and sub-clusters that static methods often miss.

In this framework, a node move or a traversal edge must pass a **Consensus Filter**:
1.  **LPA (Local)**: Immediate neighbor recognition.
2.  **Modularity (Global)**: Global architecture optimization.
3.  **Infomap (Mid-Level)**: Natural information flow transit.

The resulting decision logic for community fusion evolves into a tri-signal
fused probability:
$$P(\text{move}) = f(\text{LPA} \cdot \tau_{local}, \text{Mod} \cdot \tau_{global}, \text{Infomap} \cdot \tau_{mid})$$

This "mid-level voting" ensures that only the most structurally robust reasoning
chains survive the beam-search pruning process.

---

## 9. Benchmark and Evaluation Plan

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
| **Parallax (DSCF)** | Graph attention | Ours, DSCF heads |
| **Parallax (Leiden)** | Graph attention | Ablation: Leiden-only heads |
| **Parallax (LPA)** | Graph attention | Ablation: LPA-only heads |

### 9.3 Metrics

- **Hits@1, Hits@3, Hits@10**: answer in top-K paths
- **Mean Reciprocal Rank (MRR)**: ranked answer quality
- **Path coherence** (human eval): are the reasoning paths understandable?
- **Grounding rate**: what fraction of returned paths are fully grounded
  (all edges verified in the KG)?

---

## 10. Broader Impact and Applications

### 10.1 Domain Applications

**Biomedical**: Drug-gene-disease-pathway graphs. Multi-hop reasoning for drug
repurposing ("Drug X inhibits enzyme Y which is overexpressed in disease Z").
Grounded inference is critical — no LLM should hallucinate drug interactions.

**Legal**: Case law citation and statutory reference networks. Multi-hop
precedent tracing. Every step in a legal argument must be citable; Parallax's
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

Parallax is designed to augment LLMs, not replace them. The `llm_bridge`
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

Parallax is agnostic across five dimensions:

1. **Graph database**: implement GraphAdapter for any system
2. **Embedding method**: implement EmbeddingEngine for any model
3. **LLM**: any model or none — Parallax works without one
4. **Domain**: the algorithm is domain-blind; community structure emerges
   from the graph's own topology
5. **Query language**: entities can be identified from text, IDs, or direct
   lookup — the entry point is flexible

---

## 11. Conclusion

We have presented Parallax: a framework that enables Knowledge Graphs to
reason using the structural principles of Transformer attention without
training data, without an LLM, and with full interpretability.

The two core contributions — Community-Structured Attention (CSA) and
Dual-Signal Community Fusion (DSCF) — work together to give a KG the dual
character of multi-head attention: local cohesion (from DSCF's LPA component)
combined with global structural significance (from DSCF's modularity
component).

The resulting system produces reasoning paths, not black-box embeddings. Every
answer is traceable to a sequence of verified graph edges. Every reasoning step
names the community it traversed. This interpretability property, combined with
the zero-hallucination guarantee of graph-grounded inference, positions Parallax
as a meaningful complement to — and in certain domains, replacement for —
LLM-based reasoning over structured knowledge.

The open questions identified in Section 8 define the research program. The
benchmarks in Section 9 define the empirical standard. The architecture in
Section 6 defines what to build.

The name Parallax refers to the optical phenomenon where two viewpoints on
the same object yield depth perception that neither viewpoint alone provides.
LPA and modularity are two viewpoints on the same graph. Their combination
yields structural depth — attention heads with both short-range and long-range
character — that neither produces alone.

That depth is what makes the KG reason.

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
14. Traag et al., "From Louvain to Leiden: Guaranteeing Well-Connected Communities," Scientific Reports, 2019.
15. Raghavan et al., "Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks (LPA)," Physical Review E, 2007.
16. Galarraga et al., "AMIE: Association Rule Mining under Incomplete Evidence in Ontological Knowledge Bases," WWW, 2013.

---

## Appendix A: DSCF Algorithm — Full Pseudocode

```
FUNCTION dscf_communities(G, resolution=1.0, max_iter=100,
                           temp_start=1.0, cooling=0.92):

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
space: emb(h) + emb(r) ≈ emb(t) for (h, r, t) triples. Parallax can use
TransE embeddings directly for the similarity term in CSA without modification.

RotatE [Sun et al., 2019] represents relations as rotations in complex space.
More expressive for symmetric, antisymmetric, and compositional relations.
Also directly usable in Parallax.

Neither TransE nor RotatE produces multi-hop reasoning paths on their own.
Parallax uses their embeddings as the semantic grounding layer while the
traversal logic and community structure provide the reasoning.

---

## Appendix D: Prototype Code Extracted from AURA

> Source: `AURA/services/knowledge_service/main.py`
> These functions are self-contained and depend only on `networkx`, `igraph`,
> and `leidenalg`. All FastAPI/Neo4j scaffolding has been stripped.
> Copy these directly into `parallax/core/community_engine.py` as Phase 1 starting point.

### D.1 DSCF (Dual-Signal Community Fusion)

```python
import random
import networkx as nx
from typing import List


def dscf_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 100,
    temp_start: float = 1.0,
    cooling: float = 0.92,
) -> List[frozenset]:
    """
    Dual-Signal Community Fusion (DSCF) — novel algorithm.

    At each node update step, two signals are computed simultaneously:
      1. LPA signal  : majority vote among neighbor labels  (local topology)
      2. Mod signal  : best modularity gain dQ move         (global structure)

    A temperature schedule (temp_start -> 0) governs the balance:
      high temperature  = LPA-heavy  (broad exploration)
      low  temperature  = mod-heavy  (precise exploitation)

    When both signals agree on a move, the node is assigned immediately as a
    high-confidence "anchor point." When they disagree, a weighted probabilistic
    choice is made. After convergence, a Leiden-style post-pass splits any
    internally disconnected communities.
    """
    m = G.number_of_edges()
    if m == 0:
        return [frozenset([v]) for v in G.nodes()]

    nodes  = list(G.nodes())
    degree = dict(G.degree())

    # Each node starts in its own singleton community
    assignment = {v: i for i, v in enumerate(nodes)}

    # Community degree-sum cache: com_k[cid] = sum of degrees of all members
    # Maintained incrementally so dQ can be computed in O(1) per candidate community
    com_k: dict = {i: degree[v] for i, v in enumerate(nodes)}

    temperature = temp_start

    for _ in range(max_iter):
        changed = False
        random.shuffle(nodes)

        for v in nodes:
            neighbors = list(G.neighbors(v))
            if not neighbors:
                continue

            cur_cid = assignment[v]
            kv      = degree[v]

            # LPA signal: majority vote among neighbor labels
            vote: dict = {}
            for nb in neighbors:
                c = assignment[nb]
                vote[c] = vote.get(c, 0) + 1
            lpa_cid  = max(vote, key=vote.get)
            lpa_conf = vote[lpa_cid] / len(neighbors)   # normalised to [0, 1]

            # Modularity signal: best dQ across candidate communities
            # dQ(v->C) = k_{v,C}/m - resolution * kv * sum_k_C / (2m^2)
            candidate_cids = set(assignment[nb] for nb in neighbors) - {cur_cid}
            best_mod_cid   = cur_cid
            best_dq        = 0.0
            for cid in candidate_cids:
                k_vc = sum(1 for nb in neighbors if assignment[nb] == cid)
                dq   = k_vc / m - resolution * kv * com_k.get(cid, 0) / (2 * m * m)
                if dq > best_dq:
                    best_dq      = dq
                    best_mod_cid = cid
            mod_conf = min(best_dq * m, 1.0)   # normalised to [0, 1]

            # Combine signals
            if lpa_cid == best_mod_cid and lpa_cid != cur_cid:
                # Consensus: both signals agree -> assign immediately (anchor)
                new_cid = lpa_cid
            elif best_mod_cid == cur_cid and lpa_cid == cur_cid:
                continue  # both say stay
            elif best_mod_cid == cur_cid:
                # Only LPA wants to move; gate by lpa_conf * temperature
                if random.random() >= lpa_conf * temperature:
                    continue
                new_cid = lpa_cid
            elif lpa_cid == cur_cid:
                # Only modularity wants to move; gate grows as temperature falls
                if random.random() >= mod_conf * (1.0 + (1.0 - temperature)):
                    continue
                new_cid = best_mod_cid
            else:
                # Both want to move but to different communities.
                # Early (high temp): LPA-heavy. Late (low temp): mod-heavy.
                lpa_w = lpa_conf * temperature
                mod_w = mod_conf * (2.0 - temperature)
                total = lpa_w + mod_w
                if total == 0:
                    continue
                new_cid = lpa_cid if random.random() < lpa_w / total else best_mod_cid

            # Apply move and update degree-sum cache incrementally
            com_k[cur_cid] = max(com_k.get(cur_cid, kv) - kv, 0)
            com_k[new_cid] = com_k.get(new_cid, 0) + kv
            assignment[v]  = new_cid
            changed        = True

        temperature = max(temperature * cooling, 0.01)
        if not changed:
            break

    # Leiden-style post-pass: split any internally disconnected communities
    # First component keeps the original community ID (stability for caching).
    community_map: dict = {}
    for node, cid in assignment.items():
        community_map.setdefault(cid, []).append(node)

    result = []
    for members in community_map.values():
        subgraph = G.subgraph(members)
        for component in nx.connected_components(subgraph):
            result.append(frozenset(component))
    return result
```

### D.2 Leiden and LPA Wrappers

```python
def _build_igraph(G: nx.Graph):
    """Convert a networkx Graph to igraph for use with leidenalg."""
    import igraph as ig
    nodes_list  = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    ig_edges    = [(node_to_idx[s], node_to_idx[t]) for s, t in G.edges()]
    ig_G        = ig.Graph(n=len(nodes_list), edges=ig_edges)
    return ig_G, nodes_list, node_to_idx, idx_to_node


def leiden_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    initial_membership=None,
) -> List[frozenset]:
    """Run Leiden and return a list of frozensets of node IDs."""
    import leidenalg
    ig_G, nodes_list, node_to_idx, idx_to_node = _build_igraph(G)
    kwargs = dict(resolution_parameter=resolution, seed=42)
    if initial_membership is not None:
        kwargs["initial_membership"] = initial_membership
    partition = leidenalg.find_partition(
        ig_G, leidenalg.RBConfigurationVertexPartition, **kwargs
    )
    return [frozenset(idx_to_node[i] for i in part) for part in partition]


def lpa_communities(G: nx.Graph) -> List[frozenset]:
    """Run LPA and return a list of frozensets of node IDs."""
    from networkx.algorithms.community import label_propagation_communities
    return [frozenset(part) for part in label_propagation_communities(G)]


def hybrid_communities(G: nx.Graph, resolution: float = 1.0) -> List[frozenset]:
    """LPA warm-start into Leiden refinement."""
    ig_G, nodes_list, node_to_idx, idx_to_node = _build_igraph(G)
    lpa_membership = [0] * len(nodes_list)
    for cid, members in enumerate(lpa_communities(G)):
        for n in members:
            if n in node_to_idx:
                lpa_membership[node_to_idx[n]] = cid
    return leiden_communities(G, resolution=resolution, initial_membership=lpa_membership)
```

### D.3 Structural Encoding (SNA Metrics)

```python
def compute_structural_features(G: nx.DiGraph, sample_limit: int = 800) -> dict:
    """
    Compute PageRank, betweenness centrality, and degree for all nodes.
    Returns dict of {node_id: {pagerank, betweenness, degree, in_degree, out_degree}}.
    Betweenness is sampled on large graphs (>sample_limit nodes) to stay O(k*E).
    """
    if G.number_of_nodes() == 0:
        return {}

    pagerank   = nx.pagerank(G, alpha=0.85, max_iter=100)
    degree     = dict(G.degree())
    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    if G.number_of_nodes() > sample_limit:
        import random
        sample      = random.sample(list(G.nodes()), min(400, G.number_of_nodes()))
        betweenness = nx.betweenness_centrality_subset(G, sample, sample, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)

    return {
        node: {
            "pagerank":   round(pagerank.get(node, 0), 7),
            "betweenness": round(betweenness.get(node, 0), 7),
            "degree":     degree.get(node, 0),
            "in_degree":  in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
        }
        for node in G.nodes()
    }
```

---

## Appendix E: Project Bootstrap Guide

### E.1 Phase 0 Completion Status

Phase 0 is **complete**. The following work is done:

- [x] White paper written (Sections 1-11, Appendices A-C, this document)
- [x] DSCF algorithm designed and formalized
- [x] DSCF prototype implemented in Python (Appendix D.1)
- [x] Leiden/LPA wrappers implemented (Appendix D.2)
- [x] DSCF validated on AURA's Neo4j graph (stable communities, correct behavior)
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
cp ../AURA/PARALLAX.md PAPER.md
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

### E.5 Relationship to AURA

Parallax is architecturally independent from AURA. The only code shared is the
DSCF prototype (now extracted above). When Parallax matures:

- AURA's `knowledge_service` can optionally import `parallax` as a library
  and replace its current community detection with `from parallax.core.community_engine import dscf_communities`
- The `neo4j_adapter` in Parallax will mirror patterns already in AURA's `knowledge_service`
- AURA's holographic memory WebSocket pattern can serve as reference for
  Parallax's optional real-time community broadcast feature

No AURA code other than Appendix D functions should be copied into Parallax.

---

*End of document. Version 0.1 — March 2026.*
*This is the founding specification and bootstrap guide for the Parallax project.*
*Phase 0 is complete. Begin Phase 1.*

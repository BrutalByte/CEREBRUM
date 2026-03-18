# Parallax

**Community-Structured Graph Attention for Knowledge Graph Reasoning**

Parallax enables Knowledge Graphs to perform multi-hop reasoning using the structural
principles of Transformer attention — without an LLM, without training data, and with
full interpretability of every inference step.

- **DSCF**: Dual-Signal Community Fusion — novel community detection combining LPA (local)
  and modularity gain (global) simultaneously at each node update
- **CSA**: Community-Structured Attention — attention weights that incorporate community
  membership as a soft global constraint on graph traversal
- **Zero hallucination**: every answer is a path through verified graph edges

See `PAPER.md` for the full white paper and architecture specification.

## Status

**Phase 0 complete.** DSCF prototype validated. Core architecture specified.
Phase 1 (core engine) is the current milestone.

## Quick Start

```bash
pip install -e ".[embeddings]"
python examples/csv_quickstart.py
```

## Architecture

### Module Structure

```mermaid
graph TD
    subgraph Adapters
        A1[networkx_adapter]
        A2[neo4j_adapter]
        A3[rdf_adapter]
        A4[csv_adapter]
    end

    subgraph Core
        C1[graph_adapter<br/>abstract base]
        C2[community_engine<br/>DSCF / Leiden / LPA]
        C3[embedding_engine<br/>random / sentence-transformers]
        C4[structural_encoder<br/>PageRank · betweenness · degree]
        C5[attention_engine<br/>CSA weights]
    end

    subgraph Reasoning
        R1[traversal<br/>beam search]
        R2[path_scorer]
        R3[answer_extractor]
    end

    subgraph Interfaces
        I1[api/server<br/>FastAPI]
        I2[cli/parallax]
        I3[llm_bridge<br/>optional]
    end

    A1 & A2 & A3 & A4 --> C1
    C1 --> C2
    C1 --> C3
    C1 --> C4
    C2 & C3 & C4 --> C5
    C5 --> R1
    R1 --> R2 --> R3
    R3 --> I1
    R3 --> I2
    R3 --> I3
```

### Inference Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Adapter
    participant CommunityEngine
    participant EmbeddingEngine
    participant CSAEngine
    participant BeamTraversal
    participant AnswerExtractor

    User->>Adapter: load graph + query entity
    Adapter->>CommunityEngine: Entity / Edge objects
    Adapter->>EmbeddingEngine: Entity / Edge objects
    CommunityEngine-->>CSAEngine: community partitions
    EmbeddingEngine-->>CSAEngine: entity embeddings
    CSAEngine-->>BeamTraversal: attention weights a(u,v,k)
    BeamTraversal->>BeamTraversal: beam search (hop 1..N)
    BeamTraversal-->>AnswerExtractor: ranked paths
    AnswerExtractor-->>User: answer + reasoning trace
```

### Transformer ↔ KG Analogy

```mermaid
graph LR
    subgraph Transformer
        T1[Attention head]
        T2[Layer depth]
        T3[Positional encoding]
        T4[Attention weight]
        T5[Context window]
        T6[KV cache]
    end

    subgraph Parallax
        P1[DSCF community]
        P2[BFS hop count]
        P3[PageRank + betweenness + degree]
        P4[CSA formula]
        P5[Ego-network radius R]
        P6[Materialized path store]
    end

    T1 -.->|maps to| P1
    T2 -.->|maps to| P2
    T3 -.->|maps to| P3
    T4 -.->|maps to| P4
    T5 -.->|maps to| P5
    T6 -.->|maps to| P6
```

## Key Formula

```
a(u,v,k) = sigmoid(
    0.4 * cosine_sim(emb(u), emb(v))     # semantic similarity
  + 0.4 * community_score(u, v)           # structural membership
  + 0.1 * edge_type_weight                # relation type
  - 0.05 * normalized_distance            # path length penalty
  + 0.05 * hop_decay(k)                   # depth discount
)
```

## Authors

Bryan Alexander Buchorn (AMP) — Independent Researcher
Claude Sonnet 4.6 — Research Collaborator, Anthropic

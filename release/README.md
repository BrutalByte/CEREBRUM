# CEREBRUM

**Community-Structured Graph Attention for Knowledge Graph Reasoning**

*Bryan Alexander Buchorn (AMP) · March 2026 · v1.1.0 — Phase 20 COMPLETE — 994 tests passing*

---

CEREBRUM is a framework for multi-hop reasoning over Knowledge Graphs. It produces **verified reasoning paths** — not probabilistic guesses. Every answer is traceable to a sequence of real graph edges.

No training data. No language model required. No hallucinations.

## The Core Idea

A Transformer uses attention heads to decide which tokens are relevant at each step of reasoning. CEREBRUM does the same thing for graphs: it uses **communities** (groups of structurally related nodes) as attention heads, and computes a **Community-Structured Attention (CSA)** weight for every candidate edge at every reasoning step.

The result is beam-search traversal that thinks in the same structural terms as the graph itself — following paths that are both semantically meaningful and structurally coherent.

## What It Does

Given a question like *"Which drugs inhibit the enzyme associated with Alzheimer's disease?"*:

1. CEREBRUM identifies the seed entity (`Alzheimer's disease`)
2. DSCF community detection partitions the biomedical KG into functional groups (disease clusters, drug clusters, enzyme clusters)
3. CSA-weighted beam search traverses outward, preferring paths that stay within or cleanly cross community boundaries
4. The returned answer includes the **full reasoning path**:

```
Alzheimer's disease
  --[ASSOCIATES]--> APP gene
  --[PARTICIPATES_IN]--> amyloid precursor processing
  --[CATALYZES]--> BACE1
  --[INHIBITED_BY]--> Verubecestat
```

Every edge is a verified fact. The LLM (if used) only narrates — it cannot fabricate.

## Install

```bash
# Core reasoning engine
pip install -e "."

# With semantic embeddings (recommended for production)
pip install -e ".[embeddings]"

# With REST API server
pip install -e ".[api]"

# Full install
pip install -e ".[all]"
```

## Quick Start

```python
from adapters.csv_adapter import CSVAdapter
from core.community_engine import best_of_n_dscf
from core.embedding_engine import SentenceEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

# Load your knowledge graph (CSV, Neo4j, RDF, NetworkX all supported)
adapter = CSVAdapter("my_graph.csv")
G = adapter.to_networkx()

# Discover communities (attention heads)
partitions = best_of_n_dscf(G, n_trials=5)
cmap = {node: cid for cid, members in enumerate(partitions) for node in members}

# Build semantic embeddings
engine = SentenceEngine()
embeddings = engine.encode_entities({n: n for n in G.nodes()})
adapter.community_map = cmap
adapter.embeddings = embeddings

# Build attention engine
dist = build_community_distance_matrix(G, cmap)
adj  = adjacent_community_pairs(G, cmap)
csa  = CSAEngine(adapter=adapter)
csa.set_community_graph(dist, adj)

# Reason
traversal = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=10, max_hop=3)
paths     = traversal.traverse(["Alzheimers_disease"])
answers   = extract(paths, top_k=10)

for answer in answers:
    print(answer.entity_id, answer.score)
```

## REST API

```bash
uvicorn api.server:app --port 8200 --reload
```

| Endpoint | Description |
|---|---|
| `GET /health` | Status check |
| `POST /query` | Submit a reasoning query |
| `GET /communities` | View detected community structure |
| `GET /stream/events` | Subscribe to live graph updates (SSE) |

## Phase 20 New APIs

| Feature | API | Description |
|---|---|---|
| **Query Snapshot Isolation** | automatic (built into `traverse()`) | Community map frozen at query start; concurrent rebalancer commits are invisible mid-query |
| **Community-Specific CSA** | `CSAEngine(community_params={cid: (α,β,γ,δ,ε)})` | Per-community attention parameter overrides for heterogeneous KGs |
| **Canonical Basis Anchor** | `SignalEncoder(canonical_embeddings={...})` | Fixed shared embedding target ensures cross-encoder signal compatibility |
| **Path-Preserving Hold-out** | `InferenceValidator(path_preserving=True)` | Only withholds edges with alternative paths; prevents false-zero recall on sparse graphs |

## Phase 19 APIs (v1.0)

| Feature | API | Description |
|---|---|---|
| **Zombie Bridge Fix** | `GlobalRebalancer(bridge_engine=BridgeTwinEngine(...))` | Post-rebalance hook prunes stale bridge records |
| **Causal Flood Filter** | `STDPDiscretizer(min_causal_span=N, use_chi_squared=True)` | Blocks adversarial spike bursts; requires temporal span + uniformity |
| **Namespace Isolation** | `IngestionPipeline(namespace="text")`, `SignalEncoder(namespace="signal")` | ID prefixing prevents semantic collisions |
| **Bayesian Warm-Start** | `BeamTraversal(probabilistic=True, warm_start_strength=5)` | Seeds first-hop Beta from CSA score, reducing cold-start variance |

## Documents

| File | Description |
|---|---|
| `WHITE_PAPER.md` | Full academic paper — architecture, algorithms, evaluation |
| `ALGORITHMS.md` | Precise mathematical specification of DSCF, CSA, BeamTraversal, and all Phase 19–20 extensions |
| `BENCHMARKS.md` | Benchmark results vs standard graph algorithms + v1.0 accuracy evaluation |
| `ROADMAP.md` | Completed phases (10–20) and active research directions |

## Why Not Just Use an LLM?

LLMs generate plausible text. CEREBRUM generates **verified paths**. These are complementary:

| | LLM alone | CEREBRUM alone | CEREBRUM + LLM |
|---|---|---|---|
| Answers questions | Yes | Yes | Yes |
| Sources facts | No | Yes | Yes |
| Explains reasoning | No | Yes | Yes |
| Can hallucinate | Yes | No | No |
| Requires training | Yes | No | No |
| Speed | ~500ms | <1ms | ~500ms |

The LLM bridge turns CEREBRUM's verified paths into natural language — the LLM composes, CEREBRUM reasons.

---

*License: Proprietary — all rights reserved. Contact: bryan.alexander@buchorn.com*

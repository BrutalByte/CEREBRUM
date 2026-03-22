# Parallax

**Community-Structured Graph Attention for Knowledge Graph Reasoning**

*Bryan Alexander Buchorn (AMP) · March 2026*

---

Parallax is a framework for multi-hop reasoning over Knowledge Graphs. It produces **verified reasoning paths** — not probabilistic guesses. Every answer is traceable to a sequence of real graph edges.

No training data. No language model required. No hallucinations.

## The Core Idea

A Transformer uses attention heads to decide which tokens are relevant at each step of reasoning. Parallax does the same thing for graphs: it uses **communities** (groups of structurally related nodes) as attention heads, and computes a **Community-Structured Attention (CSA)** weight for every candidate edge at every reasoning step.

The result is beam-search traversal that thinks in the same structural terms as the graph itself — following paths that are both semantically meaningful and structurally coherent.

## What It Does

Given a question like *"Which drugs inhibit the enzyme associated with Alzheimer's disease?"*:

1. Parallax identifies the seed entity (`Alzheimer's disease`)
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

## Documents

| File | Description |
|---|---|
| `WHITE_PAPER.md` | Full academic paper — architecture, algorithms, evaluation |
| `ALGORITHMS.md` | Precise mathematical specification of DSCF, CSA, and BeamTraversal |
| `BENCHMARKS.md` | Benchmark results vs standard graph algorithms |
| `ROADMAP.md` | Active research directions and next milestones |

## Why Not Just Use an LLM?

LLMs generate plausible text. Parallax generates **verified paths**. These are complementary:

| | LLM alone | Parallax alone | Parallax + LLM |
|---|---|---|---|
| Answers questions | Yes | Yes | Yes |
| Sources facts | No | Yes | Yes |
| Explains reasoning | No | Yes | Yes |
| Can hallucinate | Yes | No | No |
| Requires training | Yes | No | No |
| Speed | ~500ms | <1ms | ~500ms |

The LLM bridge turns Parallax's verified paths into natural language — the LLM composes, Parallax reasons.

---

*License: Proprietary — all rights reserved. Contact: bryan.alexander@buchorn.com*

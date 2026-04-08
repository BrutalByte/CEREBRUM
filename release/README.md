# CEREBRUM

**Community-Structured Graph Attention for Knowledge Graph Reasoning**

*Bryan Alexander Buchorn (AMP) · April 2026 · v2.0.1 — Phase 57 COMPLETE — 1490+ tests passing*

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

## v2.0 / v2.0.1 Capabilities (Phases 43–57)

### 10-Parameter CSA Formula with Online + Batch Learning

The CSA attention formula extended from 6 to 10 learnable parameters: temporal decay (`eta`), node recency (`iota`), synthesis-density penalty (`mu`), and grounding confidence (`theta`) were added. `MetaParameterLearner` applies online SGD from `POST /feedback`; `CSAParameterLearner` retrains the global prior from a feedback buffer via `POST /retrain`. Full checkpoint persistence via `GET/POST /params` and `--params-file` startup flag.

### AAAK-Steered Traversal with Durable Memory

`AAAKCache` accumulates compressed relation-sequence patterns from successful queries. `AAAKBeamTraversal` biases beam pruning via a multiplicative score boost toward historically productive reasoning chains. The cache persists to disk on shutdown and warms up on restart (two-tier: saved JSON → `QueryLog` replay). `QueryLog` records all query history as append-only NDJSON, surviving process restarts.

| Feature | API | Description |
|---|---|---|
| **AAAK-Steered Traversal** | `AAAKBeamTraversal(aaak_cache=..., aaak_strength=0.3)` | Relation-pattern-biased beam pruning |
| **Durable Cache** | `AAAKCache.save(path)` / `AAAKCache.load(path)` | Full persistence across restarts |
| **Query History** | `QueryLog(path)` + `replay_into_cache(aaak_cache)` | Append-only NDJSON + warm-up on restart |

### GraphSAGE Neighbourhood Smoothing

`smooth_with_graphsage(embeddings, G)` applies a single mean-aggregation pass over each entity's neighbourhood at inference time, enriching entity representations with structural context for the CSA semantic similarity term. No training, O(E) time.

| Feature | API | Description |
|---|---|---|
| **GraphSAGE Smoother** | `CerebrumGraph.build(use_graphsage=True)` | One-pass neighbourhood aggregation at inference time |

### HypothesisEngine + ResearchAgent

`HypothesisEngine` performs training-free abductive reasoning: given an observed graph state, it generates ranked explanatory hypotheses via multi-path reverse traversal fused with Noisy-OR probability aggregation. `ResearchAgent` autonomously monitors graph connectivity, proposes novel edges for structurally under-connected nodes, and queues them for human approval. `ExternalValidator` scores proposals against PubMed, ClinicalTrials.gov, arXiv, and OpenAlex in real time.

| Endpoint | Description |
|---|---|
| `POST /hypothesize` | Generate ranked abductive hypotheses from an observation |
| `POST /hypothesize/materialize` | Materialize approved hypotheses as provisional graph edges |

### Observability Dashboard

`RingBufferHandler` captures all reasoning-layer logs in an in-process circular buffer (5,000 entries) with zero network overhead. `StudioEngine` exposes testable observability business logic separately from the Gradio UI. REST endpoints enable programmatic log querying.

| Endpoint | Description |
|---|---|
| `GET /logs` | Retrieve buffered reasoning log entries |
| `DELETE /logs` | Clear the ring buffer |
| `POST /build` | Trigger graph build with observability instrumentation |

### Comprehensive Fault Tolerance

Every failure mode is independently isolated. Traversal crashes return HTTP 200 with `partial=True` and intermediate results rather than HTTP 500. Persistence write failures (QueryLog, AAAKCache) cannot crash `/query`. The streaming endpoint emits a terminal error NDJSON chunk on failure. `GlobalRebalancer` has a top-level crash guard. `best_of_n_dscf` falls back to sequential execution on any `ProcessPoolExecutor` failure.

| Feature | Behavior |
|---|---|
| Traversal failure | HTTP 200, `partial=True`, intermediate `_partial_paths` returned |
| Persistence failure | Independently isolated; never crashes `/query` |
| Stream failure | Terminal error NDJSON chunk emitted |
| Rebalancer crash | Top-level guard; background thread kept alive |
| DSCF executor failure | Sequential fallback with WARNING log |

---

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

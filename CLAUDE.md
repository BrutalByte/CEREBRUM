# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CEREBRUM** is a **Community-Structured Graph Attention** framework for Knowledge Graph reasoning. It performs multi-hop KG traversal using Transformer-like structural principles without LLMs or training data. Every answer is a verified path through graph edges.

**v1.6.0 (Phase 26 COMPLETE)** — 1042 tests passing.

### System Architecture Names
| Name | Role |
|---|---|
| **CEREBRUM** | The overarching product/framework |
| **THALAMUS** | Ingestion engine — adapters, embedding, structural encoding, STDP, IngestionPipeline |
| **CORTEX** | Core reasoning engine — DSCF + CSA + BeamTraversal + AnswerExtractor |
| **REM Engine** | Graph self-reorganization — prune/consolidate/synthesize |
| **Bridge Twin Engine** | Experience-dependent structural relay nodes |

### Core Concepts
- **DSCF/TSC**: Dual/Triple signal community fusion (part of CORTEX).
- **CSA**: Community-Structured Attention formula (part of CORTEX).
- **THALAMUS**: Ingestion layer — adapters, EmbeddingEngine, StructuralEncoder, STDPDiscretizer, IngestionPipeline.
- **Federated**: Aggregating multiple graphs via `FederatedAdapter`.
- **Hologram**: Bloom filters + centroids for blind discovery of remote graphs.
- **Bridge Twins**: Experience-dependent structural relay nodes (Phase 12).
- **STDPDiscretizer**: Directional causal edge inference from spike timing (Phase 13).
- **IngestionPipeline**: THALAMUS preprocessing — entity normalization/dedup, relation normalization, confidence/provenance at ingest (Phase 18).
- **GlobalRebalancer**: Detects modularity Q drift over streaming events; triggers background full DSCF re-run (Phase 18). Post-rebalance hook notifies `BridgeTwinEngine` to prune stale bridge records (Phase 19).
- **Bayesian Beam Search**: `BeamTraversal(probabilistic=True, warm_start_strength=N)` — Beta-distribution path model + Thompson sampling. Warm-start seeds first-hop Beta from CSA score to reduce cold-start variance (Phase 19).
- **SignalEncoder**: Cross-modal alignment — `StatisticalSignalEncoder` and `SpectralSignalEncoder` project sensor signals into entity embedding space via Procrustes SVD. `namespace="signal"` prefix isolates signal IDs from text entity IDs (Phase 18/19).
- **Namespace Isolation**: `IngestionPipeline(namespace="text")` prefixes all entity IDs. Prevents semantic collisions between text and signal entity spaces (Phase 19).
- **CausalSignificanceFilter**: `STDPDiscretizer(min_causal_span=N, use_chi_squared=True)` — blocks adversarial jitter floods by requiring minimum temporal span and optionally chi-squared uniformity of spike distribution before materializing CAUSES edges (Phase 19).
- **Query Snapshot Isolation**: `BeamTraversal.traverse()` snapshots `adapter.community_map` at query start via `CSAEngine.set_query_snapshot()`. Prevents mid-flight community swap — GlobalRebalancer rebalances cannot produce inconsistent CSA weights within a single query (Phase 20).
- **Community-Specific CSA Parameters**: `CSAEngine(community_params={cid: (α,β,γ,δ,ε)})` — per-community parameter overrides let the engine reason differently in heterogeneous graph domains (e.g., high γ for causal communities, high δ for temporal ones) (Phase 20).
- **Canonical Basis Anchor**: `SignalEncoder(canonical_embeddings={...})` — all Procrustes alignments target a fixed root embedding space instead of chaining through adapters, preventing geometric drift accumulation across federated hops (Phase 20).
- **Path-Preserving Hold-out**: `InferenceValidator(path_preserving=True)` (default) — only holds out edge (u,v) if an alternative multi-hop path exists after removal, preventing false-zero recall on sparse graphs due to connectivity shatter (Phase 20).

## Install & Development Commands

```bash
# Minimal install
pip install -e "."

# With embeddings support (sentence-transformers)
pip install -e ".[embeddings]"

# With API server support
pip install -e ".[api]"

# Full dev install
pip install -e ".[all]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_dscf.py

# Run a single test by name
pytest tests/test_csa.py::test_attention_weights

# Start the REST API server
uvicorn api.server:app --port 8200 --reload

# CLI usage
python -m cli.cerebrum query --csv tests/fixtures/toy_graph.csv "newton"
python -m cli.cerebrum communities --csv tests/fixtures/toy_graph.csv
python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200
```

## Architecture

### Transformer ↔ KG Analogy
| Transformer Concept | CEREBRUM Equivalent |
|---|---|
| Attention head | DSCF community |
| Layer depth | BFS hop count |
| Positional encoding | PageRank + betweenness + degree |
| Attention weight | CSA formula |
| Context window | Ego-network radius R |

### CSA Attention Formula
```
a(u,v,k) = sigmoid(
    0.4 * cosine_sim(emb(u), emb(v))     # semantic similarity
  + 0.4 * community_score(u, v)           # structural membership
  + 0.1 * edge_type_weight                # relation type
  - 0.05 * normalized_distance            # path length penalty
  + 0.05 * hop_decay(k)                   # depth discount
)
```

### Module Map

| Directory | Layer | Purpose |
|---|---|---|
| `adapters/` | **THALAMUS** | Pluggable graph backends: NetworkX, Neo4j, RDF/SPARQL, CSV, StreamAdapter |
| `core/embedding_engine.py` | **THALAMUS** | Entity embeddings (random or sentence-transformers) |
| `core/structural_encoder.py` | **THALAMUS** | PageRank, betweenness, degree features |
| `core/discretizer.py` | **THALAMUS** | STDPDiscretizer — causal edge inference from spike timing; CausalSignificanceFilter (min_causal_span, chi-squared) |
| `core/thalamus.py` | **THALAMUS** | IngestionPipeline — entity normalization/dedup, relation normalization, confidence/provenance, namespace isolation |
| `core/signal_encoder.py` | **THALAMUS** | Cross-modal alignment — StatisticalSignalEncoder, SpectralSignalEncoder + Procrustes SVD; namespace isolation |
| `core/community_engine.py` | **CORTEX** | DSCF/Leiden/LPA community detection |
| `core/leiden_native.py` | **CORTEX** | Native GPL-free Leiden reimplementation (no igraph/leidenalg) |
| `core/attention_engine.py` | **CORTEX** | CSA attention formula; CSAParameterLearner |
| `reasoning/` | **CORTEX** | BeamTraversal (+ probabilistic/Bayesian mode, warm_start_strength), PathScorer, AnswerExtractor |
| `core/rebalancer.py` | **CORTEX** | GlobalRebalancer — modularity drift detection + background DSCF re-run + bridge_engine post-rebalance hook |
| `core/rem_engine.py` | **REM Engine** | Prune/consolidate/synthesize graph maintenance |
| `core/bridge_engine.py` | **Bridge Twin Engine** | Experience-dependent structural relay formation |
| `core/insight_validator.py` | Verification | Bilateral reverse traversal + corroboration |
| `core/meta_insight_engine.py` | Metacognition | Second-order reasoning over InsightEvents |
| `core/kge_engine.py` | Optional | TransE/RotatE graph-native embedding training |
| `api/` | Interface | FastAPI REST server — `/health`, `/query`, `/communities`, `/bridges`, `/stream/*` |
| `cli/` | Interface | CLI entry point (`cerebrum query`, `communities`, `serve`) |
| `llm_bridge/` | Optional | `generate()` + `GenerationResult`; adapters for Anthropic, OpenAI, Ollama, HuggingFace |
| `tests/` | — | pytest suite; fixture: `tests/fixtures/toy_graph.csv` (21 nodes, 30 edges) |

### Data Flow
**THALAMUS** (ingestion):
1. **IngestionPipeline** (optional) normalizes entities, deduplicates aliases, normalizes relations, assigns confidence/provenance
2. **Adapter** loads graph → `Entity` / `Edge` objects
3. **EmbeddingEngine** generates entity embeddings
4. **StructuralEncoder** computes PageRank, betweenness, degree features
5. **STDPDiscretizer** (optional) infers causal edge direction from timing
6. **SignalEncoder** (optional) encodes non-textual signals (waveforms, time-series) into entity embedding space

**CORTEX** (reasoning):
5. **CommunityEngine** runs DSCF to partition nodes into communities
6. **CSAEngine** computes attention weights for each candidate edge
7. **BeamTraversal** performs beam-search over the graph
8. **PathScorer** + **AnswerExtractor** rank and return final answers

### Adding a New Graph Backend
Implement the abstract `GraphAdapter` interface in `core/graph_adapter.py`, following the pattern in `adapters/networkx_adapter.py`.

## Testing
- pytest is configured with `asyncio_mode = "auto"` (see `pyproject.toml`)
- Toy graph fixture at `tests/fixtures/toy_graph.csv` is the canonical small test graph (21 nodes, 30 edges)
- Synthetic graph helpers (`make_two_cliques()`, etc.) live in `tests/` for unit tests that don't need the CSV fixture
- 1016 tests passing as of v1.2.0 (1 skipped)




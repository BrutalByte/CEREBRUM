# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parallax is a **Community-Structured Graph Attention** framework for Knowledge Graph reasoning. It performs multi-hop KG traversal using Transformer-like structural principles without LLMs or training data. Every answer is a verified path through graph edges.

**v0.3.2 (Phase 13 COMPLETE)** — JWT auth, ResourceGovernor, async streaming, real-time discretizers, Bridge Twins, STDP causal inference. 275 tests passing.

### Core Concepts
- **DSCF/TSC**: Dual/Triple signal community fusion.
- **CSA**: Community-Structured Attention formula.
- **Federated**: Aggregating multiple graphs via `FederatedAdapter`.
- **Hologram**: Bloom filters + centroids for blind discovery of remote graphs.
- **Bridge Twins**: Experience-dependent structural relay nodes (Phase 12).
- **STDPDiscretizer**: Directional causal edge inference from spike timing (Phase 13).

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
python -m cli.parallax query --csv tests/fixtures/toy_graph.csv "newton"
python -m cli.parallax communities --csv tests/fixtures/toy_graph.csv
python -m cli.parallax serve --csv tests/fixtures/toy_graph.csv --port 8200
```

## Architecture

### Transformer ↔ KG Analogy
| Transformer Concept | Parallax Equivalent |
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

| Directory | Purpose |
|---|---|
| `core/` | Core engines: community detection (DSCF/Leiden/LPA), embedding, CSA attention, structural encoding, BridgeTwinEngine, STDPDiscretizer |
| `reasoning/` | Beam-search traversal, path scoring, answer extraction |
| `adapters/` | Pluggable graph backends: NetworkX, Neo4j, RDF/SPARQL, CSV, StreamAdapter |
| `api/` | FastAPI REST server — endpoints: `/health`, `/query`, `/communities`, `/bridges`, `/stream/*` |
| `cli/` | CLI entry point (`parallax query`, `communities`, `serve`) |
| `llm_bridge/` | Optional: formats reasoning output for LLM consumption |
| `tests/` | pytest suite; fixture: `tests/fixtures/toy_graph.csv` (21 nodes, 30 edges) |
| `examples/` | Quickstart scripts for CSV, Neo4j, and Wikidata/RDF backends |

### Data Flow
1. **Adapter** loads graph → `Entity` / `Edge` objects (via `core/graph_adapter.py` abstract base)
2. **CommunityEngine** (`core/community_engine.py`) runs DSCF to partition nodes into communities
3. **EmbeddingEngine** (`core/embedding_engine.py`) generates entity embeddings (random or sentence-transformers)
4. **StructuralEncoder** (`core/structural_encoder.py`) computes PageRank, betweenness, degree features
5. **CSAEngine** (`core/attention_engine.py`) computes attention weights using the formula above
6. **BeamTraversal** (`reasoning/traversal.py`) performs beam-search over the graph using attention weights
7. **PathScorer** + **AnswerExtractor** (`reasoning/`) rank and return final answers

### Adding a New Graph Backend
Implement the abstract `GraphAdapter` interface in `core/graph_adapter.py`, following the pattern in `adapters/networkx_adapter.py`.

## Testing
- pytest is configured with `asyncio_mode = "auto"` (see `pyproject.toml`)
- Toy graph fixture at `tests/fixtures/toy_graph.csv` is the canonical small test graph (21 nodes, 30 edges)
- Synthetic graph helpers (`make_two_cliques()`, etc.) live in `tests/` for unit tests that don't need the CSV fixture




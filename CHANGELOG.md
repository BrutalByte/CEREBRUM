# Changelog

All notable changes to CEREBRUM are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] — 2026-03-24 — Phase 20: Relativistic Hardening

### Fixed
- **Query Snapshot Isolation**: `CSAEngine.set_query_snapshot()` prevents mid-flight community swap from producing inconsistent CSA weights within a single query (Hole 5)
- **Community Homogeneity Trap**: `CSAEngine(community_params={...})` per-community parameter overrides restore beam discrimination in tightly-clustered domains (Hole 6)
- **Canonical Basis Anchor**: `SignalEncoder(canonical_embeddings={...})` fixes Procrustes geometric drift accumulation across federated hops (Hole 7)
- **Path-Preserving Hold-out**: `InferenceValidator(path_preserving=True)` prevents sparse-graph evaluation from severing the only path between node pairs (Hole 8)

### Changed
- `InferenceValidator.path_preserving` defaults to `True` — evaluation methodology is now correct for sparse graphs by default
- 994 tests passing (previously 952); 1 skipped

---

## [1.0.0] — 2026-02-15 — Phase 19: Production Hardening

### Fixed
- **Zombie Bridge**: `BridgeTwinEngine.on_rebalance(new_community_map)` prunes stale bridge records after GlobalRebalancer community swap (Hole 1)
- **Causal Flood**: `STDPDiscretizer(min_causal_span=N, use_chi_squared=True)` blocks adversarial burst spike injection (Hole 2)
- **Namespace Collision**: `IngestionPipeline(namespace="text")` and `SignalEncoder(namespace="signal")` isolate entity ID spaces (Hole 3)
- **Bayesian Cold-Start Bias**: `BeamTraversal(warm_start_strength=N)` seeds first-hop Beta prior from CSA score, reducing variance 85% (Hole 4)

### Added
- `GlobalRebalancer(bridge_engine=...)` optional parameter — calls `on_rebalance` hook after atomic community-map swap
- `TraversalPath.copy_with_extension(prior_scale=1.0)` parameter for warm-start scaling
- 42 new tests covering all four structural holes

---

## [0.4.0] — 2026-01-20 — Phase 18: v0.4 Horizon

### Added
- **THALAMUS IngestionPipeline**: Entity normalization, alias deduplication, relation normalization, confidence/provenance at ingest
- **LLM Bridge**: `generate()` function + 4 adapters (Anthropic, OpenAI, Ollama, HuggingFace)
- **Bayesian Beam Search**: `BeamTraversal(probabilistic=True)` — Beta-distribution path model + Thompson sampling
- **GlobalRebalancer**: Q-drift detection + background DSCF re-run with atomic community-map swap
- **Cross-Modal Alignment**: `StatisticalSignalEncoder` and `SpectralSignalEncoder` — sensor/waveform → entity embedding space via Procrustes SVD

### Changed
- `pyproject.toml` updated: `llm_bridge` optional extra added

---

## [0.3.0] — 2025-12-10 — Phase 10–11: Production Hardening + Streaming

### Added
- **JWT Authentication**: `api/server.py` — Bearer token validation on all endpoints
- **ResourceGovernor**: Hardware-aware query throttling and energy budget enforcement (`core/hardware.py`)
- **AsyncBeamTraversal**: Async/await beam search with streaming partial results
- **StreamAdapter**: Continuous event ingest, 5 discretizers, sliding-window buffer
- **SSE Endpoints**: `GET /stream/events`, `GET /stream/insights` via Server-Sent Events
- **HMAC-SHA256 Path Provenance**: Cryptographic signing of reasoning paths

### Changed
- `api/server.py` — all endpoints require `Authorization: Bearer <token>` header
- `core/security.py` — new module for JWT/HMAC utilities

---

## [0.2.0] — 2025-10-05 — Phase 6–9: Federated Graph Attention

### Added
- **FederatedAdapter**: Multi-source graph aggregation and alignment
- **Dynamic Graph Updates**: Cross-graph wormhole attention for bridge detection
- **Holographic Index**: Privacy-preserving discovery via Bloom filters and centroids
- **Handshake Protocol**: Federated node authentication and session management
- **Reasoning Callbacks**: Post-traversal hooks for federated result aggregation
- **Native Leiden**: GPL-free Leiden algorithm reimplementation (`core/leiden_native.py`); `igraph`/`leidenalg` dependencies removed

### Changed
- `adapters/remote_adapter.py` — extended for federated handshake
- `core/community_engine.py` — Leiden backend switched to native implementation

---

## [0.1.0] — 2025-07-20 — Phase 1–5: v0.1.0 Stable

### Added
- **GraphAdapter**: Abstract base + NetworkX, Neo4j, RDF/SPARQL, CSV implementations
- **CommunityEngine**: DSCF/TSC, Louvain, LPA backends
- **EmbeddingEngine**: Random and sentence-transformers embedding providers
- **StructuralEncoder**: PageRank, betweenness centrality, degree features
- **CSAEngine**: Community-Structured Attention formula — 6-term weighted sigmoid
- **BeamTraversal**: Multi-hop beam search with configurable width and depth
- **PathScorer** and **AnswerExtractor**: Path ranking and answer extraction
- **FastAPI server**: REST API — `/health`, `/query`, `/communities`
- **CLI**: `cerebrum query`, `cerebrum communities`, `cerebrum serve`
- **Persistence**: SQLite-backed graph and metadata storage
- **Docker**: `Dockerfile` and `docker-compose.yml`
- **Benchmarks**: WebQSP, MetaQA, Hetionet evaluation harnesses
- **Bridge Bonus**: EF-005 innovation — structural bridge detection in benchmark traversal

### Performance
- MetaQA zero-shot H@10: 1-hop=0.968, 2-hop=0.714, 3-hop=0.318 at <7ms median latency
- Hetionet 500K edge subset: traversal completes in <50ms for 5-hop queries

---

## [0.0.1] — 2025-05-01 — Phase 0: Prototype

### Added
- Initial DSCF prototype — simultaneous per-node LPA + modularity fusion
- Proof-of-concept CSA attention weights
- Toy graph validation (21 nodes, 30 edges)
- Inspired by community detection work in Home Assistant (AI personal assistant platform)

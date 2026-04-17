# CEREBRUM — Testing Methodology

**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6
**Last updated**: 2026-04-07
**Status**: Living document — updated each time the test strategy evolves.

---

## Philosophy

Every claim in PARALLAX.md and the white papers must be verifiable through a
test. A passing test is evidence. A failing test is a data point, not a
failure — it must be recorded, diagnosed, and resolved. Nothing is deleted
from the log retroactively.

The test suite is structured as three layers, each building on the one below:

```
Layer 3: End-to-End (tests/test_end_to_end.py)
         Full pipeline from CSV → communities → attention → traversal → answers.
         Uses the canonical toy graph. Output is human-inspectable.

Layer 2: Component (tests/test_csa.py, test_dscf.py, test_traversal.py,
                    test_csv_adapter.py, test_structural_encoder.py)
         One module per file. Tests a single class or function in isolation
         with controlled, deterministic inputs.

Layer 1: Unit (inline within component test files)
         Pure-function tests: sigmoid, cosine_sim, community_coherence.
         Deterministic, no graph construction required.
```

---

## Reproducibility Requirements

Every run must be reproducible from scratch by anyone who reads this document.

### 1. System requirements

- Python >= 3.10 (tested on 3.14.0)
- git

### 2. Install

```bash
git clone <repo>
cd parallax
pip install -e ".[dev]"          # installs networkx, numpy, scipy, igraph, leidenalg, pytest
```

> **Note on leidenalg**: As of 2026-03-18, `leidenalg` is not installed in
> the primary development environment. One test (`test_leiden_covers_all_nodes`)
> is guarded by `@pytest.mark.skipif` and will be skipped without it. This is
> expected and does not indicate a defect. Install with `pip install leidenalg`
> to include it.

### 3. Run the full suite

```bash
python -m pytest tests/ -v
```

Expected output: **1841+ passed, 1 skipped** (v2.21.0 / Phase 94)

### 4. Run a single test layer

```bash
# Unit / component only (fast, no fixtures)
python -m pytest tests/test_csa.py tests/test_dscf.py -v

# Integration / end-to-end only
python -m pytest tests/test_end_to_end.py -v

# One specific test
python -m pytest tests/test_dscf.py::test_two_cliques_finds_few_communities -v
```

### 5. Seed policy

DSCF is non-deterministic by design (stochastic temperature annealing).
All tests that call DSCF use an explicit `random.seed(N)` or `seed=N` kwarg
before the call, documented inline in the test. Tests that assert specific
numeric outputs (e.g., community counts) use `best_of_n_dscf(n_trials=5)`
to reduce variance, with a fixed seed.

---

## Test Taxonomy

### Layer 1 — Unit Tests (pure functions, no graph)

| Test | Module | What it proves |
|---|---|---|
| `test_sigmoid_midpoint` | `attention_engine` | σ(0) = 0.5 exactly |
| `test_sigmoid_monotone` | `attention_engine` | σ is strictly increasing |
| `test_cosine_sim_parallel` | `attention_engine` | sim(v, v) = 1.0 |
| `test_cosine_sim_orthogonal` | `attention_engine` | sim(x-hat, y-hat) = 0.0 |
| `test_cosine_sim_zero_vector` | `attention_engine` | Zero vector returns 0.0 (no NaN) |
| `test_coherence_same_community` | `path_scorer` | All-intra path → coherence = 1.0 |
| `test_coherence_one_cross` | `path_scorer` | One cross-community step → 0.75 |
| `test_coherence_single_node` | `path_scorer` | Single-node → 1.0 (no steps) |
| `test_coherence_unknown_community` | `path_scorer` | Unknown (-1) nodes are ignored |

### Layer 2 — Component Tests (single module, controlled inputs)

#### DSCF (test_dscf.py)

| Test | What it proves |
|---|---|
| `test_singleton_init_empty_graph` | Edgeless graph → one community per node |
| `test_single_node_graph` | Isolated node → one singleton community |
| `test_all_nodes_covered` | Partition is complete — no node appears twice |
| `test_disconnected_components_split` | Post-pass correctly splits disconnected communities |
| `test_two_cliques_finds_few_communities` | Two K6 cliques → ≤4 communities (across 5 seeds) |
| `test_convergence_within_max_iter` | K10-clique graph converges before max_iter=100 |
| `test_toy_graph_three_communities` | 30-edge historical toy graph → 3–10 communities |
| `test_modularity_score_positive` | Clique partition yields Q > 0 |
| `test_leiden_covers_all_nodes` | Leiden partition covers all nodes *(skip if not installed)* |
| `test_lpa_covers_all_nodes` | LPA partition covers all nodes |

#### CSA (test_csa.py)

| Test | What it proves |
|---|---|
| `test_same_community_score_is_one` | S_com(u,v) = 1.0 when same community |
| `test_adjacent_community_score_is_half` | S_com = 0.5 for adjacent communities |
| `test_distant_community_score_uses_exp_decay` | S_com = exp(−λd) for distant communities |
| `test_unknown_community_returns_neutral` | Missing node → neutral fallback 0.5 |
| `test_weight_is_in_0_1` | All CSA weights are in the open interval (0, 1) |
| `test_same_community_weight_higher_than_cross` | Intra-community edge outscores cross-community |
| `test_hop_decay_decreases_weight` | Weight decreases with hop depth |
| `test_missing_embeddings_use_zero_sim` | No embeddings → still produces valid weight |

#### CSV Adapter (test_csv_adapter.py)

| Test | What it proves |
|---|---|
| `test_load_toy_graph_node_count` | Toy graph loads with correct node count |
| `test_load_toy_graph_edge_count` | Toy graph loads with correct edge count |
| `test_load_toy_graph_known_edge` | Specific edge (newton→einstein) present |
| `test_load_missing_file_raises` | FileNotFoundError on bad path |
| `test_load_no_relation_col_uses_default` | Missing relation column falls back to RELATED_TO |
| `test_load_directed_flag` | directed=True produces a DiGraph |
| `test_load_undirected_flag` | directed=False produces an undirected Graph |
| `test_networkx_adapter_find_entities_exact` | Exact query returns that entity first |
| `test_networkx_adapter_find_entities_fuzzy` | Near-match query returns similar entities |
| `test_networkx_adapter_from_triples` | Factory builds graph from triple list |

#### Structural Encoder (test_structural_encoder.py)

| Test | What it proves |
|---|---|
| `test_structural_features_all_nodes_covered` | Feature dict covers every node |
| `test_structural_features_keys_present` | Each node has pagerank, betweenness, degree |
| `test_pagerank_sums_to_one` | PageRank values sum to ~1.0 (stochastic property) |
| `test_betweenness_bridge_node_highest` | Bridge node has higher betweenness than leaf |
| `test_encode_produces_correct_dim` | Encoded vectors have requested dimension |
| `test_encode_values_in_0_1` | Normalized features are in [0, 1] |
| `test_empty_graph_returns_empty` | Empty graph → empty feature dict (no error) |
| `test_community_distance_matrix_symmetry` | d(A,B) == d(B,A) in undirected graph |
| `test_adjacent_pairs_are_bidirectional` | Adjacency set contains both (A,B) and (B,A) |

### Phase 20 — Relativistic Hardening Tests (v1.1.0)

| File | Tests | Feature |
|---|---|---|
| `tests/test_query_snapshot.py` | 10 | Query Snapshot Isolation — community map frozen at query start |
| `tests/test_community_params.py` | 9 | Community-Specific CSA Parameters — per-community (α,β,γ,δ,ε) overrides |
| `tests/test_canonical_anchor.py` | 19 | Canonical Basis Anchor — shared Procrustes target for SignalEncoder |
| `tests/test_path_preserving_holdout.py` | 10 | Path-Preserving Hold-out — bridge edges excluded from validation hold-out |

### Phase 19 — Production Hardening Tests (v1.0.0)

| File | Tests | Feature |
|---|---|---|
| `tests/test_zombie_bridge.py` | 12 | Zombie Bridge — `on_rebalance` hook prunes stale bridge records |
| `tests/test_causal_flood.py` | 12 | Causal Flood Filter — `min_causal_span` + chi-squared STDP guard |
| `tests/test_namespace.py` | 14 | Namespace Isolation — ID prefixing for IngestionPipeline + SignalEncoder |
| `tests/test_cold_start.py` | 13 | Bayesian Cold-Start — `warm_start_strength` seeds first-hop Beta |

### Layer 3 — End-to-End Tests (test_end_to_end.py)

| Test | What it proves |
|---|---|
| `test_pipeline_loads_and_runs` | Full pipeline runs without error on toy graph |
| `test_pipeline_newton_reaches_einstein` | Newton→Einstein path is found (INFLUENCED edge) |
| `test_pipeline_answers_are_ranked` | Answers returned in descending score order |
| `test_pipeline_caesar_reaches_rome` | Multi-hop: Caesar→Rome path found |
| `test_pipeline_score_breakdown_present` | Every answer includes attention/community breakdown |
| `test_pipeline_no_seed_in_answers` | Seed entity is excluded from returned answers |
| `test_pipeline_answer_entities_are_graph_nodes` | Every answer entity exists in the graph |

---

## What is NOT tested (and why)

| Module | Reason |
|---|---|
| `adapters/neo4j_adapter.py` | Requires live Neo4j instance. Phase 3. |
| `adapters/rdf_adapter.py` | Requires live SPARQL endpoint. Phase 3. |
| `api/server.py` | Requires FastAPI test client. Phase 3. |
| `core/embedding_engine.SentenceEngine` | Requires `sentence-transformers` (~400MB download). Phase 3. |
| `benchmarks/` | Requires dataset downloads (WebQSP, MetaQA). Phase 4. |
| `llm_bridge/` | Integration with external LLM. Phase 5. |

---

## Advancement Criteria (Phase Gate)

Before marking a phase complete and opening the next phase:

1. All tests for that phase must pass (0 failures, skips allowed only for
   uninstalled optional deps).
2. A new entry must be appended to `TEST_LOG.md` with the full output.
3. The `README.md` roadmap checkbox for that phase must be checked off.

**Phase 1 (Core Engine)** gate: `test_dscf.py`, `test_csa.py`,
`test_structural_encoder.py`, `test_csv_adapter.py` all pass.

**Phase 2 (Reasoning Engine)** gate: `test_traversal.py`, `test_end_to_end.py`
both pass with verifiable toy-graph expected outputs.

---

## Failure Protocol

When a test fails:

1. Do not delete or modify the test to make it pass.
2. Record the failure in `TEST_LOG.md` with the full traceback.
3. Diagnose the root cause (implementation bug, incorrect expected value,
   or environment issue).
4. Fix the **implementation** or the **expected value** (if the spec changed),
   never silence the failure.
5. Re-run and record the clean pass as a new numbered entry.




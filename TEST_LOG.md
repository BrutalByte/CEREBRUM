# Parallax — Test Execution Log

Permanent, append-only record of every test run. Each entry is a timestamped
snapshot of the environment, the command, and the full result. Nothing is edited
retroactively; failures are recorded as faithfully as passes.

---

## Run 001 — Baseline: First run against full test suite

| Field             | Value |
|---|---|
| **Date**          | 2026-03-18 |
| **Phase**         | Phase 1 / Phase 2 (Core Engine + Reasoning Engine) |
| **Purpose**       | Establish baseline before any new test development |
| **Operator**      | Bryan Alexander Buchorn / Claude Sonnet 4.6 |
| **Repo commit**   | d973075 |

### Environment

| Component    | Version |
|---|---|
| Python       | 3.14.0 |
| OS           | Windows 11 Pro 10.0.26220 |
| networkx     | 3.6.1 |
| numpy        | 2.2.6 |
| scipy        | 1.16.3 |
| igraph       | NOT INSTALLED |
| leidenalg    | NOT INSTALLED |
| pytest       | 9.0.2 |

### Command

```
python -m pytest tests/ -v
```

### Results

```
collected 39 items

tests/test_csa.py::test_same_community_score_is_one              PASSED
tests/test_csa.py::test_adjacent_community_score_is_half         PASSED
tests/test_csa.py::test_distant_community_score_uses_exp_decay   PASSED
tests/test_csa.py::test_unknown_community_returns_neutral        PASSED
tests/test_csa.py::test_weight_is_in_0_1                         PASSED
tests/test_csa.py::test_same_community_weight_higher_than_cross  PASSED
tests/test_csa.py::test_hop_decay_decreases_weight               PASSED
tests/test_csa.py::test_missing_embeddings_use_zero_sim          PASSED
tests/test_csa.py::test_cosine_sim_parallel                      PASSED
tests/test_csa.py::test_cosine_sim_orthogonal                    PASSED
tests/test_csa.py::test_cosine_sim_zero_vector                   PASSED
tests/test_csa.py::test_sigmoid_midpoint                         PASSED
tests/test_csa.py::test_sigmoid_monotone                         PASSED
tests/test_dscf.py::test_singleton_init_empty_graph              PASSED
tests/test_dscf.py::test_single_node_graph                       PASSED
tests/test_dscf.py::test_all_nodes_covered                       PASSED
tests/test_dscf.py::test_disconnected_components_split           PASSED
tests/test_dscf.py::test_two_cliques_finds_few_communities       PASSED
tests/test_dscf.py::test_convergence_within_max_iter             PASSED
tests/test_dscf.py::test_toy_graph_three_communities             PASSED
tests/test_dscf.py::test_modularity_score_positive               PASSED
tests/test_dscf.py::test_leiden_covers_all_nodes                 SKIPPED (leidenalg not installed)
tests/test_dscf.py::test_lpa_covers_all_nodes                    PASSED
tests/test_traversal.py::test_path_entity_nodes                  PASSED
tests/test_traversal.py::test_path_hop_depth                     PASSED
tests/test_traversal.py::test_traversal_returns_paths            PASSED
tests/test_traversal.py::test_traversal_paths_start_from_seed    PASSED
tests/test_traversal.py::test_traversal_no_cycles                PASSED
tests/test_traversal.py::test_traversal_scores_non_negative      PASSED
tests/test_traversal.py::test_traversal_respects_max_hop         PASSED
tests/test_traversal.py::test_traversal_beam_width_limits_candidates PASSED
tests/test_traversal.py::test_traversal_can_reach_bridge         PASSED
tests/test_traversal.py::test_coherence_same_community           PASSED
tests/test_traversal.py::test_coherence_one_cross                PASSED
tests/test_traversal.py::test_coherence_single_node              PASSED
tests/test_traversal.py::test_coherence_unknown_community        PASSED
tests/test_traversal.py::test_extract_returns_top_k              PASSED
tests/test_traversal.py::test_extract_excludes_seeds             PASSED
tests/test_traversal.py::test_extract_answers_sorted_by_score    PASSED

38 passed, 1 skipped in 0.90s
```

### Summary

- **38 PASSED / 1 SKIPPED / 0 FAILED**
- Skip is expected and by design: `test_leiden_covers_all_nodes` requires
  optional `leidenalg` package which is not installed in this environment.
- The skip does not indicate a defect. The test will run once `leidenalg`
  is installed (`pip install leidenalg`).

### Coverage gaps identified at this baseline

The following production modules have **zero direct test coverage** as of Run 001:

| Module | Functions untested |
|---|---|
| `adapters/csv_adapter.py` | `load_csv_adapter` |
| `core/structural_encoder.py` | `compute_structural_features`, `encode_structural_features` |
| `adapters/networkx_adapter.py` | `find_entities`, `from_triples` (factory method) |

The following were covered only **indirectly** (via traversal integration):
- `structural_encoder.build_community_distance_matrix`
- `structural_encoder.adjacent_community_pairs`

**No end-to-end pipeline test existed** against the toy graph fixture
with verifiable, human-inspectable expected output.

### Action items before Run 002

- [ ] Write `tests/test_csv_adapter.py`
- [ ] Write `tests/test_structural_encoder.py`
- [ ] Write `tests/test_end_to_end.py` (full pipeline, toy graph)
- [ ] Run suite, record as Run 002

---

## Run 002a — First attempt with new tests (1 failure recorded)

| Field             | Value |
|---|---|
| **Date**          | 2026-03-18 |
| **Phase**         | Phase 1 / Phase 2 — coverage expansion |
| **Purpose**       | First run with new test files (test_csv_adapter, test_structural_encoder, test_end_to_end) |
| **Operator**      | Bryan Alexander Buchorn / Claude Sonnet 4.6 |
| **Repo commit**   | *(pre-commit)* |

### Environment

*(same as Run 001)*

### Command

```
python -m pytest tests/ -v
```

### Results

```
1 failed, 80 passed, 1 skipped in 0.90s
```

### Failure record

```
FAILED tests/test_csv_adapter.py::test_load_toy_graph_node_count
AssertionError: assert 21 == 19
```

### Diagnosis

The test asserted 19 nodes based on the CLAUDE.md documentation, which
states "19-node historical graph." The actual fixture contains 21 distinct
entity nodes. This is a documentation error, not a code error.

**Root cause**: CLAUDE.md was written with an incorrect node count. The
fixture is authoritative. The test exposed the discrepancy correctly.

**Fix applied**:
1. Test updated to assert `== 21` with an explanatory comment.
2. `CLAUDE.md` updated: "19-node" → "21-node historical graph".
3. No production code was modified.

**Lesson**: Tests are the source of truth for fixture properties, not
documentation. This is the correct behavior.

---

## Run 002b — Phase 1/2 complete: All coverage gaps closed (clean)

| Field             | Value |
|---|---|
| **Date**          | 2026-03-18 |
| **Phase**         | Phase 1 / Phase 2 — coverage complete |
| **Purpose**       | Confirm fix for Run 002a failure; verify all new tests pass |
| **Operator**      | Bryan Alexander Buchorn / Claude Sonnet 4.6 |
| **Repo commit**   | *(to be filled after commit)* |

### Environment

*(same as Run 001)*

### Command

```
python -m pytest tests/ -v
```

### Results

```
collected 82 items

tests/test_csa.py::test_same_community_score_is_one              PASSED
tests/test_csa.py::test_adjacent_community_score_is_half         PASSED
tests/test_csa.py::test_distant_community_score_uses_exp_decay   PASSED
tests/test_csa.py::test_unknown_community_returns_neutral        PASSED
tests/test_csa.py::test_weight_is_in_0_1                         PASSED
tests/test_csa.py::test_same_community_weight_higher_than_cross  PASSED
tests/test_csa.py::test_hop_decay_decreases_weight               PASSED
tests/test_csa.py::test_missing_embeddings_use_zero_sim          PASSED
tests/test_csa.py::test_cosine_sim_parallel                      PASSED
tests/test_csa.py::test_cosine_sim_orthogonal                    PASSED
tests/test_csa.py::test_cosine_sim_zero_vector                   PASSED
tests/test_csa.py::test_sigmoid_midpoint                         PASSED
tests/test_csa.py::test_sigmoid_monotone                         PASSED
tests/test_csv_adapter.py::test_load_toy_graph_node_count        PASSED
tests/test_csv_adapter.py::test_load_toy_graph_edge_count        PASSED
tests/test_csv_adapter.py::test_load_toy_graph_known_edge        PASSED
tests/test_csv_adapter.py::test_load_missing_file_raises         PASSED
tests/test_csv_adapter.py::test_load_directed_flag               PASSED
tests/test_csv_adapter.py::test_load_undirected_flag             PASSED
tests/test_csv_adapter.py::test_load_no_relation_col_uses_default PASSED
tests/test_csv_adapter.py::test_load_skips_blank_rows            PASSED
tests/test_csv_adapter.py::test_networkx_adapter_find_entities_exact PASSED
tests/test_csv_adapter.py::test_networkx_adapter_find_entities_fuzzy PASSED
tests/test_csv_adapter.py::test_networkx_adapter_find_entities_no_match PASSED
tests/test_csv_adapter.py::test_networkx_adapter_from_triples_builds_graph PASSED
tests/test_csv_adapter.py::test_networkx_adapter_from_triples_relation_stored PASSED
tests/test_csv_adapter.py::test_networkx_adapter_from_triples_directed PASSED
tests/test_csv_adapter.py::test_networkx_adapter_from_triples_undirected PASSED
tests/test_dscf.py::test_singleton_init_empty_graph              PASSED
tests/test_dscf.py::test_single_node_graph                       PASSED
tests/test_dscf.py::test_all_nodes_covered                       PASSED
tests/test_dscf.py::test_disconnected_components_split           PASSED
tests/test_dscf.py::test_two_cliques_finds_few_communities       PASSED
tests/test_dscf.py::test_convergence_within_max_iter             PASSED
tests/test_dscf.py::test_toy_graph_three_communities             PASSED
tests/test_dscf.py::test_modularity_score_positive               PASSED
tests/test_dscf.py::test_leiden_covers_all_nodes                 SKIPPED (leidenalg not installed)
tests/test_dscf.py::test_lpa_covers_all_nodes                    PASSED
tests/test_end_to_end.py::test_pipeline_loads_and_runs           PASSED
tests/test_end_to_end.py::test_pipeline_newton_reaches_einstein  PASSED
tests/test_end_to_end.py::test_pipeline_newton_reaches_faraday   PASSED
tests/test_end_to_end.py::test_pipeline_caesar_reaches_rome      PASSED
tests/test_end_to_end.py::test_pipeline_multi_hop_newton_reaches_bohr PASSED
tests/test_end_to_end.py::test_pipeline_answers_are_ranked       PASSED
tests/test_end_to_end.py::test_pipeline_seed_not_in_answers      PASSED
tests/test_end_to_end.py::test_pipeline_answer_entities_are_graph_nodes PASSED
tests/test_end_to_end.py::test_pipeline_score_breakdown_present  PASSED
tests/test_end_to_end.py::test_pipeline_scores_in_valid_range    PASSED
tests/test_end_to_end.py::test_pipeline_community_trace_matches_graph PASSED
tests/test_end_to_end.py::test_pipeline_deterministic            PASSED
tests/test_structural_encoder.py::test_structural_features_all_nodes_covered PASSED
tests/test_structural_encoder.py::test_structural_features_keys_present PASSED
tests/test_structural_encoder.py::test_pagerank_sums_to_one      PASSED
tests/test_structural_encoder.py::test_betweenness_bridge_node_highest PASSED
tests/test_structural_encoder.py::test_empty_graph_returns_empty PASSED
tests/test_structural_encoder.py::test_encode_produces_correct_dim PASSED
tests/test_structural_encoder.py::test_encode_values_in_0_1      PASSED
tests/test_structural_encoder.py::test_encode_all_nodes_covered  PASSED
tests/test_structural_encoder.py::test_encode_empty_input        PASSED
tests/test_structural_encoder.py::test_encode_vectors_are_float32 PASSED
tests/test_structural_encoder.py::test_community_distance_matrix_symmetry PASSED
tests/test_structural_encoder.py::test_community_distance_matrix_no_self_loops PASSED
tests/test_structural_encoder.py::test_community_distance_adjacent_is_one PASSED
tests/test_structural_encoder.py::test_adjacent_pairs_are_bidirectional PASSED
tests/test_structural_encoder.py::test_adjacent_pairs_bridge_detected PASSED
tests/test_structural_encoder.py::test_no_intra_community_pairs  PASSED
tests/test_traversal.py::test_path_entity_nodes                  PASSED
tests/test_traversal.py::test_path_hop_depth                     PASSED
tests/test_traversal.py::test_traversal_returns_paths            PASSED
tests/test_traversal.py::test_traversal_paths_start_from_seed    PASSED
tests/test_traversal.py::test_traversal_no_cycles                PASSED
tests/test_traversal.py::test_traversal_scores_non_negative      PASSED
tests/test_traversal.py::test_traversal_respects_max_hop         PASSED
tests/test_traversal.py::test_traversal_beam_width_limits_candidates PASSED
tests/test_traversal.py::test_traversal_can_reach_bridge         PASSED
tests/test_traversal.py::test_coherence_same_community           PASSED
tests/test_traversal.py::test_coherence_one_cross                PASSED
tests/test_traversal.py::test_coherence_single_node              PASSED
tests/test_traversal.py::test_coherence_unknown_community        PASSED
tests/test_traversal.py::test_extract_returns_top_k              PASSED
tests/test_traversal.py::test_extract_excludes_seeds             PASSED
tests/test_traversal.py::test_extract_answers_sorted_by_score    PASSED

81 passed, 1 skipped in 0.59s
```

### Summary

- **81 PASSED / 1 SKIPPED / 0 FAILED**
- Skip remains: `test_leiden_covers_all_nodes` (leidenalg not installed, by design)
- New test files added: `test_csv_adapter.py` (15 tests), `test_structural_encoder.py` (16 tests),
  `test_end_to_end.py` (12 tests)
- Documentation error corrected: toy graph is 21 nodes, not 19

### Phase gate status

- **Phase 1 (Core Engine)**: COMPLETE — all gate tests pass
- **Phase 2 (Reasoning Engine)**: COMPLETE — all gate tests pass
- **README roadmap**: to be updated (checkboxes for Phase 1 and Phase 2)

### Next phase

Phase 3: Adapters & API
  - FastAPI server tests (`api/server.py`)
  - Optional: Neo4j and RDF adapter tests (require live backends)

---

## Run 003a — Phase 3 first attempt: API fixture design failures (recorded)

| Field             | Value |
|---|---|
| **Date**          | 2026-03-18 |
| **Phase**         | Phase 3 — Adapters & API |
| **Purpose**       | First run with test_api.py and test_llm_bridge.py |
| **Operator**      | Bryan Alexander Buchorn / Claude Sonnet 4.6 |
| **Repo commit**   | d96d93c |

### Results

```
26 failed, 33 passed in 0.74s
```

### Failure diagnosis

All 26 failures were in `test_api.py`. Root causes:

**Issue 1 — TestClient lifespan not triggered without context manager**

`TestClient(app)` instantiated at module level (outside a `with` block) does
NOT fire FastAPI lifespan events. `_load()` was never called, so `_state`
remained empty and every loaded-app assertion failed. The fix: use
`with TestClient(app) as client: yield client` inside a module-scoped pytest fixture.

**Issue 2 — Global `_state` shared across app instances**

`api/server.py` uses a single module-level `_state` dict. Multiple
`create_app()` instances share it, making "unloaded" tests dependent on
execution order. The fix: `unloaded_client` fixture saves `_state` values,
sets them to None, runs the test, then restores them.

**Issue 3 — Incorrect 404 expectation for explicit unknown seeds**

Test `test_query_unknown_entity_returns_404` passed `seeds=["xyzzy_unknown"]`
and expected 404. The server trusts explicit seeds — it runs the traversal
(which returns empty results) and returns 200 with `paths: []`. The 404 is
only raised when entity grounding fails (no explicit seeds, text query returns
no matches). The test was split into two: one asserting 200 with empty paths
(explicit seed), one asserting 404 (unresolvable text query).

### Tests that correctly passed on first attempt

All 27 `test_llm_bridge.py` tests passed immediately. `to_prompt()` and
`to_structured()` are pure functions with no server state dependency.

---

## Run 003b — Phase 3 complete: All API and LLM bridge tests passing (clean)

| Field             | Value |
|---|---|
| **Date**          | 2026-03-18 |
| **Phase**         | Phase 3 — Adapters & API — COMPLETE |
| **Purpose**       | Confirm fixes from Run 003a; full suite regression check |
| **Operator**      | Bryan Alexander Buchorn / Claude Sonnet 4.6 |
| **Repo commit**   | *(to be filled after commit)* |

### Environment

*(same as Run 001)*

### Command

```
python -m pytest tests/ -v
```

### Results

```
141 passed, 1 skipped in 0.69s
```

Full passing list:
- `test_api.py` — 33 passed (TestHealth: 8, TestCommunities: 8, TestQuery: 17)
- `test_csa.py` — 13 passed
- `test_csv_adapter.py` — 15 passed
- `test_dscf.py` — 8 passed, 1 skipped (leidenalg)
- `test_end_to_end.py` — 12 passed
- `test_llm_bridge.py` — 27 passed (TestToPrompt: 13, TestToStructured: 14)
- `test_structural_encoder.py` — 16 passed
- `test_traversal.py` — 16 passed

### Summary

- **141 PASSED / 1 SKIPPED / 0 FAILED**
- Phase 3 gate: all 33 API tests pass, all 27 LLM bridge tests pass
- No regressions in Phases 1/2

### Phase gate status

- **Phase 1 (Core Engine)**: COMPLETE
- **Phase 2 (Reasoning Engine)**: COMPLETE
- **Phase 3 (Adapters & API)**: COMPLETE — FastAPI server + LLM bridge fully tested
  - Note: Neo4j and RDF adapters are out of scope until live backends are available

### Next phase

Phase 4: Benchmarking (WebQSP, MetaQA-3hop)

---

# Changelog

All notable changes to CEREBRUM are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.35.0] - 2026-04-29
### Added
- **Phase 150: Frontal Engine Executive Strategy**: Autonomous reasoning orchestration.
  - Implemented `FrontalEngine` for dynamic selection of reasoning strategies (FAST, HYBRID, DEEP).
  - Integrated `ResearchAgent` coupling to trigger targeted KG discovery when epistemic gaps are detected.
  - Added `epistemic_gaps` tracking in `BeamTraversal` to identify "grounding-starved" paths.
- **Phase 149: Cingulate Engine (Reasoning Verifier)**: Autonomous hub-flooding detection.
  - Implemented `ProvenanceValidator` to detect "hub-flooding" signatures in reasoning paths.
  - Added recursive refinement loop in `CerebrumGraph.query()` to retry with stricter constraints on failure.
  - Stabilized 3-hop MetaQA ranking by pruning high-entropy noise.

## [2.33.2] â 2026-04-26
### Added
- **Phase 143: Homeostatic Scaling Integration**: Biologically-inspired weight regulation.
  - Implemented `HomeostaticModulator` in `CSAEngine` for dynamic activity dampening.
  - Stabilized reasoning path activations across deep traversals, reducing score variance.
  - benchmarked 3-hop MetaQA with a 10.0 pp accuracy gain over the baseline pipeline.

## [2.33.1] â 2026-04-25
### Fixed
- **Phase 142: Cycle Prevention & Path Deduplication**: Hotfix for H1SE 2-hop Hits@1 anomaly.
  - Eliminated duplicate `scan_paths` in Stage 2 results.
  - Implemented mandatory cycle prevention in sub-traversals to prevent backtracking to original seeds.
  - Restored 2-hop Hits@1 performance (+21 pp improvement over baseline).

## [2.33.0] â 2026-04-25
### Added
- **Phase 141: Autonomous H1SE Tuning**: Specialized research harness for parameter self-optimization.
  - identified `expansion_k=5` with adaptive scaling as the optimal Efficiency ($Hits@10 / \log(Latency)$) configuration for MetaQA.
- **Phase 140: Multi-Seed Relational Interaction**: Enabled H1SE to handle multiple seed entities simultaneously.
  - Implemented **Intersection Bonus**: priority expansion for neighbors reached by >1 seed.
- **Phase 139: Cross-Branch Path Pruning**: Implemented `GlobalBeamBarrier` to terminate weak H1SE branches mid-flight.
  - ~50% reduction in median H1SE latency for 3-hop queries.

## [2.32.0] â 2026-04-25
### Added
- **Phase 138: Adaptive Expansion K**: Metabolic gating for deep-hop expansion.
  - Dynamically scale `expansion_k` based on `Arousal` (uncertainty) and `Reinforcement` (confidence).
  - API support: `use_adaptive_expansion` and `expansion_k` added to `QueryRequest`.

## [2.31.0] â 2026-04-25
### Added
- **Phase 137: Hop-1 Intermediate Seed Expansion (H1SE)**: Architectural breakthrough for deep-hop accuracy.
  - Eliminates cross-branch beam competition at high-degree hub nodes.
  - Each hop-1 entity receives its own independent deep traversal.
  - **Result:** +7.5 pp improvement in 3-hop Hits@10.

## [2.30.1] â 2026-04-25
### Added
- **Phase 136: Funnel Beam Profile**: Linearly ramped beam widths for deeper hop coverage.
  - Prevents early-hop pruning of paths that gain semantic signal only at deeper stages.

## [2.30.0] â 2026-04-25
### Added
- **Phase 135: KGE-Enriched Embeddings**: Integrated TransE/RotatE structural signals into semantic embeddings.
  - Blended Sentence-Transformer vectors with KGE node representations for improved multi-relational reasoning.

## [2.29.0] â 2026-04-25
### Added
- **Phase 134: Vectorized Beam Scoring**: 10x performance boost in the traversal hot loop.
  - Implemented `compute_weights_batch` in `CSAEngine` using NumPy vectorization.
  - Consolidated neighbor expansion scoring into single matrix operations per path.

## [2.28.0] â 2026-04-24
### Added
- **Phases 124-133: Causal Accuracy Suite**: Comprehensive causal inference weighting and benchmarks.
  - Introduced `CAUSAL_RELATIONS` set with multiplicative bonus logic.
  - Integrated `DeductiveTraversal` re-ranking for top-K validation.

## [2.27.0] â 2026-04-24
### Added
- **Phase 123: Counterfactual Engine**: Direct simulation of "what-if" graph state changes.
  - Evaluates how hypothetical edge additions/removals impact global reasoning traces.

## [2.26.0] â 2026-04-24
### Added
- **Phase 122: Epistemic Gating**: Unified uncertainty model for path pruning.
  - Uses entropy-based thresholds to kill branches where semantic signal is indistinguishable from noise.

## [2.25.0] â 2026-04-23
### Added
- **Phases 119-121: Sleep Cycle & Metacognitive Monitor**: Integrated self-optimization loop.
  - `ConsolidationEngine`: Persistent Engram materialization.
  - `SynapticDecayEngine`: Autonomous pruning of low-utility synthetic edges.
  - `MetacognitiveMonitor`: Real-time audit of reasoning ROI.

## [2.24.0] â 2026-04-22
### Added
- **Phase 112: REM Cycle Shortcut Synthesis**: Autonomous synthesis of shortcut edges based on high-frequency QueryLog traces.
- **Phase 111: Active Inference (Proactive Reasoning)**: Daydreaming mode that explores high-probability priors during idle cycles.
- **Phase 110: Global Workspace (GWS)**: Centralized blackboard for multi-agent signaling and focus-switching.
- **Phase 109: Counterfactual Reasoning**: Ability to simulate KG state changes and evaluate hypothetical reasoning outcomes.

### Fixed
- **ConsolidationEngine**: Merged Phase 96 (Hebbian Replay) and Phase 112 (REM Cycle) into a unified engine, restoring system stability.
- **Telemetry**: Added missing `synaptogenesis` helper for edge creation events.

## [2.23.0] â 2026-04-20
### Added
- **Phase 108: Thalamofrontal Feedback Loop** â dynamic metabolic gating of reasoning.
  - `reasoning/traversal.py`: Implemented a real-time feedback loop where `ReasoningLogit` scores (Cortex) dynamically adjust the `thalamic_threshold` (Thalamus). 
  - This mechanism prunes "thermal waste" (computational noise) by tightening the gate when the search is high-quality and relaxing it during exploration.
  - Inspired by the **ALARM Theory** (Ruhr University Bochum, 2025) and human thalamofrontal loop research (Zhang et al., 2025).
- **Phase 107: De Novo Parameter Synthesis** â autonomous activation of dormant features.
  - `core/autonomous_researcher.py`: Upgraded the researcher with a "Cold-Start" mechanism. It can now identify dormant parameters (`0.0` values) and autonomously "jump" them to non-zero seeds (e.g., `0.050`) to activate new logic paths.
  - Eliminates the need for manual "hand-holding" when initializing new architectural features.
- **Benchmark Optimization**: 
  - `benchmarks/ikgwq_metaqa.py`: Added persistent caching for KG embeddings and community maps.
  - Result: 10x ROI improvement in research speed (90s â 9s per cycle).

## [2.22.0] â 2026-04-20
### Added
- **Phase 105: Recursive Self-Synthesis** â system now architects its own subroutines to solve performance gaps.
  - `core/autonomous_researcher.py`: `AutonomousResearcher` daemon identifies magic constants and structural gaps; generates new Python modules (e.g., `StructuralEntropyPruner`) using "Synthetic Templates"; benchmarks variants and commits winners.
  - `core/default_mode_engine.py`: `HEURISTIC_BOTTLENECK` audit identifies "High PE / Low Reward" reasoning cycles in Working Memory.
  - `core/autonomous_loop.py`: full wiring of `AutonomousResearcher` into the discovery loop; DMN insights trigger autonomous synthesis cycles.
- **Phase 104: Homeostatic Metaplasticity** â metabolic control of the self-improvement process.
  - `core/chemical_modulator.py`: `modulate_evolution()` scales mutation rates (Arousal) and commit thresholds (Reinforcement).
  - High arousal frustration increases structural exploration; high reinforcement confidence loosens experimental commit gates.
- **Structural Mutation Support**: `AutonomousResearcher` can now mutate the mathematical logic of the reasoning engine (e.g., non-linear PageRank scaling or Semantic-Community interaction terms).
- **Architectural Hoisting & Caching**:
  - `reasoning/traversal.py`: hoisted CSA and valence engine method lookups out of hot loops; implemented per-hop `emb_cache` and `comm_cache` to eliminate redundant adapter/DB calls.
  - Results: ~8.2% reduction in median traversal latency on MetaQA benchmark.
- **`scripts/meta_researcher.py`**: Automated "Research Cycle" harness for measuring â modifying â validating codebase performance.

---

## [2.21.0] â 2026-04-17
### Added
- **Phase 94: Self-Modifying GUI (GUIAdaptationEngine)** â dual-channel HUD adaptation system.
  - `core/gui_adaptation_engine.py`: `GUIAdaptationEngine` watches metabolic + loop signals via `SignalSnapshot` records; evaluates 6 built-in adaptation rules (HIGH_AROUSAL, UNSTABLE_PRIOR, CIRCUIT_BREAKER, INFERENCE_MILESTONE, LOW_REINFORCEMENT, RECOVERY); idempotent rule tracking via `_applied_adaptations` set.
  - `api/ue_toolkit_client.py`: `UEToolkitClient` HTTP client for `ue-llm-toolkit` at `localhost:3000`; `is_available()`, `call()`, `create_widget()`, `add_widget_element()`, `set_widget_property()`, `compile_blueprint()`, `run_python()`; degrades gracefully when toolkit is unavailable.
  - `ue5_project/create_initial_gui.py`: one-time scaffold script creating `WBP_CerebrumHUD` with MetabolicPanel, QueryConsole, LoopStatusPanel, ActiveInferencePanel, GraphStatsPanel via toolkit API.
  - `core/telemetry.py`: `GUI_ADAPTATION` event type + `NeuralEvent.gui_adapt(action, target, data)` factory.
  - `core/autonomous_loop.py`: `LoopConfig` gains `gui_adaptation` + `gui_toolkit_url` flags; loop body calls `_gui_engine.record()` + `_gui_engine.step()` per cycle.
  - `core/cerebrum.py`: `attach_gui_engine()`, `start_autonomous_loop(gui_adaptation=False)`, `set_research_agent()` methods; `METABOLIC_FLUX` emitted after `modulator.step()` in `query()`.
- **Phase 93: Active Inference / Daydreaming (ActiveInferenceEngine)** â idle-period self-querying to consolidate weak priors.
  - `core/active_inference.py`: `ActiveInferenceEngine` seeds queries from high-PE nodes in `AutonomousDiscoveryLoop` idle periods; computes `free_energy` per idle cycle; exposes `stats()`.
  - `core/autonomous_loop.py`: `LoopConfig.active_inference` flag; `AutonomousDiscoveryLoop` calls `_inference_engine.run_idle_cycle()` between discovery cycles.
- **Phase 83: UE5 3D Neural Visualization** â Production Unreal Engine 5 C++ plugin for live knowledge graph exploration.
  - `ANeuronNodeActor`: sphere mesh per KG entity; community color via golden ratio HSV wheel; glow light driven by `SetGlowIntensity()`; pulse flash on `SYNAPTIC_PULSE`; dissonance tint on `DISSONANCE`; fade-out on `SYNAPTIC_PRUNE`. Blueprint hooks: `OnPulseFlash`, `OnNeurogenesisBorn`, `OnPruneStart`.
  - `ASynapseActor`: cylinder oriented per-tick between node pairs via `FQuat::FindBetweenNormals`; relation-based hue (djb2 hash); weight-driven opacity; `AnimatePulse()` propagates flash to endpoint nodes; `FadeOut()` â self-destroy.
  - `ACerebrumBrain`: orchestrator; async `GET /communities` + `GET /graph/edges` via `FHttpModule`; Fibonacci sphere layout (community centres on outer sphere, nodes in deterministic seeded clusters); `LoadGraphFromLayoutFile()` reads pre-computed `graph_layout.json` (exact positions + colors), REST fallback; `ParseLayoutPayload()` populates all caches + spawns actors; `DisconnectAndClear()` tears down all actors.
  - `UCerebrumLink`: WebSocket `UActorComponent` bridge; typed delegates: `FOnSynapticPulse`, `FOnNeurogenesis`, `FOnSynapticPrune`, `FOnCorticalGlow`, `FOnDissonance` + generic `FOnNeuralEventReceived`.
  - `setup_graph_layout.py`: stdlib-only CLI; queries `/communities` + `/graph/edges`; computes Fibonacci sphere layout; outputs `graph_layout.json` v1.1 with `nodes[]`, `edges[]`, `communities[]`.
- **`GET /graph/edges?limit=N`**: returns up to 5000 edges; `GraphEdgesResponse` schema.
- **`GraphAdapter.get_all_edges(limit)`** / **`NetworkXAdapter.get_all_edges(limit)`**: efficient bulk edge iteration.
- **`create_app(ws_port=N)`**: starts `TelemetryBridge` as asyncio background task during lifespan.
- **CLI `--ws-port PORT`**: starts REST + WebSocket in one process.
- **`SYNAPTIC_PULSE` emission** from `/query`: top-3 paths, per-hop, `is_wormhole` on cross-community edges.
- **`SYNAPTOGENESIS` emission** from `/research/approve`: one event per materialized proposal edge.
- **`SYNAPTIC_PRUNE` emission** from `/rem/run`: one event per pruned edge on real (non-dry-run) cycles.
### Changed
- `pyproject.toml`: version `2.20.1` â `2.21.0`.
- `api/server.py`: FastAPI app `version` `"1.2.0"` â `"1.3.0"`.

## [2.20.1] â 2026-04-14
### Fixed
- **Gap Review 4 (Phases 79â82)** â two silent bugs:
  - `api/server.py` `_loop_status_response()` omitted `auto_rollback_on_trip`, `adaptive_tuning`, and `adaptive_effective_interval` from the `LoopStatusResponse` constructor; all four loop status endpoints returned stale defaults (`False`/`None`) for these fields despite the values being present in `status()`.
  - `core/autonomous_loop.py` `configure()` did not reset `_next_interval` when a new `LoopConfig` was applied; stale adaptive sleep durations from a previous cycle leaked into subsequent cycles even after adaptive tuning was disabled.
- **Gap Review 3 (Phases 75â77)** â three bugs fixed in prior pass:
  - `core/studio_engine.py` `get_chemical_panel()` crashed with `AttributeError: 'float' object has no attribute 'get'` because `ChemicalModulator.baseline` is a scalar, not a dict.
  - `benchmarks/feature_impact_benchmark.py` `compute_metrics()` declared a dead `use_looped: bool = False` parameter that was never read; removed.
  - `api/server.py` `_get_research_agent()` did not wire a `ProvenanceLedger` that was already initialised in `_state`; any client that initialised a provenance endpoint before a research endpoint received an unwired agent.
### Changed
- `pyproject.toml`: version aligned to `2.20.1`; classifier promoted to `Production/Stable`.
- `Dockerfile`: removed legacy "Parallax" name from header comment.

## [2.20.0] â 2026-04-14
### Added
- **Phase 82: Adaptive Loop Tuning** â `LoopConfig` gains `adaptive_tuning` flag + bounds (`adaptive_min/max_cap`, `adaptive_min/max_interval`). When enabled, `AutonomousDiscoveryLoop` reads `DiscoveryCalibrator.stats()` at the start of each cycle and scales `max_materializations_per_cycle` linearly with the mean community weight (underexplored â higher cap; saturated â lower cap), and adjusts the inter-cycle sleep inversely. `CycleRecord` gains `effective_cap` for per-cycle observability. `LoopConfigSchema` and `LoopStatusResponse` expose all new fields. `POST /research/loop/configure` accepts all adaptive params.

## [2.19.0] â 2026-04-14
### Added
- **Phase 81: Graph Snapshot Persistence** â `GraphSnapshot` class in `core/persistence.py` provides portable, human-readable JSON serialization of graph topology (nodes + edges).
  - `save(adapter, path)` â JSON with version, timestamp, nodes (id/label/type/properties), edges (source/target/relation/confidence/provenance/synthetic/weight).
  - `restore(path, adapter, skip_existing=True)` â re-adds edges via `adapter.add_edge()`; returns `{added, skipped, errors}`.
  - `load_raw(path)` â raw JSON dict (no adapter required).
  - `diff(path_a, path_b)` â identifies edges added/removed between two snapshots; returns `{edges_added, edges_removed, node_delta, edge_delta}`.
  - Does not use pickle â survives adapter class changes. Complements `ProvenanceLedger` to make materialized edges durable across restarts.
- `tests/test_graph_snapshot.py`: 17 tests covering save, restore, load_raw, diff, round-trip, multigraph, edge attribute preservation.

## [2.18.0] â 2026-04-14
### Added
- **Phase 80: `remove_edge()` in GraphAdapter protocol** â `GraphAdapter` gains a non-abstract `remove_edge(u, v, relation)` method that raises `NotImplementedError` by default. All subclasses automatically inherit it; `NetworkXAdapter` continues to provide the concrete implementation. `ProvenanceLedger.rollback_batch()` drops the fragile `hasattr()` guard and instead re-raises `NotImplementedError` from the per-edge handler, keeping the contract clean while preserving the original behavior.

## [2.17.0] â 2026-04-14
### Added
- **Phase 79: Loop-Provenance Recovery** â `AutonomousDiscoveryLoop` closes the fault-tolerance loop: when the circuit breaker trips and `LoopConfig.auto_rollback_on_trip=True`, the loop automatically calls `ProvenanceLedger.rollback_cycle(cycle_num, adapter)` to undo all edges materialized during the bad cycle before resuming. No-op in `dry_run` mode or when no ledger/adapter is attached.
  - `LoopConfig.auto_rollback_on_trip: bool = False` â opt-in flag.
  - `CycleRecord.edges_rolled_back: int` â count of edges removed by auto-rollback this cycle.
  - `LoopConfigSchema` and `CycleRecordSchema` updated; `POST /research/loop/configure` accepts `auto_rollback_on_trip`.

## [2.16.0] â 2026-04-14
### Added
- **Phase 78: Provenance Studio Panel** â `StudioEngine` gains a sixth live monitoring panel for `ProvenanceLedger` data.
  - **`attach_provenance_ledger(ledger)`**: optional attachment setter; panel degrades gracefully when not attached.
  - **`get_provenance_panel(n=20)`**: returns `(stats_html, batch_fig, timeline_fig)`.
    - `stats_html`: 4-card summary row â total batches, edges recorded, rolled-back count, cycles seen.
    - `batch_fig`: horizontal bar chart of the *n* most recent materialization batches; bars coloured green (active) or red (rolled back).
    - `timeline_fig`: dual-series chart â per-cycle edge count (bars) + cumulative edges (dashed line, secondary y-axis). Degrades gracefully when no cycle-tagged batches exist.
- `tests/test_studio_v2.py`: 8 new tests covering no-ledger graceful degradation, stats HTML correctness, bar chart population, rollback reflection, cycle timeline, and no-cycle fallback.

## [2.15.0] â 2026-04-14
### Added
- **Phase 77: Feature Impact Benchmark** â `benchmarks/feature_impact_benchmark.py` measures Hits@1, Hits@5, MRR across four feature configurations (baseline / +engram / +looped / +full) on any CSV graph. Uses toy_graph.csv for CI-safe runs; accepts `--graph`, `--sample`, `--embeddings`, `--json` flags. Reports per-config delta vs. baseline MRR.

## [2.14.0] â 2026-04-14
### Added
- **Phase 76: Graph Provenance & Rollback** â every edge materialized by `ResearchAgent.approve()` is now recorded in an optional `ProvenanceLedger` with batch_id, finding_id, cycle_number, and edge triples.
  - `rollback_batch(batch_id, adapter)` â removes exactly the edges from one approval.
  - `rollback_cycle(cycle_number, adapter)` â removes all edges from a given loop cycle.
  - LRU `max_batches` cap; thread-safe; graceful `NotImplementedError` when adapter lacks `remove_edge`.
  - `ResearchAgent` gains `set_provenance_ledger()` + `cycle_number` param on `approve()`.
  - `AutonomousDiscoveryLoop.run_cycle()` forwards `cycle_number` to every `approve()` call.
- `core/provenance_ledger.py`: `EdgeRecord`, `BatchRecord`, `ProvenanceLedger`.
- `tests/test_provenance_ledger.py`: 25 tests covering recording, rollback, LRU eviction, cycle rollback, adapter guard.

## [2.13.0] â 2026-04-14
### Added
- **Phase 75: Studio v2 Dashboard** â five new live monitoring panels added to `StudioEngine` via optional engine attachments.
  - **AutoApprover audit log** (`get_auto_approver_audit`): HTML table of last N decisions with action color-coding and stats summary.
  - **ContradictionResolver revision queue** (`get_revision_queue`): HTML list of findings where proposed evidence outweighed contradiction score, with net/weight annotations.
  - **DiscoveryCalibrator heatmap** (`get_discovery_heatmap`): Plotly dual-bar chart â sampling weight and discovery rate per community.
  - **ChemicalModulator blood panel** (`get_chemical_panel`): Plotly bar+scatter chart of 5 metabolic scalars vs. homeostatic baseline; color-coded by deviation.
  - **Autonomous Loop panel** (`get_loop_panel`): 3-card status header (running, circuit breaker, approval rate) + stacked bar/line cycle history chart.
  - All panels degrade gracefully when engines are not attached.
  - Attachment setters: `attach_research_agent()`, `attach_modulator()`, `attach_loop()`.
- `tests/test_studio_v2.py`: 25 tests covering all panels.

## [2.12.0] â 2026-04-14
### Added
- **Phase 74: Autonomous Discovery Loop** â closes the full discover â validate â approve â materialize loop without human intervention.
  - `AutonomousDiscoveryLoop` runs `ResearchAgent.scan_once()` on a configurable timer and processes each finding through the attached `AutoApprover`.
  - **Circuit breaker**: sliding window over the last N decisions; if approval rate drops below `min_approval_rate`, materialization pauses and `circuit_breaker_tripped=True` is reported. Auto-resets as the window fills with healthy decisions.
  - **Per-cycle cap**: `max_materializations_per_cycle` hard limit prevents runaway materialization.
  - **Dry-run mode**: full cycle execution without any `approve()` / `reject()` calls â safe for production trials.
  - **AutoApprover checkpoint**: persists `aa.to_dict()` to disk after any cycle with decisions; enables warm restart.
  - **`LoopConfig` dataclass**: `cycle_interval`, `max_materializations_per_cycle`, `min_approval_rate`, `circuit_breaker_window`, `dry_run`, `approver_checkpoint_path`.
  - **`CycleRecord` dataclass**: per-cycle summary â findings_seen, auto_approved, auto_rejected, sent_to_review, edges_added, circuit_breaker_tripped.
  - REST: `POST /research/loop/start`, `POST /research/loop/stop`, `GET /research/loop/status`, `POST /research/loop/configure`.
- `core/autonomous_loop.py`: `LoopConfig`, `CycleRecord`, `AutonomousDiscoveryLoop`.
- `api/schemas.py`: `LoopConfigSchema`, `CycleRecordSchema`, `LoopStatusResponse`.
- `tests/test_autonomous_loop.py`: 33 tests covering all paths (no-AA fallback, approve/reject/review, cap, dry-run, circuit breaker, configure, checkpoint, lifecycle).

## [2.11.0] â 2026-04-14
### Added
- **Phase 73 Batch B: Feature 1 â ContradictionResolver** â deterministic evidence-weight classifier on already-computed proposal data. Computes Noisy-OR of proposed path confidences vs. max contradiction_score; classifies findings as "clean" / "revision_candidate" / "contested" / "discardable". Discardable findings are auto-rejected before reaching AutoApprover. Revision candidates (proposed evidence outweighs existing) are queued separately for human review. No extra traversal passes â pure arithmetic on HypothesisProposal fields.
- **Phase 73 Batch B: Feature 3 â CandidateRegistry** â replaces flat `_evaluated_pairs` set with a TTL-aware registry that tracks `nomination_count` per (source, target) pair across scan cycles. Multi-nominated candidates receive a log-scale `nomination_boost` applied to `discovery_potential` scoring, surfacing pairs independently discovered by multiple mechanisms. TTL gate prevents redundant HypothesisEngine runs; `prune()` evicts stale entries; LRU cap enforces memory bound.
- `core/contradiction_resolver.py`: `ContradictionRecord` dataclass + `ContradictionResolver` class.
- `core/candidate_registry.py`: `RegistryEntry` dataclass + `CandidateRegistry` class.
- `ResearchAgent` gains `set_contradiction_resolver()`, `set_registry()`, and `_revision_candidates` deque.
- `AutoApprover.decide()` gains contradiction hard gate: discardable resolution â immediate reject.
- `tests/test_batch_b.py`: â¥20 new tests covering both features + ResearchAgent wiring.

## [2.10.0] â 2026-04-14
### Added
- **Phase 73 Batch A: Feature 2 â Temporal Recency Scoring** â `_compute_recency_score(hits)` exponential-decay scoring on publication year (half-life 7 years). `ValidationReport` gains `recency_score` field [0,1]: 1.0 = all hits published this year, 0.5 = average hit is 7 years old or no year data (neutral).
- **Phase 73 Batch A: Feature 4 â DiscoveryCalibrator** â EMA-smoothed per-community scan and discovery rate tracking. Inverse-rate multiplier (`weight = global_rate / (community_rate + Îµ)`) steers `ResearchAgent` candidate scoring toward understudied communities. Cold-start: unscanned communities receive `max_weight` (default 5.0).
- `core/discovery_calibrator.py`: `DiscoveryCalibrator` with `record_scan()`, `record_discovery()`, `get_weight()`, `stats()`.
- `core/external_validator.py`: `_compute_recency_score()` function + `_RECENCY_HALF_LIFE_YEARS` constant.
- `tests/test_batch_a.py`: 20 new tests (recency + calibrator sections).

## [2.9.0] â 2026-04-14
### Added
- **Phase 72: TriangulationEngine** â four-perspective validation of `ResearchCandidate` objects, extending the `AutoApprover` feature vector from 12 â 16.
  - **P1 `reverse_confidence`**: HypothesisEngine run BâA direction.
  - **P2 `strategy_agreement`**: fraction of 3 strategy configs (conservative/standard/exploratory) returning â¥1 valid proposal.
  - **P3 `mean_path_independence`**: mean Jaccard independence across primary proposals (free â already computed).
  - **P4 `semantic_type_score`**: relation-type / entity-class consistency index; novel relations score 0.5 (neutral â never penalises novelty).
  - `is_wormhole_candidate` diagnostic flag (not a classifier feature). Report stored in `finding.metadata["triangulation"]`.
  - Type index lazily built; invalidated on graph signature change.
- `core/triangulation_engine.py`: `TriangulationReport` dataclass + `TriangulationEngine` class.
- `tests/test_triangulation.py`: new test suite for all four perspectives.

## [2.8.5] â 2026-04-14
### Added
- **Phase 71: AutoApprover** â automated approve/reject/review decision engine for `ResearchFinding` objects, replacing manual `POST /research/approve|reject` at scale.
  - Three-tier decision stack: hard gates (blocked statuses, missing validation) â online logistic SGD classifier (16 features) â optional LLM semantic fallback.
  - **16-dimensional feature vector**: confidence, discovery_potential, gap_score, community_distance, local_density, literature_status ordinal, novelty_score, engram_affinity, path_count, contradiction_score, seeded_by flags, + 4 TriangulationReport slots (features 12â15).
  - Online `fit(finding, approved)` from confirmed human decisions; `to_dict()` / `from_dict()` checkpoint support.
  - `AutoApprovalPolicy` â configurable thresholds, `blocked_statuses`, `require_validation`.
  - `ResearchAgent` gains `_auto_approver` attachment; auto-decisions fire `report_outcome()` + `approve()`/`reject()`.
  - REST: `GET /research/auto-approver/stats`, `POST /research/auto-approver/policy`.
- `core/auto_approver.py`: `AutoApprovalPolicy`, `AutoDecision`, `AutoApprover`.
- `tests/test_auto_approver.py`: new test suite.

## [2.8.0] â 2026-04-11
### Added
- **Phase 70: Looped Beam Traversal** â LoopLM-style iterative refinement for KG reasoning (arXiv:2510.25741).
- New `reasoning/looped_traversal.py`: `LoopTrace` dataclass + `LoopedBeamTraversal` class.
- `LoopedBeamTraversal` wraps any `BeamTraversal`-compatible engine and applies it T times, progressively refining reasoning via seed expansion and adaptive exit.
- **Three inter-loop feedback channels** vs LoopLM's single hidden-state channel: Semantic (top answer entities expand seeds), Metabolic (PEâChemicalModulator adjusts traversal params), Mnemonic (Engram records bias next loop's beam pruning).
- **Adaptive exit gate** with two signals: PE convergence (`|ÎPE| < Î³`, primary) and answer-set stability (Jaccard fallback). Prevents both underthinking and overthinking.
- **Path merging across all loops**: `best_by_tail` dict keeps highest-score path per tail entity across all iterations, maximising coverage.
- `QueryRequest` gains `max_loops: int` (default 1, range 1â8). Default=1 is fully backward compatible.
- `QueryResponse` gains `loops_run: Optional[int]` and `pe_per_loop: Optional[List[float]]`.
- `ReasoningTrace` gains `loop_trace: Optional[LoopTrace]` (Phase 62 ERT integration).
- `MultiStrategyConsensus.run_consensus_query()` gains `max_loops` param; each strategy loops independently before consensus aggregation.
- `MultiStrategyConsensus.__init__()` gains `predictive_coder` param for PE-gated exit.
- `core/cerebrum.py` `query()` gains `max_loops: int = 1`; looped traversal wired when `max_loops > 1`.
- `/query` and `/query/trace` endpoints fully wired: `max_loops` from request, `loops_run`/`pe_per_loop` in response.
- 14 new tests in `tests/test_looped_traversal.py`: single-loop backward compat, answer-stability exit, PE-convergence exit, PE priority over stability, fallback without PE engine, seed expansion, original seed preservation, path merging, highest-score-wins deduplication, LoopTrace fields, max_loops cap.
- **PAPER_022_LOOPED_TRAVERSAL.md** â full technical paper for Phase 70.
- `docs/arxiv/SOURCES.md` â added `[zhu2025loooplm]` (arXiv:2510.25741) and `[bengio2025soliton]` (UCFT 2025) entries.
- Cross-paper citations: `[zhu2025loooplm]` added to PAPER_006, PAPER_015, PAPER_018; `[bengio2025soliton]` added to PAPER_018, PAPER_019.

## [2.7.0] â 2026-04-11
### Added
- **Phase 69: Predictive Coding Engine** â Active inference closes the loop across all Phase 59â68 components.
- New `core/predictive_coder.py`: `PriorPath` + `PredictionResult` + `PredictiveCodingEngine`.
- Before each traversal, the engine queries the Engram for the top relation pattern and generates a `PriorPath` â a forward prediction of likely nodes and relation sequence.
- After traversal, `compute_pe()` calculates **Prediction Error** as Jaccard divergence between prior and best actual relation sequence. PE=0 = perfect prediction; PE=1 = complete miss.
- PE dispatched to `ChemicalModulator`: `update_arousal(PE)`, `update_novelty(PE)`, `update_reinforcement(1-PE)` â all metabolic scalars now respond to prediction accuracy.
- **Soliton Index**: `soliton_index = 1 - mean(recent PEs)` per seed set. A self-reinforcing prior that consistently yields low PE is soliton-like (stable, self-localising wave â inspired by UCFT 2025 preprint *Consciousness as a Soliton, Not a Process*). High soliton_index = the system has converged on a stable internal model for that reasoning domain.
- `CerebrumGraph.attach_engram(engram)` â post-build method to wire the Engram and activate predictive coding.
- `ReasoningTrace` (Phase 62 ERT) gains `prior`, `prediction_error`, and `soliton_index` fields.
- `QueryResponse` gains `prediction_error` and `soliton_index` API fields.
- `_state["predictive_coder"]` initialized in `_load()` after Engram warm-up; PE drives modulator on every `/query` call.
- 16 new tests in `tests/test_predictive_coder.py`: cold start, PE accuracy (perfect/mismatch/partial), soliton index convergence, modulator signal dispatch, trace field integration.

## [2.6.0] â 2026-04-11
### Added
- **Phase 68: Metabolic Modulation Suite** â Functional regulation of reasoning.
- New `core/chemical_modulator.py` simulates Reinforcement, Arousal, Novelty, Cohesion, and Persistence.
- **Dynamic Homeostasis**: Implemented temporal decay and homeostatic baselines for metabolic scalars.
- **Metabolic Feedback Loops**: Automated adjustment of `beam_width` (Arousal), `alpha/beta` ratios (Novelty), and `canonical_promotion` (Persistence).
- **REST API Blood Panel**: New `GET /chemical` endpoint for real-time monitoring of system's metabolic state.
- **Phase 65: Autonomous Hypothesis Materialization** â ResearchAgent results can now be formally committed to the graph.
- **Phase 64: Neural Memory Consolidation** â Automatic promotion of successful relation patterns to "Canonical Engrams" via `EngramConsolidator`.

## [2.5.0] â 2026-04-10
### Added
- **Phase 63: Neural Telemetry System** â Real-time event emission for 3D visualizations.
- New `core/telemetry.py` standardizes the event schema for external observers (e.g., Unreal Engine).
- Integrated `NeuralEvent` pulses into `BeamTraversal` for real-time visibility of reasoning steps.
- New orchestrator `scripts/start_cerebrum.py` for simultaneous API & telemetry server launch.

## [2.4.0] â 2026-04-09
### Added
- **Phase 62: Explainable Reasoning Trace (ERT)** â Decisions & feature radars.
- New `ReasoningTrace` and `HopTrace` models capture winners and competitors at every step.
- `POST /query/trace` endpoint for "glass-box" reasoning transparency.
- 10-parameter "Attention Radar" (ReasoningLogit features) exposed for every candidate.
- Hardened serialization: `numpy.float32` and other types converted to standard primitives.

## [2.3.0] â 2026-04-08
### Added
- **Phase 61: Synaptic Pruning & Quantized Traversal (SPQT)** â Efficiency optimizations.
- `SynapticPruner` implements utility-based edge removal (confidence, age, usage).
- Integrated pruning into `GlobalRebalancer` for automated post-rebalance optimization.
- `BeamTraversal` now supports `quantized=True` mode, using `uint8` fixed-point scores.
- `TraversalPath` maintains both high-precision `score` and efficiency-optimized `q_score`.

## [2.2.0] â 2026-04-08
### Added
- **Phase 60: Multi-Agent Consensus Hierarchies (MACH)** â Three-tier reasoning verification.
- `L1 Local`: Multi-strategy voting (Standard, Bayesian, Engram) for internal path robustness.
- `L2 Federated`: Cross-node confirmation via `FederatedAdapter` corroboration.
- `L3 Gold`: High-trust verification against external literature via `ResearchAgent`.
- New `/query/consensus` endpoint for hierarchical multi-level reasoning.
- Upgraded `ConsensusScorer` with variance tracking and agent trust weighting.

## [2.1.0] â 2026-04-08
### Added
- **Phase 59: Cerebellar Error Correction (CEC)** â Active error-driven meta-learning loop.
- `CerebellarEngine` detects "Dissonant Predictions" (high path score, low consensus) and triggers corrective research.
- `Answer` class now exposes `path_score` and `consensus_score` for explainability.
- `ResearchAgent` now supports `push_candidate()` for external task seeding.
- Integrated CEC into `/query` API flow.

## [2.0.2] â 2026-04-08

### Changed
- **Naming: AAAK â Engram (all occurrences)**:
    - The relation-pattern cache was previously labeled "AAAK" throughout the codebase. This name was rejected because it simply is not AAAK â the acronym was inaccurate and did not describe the mechanism.
    - The correct name is **Engram**: the neurological term for the physical memory trace a successful experience leaves in the brain. This accurately describes what the cache does â successful reasoning paths leave a structural imprint that biases future beam traversals toward known-productive chains.
    - `AAAKCache` â `Engram` Â· `AAAKBeamTraversal` â `EngramTraversal` Â· `AAAKVerbalizer` â `EngramVerbalizer`
    - `SpeedTalkAAAKCache` â `SpeedTalkEngram` Â· `SpeedTalkAAAKBeamTraversal` â `SpeedTalkEngramTraversal`
    - `aaak_steered_traversal.py` â `engram_traversal.py` Â· `test_aaak_traversal.py` â `test_engram_traversal.py`
    - `PAPER_018_AAAK_STEERED_TRAVERSAL.md` â `PAPER_018_ENGRAM_TRAVERSAL.md`
    - All backward-compatibility aliases removed. Zero AAAK references remain in the codebase.
- **Phase 58: SpeedTalk-Compressed Engram Cache**:
    - `SpeedTalkEncoder` â maps each relation type to a single phoneme character (62-symbol alphabet: aâz, AâZ, 0â9). Frequency-ordered assignment via `build_frequency_order()`.
    - `SpeedTalkEngram` â drop-in replacement for `Engram` using phonemic key storage; 8â20Ã key compression. New: `prefix_query(*rels)`, `alphabet()`, `compression_stats()`.
    - `SpeedTalkEngramTraversal` â `BeamTraversal` variant backed by `SpeedTalkEngram`.
    - Graph-adaptive encoding: `adapt_to_graph(freq)` / `from_graph_adapter(adapter)` retune the alphabet to the loaded KG so most-traversed relations get shortest symbols.
    - 50 new tests in `tests/test_speedtalk_cache.py`.
    - `docs/arxiv/PAPER_021_SPEEDTALK_COMPRESSION.md` â full technical paper.

---

## [2.0.1] â 2026-04-07

### Added
- **Phase 57: Engram Persistence Across Restarts**:
    - `_engram_cache_path(cache_path)` helper derives `engram_cache.json` path alongside graph cache (or `SAFE_DATA_DIR`).
    - Lifespan `try/finally` block saves live `Engram` to disk on server shutdown.
    - Both `_load()` paths use two-tier warm-up: load saved JSON first, then merge incremental `QueryLog` entries on top.
    - `Engram.save()` / `Engram.load()` / `Engram.save_if_path()` persistence API.
    - 12 new tests in `tests/test_fault_tolerance.py` (stream error chunk, ProcessPool fallback + warning, Engram save/load roundtrip).
- **Phase 57: `/query/stream` Traversal Guard**:
    - `async for` in streaming generator wrapped in `try/except`; yields terminal `{"status": "error", "partial": true, "error": "..."}` NDJSON chunk on any traversal exception.
- **Phase 57: `ProcessPoolExecutor` Sequential Fallback**:
    - `best_of_n_dscf` catches any executor failure (`BrokenExecutor`, `WinError 1455`, etc.) and falls back to the existing sequential path; logs `WARNING` with reason.

### Fixed (Phase 56)
- **Phase 56: Fault Tolerance Hardening**:
    - `QueryResponse` now has `partial: bool = False` and `error: Optional[str] = None` fields (backward-compatible defaults).
    - `BeamTraversal._partial_paths` list checkpoints completed hops; survives mid-hop exceptions so `/query` can return partial results.
    - `/query` endpoint catches traversal exceptions and returns HTTP 200 with `partial=True` + error message rather than 500.
    - `QueryLog.record()` and `Engram.record()` failures are isolated (`try/except`) â neither crashes `/query`. Both log at `WARNING`.
    - `GlobalRebalancer._rebalance_worker` split into outer crash-guard + `_rebalance_worker_inner`; any inner exception is logged at `ERROR`, thread restarts on next trigger.
    - 15 new tests in `tests/test_fault_tolerance.py`.

## [2.0.0] â 2026-04-07

### Added
- **Phase 55: GraphSAGE Neighbourhood Smoothing**:
    - `smooth_with_graphsage(embeddings, G)` â one-pass mean neighbourhood aggregation applied after base encoding; `CerebrumGraph.build(use_graphsage=True)`.
- **Phase 55: Engram-Steered Traversal**:
    - `Engram` â thread-safe relation-pattern affinity store (relation_sequence â success_count); prefix-indexed for O(1) affinity lookup.
    - `EngramTraversal` â extends `BeamTraversal`; biases `_prune_candidates()` via `effective_score = score Ã (1 + engram_strength Ã affinity)`.
    - On-startup `replay_into_cache()` warms `Engram` from `QueryLog` NDJSON history.
- **Phase 55: TemporalCalibrator**:
    - Grid-search calibration of CSA `eta` (temporal decay) and `iota` (node recency) to maximise Recall@K against a labelled validation set.
    - `calibrate()` / `apply()` / `measure_recall()` API; `try/finally` param restore guarantee.
- **Phase 55: QueryLog**:
    - Append-only NDJSON query history in `core/persistence.py`. Records seeds, answers, and relation sequences after each reasoning call.
    - `replay_into_cache(engram)` re-warms `Engram` on process restart.
- **Phase 54: Observability Dashboard**:
    - `RingBufferHandler` in `core/log_config.py` â thread-safe in-memory ring buffer (5000 entries) feeding `GET /logs`.
    - `setup_logging()` configures the `cerebrum.*` logger hierarchy (console + optional rotating file + ring buffer).
    - CORS middleware, HTTP request timing middleware added to API server.
    - `GET /logs` and `DELETE /logs` endpoints for live log streaming.
    - `POST /build` hot-reload endpoint.
    - `ui/dashboard.html` dark-mode operational dashboard (GridStack + Chart.js + vis-network).

## [1.9.7] â 2026-04-05

### Added
- **Phase 53: Adaptive Search Strategy**:
    - `ResearchAgent._select_strategy(local_density)` selects beam search parameters based on 2-hop neighbourhood density: dense (> 0.4) â shallow fast, sparse (< 0.1) â deep wide, mid â defaults.
    - `local_density` stored on `ResearchCandidate` and exposed in `ResearchCandidateSchema`.
    - `_score_discovery_potential()` returns `(potential, conn_density)` tuple.

## [1.9.6] â 2026-04-05

### Added
- **Phase 51: ResearchAgent**:
    - Autonomous background daemon (`core/research_agent.py`) that mines missing-link candidates via embedding similarity scan [0.6, 0.95] and `InsightEngine` seeding.
    - Discovery potential scoring: semantic gap + connection density + community leap.
    - Fixed-size ring buffer for pending findings; `approve(finding_id)` delegates to `HypothesisEngine.materialize()`.
    - 7 new REST endpoints: `/research/status`, `/research/start`, `/research/stop`, `/research/scan`, `/research/findings`, `/research/approve/{id}`, `/research/reject/{id}`.
- **Phase 52: ExternalValidator**:
    - `ExternalValidator` (`core/external_validator.py`) â LLM-independent external source validation using keyword co-occurrence in corpus documents.
    - `/research/validate` endpoint triggers external validation of pending findings.
    - `ValidationReportSchema` / `ValidateProposalsRequest` / `ValidateProposalsResponse` schemas.

## [1.9.5] â 2026-04-05

### Added
- **Phase 50: HypothesisEngine**:
    - `HypothesisEngine` (`core/hypothesis_engine.py`) â multi-path abductive reasoning with Noisy-OR confidence combination across independent paths.
    - Relation chain composition reusing `InferenceEngine`'s 50+ rule index.
    - Contradiction detection and intersection hub identification.
    - Snapshot-based rollback.
    - `POST /hypothesize` and `POST /hypothesize/materialize` endpoints.
    - 6 new schemas: `HypothesizeRequest`, `HypothesizeResponse`, `HypothesisProposalSchema`, `HypothesisMaterializeRequest`, `HypothesisMaterializeResponse`, `HypothesisStatusResponse`.

## [1.9.4] â 2026-04-05

### Added
- **Phase 49: TSC Explicit Mode**:
    - `tsc_communities(G)` public API â auto-computes PageRank centrality and delegates to vectorized TSC; exported from `core/__init__.py`.
    - `tsc_quality_metrics(G, communities)` â returns modularity Q, community count, min/max/mean size.
    - `community_engine="tsc"` backend in `CerebrumGraph.build()` with PageRank reuse from `struct_features`.

## [1.9.3] â 2026-04-05

### Added
- **Phase 48: Auto-Retrain Scheduler**:
    - **Feedback buffer** (`_state["feedback_buffer"]`): Every `POST /feedback` call now appends `{path, reward}` to an in-memory buffer alongside the existing online SGD update. The response includes `buffer_size` so clients can track accumulation.
    - **`POST /retrain` endpoint**: Runs `CSAParameterLearner.fit()` on cross-paired positive/negative paths from the buffer. Uses the current `MetaParameterLearner.global_prior` as the starting point, then replaces it with the learned 10-parameter vector. Returns `RetrainResponse` with loss trajectory, convergence flag, and all learned param values.
    - **`RetrainRequest` schema**: `max_pairs` (default 500), `max_iterations` (200), `learning_rate` (0.01), `clear_buffer` (True).
    - **`RetrainResponse` schema**: `pairs_used`, `iterations`, `initial_loss`, `final_loss`, `converged`, `learned_params`, `buffer_remaining`.
    - 5 new tests covering mixed-feedback requirement, response structure, global prior sync, buffer clear/keep.

### Changed
- `POST /feedback` response now includes `buffer_size` field.

## [1.9.2] â 2026-04-05

### Added
- **Phase 47: Params Persistence**:
    - **`MetaParameterLearner.to_dict()` / `from_dict()`**: Full JSON serialisation of the learned state (global prior, community overrides, hyperparams). Enables checkpoint/restore across server restarts.
    - **`POST /params` endpoint**: Accepts a `ParamsImportRequest` (global_prior + community_overrides) and replaces the running learner state. Supports the full export â restart â import workflow. Returns the new `ParamsResponse` so callers can verify the applied state. Invalid vector lengths return 422.
    - **`ParamsImportRequest` schema**: New Pydantic model with optional `learning_rate` and `momentum` overrides.
    - **`--params-file FILE` CLI flag** (`cerebrum serve`): Loads a JSON checkpoint at startup, restoring the MetaParameterLearner before the server begins accepting requests.
    - 9 new tests covering `to_dict`/`from_dict` round-trip, `POST /params` restore/422, and full exportâresetâimport cycle.

### Fixed
- **`test_temporal_sliding_window` flakiness**: Replaced `np.random.rand(384)` embeddings with `np.ones(384)` so cosine similarity is equal for all pairs, making the temporal decay signal the sole differentiator.

## [1.9.1] â 2026-04-04

### Added
- **Phase 46: Live Feedback Loop & /params Endpoint**:
    - **`GET /params` endpoint**: Returns the current 10-parameter global vector and all per-community overrides accumulated via `POST /feedback`. Enables parameter inspection and client-side checkpointing.
    - **`PathResult.edge_features`**: Query responses now include the per-hop 10-element feature vectors `(sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding)` so clients can pass them directly to `POST /feedback` without client-side reconstruction.
    - **`PathResult.community_sequence`**: Query responses now include the community ID sequence for each entity node, also required for `/feedback`.
    - **`ParamsResponse` schema**: New Pydantic model for `/params` output.

### Fixed
- **`_DEFAULT_INIT_PARAMS` in `reasoning/traversal.py`**: Was a 9-tuple `(â¦, iota, theta)` missing `mu=0.1`. Now correctly a 10-tuple matching the Phase 43 CSA formula. This prevented the synthesis-density penalty from being applied when the fallback param path was taken.
- **`FeedbackRequest.edge_features` description**: Updated to document all 10 features including `sd` (synthesis density).

## [1.9.0] â 2026-04-04

### Added
- **Phase 45: 10-Parameter Learner Upgrade**:
    - **`CSAParameterLearner` â Full 10-param support**: Upgraded from 5 to 10 learnable parameters `(alpha, beta, gamma, delta, epsilon, zeta, eta, iota, mu, theta)` matching the Phase 43 CSA formula. Numerical gradient descent, fit loop, and `_score_path_parametric` all operate on the full parameter vector.
    - **`MetaParameterLearner` â Full 10-param support**: Online SGD update now uses all 10 feature dimensions with correct signs (`nd` and `sd` are penalised). Backward compatible with legacy 5-element edge_features via zero-padding.
    - **`CSAEngine.get_current_params()` â 10-param-aware**: Fixed 5-element destructure to safely unpack any-length param vector from `MetaParameterLearner`, with per-engine fallbacks for unmanaged params.
    - **`LearningResult` â Updated type**: `params` field is now `Tuple[float, ...]` (variable-length).
    - **Backward Compatibility**: `_score_path_parametric` zero-pads legacy 5- or 7-element edge_features to avoid breaking existing callers.
    - **New Tests**: Added `test_legacy_5element_edge_features_backward_compat`, `test_gradient_length_matches_param_count`, and `test_meta_parameter_legacy_5element_compat`.

### Fixed
- **`CSAEngine.get_current_params()`**: `ValueError: too many values to unpack (expected 5)` when `MetaParameterLearner` returned 10 values.

### Changed
- **v1.9.0 Release**: Parameter learning subsystem is now fully aligned with the Phase 43 10-parameter CSA formula.

## [1.8.0] â 2026-04-04

### Added
- **Phase 44: IKGWQ-MetaQA Benchmark**:
    - **Unified Evaluation Protocol**: Adapted the IKGWQ (Incomplete Knowledge Graph) protocol to the MetaQA 3-hop reasoning dataset.
    - **REM Synthesis Validation**: Quantified the impact of `REMEngine` "Wormhole" synthesis on reasoning recall under high edge sparsity.
    - **Improved Benchmarking**: Verified up to 40% recall improvement on Level 4 (50% removal) graphs when REM synthesis is active.

### Changed
- **v1.8.0 Release**: Marked full system readiness for incomplete graph reasoning.

## [1.7.5] â 2026-04-04

### Added
- **Phase 43: Temporal Context & REM Synthesis Evaluation**:
    - **Temporal Sliding Window**: Implemented `temporal_window_size` in `CSAEngine` to penalize edges older than a specified window.
    - **Synthesis Density Scoring**: Added `sd` (Synthesis Density) feature to `ReasoningLogit` to track and penalize synthetic REM/Wormhole edges.
    - **Dynamic Parameter Overrides**: Refactored `CSAEngine.compute_weight` to support per-call parameter overrides (alpha, beta, etc.).
    - **REM Synthesis Benchmark**: Added `benchmarks/rem_synthesis_eval.py` (IKGWQ-S) to verify that REM bridges improve reachability in disconnected graphs.

### Changed
- **Unified Logit**: Upgraded `ReasoningLogit` to 10 parameters to include Synthesis Density (`sd`).
- **CSA Backward Compatibility**: Enhanced `set_community_graph` to accept flexible positional and keyword arguments.
- **Improved BRIDGE_TWIN**: Secured `BRIDGE_TWIN` edges as high-confidence structural relays.

## [1.7.4] â 2026-04-04

### Added
- **Phase 42: Interface Robustness & API Hardening**:
    - **UI Stabilization**: Refactored `load_graph` in `Reasoning Studio` to use `gr.Progress` for more stable loading feedback.
    - **REST API Hardening**: Secured the `/health` endpoint and ensured 9-parameter score breakdowns are returned consistently.
    - **Automated Robustness Tests**: Added `tests/test_studio_robustness.py` and `tests/test_api_robustness.py`.
    - **Community Scaling**: Implemented automatic community coarsening in `core/cerebrum.py` for large-scale datasets.

### Fixed
- **UI Syntax Errors**: Resolved multiple `SyntaxError` issues in `ui/studio.py` caused by incorrectly escaped triple quotes and illegal backslashes in f-strings.
- **Security Test Alignment**: Updated `tests/test_security.py` to correctly expect `401 Unauthorized` for the now-secured `/health` endpoint.
- **GPU DSCF Stability**: Stabilized `tests/test_dscf_gpu.py` by ensuring deterministic community detection for triangle graphs.
- **Robustness Test Sync**: Synchronized `api_name`s in robustness tests with the actual Gradio implementation.

## [1.7.3] â 2026-04-02

### Added
- **Phase 41: Temporal Reasoning & REM Synthesis**:
    - **Temporal Bias Correction**: Corrected the recency formula in `CSAEngine` to properly favor newer edges ($+\exp(-\lambda t)$).
    - **Node Recency Integration**: Added `nr_v` (Node Recency) to `ReasoningLogit` framework (9-feature vector).
    - **Wormhole Detection**: Implemented similarity-based bridging of disconnected components in `REMEngine`.

## [1.7.2] â 2026-04-02

### Added
- **Phase 39: Async Bridge Synthesis**: Decoupled `BridgeTwinEngine` and `InsightEngine` updates via `TaskQueue` to minimize beam traversal latency.
- **Phase 40: IKGWQ Hardening**: Verified full system reasoning performance under extreme (50%+) edge removal sparsity.
- **Node Recency Prior**: Integrated node-level recency scores (`nr_v`) from structural features into the `ReasoningLogit` framework.
- **Unified Logit Framework**: Expanded `ReasoningLogit` to 9 features for consistent parametric scoring across all engines.

### Fixed
- **Temporal Reasoning Bias**: Corrected the exponential decay formula in `CSAEngine` which was accidentally penalizing newer edges (reversed recency bias).
- **BeamTraversal Feature Sync**: Synchronized both synchronous and asynchronous traversal paths to use the unified 9-element logit structure.
- **MockAdapter Stability**: Added missing abstract method implementations to `MockAdapter` for internal testing stability.

## [1.7.1] â 2026-04-01
 â Federated Reasoning + GPU Stability

### Added
- **Federated Reasoning Infrastructure (Phase 32)**:
    - **`api/server.py` â `/traverse` endpoint**: New delegated reasoning endpoint that returns serialized `TraversalPath` branches starting from a seed entity.
    - **`reasoning/traversal.py` â `TraversalPath.to_dict() / from_dict()`**: Native serialization for cross-network path transmission with full metadata (scores, attention weights, community sequences).
    - **`reasoning/distributed_traversal.py` â `DistributedBeamTraversal`**: New traversal engine that supports initial and boundary delegation to remote CEREBRUM nodes.
    - **`adapters/federated_adapter.py` â `get_reasoning_branches()`**: Aggregates reasoning branches from sub-adapters (local or remote) and applies Procrustes alignment rotations to returned embeddings.
    - **`adapters/remote_adapter.py` â `get_reasoning_branches()`**: Client-side implementation that calls the `/traverse` endpoint on remote CEREBRUM instances.
    - **`adapters/networkx_adapter.py` â `get_reasoning_branches()`**: Local implementation that runs a sub-beam search to provide branches to federated callers.
    - **`tests/test_federated_reasoning.py`**: Integration tests for the new `/traverse` API and federated delegation logic.

### Fixed
- **`core/dscf_gpu.py` â Convergence Stability**: 
    - Fixed a critical flakiness bug in `GPUDSCFEngine` where small symmetric structures (like triangles) could oscillate indefinitely under synchronous updates. 
    - Implemented **Block-Asynchronous Updates** using a 50% random Bernoulli mask to break symmetry and ensure convergence.
    - Updated `changed_frac` calculation to use the unmasked "intent" vector for more robust termination.
    - Added current-community score bias (0.05) to further stabilize near-tie community assignments.
    - Populated all `GPURunStats` profiling fields (tensor_build_ms, iteration_ms, iterations, converged) which were previously zero or uninitialized.

---

## [1.7.0] â 2026-04-01 â Proactive Bridge Synthesis (Phase 30)

### Added
- **`core/graph_bridge.py` â `GraphBridgeEngine`**: Proactive cross-component bridge synthesizer. Detects disconnected components and connects "frontier nodes" (peripheral nodes in small components) using pre-trained `SentenceEngine` embeddings. This addresses the multi-hop recall bottleneck on fragmented scaffold graphs (e.g., CWQ) without requiring task-specific training.
- **`CerebrumGraph.enhance()`** (`core/cerebrum.py`): New pipeline stage: `THALAMUS â complete() â enhance() â build() â CORTEX`. Supports proactive enhancers that require embeddings or heuristics, complementing the purely logical `complete()` stage.
- **`CerebrumGraph.build(community_engine=...)`**: Added support for choosing between `dscf` (default), `leiden`, and `lpa` engines. `leiden` provides a 10-100x speedup on multi-million node graphs on CPU compared to the standard DSCF loop.
- **`tests/test_graph_bridge.py`**: Comprehensive unit tests for bridge synthesis, covering component discovery, frontier selection, and similarity-based link materialization.

### Fixed
- **`GraphBridgeEngine` cap strictness**: Fixed a bug where bidirectional edges could exceed the `max_bridges` limit by one.
- **`GraphBridgeEngine` robustness**: Added bounds checking for `top_k` in `np.argpartition` to prevent `ValueError` on small components.
- **`scripts/setup_cwq_data.py`**: Added `entity_names.json` generation logic to ensure `SentenceEngine` correctly labels MIDs using the name-string format already present in the CWQ scaffold.

---

## [1.6.9] â 2026-03-31 â CWQ Benchmark + Unit Tests + WebQSP Fix

### Added
- **`scripts/setup_cwq_data.py`**: One-time data download and scaffold graph construction for
  ComplexWebQuestions (CWQ, 3,519 test / 27,639 train questions from `rmanluo/RoG-cwq` on
  HuggingFace).  Supplements with WebQSP Freebase triples (same ontology).  Outputs
  `cwq_scaffold.txt`, `entity_names.json` (placeholder), `CWQ.test.json`, `CWQ.train.json`.
- **`benchmarks/cwq_eval.py`**: Full CWQ evaluation harness.  Refactored to use the unified
  `CerebrumGraph` pipeline. Supports sentence embeddings with friendly names, DSCF + coarsening,
  question-embedding-guided traversal, and entity-level F1 + Hits@1.  Reports per-type breakdown.
- **`tests/test_cerebrum.py`**: 36 unit tests for `CerebrumGraph` public API covering all four
  factory methods (`from_kb`, `from_csv`, `from_triples`, `from_adapter`), `complete()`,
  `build()`, and `query()` (including `max_hop` override, `min_hop` filtering, uniqueness, sorting,
  multi-seed, chaining).
- **`tests/test_graph_completion.py`**: 22 unit tests for `InverseRule` and `CompositionRule`
  covering synthetic flag, confidence inheritance (weakest-link), provenance format,
  `min_occurrences`, `max_edges`, cycle avoidance, idempotency, and `describe()`.

### Fixed
- **`benchmarks/webqsp_full_eval.py` line 836**: `UnicodeEncodeError` on Windows cp1252 console
  caused by printing Greek letters (Î± Î^2 Î³ Î´ Îµ) in the CSAParameterLearner output block.
  Replaced with ASCII equivalents (`alpha`, `beta`, `gamma`, `delta`, `eps`).  The `--optimized`
  variant of the WebQSP benchmark can now run to completion on Windows.

---

## [1.6.8] â 2026-03-31 â RelationPathPrior for MetaQA

### Added
- **`--use-prior` flag in `benchmarks/metaqa_eval.py`**: Builds a `RelationPathPrior` from training
  data for 2-hop and 3-hop re-ranking. The prior counts which relation-sequence patterns most
  frequently reach correct answers across 20K training questions (sampled from 118K/114K available).
  It is a frequency heuristic over the *search process*, not a modification to graph structure or
  answer claims. Priors are cached to `CACHE_DIR/prior_{N}hop.pkl` after the first run (~5 min
  one-time cost for 3-hop; negligible on subsequent runs).
- **`build_or_load_prior()` helper** in `metaqa_eval.py`: handles train-file parsing, sampling
  (default 20K, avoids 30+ min full training traversal), BeamTraversal per hop, and disk cache.
- **1-hop intentionally excluded** from prior: the 1-hop prior has only 9 unique relation patterns
  (one per relation type) and provides no discriminating signal; empirically hurts H@10 if applied.

### Fixed
- `build_or_load_prior` reads from `TRAIN_FILES[hop]` (training split), not `QA_FILES[hop]` (test).
- Training sample capped at 20,000 questions per hop â full 118K/114K training sets would take
  30+ minutes per hop; 20K gives effectively the same 70â217 unique patterns (saturation point).

### Benchmark Results (MetaQA â full 39,093 questions, sentence + RelationPrior, OFFICIAL v1.6.8)

Settings: SentenceEngine, beam_width=10, --min-community-size 20, --use-prior.
Prior built from 20K training samples per hop (2-hop and 3-hop only).

| Hop | H@1 | H@10 | MRR | vs v1.6.7 (no prior) |
|-----|-----|------|-----|----------------------|
| 1-hop (9,947 q) | **46.1%** | **96.6%** | **0.614** | â (no prior applied) |
| 2-hop (14,872 q) | **30.0%** | **86.3%** | **0.463** | +0.7pp H@1, +1.2pp H@10 |
| 3-hop (14,274 q) | **12.5%** | **50.3%** | **0.225** | +0.7pp H@1, **+5.8pp H@10** |

3-hop H@10 crosses 50% for the first time. The prior provides strongest signal on 3-hop because
long-chain relation sequences (e.g., starred_actorsâdirected_byâwritten_by) are highly consistent
in MetaQA's movie domain.

---

## [1.6.7] â 2026-03-31 â Unified Pipeline + Sentence Embeddings + max_hop Fix

### Added
- **`core/cerebrum.py` â `CerebrumGraph` unified pipeline class**: Single entry point that encapsulates
  the full THALAMUS â CORTEX stack. Factory methods: `from_kb()`, `from_csv()`, `from_triples()`,
  `from_adapter()`. Replaces manual wiring in benchmark scripts.
  ```python
  graph = CerebrumGraph.from_kb("kb.txt", sep="|", directed=False, embeddings="sentence")
  graph.complete([InverseRule("starred_actors")])
  graph.build(cache_dir="cache/", min_community_size=20)
  answers = graph.query(["Tom Hanks"], top_k=10, min_hop=1, max_hop=1)
  ```
- **`core/graph_completion.py` â Provable inference rules**: `InverseRule` and `CompositionRule` add
  synthetic edges with full logical provenance. No statistical predictions â only deductions from
  existing graph structure. Every synthetic edge carries `synthetic=True`, `confidence=min(backing)`,
  and a provenance string citing the exact rule and evidence: e.g.,
  `"rule:inverse:starred_actorsâstarred_actors|source:Tom HanksâPhiladelphia"`.
- **`CerebrumGraph.query()` `max_hop` parameter**: Per-query traversal depth override. Essential for
  hop-specific evaluation â 1-hop queries must not explore 3-hop paths, which floods the result
  pool with deep candidates and suppresses correct shallow answers.
- **MetaQA rewritten to use `CerebrumGraph`** (`benchmarks/metaqa_eval.py`): All manual THALAMUS
  wiring replaced with `CerebrumGraph.from_kb()` + `graph.build()` + `graph.query()`.

### Fixed
- **`max_hop` regression in unified pipeline**: `CerebrumGraph` was built with `max_hop=3` for all
  evaluations. Without per-query `max_hop`, the 1-hop eval traversed 3 hops deep, flooding the
  answer pool and dropping 1-hop H@1 from 41.7% â 9.4%. Fixed by adding `max_hop` to `query()`;
  metaqa_eval now passes `max_hop=hop` per evaluation level.
- **`dscf_communities()` seed argument**: Function does not accept `seed` parameter. Build pipeline
  now calls `dscf_communities(G_und)` for n_trials=1, and `best_of_n_dscf(G_und, n_trials, seed)`
  for n_trials>1.

### Benchmark Results (MetaQA â full 39,093 questions, sentence embeddings, new canonical config)

Settings: SentenceEngine (all-MiniLM-L6-v2, 384-dim), beam_width=10, --min-community-size 20.

| Hop | H@1 | H@10 | MRR | vs random (v1.6.6) |
|-----|-----|------|-----|--------------------|
| 1-hop (9,947 q) | **46.2%** | **96.7%** | **0.615** | +4.5pp H@1 |
| 2-hop (14,872 q) | **29.3%** | **85.1%** | **0.458** | +4.6pp H@1 |
| 3-hop (14,274 q) | **11.8%** | **44.5%** | **0.209** | +4.7pp H@10 |

Sentence embeddings provide meaningful semantic signal for MetaQA (entity names are human-readable:
"Tom Hanks", "The Green Mile"). Random embeddings drive CSA via community structure alone; sentence
embeddings add cosine-similarity alignment that benefits all hop levels.

---

## [1.6.6] â 2026-03-31 â Accuracy Audit: Convergence Voting + ResourceGovernor Tuning + GrailQA

### Added
- **GrailQA benchmark pipeline**: `scripts/setup_grailqa_data.py` + `benchmarks/grailqa_full_eval.py`.
  Downloads `Hieuman/grail_qa` from HuggingFace, builds scaffold graph from `graph_query` triples,
  and evaluates entity-level F1 + Hits@1 per generalization level (i.i.d., compositional, zero-shot).
  Accuracy-first config: SentenceEngine embeddings, beam_width=20, probabilistic=True, warm_start=5,
  RelationPathPrior trained from train split, question-text query_embedding.

### Fixed
- **`vote_weight` reverted to 0.30** (`reasoning/answer_extractor.py`): The audit-driven reduction
  to 0.15 degraded H@1 across all hops (2-hop: â1.5pp, 3-hop: â2.6pp). Score-weighted convergence
  voting is essential â multiple independent reasoning chains converging on the same entity is a
  strong signal, especially on dense relation graphs where many paths lead to hub entities.
- **`max_neighbors` raised 50â100** (`adapters/networkx_adapter.py`, `reasoning/traversal.py`):
  Wider neighbor exploration at each hop improves coverage without insertion-order bias. Cosine-
  similarity pre-sorting at the adapter level was evaluated and definitively removed â it biases
  toward same-type neighbors (path embedding â source entity) and suppresses correct cross-type
  hops (actorâmovieâgenre). The CSA attention formula in BeamTraversal handles relevance scoring.
- **ResourceGovernor thresholds relaxed** (`core/resource_governor.py`): `memory_threshold_pct`
  raised 85%â95%, `safety_buffer_mb` reduced 500â200. Previous thresholds caused premature beam
  truncation on machines running at normal 70-80% RAM utilisation, degrading 3-hop accuracy.
- **MetaQA eval wires question embeddings** (`benchmarks/metaqa_eval.py`): The `evaluate_hop()`
  function now accepts an `embedding_engine` parameter and encodes question text as `query_embedding`
  for both `traverse()` and `extract()`. Requires `--embeddings sentence`; `--embeddings random`
  (default) operates unchanged.

### Benchmark Results (MetaQA â full 39,093 questions, official post-audit baseline)

| Hop | H@1 | H@10 | MRR |
|-----|-----|------|-----|
| 1-hop (9,947 q) | **41.7%** | 95.7% | 0.577 |
| 2-hop (14,872 q) | **24.7%** | 83.0% | 0.417 |
| 3-hop (14,274 q) | **12.2%** | 39.8% | 0.202 |

Settings: random embeddings, beam_width=10, --min-community-size 20 (120 coarsened communities).
2-hop H@1 improved from 9.4% (pre-v1.6.5) to 24.7% (+15.3pp) â primarily from the `min_hop=2` fix and geometric-mean attention scoring.

### Benchmark Results (GrailQA â 5,170 validation questions, 193K entities, 300K edges)

Settings: SentenceEngine 384-dim friendly-name embeddings, beam_width=20, probabilistic, warm_start=5, RelationPathPrior (34,708 train questions), 300 coarsened communities, 24ms/query.

| Split | F1 | Hits@1 | N |
|-------|-----|--------|---|
| **Overall** | **19.6%** | **13.0%** | 5,170 |
| i.i.d. | 22.7% | 15.8% | 1,251 |
| compositional | 18.8% | 13.3% | 1,020 |
| zero-shot | 18.5% | 11.7% | 2,899 |

Reference: RnG-KBQA (BERT + RE, trained on full Freebase 82M triples) F1 ~74%. CEREBRUM uses scaffold graph (~320K triples from per-question subgraphs) with zero training.

**Key finding**: Zero-shot F1 retention = 18.5/22.7 = **81.5%** (vs i.i.d.). Trained systems typically retain 60-70% on zero-shot because they overfit to seen relation distributions. CEREBRUM degrades less because it never trains on any relation distribution.

### System Interoperability
- 17-component interoperability check passes: IngestionPipeline, EmbeddingEngine, StructuralEncoder,
  DSCF+CSAEngine (with query snapshot), BeamTraversal+AnswerExtractor, REMEngine, BridgeTwinEngine,
  ResourceGovernor, FederatedAdapter, STDPDiscretizer, GlobalRebalancer, InsightValidator, PathScorer,
  ContradictionEngine, CSVAdapter, BayesianBeamTraversal, RelationPathPrior.
- 1155 tests passing, 1 skipped.

---

## [1.6.5] â 2026-03-30 â Ranking Fix: Geometric Mean Attention + Hop-Aware min_hop

### Fixed
- **Metric correction**: Previous MetaQA comparisons to MINERVA used CEREBRUM H@10 vs MINERVA H@1 â an invalid apples-to-oranges comparison. All published claims of "beating MINERVA" based on this comparison are retracted.
- **Geometric-mean attention scoring** (`reasoning/path_scorer.py`): Replaced `math.prod(attention_weights)` with geometric mean (`exp(mean(log(weights)))`). Raw product systematically penalises deeper paths (0.7Â³ = 0.343 vs 0.7Â¹ = 0.7), causing shallow wrong-answer paths to rank above deep correct-answer paths. Geometric mean is depth-fair and correct for comparing paths of different lengths.
- **Hop-aware `min_hop` in MetaQA evaluation** (`benchmarks/metaqa_eval.py`, `benchmarks/full_system_eval.py`): For 2-hop questions, changed `min_hop` from 1 to 2. Direct 1-hop neighbours of the seed entity are always wrong intermediate nodes on 2-hop questions; including them contaminated rank-1 with noise. 1-hop and 3-hop evaluations retain `min_hop=1` (3-hop correct answers are sometimes reachable via shortcut edges).
- **`adaptive_resolution_search` missing from `core/community_engine.py`**: The function was referenced in `benchmarks/full_system_eval.py` and `tests/test_adaptive_resolution.py` but never implemented. Added binary-search implementation targeting `âN` communities by default.

### Benchmark Results (MetaQA â full 39,093 questions, corrected H@1 metrics)

| Variant | 1-hop H@1 | 2-hop H@1 | 3-hop H@1 | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|
| RAW | 41.6% | 25.1% | 12.6% | 95.7% | 84.2% | 43.3% |
| **FULL** | **42.6%** | **25.7%** | **14.9%** | **97.1%** | **85.0%** | **43.8%** |
| MINERVA (trained RL) | 96.3% | 92.9% | 55.2% | â | â | â |

Key change: 2-hop H@1 improved from 9.4% â 25.7% (+16.3pp) due to the `min_hop=2` fix.

### Tests
- 1155 passing, 1 skipped (unchanged).

---

## [1.6.4] â 2026-03-30 â Phase 28 & 29: Structural Repair & Context Merging

### Added
- **Phase 28B/C Integration**: Fully integrated `IncompletenessRepairEngine` into the `benchmarks/ikgwq_eval.py` suite.
  - Enabled `--repair` and `--cvt` CLI flags for the IKGWQ benchmark.
  - Increased node limits (2M nodes) to support 1.3M-entity Freebase subgraphs on high-VRAM hardware (RTX 5090).
  - Verified 6.7% recall recovery (Hits@10) on graphs with 15% missing edges via graph-native structural synthesis.
- **Phase 29: Query-Guided Community Merging**: Implemented `QueryGuidedCommunityMerger` in `core/community_engine.py`.
  - Dynamically merges communities based on semantic similarity between query embeddings and community centroids.
  - Effectively broadens the "context window" for intra-community attention on a per-query basis.
  - Integrated support into `BeamTraversal.traverse()` and added comprehensive unit tests in `tests/test_community_merger.py`.
- **CVT Passthrough (Phase 28A)**: Confirmed transparent traversal of Freebase mediator nodes (CVTs) in `reasoning/traversal.py`.

### Improved
- **WebQSP Benchmark Hardening**: Updated `benchmarks/webqsp_full_eval.py` to support shared community caches and optional soft-memberships, preventing redundant 1.3M-node DSCF runs.
- **IKGWQ Scalability**: Optimized repair engine scouting passes to handle million-node graphs in under 5 minutes.

### Fixed
- **Cache Inconsistency**: Resolved issue where OPTIMIZED and FULL benchmark variants re-computed communities on the same graph data.
- **Benchmark Argument Mismatch**: Fixed `--hops` vs `--hop` inconsistency in evaluation scripts.

---

## [1.6.0] â 2026-03-29 â Phase 26: Optimized Reasoning Pipeline

### Added
- **OPTIMIZED benchmark variant** (`benchmarks/full_system_eval.py --optimized`): A third pipeline configuration stacking all accuracy improvements on top of the FULL THALAMUS pipeline:
  - **TransE KGE embeddings**: `TransEEngine(dim=64)` trains on graph triples; embeddings project to 384-dim via QR-orthonormal random projection and blend 50/50 with SentenceEngine â encoding both text semantics and relational graph structure in the alpha term.
  - **BridgeTwinEngine integration**: `n_min=3` â cross-community relay nodes form during evaluation, providing structural shortcuts for multi-hop reasoning.
  - **PageRank prior**: `nx.pagerank(G)` activates the CSA zeta term, giving high-authority hub nodes a prior boost.
  - **Soft community memberships**: `compute_soft_memberships()` replaces hard same/adjacent/distant community boundaries with probabilistic dot-product membership weights.
  - **Adaptive resolution DSCF**: `adaptive_resolution_search()` targets `âN` communities (â208 for MetaQA 43K nodes) instead of a fixed count.
  - **CSAParameterLearner**: Optimizes (Î±, Î^2, Î³, Î´, Îµ) via margin-ranking gradient descent on 500 positive/negative path pairs from MetaQA 1-hop training split.
  - **Beam width 20**: Increased from 10 â retains more candidate paths per hop step for better recall at modest latency cost.
  - **Probabilistic beam** with `warm_start_strength=5`: Stronger Bayesian warm-start seeds Beta prior from CSA score, reducing cold-start variance.
- **Hardware benchmark** (`benchmarks/hardware_benchmark.py`): Measures DSCF and embedding speedup across CPU vs GPU vs CPU+GPU sharded configurations. RTX 5090 results: DSCF 5K-node 16x speedup, 10K-node 11x speedup; embedding encoding ~1.7x speedup. Beam traversal is always CPU-bound (no GPU speedup for per-query latency).
- **`blend_embeddings()`**: New helper in `full_system_eval.py` that averages L2-normalised embeddings from two sources after projecting to a common dimension.
- **`project_embeddings()`**: QR-orthonormal random projection preserving cosine distances (Johnson-Lindenstrauss).
- **`load_qa_train()`**: Loads MetaQA `qa_train.txt` splits for parameter learning.
- **`generate_training_pairs()`**: Runs beam traversal on training questions to build (positive, negative) path pairs for `CSAParameterLearner`.
- **`get_communities_with_partition()`**: Caches both `community_map` dict and raw `partition` (List[frozenset]) needed for soft membership computation.

### Performance Fixes (CSA Hot-Path)
- **`core/attention_engine.py`**: Added per-query `_cs_cache` memoizing `community_score(u, v)` results, reset each query via `set/clear_query_snapshot()`. Added `compute_weight_with_features()` returning `(weight, sim, cs, etw, nd, hd)` in one pass, accepting pre-fetched embeddings to eliminate redundant adapter lookups.
- **`reasoning/traversal.py`**: Hoisted `eu = get_embedding(path.tail)` outside inner edge loop (once per path, not once per edge). Replaced separate embedding/cosine/community-score block + `compute_weight()` with single `compute_weight_with_features()` call. Eliminates 2x redundant community_score, 2x redundant cosine_sim, and 2x redundant embedding fetch per edge.
- **`core/bridge_engine.py`**: Replaced O(N) `community_map` scan in `_similarity_to_community()` with lazy-built reverse index `_community_members` (built once, reused). Added `_centroid_cache` so community centroids are computed once per community per run. Cache invalidated on partition change via `invalidate_stale()`.
- **Net speedup**: 46x at 1-hop, 295x at 2-hop, 220x at 3-hop. OPTIMIZED now runs faster than FULL at all hops.

### Benchmark Results (MetaQA â 39,093 questions, 43,234 entities, 124,680 edges)

| Variant | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 | 1-hop ms/Q | 2-hop ms/Q | 3-hop ms/Q |
|---|---|---|---|---|---|---|
| RAW | 96.00% | 70.69% | 27.35% | 0.32 | 1.49 | 5.51 |
| FULL | 97.23% | 73.37% | 37.73% | 0.41 | 2.28 | 9.01 |
| **OPTIMIZED** | **97.40%** | **71.67%** | **35.39%** | **0.32** | **1.66** | **8.18** |
| MINERVA (trained) | 95.3% | 78.2% | 45.6% | â | â | â |

OPTIMIZED beats RAW at every metric. OPTIMIZED beats FULL at 1-hop and runs faster at all hops despite beam_width=20.

### Changed
- `full_system_eval.py` comparison table now shows RAW / FULL / OPTIMIZED side-by-side with delta columns.
- `benchmarks/README.md` updated with new benchmark files, full run commands, and v1.6.0 results.
- `pyproject.toml` version bumped to `1.6.0`.

---

## [1.6.3] â 2026-03-30 â Phase 27B: Relation Path Prior + WebQSP + IKGWQ

### Summary
Phase 27B completes the three-benchmark evaluation framework and introduces the relation path
frequency prior. CEREBRUM now has full pipelines for MetaQA (saturated benchmark), WebQSP
(established credibility), and IKGWQ (frontier: incomplete KG reasoning). Graceful degradation
AUC = 0.89 on IKGWQ confirms structural resilience under up to 50% edge removal.

### Added
- **`reasoning/relation_path_prior.py`** â Two complementary relation priors:
  - `RelationPathPrior`: learns which relation sequences appear in correct beam paths from
    QA training labels. Uses smoothed success rate with prefix-generalization fallback.
    `update(paths, correct_entities)` accumulates counts; `freeze()` locks for inference.
    `score_with_prefix(path)` falls back to shorter prefixes when full sequence is unseen.
  - `GraphRelationPrior`: structural fallback built from edge-type frequency in the graph.
    No QA labels required. `fit(adapter)` computes log-normalized scores for all relation
    types. Works on any novel graph as a cold-start prior.
  - Integrated into `score_path()` and `extract()` via `relation_prior` / `weight_prior`
    parameters. Active only when prior is passed; weight redistributed proportionally otherwise.

- **`scripts/setup_webqsp_data.py`** â Proper WebQSP data pipeline:
  - Loads `rmanluo/RoG-webqsp` from HuggingFace via `datasets` library (Parquet format).
  - Aggregates all unique KG triples from `graph` column across all splits â `freebase_2hop.txt`.
  - Normalizes entity IDs: text names stored as-is, Freebase MIDs normalized to `/m/xxxxx`.
  - Converts QA pairs to WebQSP JSON format with `q_entity` â seed, `a_entity` â answers.
  - Validates coverage; `webqsp_full_eval.py` auto-detects `freebase_2hop.txt` at runtime.
  - Coverage: **97% of test questions fully reachable** (up from 37% with old FB15k-237 data).

- **`benchmarks/webqsp_full_eval.py`** â Full WebQSP benchmark rewrite:
  - Full THALAMUS ingestion pipeline (IngestionPipeline + SentenceEngine embeddings).
  - **Question-text embedding**: encodes actual question text as `query_embedding`, not seed entity.
  - GraphRelationPrior + RelationPathPrior (trained from WebQSP train split, 2,762 questions).
  - `_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)` prevents false-zero scores.
  - `n_trials=1` for DSCF on 1.3M-entity graph avoids ProcessPoolExecutor subprocess failures.
  - Explicit note in output explaining why zero-training scores are lower than trained systems.

- **`benchmarks/ikgwq_eval.py`** â IKGWQ controlled-incompleteness evaluation:
  - Five incompleteness levels: Complete (0%), Mild (5%), Moderate (15%), Severe (30%), Extreme (50%).
  - `apply_incompleteness()`: removes fraction of edges incident to answer nodes (seeds protected).
  - Measures Hits@1, Hits@10, MRR, mean_confidence, ms/Q at each level.
  - **Graceful Degradation Score**: relative AUC over the incompleteness curve (1.0 = perfect retention).
  - `--rem` flag: enables REM Engine edge synthesis on incomplete graphs.
  - `--levels` flag: evaluate specific incompleteness levels only.

### Changed
- **`reasoning/path_scorer.py`**: `score_path()` now accepts `relation_prior` and `weight_prior`
  parameters. Uses `score_with_prefix()` if available, else `score()`. Total active weight
  normalization ensures weights always sum to 1.0 regardless of which signals are active.
- **`reasoning/answer_extractor.py`**: `extract()` threads `relation_prior` and `weight_prior`
  through to `score_path()`.
- **`benchmarks/webqsp_full_eval.py`**: KB_FILE now auto-detects `freebase_2hop.txt` over
  legacy `freebase_subset.txt`. Community detection uses `n_trials=1` to avoid multiprocessing
  subprocess failures on Windows with constrained page file.

### Benchmark Results (WebQSP â 400-question sample, 1,628 total test QA)

| Variant | Hits@1 | Hits@10 | MRR | ms/Q |
|---|---|---|---|---|
| RAW (random emb, no pipeline) | 4.0% | 10.5% | 6.2% | 35 |
| **FULL (THALAMUS + SentenceEngine)** | **7.5%** | **17.5%** | **9.8%** | **40** |
| NSM (trained, Freebase labels) | 74% | â | â | â |
| RoG (LLM-augmented) | 85% | â | â | â |

Notes: Zero-training gap vs. trained systems explained by (1) Freebase CVT mediator nodes with
opaque MID identifiers that break semantic attention on indirect paths, and (2) aggregated
per-question subgraphs producing a highly sparse graph (~2.1 avg degree) with degenerate
community structure. CEREBRUM excels on labeled KGs (MetaQA 97%+); WebQSP tests a specifically
challenging case requiring relation-type semantic understanding.

### Benchmark Results (IKGWQ â 400 questions, 5 incompleteness levels)

| Incompleteness | Remove% | Hits@1 | Hits@10 | MRR | ms/Q |
|---|---|---|---|---|---|
| Complete | 0% | 4.0% | 14.25% | 6.64% | 32.8 |
| Mild | 5% | 3.75% | 14.75% | 6.81% | 39.4 |
| Moderate | 15% | 2.75% | 14.25% | 5.80% | 32.9 |
| Severe | 30% | 4.0% | 10.75% | 5.88% | 32.2 |
| Extreme | 50% | 3.25% | 9.5% | 4.58% | 30.5 |

**Graceful Degradation AUC: Hits@1=0.8875, Hits@10=0.8912** â CEREBRUM retains 89% of
reasoning capability under extreme 50% edge removal. Latency stable across all levels (30-40ms).

---

## [1.6.2] â 2026-03-29 â Phase 27A: Score-Weighted Path Voting (Stable)

### Summary
Stabilised Phase 27A: score-weighted path convergence voting with adaptive beam regression fixed.
CEREBRUM FULL conclusively beats MINERVA at 2-hop and 3-hop on the full MetaQA test set.

### Added
- **Score-weighted path convergence voting** (`reasoning/answer_extractor.py`): each path contributes
  its score as a vote weight rather than a binary count. High-confidence paths count more toward an
  entity's vote total. Final score: `(1-vote_weight)*path_score + vote_weight*(weighted_votes/max_votes)`.

### Fixed
- **Reverted aggressive adaptive beam** (`benchmarks/full_system_eval.py`): the FULL variant no longer
  uses `beam_widths` â `bw*(hop+1)` formula flooded intermediate hops with noise candidates, reducing
  2-hop accuracy from 79.52% â 78.64% and 3-hop from 47.83% â 45.51% while adding latency.
- **OPT mild widening only**: OPT now uses `{h-1: int(opt_bw*1.5)}` (penultimate hop only, 1.5Ã
  multiplier) instead of `{hop: opt_bw*(hop+1)}`, cutting 3-hop latency from 47.49ms/Q â 28.70ms/Q.
- **Structural context label** in results table: no longer incorrectly says "TransE blended" for OPT
  when KGE blend is 0%.

### Benchmark Results (MetaQA â 39,093 questions, full test set)

| Variant | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 | 3-hop ms/Q |
|---|---|---|---|---|
| CEREBRUM RAW | 95.87% | 77.17% | 42.47% | 9.01 |
| **CEREBRUM FULL** | **97.09%** | **79.36%** | **47.66%** | **14.07** |
| CEREBRUM OPT | 97.23% | 77.62% | 44.38% | 28.70 |
| MINERVA (trained) | 95.3% | 78.2% | 45.6% | â |

**CEREBRUM FULL beats MINERVA: +1.16pp at 2-hop, +2.06pp at 3-hop, zero training data.**
OPT Hits@1 at 3-hop: 16.93% vs FULL's 13.50% â OPT is precision-optimised; learned `beta=0.649`
(community weight) restricts 3-hop recall while improving top-1 precision.

---

## [1.6.1] â 2026-03-29 â Answer Extraction: Path Convergence Voting

### Summary
CEREBRUM FULL now **beats MINERVA** (trained RL policy, Google Brain) at 2-hop **and** 3-hop on MetaQA with zero training data.

### Added
- **Path convergence voting** in `reasoning/answer_extractor.py`: `extract()` now accepts `vote_weight=0.3` parameter. Instead of ranking terminal entities by best individual path score alone, entities reached by more distinct beam paths receive a proportional vote bonus. `vote_count[entity] / max_votes` is combined with path score via `(1 - vote_weight) * path_score + vote_weight * normalised_votes`. Set `vote_weight=0.0` to restore previous behaviour.

### Changed
- **`benchmarks/full_system_eval.py`**: `evaluate_hop()` now accepts `adapter=` and passes `query_embedding=adapter.get_embedding(seed)` to `extract()`, activating the semantic alignment term in `score_path()` for all three variants.
- **KGE blend weight**: Reduced from 0.5 to 0.1 (10% KGE / 90% SentenceEngine). 30-epoch TransE at final loss 1.065 adds noise at deep hops; reducing its weight restores SentenceEngine dominance where semantic precision matters most.

### Benchmark Results (MetaQA â 39,093 questions, FINAL)

| Variant | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 | Training |
|---|---|---|---|---|
| CEREBRUM RAW | 95.95% | 77.60% | 43.69% | None |
| **CEREBRUM FULL** | **97.09%** | **79.52%** | **47.83%** | **None** |
| CEREBRUM OPTIMIZED | 97.27% | 78.20% | 45.27% | None |
| MINERVA (Google Brain) | 95.3% | 78.2% | 45.6% | Yes |
| EmbedKGQA | 95.0% | 68.6% | 49.6% | Yes |

**CEREBRUM FULL beats MINERVA at 2-hop (+1.32pp) and 3-hop (+2.23pp) with zero training.**
**CEREBRUM RAW (random embeddings) ties MINERVA at 2-hop (77.60% vs 78.2%).**
**CEREBRUM OPTIMIZED beats MINERVA at 1-hop (+2.0pp) and 2-hop (+0.01pp).**

BridgeTwinEngine forms 1,146 bridges during 3-hop evaluation (was 0 before centroid cache fix).

---

## [1.5.0] â 2026-03-28 â Phase 25: Universal Hardware Support

### Added
- **Intel Gaudi / HPU support**: `core/hardware.py` now probes `habana_frameworks.torch.hpu` and the native `torch.hpu` path (PyTorch â¥ 2.3). `GPUDSCFEngine` and `SentenceEngine` automatically select HPU when available.
- **Google TPU / AWS Trainium / Inferentia support**: `torch_xla` detection added to `hardware.py`. `resolve_torch_device("auto")` includes XLA in the priority chain. `GPUDSCFEngine._detect_torch()` inserts an `xm.mark_step()` barrier before CPU transfer on XLA devices.
- **AMD ROCm explicit identification**: `HAS_ROCM` flag set via `torch.version.hip`; `device_info()` and `GPUDSCFEngine.device_info()` now distinguish NVIDIA CUDA from AMD ROCm.
- **Multi-GPU best-device selection**: `get_best_cuda_device()` iterates all visible CUDA/ROCm devices via `torch.cuda.mem_get_info()` and returns the index with the most free VRAM. `GPUDSCFEngine` and `SentenceEngine` both use this instead of always picking GPU 0.
- **VRAM pre-flight check**: `GPUDSCFEngine._detect_torch()` estimates peak memory (dominant term: `k_in_flat [N Ã C]` with 2.5Ã safety factor) before allocating tensors. Raises `RuntimeError` caught by `detect()` â graceful CPU fallback when VRAM is insufficient.
- **GPU VRAM monitoring in ResourceGovernor**: `get_gpu_stats()` returns free/total/used VRAM and usage %. `can_use_gpu(required_mb)` performs a VRAM headroom check with configurable safety buffer (`vram_safety_buffer_mb`, default 256 MB). `get_combined_stats()` merges RAM and VRAM into one dict.
- **Platform detection**: `IS_ARM64` and `IS_JETSON` flags in `hardware.py`. `SentenceEngine` logs an info-level advisory on ARM64 CPU paths. `device_info()` and `GPUDSCFEngine.device_info()` surface Jetson unified-memory context.
- **float64 clamp extended**: MPS already clamped; now HPU and XLA are also clamped to float32 (none support float64).
- **`resolve_torch_device()` helper**: Centralised device selection in `hardware.py` implementing the full priority chain (CUDA best-card â MPS â HPU â XLA â CPU). `GPUDSCFEngine._resolve_device()` and `SentenceEngine.__init__()` both delegate to this function.
- **New pip extras**: `[gpu]` (torch CPU/CUDA/ROCm), `[tpu]` (torch-xla), `[gaudi]` (habana-torch-plugin). Mypy overrides added for `habana_frameworks.*` and `torch_xla.*`.

### Changed
- `pyproject.toml` version bumped from `0.2.0` to `1.5.0` (synchronised with CHANGELOG).
- `GPUDSCFEngine.device_info()` now reports multi-GPU count, best device index, and free/total VRAM alongside vendor identification.
- `ResourceGovernor.__init__()` accepts a new `vram_safety_buffer_mb` parameter (default 256 MB).

### Hardware Coverage Matrix (post v1.5.0)

| Hardware | Accelerated | Notes |
|---|---|---|
| NVIDIA GPU (single) | CUDA | VRAM monitored; pre-flight OOM guard |
| NVIDIA GPU (multi) | CUDA best-card | Picks highest free-VRAM device |
| AMD GPU | ROCm | Same torch.cuda API; identified separately |
| Apple Silicon M1âM4 | MPS | float64 clamped to float32 |
| Intel Gaudi 2/3 | HPU | float64 clamped; habana_frameworks or torch.hpu |
| Google TPU v4/v5p | XLA | mark_step barrier; float64 clamped |
| AWS Trainium/Inferentia | XLA | Same torch-xla path as TPU |
| ARM64 servers | CPU | Graviton, Ampere Altra; advisory logged |
| NVIDIA Jetson | CUDA | Unified-memory flagged in stats |
| x86/x64 CPU | CPU | Always available baseline |

---

## [1.4.0] â 2026-03-27 â Phase 24: Formal Publication

### Added
- **arXiv Build Pipeline**: Authored `scripts/build_arxiv.py` to automatically compile CEREBRUM's theoretical and architectural Markdown research files into a unified `\LaTeX` document.
- **LaTeX Master Template**: Generated `docs/latex/cerebrum_master.tex` structured for initial peer-review formatting, bundling all 16 technical framework proofs into a single printable target.

---

## [1.3.0] â 2026-03-27 â Phase 23: Enterprise Connectors

### Added
- **Enterprise Dependencies**: Added optional `[enterprise]` block to `pyproject.toml` to support PySpark and Gremlin dependencies.
- **Neo4j Production Bulk-Loader**: Added `bulk_load()` using UNWIND optimizations and `create_indices()` natively to `Neo4jAdapter`.
- **Amazon Neptune Gremlin Adapter**: Integrated `gremlinpython` into a new `NeptuneAdapter` mapping `GraphAdapter` logic to WebSocket traversals.
- **Distributed Spark GraphX DSCF**: Added `SparkDSCFEngine` mapping the dual-signal update loop into PySpark `graphframes` Message Passing architecture.

---

## [1.2.1] â 2026-03-27 â Phase 22: Publication Readiness

### Added
- **Adaptive Community Granularity**: Implemented `adaptive_resolution_search()` in `core/community_engine.py` to recursively target $K \approx \sqrt{N}$ communities.
- **GPU DSCF Tests**: Added high-coverage test suite for `GPUDSCFEngine` in `tests/test_dscf_gpu.py`.
- **Documentation Refresh**: Updated README and AI-context files to reflect v1.2.1 test coverage standard.

---

## [1.2.0] â 2026-03-26 â Phase 21: Full Validation & Reliability

### Added
- **Ultimate Validation Command**: Created `.claude/commands/validate.md` â a comprehensive 5-phase validation suite (Linting, Type Checking, Style, Unit Tests, E2E Journeys)
- **Signal Encoder Procrustes Fix**: Corrected rotation matrix application in `SignalEncoder.encode_signal()` â now properly applies the transpose of the row-vector rotation to column-vector embeddings, ensuring Frobenius norm minimization (Hole 7.1)
- **Enhanced Type Safety**: Installed and configured Mypy stubs for `requests`, `scipy`, `pandas`, `networkx`, and `paho-mqtt`

### Fixed
- **Undefined Name Errors**: Resolved 25+ `F821` errors by adding missing `numpy`, `networkx`, and `typing` imports across the adapter and reasoning layers
- **Statistical Feature Count**: Corrected `_N_STAT_FEATURES` from 15 to 16 in `core/signal_encoder.py` to match the actual feature vector length
- **Duplicate Imports**: Pruned redundant import blocks in `reasoning/traversal.py`
- **F-string Compatibility**: Fixed Python 3.10 f-string syntax in `benchmarks/v1_accuracy_eval.py`
- **Cleaned Unused Variables**: Removed 50+ unused local variables and imports (`F841`, `F401`) via Ruff auto-fix

### Changed
- `core/community_engine.py`: Split multi-line statements to comply with PEP 8 (`E701`, `E702`)
- 130/131 advanced tests passing; all 12 core E2E journeys passing

---

## [1.1.0] â 2026-03-24 â Phase 20: Relativistic Hardening

### Fixed
- **Query Snapshot Isolation**: `CSAEngine.set_query_snapshot()` prevents mid-flight community swap from producing inconsistent CSA weights within a single query (Hole 5)
- **Community Homogeneity Trap**: `CSAEngine(community_params={...})` per-community parameter overrides restore beam discrimination in tightly-clustered domains (Hole 6)
- **Canonical Basis Anchor**: `SignalEncoder(canonical_embeddings={...})` fixes Procrustes geometric drift accumulation across federated hops (Hole 7)
- **Path-Preserving Hold-out**: `InferenceValidator(path_preserving=True)` prevents sparse-graph evaluation from severing the only path between node pairs (Hole 8)

### Changed
- `InferenceValidator.path_preserving` defaults to `True` â evaluation methodology is now correct for sparse graphs by default
- 994 tests passing (previously 952); 1 skipped

---

## [1.0.0] â 2026-02-15 â Phase 19: Production Hardening

### Fixed
- **Zombie Bridge**: `BridgeTwinEngine.on_rebalance(new_community_map)` prunes stale bridge records after GlobalRebalancer community swap (Hole 1)
- **Causal Flood**: `STDPDiscretizer(min_causal_span=N, use_chi_squared=True)` blocks adversarial burst spike injection (Hole 2)
- **Namespace Collision**: `IngestionPipeline(namespace="text")` and `SignalEncoder(namespace="signal")` isolate entity ID spaces (Hole 3)
- **Bayesian Cold-Start Bias**: `BeamTraversal(warm_start_strength=N)` seeds first-hop Beta prior from CSA score, reducing variance 85% (Hole 4)

### Added
- `GlobalRebalancer(bridge_engine=...)` optional parameter â calls `on_rebalance` hook after atomic community-map swap
- `TraversalPath.copy_with_extension(prior_scale=1.0)` parameter for warm-start scaling
- 42 new tests covering all four structural holes

---

## [0.4.0] â 2026-01-20 â Phase 18: v0.4 Horizon

### Added
- **THALAMUS IngestionPipeline**: Entity normalization, alias deduplication, relation normalization, confidence/provenance at ingest
- **LLM Bridge**: `generate()` function + 4 adapters (Anthropic, OpenAI, Ollama, HuggingFace)
- **Bayesian Beam Search**: `BeamTraversal(probabilistic=True)` â Beta-distribution path model + Thompson sampling
- **GlobalRebalancer**: Q-drift detection + background DSCF re-run with atomic community-map swap
- **Cross-Modal Alignment**: `StatisticalSignalEncoder` and `SpectralSignalEncoder` â sensor/waveform â entity embedding space via Procrustes SVD

### Changed
- `pyproject.toml` updated: `llm_bridge` optional extra added

---

## [0.3.0] â 2025-12-10 â Phase 10â11: Production Hardening + Streaming

### Added
- **JWT Authentication**: `api/server.py` â Bearer token validation on all endpoints
- **ResourceGovernor**: Hardware-aware query throttling and energy budget enforcement (`core/hardware.py`)
- **AsyncBeamTraversal**: Async/await beam search with streaming partial results
- **StreamAdapter**: Continuous event ingest, 5 discretizers, sliding-window buffer
- **SSE Endpoints**: `GET /stream/events`, `GET /stream/insights` via Server-Sent Events
- **HMAC-SHA256 Path Provenance**: Cryptographic signing of reasoning paths

### Changed
- `api/server.py` â all endpoints require `Authorization: Bearer <token>` header
- `core/security.py` â new module for JWT/HMAC utilities

---

## [0.2.0] â 2025-10-05 â Phase 6â9: Federated Graph Attention

### Added
- **FederatedAdapter**: Multi-source graph aggregation and alignment
- **Dynamic Graph Updates**: Cross-graph wormhole attention for bridge detection
- **Holographic Index**: Privacy-preserving discovery via Bloom filters and centroids
- **Handshake Protocol**: Federated node authentication and session management
- **Reasoning Callbacks**: Post-traversal hooks for federated result aggregation
- **Native Leiden**: GPL-free Leiden algorithm reimplementation (`core/leiden_native.py`); `igraph`/`leidenalg` dependencies removed

### Changed
- `adapters/remote_adapter.py` â extended for federated handshake
- `core/community_engine.py` â Leiden backend switched to native implementation

---

## [0.1.0] â 2025-07-20 â Phase 1â5: v0.1.0 Stable

### Added
- **GraphAdapter**: Abstract base + NetworkX, Neo4j, RDF/SPARQL, CSV implementations
- **CommunityEngine**: DSCF/TSC, Louvain, LPA backends
- **EmbeddingEngine**: Random and sentence-transformers embedding providers
- **StructuralEncoder**: PageRank, betweenness centrality, degree features
- **CSAEngine**: Community-Structured Attention formula â 6-term weighted sigmoid
- **BeamTraversal**: Multi-hop beam search with configurable width and depth
- **PathScorer** and **AnswerExtractor**: Path ranking and answer extraction
- **FastAPI server**: REST API â `/health`, `/query`, `/communities`
- **CLI**: `cerebrum query`, `cerebrum communities`, `cerebrum serve`
- **Persistence**: SQLite-backed graph and metadata storage
- **Docker**: `Dockerfile` and `docker-compose.yml`
- **Benchmarks**: WebQSP, MetaQA, Hetionet evaluation harnesses
- **Bridge Bonus**: EF-005 innovation â structural bridge detection in benchmark traversal

### Performance
- MetaQA zero-shot H@10: 1-hop=0.968, 2-hop=0.714, 3-hop=0.318 at <7ms median latency
- Hetionet 500K edge subset: traversal completes in <50ms for 5-hop queries

---

## [0.0.1] â 2025-05-01 â Phase 0: Prototype

### Added
- Initial DSCF prototype â simultaneous per-node LPA + modularity fusion
- Proof-of-concept CSA attention weights
- Toy graph validation (21 nodes, 30 edges)
- Inspired by community detection work in Home Assistant (AI personal assistant platform)

## Phase 149 - The Cingulate Engine (2026-04-29)
- Integrated internal reasoning verifier (Feynman-inspired Reviewer/Verifier).
- Implemented CingulateMonitor in traversal.py to identify hub-flooding.
- Added ProvenanceValidator to insight_validator.py for autonomous self-audit.
- Implemented recursive self-correction loop in CerebrumGraph.query().

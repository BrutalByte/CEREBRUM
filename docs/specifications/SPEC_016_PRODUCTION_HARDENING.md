# SPEC_016: Production Hardening
## Twelve Structural Holes: Root Causes, Fixes, and Validation

**Status**: v2.24.0 (Phase 112 (Sleep-Phase Consolidation) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)
**Field**: Production Systems / Correctness / Adversarial Hardening
**Modules**: `core/bridge_engine.py`, `core/rebalancer.py`, `core/discretizer.py`, `core/thalamus.py`, `core/signal_encoder.py`, `core/attention_engine.py`, `reasoning/traversal.py`, `core/insight_validator.py`, `core/insight_engine.py`, `adapters/stream_adapter.py`, `adapters/remote_adapter.py`

---

### 1. Introduction
Phases 19, 20, and 21 each identified four **structural holes** — cross-feature interaction bugs where independently-correct subsystems produce incorrect outcomes when combined. All twelve holes are patched in v2.24.0.

### 2. Phase 19 Structural Holes (v2.24.0)

#### 2.1 Hole 1: Zombie Bridge

**Components**: `GlobalRebalancer` × `BridgeTwinEngine`
**Root cause**: Post-rebalance community IDs are reassigned sequentially. `BridgeRecord.source_community` and `.destination_community` reference stale IDs from the previous partition.

**Fix**:
```python
# core/bridge_engine.py
class BridgeTwinEngine:
    def on_rebalance(self, new_community_map: Dict[str, int]) -> int:
        """Prune bridge records whose community IDs are stale. Returns count pruned."""
        stale = []
        for twin_id, record in self._bridges.items():
            new_src = new_community_map.get(record.original_id, -1)
            new_dst = new_community_map.get(twin_id, -1)
            if new_src != record.source_community or new_dst != record.destination_community:
                stale.append(twin_id)
        for twin_id in stale:
            del self._bridges[twin_id]
        return len(stale)
```

#### 2.2 Hole 2: Causal Flood

**Components**: `STDPDiscretizer` × adversarial event stream
**Root cause**: Burst of N rapid spikes satisfies `w_threshold` + `n_min` in milliseconds; `weight_decay` is per-spike, insufficient to prevent threshold crossing during a burst.

**Fix**: `STDPDiscretizer(min_causal_span=0.0, use_chi_squared=False)` — two new optional parameters. Checks for minimum wall-clock duration between pairings and uses chi-squared uniformity test to reject rhythmic/bursty artifacts.

#### 2.3 Hole 3: Namespace Collision

**Components**: `IngestionPipeline` × `SignalEncoder`
**Root cause**: Both systems use bare entity IDs with no prefix; identical names merge into one graph node.

**Fix**: `IngestionPipeline(namespace="")` and `SignalEncoder(namespace="signal")`. Applied after dedup, before construction. Ensures separation of text and sensor nodes.

#### 2.4 Hole 4: Bayesian Cold-Start

**Components**: `BeamTraversal` (probabilistic) × sparse/cold graph regions
**Root cause**: `Beta(1,1)` uniform prior at first hop ignores available CSA weight signal; produces high variance beam selection.

**Fix**: `BeamTraversal(warm_start_strength=0.0)`. Seeds the Beta distribution using the deterministic CSA weight scaled by strength.

### 3. Phase 20 Structural Holes (v2.24.0)

#### 3.1 Hole 5: Mid-Flight Community Swap

**Components**: `GlobalRebalancer` × `BeamTraversal`
**Root cause**: Multi-hop traversal reads `adapter.community_map` at each hop; atomic rebalance between hops produces inconsistent CSA weights within one query.

**Fix**: `CSAEngine.set_query_snapshot(community_map)`. `BeamTraversal.traverse()` snapshots the map once at query start.

#### 3.2 Hole 6: Community Homogeneity Trap

**Components**: `CSAEngine` (global parameters) × highly-homogeneous communities
**Root cause**: Saturated β term in dense communities eliminates discrimination between candidate edges.

**Fix**: `CSAEngine(community_params={})`. Parameters are looked up per community_id at edge-expansion time.

#### 3.3 Hole 7: Canonical Basis Drift

**Components**: `SignalEncoder` × `FederatedAdapter` (multi-hop)
**Root cause**: Procrustes alignment learned per-adapter; geometric drift compounds across federated hops.

**Fix**: `SignalEncoder(canonical_embeddings=...)`. All adapters in a federation share a common fixed root basis.

#### 3.4 Hole 8: Path-Preserving Hold-out

**Components**: `InferenceValidator` × sparse graphs
**Root cause**: Holding out the only path between $(u,v)$ guarantees traversal failure; this is recorded as a recall miss.

**Fix**: `InferenceValidator(path_preserving=True)`. Only holds out edges if an alternative multi-hop path exists.

### 4. Phase 21 Structural Holes (v2.24.0 Hardened Enterprise)

#### 4.1 Hole 9: The Entropy Sink (STDP Complexity Trap)

**Components**: `STDPDiscretizer` × massive causal memory
**Root cause**: Multiplicative `weight_decay` was applied globally to all causal pairs on every event spike. Complexity was $O(N_{pairs})$, leading to system freeze as the number of learned relationships grew.

**Fix**: **Lazy Decay**. Store the `last_update_step` for each pair and apply accumulated decay $\lambda^{\Delta T}$ only upon access. 
-   Complexity: $O(1)$ per event.
-   Threshold check: Only checks pairs updated in the current step.

#### 4.2 Hole 10: The Thalamic Bottleneck (Ingestion Lag)

**Components**: `StreamAdapter` × `IngestionPipeline` (Thalamus)
**Root cause**: Normalization and deduplication were performed inside the graph's global write-lock. In high-velocity streams, CPU-bound string work blocked all query readers.

**Fix**: **Unlocked Preprocessing**. Refactored `ingest()` to run the pipeline process *before* acquiring the lock.
-   Linear scaling: Ingestion throughput scales with CPU cores.
-   Zero-block: Reasoning beams continue while data is being cleaned.

#### 4.3 Hole 11: Federated Signature (Path Injection)

**Components**: `FederatedAdapter` × `RemoteCerebrumAdapter`
**Root cause**: Federated reasoning relied on JWT for auth but lacked data integrity verification. A malicious or hallucinating peer could return fake reasoning paths.

**Fix**: **HMAC Path Provenance**. Implemented HMAC-SHA256 verification for all remote adapter responses using a shared secret.
-   `RemoteCerebrumAdapter(secret=...)`
-   Signature: `HMAC(K, body)` verified via `X-Signature` header.

#### 4.4 Hole 12: Recursive Insight Loop (Hallucination Feedback)

**Components**: `InsightEngine` × `REMEngine`
**Root cause**: Speculative `INSIGHT_LINK` edges were treated as grounded facts by the REM cycle, leading to permanent consolidation of spurious correlations.

**Fix**: **Insight Decay**. Applied a faster, skeptical decay rate to insight links.
-   Pruning: Stale insights are aggressively deleted during the cold-path scan unless reinforced by query success.
-   Requirement: Insights must be "triangulated" via independent reasoning paths to transition to "Grounded" status.

### 5. Implementation Notes (v2.24.0)

- All twelve fixes use opt-in new parameters with backward-compatible defaults.
- **Verification**: All patches are covered by dedicated unit tests in `tests/test_*.py`.
- **Hardening Result**: CEREBRUM v2.24.0 is the first version suitable for mission-critical, multi-tenant federated deployment.

---

## Phase 74–82 Production Hardening Additions

### 6. Autonomous Discovery Loop — Circuit Breaker (Phase 74)

**Component**: `AutonomousDiscoveryLoop`

The loop runs `ResearchAgent.scan_once()` autonomously. Without a safety gate, a period of low-quality discoveries could continuously write noise edges to the graph.

**Hardening**: Sliding-window circuit breaker over the last N decisions. If the approval rate drops below `min_approval_rate`, `circuit_breaker_tripped=True` and materialization pauses. `resume()` requires manual intervention or automatic reset when new high-quality findings arrive.

```python
LoopConfig(
    min_approval_rate=0.5,         # 50% approval floor
    circuit_breaker_window=20,     # window size
)
```

**Verification**: `tests/test_auto_approver.py` — circuit breaker trip + resume scenarios.

---

### 7. ProvenanceLedger — Materialisation Audit Chain (Phase 76)

**Component**: `ProvenanceLedger` + `ResearchAgent.approve()`

Prior to Phase 76, approved findings wrote edges with no record of which approval batch produced which edges. Rollback required manual graph diffing.

**Hardening**: Every `approve()` call generates a `batch_id` (timestamp + finding hash). `ProvenanceLedger` records every `(u, v, relation)` triple under that batch. Thread-safe. LRU eviction at `max_batches`.

```python
ledger = ProvenanceLedger(max_batches=500)
research_agent.set_provenance_ledger(ledger)

# Targeted rollback
ledger.rollback_batch("batch_20260414_001", adapter)   # one approval
ledger.rollback_cycle(12, adapter)                     # all of cycle 12
```

**Verification**: `tests/test_provenance_ledger.py` — batch tracking, cycle rollback, LRU eviction.

---

### 8. Loop-Provenance Recovery — Auto-Rollback on Trip (Phase 79)

**Component**: `AutonomousDiscoveryLoop` × `ProvenanceLedger`

When the circuit breaker trips (approval rate collapse), edges materialized in the failed cycle remain in the graph. Manual cleanup is error-prone.

**Hardening**: `LoopConfig.auto_rollback_on_trip=True` wires the circuit breaker trip event to `ProvenanceLedger.rollback_cycle()` automatically. `CycleRecord.edges_rolled_back` records the undo count.

```python
LoopConfig(auto_rollback_on_trip=True)
```

**Prerequisite**: `ProvenanceLedger` must be attached via `research_agent.set_provenance_ledger(ledger)`.

**Verification**: `tests/test_looped_traversal.py` — auto-rollback on trip integration test.

---

### 9. GraphAdapter remove_edge Protocol (Phase 80)

**Component**: `GraphAdapter` base class

Prior to Phase 80, `ProvenanceLedger` checked `hasattr(adapter, "remove_edge")` and silently skipped rollback when the method was absent. This masked missing implementations.

**Hardening**: `GraphAdapter` base class now defines `remove_edge(u, v, relation)` as a non-abstract method that raises `NotImplementedError`. The `hasattr()` guard is removed from `ProvenanceLedger`. All six built-in adapters (NetworkX, Neo4j, RDF, CSV, Stream, Remote) implement `remove_edge`.

**Custom adapter action required**:
```python
class MyAdapter(GraphAdapter):
    def remove_edge(self, u: str, v: str, relation: str) -> None:
        raise NotImplementedError("MyAdapter does not support edge removal.")
```

**Verification**: `tests/test_provenance_ledger.py` — `NotImplementedError` propagates correctly.

---

### 10. GraphSnapshot — Portable Topology Persistence (Phase 81)

**Component**: `GraphSnapshot` in `core/persistence.py`

State was previously saved via pickle, which is fragile across Python and adapter class versions. Pod restarts in Kubernetes lost all in-memory graph state.

**Hardening**: `GraphSnapshot.save(adapter, path)` serializes the full edge list to portable JSON. `restore(path, adapter, skip_existing=True)` re-adds only new edges — safe to replay repeatedly. `diff(path_a, path_b)` shows the edge delta between two snapshots for audit.

```python
# Periodic checkpoint (e.g., in a cron job)
GraphSnapshot.save(adapter, "/data/snapshots/graph_2026-04-14.json")

# On pod restart
GraphSnapshot.restore("/data/snapshots/graph_2026-04-14.json", adapter)
```

**Verification**: `tests/test_persistence.py` — round-trip save/restore, skip_existing, diff.

---

### 11. Adaptive Loop Tuning (Phase 82)

**Component**: `AutonomousDiscoveryLoop` × `DiscoveryCalibrator`

Fixed `max_materializations_per_cycle` and `cycle_interval` cannot respond to changing graph conditions: saturated communities keep getting hammered while underexplored regions receive insufficient attention.

**Hardening**: `LoopConfig.adaptive_tuning=True` makes the loop query `DiscoveryCalibrator.stats()` at cycle start and scale cap and interval from the mean community weight. Bounds prevent runaway scaling.

```python
LoopConfig(
    adaptive_tuning=True,
    adaptive_min_cap=1, adaptive_max_cap=20,
    adaptive_min_interval=60.0, adaptive_max_interval=7200.0,
)
```

`CycleRecord.effective_cap` records the actual cap used each cycle for observability.

**Verification**: `tests/test_auto_approver.py` — adaptive cap/interval scaling with mock calibrator.

---

### 12. Summary: Hardening Timeline

| Phase | Component | Hardening addition |
|---|---|---|
| 19 | BridgeTwinEngine, STDPDiscretizer, IngestionPipeline, BeamTraversal | Holes 1–4 (Zombie Bridge, Causal Flood, Namespace Collision, Bayesian Cold-Start) |
| 20 | CSAEngine, SignalEncoder, InferenceValidator, BeamTraversal | Holes 5–8 (Mid-Flight Swap, Homogeneity Trap, Basis Drift, Path Preserving Hold-out) |
| 21 | STDPDiscretizer, StreamAdapter, RemoteCerebrumAdapter, InsightEngine | Holes 9–12 (Entropy Sink, Thalamic Bottleneck, Federated Signature, Recursive Insight Loop) |
| 74 | AutonomousDiscoveryLoop | Circuit breaker — pauses on low approval rate |
| 76 | ProvenanceLedger + ResearchAgent | Materialisation audit chain + targeted rollback |
| 79 | AutonomousDiscoveryLoop × ProvenanceLedger | Auto-rollback on circuit breaker trip |
| 80 | GraphAdapter | `remove_edge()` protocol — replaces silent `hasattr()` guard |
| 81 | GraphSnapshot | Portable JSON topology checkpoint |
| 82 | AutonomousDiscoveryLoop × DiscoveryCalibrator | Adaptive cap + interval from community weights |

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

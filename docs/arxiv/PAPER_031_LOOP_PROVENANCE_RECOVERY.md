# PAPER 031: Loop-Provenance Recovery — Automatic Rollback on Circuit Breaker Trip in Autonomous KG Discovery

**CEREBRUM Phase 79**

---

## Abstract

We present **Loop-Provenance Recovery**, an integration between `AutonomousDiscoveryLoop` (Phase 74) and `ProvenanceLedger` (Phase 76) that automatically undoes graph materializations when the circuit breaker fires. Prior to Phase 79, a circuit breaker trip paused future materialization but left already-materialized edges from the failed cycle in the graph. These edges may represent low-quality discoveries that degraded graph precision. Loop-Provenance Recovery addresses this by wiring `LoopConfig.auto_rollback_on_trip=True` to an automatic call to `ProvenanceLedger.rollback_cycle()` at trip time. `CycleRecord.edges_rolled_back` records how many edges were removed, providing observability. The combined system achieves **fail-safe materialization**: if a discovery cycle degrades below quality thresholds, both future materializations pause *and* past materializations are undone — returning the graph to its pre-cycle state.

---

## 1. Motivation: Incomplete Recovery

The Phase 74 circuit breaker addresses the *forward* problem: when approval rate drops below `min_approval_rate`, future materialization pauses. However, the cycle that triggered the trip has already materialized some edges. If those edges are low-quality (e.g., confidences just above threshold on marginally-approved findings), they remain in the graph indefinitely.

In a production deployment with continuous discovery, a single degraded cycle can introduce dozens of noisy edges before the circuit breaker fires. Manual cleanup requires identifying the exact edges from that cycle — a non-trivial task without provenance tracking.

---

## 2. Recovery Protocol

When the circuit breaker trips:

```
1. circuit_breaker_tripped = True → pause future materializations
2. if auto_rollback_on_trip and provenance_ledger is not None:
       rolled_back = provenance_ledger.rollback_cycle(cycle_number, adapter)
       cycle_record.edges_rolled_back = rolled_back
3. log: "Circuit breaker tripped. Rolled back {rolled_back} edges from cycle {N}."
4. sleep(next_interval)
5. Next cycle: check if approval rate has recovered
```

### Prerequisite: ProvenanceLedger Attachment

```python
ledger = ProvenanceLedger(max_batches=500)
research_agent.set_provenance_ledger(ledger)
loop = AutonomousDiscoveryLoop(
    agent=research_agent,
    config=LoopConfig(auto_rollback_on_trip=True),
    auto_approver=approver
)
```

If `provenance_ledger` is `None`, the auto-rollback silently skips (backward-compatible behavior).

---

## 3. CycleRecord Fields

```python
@dataclass
class CycleRecord:
    ...
    circuit_breaker_tripped: bool
    edges_rolled_back: int = 0   # 0 if not tripped or rollback not configured
```

The Studio v2 cycle history panel (Phase 75) renders `edges_rolled_back` in the cycle table, making rollback events visible to operators without requiring log inspection.

---

## 4. GraphAdapter Prerequisite

`rollback_cycle()` calls `adapter.remove_edge(u, v, relation)` for each edge in the cycle's batches. This requires the adapter to implement the Phase 80 `remove_edge()` protocol. All built-in adapters support this. Custom adapters must implement it or raise `NotImplementedError` explicitly.

---

## 5. Failure Modes and Mitigations

| Failure | Behavior | Mitigation |
|---|---|---|
| Partial rollback (adapter exception mid-cycle) | `rollback_cycle` catches exception, logs warning, continues with remaining edges | Idempotent: re-running `rollback_batch` on completed batches is a no-op |
| ProvenanceLedger evicted cycle batches (LRU cap) | Rollback is incomplete — only unevicted batches are removed | Tune `max_batches` to retain at least `circuit_breaker_window` cycles |
| Adapter does not implement `remove_edge` | `NotImplementedError` propagates, trip still recorded | Implement `remove_edge` or disable `auto_rollback_on_trip` |

---

## 6. References

- Phase 74 (PAPER_027): AutonomousDiscoveryLoop circuit breaker
- Phase 76 (PAPER_029): ProvenanceLedger data model and rollback API
- Phase 80 (PAPER_032): GraphAdapter `remove_edge` protocol

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

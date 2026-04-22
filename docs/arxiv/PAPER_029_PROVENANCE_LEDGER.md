# PAPER 029: ProvenanceLedger — Targeted Rollback and Audit for Autonomous Knowledge Graph Materialization

**CEREBRUM Phase 76**

---

## Abstract

We present **ProvenanceLedger**, an audit chain component for CEREBRUM's autonomous knowledge graph discovery pipeline. ProvenanceLedger records every edge materialized by `ResearchAgent.approve()` in structured `EdgeRecord` / `BatchRecord` objects, enabling targeted rollback at two granularities: per-approval-batch (`rollback_batch`) and per-loop-cycle (`rollback_cycle`). Unlike naive graph snapshots, which require full graph comparison to identify removed edges, ProvenanceLedger provides O(1) lookup of edges-to-remove by batch ID or cycle number. The ledger is thread-safe, LRU-capped at `max_batches` to bound memory usage, and requires adapters to implement the Phase 80 `remove_edge()` protocol. ProvenanceLedger is the prerequisite for Loop-Provenance Recovery (Phase 79), which wires circuit breaker trips to automatic cycle rollback.

---

## 1. Motivation: Accountability in Autonomous Materialization

Prior to Phase 76, `ResearchAgent.approve()` called `adapter.add_edge()` with no record of which edges were added in which approval decision. If a batch of findings later proved incorrect (e.g., when the circuit breaker tripped, indicating poor discovery quality), there was no automated way to remove exactly those edges without manual graph inspection.

Two rollback granularities are needed:
1. **Batch rollback**: Remove edges from one specific `approve()` call. Useful when a single finding is later disproved.
2. **Cycle rollback**: Remove all edges materialized in loop cycle N. Useful when the entire cycle is suspect (circuit breaker trip, external validation failure).

---

## 2. Data Model

### EdgeRecord
```python
@dataclass
class EdgeRecord:
    u: str           # source entity
    v: str           # target entity
    relation: str    # edge label
    finding_id: str  # originating ResearchFinding ID
```

### BatchRecord
```python
@dataclass
class BatchRecord:
    batch_id: str              # timestamp + finding hash
    cycle_number: int          # loop cycle that produced this batch
    edges: List[EdgeRecord]
    rolled_back: bool = False
    rolled_back_at: Optional[float] = None
```

### ProvenanceLedger
- `_batches: OrderedDict[str, BatchRecord]` — LRU-ordered dict
- `_cycle_index: Dict[int, List[str]]` — cycle → list of batch_ids

---

## 3. Recording

```python
# Called automatically by ResearchAgent.approve()
ledger.record(batch_id, cycle_number, edges: List[Tuple[str, str, str]])
```

When `max_batches` is reached, the oldest batch is evicted (LRU).

---

## 4. Rollback

### Batch rollback
```python
removed: int = ledger.rollback_batch("batch_20260414_001", adapter)
# Calls adapter.remove_edge(u, v, relation) for each EdgeRecord
# Marks batch as rolled_back=True
```

### Cycle rollback
```python
removed: int = ledger.rollback_cycle(12, adapter)
# Iterates all batches with cycle_number == 12
# Calls rollback_batch for each
```

Both methods return the count of edges removed, which is recorded in `CycleRecord.edges_rolled_back` (Phase 79).

---

## 5. Stats and Inspection

```python
stats = ledger.stats()
# {
#   "total_batches": 42,
#   "active_batches": 38,
#   "rolled_back_batches": 4,
#   "total_edges": 187,
#   "rolled_back_edges": 19,
# }

batches = ledger.recent_batches(n=10)
# List of BatchRecord dicts, newest first
```

---

## 6. REST API

| Endpoint | Method | Description |
|---|---|---|
| `/research/provenance/stats` | GET | Ledger totals |
| `/research/provenance/batches` | GET | Recent batches list (`?n=20`) |
| `/research/provenance/rollback/{batch_id}` | POST | Rollback one batch |
| `/research/provenance/rollback-cycle/{n}` | POST | Rollback all batches from cycle N |

---

## 7. Thread Safety

All read and write operations on `_batches` and `_cycle_index` are protected by a `threading.Lock`. Concurrent `approve()` calls from different threads produce distinct `batch_id` values (based on `time.time_ns()` + `uuid4` suffix) and never collide.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

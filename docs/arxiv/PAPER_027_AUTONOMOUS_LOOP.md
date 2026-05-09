# PAPER 027: AutonomousDiscoveryLoop — Closing the Discover-Validate-Approve-Materialize Loop for Knowledge Graph Self-Improvement

**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 74**

---

## Abstract

We present **AutonomousDiscoveryLoop**, the orchestration component that closes the end-to-end autonomous knowledge graph improvement loop in CEREBRUM. The loop runs `ResearchAgent.scan_once()` on a configurable timer, processes each finding through `AutoApprover` (Phase 71), and applies a **circuit breaker** — a sliding-window approval-rate monitor that pauses materialization when quality degrades below a threshold. A **per-cycle cap** (`max_materializations_per_cycle`) prevents runaway edge insertion during anomalous discovery bursts. A **dry_run** mode allows safe production trials without writing any edges. `AutoApprover` state is checkpointed to disk after every cycle, enabling warm restart. The loop exposes a structured REST API for monitoring, configuration, and lifecycle management, making it suitable for embedding in production CEREBRUM deployments without operator intervention.

---

## 1. Architecture

### 1.1 Loop Lifecycle

```
start() → background thread
    ↓
while running:
    scan_once() → findings
    for finding in findings[:cap]:
        decide = approver.decide(finding)
        if decide == APPROVE: approve(finding)
        if decide == REJECT:  reject(finding)
    record_cycle()
    checkpoint_approver()
    check_circuit_breaker()
    sleep(next_interval)
stop() → graceful shutdown
```

### 1.2 LoopConfig Dataclass

```python
@dataclass
class LoopConfig:
    cycle_interval: float = 300.0
    max_materializations_per_cycle: int = 10
    min_approval_rate: float = 0.5
    circuit_breaker_window: int = 20
    dry_run: bool = False
    auto_rollback_on_trip: bool = False   # Phase 79
    adaptive_tuning: bool = False          # Phase 82
    adaptive_min_cap: int = 1
    adaptive_max_cap: int = 20
    adaptive_min_interval: float = 60.0
    adaptive_max_interval: float = 7200.0
    approver_checkpoint_path: Optional[str] = None
```

### 1.3 CycleRecord Dataclass

```python
@dataclass
class CycleRecord:
    cycle_number: int
    started_at: float
    findings_scanned: int
    materializations: int
    approvals: int
    rejections: int
    reviews: int
    circuit_breaker_tripped: bool
    edges_rolled_back: int = 0    # Phase 79
    effective_cap: int = 0        # Phase 82
```

---

## 2. Circuit Breaker

The circuit breaker computes approval rate over a sliding window of the last N decisions (approvals + rejections):

```
approval_rate = approvals_in_window / (approvals + rejections)_in_window
```

If `approval_rate < min_approval_rate`:
- `circuit_breaker_tripped = True`
- Materialization pauses for the current cycle
- If `auto_rollback_on_trip=True` (Phase 79): `ProvenanceLedger.rollback_cycle()` is called
- Loop continues sleeping; next cycle attempts recovery

The circuit breaker resets when the approval rate recovers above threshold over a fresh window.

---

## 3. Per-Cycle Cap

`max_materializations_per_cycle` acts as a hard upper bound on approved findings per cycle. Even if `AutoApprover` would approve 100 findings in a single scan, only the first N are materialized. This prevents:
- Burst materialization that overwhelms downstream systems
- Runaway edge accumulation during transient high-discovery phases

Combined with `adaptive_tuning` (Phase 82), the cap is dynamically scaled per cycle based on `DiscoveryCalibrator` community weights.

---

## 4. Dry Run Mode

`LoopConfig(dry_run=True)` makes the loop execute all phases (scan, validate, decide) but skips `approve()` / `reject()` calls. The loop records what *would* have been materialized in `CycleRecord`, enabling safe production trials to measure expected impact before enabling writes.

---

## 5. REST API

| Endpoint | Method | Description |
|---|---|---|
| `/research/loop/start` | POST | Start loop (idempotent) |
| `/research/loop/stop` | POST | Graceful stop |
| `/research/loop/status` | GET | Running state, cycle history, approval rate, circuit breaker |
| `/research/loop/configure` | POST | Partial update: `cycle_interval`, `max_materializations_per_cycle`, `min_approval_rate`, `dry_run`, `auto_rollback_on_trip`, `adaptive_tuning` |

---

## 6. References

- [dean2008mapreduce] Dean, J. & Ghemawat, S. (2008). MapReduce: Simplified data processing on large clusters. *CACM*, 51(1), 107–113.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
## Acknowledgments

The author gratefully acknowledges the use of Claude (Anthropic) as a research assistant throughout this work. Claude assisted with mathematical formalization, code generation, manuscript preparation, and technical writing. All conceptual contributions, architectural decisions, experimental design, and intellectual claims are solely the author's.

**Reviewed on**: May 2, 2026 for version v2.51.0

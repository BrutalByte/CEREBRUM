# Five Fault-Tolerance Patterns for Production Knowledge Graph Reasoning Servers

**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)  
**Affiliations**: Independent Researcher · Anthropic  
**Status**: v2.24.0 (Phase 112 (Sleep-Phase Consolidation) COMPLETE)
**Date**: April 2026

---

### Abstract
Production Knowledge Graph reasoning servers face five distinct failure classes: traversal failures mid-hop, persistence write failures during query logging, stream interruptions visible to connected clients, process spawning failures during community detection, and state loss across server restarts. We document five corresponding fault-tolerance patterns implemented in CEREBRUM v2.24.0 (Phases 56–57): **partial-result HTTP 200 degradation** (`QueryResponse.partial`, `_partial_paths`), **write-failure isolation** (QueryLog, Engram, GlobalRebalancer worker), **streaming error signalling** (terminal NDJSON error chunk), **ProcessPoolExecutor sequential fallback** (`best_of_n_dscf`), and **durable Engram persistence** (save/load/two-tier startup). Together, these patterns guarantee that no single failure class can crash a running CEREBRUM server, corrupt an in-flight query, or silently drop accumulated reasoning experience. Each pattern is backward-compatible and adds no new required parameters to existing APIs.

### 1. Introduction
Production distributed systems face a fundamental asymmetry: failures are rare but their consequences are disproportionate. A single traversal crash that returns HTTP 500 may abort a client's multi-step reasoning workflow. A disk-full condition that propagates from a log write to a query response degrades system availability unnecessarily. A stream client that receives a TCP disconnect mid-stream has no way to distinguish a deliberate completion from a server crash.

Lamport \cite{lamport1978} established that distributed systems cannot distinguish a crashed component from a slow one without timeouts. Bernstein & Goodman \cite{bernstein1983} formalized concurrency control as the discipline of maintaining database consistency in the presence of concurrent failures. CEREBRUM's fault tolerance architecture applies these principles to the specific failure modes of a KG reasoning server: it identifies the five failure classes, bounds their blast radius, and provides clients with the information they need to recover gracefully.

This paper documents the five failure classes, the pattern applied to each, and the invariant each pattern preserves.

### 2. Pattern 1 — Partial-Result HTTP 200 (Phase 56)

**Failure class**: Traversal failure mid-hop.

**Scenario**: `BeamTraversal.traverse()` completes hops 1–3 successfully but raises an exception at hop 4. Without intervention, the exception propagates to the FastAPI route handler, which returns HTTP 500. The client receives no answers, even though three hops of valid reasoning were completed.

**Pattern**: `BeamTraversal` maintains a `_partial_paths: List[Path]` list. After each hop completes, the current best paths are checkpointed into `_partial_paths`. If a later hop raises, the exception is caught by the route handler and `_partial_paths` is returned as the answer set.

`QueryResponse` gains two optional fields:
- `partial: bool = False` — set to `True` when the response contains partial rather than full results.
- `error: Optional[str] = None` — the exception message, for client-side logging and retry decisions.

The route handler returns HTTP 200 in both cases. Clients distinguish partial from full results by checking `response.partial`, not by parsing HTTP status codes.

**Invariant preserved**: A traversal failure never returns fewer results than the last completed hop. Clients always receive the best available answer, with explicit metadata indicating whether the result is complete.

### 3. Pattern 2 — Write Failure Isolation (Phase 56)

**Failure class**: Persistence write failures during query logging.

**Scenario**: `QueryLog.record()` or `Engram.record()` raises `OSError` (disk full) or `MemoryError` (OOM) during a live query. Without isolation, the exception propagates from the write call to the query route handler, converting a persistence failure into a query failure.

**Pattern**: All write calls in the hot query path are wrapped in `try/except Exception`:

```python
try:
    query_log.record(seeds, answers, rel_seq)
except Exception as exc:
    logger.warning("QueryLog write failed: %s", exc)
```

The exception is logged at WARNING (not ERROR, since it does not affect the query result) and swallowed. The query response proceeds normally.

A parallel isolation is applied to `GlobalRebalancer`: `_rebalance_worker_inner()` is extracted as a separate method containing the actual rebalance work. The outer `_rebalance_worker()` calls `_rebalance_worker_inner()` inside `try/except` and logs any exception at ERROR. This prevents an exception in the rebalance algorithm from crashing the rebalancer thread silently — it is logged and the thread remains alive for the next scheduled rebalance.

**Invariant preserved**: A write failure never degrades query availability. The persistence layer is a best-effort side channel, not a reliability dependency for core reasoning.

### 4. Pattern 3 — Stream Error Signalling (Phase 57)

**Failure class**: Stream interruptions visible to connected clients.

**Scenario**: A client connects to `/query/stream` and begins receiving NDJSON hop chunks. The traversal raises mid-stream. Without intervention, the HTTP response stream closes with a TCP disconnect. The client cannot distinguish this from a deliberate end-of-stream.

**Pattern**: The async generator that implements `/query/stream` wraps the traversal loop in `try/except`:

```python
try:
    async for chunk in traversal.traverse_stream(seeds):
        yield json.dumps(chunk) + "\n"
except Exception as exc:
    yield json.dumps({
        "status": "error",
        "partial": True,
        "error": str(exc)
    }) + "\n"
```

The terminal error chunk is a valid NDJSON line. Clients that parse the stream line-by-line can detect failure by checking `chunk["status"] == "error"` on the final line, without inspecting HTTP trailers or catching TCP exceptions.

**Invariant preserved**: Stream clients always receive an explicit terminal signal, whether the traversal completed normally or failed. The distinction between completion and failure is always observable from stream content alone.

### 5. Pattern 4 — ProcessPoolExecutor Sequential Fallback (Phase 57)

**Failure class**: Process spawning failures during parallel community detection.

**Scenario**: `best_of_n_dscf` uses a `ProcessPoolExecutor` to run multiple DSCF community detection trials in parallel. On Windows hosts with restricted paging file sizes, or on containers with `fork()` restrictions, `ProcessPoolExecutor` submission raises `BrokenProcessPool` or `concurrent.futures.BrokenExecutor`. Without fallback, server startup fails.

**Pattern**: The `ProcessPoolExecutor` block is wrapped in `try/except`:

```python
try:
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(dscf_communities, configs))
except Exception as exc:
    logger.warning("ProcessPoolExecutor failed (%s); falling back to sequential", exc)
    results = [dscf_communities(cfg) for cfg in configs]
```

Sequential execution is slower but produces identical results. The WARNING log makes the fallback observable in the `/logs` endpoint without surfacing as an error to the client.

**Invariant preserved**: Server startup succeeds on any host that can run Python, regardless of process spawning constraints.

### 6. Pattern 5 — Durable Engram Persistence (Phase 57)

**Failure class**: State loss across server restarts.

**Scenario**: The `Engram` accumulates relation-sequence success counts during a server run. On planned or unplanned shutdown, these counts are lost. The next server run starts with a cold cache, discarding all learned reasoning experience.

**Pattern**: `Engram` implements `save(path)` / `load(path)` / `save_if_path(path)`:

- `save(path)`: Serializes `_counts` as `[[seq_tuple, count], ...]` inside a `{"version": 1, ...}` JSON envelope.
- `load(path)`: Deserializes and restores `_counts`; recomputes `_max_count` from the loaded values.
- `save_if_path(path)`: Null-safe wrapper — silently returns if `path is None`.

The FastAPI lifespan context manager integrates persistence:

```python
@asynccontextmanager
async def lifespan(app):
    # Startup: two-tier warm-up
    engram.load(cache_path)               # Tier 1: saved JSON
    query_log.replay_into_cache(engram)   # Tier 2: QueryLog entries
    try:
        yield
    finally:
        engram.save_if_path(cache_path)   # Shutdown: persist counts
```

The `try/finally` in the lifespan guarantees that `save_if_path` is called even on unhandled exceptions during the application lifetime. Save failures are isolated with `try/except` and logged at WARNING.

**Invariant preserved**: No productive reasoning trace is lost across planned restarts. On unplanned restarts, at most the traces since the last explicit `save()` call are lost — bounded by the QueryLog replay that closes this gap at the next startup.

### 7. Fault-Tolerance Taxonomy

| Pattern | Failure Class | Blast Radius Without Pattern | Invariant Preserved |
|---|---|---|---|
| Partial-Result HTTP 200 | Traversal failure mid-hop | HTTP 500, zero answers | Best available answers always returned |
| Write Failure Isolation | Disk-full / OOM during logging | HTTP 500, query aborted | Write failures never degrade query availability |
| Stream Error Signalling | Traversal failure mid-stream | Silent TCP disconnect | Stream failure always explicitly signalled |
| ProcessPoolExecutor Fallback | Process spawn failure at startup | Server fails to start | Startup succeeds on any Python-capable host |
| Engram Persistence | Server restart | All learned patterns lost | Productive traces survive planned restarts |

The five patterns are orthogonal: each addresses a distinct failure class and can be applied independently. Together, they provide defense-in-depth against the full set of operational failure modes observed in production KG reasoning deployments.

### 8. Conclusion
Fault tolerance in production systems is not a single feature but a taxonomy of patterns, each matched to a specific failure class and preserving a specific invariant. The five patterns documented here — partial-result degradation, write-failure isolation, stream error signalling, executor fallback, and durable cache persistence — together ensure that no single operational failure can crash a CEREBRUM server, corrupt an in-flight query, or silently discard accumulated reasoning experience. All five are backward-compatible with existing APIs and add no new required configuration. They represent the engineering discipline that distinguishes a research prototype from a production-ready system.

---
**References**
1. Lamport, L. (1978). Time, Clocks, and the Ordering of Events in a Distributed System. Communications of the ACM, 21(7), 558–565.
2. Bernstein, P. A., & Goodman, N. (1983). Multiversion Concurrency Control — Theory and Algorithms. ACM Transactions on Database Systems, 8(4), 465–483.
3. Vaswani, A., et al. (2017). Attention is All You Need. NIPS.

---
**Reviewed on**: April 21, 2026 for version v2.24.0

# PAPER 032: GraphAdapter remove_edge Protocol — A Formal Edge-Removal Contract for Knowledge Graph Adapters

**CEREBRUM Phase 80**

---

## Abstract

We present the **GraphAdapter `remove_edge()` Protocol**, a Phase 80 change that promotes edge removal from an ad-hoc capability to a formally defined method in the `GraphAdapter` base class. Prior to Phase 80, `ProvenanceLedger.rollback_batch()` checked for edge-removal capability via `hasattr(adapter, "remove_edge")` and silently skipped rollback if the method was absent. This silent-skip pattern allowed custom adapter subclasses to silently fail rollback without any error signal. Phase 80 removes the guard and defines `remove_edge(u, v, relation)` as a protocol method on `GraphAdapter` — non-abstract, raising `NotImplementedError` by default, overridden in all six built-in adapters. Custom adapters must explicitly override the method or accept that rollback will raise `NotImplementedError`, making the failure visible. This is a breaking change for custom adapters that previously relied on the silent-skip behavior.

---

## 1. Motivation: Silent Failure in Rollback

The ProvenanceLedger (Phase 76) enables targeted rollback of materialized edges. Rollback is only useful if the underlying adapter can actually remove edges. The original implementation used a defensive `hasattr()` check:

```python
# Pre-Phase 80 (fragile)
if hasattr(adapter, "remove_edge"):
    adapter.remove_edge(u, v, relation)
else:
    logger.warning("Adapter does not support remove_edge; skipping rollback")
```

This pattern has two failure modes:
1. **Silent no-op**: A custom adapter implementing a different interface (e.g., `delete_edge`) would fail the `hasattr` check, silently skip rollback, and report success.
2. **Untestable absence**: Tests that mock `adapter` without adding `remove_edge` would pass, masking the missing implementation.

---

## 2. Protocol Definition

The Phase 80 change to `core/graph_adapter.py`:

```python
class GraphAdapter(ABC):
    # ... existing abstract methods ...

    def remove_edge(self, u: str, v: str, relation: str) -> None:
        """Remove edge (u, v, relation) from the graph.

        All built-in adapters override this method.
        Custom subclasses must override it to enable ProvenanceLedger rollback.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement remove_edge(). "
            "Override this method to enable ProvenanceLedger rollback."
        )
```

The method is **non-abstract** (does not use `@abstractmethod`) to avoid breaking existing custom subclasses on import. The `NotImplementedError` is raised at call time, not at subclass definition time.

---

## 3. Built-in Adapter Implementations

All six built-in adapters implement `remove_edge`:

| Adapter | Implementation |
|---|---|
| `NetworkXAdapter` | `G.remove_edge(u, v)` with edge matching by relation key |
| `Neo4jAdapter` | Cypher `MATCH (u)-[r:RELATION]->(v) DELETE r` |
| `RDFAdapter` | `graph.remove((subject, predicate, obj))` |
| `CSVAdapter` | In-memory edge list filter + optional file flush |
| `StreamAdapter` | Delegates to wrapped `base_adapter.remove_edge()` |
| `RemoteCerebrumAdapter` | `DELETE /edges` REST call with signature |

---

## 4. Migration Guide for Custom Adapters

Any custom `GraphAdapter` subclass must now explicitly handle edge removal:

```python
# Option A: Implement it
class MyAdapter(GraphAdapter):
    def remove_edge(self, u: str, v: str, relation: str) -> None:
        self._edges = [(a, b, r) for (a, b, r) in self._edges
                       if not (a == u and b == v and r == relation)]

# Option B: Declare it unsupported
class MyAdapter(GraphAdapter):
    def remove_edge(self, u: str, v: str, relation: str) -> None:
        raise NotImplementedError("MyAdapter does not support edge removal.")
```

Option B is equivalent to the previous silent-skip behavior, but now raises explicitly rather than silently doing nothing.

---

## 5. ProvenanceLedger Changes

```python
# Post-Phase 80 (no hasattr guard)
def rollback_batch(self, batch_id: str, adapter: GraphAdapter) -> int:
    batch = self._batches[batch_id]
    for edge in batch.edges:
        adapter.remove_edge(edge.u, edge.v, edge.relation)  # raises if not implemented
    batch.rolled_back = True
    return len(batch.edges)
```

If `adapter.remove_edge()` raises, the exception propagates to the caller. `ProvenanceLedger` does not catch it — callers are expected to handle `NotImplementedError` if they choose to continue on partial rollback.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.24.0

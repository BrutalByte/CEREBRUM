# PAPER 033: GraphSnapshot — Portable JSON Topology Persistence for Knowledge Graph Recovery

**CEREBRUM Phase 81**

---

## Abstract

We present **GraphSnapshot**, a portable JSON-based topology persistence mechanism for CEREBRUM Knowledge Graph adapters. Prior to Phase 81, graph state was saved via Python pickle, which is fragile across Python versions, adapter class renames, and dependency updates. `GraphSnapshot.save(adapter, path)` serializes the full edge list (source, target, relation, weight, metadata) to a portable JSON file that survives class changes. `GraphSnapshot.restore(path, adapter, skip_existing=True)` re-adds only new edges — idempotent and safe to run on every pod restart. `GraphSnapshot.diff(path_a, path_b)` computes the edge delta between two snapshots, enabling change auditing between any two checkpoints. GraphSnapshot is the recommended mechanism for disaster recovery in Kubernetes deployments, replacing manual graph CSV exports and pickle-based state files.

---

## 1. Motivation: Fragile Persistence

CEREBRUM's `save_state()` / `load_state()` (Phase 55) persisted graph topology via `pickle.dump()`. Pickle is convenient but fragile:

1. **Class renames**: If `GraphAdapter` or `Edge` class names change (as happened with the Parallax → CEREBRUM rename), pickled files are unloadable.
2. **Python version changes**: Pickle protocol versions differ across Python releases. A file pickled on Python 3.10 may fail on 3.12.
3. **Dependency changes**: Pickling `networkx.DiGraph` embeds the networkx version; a dependency upgrade can break loading.
4. **No diff semantics**: Pickle files are binary blobs; computing what changed between two states requires unpickling both and diffing in Python.

---

## 2. File Format

GraphSnapshot files are UTF-8 JSON with a standardized schema:

```json
{
  "cerebrum_snapshot": "1.0",
  "saved_at": "2026-04-14T12:00:00Z",
  "adapter_type": "NetworkXAdapter",
  "node_count": 21,
  "edge_count": 30,
  "edges": [
    {
      "u": "marie_curie",
      "v": "radium",
      "relation": "discovered",
      "weight": 0.95,
      "metadata": {"source": "wikipedia", "confidence": 0.99}
    },
    ...
  ]
}
```

The `metadata` field captures any arbitrary per-edge metadata written at ingest time (confidence, source, provenance batch_id, etc.).

---

## 3. API

### Save

```python
from core.persistence import GraphSnapshot

GraphSnapshot.save(adapter, "/data/snapshots/graph_2026-04-14.json")
# Writes full edge list to JSON
```

### Restore

```python
result = GraphSnapshot.restore(
    "/data/snapshots/graph_2026-04-14.json",
    adapter,
    skip_existing=True    # default: don't re-add edges already in the graph
)
print(f"Added: {result['added']}, Skipped: {result['skipped']}")
```

`skip_existing=True` makes restore idempotent — calling it multiple times on the same adapter is safe. `skip_existing=False` forces all edges to be re-added (useful after a full graph wipe).

### Diff

```python
diff = GraphSnapshot.diff(
    "/data/snapshots/before.json",
    "/data/snapshots/after.json"
)
# {
#   "edge_delta": +12,
#   "added_edges": [...],
#   "removed_edges": [...],
#   "node_delta": +3,
# }
```

Diff does not require a live adapter — it compares two snapshot files directly.

---

## 4. Integration with AutonomousDiscoveryLoop

For disaster recovery in production deployments:

```bash
# Cron: save snapshot every hour
0 * * * * python -c "
from core.persistence import GraphSnapshot
from core.cerebrum import CerebrumGraph
graph = CerebrumGraph.load_from_db()
GraphSnapshot.save(graph.adapter, f'/data/snapshots/graph_{date}.json')
"
```

On pod restart:
```python
result = GraphSnapshot.restore("/data/snapshots/graph_latest.json", adapter)
```

Pairs with `ProvenanceLedger.rollback_cycle()` for fine-grained recovery: if the loop materialized bad edges after the last snapshot, roll them back via provenance rather than restoring the full snapshot.

---

## 5. Comparison to Alternatives

| Mechanism | Portable | Diffable | Idempotent restore | Survives class changes |
|---|---|---|---|---|
| `pickle` | No | No | No | No |
| CSV export | Yes | Manual | Partial | Yes |
| **GraphSnapshot** | Yes | Yes | Yes | Yes |
| Neo4j native backup | Yes | No | Yes | N/A |

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

# CEREBRUM Migration Guide

This guide covers what operators and developers need to **do** (not just what changed) when upgrading between major versions. For a list of *what* changed, see `CHANGELOG.md`.

---

## v1.7.0 → v1.7.1 (Phase 32: Federated Reasoning)

### Required actions: GraphAdapter subclasses
If you have custom `GraphAdapter` implementations outside the core library, you should implement the new `get_reasoning_branches` method. While it has a base implementation that returns an empty list, providing a real implementation enables delegated reasoning.

```python
# In your custom adapter:
def get_reasoning_branches(self, seed_id, context_embedding=None, max_hop=2, beam_width=5):
    # Optional: run local reasoning and return serialized TraversalPaths
    return []
```

### Recommended changes

**1. Switch to `DistributedBeamTraversal` for Federated deployments**

If you are using `FederatedAdapter` with `RemoteCerebrumAdapter`, replace `BeamTraversal` with `DistributedBeamTraversal` to enable multi-hop delegation. This significantly reduces network overhead by fetching full reasoning branches instead of individual neighbors.

```python
# Before
from reasoning.traversal import BeamTraversal
traversal = BeamTraversal(adapter=federated, ...)

# After
from reasoning.distributed_traversal import DistributedBeamTraversal
traversal = DistributedBeamTraversal(adapter=federated, ...)
```

**2. Update Remote Node API Keys**

The new `/traverse` endpoint requires the `query` scope. Ensure your JWT tokens for remote nodes include this scope.

---

## v1.1.0 → v1.7.0 (Phases 21–30: Pipeline & Bridge)

### Required actions: none
All four Phase 20 fixes use backward-compatible defaults. No existing code breaks.

### Recommended changes

**1. Enable Query Snapshot Isolation (Hole 5)**

If your deployment uses the `GlobalRebalancer` for live graph updates alongside concurrent queries, enable snapshot isolation explicitly:

```python
# Before (v1.0.0 — no snapshot)
traversal = BeamTraversal(adapter=adapter, csa_engine=csa, ...)

# After (v1.1.0 — add snapshot at query start)
csa.set_query_snapshot(adapter.community_map)
traversal = BeamTraversal(adapter=adapter, csa_engine=csa, ...)
```

The `api/server.py` already does this automatically for REST queries.

**2. Per-community CSA parameters for homogeneous graphs (Hole 6)**

If your graph has tightly-clustered domains (proteins, legal clauses, financial instruments) where intra-community similarity is uniformly high, add parameter overrides for those communities:

```python
# Identify homogeneous communities (avg intra-community S_C > 0.85)
# Then reduce β and increase α/γ for those communities:
csa = CSAEngine(
    community_params={3: (0.5, 0.15, 0.25, 0.05, 0.05, 0.0)},
    ...
)
```

**3. Path-preserving hold-out is now the default**

`InferenceValidator` now defaults to `path_preserving=True`. On dense graphs this is a no-op. On sparse graphs (avg degree < 3) this will improve recall estimates. No code changes needed unless you specifically want the old behavior:

```python
# To restore v1.0.0 behavior (not recommended):
validator = InferenceValidator(adapter, traversal, path_preserving=False)
```

---

## v0.4.0 → v1.0.0 (Phase 19: Production Hardening)

### Required actions

**1. Namespace isolation for mixed-modality deployments**

If you use both `IngestionPipeline` and `SignalEncoder` in the same graph, you must now add namespace prefixes to prevent entity collisions:

```python
# Before (v0.4.0 — collision risk)
pipeline = IngestionPipeline(adapter)
encoder  = SignalEncoder(entity_dim=128)

# After (v1.0.0 — isolated namespaces)
pipeline = IngestionPipeline(adapter, namespace="text")
encoder  = SignalEncoder(entity_dim=128, namespace="signal")
```

If your graph contains entities with the same name in both text and signal sources, they will now appear as separate nodes (`text:Name` and `signal:Name`). To intentionally merge them, use `entity_dedup_map={"signal:Name": "text:Name"}` in `IngestionPipeline`.

**2. Wire the Zombie Bridge hook if using GlobalRebalancer + BridgeTwinEngine**

```python
# Before (v0.4.0 — stale bridges after rebalance)
rebalancer = GlobalRebalancer(adapter, q_drift_threshold=0.05)

# After (v1.0.0 — bridges stay consistent)
rebalancer = GlobalRebalancer(adapter, q_drift_threshold=0.05, bridge_engine=bridge_engine)
```

### Recommended changes

**3. Enable warm-start for probabilistic traversal on sparse graphs**

```python
# Before
traversal = BeamTraversal(adapter, csa, probabilistic=True)

# After — reduce cold-start variance 85%
traversal = BeamTraversal(adapter, csa, probabilistic=True, warm_start_strength=5.0)
```

**4. Enable causal flood protection for untrusted event streams**

```python
# Before
discretizer = STDPDiscretizer(w_threshold=0.5, n_min=5)

# After — adversarial hardening
discretizer = STDPDiscretizer(w_threshold=0.5, n_min=5, min_causal_span=1.0, use_chi_squared=True)
```

---

## v0.3.0 → v0.4.0 (Phase 18: v0.4 Horizon)

### New required environment variable (if using API auth)
```bash
export CEREBRUM_JWT_SECRET="your-secret-key"
```

The API server now requires JWT authentication on all endpoints. Anonymous access can be re-enabled for development:
```bash
export CEREBRUM_ALLOW_ANONYMOUS=true
```

### IngestionPipeline replaces direct adapter.add_edge for text ingest
```python
# Before (v0.3.0)
for subj, pred, obj in triples:
    adapter.add_edge(subj, obj, pred)

# After (v0.4.0) — preferred for text data
pipeline = IngestionPipeline(adapter)
for subj, pred, obj in triples:
    pipeline.process(subj, pred, obj, confidence=0.9)
```

Direct `add_edge` still works — the pipeline is additive.

### LLM bridge import path changed
```python
# Before
from cerebrum.bridge import generate

# After
from llm_bridge import generate, GenerationResult
```

---

## v0.2.0 → v0.3.0 (Phase 10–11: Production Hardening + Streaming)

### igraph / leidenalg dependency removed
The native Leiden reimplementation (`core/leiden_native.py`) is now the default backend. Remove these from your environment if installed:

```bash
pip uninstall leidenalg python-igraph igraph
```

The native implementation is API-compatible; no code changes are needed.

### JWT authentication added to API
See v0.3.0 → v0.4.0 section above (authentication was introduced in v0.3.0).

### StreamAdapter replaces manual polling loops
```python
# Before (v0.2.0 — polling)
while True:
    event = get_next_event()
    adapter.add_edge(event.subject, event.object, event.predicate)
    time.sleep(0.1)

# After (v0.3.0 — StreamAdapter)
stream = StreamAdapter(base_adapter=adapter, window_size=10_000)
stream.push_event(subject, predicate, object, timestamp, weight)
```

---

## v0.1.0 → v0.2.0 (Phase 6–9: Federated)

No breaking changes. All Phase 6–9 features (FederatedAdapter, HolographicIndex, Handshake) are additive and do not affect single-graph deployments.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

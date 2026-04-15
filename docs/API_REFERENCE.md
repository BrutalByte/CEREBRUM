# CEREBRUM REST API Reference

**Base URL**: `http://localhost:8200`
**API Version**: v2.20.1
**Authentication**: JWT Bearer token (all endpoints except `/health`)

---

## High-Level API (`CerebrumGraph`)

While the REST API provides remote access, the primary programmatic entry point for local integration is the `CerebrumGraph` class in `core/cerebrum.py`.

### `CerebrumGraph` Lifecycle

1.  **Initialize**: `graph = CerebrumGraph.from_kb("path/to/kb.csv", embeddings="sentence")`
2.  **Complete (Optional)**: `graph.complete([InverseRule(), CompositionRule()])`
3.  **Enhance (Optional)**: `graph.enhance([GraphBridgeEngine()])`
4.  **Build**: `graph.build(cache_dir="cache/", community_engine="dscf")`
5.  **Query**: `answers = graph.query(["start_node"], top_k=10)`

---

## Authentication

All endpoints (except `GET /health`) require a JWT Bearer token:

```
Authorization: Bearer <jwt_token>
```

Tokens are generated using HMAC-SHA256 with the `CEREBRUM_JWT_SECRET` environment variable. Token lifetime defaults to 3600 seconds; configure via `CEREBRUM_JWT_TTL`.

**Error response (401 Unauthorized):**
```json
{"detail": "Missing or invalid authorization token"}
```

---

## Endpoints

### Core Reasoning

#### `POST /query`
Execute a multi-hop reasoning query against the loaded graph.

**Request body:**
```json
{
    "entity": "Marie Curie",
    "max_hops": 3,
    "beam_width": 5,
    "top_k": 10,
    "probabilistic": false,
    "warm_start_strength": 0.0,
    "community_snapshot": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `entity` | string | required | Starting entity for traversal |
| `max_hops` | int | 3 | Maximum reasoning depth |
| `beam_width` | int | 5 | Number of candidates per hop |
| `top_k` | int | 10 | Number of answer paths to return |
| `probabilistic` | bool | false | Enable Bayesian beam search |
| `warm_start_strength` | float | 0.0 | First-hop Beta prior seeding (Phase 19) |
| `community_snapshot` | bool | true | Enable query snapshot isolation (Phase 20) |

**Response (200 OK):**
```json
{
    "query_entity": "Marie Curie",
    "answers": [
        {
            "entity": "Radioactivity",
            "score": 0.847,
            "path": ["Marie Curie", "discovered", "Polonium", "property_of", "Radioactivity"],
            "path_confidence": 0.823,
            "confidence_interval": [0.78, 0.87],
            "communities": [2, 2, 5],
            "hmac": "sha256:a3f4b2..."
        }
    ],
    "traversal_ms": 6.3,
    "hops_explored": 847,
    "snapshot_id": "snap_1743000000",
    "partial": false,
    "error": null
}
```

**Fault-tolerance fields (Phase 56):**

| Field | Type | Description |
|---|---|---|
| `partial` | bool | `true` when the traversal raised an unrecoverable exception mid-execution. The response contains whatever paths were collected in `_partial_paths` before the failure. HTTP status remains 200 so clients can still consume partial results. |
| `error` | string \| null | Exception message when `partial=true`; `null` on successful traversal. |

**curl example:**
```bash
curl -X POST http://localhost:8200/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entity": "Marie Curie", "max_hops": 3}'
```

---

#### `GET /query`
Simple GET-style query for integrations that cannot send POST bodies.

**Query parameters:** `entity` (required), `max_hops` (default: 3), `top_k` (default: 10)

**Response:** Same schema as `POST /query`.

---

### Autonomous Reasoning

#### `POST /hypothesize`
Generate hypothesis proposals using the HypothesisEngine. Multi-path abductive reasoning identifies candidate edges that are strongly implied by indirect evidence but not yet present in the graph.

**Request body:**
```json
{
    "entity": "BRCA1",
    "max_hops": 3,
    "top_k": 10,
    "min_support_paths": 2
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `entity` | string | required | Anchor entity for abductive reasoning |
| `max_hops` | int | 3 | Depth of supporting path search |
| `top_k` | int | 10 | Maximum number of hypotheses to generate |
| `min_support_paths` | int | 2 | Minimum supporting paths required for a hypothesis |

**Response (200 OK):**
```json
{
    "hypotheses": [
        {
            "id": "hyp_001",
            "source": "BRCA1",
            "relation": "associated_with",
            "target": "Ovarian_Cancer",
            "confidence": 0.83,
            "supporting_paths": 4,
            "status": "PENDING"
        }
    ],
    "generated_at": 1743000000,
    "engine_version": "2.0.1"
}
```

---

#### `POST /hypothesize/materialize`
Materialize approved hypotheses as real edges in the graph. Only hypotheses with `status: APPROVED` are materialized.

**Request body:**
```json
{
    "hypothesis_ids": ["hyp_001", "hyp_003"]
}
```

**Response (200 OK):**
```json
{
    "materialized": ["hyp_001", "hyp_003"],
    "skipped": [],
    "edges_added": 2
}
```

---

#### `GET /hypothesize/status`
Return the current status of all hypothesis proposals in the engine's working set.

**Response (200 OK):**
```json
{
    "total": 12,
    "pending": 8,
    "approved": 3,
    "rejected": 1,
    "materialized": 2,
    "hypotheses": [
        {
            "id": "hyp_001",
            "source": "BRCA1",
            "target": "Ovarian_Cancer",
            "confidence": 0.83,
            "status": "MATERIALIZED"
        }
    ]
}
```

---

#### `POST /research/scan`
Trigger the ResearchAgent to scan the graph for missing links. The agent identifies structural signatures that typically indicate an unrecorded edge and produces a ranked list of candidate findings for human review.

**Request body:**
```json
{
    "scope": "full",
    "max_findings": 50,
    "min_confidence": 0.6
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `scope` | string | `"full"` | Scan scope: `"full"` or entity ID for local scan |
| `max_findings` | int | 50 | Maximum findings to surface |
| `min_confidence` | float | 0.6 | Minimum confidence threshold for findings |

**Response (202 Accepted):**
```json
{
    "scan_id": "scan_1743000000",
    "status": "running",
    "estimated_duration_s": 12
}
```

Poll `GET /research/scan/{scan_id}` for completion. When complete, findings are available via `GET /hypothesize/status`.

---

#### `POST /research/approve/{id}`
Approve a specific research finding. Approved findings are queued for materialization.

**Path parameter:** `id` — finding identifier (e.g., `hyp_005`)

**Response (200 OK):**
```json
{
    "id": "hyp_005",
    "status": "APPROVED",
    "approved_at": 1743000000
}
```

---

#### `POST /research/reject/{id}`
Reject a specific research finding. Rejected findings are retained in the audit log but will not be materialized.

**Path parameter:** `id` — finding identifier

**Response (200 OK):**
```json
{
    "id": "hyp_005",
    "status": "REJECTED",
    "rejected_at": 1743000000
}
```

---

#### `GET /research/status`
Return the current status of the ResearchAgent (running / idle) and summary statistics.

**Response (200 OK):**
```json
{
    "status": "idle",
    "last_scan_id": "scan_1743000000",
    "last_scan_at": 1743000000,
    "total_findings": 12,
    "pending_review": 5
}
```

---

#### `POST /research/start`
Start the ResearchAgent background worker. The agent continuously monitors the graph for structural gap signatures and surfaces findings for human review.

**Request body:** Empty `{}` or optional `{"interval_s": 300}` to configure the polling interval.

**Response (200 OK):**
```json
{"started": true, "interval_s": 300}
```

---

#### `POST /research/stop`
Stop the ResearchAgent background worker.

**Response (200 OK):**
```json
{"stopped": true}
```

---

#### `GET /research/findings`
Return the current list of findings surfaced by the ResearchAgent, with optional status filter.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `status` | string | null | Filter by status: `PENDING`, `APPROVED`, `REJECTED`, `MATERIALIZED` |
| `limit` | int | 50 | Maximum findings to return |

**Response (200 OK):**
```json
{
    "findings": [
        {
            "id": "hyp_007",
            "source": "TP53",
            "relation": "regulates",
            "target": "MDM2",
            "confidence": 0.79,
            "supporting_paths": 3,
            "status": "PENDING",
            "discovered_at": 1743000000
        }
    ],
    "total": 12
}
```

---

#### `POST /research/validate`
Submit a batch of research findings to the ExternalValidator. Equivalent to `POST /validate/proposals` but takes finding IDs rather than hypothesis IDs.

**Request body:**
```json
{
    "finding_ids": ["hyp_007", "hyp_009"],
    "sources": ["pubmed", "arxiv"],
    "max_results_per_source": 5
}
```

**Response (200 OK):** Same schema as `POST /validate/proposals`.

---

#### `POST /validate/proposals`
Submit hypothesis proposals to the ExternalValidator for literature cross-referencing. The validator searches PubMed, ClinicalTrials.gov, arXiv, and OpenAlex and returns corroboration evidence for each proposal.

**Request body:**
```json
{
    "proposal_ids": ["hyp_001", "hyp_003"],
    "sources": ["pubmed", "clinicaltrials", "arxiv", "openalex"],
    "max_results_per_source": 5
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `proposal_ids` | list[string] | required | Hypothesis IDs to validate |
| `sources` | list[string] | all | External databases to query |
| `max_results_per_source` | int | 5 | Maximum literature hits per source per proposal |

**Response (200 OK):**
```json
{
    "validations": [
        {
            "proposal_id": "hyp_001",
            "source": "pubmed",
            "hits": 3,
            "top_result": {
                "title": "BRCA1 mutations and ovarian cancer risk...",
                "pmid": "12345678",
                "relevance_score": 0.91
            },
            "overall_support": "CORROBORATED"
        }
    ],
    "validated_at": 1743000000
}
```

---

### Autonomous Discovery Loop

#### `POST /research/loop/start`
Start the autonomous discovery loop (idempotent — safe to call if already running).

**Response (200 OK):** `LoopStatusResponse` — see `GET /research/loop/status`.

---

#### `POST /research/loop/stop`
Stop the autonomous discovery loop gracefully. In-progress cycle completes before stopping.

**Response (200 OK):** `LoopStatusResponse`

---

#### `GET /research/loop/status`
Return a full health snapshot of the autonomous discovery loop.

**Response (200 OK):**
```json
{
    "running": true,
    "cycle_interval": 300.0,
    "max_materializations_per_cycle": 10,
    "min_approval_rate": 0.5,
    "circuit_breaker_window": 20,
    "dry_run": false,
    "auto_rollback_on_trip": false,
    "adaptive_tuning": false,
    "adaptive_effective_interval": null,
    "circuit_breaker_tripped": false,
    "current_approval_rate": 0.75,
    "total_cycles": 14,
    "total_approved": 42,
    "total_rejected": 18,
    "total_review": 5,
    "total_edges_added": 42,
    "started_at": 1743000000,
    "last_cycle_at": 1743005200,
    "recent_cycles": [
        {
            "cycle_number": 14,
            "started_at": 1743005200,
            "duration_seconds": 4.2,
            "findings_seen": 5,
            "auto_approved": 3,
            "auto_rejected": 2,
            "sent_to_review": 0,
            "edges_added": 3,
            "circuit_breaker_tripped": false,
            "edges_rolled_back": 0,
            "effective_cap": 10,
            "dry_run": false
        }
    ]
}
```

---

#### `POST /research/loop/configure`
Partially update the loop configuration. All fields are optional; unspecified fields retain their current value.

**Request body:**
```json
{
    "cycle_interval": 600.0,
    "max_materializations_per_cycle": 5,
    "min_approval_rate": 0.6,
    "circuit_breaker_window": 10,
    "dry_run": false,
    "auto_rollback_on_trip": true,
    "adaptive_tuning": true,
    "adaptive_min_cap": 1,
    "adaptive_max_cap": 20,
    "adaptive_min_interval": 60.0,
    "adaptive_max_interval": 3600.0
}
```

**Response (200 OK):** `LoopStatusResponse`

---

### Graph Provenance & Rollback

#### `GET /research/provenance/stats`
Return aggregate provenance statistics.

**Response (200 OK):**
```json
{
    "total_batches": 42,
    "total_edges": 138,
    "rollback_count": 2,
    "cycles_seen": 14
}
```

---

#### `GET /research/provenance/batches`
List recent materialization batches, newest first.

**Query params:** `?n=20` (default 20)

**Response (200 OK):**
```json
{
    "batches": [
        {
            "batch_id": "batch_20260414_001",
            "finding_id": "hyp_007",
            "cycle_number": 14,
            "edge_count": 3,
            "rolled_back": false,
            "recorded_at": 1743005204
        }
    ]
}
```

---

#### `POST /research/provenance/rollback/{batch_id}`
Remove all edges recorded under a single `batch_id`. Adapter must implement `remove_edge()`.

**Response (200 OK):**
```json
{"rolled_back": 3, "batch_id": "batch_20260414_001"}
```

---

#### `POST /research/provenance/rollback-cycle/{n}`
Remove all edges materialized during loop cycle `n`.

**Response (200 OK):**
```json
{"rolled_back": 7, "cycle_number": 12}
```

---

### Parameter Learning

#### `POST /feedback`
Submit feedback for a completed query path. Triggers an immediate online SGD update on community-specific CSA parameters via `MetaParameterLearner`. Also buffers the (positive, negative) pair for later batch retraining.

**Request body:**
```json
{
    "query_entity": "Marie Curie",
    "positive_path": ["Marie Curie", "discovered", "Polonium"],
    "negative_path": ["Marie Curie", "worked_at", "Sorbonne"],
    "community_id": 2
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query_entity` | string | required | The query entity that produced the paths |
| `positive_path` | list[string] | required | The correct/preferred path |
| `negative_path` | list[string] | null | An incorrect/rejected path (optional) |
| `community_id` | int | null | Community to update; null updates global prior |

**Response (200 OK):**
```json
{
    "updated": true,
    "community_id": 2,
    "buffer_size": 47,
    "new_params": [0.41, 0.39, 0.11, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10, 1.0]
}
```

---

#### `POST /retrain`
Batch retrain the global CSA parameter prior from all buffered feedback pairs using `CSAParameterLearner.fit()` (gradient descent). Clears the buffer on completion.

**Request body:** Empty `{}` or optional `{"learning_rate": 0.01, "epochs": 100}`

**Response (200 OK):**
```json
{
    "retrained": true,
    "pairs_used": 47,
    "epochs": 100,
    "final_loss": 0.0034,
    "new_global_prior": [0.42, 0.38, 0.11, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10, 1.0],
    "duration_ms": 380
}
```

---

#### `GET /params`
Inspect the current 10-parameter global vector and any community-specific overrides.

**Response (200 OK):**
```json
{
    "global_prior": [0.4, 0.4, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 1.0],
    "param_names": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "iota", "mu", "theta"],
    "community_overrides": {
        "2": [0.45, 0.35, 0.12, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10, 1.0]
    },
    "feedback_buffer_size": 0,
    "last_retrain_at": 1743000000
}
```

---

#### `POST /params`
Restore a parameter checkpoint (replaces global prior and community overrides). Used with `--params-file` CLI flag for startup restoration.

**Request body:**
```json
{
    "global_prior": [0.42, 0.38, 0.11, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10, 1.0],
    "community_overrides": {
        "2": [0.45, 0.35, 0.12, 0.05, 0.05, 0.10, 0.10, 0.05, 0.10, 1.0]
    }
}
```

**Response (200 OK):**
```json
{"restored": true, "communities_overridden": 1}
```

---

### Federated Reasoning

#### `POST /traverse`
Delegate multi-hop reasoning branch expansion to this node. Returns a set of `TraversalPath` objects (sub-beams) starting from the provided seed. Used by `DistributedBeamTraversal` to orchestrate cross-node federated reasoning.

**Request body:**
```json
{
    "seed_id": "Marie Curie",
    "context_embedding": [0.12, -0.05, ...],
    "max_hop": 2,
    "beam_width": 5,
    "max_budget": 500
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `seed_id` | string | required | Entity to start the local expansion from |
| `context_embedding` | list[float] | null | Optional semantic vector to seed the CSA attention |
| `max_hop` | int | 2 | Maximum depth for this sub-branch |
| `beam_width` | int | 5 | Width of the local beam search |
| `max_budget` | int | 500 | Max edges to explore in this branch |

**Response (200 OK):**
```json
{
    "seed_id": "Marie Curie",
    "branches": [
        {
            "nodes": ["Marie Curie", "discovered", "Polonium"],
            "score": 0.892,
            "embedding": [0.11, 0.04, ...],
            "attention_weights": [0.95, 0.88],
            "community_sequence": [0, 0, 0],
            "edge_confidences": [1.0, 1.0],
            "edge_provenances": ["", ""],
            "beta_alpha": 1.0,
            "beta_beta": 1.0
        }
    ]
}
```

---

### Observability (Phase 54)

#### `GET /logs`
Retrieve recent log entries from the in-memory `RingBufferHandler`. Supports filtering by level, time, and content.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"DEBUG"` | Minimum log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `limit` | int | 100 | Maximum number of entries to return |
| `since` | string | null | ISO 8601 timestamp — only entries after this time |
| `search` | string | null | Substring filter on log message text |

**Response (200 OK):**
```json
{
    "entries": [
        {
            "timestamp": "2026-04-05T12:00:01.234Z",
            "level": "INFO",
            "logger": "cerebrum.traversal",
            "message": "BeamTraversal: 847 hops in 6.3ms",
            "trace_id": "req_abc123"
        }
    ],
    "total_in_buffer": 500,
    "returned": 100
}
```

**curl example:**
```bash
curl "http://localhost:8200/logs?level=WARNING&limit=50&search=error" \
  -H "Authorization: Bearer $TOKEN"
```

---

#### `DELETE /logs`
Clear all entries from the log ring buffer.

**Response (200 OK):**
```json
{"cleared": true, "entries_removed": 500}
```

---

#### `POST /build`
Hot-reload the graph from an uploaded CSV file. Rebuilds embeddings, community partitions, and structural encodings in-process without a server restart. Accepts `multipart/form-data` with a `file` field.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | CSV file with format `subject,predicate,object[,weight]` |

**Response (200 OK):**
```json
{
    "built": true,
    "num_nodes": 142,
    "num_edges": 389,
    "num_communities": 11,
    "modularity_q": 0.451,
    "build_duration_ms": 1240
}
```

**curl example:**
```bash
curl -X POST http://localhost:8200/build \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@new_graph.csv"
```

---

### Graph Information

#### `GET /communities`
Return the current community partition of the loaded graph.

**Response (200 OK):**
```json
{
    "algorithm": "DSCF",
    "modularity_q": 0.432,
    "num_communities": 7,
    "communities": {
        "0": ["Marie Curie", "Pierre Curie", "Polonium", "Radium"],
        "1": ["Einstein", "Special Relativity", "General Relativity"],
        "2": ["Newton", "Gravity", "Calculus"]
    },
    "computed_at": 1743000000
}
```

---

#### `GET /graph/stats`
Return structural statistics for the loaded graph.

**Response (200 OK):**
```json
{
    "num_nodes": 21,
    "num_edges": 30,
    "avg_degree": 2.86,
    "density": 0.143,
    "num_communities": 7,
    "modularity_q": 0.432,
    "avg_clustering_coefficient": 0.41
}
```

---

#### `GET /graph/entity/{entity_id}`
Return all edges and community membership for a specific entity.

**Path parameter:** `entity_id` — URL-encoded entity string

**Response (200 OK):**
```json
{
    "entity": "Marie Curie",
    "community": 0,
    "soft_memberships": {"0": 0.72, "2": 0.18, "5": 0.10},
    "edges_out": [
        {"target": "Polonium", "relation": "discovered", "weight": 0.95},
        {"target": "Radium", "relation": "discovered", "weight": 0.93}
    ],
    "edges_in": [
        {"source": "Nobel_Prize_1903", "relation": "awarded_to", "weight": 0.99}
    ],
    "pagerank": 0.043,
    "betweenness": 0.127,
    "degree": 5
}
```

---

### Bridge Twins

#### `GET /bridges`
Return all active bridge twin nodes.

**Response (200 OK):**
```json
{
    "bridge_count": 3,
    "bridges": [
        {
            "twin_id": "bridge_7",
            "original_id": "Curie_Institute",
            "source_community": 0,
            "destination_community": 3,
            "crossing_count": 47,
            "ltp_weight": 0.84,
            "created_at": 1743000000
        }
    ]
}
```

---

#### `GET /bridges/{twin_id}`
Return details for a specific bridge twin node.

**Path parameter:** `twin_id` — bridge twin identifier

**Response:** Single bridge object (same schema as items in `/bridges`).

---

### Streaming

#### `GET /stream/events`
Server-Sent Event stream of raw graph update events.

**Headers:**
```
Accept: text/event-stream
Authorization: Bearer <token>
```

**Event format:**
```
event: edge_added
data: {"u": "A", "v": "B", "relation": "CAUSES", "weight": 0.72, "timestamp": 1743000000.0}

event: community_rebalanced
data: {"communities": 7, "q": 0.43, "pruned_bridges": 2}

event: bridge_formed
data: {"twin_id": "bridge_8", "src_community": 0, "dst_community": 3}
```

---

#### `GET /stream/insights`
Server-Sent Event stream of materialized insight events.

**Headers:** Same as `/stream/events`.

**Event format:**
```
event: insight_link
data: {"source": "A", "target": "B", "score": 0.81, "state": "SPECULATIVE"}

event: insight_verified
data: {"source": "A", "target": "B", "fwd_conf": 0.78, "rev_conf": 0.71}

event: insight_refuted
data: {"source": "A", "target": "B", "reason": "bilateral_probe_failed"}
```

---

#### `GET /stream/query`
Streaming NDJSON reasoning — emits partial results as they are discovered during traversal, rather than waiting for the full beam to complete.

**Query parameters:** Same as `GET /query`.

**Response:** Newline-delimited JSON, one path object per line, followed by a summary object.

**Fault tolerance (Phase 57):** If the traversal raises an unrecoverable exception, the stream terminates with a final JSON line before closing:
```json
{"status": "error", "partial": true, "error": "<exception message>"}
```
Any path objects already emitted before the failure remain valid and consumable.

---

#### `POST /stream/push`
Push a single event into the streaming ingest pipeline.

**Request body:**
```json
{
    "subject": "sensor_42",
    "predicate": "READS",
    "object": "temperature_chamber_1",
    "timestamp": 1743000000.0,
    "weight": 0.85
}
```

**Response (202 Accepted):**
```json
{"queued": true, "queue_depth": 14}
```

---

### Administration

#### `GET /health`
Health check endpoint. No authentication required.

**Response (200 OK):**
```json
{
    "status": "healthy",
    "version": "2.0.1",
    "graph_loaded": true,
    "num_nodes": 21,
    "num_edges": 30,
    "uptime_s": 3600
}
```

---

#### `POST /admin/rebalance`
Trigger an immediate synchronous DSCF community rebalance. Requires `scope: admin` in JWT.

**Response (200 OK):**
```json
{
    "rebalanced": true,
    "new_q": 0.448,
    "communities": 7,
    "pruned_bridges": 1,
    "duration_ms": 480
}
```

---

#### `GET /admin/resource-stats`
Return current ResourceGovernor metrics.

**Response (200 OK):**
```json
{
    "cpu_percent": 12.4,
    "memory_mb": 384,
    "active_queries": 3,
    "queued_events": 47,
    "rebalance_count": 12,
    "last_rebalance_at": 1743000000
}
```

---

#### `DELETE /admin/cache`
Clear the materialized path cache.

**Response (200 OK):**
```json
{"cleared": true, "entries_removed": 1024}
```

---

## Error Codes

| HTTP Status | Code | Description |
|---|---|---|
| 400 | `INVALID_ENTITY` | Entity not found in graph |
| 400 | `INVALID_PARAMS` | Request parameter validation failed |
| 401 | `AUTH_MISSING` | Authorization header absent |
| 401 | `AUTH_EXPIRED` | JWT token has expired |
| 401 | `AUTH_INVALID` | JWT signature invalid |
| 403 | `SCOPE_INSUFFICIENT` | Token scope does not permit operation |
| 404 | `NOT_FOUND` | Requested resource not found |
| 429 | `RATE_LIMITED` | Request rate exceeded; retry after N seconds |
| 500 | `TRAVERSAL_FAILED` | Internal traversal error (check logs) |
| 503 | `GRAPH_NOT_LOADED` | No graph loaded; call /build or restart with --csv |

**Error response format:**
```json
{
    "error": "INVALID_ENTITY",
    "detail": "Entity 'xyz' not found in graph",
    "status": 400
}
```

---

## Rate Limiting

Default rate limits (configurable via environment variables):

| Endpoint | Default Limit |
|---|---|
| `POST /query` | 100 requests/minute |
| `GET /query` | 100 requests/minute |
| `GET /communities` | 30 requests/minute |
| `POST /hypothesize` | 20 requests/minute |
| `POST /research/scan` | 5 requests/minute |
| `POST /validate/proposals` | 10 requests/minute |
| `POST /feedback` | 200 requests/minute |
| `POST /retrain` | 5 requests/minute |
| `GET /logs` | 60 requests/minute |
| `POST /build` | 10 requests/minute |
| `POST /stream/push` | 1000 requests/minute |
| `GET /stream/events` | 10 concurrent connections |
| `GET /stream/insights` | 10 concurrent connections |
| `POST /admin/*` | 10 requests/minute |

Rate limit responses include:
```
Retry-After: 15
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1743000060
```

---

## SDK Usage

```python
# Python client (using httpx)
import httpx, os

client = httpx.Client(
    base_url="http://localhost:8200",
    headers={"Authorization": f"Bearer {os.environ['CEREBRUM_TOKEN']}"}
)

# Core reasoning
response = client.post("/query", json={"entity": "Marie Curie", "max_hops": 3})
result = response.json()
print(result["answers"][0]["path"])

# Generate hypotheses
hyps = client.post("/hypothesize", json={"entity": "BRCA1", "top_k": 5}).json()

# Validate against literature
validated = client.post("/validate/proposals", json={
    "proposal_ids": [h["id"] for h in hyps["hypotheses"]],
    "sources": ["pubmed", "arxiv"]
}).json()

# Submit feedback and retrain
client.post("/feedback", json={
    "query_entity": "Marie Curie",
    "positive_path": ["Marie Curie", "discovered", "Polonium"]
})
client.post("/retrain", json={})

# Export learned parameters
checkpoint = client.get("/params").json()
```

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

# Multi-Tenant API

CEREBRUM supports multiple tenants on a single server — each with an isolated knowledge graph, independent usage metering, and per-key access control.

## Quick start

```bash
# Set an admin key before starting the server
export CEREBRUM_ADMIN_KEY=my-secret-admin-key

cerebrum serve --csv kb.csv
```

All admin endpoints require the `X-API-Key: <CEREBRUM_ADMIN_KEY>` header.

---

## API Key Management

### Create a key

```bash
curl -X POST http://localhost:8200/v1/admin/keys \
  -H "X-API-Key: my-secret-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"label": "production-app", "tenant_id": "default"}'
```

Response (raw secret shown once):
```json
{
  "key_id": "abc123def456",
  "tenant_id": "default",
  "label": "production-app",
  "created_at": 1748290420.0,
  "is_active": true,
  "raw_secret": "zXk9....<store this>"
}
```

### List keys

```bash
curl http://localhost:8200/v1/admin/keys \
  -H "X-API-Key: my-secret-admin-key"

# Filter by tenant
curl "http://localhost:8200/v1/admin/keys?tenant_id=acme" \
  -H "X-API-Key: my-secret-admin-key"
```

### Revoke a key

```bash
curl -X DELETE http://localhost:8200/v1/admin/keys/abc123def456 \
  -H "X-API-Key: my-secret-admin-key"
```

---

## Usage Metering

### Per-key usage

```bash
curl http://localhost:8200/v1/admin/keys/abc123def456/usage \
  -H "X-API-Key: my-secret-admin-key"
```

```json
{
  "key_id": "abc123def456",
  "tenant_id": "default",
  "label": "production-app",
  "queries_today": 142,
  "total_queries": 4831,
  "avg_latency_ms": 18.4,
  "last_used_at": 1748290420.0
}
```

### All keys aggregate

```bash
curl http://localhost:8200/v1/admin/usage \
  -H "X-API-Key: my-secret-admin-key"
```

---

## Per-Tenant Knowledge Graphs

Each API key has a `tenant_id`. If a graph is registered for that tenant, queries from that key are routed to the tenant-specific graph instead of the default graph.

### Register a tenant KB

```bash
curl -X POST http://localhost:8200/v1/admin/tenants \
  -H "X-API-Key: my-secret-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme",
    "csv_path": "/data/acme_kb.csv",
    "source_col": "subject",
    "target_col": "object",
    "relation_col": "predicate"
  }'
```

### List tenants

```bash
curl http://localhost:8200/v1/admin/tenants \
  -H "X-API-Key: my-secret-admin-key"
```

### Route a key to a tenant

Create the API key with `tenant_id: "acme"`:

```bash
curl -X POST http://localhost:8200/v1/admin/keys \
  -H "X-API-Key: my-secret-admin-key" \
  -d '{"label": "acme-app", "tenant_id": "acme"}'
```

Queries using that key now run against the `acme` graph.

---

## Key persistence

Dynamic keys survive server restarts — they are stored in:

```
data/cerebrum/api_keys.json
```

Keys are stored as SHA-256 hashes; raw secrets are never written to disk.

## Authentication modes

| Scenario | Behaviour |
|----------|-----------|
| No `CEREBRUM_API_KEYS` and no `CEREBRUM_ADMIN_KEY` set | Dev mode — any key accepted |
| `CEREBRUM_API_KEYS=key1,key2` set | Static allow-list |
| Dynamic key (created via admin API) | Identity-aware, per-tenant routing |
| `CEREBRUM_ADMIN_KEY=X` set | Admin endpoints available |

All modes are additive — a server with both static keys and dynamic keys accepts either.

# Compliance Mode

Compliance mode activates full query audit logging — every query is recorded with its timestamp, client identity, answer, confidence score, and complete hop-by-hop reasoning trace.

Designed for GDPR, HIPAA, SOX, and any regulatory environment requiring AI decision auditability.

## Enable

```bash
# Default audit log file: cerebrum_audit.jsonl
cerebrum serve --csv kb.csv --compliance

# Custom audit log path
cerebrum serve --csv kb.csv --compliance --audit-log /var/log/cerebrum/queries.jsonl
```

## What gets logged

Every call to `POST /v1/query` appends a record:

```json
{
  "record_id": 42,
  "timestamp": 1748290420.123,
  "iso_time": "2026-05-18T14:30:20.123Z",
  "client_id": "user@example.com",
  "query": "Who directed Inception?",
  "answer": "Christopher_Nolan",
  "confidence": 0.923,
  "hop_depth": 2,
  "elapsed_ms": 14.2,
  "trace_path": [
    {"entity": "Inception", "relation": "directed_by"},
    {"entity": "Christopher_Nolan", "relation": ""}
  ]
}
```

`client_id` is extracted from the JWT `sub` claim, or `"anonymous"` if unauthenticated.

## Export the audit log

### Via REST API

```bash
# JSON (last 1000 records)
curl http://localhost:8200/v1/compliance/audit

# CSV (all records)
curl "http://localhost:8200/v1/compliance/audit?fmt=csv&n=0" \
  -o audit_export.csv

# Summary statistics
curl http://localhost:8200/v1/compliance/stats
```

Stats response:
```json
{
  "total": 14234,
  "first_at": "2026-05-01T09:00:00Z",
  "last_at": "2026-05-18T14:30:20Z",
  "avg_confidence": 0.7842,
  "avg_elapsed_ms": 18.3,
  "unique_clients": 47
}
```

### Via Python

```python
from core.query_audit_ledger import QueryAuditLedger

ledger = QueryAuditLedger(log_file="cerebrum_audit.jsonl")

# Export
ledger.export_json("audit_2026_05.json")
ledger.export_csv("audit_2026_05.csv")

# In-memory stats
print(ledger.stats())
```

## JSONL file format

The audit log is an append-only JSONL file (one JSON record per line). It survives server restarts — each launch appends to the same file if `--audit-log` points to an existing path.

This format is compatible with:

- **Splunk**, **Datadog**, **Elasticsearch** (direct log ingestion)
- **pandas**: `df = pd.read_json("cerebrum_audit.jsonl", lines=True)`
- **SQL**: import via any ETL that supports JSONL

## Retention

The in-memory ring buffer holds the last 10,000 records. For longer retention, rely on the JSONL file — it is unlimited and append-only.

To purge the file, simply delete or rotate it; the server will create a new one on the next query.

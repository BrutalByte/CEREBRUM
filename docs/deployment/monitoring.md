# Monitoring & Observability

CEREBRUM exposes a Prometheus-compatible `/metrics` endpoint and ships a pre-built Grafana dashboard for operational visibility.

## Install

```bash
pip install "cerebrum-kg[monitoring]"
# or
pip install prometheus-client
```

Without `prometheus-client` the `/metrics` endpoint returns a plain-text hint instead of metrics data. The server runs normally in either case.

## Metrics endpoint

```
GET /v1/metrics
```

Returns Prometheus text-format exposition. Configure your Prometheus scrape config to poll this endpoint.

### Metrics exposed

| Metric | Type | Description |
|--------|------|-------------|
| `cerebrum_queries_total{status}` | Counter | Total queries by status (`ok` / `partial` / `error`) |
| `cerebrum_query_latency_seconds` | Histogram | End-to-end query latency |
| `cerebrum_paths_explored_total` | Counter | Beam paths explored across all queries |
| `cerebrum_hop_depth` | Histogram | Hop depth of top-1 answer path |
| `cerebrum_answer_confidence` | Histogram | Confidence score of top-1 answer |
| `cerebrum_graph_nodes` | Gauge | Entity count of the loaded graph |
| `cerebrum_graph_edges` | Gauge | Edge count of the loaded graph |
| `cerebrum_active_tenants` | Gauge | Registered tenant knowledge graphs |
| `cerebrum_api_keys_active` | Gauge | Active dynamic API keys |
| `cerebrum_audit_records_total` | Gauge | Compliance audit records in memory buffer |

## Prometheus + Grafana via Docker Compose

```bash
# Start CEREBRUM + Prometheus + Grafana
docker compose \
  -f docker-compose.yml \
  -f monitoring/docker-compose.monitoring.yml \
  up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / cerebrum)

The Grafana datasource and dashboard are provisioned automatically on first start — no manual setup required.

## Dashboard panels

The pre-built dashboard (`monitoring/grafana_provisioning/dashboards/cerebrum.json`) includes:

- **Stat row**: Queries/min, p95 latency, avg confidence, graph node count, active tenants
- **Query Rate** by status (ok / partial / error) — time series
- **Latency Percentiles** — p50 / p95 / p99 over time
- **Beam Paths Explored** — throughput indicator
- **Answer Confidence** — avg confidence over time
- **Hop Depth** — avg reasoning depth
- **API Keys & Tenants** — active keys, tenants, audit record count

Default refresh: 10 seconds.

## Standalone Prometheus config

```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: cerebrum
    static_configs:
      - targets: ["localhost:8200"]
    metrics_path: /v1/metrics
```

## SLO recommendations

| Signal | Warning | Critical |
|--------|---------|----------|
| p95 latency | > 500ms | > 2s |
| Error rate | > 1% | > 5% |
| Avg confidence | < 0.5 | < 0.3 |

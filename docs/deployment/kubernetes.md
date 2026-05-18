# Kubernetes & Horizontal Scale

CEREBRUM's query path is stateless — the knowledge graph is loaded read-only from a shared volume, and all writes (API key store, audit log, Engram cache) go to the same shared volume. This makes it straightforward to run multiple replicas behind a load balancer.

## Architecture

```
               ┌─────────────────────────────────┐
Internet ──>   │  LoadBalancer / Ingress          │
               └────────────┬────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   cerebrum pod (replica 1) │
              │   cerebrum pod (replica 2) ├──> Shared PVC (NFS / EFS)
              │   cerebrum pod (replica N) │      /data/graph.csv (read)
              └───────────────────────────┘      /data/api_keys.json (rw)
                                                 /data/cerebrum_audit.jsonl (rw)
```

All replicas are identical and stateless at query time. The only shared mutable state is small (key store + audit log JSON/JSONL files) and write-rarely. For very high write rates, swap the file-backed `ApiKeyStore` for a Redis-backed variant.

---

## Quick deploy (Kustomize)

```bash
# 1. Clone the repo and set your image tag
cd k8s
kustomize edit set image ghcr.io/brutalbyte/cerebrum:v2.58.0

# 2. Set real credentials (never commit this file)
kubectl create secret generic cerebrum-secrets -n cerebrum \
  --from-literal=CEREBRUM_API_KEYS="prod-key-1,prod-key-2" \
  --from-literal=CEREBRUM_ADMIN_KEY="your-secure-admin-key"

# 3. Create a StorageClass that supports ReadWriteMany, then apply
kubectl apply -k k8s/

# 4. Check rollout
kubectl rollout status deployment/cerebrum -n cerebrum
```

---

## Manifests

| File | Purpose |
|------|---------|
| `k8s/namespace.yaml` | `cerebrum` namespace |
| `k8s/pvc.yaml` | ReadWriteMany PVC for shared data |
| `k8s/configmap.yaml` | Non-secret environment config |
| `k8s/secret.yaml` | API keys (example — replace before applying) |
| `k8s/deployment.yaml` | 2-replica Deployment with readiness/liveness probes |
| `k8s/service.yaml` | ClusterIP + LoadBalancer services |
| `k8s/hpa.yaml` | HPA: scale 2→10 replicas at 60% CPU / 70% memory |
| `k8s/kustomization.yaml` | Kustomize entry point |

---

## Docker Compose scale-out

For Docker Compose environments (without Kubernetes):

```bash
# Start 4 replicas behind Nginx
docker compose \
  -f docker-compose.yml \
  -f docker-compose.scale.yml \
  up --scale cerebrum=4
```

The `docker-compose.scale.yml` overlay:
- Removes the direct port binding from `cerebrum` containers
- Adds an Nginx load balancer on port 8200
- Uses Docker's built-in DNS to round-robin across all replicas

---

## Storage requirements

| File | Size (typical) | Access pattern |
|------|---------------|----------------|
| `graph.csv` or `.pkl` state | 10 MB – 10 GB | Read-only at startup |
| `api_keys.json` | < 1 MB | Append on key creation, read on every auth |
| `cerebrum_audit.jsonl` | Grows at ~1 KB/query | Append-only |
| `engram_cache.json` | < 50 MB | Read at startup, write at shutdown |

A 10 Gi PVC covers most production deployments. Scale up if your KB exceeds 5 GB.

---

## Resource guidance

| Replicas | CPU request | Memory request | Use case |
|----------|-------------|----------------|----------|
| 2 | 500m each | 1 Gi each | Development / staging |
| 4 | 1 each | 2 Gi each | Production (< 100 qps) |
| 8+ | 2 each | 4 Gi each | High-throughput (> 500 qps) |

Enable GPU scheduling by adding `resources.limits.nvidia.com/gpu: 1` to the container spec and using a GPU-enabled node pool.

---

## Prometheus metrics in K8s

The Deployment manifest includes standard Prometheus annotations:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/path: "/v1/metrics"
  prometheus.io/port: "8200"
```

If you use the Prometheus Operator, create a `ServiceMonitor` targeting the `cerebrum` service instead.

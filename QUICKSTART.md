# CEREBRUM Quickstart

CEREBRUM is a Community-Structured Graph Attention engine for Knowledge Graph reasoning.
It answers multi-hop queries over any graph by traversing edges — no LLM, no training data.

---

## Option A — Docker (recommended)

### 1. Generate an API key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
# e.g. a3f8c2d1e9b04f7a...
```

### 2. Start the server

```bash
# Set your accepted keys (comma-separated)
export CEREBRUM_API_KEYS=your-key-here

# Start with a CSV graph
python -m cli.cerebrum serve --csv my_graph.csv --port 8200
```

The API is now available at `http://localhost:8200`.
Authentication: Send `-H "X-API-Key: your-key-here"` or `-H "Authorization: Bearer <jwt>"`.

---

## Zero-Config Reasoning (Phase 172)

CEREBRUM now automatically configures itself for your graph. Just load it and query:

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "X-API-Key: your-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What compound treats Diabetes?",
    "max_hop": 3
  }'
```

The **GraphProfiler** (Phase 172) detects your graph type, and **STRB** (Phase 172) automatically boosts the "treats" relationship based on the question text.

---

## Option B — pip install

### 1. Install

```bash
pip install -e ".[api,embeddings]"
```

### 2. Start the server

```bash
# Dev mode (no auth):
python -m cli.cerebrum serve --csv my_graph.csv --port 8200

# With API key:
CEREBRUM_API_KEYS=your-key-here python -m cli.cerebrum serve --csv my_graph.csv --port 8200
```

---

## Running a query

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "X-API-Key: your-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "newton",
    "top_k": 5,
    "max_hop": 3
  }'
```

**Response** — ranked paths through the graph:

```json
{
  "query": "newton",
  "results": [
    {
      "rank": 1,
      "answer_entity": "gravity",
      "score": 0.87,
      "path": [
        {"entity": "newton",  "relation": "discovered"},
        {"entity": "gravity", "relation": null}
      ]
    }
  ]
}
```

---

## Reading a reasoning trace

The `/query/trace` endpoint returns the full beam-search trace with per-hop attention weights:

```bash
curl -X POST http://localhost:8200/v1/query/trace \
  -H "X-API-Key: your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "newton", "top_k": 3, "max_hop": 3}'
```

Each hop in the trace includes the **10 CSA attention parameters** — semantic similarity,
community score, edge-type weight, PageRank prior, temporal decay, and more — showing
exactly why the engine ranked one path over another.

---

## Check server health (no key required)

```bash
curl http://localhost:8200/health
# {"status":"ok","node_count":21,"community_count":4,...}
```

---

## API key setup reference

| Env var | Purpose |
|---|---|
| `CEREBRUM_API_KEYS` | Comma-separated accepted keys, e.g. `key1,key2` |
| `CEREBRUM_CSV_PATH` | CSV to load at startup (Docker / `api.main`) |
| `CEREBRUM_DATA_DIR` | Root for persistent data (default: `data/cerebrum`) |
| `CEREBRUM_CACHE_PATH` | State pickle for fast restart after community detection |
| `CEREBRUM_WS_PORT` | WebSocket port for UE5 / real-time visualization feed |

If `CEREBRUM_API_KEYS` is **not set**, the server runs in open dev mode — all requests are accepted without a key. Always set a key before exposing the server on a network.

## Background maintenance (REM Cycle)

CEREBRUM automatically optimizes its internal structure during idle periods. You can also trigger a manual **REM Cycle** pass to strengthen successful paths and synthesize shortcuts:

```bash
curl -X POST http://localhost:8200/v1/rem/run \
  -H "X-API-Key: your-key-here"
```

---

## Next steps

- **Feedback loop**: `POST /feedback` with `{"path": ..., "reward": 1.0}` to online-train CSA attention weights
- **Autonomous research**: `POST /research/loop/start` to enable autonomous missing-link discovery
- **Full API reference**: `http://localhost:8200/v1/docs` (Swagger UI, or browse to `/` for auto-redirect)

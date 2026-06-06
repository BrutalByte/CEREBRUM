---
name: hotswap-graph
description: Hot-swap the loaded graph on a running CEREBRUM server without restarting. Calls the Studio hot-swap endpoint (Phase 175). Usage: /hotswap-graph <path-to-csv>
disable-model-invocation: true
---

Hot-swap the active graph via the Studio engine API:

```bash
# Default: reload toy graph
CSV="${ARGS:-tests/fixtures/toy_graph.csv}"

curl -s -X POST "http://localhost:8200/graph/hotswap" \
  -H "Content-Type: application/json" \
  -d "{\"csv_path\": \"${CSV}\"}" | python -m json.tool
```

If the hot-swap endpoint isn't available on the running server, use the Studio UI:
- Open http://localhost:8200/web/ → Studio tab → Settings → Hot-swap graph

Notes:
- Server must be running on port 8200
- Engram cache and CSA parameters are preserved across hot-swaps
- New graph triggers a full DSCF community re-partition

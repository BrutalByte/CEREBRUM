# Quick Start

Get CEREBRUM running in under 5 minutes.

## Install

```bash
pip install cerebrum-kg[all]
```

For GPU acceleration (recommended):

```bash
pip install cerebrum-kg[all] torch --index-url https://download.pytorch.org/whl/cu124
```

## Option A — CLI Wizard (fastest)

```bash
# Instant demo with built-in toy KB
cerebrum init --demo

# Your own CSV → KB → API server → browser
cerebrum init --from-csv mydata.csv --serve --open
```

The wizard will:

1. Load your CSV and detect the graph structure
2. Run community detection
3. Print a KB summary (entities, relations, communities)
4. Optionally start the REST API server and open Swagger UI

## Option B — Python SDK

```python
from cerebrum_sdk import Cerebrum

# Load from any CSV with source/target/relation columns
c = Cerebrum.from_csv("kb.csv")

# Ask a natural-language question
result = c.ask("Who directed Inception?")

print(result.answer)       # Christopher_Nolan
print(result.confidence)   # 0.923
print(result.elapsed_ms)   # 14.2

# Walk the crystal-box reasoning path
for step in result.trace_path:
    print(f"  {step.entity} --[{step.relation}]-->")
# Inception --[directed_by]-->

# Top-5 candidates with confidence scores
for cand in result.top_k:
    print(cand["entity"], cand["confidence"])
```

## Option C — REST API

```bash
cerebrum serve --csv kb.csv --port 8200
```

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who directed Inception?", "top_k": 5}'
```

Interactive API docs: `http://localhost:8200/v1/docs`

## Option D — Studio UI (no-code)

```bash
python ui/studio.py
```

Open `http://localhost:7860` and use the **KB Builder** tab to drag-and-drop your CSV, map columns, and start querying — no Python required.

## Next Steps

- [Your First KB](first-kb.md) — build a knowledge base from scratch
- [Python SDK reference](../sdk/python.md) — full API
- [Data Connectors](../connectors/overview.md) — CSV, SQL, Parquet, Neo4j

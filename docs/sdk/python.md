# Python SDK Reference

## Installation

```bash
pip install cerebrum-kg[all]
```

Then add the SDK to your path (or install as a package):

```python
import sys
sys.path.insert(0, "sdk/python")
from cerebrum_sdk import Cerebrum, Result, TraceStep
```

---

## `Cerebrum` — main class

### Factory methods

#### `Cerebrum.from_csv(path, *, source_col, target_col, relation_col, embeddings, beam_width, max_hop, top_k)`

Load from a CSV edge-list.

```python
c = Cerebrum.from_csv("kb.csv")

# Custom column names
c = Cerebrum.from_csv(
    "edges.csv",
    source_col="head",
    target_col="tail",
    relation_col="predicate",
)

# With sentence embeddings (better accuracy, slower)
c = Cerebrum.from_csv("kb.csv", embeddings="sentence")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source_col` | `"source"` | Source/head entity column |
| `target_col` | `"target"` | Target/tail entity column |
| `relation_col` | `"relation"` | Relation type column |
| `embeddings` | `"random"` | `"random"` or `"sentence"` |
| `beam_width` | `10` | Beam search width |
| `max_hop` | `3` | Maximum reasoning hops |
| `top_k` | `5` | Number of candidates |

---

#### `Cerebrum.from_sql(connection, query, *, source_col, target_col, relation_col, ...)`

Load from a SQL query.

```python
import sqlite3

conn = sqlite3.connect("knowledge.db")
c = Cerebrum.from_sql(conn, "SELECT subject, predicate, object FROM triples",
                      source_col="subject", target_col="object", relation_col="predicate")

# SQLAlchemy URL
c = Cerebrum.from_sql("postgresql://user:pass@host/db",
                      "SELECT source, target, relation FROM kg")
```

---

#### `Cerebrum.from_parquet(source, *, source_col, target_col, relation_col, ...)`

Load from a Parquet file, Feather/Arrow file, pandas DataFrame, or pyarrow Table.

```python
c = Cerebrum.from_parquet("kg.parquet")

# DataFrame
import pandas as pd
df = pd.read_parquet("kg.parquet")
c = Cerebrum.from_parquet(df)
```

---

#### `Cerebrum.from_triples(triples, *, directed, embeddings, ...)`

Load from a list of `(subject, relation, object)` tuples.

```python
triples = [
    ("Inception", "directed_by", "Christopher_Nolan"),
    ("Christopher_Nolan", "born_in", "London"),
]
c = Cerebrum.from_triples(triples)
```

---

#### `Cerebrum.from_kb(path, *, sep, embeddings, ...)`

Load from a pipe-separated triples file (MetaQA/Freebase format).

```python
c = Cerebrum.from_kb("kb.txt", sep="|")
```

---

### Query methods

#### `c.ask(question) → Result`

Answer a natural-language question.

```python
result = c.ask("Who directed Inception?")
```

Optional overrides per call:

```python
result = c.ask("Who directed Inception?", beam_width=15, max_hop=4, top_k=10)
```

---

#### `c.query(entity) → Result`

Query by entity ID or label (no NL parsing step, slightly faster).

```python
result = c.query("Inception")
```

---

### Properties

#### `c.stats → dict`

```python
print(c.stats)
# {'entities': 14541, 'relations': 210448, 'relation_types': 9, 'communities': 43}
```

---

## `Result` — return type

```python
result = c.ask("Who directed Inception?")

result.answer        # "Christopher_Nolan"
result.confidence    # 0.9231
result.trace_path    # [TraceStep(entity='Inception', relation='directed_by'), ...]
result.top_k         # [{'entity': 'Christopher_Nolan', 'confidence': 0.923}, ...]
result.elapsed_ms    # 14.2
str(result)          # "Christopher_Nolan (conf=0.923, path: Inception[directed_by]->...)"
```

### `TraceStep`

```python
for step in result.trace_path:
    print(step.entity)    # "Inception"
    print(step.relation)  # "directed_by"
```

# Your First Knowledge Base

This guide walks you through building a KB from scratch and querying it.

## Step 1 — Prepare your data

CEREBRUM needs a list of triples: `(subject, relation, object)`.

The simplest format is a CSV with three columns:

```csv
source,relation,target
Inception,directed_by,Christopher_Nolan
Inception,starring,Leonardo_DiCaprio
The_Dark_Knight,directed_by,Christopher_Nolan
Christopher_Nolan,born_in,London
London,located_in,United_Kingdom
```

Column names are flexible — `source/target/relation` are the defaults. You can use any names and map them at load time.

## Step 2 — Load and query

```python
from cerebrum_sdk import Cerebrum

c = Cerebrum.from_csv("movies.csv")

# Natural-language question
r = c.ask("Where was the director of Inception born?")
print(r.answer)      # London
print(r.trace_path)  # Inception → directed_by → Christopher_Nolan → born_in → London

# Direct entity query
r2 = c.query("Christopher_Nolan")
print(r2.top_k[:3])
```

## Step 3 — Custom column names

```python
c = Cerebrum.from_csv(
    "edges.csv",
    source_col="head",
    target_col="tail",
    relation_col="predicate",
)
```

## Step 4 — From a database

```python
import sqlite3

conn = sqlite3.connect("knowledge.db")
c = Cerebrum.from_sql(
    conn,
    "SELECT subject, predicate, object FROM triples",
    source_col="subject",
    target_col="object",
    relation_col="predicate",
)
```

## Step 5 — From a Parquet file

```python
c = Cerebrum.from_parquet("kg.parquet")
```

## Step 6 — Start the API server

```bash
cerebrum serve --csv movies.csv --port 8200
```

Then query via HTTP:

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Inception director", "top_k": 5}'
```

## What makes a good KB?

- **Specificity**: each relation should have a single clear meaning (`directed_by` not `associated_with`)
- **Coverage**: multi-hop queries need intermediate entities to be present (`Inception → Christopher_Nolan → London`)
- **Consistency**: use the same entity ID everywhere (`Christopher_Nolan` not `Chris Nolan` in one row)
- **Size**: CEREBRUM works on KBs from hundreds to millions of edges; no minimum required

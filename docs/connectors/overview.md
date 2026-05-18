# Data Connectors

CEREBRUM is fully data-agnostic — it works with any structured dataset that can be expressed as triples `(subject, relation, object)`.

## Available connectors

| Connector | Class / Function | Source |
|-----------|-----------------|--------|
| CSV | `Cerebrum.from_csv()` | Any CSV with named columns |
| SQL | `Cerebrum.from_sql()` | SQLite, PostgreSQL, MySQL, any SQLAlchemy DB |
| Parquet / Arrow | `Cerebrum.from_parquet()` | .parquet, .arrow, .feather, pandas DataFrame |
| In-memory triples | `Cerebrum.from_triples()` | Python list of (s, r, o) tuples |
| KB triples file | `Cerebrum.from_kb()` | Pipe- or comma-separated flat files |
| Neo4j | `Neo4jAdapter` | Neo4j graph database |
| Amazon Neptune | `NeptuneAdapter` | AWS Neptune (Gremlin) |
| RDF / SPARQL | `RDFAdapter` | Any SPARQL endpoint |
| Federated | `FederatedGraphRegistry` | Multiple sources merged at query time |
| Memory-mapped | `MmapAdapter` | Large KBs exceeding RAM |

## Choosing a connector

```
Your data is in...            Use...
─────────────────────────────────────────────────────
A CSV file                 →  Cerebrum.from_csv()
A relational database      →  Cerebrum.from_sql()
A data pipeline (Parquet)  →  Cerebrum.from_parquet()
Neo4j / Neptune            →  Neo4jAdapter / NeptuneAdapter
SPARQL endpoint            →  RDFAdapter
Multiple sources           →  FederatedGraphRegistry
Very large graph (>RAM)    →  MmapAdapter
```

## Column mapping

All connectors accept `source_col`, `target_col`, and `relation_col` to map your schema:

```python
# Your CSV has "entity1", "entity2", "type" columns
c = Cerebrum.from_csv("data.csv",
    source_col="entity1",
    target_col="entity2",
    relation_col="type",
)
```

If `relation_col` is absent or the column doesn't exist, CEREBRUM uses `"RELATED_TO"` for all edges.

## What makes a good triple?

- **Specific relations**: `directed_by` is better than `associated_with`
- **Consistent entity IDs**: use the same string everywhere (`Christopher_Nolan`, not `Chris Nolan`)
- **Coverage**: intermediate entities must be present for multi-hop queries to work
- **No required minimum size**: CEREBRUM works from tens to millions of edges

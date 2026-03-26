# CEREBRUM Examples

Minimal, runnable demonstrations of the CEREBRUM Python API. All examples use only core dependencies — no API server required.

---

## Prerequisites

```bash
pip install -e "."                    # core only (all examples work)
pip install -e ".[embeddings]"        # for semantic embedding examples
```

---

## Examples

### `csv_quickstart.py` — Minimal end-to-end pipeline

The simplest possible CEREBRUM session: load a CSV graph, detect communities, run a query, print results.

```bash
python examples/csv_quickstart.py
```

**What it demonstrates:**
- Loading a graph from `tests/fixtures/toy_graph.csv` (21 nodes, 30 edges)
- DSCF community detection (`best_of_n_dscf`)
- `RandomEngine` embeddings (no sentence-transformers needed)
- `CSAEngine` construction
- `BeamTraversal` with 3-hop beam search
- `AnswerExtractor` top-5 results
- `to_prompt()` — formatting results for an LLM

**Expected output:**
```
Graph loaded: 21 nodes, 30 edges
Communities: 5  |  Modularity Q: 0.4312
Query: What is connected to 'newton'?
Paths explored: 847
Top-5 answers:
  gravity               score=0.7841  path: newton -> INFLUENCED -> leibniz -> ...
  calculus              score=0.7203  path: newton -> CONTRIBUTED_TO -> calculus
  ...
```

---

### `neo4j_quickstart.py` — Neo4j backend

Demonstrates connecting to a Neo4j instance and running the same pipeline against a live graph database.

```bash
# Requires Neo4j running (Docker: docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j)
NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=password \
    python examples/neo4j_quickstart.py
```

**What it demonstrates:**
- `Neo4jAdapter` connection and graph loading
- Same DSCF + CSA + BeamTraversal pipeline on a database backend
- Query result formatting

---

### `wikidata_quickstart.py` — RDF / SPARQL backend

Demonstrates running CEREBRUM against a live SPARQL endpoint (Wikidata public endpoint).

```bash
python examples/wikidata_quickstart.py
```

**What it demonstrates:**
- `RDFAdapter` connection to a SPARQL endpoint
- Entity resolution from SPARQL results
- Multi-hop reasoning over a public knowledge graph

**Note**: Wikidata's public SPARQL endpoint has rate limits. This example uses a small focused query.

---

### `Validation_Walkthrough.ipynb` — Interactive white paper

A Jupyter notebook that serves as an interactive tutorial and visual white paper. Recommended as the first stop for new users.

```bash
pip install -e ".[dev]"
jupyter notebook examples/Validation_Walkthrough.ipynb
```

**What it demonstrates:**
- Step-by-step visualization of community formation
- CSA attention weight breakdown per edge
- 3-hop reasoning path tracing with community annotations
- The Transformer ↔ KG structural analogy illustrated with live data

---

## Adding an Example

Place new example scripts in `examples/` following these conventions:
- Filename: `<backend_or_feature>_quickstart.py` or `<feature>_demo.py`
- Self-contained: must run with `python examples/<name>.py` from the repo root
- Commented: key steps should have `# N. Description` comments matching the pattern in `csv_quickstart.py`
- Update this README with the new entry

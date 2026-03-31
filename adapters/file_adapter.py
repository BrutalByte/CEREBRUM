"""
Universal file loader for CEREBRUM.

Dispatches to the correct loader based on file extension. Supports:
  .csv / .tsv       — edge-list (source, target, relation columns)
  .json / .jsonl    — edge list as JSON array or JSON Lines
  .graphml          — GraphML (NetworkX native)
  .gexf             — GEXF (NetworkX native)
  .gml              — GML (NetworkX native)
  .parquet          — columnar edge list (pandas/pyarrow)
  .xlsx / .xls      — Excel edge list (pandas/openpyxl)

All loaders return a NetworkXAdapter.

JSON formats accepted
---------------------
Array-of-objects (edge list):
    [{"source": "a", "target": "b", "relation": "R"}, ...]

Nodes-and-edges object:
    {"nodes": ["a", "b", ...], "edges": [{"source": "a", "target": "b", "relation": "R"}, ...]}

Triples array (3-element arrays):
    [["a", "R", "b"], ...]

JSONL (one edge per line):
    {"source": "a", "target": "b", "relation": "R"}
    {"source": "b", "target": "c", "relation": "S"}
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_file_adapter(
    path: str,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
    encoding: str = "utf-8",
) -> NetworkXAdapter:
    """
    Load any supported graph file into a NetworkXAdapter.

    Parameters
    ----------
    path         : path to the file
    source_col   : column/key name for the source entity (CSV/JSON/Parquet/Excel)
    target_col   : column/key name for the target entity
    relation_col : column/key name for the relation type (optional in source)
    directed     : use DiGraph if True, Graph if False
    encoding     : text file encoding (CSV, JSON, GML, GraphML, GEXF)

    Returns
    -------
    NetworkXAdapter wrapping the loaded graph

    Raises
    ------
    FileNotFoundError  : file does not exist
    ValueError         : unsupported extension or malformed content
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")

    ext = p.suffix.lower()

    if ext == ".csv":
        return _load_csv(p, source_col, target_col, relation_col, directed, encoding)
    elif ext == ".tsv":
        return _load_csv(p, source_col, target_col, relation_col, directed, encoding, delimiter="\t")
    elif ext in (".json", ".jsonl"):
        return _load_json(p, source_col, target_col, relation_col, directed, encoding)
    elif ext == ".graphml":
        return _load_graphml(p, directed)
    elif ext == ".gexf":
        return _load_gexf(p, directed)
    elif ext == ".gml":
        return _load_gml(p, directed)
    elif ext == ".parquet":
        return _load_parquet(p, source_col, target_col, relation_col, directed)
    elif ext in (".xlsx", ".xls"):
        return _load_excel(p, source_col, target_col, relation_col, directed)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: .csv, .tsv, .json, .jsonl, .graphml, .gexf, .gml, .parquet, .xlsx, .xls"
        )


# ---------------------------------------------------------------------------
# Format loaders
# ---------------------------------------------------------------------------

def _make_graph(directed: bool) -> nx.Graph:
    return nx.DiGraph() if directed else nx.Graph()


def _load_csv(
    p: Path,
    source_col: str,
    target_col: str,
    relation_col: str,
    directed: bool,
    encoding: str,
    delimiter: str = ",",
) -> NetworkXAdapter:
    import csv
    G = _make_graph(directed)
    with open(p, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"File has no header row: {p}")
        for row in reader:
            src = row.get(source_col, "").strip()
            tgt = row.get(target_col, "").strip()
            if not src or not tgt or src.startswith("#"):
                continue
            rel = row.get(relation_col, "RELATED_TO").strip() or "RELATED_TO"
            G.add_edge(src, tgt, relation=rel)
    return NetworkXAdapter(G)


def _load_json(
    p: Path,
    source_col: str,
    target_col: str,
    relation_col: str,
    directed: bool,
    encoding: str,
) -> NetworkXAdapter:
    import json
    G = _make_graph(directed)
    text = p.read_text(encoding=encoding)

    # JSONL: multiple JSON objects, one per line
    if p.suffix.lower() == ".jsonl" or "\n{" in text.lstrip():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//")]
        try:
            edges = [json.loads(ln) for ln in lines]
        except json.JSONDecodeError:
            edges = json.loads(text)  # fall back to array parse
    else:
        data = json.loads(text)
        if isinstance(data, list):
            edges = data
        elif isinstance(data, dict) and "edges" in data:
            # {"nodes": [...], "edges": [...]}
            for node in data.get("nodes", []):
                G.add_node(str(node))
            edges = data["edges"]
        else:
            raise ValueError(
                "JSON must be an array of edge objects, a {nodes, edges} dict, "
                "or a list of [source, relation, target] triples."
            )

    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            # [source, relation, target] or [source, target]
            src = str(edge[0])
            if len(edge) >= 3:
                rel = str(edge[1])
                tgt = str(edge[2])
            else:
                rel = "RELATED_TO"
                tgt = str(edge[1])
        elif isinstance(edge, dict):
            src = str(edge.get(source_col, "")).strip()
            tgt = str(edge.get(target_col, "")).strip()
            rel = str(edge.get(relation_col, "RELATED_TO")).strip() or "RELATED_TO"
        else:
            continue
        if src and tgt:
            G.add_edge(src, tgt, relation=rel)

    return NetworkXAdapter(G)


def _load_graphml(p: Path, directed: bool) -> NetworkXAdapter:
    G = nx.read_graphml(str(p))
    if directed and not G.is_directed():
        G = G.to_directed()
    elif not directed and G.is_directed():
        G = G.to_undirected()
    # Normalize edge relation attribute
    for u, v, data in G.edges(data=True):
        if "relation" not in data:
            data["relation"] = data.get("label", data.get("type", "RELATED_TO"))
    return NetworkXAdapter(G)


def _load_gexf(p: Path, directed: bool) -> NetworkXAdapter:
    G = nx.read_gexf(str(p))
    if directed and not G.is_directed():
        G = G.to_directed()
    elif not directed and G.is_directed():
        G = G.to_undirected()
    for u, v, data in G.edges(data=True):
        if "relation" not in data:
            data["relation"] = data.get("label", data.get("type", "RELATED_TO"))
    return NetworkXAdapter(G)


def _load_gml(p: Path, directed: bool) -> NetworkXAdapter:
    G = nx.read_gml(str(p))
    if directed and not G.is_directed():
        G = G.to_directed()
    elif not directed and G.is_directed():
        G = G.to_undirected()
    for u, v, data in G.edges(data=True):
        if "relation" not in data:
            data["relation"] = data.get("label", data.get("type", "RELATED_TO"))
    return NetworkXAdapter(G)


def _load_parquet(
    p: Path,
    source_col: str,
    target_col: str,
    relation_col: str,
    directed: bool,
) -> NetworkXAdapter:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load Parquet files: pip install pandas pyarrow")

    df = pd.read_parquet(p)
    return _df_to_adapter(df, source_col, target_col, relation_col, directed, str(p))


def _load_excel(
    p: Path,
    source_col: str,
    target_col: str,
    relation_col: str,
    directed: bool,
) -> NetworkXAdapter:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load Excel files: pip install pandas openpyxl")

    df = pd.read_excel(p)
    return _df_to_adapter(df, source_col, target_col, relation_col, directed, str(p))


def _df_to_adapter(
    df,
    source_col: str,
    target_col: str,
    relation_col: str,
    directed: bool,
    path_hint: str,
) -> NetworkXAdapter:
    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"DataFrame from '{path_hint}' must have '{source_col}' and '{target_col}' columns. "
            f"Found: {list(df.columns)}"
        )
    G = _make_graph(directed)
    for _, row in df.iterrows():
        src = str(row[source_col]).strip()
        tgt = str(row[target_col]).strip()
        if not src or not tgt or src.lower() in ("nan", "none"):
            continue
        rel = str(row.get(relation_col, "RELATED_TO")).strip() if relation_col in df.columns else "RELATED_TO"
        rel = rel if rel and rel.lower() not in ("nan", "none") else "RELATED_TO"
        G.add_edge(src, tgt, relation=rel)
    return NetworkXAdapter(G)

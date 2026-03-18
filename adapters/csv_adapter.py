"""
Bootstrap a Parallax graph from an edge-list CSV file.

Expected CSV format (with header row):
    source,target,relation
    newton,einstein,INFLUENCED
    einstein,bohr,COLLABORATED
    ...

Column names are configurable. The relation column is optional.
"""
import csv
from pathlib import Path
from typing import Optional

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter


def load_csv_adapter(
    path: str,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
    encoding: str = "utf-8",
) -> NetworkXAdapter:
    """
    Load an edge-list CSV into a NetworkXAdapter.

    Parameters
    ----------
    path         : path to the CSV file
    source_col   : column name for the source entity
    target_col   : column name for the target entity
    relation_col : column name for the relation type (optional in CSV)
    directed     : use DiGraph if True, Graph if False
    encoding     : file encoding

    Returns
    -------
    NetworkXAdapter wrapping the loaded graph

    Example:
        from adapters.csv_adapter import load_csv_adapter
        adapter = load_csv_adapter("tests/fixtures/toy_graph.csv")
        print(f"Loaded {adapter.node_count()} nodes")
    """
    G        = nx.DiGraph() if directed else nx.Graph()
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath.resolve()}")

    with open(filepath, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        for row in reader:
            # Skip comment rows (lines starting with #)
            src = row.get(source_col, "").strip()
            tgt = row.get(target_col, "").strip()
            if not src or not tgt or src.startswith("#"):
                continue

            rel = row.get(relation_col, "RELATED_TO").strip() or "RELATED_TO"
            G.add_edge(src, tgt, relation=rel)

    return NetworkXAdapter(G)

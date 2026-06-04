import csv
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter

if TYPE_CHECKING:
    from core.thalamus import IngestionPipeline


def load_csv_adapter(
    path: str,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
    encoding: str = "utf-8",
    pipeline: Optional["IngestionPipeline"] = None,
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
    pipeline     : optional IngestionPipeline for entity/relation normalization
                   and confidence-at-ingest. Any CSV columns beyond source,
                   target, and relation are passed as metadata to the pipeline.
                   If None, raw values are stored as-is (backward-compatible).

    Returns
    -------
    NetworkXAdapter wrapping the loaded graph

    Example:
        from adapters.csv_adapter import load_csv_adapter
        from core.thalamus import IngestionPipeline

        pipeline = IngestionPipeline(
            relation_map={"activates": "ACTIVATES"},
            confidence_fn=lambda s, t, r, m: float(m.get("score", 1.0)),
        )
        adapter = load_csv_adapter("kg.csv", pipeline=pipeline)
    """
    G        = nx.DiGraph() if directed else nx.Graph()
    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath.resolve()}")

    with open(filepath, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path}")

        _key_cols = {source_col, target_col, relation_col}

        for row in reader:
            # Skip comment rows (lines starting with #)
            src = row.get(source_col, "").strip()
            tgt = row.get(target_col, "").strip()
            if not src or not tgt or src.startswith("#"):
                continue

            rel = row.get(relation_col, "RELATED_TO").strip() or "RELATED_TO"

            if pipeline is not None:
                # Extra columns beyond the three key columns become metadata
                meta = {k: v for k, v in row.items() if k not in _key_cols}
                edge = pipeline.process(src, tgt, rel, meta)
                G.add_edge(
                    edge.source,
                    edge.target,
                    relation=edge.relation,
                    confidence=edge.confidence,
                    provenance=edge.provenance,
                    weight=edge.weight,
                    **edge.properties,
                )
            else:
                G.add_edge(src, tgt, relation=rel)

    return NetworkXAdapter(G)

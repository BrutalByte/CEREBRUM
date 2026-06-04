"""
Parquet / Arrow adapter for CEREBRUM â€” load a knowledge graph from columnar data files.

Supports Parquet (.parquet), Feather/Arrow (.arrow, .feather), and
in-memory pandas DataFrames / pyarrow Tables.

Usage
-----
    from adapters.parquet_adapter import load_parquet_adapter

    # From a Parquet file
    adapter = load_parquet_adapter("triples.parquet")

    # Custom column names
    adapter = load_parquet_adapter(
        "edges.parquet",
        source_col="head", target_col="tail", relation_col="predicate",
    )

    # From a pandas DataFrame
    import pandas as pd
    df = pd.DataFrame({"source": ["A", "B"], "target": ["B", "C"], "relation": ["knows", "likes"]})
    adapter = load_parquet_adapter(df)

    # From a pyarrow Table
    import pyarrow as pa
    table = pa.table({"source": ["A"], "target": ["B"], "relation": ["knows"]})
    adapter = load_parquet_adapter(table)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter


def load_parquet_adapter(
    source: Union[str, Path, object],
    *,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
    columns: list = None,
    filters=None,
) -> NetworkXAdapter:
    """
    Load a knowledge graph from a Parquet file, Arrow file, pandas DataFrame,
    or pyarrow Table.

    Parameters
    ----------
    source       : File path (str/Path), pandas DataFrame, or pyarrow Table
    source_col   : Column for source/head entity (default "source")
    target_col   : Column for target/tail entity (default "target")
    relation_col : Column for relation type (default "relation")
    directed     : Use DiGraph if True, Graph if False (default)
    columns      : Optional list of columns to load from file (pushdown projection)
    filters      : Optional pyarrow filters for predicate pushdown
    """
    df = _to_dataframe(source, source_col, target_col, relation_col, columns, filters)
    return _df_to_adapter(df, source_col, target_col, relation_col, directed)


def load_arrow_adapter(
    source: Union[str, Path, object],
    *,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
) -> NetworkXAdapter:
    """Alias for load_parquet_adapter â€” accepts .arrow / .feather files."""
    return load_parquet_adapter(
        source, source_col=source_col, target_col=target_col,
        relation_col=relation_col, directed=directed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_dataframe(source, src_col, tgt_col, rel_col, columns, filters):
    """Normalise input to a pandas DataFrame."""
    # Already a DataFrame
    try:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return source
    except ImportError:
        pass

    # pyarrow Table
    try:
        import pyarrow as pa
        if isinstance(source, pa.Table):
            return source.to_pandas()
    except ImportError:
        pass

    # File path
    path = Path(source)
    suffix = path.suffix.lower()

    read_cols = None
    if columns:
        read_cols = list(columns)
    else:
        # Push down only the columns we need
        read_cols = _needed_cols(path, src_col, tgt_col, rel_col)

    if suffix == ".parquet":
        return _read_parquet(path, read_cols, filters)
    elif suffix in (".arrow", ".feather"):
        return _read_arrow(path, read_cols)
    elif suffix in (".csv", ".tsv"):
        # Graceful fallback for CSV passed by mistake
        import pandas as pd
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, usecols=read_cols)
    else:
        # Try parquet first, then arrow
        try:
            return _read_parquet(path, read_cols, filters)
        except Exception:
            return _read_arrow(path, read_cols)


def _needed_cols(path: Path, src_col, tgt_col, rel_col):
    """Return the minimal column list needed (if they exist in the schema)."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        available = {f.name for f in schema}
        needed = [c for c in [src_col, tgt_col, rel_col] if c in available]
        return needed if needed else None
    except Exception:
        return None


def _read_parquet(path: Path, columns, filters):
    try:
        import pandas as pd
        kwargs = {}
        if columns:
            kwargs["columns"] = columns
        if filters:
            kwargs["filters"] = filters
        return pd.read_parquet(path, **kwargs)
    except ImportError:
        raise ImportError(
            "pandas and pyarrow are required to read Parquet files: "
            "pip install pandas pyarrow"
        )


def _read_arrow(path: Path, columns):
    try:
        import pyarrow.feather as feather
        tbl = feather.read_table(path, columns=columns)
        return tbl.to_pandas()
    except ImportError:
        raise ImportError(
            "pyarrow is required to read Arrow/Feather files: pip install pyarrow"
        )


def _df_to_adapter(df, src_col, tgt_col, rel_col, directed) -> NetworkXAdapter:
    G = nx.DiGraph() if directed else nx.Graph()

    col_names = set(df.columns)
    if src_col not in col_names or tgt_col not in col_names:
        raise ValueError(
            f"Source column {src_col!r} or target column {tgt_col!r} "
            f"not found. Available: {sorted(col_names)}"
        )

    has_rel = rel_col in col_names

    for row in df.itertuples(index=False):
        src = str(getattr(row, src_col, "")).strip()
        tgt = str(getattr(row, tgt_col, "")).strip()
        if not src or not tgt or src == "nan" or tgt == "nan":
            continue
        rel = (
            str(getattr(row, rel_col, "RELATED_TO")).strip()
            if has_rel else "RELATED_TO"
        ) or "RELATED_TO"
        G.add_edge(src, tgt, relation=rel)

    return NetworkXAdapter(G)

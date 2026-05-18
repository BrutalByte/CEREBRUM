"""
SQL adapter for CEREBRUM — load a knowledge graph from any relational database.

Supports SQLite (built-in), PostgreSQL (psycopg2), MySQL (pymysql), and any
other database with a SQLAlchemy-compatible connection URL.

Usage
-----
    # SQLite (zero-install)
    from adapters.sql_adapter import load_sql_adapter
    adapter = load_sql_adapter(
        "sqlite:///my_db.sqlite",
        query="SELECT source, target, relation FROM edges",
    )

    # PostgreSQL
    adapter = load_sql_adapter(
        "postgresql://user:pass@host/dbname",
        query="SELECT head, tail, rel_type FROM knowledge_graph",
        source_col="head", target_col="tail", relation_col="rel_type",
    )

    # From an existing connection
    import sqlite3
    conn = sqlite3.connect("my_db.sqlite")
    adapter = load_sql_adapter(conn, query="SELECT * FROM triples")
"""
from __future__ import annotations

from typing import Optional, Union

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter


def load_sql_adapter(
    connection: Union[str, object],
    query: str,
    *,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    directed: bool = False,
    params: Optional[tuple] = None,
) -> NetworkXAdapter:
    """
    Load a knowledge graph from a SQL query result.

    Parameters
    ----------
    connection   : SQLAlchemy URL string (e.g. "sqlite:///db.sqlite",
                   "postgresql://user:pass@host/db") OR a DBAPI2 connection
                   object (sqlite3, psycopg2, pymysql, etc.)
    query        : SQL SELECT that returns rows with at least source and target
                   columns. The relation column is optional — defaults to
                   "RELATED_TO" if absent.
    source_col   : Column name for the source/head entity (default "source")
    target_col   : Column name for the target/tail entity (default "target")
    relation_col : Column name for the relation type (default "relation")
    directed     : Use DiGraph if True, Graph if False (default)
    params       : Optional query parameters passed to cursor.execute()

    Returns
    -------
    NetworkXAdapter wrapping the loaded graph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    rows, columns = _run_query(connection, query, params)

    col_lower = [c.lower() for c in columns]

    def _idx(name: str) -> Optional[int]:
        try:
            return col_lower.index(name.lower())
        except ValueError:
            return None

    src_i = _idx(source_col)
    tgt_i = _idx(target_col)
    rel_i = _idx(relation_col)

    if src_i is None or tgt_i is None:
        available = ", ".join(columns)
        raise ValueError(
            f"Source column {source_col!r} or target column {target_col!r} "
            f"not found in query result. Available columns: {available}"
        )

    for row in rows:
        src = str(row[src_i]).strip() if row[src_i] is not None else ""
        tgt = str(row[tgt_i]).strip() if row[tgt_i] is not None else ""
        if not src or not tgt:
            continue
        rel = (
            str(row[rel_i]).strip()
            if rel_i is not None and row[rel_i] is not None
            else "RELATED_TO"
        ) or "RELATED_TO"
        G.add_edge(src, tgt, relation=rel)

    return NetworkXAdapter(G)


def load_sql_adapter_from_table(
    connection: Union[str, object],
    table: str,
    *,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: str = "relation",
    where: Optional[str] = None,
    directed: bool = False,
) -> NetworkXAdapter:
    """
    Convenience wrapper: load from a specific table instead of a raw query.

    Parameters
    ----------
    table  : Table name
    where  : Optional WHERE clause (without the WHERE keyword), e.g. "confidence > 0.5"
    """
    cols = f"{source_col}, {target_col}, {relation_col}"
    q = f"SELECT {cols} FROM {table}"
    if where:
        q += f" WHERE {where}"
    return load_sql_adapter(
        connection, q,
        source_col=source_col, target_col=target_col, relation_col=relation_col,
        directed=directed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_query(connection, query: str, params):
    """Execute query and return (rows, column_names) regardless of connection type."""
    # SQLAlchemy URL string
    if isinstance(connection, str):
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(connection)
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                columns = list(result.keys())
                rows = result.fetchall()
            return rows, columns
        except ImportError:
            # Fall back: try sqlite3 for sqlite:// URLs
            if connection.startswith("sqlite:///"):
                import sqlite3
                db_path = connection[len("sqlite:///"):]
                dbconn = sqlite3.connect(db_path)
                return _run_dbapi(dbconn, query, params)
            raise ImportError(
                "sqlalchemy is required for non-SQLite databases: pip install sqlalchemy"
            )

    # DBAPI2 connection object (sqlite3, psycopg2, pymysql, etc.)
    return _run_dbapi(connection, query, params)


def _run_dbapi(conn, query: str, params):
    cursor = conn.cursor()
    cursor.execute(query, params or ())
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return rows, columns

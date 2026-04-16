"""
CEREBRUM API entrypoint — module-level ``app`` for uvicorn / gunicorn.

Environment variables
---------------------
CEREBRUM_CSV_PATH     Path to an edge-list CSV to load at startup (optional).
                      If omitted the server starts empty; upload a graph later
                      via POST /upload/csv.
CEREBRUM_CACHE_PATH   Path prefix for state pickle cache (optional).
                      Speeds up restarts by skipping community detection on
                      graphs that haven't changed.
CEREBRUM_DATA_DIR     Root directory for all persistent data (default: data/cerebrum).
CEREBRUM_API_KEYS     Comma-separated list of accepted API keys (optional).
                      If unset the server runs in open dev mode.
CEREBRUM_WS_PORT      Port for the neural telemetry WebSocket bridge (optional).
                      Set to an integer to enable the UE5 / visualization feed.
CEREBRUM_LARQL_ENDPOINT URL of a LARQL-enabled API for neural context (optional).
CEREBRUM_LARQL_VINDEX   Path to a local LARQL VIndex file (optional).

Usage
-----
    uvicorn api.main:app --host 0.0.0.0 --port 8200
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("cerebrum.main")

from api.server import create_app
from core.graph_adapter import GraphAdapter
from core.embedding_engine import EmbeddingEngine

_csv_path   = os.getenv("CEREBRUM_CSV_PATH")
_cache_path = os.getenv("CEREBRUM_CACHE_PATH")
_ws_port_raw = os.getenv("CEREBRUM_WS_PORT")
_ws_port    = int(_ws_port_raw) if _ws_port_raw else None

if _csv_path:
    from adapters.csv_adapter import load_csv_adapter
    from core.embedding_engine import RandomEngine

    logger.info("Loading graph from %s", _csv_path)
    _adapter: Optional[GraphAdapter] = load_csv_adapter(_csv_path)
    _embedding_engine: Optional[EmbeddingEngine] = RandomEngine(dim=64)
else:
    logger.info(
        "No CEREBRUM_CSV_PATH set ΓÇö starting empty. "
        "Upload a graph via POST /upload/csv."
    )
    _adapter          = None
    _embedding_engine = None

app = create_app(
    adapter=_adapter,
    embedding_engine=_embedding_engine,
    cache_path=_cache_path,
    ws_port=_ws_port,
)

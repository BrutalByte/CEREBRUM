"""
GraphWAL — Phase 227: Write-Ahead Log for live edge additions.

Every add_edge() call appends one JSON record to edges.wal on NVMe.
If the process is killed before REM runs, the next startup replays the WAL
into the in-memory graph — zero edge loss.

After a successful MmapConsolidator.flush(), the WAL is truncated (all
replayed edges are now baked into graph.a / graph.e).

File format: NDJSON, one record per line.
    {"op":"add","ts":1234.5,"src":"A","tgt":"B","rel":"CAUSES",
     "conf":0.9,"prov":"orin","syn":false}

Thread safety: a single threading.Lock serialises all appends and truncations.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("cerebrum.graph_wal")

_WAL_FILENAME = "edges.wal"


class GraphWAL:
    """
    Append-only write-ahead log for graph edge additions.

    Parameters
    ----------
    data_dir : Path
        Directory on NVMe where edges.wal lives.
        The file is created on first append if it doesn't exist.
    """

    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / _WAL_FILENAME
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        src:      str,
        tgt:      str,
        relation: str,
        confidence: float = 1.0,
        provenance: str   = "",
        synthetic:  bool  = False,
    ) -> None:
        """Append one edge addition record. O(1), thread-safe."""
        record = json.dumps({
            "op":   "add",
            "ts":   time.time(),
            "src":  src,
            "tgt":  tgt,
            "rel":  relation,
            "conf": round(float(confidence), 6),
            "prov": provenance,
            "syn":  synthetic,
        }, ensure_ascii=False)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(record + "\n")

    # ------------------------------------------------------------------
    # Read / replay path
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        """Yield parsed records. Skips malformed lines."""
        if not self._path.exists():
            return
        with self._lock:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError:
                return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("GraphWAL: skipping malformed record: %.80s", line)

    def replay(self, adapter) -> int:
        """
        Replay all WAL records into *adapter* by calling adapter.add_edge().
        Returns the number of edges replayed.
        Silently skips records whose op is not "add".
        """
        if not self._path.exists():
            return 0

        count = 0
        for rec in self:
            if rec.get("op") != "add":
                continue
            try:
                adapter.add_edge(
                    rec["src"],
                    rec["tgt"],
                    rec["rel"],
                    confidence  = rec.get("conf", 1.0),
                    provenance  = rec.get("prov", "wal_replay"),
                    synthetic   = rec.get("syn", False),
                )
                count += 1
            except Exception as exc:
                logger.warning("GraphWAL.replay: skipping edge %s→%s: %s",
                               rec.get("src"), rec.get("tgt"), exc)
        if count:
            logger.info("GraphWAL: replayed %d edges from %s", count, self._path)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def truncate(self) -> None:
        """
        Remove the WAL file. Called after MmapConsolidator.flush() so the
        replayed edges are now in graph.a/graph.e and the WAL is redundant.
        """
        with self._lock:
            try:
                if self._path.exists():
                    self._path.unlink()
                    logger.info("GraphWAL: truncated %s", self._path)
            except OSError as exc:
                logger.warning("GraphWAL: truncate failed: %s", exc)

    def entry_count(self) -> int:
        """Return the number of records in the WAL (O(n) scan)."""
        if not self._path.exists():
            return 0
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def size_bytes(self) -> int:
        """Current WAL file size in bytes, 0 if missing."""
        try:
            return self._path.stat().st_size
        except OSError:
            return 0

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

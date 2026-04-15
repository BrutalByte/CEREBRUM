"""
Persistence layer for CEREBRUM — save/load graph state, communities, metadata,
and the durable query log.

CEREBRUM states (especially DSCF communities and structural encodings) can
be expensive to compute on large graphs. This module provides a unified
mechanism to serialize the current state to disk and reload it instantly.

QueryLog
--------
Append-only NDJSON record of every completed reasoning query.  Each line is a
JSON object: {timestamp, seeds, answers:[{entity, score, path}], rel_sequences}.
The log feeds Engram on startup (call QueryLog.replay_into_cache(cache)) so
learned relation patterns survive process restarts.
"""
import json
import pickle
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Security: Define the root for all persistent data. 
# In production, this should be configurable but restricted.
SAFE_DATA_DIR = Path(os.getenv("CEREBRUM_DATA_DIR", "data/cerebrum")).absolute()

def _resolve_safe_path(file_path: str) -> Path:
    """
    Ensure the file_path is within the SAFE_DATA_DIR to prevent path traversal.
    """
    requested_path = Path(file_path)
    # If the user passed an absolute path, we only allow it if it starts with SAFE_DATA_DIR
    if requested_path.is_absolute():
        final_path = requested_path.resolve()
    else:
        # Join relative path to our sandbox
        final_path = (SAFE_DATA_DIR / requested_path).resolve()

    if not str(final_path).startswith(str(SAFE_DATA_DIR)):
        raise PermissionError(f"Security: Path traversal attempt blocked: {file_path}")
    
    return final_path

def save_state(
    file_path: str,
    adapter: Any,
    community_map: Dict[str, int],
    embeddings: Dict[str, np.ndarray],
    csa_metadata: Dict[str, Any],
    default_edge_type_weights: Optional[Dict[str, float]] = None,
    hologram: Optional[Any] = None,
) -> None:
    """
    Serialize the entire CEREBRUM state to a binary pickle file.
    Only allows paths within the SAFE_DATA_DIR sandbox.
    """
    path = _resolve_safe_path(file_path)
    
    state = {
        "version": "0.2.0",
        "timestamp": time.time(),
        "adapter": adapter,
        "community_map": community_map,
        "embeddings": embeddings,
        "csa_metadata": csa_metadata,
        "default_edge_type_weights": default_edge_type_weights,
        "hologram": hologram,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(state, f)
    
    print(f"  [Persistence] State saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_state(file_path: str) -> Dict[str, Any]:
    """
    Load a CEREBRUM state from a pickle file.
    Only allows paths within the SAFE_DATA_DIR sandbox.
    """
    path = _resolve_safe_path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CEREBRUM state file not found: {path}")
    
    t0 = time.time()
    with open(path, "rb") as f:
        state = pickle.load(f)
    
    print(f"  [Persistence] State loaded from {path} in {time.time() - t0:.2f}s")
    return state


def is_state_cached(file_path: str) -> bool:
    """Return True if the state file exists and is within sandbox."""
    try:
        path = _resolve_safe_path(file_path)
        return path.exists()
    except PermissionError:
        return False


# ---------------------------------------------------------------------------
# QueryLog — durable, append-only query history
# ---------------------------------------------------------------------------

class QueryLog:
    """
    Append-only NDJSON log of completed CEREBRUM reasoning queries.

    Each record captures the seeds, top-K answer entity IDs, scores, path
    node sequences, and the compressed relation-type sequences used by
    Engram.  The log survives process restarts and is the authoritative
    source for warming up Engram at startup.

    Parameters
    ----------
    path : path to the NDJSON log file (created on first write if absent).
           Defaults to ``data/cerebrum/query_log.ndjson`` inside SAFE_DATA_DIR.
    max_entries : if > 0, the log is trimmed to the most recent max_entries
                  records when it grows beyond 2× that size (default 10 000).
    """

    _DEFAULT = "query_log.ndjson"

    def __init__(
        self,
        path: Optional[str] = None,
        max_entries: int = 10_000,
    ) -> None:
        # QueryLog paths are user-configurable (not sandbox-restricted like
        # graph state).  Use the default SAFE_DATA_DIR location when no path
        # is given; accept any explicit path as-is.
        if path is None:
            self._path = SAFE_DATA_DIR / self._DEFAULT
        else:
            self._path = Path(path)
        self.max_entries = max_entries
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        seeds: List[str],
        answers: List[Any],          # Answer objects from answer_extractor
        min_score: float = 0.0,
    ) -> None:
        """
        Append one query result to the log.

        Parameters
        ----------
        seeds     : seed entity IDs used for this query
        answers   : list of Answer objects (answer.entity_id, answer.score,
                    answer.best_path.nodes)
        min_score : only log answers above this score threshold
        """
        entries: List[Dict] = []
        for ans in answers:
            if ans.score < min_score:
                continue
            path = getattr(ans, "best_path", None)
            nodes = list(path.nodes) if path is not None else []
            # Extract relation sequence (odd-indexed elements)
            rel_seq = [nodes[i] for i in range(1, len(nodes), 2)]
            entries.append({
                "entity": ans.entity_id,
                "score":  round(float(ans.score), 6),
                "path":   nodes,
                "rels":   rel_seq,
            })

        if not entries:
            return

        record = {
            "ts":      time.time(),
            "seeds":   seeds,
            "answers": entries,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Trim if grown too large
        if self.max_entries > 0:
            self._maybe_trim()

    # ------------------------------------------------------------------
    # Read / replay
    # ------------------------------------------------------------------

    def read_recent(self, n: int = 1000) -> List[Dict]:
        """Return the last n log records (most recent last)."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def replay_into_cache(self, cache: Any, min_score: float = 0.3) -> int:
        """
        Warm up an Engram from the persisted query log.

        Reads all log records, extracts relation sequences from answers that
        meet *min_score*, and feeds them into *cache* via cache.record().

        Parameters
        ----------
        cache     : Engram instance
        min_score : only replay answers at or above this score

        Returns
        -------
        Number of relation sequences replayed.
        """
        replayed = 0
        for rec in self.read_recent(n=self.max_entries):
            for ans in rec.get("answers", []):
                if ans.get("score", 0.0) < min_score:
                    continue
                rel_seq = tuple(ans.get("rels", []))
                if rel_seq:
                    weight = max(1, int(ans["score"] * 10))
                    cache.record(rel_seq, weight=weight)
                    replayed += 1
        return replayed

    def count(self) -> int:
        """Return the total number of log records."""
        if not self._path.exists():
            return 0
        return sum(1 for ln in self._path.read_text(encoding="utf-8").splitlines() if ln.strip())

    def clear(self) -> None:
        """Delete all log records."""
        if self._path.exists():
            self._path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _maybe_trim(self) -> None:
        """Keep only the most recent max_entries records."""
        if not self._path.exists():
            return
        lines = [ln for ln in self._path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) > self.max_entries * 2:
            kept = lines[-self.max_entries:]
            self._path.write_text("\n".join(kept) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# GraphSnapshot — portable JSON topology snapshot (Phase 81)
# ---------------------------------------------------------------------------

class GraphSnapshot:
    """
    Lightweight, format-stable snapshot of graph topology (nodes + edges).

    Complements ``save_state()`` / ``load_state()`` (which pickle the full
    CEREBRUM state) with a human-readable, adapter-version-independent format
    that captures *only* topology — no embeddings or community structure.

    Key use cases
    -------------
    - Make ResearchAgent-materialized edges durable across restarts.
    - Migrate graphs between adapter backends (export → re-import).
    - Pair with ProvenanceLedger: snapshot after each approved batch to create
      a point-in-time restore capability.

    Snapshot format (JSON)
    ----------------------
    ::

        {
          "version": "1.0",
          "saved_at": <unix timestamp>,
          "node_count": <int>,
          "edge_count": <int>,
          "nodes": [{"id": "...", "label": "...", "type": "...", "properties": {...}}],
          "edges": [{"source": "...", "target": "...", "relation": "...",
                     "confidence": 1.0, "provenance": "...", "synthetic": false,
                     "weight": 1.0}]
        }

    Usage
    -----
    ::

        snap = GraphSnapshot()
        stats = snap.save(adapter, "data/graph_snapshot.json")
        # ... restart / new adapter ...
        result = snap.restore("data/graph_snapshot.json", new_adapter)
        print(result)  # {"added": 120, "skipped": 0, "errors": 0}
    """

    VERSION = "1.0"

    # Core edge attribute keys to capture; others are ignored
    _EDGE_KEYS = ("relation", "confidence", "provenance", "synthetic", "weight",
                  "valid_from", "valid_to")

    def save(self, adapter, path: str) -> Dict[str, Any]:
        """
        Serialize all nodes and edges in *adapter* to a JSON file at *path*.

        Parameters
        ----------
        adapter : any GraphAdapter (must implement ``to_networkx()``).
        path    : destination file path (created with parent dirs as needed).

        Returns
        -------
        dict with keys ``node_count``, ``edge_count``, ``path``.
        """
        G = adapter.to_networkx()

        nodes = []
        for nid, attrs in G.nodes(data=True):
            nodes.append({
                "id": nid,
                "label": attrs.get("label", nid),
                "type": attrs.get("type", "entity"),
                "properties": {k: v for k, v in attrs.items()
                               if k not in ("label", "type")},
            })

        edges = []
        if G.is_multigraph():
            edge_iter = (
                (u, v, data)
                for u, v, data in G.edges(data=True)
            )
        else:
            edge_iter = G.edges(data=True)

        for u, v, data in edge_iter:
            entry: Dict[str, Any] = {"source": u, "target": v}
            for k in self._EDGE_KEYS:
                val = data.get(k)
                if val is not None:
                    # numpy scalars → Python scalars for JSON serialization
                    if hasattr(val, "item"):
                        val = val.item()
                    entry[k] = val
            edges.append(entry)

        snapshot = {
            "version": self.VERSION,
            "saved_at": time.time(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False),
                        encoding="utf-8")

        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "path": str(dest),
        }

    def restore(
        self,
        path: str,
        adapter,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """
        Re-add edges from a saved snapshot into *adapter*.

        Parameters
        ----------
        path          : path to the JSON snapshot file.
        adapter       : target GraphAdapter — must implement ``add_edge()``.
        skip_existing : if True (default), check whether the edge already
                        exists (via ``adapter.to_networkx().has_edge()``)
                        before calling ``add_edge()``.

        Returns
        -------
        dict with keys ``added``, ``skipped``, ``errors``.
        """
        raw = self.load_raw(path)
        G_existing = adapter.to_networkx() if skip_existing else None

        added = skipped = errors = 0

        for edge in raw.get("edges", []):
            u = edge.get("source", "")
            v = edge.get("target", "")
            relation = edge.get("relation", "RELATED_TO")

            if not u or not v:
                errors += 1
                continue

            if skip_existing and G_existing is not None and G_existing.has_edge(u, v):
                skipped += 1
                continue

            try:
                adapter.add_edge(
                    u, v,
                    relation=relation,
                    confidence=float(edge.get("confidence", 1.0)),
                    provenance=str(edge.get("provenance", "snapshot")),
                    synthetic=bool(edge.get("synthetic", False)),
                )
                added += 1
            except Exception:
                errors += 1

        return {"added": added, "skipped": skipped, "errors": errors}

    @staticmethod
    def load_raw(path: str) -> Dict[str, Any]:
        """Return the raw JSON dict from a snapshot file (no adapter required)."""
        data = Path(path).read_text(encoding="utf-8")
        return json.loads(data)

    def diff(self, path_a: str, path_b: str) -> Dict[str, Any]:
        """
        Compare two snapshots and return added / removed edge sets.

        Returns
        -------
        dict with keys:
          - ``edges_added``   : edges in B but not A (list of dicts)
          - ``edges_removed`` : edges in A but not B (list of dicts)
          - ``node_delta``    : node_count(B) - node_count(A)
          - ``edge_delta``    : edge_count(B) - edge_count(A)
        """
        snap_a = self.load_raw(path_a)
        snap_b = self.load_raw(path_b)

        def _key(e: Dict) -> Tuple[str, str, str]:
            return (e.get("source", ""), e.get("target", ""),
                    e.get("relation", ""))

        set_a = {_key(e) for e in snap_a.get("edges", [])}
        set_b = {_key(e) for e in snap_b.get("edges", [])}

        idx_b = {_key(e): e for e in snap_b.get("edges", [])}
        idx_a = {_key(e): e for e in snap_a.get("edges", [])}

        return {
            "edges_added":   [idx_b[k] for k in set_b - set_a],
            "edges_removed":  [idx_a[k] for k in set_a - set_b],
            "node_delta": snap_b.get("node_count", 0) - snap_a.get("node_count", 0),
            "edge_delta": snap_b.get("edge_count", 0) - snap_a.get("edge_count", 0),
        }

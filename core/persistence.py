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
The log feeds AAAKCache on startup (call QueryLog.replay_into_cache(cache)) so
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
    AAAKCache.  The log survives process restarts and is the authoritative
    source for warming up AAAKCache at startup.

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
        Warm up an AAAKCache from the persisted query log.

        Reads all log records, extracts relation sequences from answers that
        meet *min_score*, and feeds them into *cache* via cache.record().

        Parameters
        ----------
        cache     : AAAKCache instance
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

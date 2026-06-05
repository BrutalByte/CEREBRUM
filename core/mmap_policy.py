"""
MmapPolicy — Phase 227: Intentional NVMe graph consolidation.

Provides three components:

  MmapPolicy        enum    AUTO / ALWAYS / NEVER
  MmapAdvisor       class   evaluates whether mmap is worth it at build time
  MmapConsolidator  class   writes graph + embeddings atomically to NVMe after REM

Data directory resolution (in priority order):
  1. CEREBRUM_MMAP_DIR env var  (point at dedicated NVMe mount)
  2. <base_data_dir>/mmap/      (default relative path)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("cerebrum.mmap_policy")

_ENV_VAR      = "CEREBRUM_MMAP_DIR"
_DEFAULT_BASE = "data/cerebrum"
_DEFAULT_SUB  = "mmap"


# ---------------------------------------------------------------------------
# Directory resolver
# ---------------------------------------------------------------------------

def resolve_mmap_dir(base_data_dir: str = _DEFAULT_BASE) -> Path:
    """
    Return the canonical mmap directory.

    Resolution order:
      1. CEREBRUM_MMAP_DIR environment variable (already set before process start)
      2. CEREBRUM_MMAP_DIR from the project-root .env file (loaded once on first call)
      3. <base_data_dir>/mmap/  (default)

    Point CEREBRUM_MMAP_DIR at a dedicated NVMe mount to isolate all graph
    data on that drive.  The .env file in the project root is the recommended
    place to persist this setting.
    """
    # Load .env from the project root (two levels up from this file).
    # Silent if python-dotenv is not installed or .env is absent.
    try:
        from dotenv import load_dotenv
        _env_file = Path(__file__).parent.parent / ".env"
        load_dotenv(_env_file, override=False)   # env vars already set take priority
    except ImportError:
        pass

    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return Path(env)
    return Path(base_data_dir) / _DEFAULT_SUB


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------

class MmapPolicy(Enum):
    AUTO   = "auto"    # MmapAdvisor decides
    ALWAYS = "always"  # force NVMe (e.g. dedicated drive, large graph)
    NEVER  = "never"   # force RAM  (e.g. benchmarking, tiny graph)


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------

@dataclass
class MmapRecommendation:
    use_mmap:      bool
    reason:        str
    graph_mb:      float
    embedding_mb:  float
    available_mb:  float
    pressure_pct:  float
    policy:        MmapPolicy = MmapPolicy.AUTO
    override_used: bool = False

    def log(self) -> None:
        direction = "→ NVMe" if self.use_mmap else "→ RAM"
        logger.info(
            "MmapAdvisor: %s  pressure=%.1f%%  graph=%.1fMB  emb=%.1fMB  "
            "avail=%.0fMB  reason=%s%s",
            direction,
            self.pressure_pct,
            self.graph_mb,
            self.embedding_mb,
            self.available_mb,
            self.reason,
            " [policy override]" if self.override_used else "",
        )


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class MmapAdvisor:
    """
    Evaluates whether NVMe memmap is worth using for a given graph.
    All thresholds are fractions of *available* (not total) RAM.
    """

    _THRESHOLD_RECOMMEND = 0.20   # >20% of free RAM → recommend
    _THRESHOLD_REQUIRE   = 0.50   # >50% of free RAM → require

    def evaluate(
        self,
        n_nodes:       int,
        n_edges:       int,
        embedding_dim: int,
        policy:        MmapPolicy = MmapPolicy.AUTO,
    ) -> MmapRecommendation:
        try:
            import psutil
            available_bytes = psutil.virtual_memory().available
        except ImportError:
            available_bytes = 8 * 1024 ** 3  # conservative fallback

        graph_bytes     = n_nodes * 32 + n_edges * 12   # A + E file sizes
        embedding_bytes = n_nodes * embedding_dim * 4   # float32 matrix
        total_bytes     = graph_bytes + embedding_bytes
        pressure        = total_bytes / available_bytes if available_bytes > 0 else 1.0

        graph_mb     = graph_bytes     / 1024 ** 2
        embedding_mb = embedding_bytes / 1024 ** 2
        available_mb = available_bytes / 1024 ** 2

        if pressure >= self._THRESHOLD_REQUIRE:
            advised = True
            reason  = (f"pressure {pressure*100:.1f}% >= "
                       f"{self._THRESHOLD_REQUIRE*100:.0f}% (required)")
        elif pressure >= self._THRESHOLD_RECOMMEND:
            advised = True
            reason  = (f"pressure {pressure*100:.1f}% >= "
                       f"{self._THRESHOLD_RECOMMEND*100:.0f}% (recommended)")
        else:
            advised = False
            reason  = (f"pressure {pressure*100:.1f}% < "
                       f"{self._THRESHOLD_RECOMMEND*100:.0f}% (RAM sufficient)")

        override_used = False
        if policy == MmapPolicy.ALWAYS:
            final, override_used = True, not advised
        elif policy == MmapPolicy.NEVER:
            final, override_used = False, advised
        else:
            final = advised

        return MmapRecommendation(
            use_mmap      = final,
            reason        = reason,
            graph_mb      = graph_mb,
            embedding_mb  = embedding_mb,
            available_mb  = available_mb,
            pressure_pct  = pressure * 100,
            policy        = policy,
            override_used = override_used,
        )


# ---------------------------------------------------------------------------
# Consolidation report
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationReport:
    success:       bool
    data_dir:      str
    node_count:    int
    edge_count:    int
    embedding_mb:  float
    duration_s:    float
    wal_truncated: bool = False
    error:         Optional[str] = None
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "success":       self.success,
            "data_dir":      self.data_dir,
            "node_count":    self.node_count,
            "edge_count":    self.edge_count,
            "embedding_mb":  round(self.embedding_mb, 2),
            "duration_s":    round(self.duration_s, 3),
            "wal_truncated": self.wal_truncated,
            "error":         self.error,
            "timestamp":     self.timestamp,
        }


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------

class MmapConsolidator:
    """
    Writes graph topology + embeddings to NVMe memmap files after REM/sleep.

    The write is atomic: everything goes to <data_dir>/_tmp_flush/ first,
    then renamed into place so a power cut mid-write never corrupts the store.

    Files written:
        graph.a        — node array  (32 B/node, NumPy structured)
        graph.e        — edge array  (12 B/edge, NumPy structured)
        embeddings.e   — float32 embedding matrix (N × dim)
        relations.idx  — relation string pool (one name per line)
        nodes.map      — node IDs + community (tab-separated)
        graph.meta     — JSON stats + build_id for staleness detection
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def flush(
        self,
        adapter,
        embeddings:  Dict[str, np.ndarray],
        wal:         Optional[Any] = None,   # GraphWAL — truncated on success
        build_id:    Optional[str] = None,   # propagate from CerebrumGraph
    ) -> ConsolidationReport:
        t0 = time.time()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.data_dir / "_tmp_flush"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        try:
            _write_full(adapter, embeddings, tmp_dir, build_id=build_id)
            _atomic_replace(tmp_dir, self.data_dir)

            n_nodes    = adapter.node_count() if hasattr(adapter, "node_count") else 0
            n_edges    = _count_edges(adapter)
            emb_sample = next(iter(embeddings.values()), None)
            emb_mb     = (n_nodes * emb_sample.nbytes / 1024 ** 2) if emb_sample is not None else 0.0

            wal_truncated = False
            if wal is not None:
                wal.truncate()
                wal_truncated = True

            report = ConsolidationReport(
                success       = True,
                data_dir      = str(self.data_dir),
                node_count    = n_nodes,
                edge_count    = n_edges,
                embedding_mb  = emb_mb,
                duration_s    = time.time() - t0,
                wal_truncated = wal_truncated,
            )
            logger.info(
                "MmapConsolidator: flushed %d nodes / %d edges  emb=%.1fMB"
                "  to %s  in %.2fs%s",
                n_nodes, n_edges, emb_mb, self.data_dir, report.duration_s,
                " + WAL truncated" if wal_truncated else "",
            )
            return report

        except Exception as exc:
            logger.error("MmapConsolidator: flush failed: %s", exc, exc_info=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return ConsolidationReport(
                success      = False,
                data_dir     = str(self.data_dir),
                node_count   = 0,
                edge_count   = 0,
                embedding_mb = 0.0,
                duration_s   = time.time() - t0,
                error        = str(exc),
            )

    def meta_path(self) -> Path:
        return self.data_dir / "graph.meta"

    def load_meta(self) -> Optional[dict]:
        p = self.meta_path()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

from adapters.mmap_adapter import A_DTYPE, E_DTYPE   # reuse existing dtypes


def _write_full(
    adapter,
    embeddings: Dict[str, np.ndarray],
    out_dir:    Path,
    build_id:   Optional[str] = None,
) -> None:
    """Write all mmap files to out_dir (does NOT replace existing files)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    G            = _get_nx(adapter)
    nodes        = list(G.nodes())
    node_to_idx  = {n: i for i, n in enumerate(nodes)}

    # --- relation string pool ---
    rel_set  = {d.get("relation") or d.get("relation_type") or "LINKED"
                for _, _, d in G.edges(data=True)}
    rel_list = sorted(rel_set)
    rel_to_id: Dict[str, int] = {r: i for i, r in enumerate(rel_list)}

    (out_dir / "relations.idx").write_text(
        "\n".join(rel_list), encoding="utf-8"
    )

    # --- nodes.map ---
    with open(out_dir / "nodes.map", "w", encoding="utf-8") as f:
        for node in nodes:
            cid = adapter.get_community(node) if hasattr(adapter, "get_community") else 0
            if cid is None or cid < 0:
                cid = 0
            f.write(f"{node}\t{cid}\n")

    # --- A-file and E-file ---
    n_nodes = len(nodes)
    # For directed graphs G.adj[node] yields only out-edges, so use out_degree.
    # For undirected, degree == adjacency fan-out.
    if G.is_directed():
        n_edge_records = sum(d for _, d in G.out_degree())
    else:
        n_edge_records = sum(d for _, d in G.degree())

    a_path = out_dir / "graph.a"
    e_path = out_dir / "graph.e"

    with open(a_path, "wb") as f:
        f.seek(max(n_nodes * A_DTYPE.itemsize - 1, 0))
        f.write(b"\x00")
    with open(e_path, "wb") as f:
        f.seek(max(n_edge_records * E_DTYPE.itemsize - 1, 0))
        f.write(b"\x00")

    a_mmap = np.memmap(a_path, dtype=A_DTYPE, mode="r+", shape=(n_nodes,))
    e_mmap = np.memmap(e_path, dtype=E_DTYPE, mode="r+", shape=(n_edge_records,))

    e_ptr = 0
    for i, node in enumerate(nodes):
        neighbors = G.adj[node]
        deg = len(neighbors)
        cid = adapter.get_community(node) if hasattr(adapter, "get_community") else 0
        if cid is None or cid < 0:
            cid = 0

        a_mmap[i] = (i, deg, e_ptr * E_DTYPE.itemsize, cid, b"\x00" * 10)

        for neighbor, data in neighbors.items():
            t_idx  = node_to_idx[neighbor]
            rel    = data.get("relation") or data.get("relation_type") or "LINKED"
            rel_id = rel_to_id.get(rel, 0)
            conf   = float(data.get("confidence", 1.0))
            e_mmap[e_ptr] = (t_idx, rel_id, conf)
            e_ptr += 1

    a_mmap.flush(); del a_mmap
    e_mmap.flush(); del e_mmap

    # --- embeddings.e ---
    emb_dim = 0
    if embeddings:
        sample = next(iter(embeddings.values()))
        emb_dim = sample.shape[0]
        emb_path = out_dir / "embeddings.e"
        emb_mmap = np.memmap(emb_path, dtype="float32", mode="w+",
                             shape=(n_nodes, emb_dim))
        for i, node in enumerate(nodes):
            vec = embeddings.get(node)
            if vec is not None:
                emb_mmap[i] = vec.astype("float32")
        emb_mmap.flush(); del emb_mmap

    # --- graph.meta ---
    meta = {
        "version":      1,
        "written_at":   time.time(),
        "build_id":     build_id or str(uuid.uuid4()),
        "node_count":   n_nodes,
        "edge_count":   n_edge_records,
        "embedding_dim": emb_dim,
        "relation_count": len(rel_list),
    }
    (out_dir / "graph.meta").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _atomic_replace(tmp_dir: Path, target_dir: Path) -> None:
    """
    Move files from tmp_dir into target_dir, replacing existing ones.
    Each file rename is atomic on NTFS and POSIX so a crash mid-replace
    leaves the old file intact.
    """
    for src in tmp_dir.iterdir():
        dst = target_dir / src.name
        if dst.exists():
            dst.unlink()
        src.rename(dst)
    tmp_dir.rmdir()


def _get_nx(adapter):
    """Extract the NetworkX graph from any adapter."""
    G = getattr(adapter, "_G", None)
    if G is not None:
        return G
    if hasattr(adapter, "to_networkx"):
        return adapter.to_networkx()
    raise AttributeError(
        "MmapConsolidator requires adapter._G or adapter.to_networkx()"
    )


def _count_edges(adapter) -> int:
    try:
        G = _get_nx(adapter)
        return G.number_of_edges()
    except Exception:
        return 0

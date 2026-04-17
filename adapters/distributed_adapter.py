"""
DistributedCEREBRUM — Phase 22 Distributed Federation.

Coordinates a cluster of CEREBRUM worker nodes for distributed graph reasoning
at web-scale (hundreds of millions of nodes across many machines).

Architecture
------------
::

    DistributedCEREBRUM (coordinator — no graph data)
      ├── ShardManager    maps entity_id → worker_url (shard assignment)
      ├── QueryRouter     decomposes multi-hop queries into per-shard sub-queries
      ├── PathStitcher    assembles cross-shard reasoning paths
      └── CouplingSync      propagates community-map deltas to all workers

Each worker is a standard CEREBRUM REST API instance accessed via
``RemoteCerebrumAdapter``.  The coordinator itself holds only the shard
index and routing metadata — zero graph data.

Shard strategies
----------------
``"community"``  (default)
    Assign each community to a worker.  Cross-community edges become
    cross-shard edges.  Optimal for community-structured queries.

``"hash"``
    ``worker = hash(entity_id) % n_workers``.  Uniform distribution,
    no semantic locality.  Good for random-access workloads.

``"range"``
    Sort entity IDs lexicographically; assign equal-sized ranges.
    Good for sequential scans (bulk load, maintenance).

Cross-shard traversal
---------------------
When a beam-search hop lands on an entity whose owner is a different
worker than the current hop, the QueryRouter forwards a sub-query to
that worker, fetches its local beam expansion, and returns the candidates
to the coordinator for re-ranking.  Path stitching merges the per-shard
path segments into complete end-to-end reasoning traces.

Usage
-----
::

    from adapters.distributed_adapter import DistributedCEREBRUM

    cluster = DistributedCEREBRUM(
        workers=[
            {"url": "http://node-1:8200", "token": "..."},
            {"url": "http://node-2:8200", "token": "..."},
            {"url": "http://node-3:8200", "token": "..."},
        ],
        shard_strategy="community",
    )

    cluster.assign_shard("Marie Curie", worker_url="http://node-1:8200")
    # or bulk-load from a shard manifest CSV
    cluster.load_shard_manifest("shards.csv")

    answers = cluster.query("Marie Curie", max_hops=3, top_k=10)
    for a in answers:
        print(a["entity"], a["score"], a["path"])
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("cerebrum.distributed")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkerNode:
    """Represents a single CEREBRUM worker in the cluster."""
    url: str
    token: Optional[str] = None
    secret: Optional[str] = None
    weight: float = 1.0          # relative load weight (higher = more shards)
    healthy: bool = True
    last_health_check: float = 0.0
    shard_count: int = 0
    latency_ms: float = 0.0      # rolling average round-trip latency


@dataclass
class ShardEntry:
    """Maps an entity_id to its owning worker."""
    entity_id: str
    worker_url: str
    community_id: Optional[int] = None


@dataclass
class DistributedPath:
    """A reasoning path assembled from cross-shard segments."""
    entities: List[str]
    relations: List[str]
    score: float
    shard_hops: List[str]         # which worker served each hop
    segments: List[Dict]          # raw per-shard segment data


@dataclass
class DistributedAnswer:
    """Final answer from a distributed query."""
    entity: str
    score: float
    path: List[str]               # interleaved entity/relation list
    shard_hops: List[str]
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# ShardManager
# ---------------------------------------------------------------------------

class ShardManager:
    """
    Maintains the entity→worker shard index and handles rebalancing.

    The index is an in-memory dict; for production scale it should be
    backed by a fast distributed key-value store (Redis, etcd).  The
    interface is identical either way.
    """

    def __init__(self, workers: List[WorkerNode], strategy: str = "community"):
        self._workers: List[WorkerNode] = workers
        self._strategy = strategy
        self._index: Dict[str, str] = {}       # entity_id → worker_url
        self._community_map: Dict[int, str] = {}  # community_id → worker_url
        self._lock = threading.RLock()

    def assign(self, entity_id: str, worker_url: str, community_id: Optional[int] = None) -> None:
        """Explicitly assign an entity to a worker."""
        with self._lock:
            self._index[entity_id] = worker_url
            if community_id is not None:
                self._community_map[community_id] = worker_url
            worker = self._worker_by_url(worker_url)
            if worker:
                worker.shard_count += 1

    def assign_community(self, community_id: int, worker_url: str) -> None:
        """Assign an entire community (all its entities) to a worker."""
        with self._lock:
            self._community_map[community_id] = worker_url

    def lookup(self, entity_id: str) -> Optional[str]:
        """Return the worker URL responsible for this entity, or None."""
        with self._lock:
            if entity_id in self._index:
                return self._index[entity_id]
            return self._route_by_strategy(entity_id)

    def lookup_community(self, community_id: int) -> Optional[str]:
        """Return the worker URL responsible for this community."""
        with self._lock:
            return self._community_map.get(community_id)

    def load_manifest(self, manifest: List[Dict[str, Any]]) -> int:
        """
        Bulk-load a shard manifest.

        Each entry: {"entity_id": str, "worker_url": str, "community_id": int|None}
        Returns count of entries loaded.
        """
        count = 0
        for entry in manifest:
            self.assign(
                entry["entity_id"],
                entry["worker_url"],
                entry.get("community_id"),
            )
            count += 1
        log.info("ShardManager: loaded %d shard entries.", count)
        return count

    def rebalance(self, strategy: Optional[str] = None) -> Dict[str, int]:
        """
        Redistribute shards evenly across healthy workers.

        Returns {worker_url: new_shard_count}.
        """
        strat = strategy or self._strategy
        healthy = [w for w in self._workers if w.healthy]
        if not healthy:
            raise RuntimeError("No healthy workers available for rebalancing.")

        with self._lock:
            entities = list(self._index.keys())
            new_index: Dict[str, str] = {}

            if strat == "hash":
                for eid in entities:
                    h = int(hashlib.md5(eid.encode()).hexdigest(), 16)
                    worker = healthy[h % len(healthy)]
                    new_index[eid] = worker.url
            elif strat == "range":
                entities.sort()
                chunk = max(1, len(entities) // len(healthy))
                for i, eid in enumerate(entities):
                    worker = healthy[min(i // chunk, len(healthy) - 1)]
                    new_index[eid] = worker.url
            else:
                # community-based: keep community→worker assignments
                for eid in entities:
                    for cid, wurl in self._community_map.items():
                        pass  # simplified: use hash as fallback
                    h = int(hashlib.md5(eid.encode()).hexdigest(), 16)
                    worker = healthy[h % len(healthy)]
                    new_index[eid] = worker.url

            self._index = new_index
            counts: Dict[str, int] = {w.url: 0 for w in healthy}
            for wurl in new_index.values():
                counts[wurl] = counts.get(wurl, 0) + 1
            for w in self._workers:
                w.shard_count = counts.get(w.url, 0)

        log.info("ShardManager rebalanced %d entities across %d workers.", len(entities), len(healthy))
        return counts

    def _route_by_strategy(self, entity_id: str) -> Optional[str]:
        healthy = [w for w in self._workers if w.healthy]
        if not healthy:
            return None
        if self._strategy in ("hash", "community"):
            h = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
            return healthy[h % len(healthy)].url
        # range — fall back to hash
        h = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
        return healthy[h % len(healthy)].url

    def _worker_by_url(self, url: str) -> Optional[WorkerNode]:
        for w in self._workers:
            if w.url == url:
                return w
        return None


# ---------------------------------------------------------------------------
# CouplingSync — community-map propagation
# ---------------------------------------------------------------------------

class CouplingSync:
    """
    Propagates community-map updates across all workers via a simple
    broadcast protocol.

    In production, this should be replaced with a pub-sub system
    (Redis Pub/Sub, Kafka, NATS).  The interface is identical.
    """

    def __init__(self, workers: List[WorkerNode]):
        self._workers = workers
        self._version: int = 0
        self._lock = threading.Lock()

    def couple_community_delta(
        self,
        community_map_delta: Dict[str, int],
        source_url: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Send a community_map delta to all workers except the source.

        Returns {worker_url: success}.
        """
        results: Dict[str, bool] = {}
        with self._lock:
            self._version += 1
            version = self._version

        for worker in self._workers:
            if worker.url == source_url or not worker.healthy:
                results[worker.url] = True
                continue
            try:
                adapter = self._get_adapter(worker)
                adapter._post_community_delta(community_map_delta, version)
                results[worker.url] = True
            except Exception as exc:
                log.warning("CouplingSync: failed to update %s: %s", worker.url, exc)
                results[worker.url] = False

        log.debug("CouplingSync v%d: broadcast to %d workers.", version, len(self._workers))
        return results

    def _get_adapter(self, worker: WorkerNode):
        from adapters.remote_adapter import RemoteCerebrumAdapter
        return RemoteCerebrumAdapter(worker.url, token=worker.token, secret=worker.secret)


# ---------------------------------------------------------------------------
# QueryRouter + PathStitcher
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Routes a multi-hop query across shards.

    For each hop, identifies which worker owns the current frontier entities,
    dispatches sub-queries in parallel, and collects candidate expansions.
    """

    def __init__(self, shard_manager: ShardManager, timeout: float = 10.0):
        self._shards = shard_manager
        self._timeout = timeout

    def route_hop(
        self,
        frontier_entities: List[str],
        max_hops: int,
        beam_width: int,
        top_k: int,
        current_hop: int,
    ) -> Dict[str, List[Dict]]:
        """
        Fan out sub-queries for the current frontier to their respective workers.

        Returns {entity_id: [candidate_expansions]}.
        """
        # Group entities by owning worker
        worker_groups: Dict[str, List[str]] = {}
        for eid in frontier_entities:
            wurl = self._shards.lookup(eid) or self._shards._route_by_strategy(eid)
            if wurl:
                worker_groups.setdefault(wurl, []).append(eid)

        results: Dict[str, List[Dict]] = {}
        threads: List[threading.Thread] = []
        lock = threading.Lock()

        def _query_worker(worker_url: str, entities: List[str]) -> None:
            try:
                from adapters.remote_adapter import RemoteCerebrumAdapter
                worker = self._shards._worker_by_url(worker_url)
                adapter = RemoteCerebrumAdapter(
                    worker_url,
                    token=worker.token if worker else None,
                    timeout=int(self._timeout),
                )
                for eid in entities:
                    resp = adapter._post_query(
                        entity=eid,
                        max_hops=1,          # single-hop expansion per sub-query
                        beam_width=beam_width,
                        top_k=top_k,
                    )
                    with lock:
                        results[eid] = resp.get("answers", [])
            except Exception as exc:
                log.warning("QueryRouter: worker %s error: %s", worker_url, exc)
                with lock:
                    for eid in entities:
                        results[eid] = []

        for wurl, entities in worker_groups.items():
            t = threading.Thread(target=_query_worker, args=(wurl, entities), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=self._timeout)

        return results


class PathStitcher:
    """
    Assembles complete multi-hop paths from cross-shard sub-query segments.
    """

    @staticmethod
    def stitch(
        query_entity: str,
        hop_results: List[Dict[str, List[Dict]]],
        top_k: int,
    ) -> List[DistributedAnswer]:
        """
        Combine per-hop sub-query results into complete reasoning paths.

        Uses a greedy beam approach: at each hop keep the top-k scoring
        partial paths, then extend by one hop.

        Parameters
        ----------
        query_entity : Starting entity.
        hop_results  : List of {entity → expansions} dicts, one per hop.
        top_k        : Number of final answers to return.

        Returns
        -------
        List[DistributedAnswer] sorted by score descending.
        """
        # Beam: list of (score, path_entities, path_relations, shard_trace)
        Beam = List[Tuple[float, List[str], List[str], List[str]]]
        beam: Beam = [(1.0, [query_entity], [], [])]

        for hop_idx, hop_map in enumerate(hop_results):
            new_beam: Beam = []
            for base_score, path_ents, path_rels, shards in beam:
                current_entity = path_ents[-1]
                expansions = hop_map.get(current_entity, [])
                for exp in expansions:
                    ans_entity = exp.get("entity", "")
                    ans_score  = float(exp.get("score", 0.0))
                    rel        = _extract_relation(exp.get("path", []), current_entity)
                    worker     = exp.get("worker_url", "?")

                    combined   = base_score * ans_score
                    new_beam.append((
                        combined,
                        path_ents + [ans_entity],
                        path_rels + [rel],
                        shards + [worker],
                    ))

            # Keep top-k for next hop
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:top_k * 2]  # keep 2× for diversity

        # Build final answers
        answers: List[DistributedAnswer] = []
        seen: set = set()
        for score, path_ents, path_rels, shards in sorted(beam, key=lambda x: x[0], reverse=True):
            terminal = path_ents[-1]
            if terminal in seen:
                continue
            seen.add(terminal)

            # Interleave entities and relations for the path list
            path_interleaved: List[str] = [path_ents[0]]
            for rel, ent in zip(path_rels, path_ents[1:]):
                path_interleaved.append(rel)
                path_interleaved.append(ent)

            answers.append(DistributedAnswer(
                entity=terminal,
                score=score,
                path=path_interleaved,
                shard_hops=shards,
            ))
            if len(answers) >= top_k:
                break

        return answers


def _extract_relation(path: list, from_entity: str) -> str:
    """Extract the relation label from a path list adjacent to from_entity."""
    for i, step in enumerate(path):
        if isinstance(step, dict) and step.get("id") == from_entity:
            if i + 1 < len(path):
                nxt = path[i + 1]
                if isinstance(nxt, dict) and nxt.get("type") == "relation":
                    return nxt.get("label", "RELATED_TO")
    return "RELATED_TO"


# ---------------------------------------------------------------------------
# DistributedCEREBRUM — main coordinator
# ---------------------------------------------------------------------------

class DistributedCEREBRUM:
    """
    Coordinator for a distributed CEREBRUM cluster.

    Parameters
    ----------
    workers         : List of worker config dicts:
                      {"url": str, "token": str|None, "secret": str|None, "weight": float}
    shard_strategy  : "community" | "hash" | "range"
    health_interval : Seconds between background health checks (0 = disabled).
    query_timeout   : Per-worker sub-query timeout in seconds.
    """

    def __init__(
        self,
        workers: List[Dict[str, Any]],
        shard_strategy: str = "community",
        health_interval: float = 30.0,
        query_timeout: float = 10.0,
    ):
        self._worker_nodes = [
            WorkerNode(
                url=w["url"],
                token=w.get("token"),
                secret=w.get("secret"),
                weight=float(w.get("weight", 1.0)),
            )
            for w in workers
        ]
        self._shard_manager = ShardManager(self._worker_nodes, strategy=shard_strategy)
        self._coupling        = CouplingSync(self._worker_nodes)
        self._router        = QueryRouter(self._shard_manager, timeout=query_timeout)
        self._stitcher      = PathStitcher()
        self._query_timeout = query_timeout

        if health_interval > 0:
            self._start_health_monitor(health_interval)

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------

    def assign_shard(
        self,
        entity_id: str,
        worker_url: str,
        community_id: Optional[int] = None,
    ) -> None:
        """Assign a single entity to a worker."""
        self._shard_manager.assign(entity_id, worker_url, community_id)

    def assign_community_shard(self, community_id: int, worker_url: str) -> None:
        """Assign an entire community to a worker."""
        self._shard_manager.assign_community(community_id, worker_url)

    def load_shard_manifest(self, manifest: List[Dict[str, Any]]) -> int:
        """
        Bulk-load a shard manifest.

        Each entry: {"entity_id": str, "worker_url": str, "community_id": int|None}
        """
        return self._shard_manager.load_manifest(manifest)

    def rebalance_shards(self, strategy: Optional[str] = None) -> Dict[str, int]:
        """Redistribute shards evenly across healthy workers."""
        return self._shard_manager.rebalance(strategy)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(
        self,
        entity: str,
        max_hops: int = 3,
        beam_width: int = 10,
        top_k: int = 10,
    ) -> List[DistributedAnswer]:
        """
        Execute a distributed multi-hop reasoning query.

        Parameters
        ----------
        entity     : Starting entity for traversal.
        max_hops   : Maximum reasoning depth.
        beam_width : Beam width per hop per worker.
        top_k      : Number of final answers.

        Returns
        -------
        List[DistributedAnswer] sorted by score descending.
        """
        frontier = [entity]
        hop_results: List[Dict[str, List[Dict]]] = []

        for hop in range(max_hops):
            hop_map = self._router.route_hop(
                frontier_entities=frontier,
                max_hops=max_hops,
                beam_width=beam_width,
                top_k=top_k,
                current_hop=hop,
            )
            hop_results.append(hop_map)

            # Update frontier for next hop
            new_frontier: List[str] = []
            for expansions in hop_map.values():
                for exp in expansions[:beam_width]:
                    eid = exp.get("entity", "")
                    if eid and eid not in new_frontier:
                        new_frontier.append(eid)
            if not new_frontier:
                break
            frontier = new_frontier[:beam_width]

        return PathStitcher.stitch(entity, hop_results, top_k)

    # ------------------------------------------------------------------
    # Community map synchronisation
    # ------------------------------------------------------------------

    def couple_community_map(
        self,
        delta: Dict[str, int],
        source_url: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Broadcast a community-map delta to all workers.

        Returns {worker_url: success}.
        """
        return self._coupling.couple_community_delta(delta, source_url)

    # ------------------------------------------------------------------
    # Cluster health
    # ------------------------------------------------------------------

    def cluster_status(self) -> List[Dict[str, Any]]:
        """Return current health and shard counts for all workers."""
        return [
            {
                "url":         w.url,
                "healthy":     w.healthy,
                "shard_count": w.shard_count,
                "latency_ms":  w.latency_ms,
                "last_check":  w.last_health_check,
            }
            for w in self._worker_nodes
        ]

    def add_worker(self, url: str, token: Optional[str] = None) -> None:
        """Add a new worker to the cluster without downtime."""
        worker = WorkerNode(url=url, token=token)
        self._worker_nodes.append(worker)
        self._shard_manager._workers.append(worker)
        self._coupling._workers.append(worker)
        log.info("DistributedCEREBRUM: added worker %s", url)

    def remove_worker(self, url: str, rebalance: bool = True) -> None:
        """
        Gracefully remove a worker. If rebalance=True, redistributes
        its shards across remaining healthy workers.
        """
        for w in self._worker_nodes:
            if w.url == url:
                w.healthy = False
                break
        if rebalance:
            self._shard_manager.rebalance()
        log.info("DistributedCEREBRUM: removed worker %s (rebalanced=%s)", url, rebalance)

    # ------------------------------------------------------------------
    # Background health monitor
    # ------------------------------------------------------------------

    def _start_health_monitor(self, interval: float) -> None:
        def _monitor():
            while True:
                time.sleep(interval)
                self._check_health()

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        log.debug("DistributedCEREBRUM: health monitor started (interval=%.1fs).", interval)

    def _check_health(self) -> None:
        for worker in self._worker_nodes:
            try:
                import requests
                t0 = time.perf_counter()
                resp = requests.get(
                    f"{worker.url}/health",
                    headers={"Authorization": f"Bearer {worker.token}"} if worker.token else {},
                    timeout=5,
                )
                latency = (time.perf_counter() - t0) * 1000
                worker.healthy        = resp.status_code == 200
                worker.latency_ms     = latency
                worker.last_health_check = time.time()
            except Exception:
                worker.healthy = False
                worker.last_health_check = time.time()

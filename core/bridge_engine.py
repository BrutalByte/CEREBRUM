"""
Bridge Twin Engine — Phase 12: Experience-Dependent Structural Relay Formation.

When a cross-community traversal occurs repeatedly (>= n_min times) and the
crossing node has high semantic similarity to the destination community centroid
(>= similarity_threshold), a "bridge twin" node is created:

  - Same embedding as the original node
  - Assigned to the destination community
  - Connected to the original by a bidirectional BRIDGE_TWIN edge (w_rel = 1.0)
  - Can accumulate its own community-specific edges over time (divergence)

The circuit is complete: queries arriving from either side of the community
boundary find a local relay node, eliminating the exponential distance penalty
for a frequently-used crossing.

Biological analog
-----------------
Thalamic relay nuclei: faithful copies of source signals re-expressed in an
intermediate structural location, completing circuits between otherwise-distant
brain regions. The LGN, for example, is a bridge twin of the retina that lives
inside the thalamus, holding the same retinotopic map while projecting into
the visual cortex.

LTP analog: bridge creation is triggered by repeated use (n_min crossings),
exactly as LTP requires repeated co-activation before a synapse potentiates.

LTD analog: idle bridges are pruned after prune_after_days, the same as the
sliding window evicts edges that haven't been recently reinforced.
"""
import threading
import time
import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from core.security import FederatedAuth
from core.node_registry import NodeRegistry

# The relation type label for bridge edges — gets w_rel = 1.0 in CSA
BRIDGE_RELATION = "BRIDGE_TWIN"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BridgeRecord:
    """Metadata for one bridge twin relationship."""

    original_id: str
    twin_id: str
    source_community: int
    destination_community: int
    traversal_count: int
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    similarity_at_creation: float = 0.0

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400.0

    def idle_days(self) -> float:
        return (time.time() - self.last_used) / 86400.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
logger = logging.getLogger("cerebrum.bridge_engine")
...
class BridgeTwinEngine:
    """
    Tracks cross-community traversals and materialises bridge twin nodes when
    a crossing has been used enough times to justify a permanent structural relay.

    Thread-safe: all state is protected by a reentrant lock so it can be shared
    between concurrent BeamTraversal instances (e.g. async streaming queries).

    Usage
    -----
    engine = BridgeTwinEngine(n_min=5, similarity_threshold=0.65)

    # In BeamTraversal, after each cross-community hop:
    twin_id = engine.record_crossing(
        node_id, source_community, dest_community, adapter
    )

    # BRIDGE_TWIN edges are added directly to the adapter's underlying graph.
    # BeamTraversal discovers them naturally via adapter.get_neighbors().
    # CSAEngine gives BRIDGE_TWIN edges w_rel = 1.0 (effectively free crossing).
    """

    def __init__(
        self,
        n_min: int = 5,
        similarity_threshold: float = 0.65,
        prune_after_days: float = 7.0,
        node_id: str = "local-node",
    ):
        """
        Parameters
        ----------
        n_min                : crossings required before a bridge is considered
        similarity_threshold : min cosine similarity to destination centroid
        prune_after_days     : idle bridges older than this are pruned (LTD)
        node_id              : unique identity for this agent
        """
        self.n_min = n_min
        self.similarity_threshold = similarity_threshold
        self.prune_after_days = prune_after_days
        self.node_id = node_id

        # (original_id, destination_community_id) -> crossing count
        self._candidates: Dict[Tuple[str, int], int] = {}

        # twin_id -> BridgeRecord
        self._bridges: Dict[str, BridgeRecord] = {}

        # (original_id, destination_community_id) -> twin_id (fast lookup)
        self._bridge_index: Dict[Tuple[str, int], str] = {}

        self._lock = threading.RLock()

        # Lazily-built reverse index: community_id -> [node_ids] (built once from
        # adapter.community_map on first _similarity_to_community call, then cached).
        self._community_members: Optional[Dict[int, List[str]]] = None
        # Per-community centroid embeddings: built incrementally as communities are
        # first requested, never recomputed for the same community in a run.
        self._centroid_cache: Dict[int, Optional[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_crossing(
        self,
        node_id: str,
        source_community: int,
        dest_community: int,
        adapter,                   # GraphAdapter — typed loosely to avoid circular import
    ) -> Optional[str]:
        """
        Record one cross-community traversal.

        If the crossing count reaches n_min and the node semantically fits
        the destination community (cosine similarity >= similarity_threshold),
        a bridge twin node is created and its twin_id is returned.

        If a bridge for this crossing already exists, its use counter is
        incremented and None is returned (bridge already live).

        Parameters
        ----------
        node_id          : node being traversed (source of the crossing)
        source_community : node's home community
        dest_community   : community being traversed into
        adapter          : NetworkXAdapter — twin is added here if created

        Returns
        -------
        twin_id if a new bridge was just created, else None.
        """
        key = (node_id, dest_community)

        with self._lock:
            # Bridge already exists — just refresh the timestamp
            if key in self._bridge_index:
                twin_id = self._bridge_index[key]
                rec = self._bridges[twin_id]
                rec.last_used = time.time()
                rec.traversal_count += 1
                return None

            # Accumulate crossing count
            self._candidates[key] = self._candidates.get(key, 0) + 1
            count = self._candidates[key]

            if count < self.n_min:
                return None

            # Semantic fit check — does the node actually belong in dest?
            sim = self._similarity_to_community(node_id, dest_community, adapter)
            if sim < self.similarity_threshold:
                return None

            # All criteria met — materialise the bridge twin
            return self._create_twin(
                node_id=node_id,
                source_community=source_community,
                dest_community=dest_community,
                sim=sim,
                count=count,
                adapter=adapter,
            )

    def get_twin(self, original_id: str, dest_community: int) -> Optional[str]:
        """Return twin_id if a bridge exists for this crossing, else None."""
        with self._lock:
            return self._bridge_index.get((original_id, dest_community))

    def record_twin_use(self, twin_id: str) -> None:
        """Mark a bridge twin as recently used (resets the idle/LTD timer)."""
        with self._lock:
            if twin_id in self._bridges:
                self._bridges[twin_id].last_used = time.time()

    def prune_unused(self, adapter=None) -> List[str]:
        """
        Remove bridge twins idle for >= prune_after_days (LTD analog).

        If adapter is provided, also removes twin nodes from the underlying graph.

        Returns list of pruned twin_ids.
        """
        pruned: List[str] = []
        with self._lock:
            for twin_id, record in list(self._bridges.items()):
                if record.idle_days() >= self.prune_after_days:
                    pruned.append(twin_id)
                    del self._bridges[twin_id]
                    self._bridge_index.pop(
                        (record.original_id, record.destination_community), None
                    )
                    self._candidates.pop(
                        (record.original_id, record.destination_community), None
                    )
                    if adapter is not None:
                        try:
                            adapter._G.remove_node(twin_id)
                        except Exception:
                            pass
        return pruned

    def describe(self) -> str:
        return (
            f"GraphBridgeEngine: cross-component bridges via embedding similarity "
            f"(min_sim={self.min_similarity}, top_k={self.top_k}, "
            f"max_degree={self.max_degree}, max_bridges={self.max_bridges:,})"
        )

    def active_bridges(self) -> List[BridgeRecord]:
        """Return a snapshot of all live bridge records."""
        with self._lock:
            return list(self._bridges.values())

    def observe_peer(self, proposal: Dict[str, Any], adapter: Any):
        """Observe peer success and synthesize local reinforcement."""
        path = proposal["path"]
        valence = proposal["valence"]
        logger.info(f"BridgeEngine: Observing peer success. Valence={valence}, Path={len(path)} edges")
        
        # Local validation: simple threshold based on similarity if applicable,
        # otherwise accept if valence is sufficiently high
        if valence > 0.7:
            # Here, we could potentially materialize similar edges locally
            # or perform Hebbian boosting of the path.
            for (u, rel, v) in path:
                adapter.update_edge_weight(u, v, rel, delta=0.01) # Mild reinforcement

    def propagate_consensus(self, node_id: str, proposal: Dict[str, Any], gws: Any):
        """Propagate consensus signal to the GWS."""
        logger.info(f"BridgeEngine: Propagating consensus bid for node {node_id}")
        # Use create_task to schedule the coroutine in the running loop
        asyncio.create_task(gws.post("consensus_bid", proposal, node_id))

    def candidate_count(self, node_id: str, dest_community: int) -> int:
        """How many crossings have been recorded for this (node, dest) pair."""
        with self._lock:
            return self._candidates.get((node_id, dest_community), 0)

    def is_bridge_twin(self, node_id: str) -> bool:
        """True if node_id is a bridge twin (not an original graph node)."""
        with self._lock:
            return node_id in self._bridges

    def on_rebalance(self, new_community_map: Dict[str, int]) -> int:
        """
        Called by GlobalRebalancer after a full DSCF re-run commits a new
        community_map to the adapter.

        Validates each existing BridgeRecord against the new partition:
          - ``record.source_community`` must still match the original node's
            community in ``new_community_map``
          - ``record.destination_community`` must still match the twin node's
            community in ``new_community_map``

        Records whose community IDs are now stale are removed from ``_bridges``
        and ``_bridge_index``. The ``_candidates`` crossing-count dict is left
        intact because crossing frequencies remain meaningful regardless of
        partition changes.

        Bridge twin *nodes* are NOT removed from the graph here — the existing
        ``prune_unused()`` LTD mechanism handles node-level cleanup. This method
        only invalidates the BridgeRecord bookkeeping.

        Parameters
        ----------
        new_community_map : ``{entity_id: community_id}`` dict from the fresh DSCF run

        Returns
        -------
        int — number of bridge records pruned as stale
        """
        with self._lock:
            stale: List[str] = []
            for twin_id, record in self._bridges.items():
                current_src = new_community_map.get(record.original_id, -1)
                current_dst = new_community_map.get(twin_id, -1)
                if (
                    current_src != record.source_community
                    or current_dst != record.destination_community
                ):
                    stale.append(twin_id)

            for twin_id in stale:
                record = self._bridges.pop(twin_id)
                self._bridge_index.pop((record.original_id, record.destination_community), None)

            # Community partition changed — invalidate reverse index and centroid cache
            self._community_members = None
            self._centroid_cache = {}

        return len(stale)

# ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _similarity_to_community(
        self,
        node_id: str,
        dest_community: int,
        adapter,
    ) -> float:
        """
        Cosine similarity between node_id's embedding and the centroid of
        dest_community (mean of member embeddings).

        Reverse index and per-community centroids are computed lazily and
        cached after first use so repeated calls are O(1) dict lookups.
        """
        node_emb = adapter.get_embedding(node_id)
        if node_emb is None:
            return 0.0

        # Build reverse index once: community_id -> [node_ids]
        if self._community_members is None:
            community_map: Dict[str, int] = getattr(adapter, "community_map", {})
            index: Dict[int, List[str]] = {}
            for n, c in community_map.items():
                index.setdefault(c, []).append(n)
            self._community_members = index

        members = self._community_members.get(dest_community)
        if not members:
            return 0.0

        # Compute centroid for this community once, then cache it
        if dest_community not in self._centroid_cache:
            vecs = []
            for m in members:
                e = adapter.get_embedding(m)
                if e is not None:
                    vecs.append(e)
            self._centroid_cache[dest_community] = np.mean(vecs, axis=0) if vecs else None

        centroid = self._centroid_cache[dest_community]
        if centroid is None:
            return 0.0

        return _cosine_sim(node_emb, centroid)

    def _create_twin(
        self,
        node_id: str,
        source_community: int,
        dest_community: int,
        sim: float,
        count: int,
        adapter,
    ) -> str:
        """
        Materialise the twin node in the adapter's graph and register
        the bridge record.
        """
        twin_id = f"{node_id}::twin::{dest_community}"
        
        # Provenance and Authenticated Signature
        prov = f"rule:bridge_twin|sim:{sim:.4f}|from:{node_id}|to:{dest_community}|node:{'local-node'}"
        payload = {"original_id": node_id, "twin_id": twin_id, "prov": prov}
        signature = self._sign_signal(payload).hex()

        # Copy node attributes, mark as bridge twin
        original_attrs = dict(adapter._G.nodes.get(node_id, {}))
        original_attrs.update({
            "is_bridge_twin": True,
            "original_id": node_id,
            "twin_community": dest_community,
            "signature": signature,
            "signer": "local-node"
        })
        original_label = original_attrs.get("label", node_id)
        original_attrs["label"] = f"{original_label} [relay@c{dest_community}]"

        adapter._G.add_node(twin_id, **original_attrs)

        # Copy embedding
        original_emb = adapter.get_embedding(node_id)
        if original_emb is not None:
            if not hasattr(adapter, "embeddings"):
                adapter.embeddings = {}
            adapter.embeddings[twin_id] = original_emb.copy()

        # Assign to destination community
        if not hasattr(adapter, "community_map"):
            adapter.community_map = {}
        adapter.community_map[twin_id] = dest_community

        # Bidirectional BRIDGE_TWIN edges — the circuit is now complete
        bridge_attrs = {"relation": BRIDGE_RELATION, "weight": 1.0, "is_bridge": True}
        adapter._G.add_edge(node_id, twin_id, **bridge_attrs)
        adapter._G.add_edge(twin_id, node_id, **bridge_attrs)

        # Register
        record = BridgeRecord(
            original_id=node_id,
            twin_id=twin_id,
            source_community=source_community,
            destination_community=dest_community,
            traversal_count=count,
            similarity_at_creation=sim,
        )
        self._bridges[twin_id] = record
        self._bridge_index[(node_id, dest_community)] = twin_id

        return twin_id


    def _sign_signal(self, payload: Dict[str, Any]) -> bytes:
        """Sign a signal payload."""
        data = json.dumps(payload, sort_keys=True).encode()
        return FederatedAuth.sign_payload(data)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

"""
MetaInsightEngine â€” second-order reasoning over InsightEvents (Phase 16).

Biological analogy: when the brain notices that two separate "Aha!" moments
are themselves related, it creates a meta-memory â€” a higher-order abstraction
that links two insights into a unified conceptual structure. This is the
neurological basis of analogical reasoning and the feeling of "I've seen this
pattern before."

Architecture
------------
The engine maintains an InsightGraph: a directed NetworkX graph where:
  - Nodes are InsightEvent IDs
  - Edges are structural relationships between insights:
      "chain"             â€” a.target == b.source (insight B continues insight A)
      "shared_entity"     â€” A and B reference the same source or target entity
      "community_overlap" â€” A and B cross the same community boundary
      "temporal_cluster"  â€” A and B fired within a short time window

When the InsightGraph itself develops a surprising new connection â€” i.e., when
two previously unrelated InsightEvents are discovered to be structurally linked â€”
a MetaInsightEvent fires. This is the system observing a pattern in its own
discovery history.

Depth-2 detection finds chains of meta-connections:
  Insight A â†’ Insight B (meta-edge) and Insight B â†’ Insight C (meta-edge)
  â†’ MetaInsight at depth=2: "My insight about B is itself connected to insights
    about both A and C â€” the three form a coherent higher-order pattern."

The "I think that you think that I think" recursive awareness maps directly to
depth-k meta-reasoning. In practice, depth 2 yields meaningful signal; depth 3
approaches the boundary of interpretability.

Usage
-----
    meta = MetaInsightEngine(meta_depth=2)

    # Feed it InsightEvents as they arrive
    new_meta_events = meta.observe(insight_event)

    # Inspect
    print(meta.total_meta_events)
    print(meta.recent_meta_events(5))
    nodes, edges = meta.insight_graph_size
    graph_data = meta.export_insight_graph()
"""
from __future__ import annotations

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple

import networkx as nx

from core.insight_engine import InsightEvent


# ---------------------------------------------------------------------------
# MetaInsightEvent
# ---------------------------------------------------------------------------

@dataclass
class MetaInsightEvent:
    """
    A second-order (or higher) pattern detected in the InsightGraph.

    Fields
    ------
    insight_a_id : str
        ID of the first InsightEvent in the relationship.
    insight_b_id : str
        ID of the second InsightEvent.
    connection_type : str
        How the two insights are related:
        "chain" | "shared_entity" | "community_overlap" | "temporal_cluster"
    meta_score : float
        Combined score of the two underlying insights â€” a proxy for how strong
        the meta-connection is.
    depth : int
        Recursion level. 1 = direct connection between two insights.
        2 = a chain of two meta-connections (Aâ†’B and Bâ†’C detected together).
    timestamp : float
        Unix time the MetaInsightEvent was created.
    id : str
        Short unique identifier.
    chain_ids : list[str]
        For depth >= 2, the full ordered list of InsightEvent IDs forming the
        chain (e.g., [A_id, B_id, C_id] for depth=2).
    """
    insight_a_id: str
    insight_b_id: str
    connection_type: str
    meta_score: float
    depth: int = 1
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    chain_ids: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MetaInsightEvent(type={self.connection_type!r}, "
            f"depth={self.depth}, score={self.meta_score:.3f}, "
            f"a={self.insight_a_id!r}â†’b={self.insight_b_id!r})"
        )


# ---------------------------------------------------------------------------
# MetaInsightEngine
# ---------------------------------------------------------------------------

class MetaInsightEngine:
    """
    Detects structural patterns between InsightEvents.

    Parameters
    ----------
    meta_depth : int
        Maximum depth of recursive meta-detection. 1 = pairwise connections,
        2 = chains of connections (default). Values above 3 are unlikely to
        yield interpretable signal.
    chain_score_threshold : float
        Minimum meta_score for a MetaInsightEvent to be recorded. Lower values
        capture weaker connections at the cost of more noise. Default 0.3.
    temporal_window : float
        Seconds within which two InsightEvents are considered temporally
        clustered. Default 300 (5 minutes).
    max_meta_events : int
        Maximum MetaInsightEvents kept in memory (ring buffer). Default 500.
    """

    def __init__(
        self,
        meta_depth: int = 2,
        chain_score_threshold: float = 0.3,
        temporal_window: float = 300.0,
        max_meta_events: int = 500,
    ):
        self.meta_depth             = meta_depth
        self.chain_score_threshold  = chain_score_threshold
        self.temporal_window        = temporal_window

        # InsightGraph: nodes = event IDs, edges = meta-relationships
        self._graph: nx.DiGraph = nx.DiGraph()

        # ID â†’ InsightEvent storage
        self._events: Dict[str, InsightEvent] = {}

        # Ring buffer of MetaInsightEvents
        self._meta_events: collections.deque = collections.deque(maxlen=max_meta_events)

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, event: InsightEvent) -> List[MetaInsightEvent]:
        """
        Add an InsightEvent to the InsightGraph and detect meta-patterns.

        Returns a list of any MetaInsightEvents fired by adding this event.
        Thread-safe.
        """
        with self._lock:
            self._events[event.id] = event
            self._graph.add_node(
                event.id,
                source=event.source,
                target=event.target,
                score=event.insight_score,
                community_leap=event.community_leap,
                timestamp=event.timestamp,
            )

            fired: List[MetaInsightEvent] = []

            # Connect to all existing events and detect depth-1 patterns
            for existing_id, existing in self._events.items():
                if existing_id == event.id:
                    continue

                connections = self._find_connections(existing, event)
                for conn_type, score in connections:
                    if not self._graph.has_edge(existing_id, event.id):
                        self._graph.add_edge(
                            existing_id, event.id,
                            connection_type=conn_type,
                            score=score,
                        )

                    if score >= self.chain_score_threshold:
                        meta_ev = MetaInsightEvent(
                            insight_a_id=existing_id,
                            insight_b_id=event.id,
                            connection_type=conn_type,
                            meta_score=score,
                            depth=1,
                            chain_ids=[existing_id, event.id],
                        )
                        self._meta_events.append(meta_ev)
                        fired.append(meta_ev)

            # Depth-2+: detect chains through the newly connected event
            if self.meta_depth >= 2:
                fired.extend(self._detect_higher_order(event.id))

            return fired

    def recent_meta_events(self, n: int = 20) -> List[MetaInsightEvent]:
        """Return the last n MetaInsightEvents (most recent last)."""
        with self._lock:
            items = list(self._meta_events)
        return items[-n:]

    @property
    def total_meta_events(self) -> int:
        with self._lock:
            return len(self._meta_events)

    @property
    def insight_graph_size(self) -> Tuple[int, int]:
        """(node_count, edge_count) of the InsightGraph."""
        with self._lock:
            return self._graph.number_of_nodes(), self._graph.number_of_edges()

    def export_insight_graph(self) -> dict:
        """
        Export the InsightGraph as a JSON-serialisable dict with nodes/edges.
        Suitable for the /meta-insight/graph API endpoint.
        """
        with self._lock:
            nodes = []
            for nid, data in self._graph.nodes(data=True):
                nodes.append({
                    "id": nid,
                    "source_entity": data.get("source", ""),
                    "target_entity": data.get("target", ""),
                    "insight_score": data.get("score", 0.0),
                    "community_leap": data.get("community_leap", 0),
                    "timestamp": data.get("timestamp", 0.0),
                })
            edges = []
            for a, b, data in self._graph.edges(data=True):
                edges.append({
                    "from": a,
                    "to": b,
                    "connection_type": data.get("connection_type", ""),
                    "score": data.get("score", 0.0),
                })
            return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Connection detection
    # ------------------------------------------------------------------

    def _find_connections(
        self, a: InsightEvent, b: InsightEvent
    ) -> List[Tuple[str, float]]:
        """
        Find all structural relationships between InsightEvents a and b.
        Returns a list of (connection_type, meta_score) tuples.
        Multiple connection types can fire for the same pair.
        """
        connections: List[Tuple[str, float]] = []
        base = (a.insight_score + b.insight_score) / 2.0

        # --- Chain: A's target is B's source (reasoning continues) ------
        if a.target == b.source:
            # Forward chain: a's conclusion is b's starting point
            connections.append(("chain", base))
        if b.target == a.source:
            # Reverse chain: b's conclusion leads back to a's start
            connections.append(("chain", base))

        # --- Shared entity: both insights reference the same node --------
        a_entities = {a.source, a.target, a.bridging_node}
        b_entities = {b.source, b.target, b.bridging_node}
        if a_entities & b_entities:
            connections.append(("shared_entity", base * 0.8))

        # --- Community overlap: both cross the same community boundary ---
        # (using community_leap as a proxy â€” same leap count and at least 1)
        if a.community_leap == b.community_leap and a.community_leap >= 1:
            connections.append(("community_overlap", base * 0.6))

        # --- Temporal cluster: fired close together in time --------------
        if abs(a.timestamp - b.timestamp) <= self.temporal_window:
            connections.append(("temporal_cluster", base * 0.4))

        return connections

    # ------------------------------------------------------------------
    # Higher-order detection
    # ------------------------------------------------------------------

    def _detect_higher_order(self, new_id: str) -> List[MetaInsightEvent]:
        """
        Detect depth-2 meta-patterns involving the newly added InsightEvent.

        Two patterns are detected:

        Pattern 1 â€” new_id completes a chain tail:
            pred_of_pred â†’ pred â†’ new_id
            The new event is the third link in a chain that already existed
            between pred_of_pred and pred. This is the most common pattern:
            Aâ†’B insight + Bâ†’C insight, then Câ†’D arrives and completes the
            meta-chain [Aâ†’B] â†’ [Bâ†’C] â†’ [Câ†’D].

        Pattern 2 â€” new_id starts a chain into existing successors:
            new_id â†’ succ â†’ succ_of_succ
            The new event feeds into a chain that already exists downstream.
            This fires if a third event was observed before the first.

        When this occurs, the system recognises that three insights form a
        coherent higher-order reasoning chain â€” the graph is thinking about
        its own thinking.
        """
        fired: List[MetaInsightEvent] = []

        # Pattern 1: pred_of_pred â†’ pred â†’ new_id
        for pred_id in list(self._graph.predecessors(new_id)):
            pred_data = self._graph.get_edge_data(pred_id, new_id) or {}
            for pp_id in list(self._graph.predecessors(pred_id)):
                if pp_id == new_id:
                    continue
                pp_data = self._graph.get_edge_data(pp_id, pred_id) or {}
                score = (pp_data.get("score", 0.0) + pred_data.get("score", 0.0)) / 2.0
                if score >= self.chain_score_threshold:
                    meta_ev = MetaInsightEvent(
                        insight_a_id=pp_id,
                        insight_b_id=new_id,
                        connection_type="chain",
                        meta_score=score,
                        depth=2,
                        chain_ids=[pp_id, pred_id, new_id],
                    )
                    self._meta_events.append(meta_ev)
                    fired.append(meta_ev)

        # Pattern 2: new_id â†’ succ â†’ succ_of_succ
        for succ_id in list(self._graph.successors(new_id)):
            succ_data = self._graph.get_edge_data(new_id, succ_id) or {}
            for ss_id in list(self._graph.successors(succ_id)):
                if ss_id == new_id:
                    continue
                ss_data = self._graph.get_edge_data(succ_id, ss_id) or {}
                score = (succ_data.get("score", 0.0) + ss_data.get("score", 0.0)) / 2.0
                if score >= self.chain_score_threshold:
                    meta_ev = MetaInsightEvent(
                        insight_a_id=new_id,
                        insight_b_id=ss_id,
                        connection_type="chain",
                        meta_score=score,
                        depth=2,
                        chain_ids=[new_id, succ_id, ss_id],
                    )
                    self._meta_events.append(meta_ev)
                    fired.append(meta_ev)

        return fired

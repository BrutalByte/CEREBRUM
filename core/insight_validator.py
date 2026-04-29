"""
InsightValidator — bilateral and corroborative verification of InsightEvents (Phase 16).

After the InsightEngine fires an InsightEvent based on *surprise*, the validator
verifies whether the connection is structurally sound using two independent checks:

  1. Bilateral reverse traversal
     Does a path exist from the insight target back to the source? A connection
     that only works in one direction is weaker than one where the knowledge
     graph can reason both ways. Implemented via undirected reachability on the
     graph (O(V+E) BFS — fast enough for post-hoc validation).

  2. Multi-path corroboration (triangulation)
     Do other nodes in the same source community also reach the insight target?
     If multiple independent entry points converge on the same target, the
     connection is structurally robust, not an artifact of one unusual path.

Validation produces one of four statuses (field: InsightEvent.validation_status):

  "corroborated"  — bilateral + corroboration_count >= 2  → confidence promoted to 0.95
  "bilateral"     — bilateral confirmed, corroboration_count < 2 → confidence 0.92
  "unilateral"    — not bilateral, but corroboration_count >= 1 → confidence unchanged
  "isolated"      — neither bilateral nor corroborated → confidence unchanged, flagged

Confidence promotion updates the INSIGHT_LINK edge in the graph so future traversal
strongly prefers paths that have been independently verified.
"""
from __future__ import annotations

import logging
import threading
from typing import List

import networkx as nx

from core.graph_adapter import GraphAdapter
from core.insight_engine import INSIGHT_RELATION, InsightEvent

_log = logging.getLogger("cerebrum.insight")

# Status constants — also used in tests
STATUS_CORROBORATED = "corroborated"
STATUS_BILATERAL    = "bilateral"
STATUS_UNILATERAL   = "unilateral"
STATUS_ISOLATED     = "isolated"

# Confidence values after promotion
CONFIDENCE_CORROBORATED = 0.95
CONFIDENCE_BILATERAL    = 0.92


class InsightValidator:
    """
    Validates InsightEvents via reverse-path and corroboration checks.

    Parameters
    ----------
    adapter : GraphAdapter
        Must expose ``adapter._G`` (NetworkXAdapter) and
        ``adapter.get_community(entity_id) -> int``.
    corroboration_seeds : int
        Maximum number of alternate community members to test as independent
        seeds. Higher = more thorough but slower. Default 5.
    corroboration_threshold : int
        Minimum alternate-seed hits to count as corroborated. Default 2.
    max_hop : int
        Maximum path length considered for bilateral/corroboration checks.
        Paths longer than this are too indirect to count as confirmation.
        Default 4 (one more than default traversal max_hop=3).
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        corroboration_seeds: int = 5,
        corroboration_threshold: int = 2,
        max_hop: int = 4,
    ):
        self.adapter                  = adapter
        self.corroboration_seeds      = corroboration_seeds
        self.corroboration_threshold  = corroboration_threshold
        self.max_hop                  = max_hop
        self._lock                    = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, event: InsightEvent) -> InsightEvent:
        """
        Validate a single InsightEvent. Mutates event in-place and returns it.

        Sets:
          - event.validation_status
          - event.corroboration_count
        Promotes the INSIGHT_LINK edge confidence in the graph if validated.
        """
        G = self._get_graph()
        G_und = G.to_undirected() if G.is_directed() else G

        bilateral     = self._check_bilateral(event, G_und)
        corroboration = self._check_corroboration(event, G, G_und)

        event.corroboration_count = corroboration

        if bilateral and corroboration >= self.corroboration_threshold:
            event.validation_status = STATUS_CORROBORATED
        elif bilateral:
            event.validation_status = STATUS_BILATERAL
        elif corroboration >= 1:
            event.validation_status = STATUS_UNILATERAL
        else:
            event.validation_status = STATUS_ISOLATED

        self._promote_edge(event, G)
        return event

    def validate_all(self, events: list) -> list:
        """Validate a list of InsightEvents, returning the same list mutated."""
        for event in events:
            self.validate(event)
        return events

    # ------------------------------------------------------------------
    # Bilateral check
    # ------------------------------------------------------------------

    def _check_bilateral(self, event: InsightEvent, G_und) -> bool:
        """
        True if a path of length <= max_hop exists from target back to source
        in the undirected view of the graph, EXCLUDING INSIGHT_LINK edges.

        Insight edges are excluded because the link itself must not serve as
        its own validation (that would be circular). We check whether the
        *original* graph structure independently supports a return path.
        """
        # Build a subgraph that strips out:
        #   1. INSIGHT_LINK edges (circular: the link can't validate itself)
        #   2. Direct source↔target edges (the original cross-community edge
        #      being validated cannot serve as its own return path)
        direct_pair = frozenset({event.source, event.target})
        filtered = nx.Graph()
        filtered.add_nodes_from(G_und.nodes())
        for u, v, data in G_und.edges(data=True):
            if data.get("relation") == INSIGHT_RELATION:
                continue
            if frozenset({u, v}) == direct_pair:
                continue
            filtered.add_edge(u, v)

        try:
            length = nx.shortest_path_length(filtered, event.target, event.source)
            return length <= self.max_hop
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    # ------------------------------------------------------------------
    # Corroboration check
    # ------------------------------------------------------------------

    def _check_corroboration(
        self, event: InsightEvent, G, G_und
    ) -> int:
        """
        Count how many other nodes in the source community can also reach
        the insight target (via a path of length <= max_hop).

        Only nodes in the same community as the source are used as seeds —
        they represent the same knowledge domain approaching the same target
        by an independent route.
        """
        source_cid = self.adapter.get_community(event.source)

        # Collect alternate seeds from the same community
        alt_seeds = []
        for node in G.nodes():
            if node in (event.source, event.target):
                continue
            if self.adapter.get_community(node) == source_cid:
                alt_seeds.append(node)
            if len(alt_seeds) >= self.corroboration_seeds:
                break

        # Remove the source node so corroboration paths cannot simply route
        # through the source's existing connection to the target.
        # A genuine corroborating path must reach the target independently.
        G_test = G_und.copy()
        if event.source in G_test:
            G_test.remove_node(event.source)

        count = 0
        for seed in alt_seeds:
            if seed not in G_test or event.target not in G_test:
                continue
            try:
                length = nx.shortest_path_length(G_test, seed, event.target)
                if length <= self.max_hop:
                    count += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        return count

    # ------------------------------------------------------------------
    # Confidence promotion
    # ------------------------------------------------------------------

    def _promote_edge(self, event: InsightEvent, G) -> None:
        """
        Raise the INSIGHT_LINK edge confidence for validated insights so future
        traversal strongly prefers them. Only "bilateral" and "corroborated"
        status triggers promotion.
        """
        if event.validation_status not in (STATUS_BILATERAL, STATUS_CORROBORATED):
            return

        target_confidence = (
            CONFIDENCE_CORROBORATED
            if event.validation_status == STATUS_CORROBORATED
            else CONFIDENCE_BILATERAL
        )

        for u, v in [(event.source, event.target), (event.target, event.source)]:
            if G.has_edge(u, v):
                data = G.get_edge_data(u, v) or {}
                if data.get("relation") == INSIGHT_RELATION:
                    data["confidence"] = max(data.get("confidence", 0.85), target_confidence)
                    return  # found and updated — done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_graph(self) -> nx.Graph:
        G = getattr(self.adapter, "_G", None)
        if G is None:
            raise AttributeError(
                "InsightValidator requires adapter._G (NetworkXAdapter). "
                "Override _get_graph() for other backends."
            )
        return G


class ProvenanceValidator:
    """Phase 149: Verifier agent logic to detect hub-flooding."""

    @staticmethod
    def is_hub_flooded(paths: List, threshold: float = 0.5) -> bool:
        """Analyze path results for high-degree hub signatures."""
        if not paths:
            return False

        # Case 1: Answer objects (already extracted)
        if hasattr(paths[0], 'branch_count'):
            hubs = [p for p in paths[:5] if p.branch_count > 50 or p.score > 0.9]
            if len(hubs) >= 3:
                _log.warning("CingulateEngine: Hub flooding detected (Answer mode).")
                return True
            return False

        # Case 2: TraversalPath objects (pre-extraction)
        tail_branches: dict = {}
        for p in paths:
            if not hasattr(p, 'nodes') or len(p.nodes) < 1:
                continue
            tail = p.tail
            branch_id = p.nodes[2] if len(p.nodes) >= 3 else p.nodes[0]
            if tail not in tail_branches:
                tail_branches[tail] = set()
            tail_branches[tail].add(branch_id)

        hubs = [t for t, branches in tail_branches.items() if len(branches) > 50]
        if len(hubs) >= 3:
            _log.warning("CingulateEngine: Hub flooding detected (Path mode). Hubs: %s", hubs[:3])
            return True

        return False

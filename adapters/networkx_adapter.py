"""
In-memory graph adapter backed by a NetworkX graph.

This is the default adapter — no external dependencies beyond networkx.
Use it for:
  - Loading from CSV (via csv_adapter.py)
  - Testing and prototyping
  - In-memory graph construction from Python dicts or triples
"""
import difflib
from typing import List, Optional, Dict

import networkx as nx

from core.graph_adapter import GraphAdapter, Entity, Edge


class NetworkXAdapter(GraphAdapter):
    """
    GraphAdapter implementation wrapping a networkx Graph or DiGraph.

    Node data convention:
      G.nodes[node_id] may carry {label, type, ...} attributes.
      Edge data may carry {relation, weight, ...} attributes.
      If absent, node ID is used as label and "entity" as type.
    """

    def __init__(
        self,
        G: nx.Graph,
        entity_types: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        G            : networkx Graph or DiGraph
        entity_types : optional {node_id -> type_string} override
        """
        self._G            = G
        self._entity_types = entity_types or {}

    # ------------------------------------------------------------------
    # Required GraphAdapter methods
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if entity_id not in self._G:
            return None
        data = self._G.nodes[entity_id]
        return Entity(
            id=entity_id,
            label=data.get("label", entity_id),
            type=self._entity_types.get(entity_id, data.get("type", "entity")),
            properties=dict(data),
        )

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        if entity_id not in self._G:
            return []

        edges = []
        neighbors = (
            self._G.successors(entity_id)
            if self._G.is_directed()
            else self._G.neighbors(entity_id)
        )

        for neighbor in neighbors:
            edge_data = self._G.get_edge_data(entity_id, neighbor) or {}
            rel_type  = edge_data.get("relation", "RELATED_TO")

            if edge_types and rel_type not in edge_types:
                continue

            edges.append(
                Edge(
                    source_id=entity_id,
                    target_id=neighbor,
                    relation_type=rel_type,
                    weight=float(edge_data.get("weight", 1.0)),
                    properties=dict(edge_data),
                )
            )

            if len(edges) >= max_neighbors:
                break

        return edges

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """Fuzzy match query against node IDs and labels."""
        all_nodes = list(self._G.nodes())

        # Exact match first
        if query in self._G:
            exact = self.get_entity(query)
            rest  = difflib.get_close_matches(query, all_nodes, n=top_k, cutoff=0.4)
            seen  = {query}
            result = [exact]
            for n in rest:
                if n not in seen:
                    e = self.get_entity(n)
                    if e:
                        result.append(e)
                    seen.add(n)
            return result[:top_k]

        matches = difflib.get_close_matches(query, all_nodes, n=top_k, cutoff=0.3)
        return [self.get_entity(m) for m in matches if self.get_entity(m)]

    def to_networkx(self) -> nx.Graph:
        return self._G

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def get_all_entities(self) -> List[Entity]:
        return [self.get_entity(n) for n in self._G.nodes()]

    def get_edge_types(self) -> List[str]:
        types = set()
        for _, _, data in self._G.edges(data=True):
            types.add(data.get("relation", "RELATED_TO"))
        return sorted(types)

    def node_count(self) -> int:
        return self._G.number_of_nodes()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_triples(
        cls,
        triples: List[tuple],
        directed: bool = True,
    ) -> "NetworkXAdapter":
        """
        Build an adapter from a list of (subject, predicate, object) triples.

        Example:
            adapter = NetworkXAdapter.from_triples([
                ("newton", "INFLUENCED", "einstein"),
                ("einstein", "COLLABORATED", "bohr"),
            ])
        """
        G = nx.DiGraph() if directed else nx.Graph()
        for s, p, o in triples:
            G.add_edge(str(s), str(o), relation=str(p))
        return cls(G)

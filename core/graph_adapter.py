"""
Abstract base class for graph backends.

Implement these four methods for any graph system (NetworkX, Neo4j, RDF, etc.)
and the full Parallax stack works with it unchanged.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Entity:
    id: str
    label: str
    type: str = "entity"
    properties: dict = field(default_factory=dict)


@dataclass
class Edge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: dict = field(default_factory=dict)


class GraphAdapter(ABC):
    """
    Minimal interface that any graph backend must implement to work with Parallax.

    All four abstract methods are required. Two optional helpers (get_all_entities,
    get_edge_types) have default NotImplementedError stubs — override when available.
    """

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Return the Entity for entity_id, or None if not found."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        edge_types: List[str] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        """
        Return outgoing edges from entity_id.

        If edge_types is provided, filter to only those relation types.
        Callers set max_neighbors to prevent beam explosion at high-degree nodes.
        """
        ...

    @abstractmethod
    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """
        Find entities matching query (text search, fuzzy match, or exact lookup).
        Used for entity grounding in STEP 1 of the forward pass.
        """
        ...

    @abstractmethod
    def to_networkx(self) -> "nx.Graph":  # noqa: F821
        """
        Return the graph as a networkx Graph (or DiGraph).
        Used by community_engine for DSCF/Leiden/LPA community detection.
        """
        ...

    # Optional helpers — override for better performance or richer metadata.

    def get_all_entities(self) -> List[Entity]:
        raise NotImplementedError

    def get_edge_types(self) -> List[str]:
        raise NotImplementedError

    def node_count(self) -> int:
        """Convenience: number of nodes (used for adaptive resolution)."""
        try:
            return self.to_networkx().number_of_nodes()
        except Exception:
            raise NotImplementedError

    def adaptive_resolution(self) -> float:
        """
        Compute DSCF resolution targeting K ~ sqrt(N) communities (Section 8.2).
        Resolution is heuristic — tune per-graph if needed.
        """
        n = self.node_count()
        if n <= 10:
            return 0.5
        if n <= 100:
            return 0.8
        if n <= 1000:
            return 1.0
        if n <= 10000:
            return 1.2
        return 1.5

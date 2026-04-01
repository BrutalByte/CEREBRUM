"""
Abstract base class for graph backends.

Implement these four methods for any graph system (NetworkX, Neo4j, RDF, etc.)
and the full CEREBRUM stack works with it unchanged.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
import networkx as nx


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
    # Contradiction handling fields
    confidence: float = 1.0          # Claim certainty [0, 1]; 1.0 = fully certain
    provenance: str = ""             # Source ID: "pubmed:123", "wikidata:Q42", etc.
    valid_from: Optional[float] = None  # Unix timestamp; None = always valid
    valid_to: Optional[float] = None    # Unix timestamp; None = still valid


class GraphAdapter(ABC):
    """
    Minimal interface that any graph backend must implement to work with CEREBRUM.

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
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
        context_embedding: Optional["np.ndarray"] = None,
    ) -> List[Edge]:
        """
        Return outgoing edges from entity_id.

        If edge_types is provided, filter to only those relation types.
        Callers set max_neighbors to prevent beam explosion at high-degree nodes.
        
        context_embedding: optional vector used by FederatedAdapter for
                           holographic blind discovery.
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
    def to_networkx(self) -> nx.Graph:
        """
        Return the graph as a networkx Graph (or DiGraph).
        Used by community_engine for DSCF/Leiden/LPA community detection.
        """
        ...

    @abstractmethod
    def get_community(self, entity_id: str) -> int:
        """Return the community ID for entity_id, or -1 if unknown."""
        ...

    @abstractmethod
    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Return the embedding vector for entity_id, or None."""
        ...

    @abstractmethod
    def find_similar(
        self, 
        embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[Entity]:
        """
        Find entities most semantically similar to the given vector.
        Used for cross-modal and blind federated discovery.
        """
        ...

    def get_reasoning_branches(
        self,
        seed_id: str,
        context_embedding: Optional[np.ndarray] = None,
        max_hop: int = 2,
        beam_width: int = 5,
    ) -> List[Dict]:
        """
        Optional: Return a list of reasoning paths (beams) starting from seed_id.
        Used by FederatedAdapter to delegate multi-hop exploration to remote agents.

        Returns:
            List[Dict] — Serialized TraversalPath objects.
        """
        return []

    def find_entities_masked(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Find entities but return masked results (ID + Score only).
        Default implementation wraps find_entities.
        """
        entities = self.find_entities(query, top_k)
        return [{"id": e.id, "type": e.type, "score": 1.0} for e in entities if e]

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




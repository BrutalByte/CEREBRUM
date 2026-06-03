"""
Abstract base class for graph backends.

Implement these four methods for any graph system (NetworkX, Neo4j, RDF, etc.)
and the full CEREBRUM stack works with it unchanged.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
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

    def get_neighbors_batch(
        self,
        entity_ids: List[str],
        max_neighbors: int = 50,
    ) -> Dict[str, List[Edge]]:
        """
        Phase 171: Return neighbors for multiple entities at once.
        Default implementation calls get_neighbors in a loop.
        """
        return {eid: self.get_neighbors(eid, max_neighbors=max_neighbors) for eid in entity_ids}

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
        ...

    @abstractmethod
    def get_degree(self, entity_id: str) -> int:
        return 0
        ...

        ...

    @abstractmethod
    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        """Add an edge to the graph (for Phase 65 materialization)."""
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

    def remove_edge(self, u: str, v: str, relation: str) -> None:
        """
        Remove an edge from the graph (for Phase 76+ provenance rollback).

        This method is optional: adapters that support graph mutation should
        override it.  The default implementation raises ``NotImplementedError``
        so that ``ProvenanceLedger.rollback_*()`` can surface a clear error
        rather than silently doing nothing.

        Parameters
        ----------
        u, v     : node identifiers of the edge endpoints.
        relation : the ``relation`` attribute that must match the stored edge.

        Raises
        ------
        NotImplementedError
            If the adapter has not implemented mutation support.
        ValueError
            If no matching edge is found (implementation-dependent).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement remove_edge(). "
            "Override this method to enable provenance rollback."
        )

    def update_edge_valence(
        self, u: str, v: str, relation: str,
        delta: float = 0.0, min_val: float = -1.0, max_val: float = 1.0
    ) -> int:
        """Add delta to edge valence (aversive/appetitive tone), clamped to [min_val, max_val].

        Valence is stored separately from weight in edge attribute "valence".
        Raises NotImplementedError by default; override in mutable adapters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_edge_valence()."
        )

    def get_edge_valence(self, u: str, v: str, relation: str) -> float:
        """Return the valence of the matching edge, or 0.0 if absent.

        Default implementation returns 0.0 (neutral) for adapters that
        don't store valence; override for full support.
        """
        return 0.0

    def update_edge_weight(
        self, u: str, v: str, relation: str,
        delta: float = 0.0, max_weight: float = 2.0, min_weight: float = 0.0
    ) -> int:
        """Add delta to edge weight, clamped to [min_weight, max_weight].

        Supports LTP (positive delta, Phase 96) and LTD (negative delta, Phase 97).
        Returns 1 if found and updated, 0 otherwise.
        Raises NotImplementedError by default; override in adapters that support mutation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_edge_weight(). "
            "Override this method to enable Hebbian consolidation."
        )

    def get_reasoning_branches(
        self,
        seed_id: str,
        context_embedding: Optional[np.ndarray] = None,
        max_hop: int = 2,
        beam_width: int = 5,
        max_budget: int = 500,
    ) -> List[Dict]:
        """
        Optional: Return a list of reasoning paths (beams) starting from seed_id.
        Used by FederatedAdapter to delegate multi-hop exploration to remote agents.

        Returns:
            List[Dict] ΓÇö Serialized TraversalPath objects.
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

    def get_meta_neighbors(self, relation_type: str) -> List["MetaEdge"]:
        """Phase 217: Return meta-edges for this relation type (default: empty).

        Override in adapters that have built a meta-relation graph.
        """
        return []

    def get_all_edges(self, limit: int = 500) -> List[Edge]:
        """
        Return up to *limit* edges from the graph.
        Default implementation falls back to iterating all entities and their
        neighbours; adapters backed by a graph store should override for
        better performance.
        """
        seen: set = set()
        edges: List[Edge] = []
        try:
            entities = self.get_all_entities()
        except NotImplementedError:
            return edges
        for entity in entities:
            if len(edges) >= limit:
                break
            for edge in self.get_neighbors(entity.id, max_neighbors=limit):
                key = (edge.source_id, edge.relation_type, edge.target_id)
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)
                    if len(edges) >= limit:
                        break
        return edges

    def get_relation_statistics(self) -> Dict[str, Dict]:
        """
        Return per-relation structural statistics over the full graph.

        Result: {
            relation_type: {
                "freq":             int,    # total edge count for this relation
                "n_unique_targets": int,    # distinct target entities
                "n_unique_sources": int,    # distinct source entities
                "target_degree_sum": float, # sum of degrees of all target entities
            }
        }

        Default: O(E) pass via to_networkx(). Override in adapters that can
        compute this more efficiently (e.g., from a pre-built index).
        Used by StructuralRelationInferrer to build agnostic TRB hints.
        """
        from collections import defaultdict
        G = self.to_networkx()
        freq: Dict[str, int] = defaultdict(int)
        unique_targets: Dict[str, set] = defaultdict(set)
        unique_sources: Dict[str, set] = defaultdict(set)
        target_degree_sum: Dict[str, float] = defaultdict(float)
        degree = dict(G.degree())
        for src, tgt, data in G.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            freq[rel] += 1
            unique_targets[rel].add(tgt)
            unique_sources[rel].add(src)
            target_degree_sum[rel] += float(degree.get(tgt, 1))
        return {
            rel: {
                "freq": freq[rel],
                "n_unique_targets": len(unique_targets[rel]),
                "n_unique_sources": len(unique_sources[rel]),
                "target_degree_sum": target_degree_sum[rel],
            }
            for rel in freq
        }

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


@dataclass
class MetaEdge:
    """
    Phase 217: A relation between relation types (meta-relation).

    Captures the structural co-occurrence of relation types in the graph:
    if triples A→[r1]→B and B→[r2]→C appear frequently, then r1 and r2
    are linked by the meta-relation 'precedes' (or whatever is derived).

    This lifts relation types to first-class graph citizens and enables
    second-order queries: "What kinds of connections link disease to gene?"
    """
    source_relation: str    # e.g. "treats"
    target_relation: str    # e.g. "causes"
    meta_relation: str      # e.g. "precedes", "co-occurs_with", "implies"
    weight: float = 1.0     # co-occurrence count (TF-IDF normalised)
    confidence: float = 1.0


class CredibilityRegistry:
    """
    Phase 216-A: Per-source trust priors for graph edge provenance.

    Maps provenance strings (e.g. "pubmed:123", "synthetic") to a credibility
    score in [0, 1].  Edges with unknown provenance get the default score (0.7).

    The score is multiplied into the CSA 'grounding' feature so that
    authoritative sources (PubMed, structured KBs) naturally score higher than
    synthetic or inferred edges — without touching path score arithmetic.
    """

    _DEFAULTS: Dict[str, float] = {
        "pubmed":    0.95,
        "openkg":    0.85,
        "wikidata":  0.90,
        "freebase":  0.88,
        "dbpedia":   0.85,
        "inferred":  0.50,
        "synthetic": 0.30,
        "rem_synthesized": 0.25,
        "hypothesis": 0.40,
    }

    def __init__(self, default_score: float = 0.70) -> None:
        self._scores: Dict[str, float] = dict(self._DEFAULTS)
        self.default_score = default_score

    def get(self, provenance: str) -> float:
        """Return credibility for a provenance string.

        Matches by prefix so "pubmed:12345" returns the 'pubmed' score.
        """
        if not provenance:
            return self.default_score
        # Exact match first
        if provenance in self._scores:
            return self._scores[provenance]
        # Prefix match (e.g. "pubmed:123" → "pubmed")
        for key, score in self._scores.items():
            if provenance.startswith(key):
                return score
        return self.default_score

    def register(self, source: str, score: float) -> None:
        """Register or update the credibility score for a source prefix."""
        self._scores[source] = float(max(0.0, min(1.0, score)))

    def to_dict(self) -> dict:
        return {"default_score": self.default_score, "scores": dict(self._scores)}

    @classmethod
    def from_dict(cls, d: dict) -> "CredibilityRegistry":
        obj = cls(default_score=d.get("default_score", 0.70))
        obj._scores.update(d.get("scores", {}))
        return obj

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CredibilityRegistry":
        with open(path) as f:
            return cls.from_dict(json.load(f))

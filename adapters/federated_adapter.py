"""
Federated Graph Adapter for Parallax.

Aggregates multiple GraphAdapter instances (local or remote) into a single 
virtual graph. Handles entity resolution and neighbor merging across 
disparate knowledge bases.
"""
from typing import List, Optional, Dict, Set
from core.graph_adapter import GraphAdapter, Entity, Edge


class FederatedAdapter(GraphAdapter):
    """
    Combines multiple GraphAdapters into a single logical view.
    
    Attributes:
        adapters (Dict[str, GraphAdapter]): Named adapters (e.g., {'local': nx, 'remote': api})
        id_to_adapter (Dict[str, str]): Mapping from entity_id to adapter name
    """

    def __init__(self, adapters: Dict[str, GraphAdapter]):
        self.adapters = adapters
        self._id_cache: Dict[str, str] = {}

    def _resolve_adapter(self, entity_id: str) -> Optional[str]:
        """Find which adapter owns this entity_id."""
        if entity_id in self._id_cache:
            return self._id_cache[entity_id]
            
        for name, adapter in self.adapters.items():
            if adapter.get_entity(entity_id):
                self._id_cache[entity_id] = name
                return name
        return None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        name = self._resolve_adapter(entity_id)
        if name:
            return self.adapters[name].get_entity(entity_id)
        return None

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: List[str] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        """
        Merge neighbors from all adapters. 
        Note: Current implementation assumes entity_id is globally unique 
        or the first adapter found is the 'owner'.
        """
        all_edges: List[Edge] = []
        seen_targets: Set[str] = set()
        
        # Primary: check the owner
        owner_name = self._resolve_adapter(entity_id)
        if owner_name:
            edges = self.adapters[owner_name].get_neighbors(entity_id, edge_types, max_neighbors)
            for e in edges:
                if e.target_id not in seen_targets:
                    all_edges.append(e)
                    seen_targets.add(e.target_id)
                    
        # Secondary: Check other adapters for cross-graph links 
        # (This is where cross-node alignment happens in Phase 6.2)
        # For now, we just aggregate if multiple adapters have info on the same ID.
        for name, adapter in self.adapters.items():
            if name == owner_name:
                continue
            # Some adapters might have partial info or cross-links
            edges = adapter.get_neighbors(entity_id, edge_types, max_neighbors)
            for e in edges:
                if e.target_id not in seen_targets:
                    all_edges.append(e)
                    seen_targets.add(e.target_id)
                    
        return all_edges[:max_neighbors]

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """Aggregate search results from all adapters."""
        results: List[Entity] = []
        seen_ids: Set[str] = set()
        
        for adapter in self.adapters.values():
            for entity in adapter.find_entities(query, top_k):
                if entity.id not in seen_ids:
                    results.append(entity)
                    seen_ids.add(entity.id)
                    
        # Sort or rank could happen here if adapters provide scores
        return results[:top_k]

    def to_networkx(self) -> "nx.Graph":
        """
        Merged NetworkX graph. Warning: can be expensive for large federated KGs.
        """
        import networkx as nx
        merged = nx.Graph()
        for adapter in self.adapters.values():
            merged = nx.compose(merged, adapter.to_networkx())
        return merged

    def node_count(self) -> int:
        return sum(a.node_count() for a in self.adapters.values())

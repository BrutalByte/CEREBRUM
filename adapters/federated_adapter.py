"""
Federated Graph Adapter for Parallax.

Aggregates multiple GraphAdapter instances (local or remote) into a single 
virtual graph. Handles entity resolution and neighbor merging across 
disparate knowledge bases.
"""
from typing import List, Optional, Dict, Set, Tuple
from core.graph_adapter import GraphAdapter, Entity, Edge
from core.alignment_engine import AlignmentIndex


class FederatedAdapter(GraphAdapter):
    """
    Combines multiple GraphAdapters into a single logical view.
    
    Attributes:
        adapters (Dict[str, GraphAdapter]): Named adapters (e.g., {'local': nx, 'remote': api})
        alignment (AlignmentIndex): Mapping for entity resolution across graphs.
    """

    def __init__(self, adapters: Dict[str, GraphAdapter], alignment: Optional[AlignmentIndex] = None):
        self.adapters = adapters
        self.alignment = alignment or AlignmentIndex()
        self._id_cache: Dict[str, str] = {}

    def _resolve_adapter(self, entity_id: str) -> Optional[str]:
        """Find which adapter owns this entity_id (primary owner)."""
        if entity_id in self._id_cache:
            return self._id_cache[entity_id]
            
        for name, adapter in self.adapters.items():
            if adapter.get_entity(entity_id):
                self._id_cache[entity_id] = name
                return name
        return None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        # Implementation: Check all owners of this ID or any alias
        owner_name = self._resolve_adapter(entity_id)
        if not owner_name:
            return None
            
        return self.adapters[owner_name].get_entity(entity_id)

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: List[str] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        """
        Merge neighbors from all adapters based on entity aliases.
        """
        all_edges: List[Edge] = []
        seen_targets: Set[str] = set()
        
        # 1. Identify which adapter(s) have this entity (primary and aliases)
        owner_name = self._resolve_adapter(entity_id)
        if not owner_name:
            return []
            
        # 2. Get all known (adapter, id) pairs for this entity via alignment index
        aliases = self.alignment.resolve_aliases(owner_name, entity_id)
        
        # 3. Aggregate neighbors from every alias
        for alias_adapter_name, alias_id in aliases:
            if alias_adapter_name not in self.adapters:
                continue
            
            adapter = self.adapters[alias_adapter_name]
            edges = adapter.get_neighbors(alias_id, edge_types, max_neighbors)
            
            for e in edges:
                # Deduplication by target ID. 
                # Future: Map target_id to its canonical ID for global deduplication.
                target_canonical = self.alignment.get_canonical(alias_adapter_name, e.target_id)
                dedupe_key = target_canonical or e.target_id
                
                if dedupe_key not in seen_targets:
                    all_edges.append(e)
                    seen_targets.add(dedupe_key)
                    
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

    def get_community(self, entity_id: str) -> int:
        """
        Get community ID for entity. Uses primary adapter's community
        and maps it to a federated namespace.
        """
        owner_name = self._resolve_adapter(entity_id)
        if not owner_name:
            return -1
        
        adapter = self.adapters[owner_name]
        # We assume local adapters might have a .community_map attribute 
        # or the FederatedAdapter is initialized with them.
        # For now, let's look for a community_map in the adapter if it's a local one.
        if hasattr(adapter, "community_map"):
            local_cid = adapter.community_map.get(entity_id, -1)
            if local_cid == -1:
                return -1
            # Federated CID = hash(adapter_name) + local_cid to avoid collisions
            return hash(owner_name) % 1000000 + local_cid
        return -1

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding from the owner adapter."""
        owner_name = self._resolve_adapter(entity_id)
        if not owner_name:
            return None
        
        adapter = self.adapters[owner_name]
        if hasattr(adapter, "embeddings"):
            return adapter.embeddings.get(entity_id)
        return None

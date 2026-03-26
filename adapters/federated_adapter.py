"""
Federated Graph Adapter for CEREBRUM.

Aggregates multiple GraphAdapter instances (local or remote) into a single 
virtual graph. Handles entity resolution and neighbor merging across 
disparate knowledge bases.
"""
from typing import List, Optional, Dict, Set, Tuple
import numpy as np
from core.graph_adapter import GraphAdapter, Entity, Edge
from core.alignment_engine import AlignmentIndex
from core.holographic_index import HolographicIndex, CommunitySignature


class FederatedAdapter(GraphAdapter):
    """
    Combines multiple GraphAdapters into a single logical view.
    
    Attributes:
        adapters (Dict[str, GraphAdapter]): Named adapters (e.g., {'local': nx, 'remote': api})
        alignment (AlignmentIndex): Mapping for entity resolution across graphs.
        hologram_index (HolographicIndex): Compressed structural signatures for fast discovery.
    """

    def __init__(self, adapters: Dict[str, GraphAdapter], alignment: Optional[AlignmentIndex] = None):
        self.adapters = adapters
        self.alignment = alignment or AlignmentIndex()
        self.hologram_index = HolographicIndex()
        self._id_cache: Dict[str, str] = {}
        # Procrustes rotation matrices: {secondary_adapter_name -> R (d×d)}
        self._alignment_rotations: Dict[str, np.ndarray] = {}

    def refresh_holograms(self):
        """
        Fetch holographic signatures from all sub-adapters to enable
        optimized blind discovery.
        """
        for name, adapter in self.adapters.items():
            if hasattr(adapter, "get_hologram"):
                h = adapter.get_hologram()
                if h and "signatures" in h:
                    sigs = [CommunitySignature.from_dict(s) for s in h["signatures"]]
                    self.hologram_index.add_adapter_signatures(name, sigs)

    def validate_all_connections(self) -> Dict[str, bool]:
        """Verify health and compatibility of all member adapters."""
        results = {}
        for name, adapter in self.adapters.items():
            if hasattr(adapter, "validate_connection"):
                results[name] = adapter.validate_connection()
            else:
                results[name] = True # Local adapters are always 'connected'
        return results

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
        context_embedding: Optional[np.ndarray] = None,
    ) -> List[Edge]:
        """
        Merge neighbors from all adapters based on entity aliases.
        Includes holographic blind discovery (Wormholes) if context_embedding is provided.
        """
        all_edges: List[Edge] = []
        seen_targets: Set[str] = set()
        
        # 1. Identify which adapter(s) have this entity (primary and aliases)
        owner_name = self._resolve_adapter(entity_id)
        
        # Track which adapters we've already queried for this node
        queried_adapters: Set[str] = set()

        if owner_name:
            # 2. Get all known (adapter, id) pairs for this entity via alignment index
            aliases = self.alignment.resolve_aliases(owner_name, entity_id)
            
            # 3. Aggregate neighbors from every alias
            for alias_adapter_name, alias_id in aliases:
                if alias_adapter_name not in self.adapters:
                    continue
                
                adapter = self.adapters[alias_adapter_name]
                edges = adapter.get_neighbors(alias_id, edge_types, max_neighbors)
                queried_adapters.add(alias_adapter_name)
                
                for e in edges:
                    target_canonical = self.alignment.get_canonical(alias_adapter_name, e.target_id)
                    dedupe_key = target_canonical or e.target_id
                    
                    if dedupe_key not in seen_targets:
                        all_edges.append(e)
                        seen_targets.add(dedupe_key)

        # 4. Blind Discovery (Wormholes) - Milestone 1
        # If we have a context embedding, check OTHER adapters for relevant communities.
        if context_embedding is not None:
            # Find remote adapters whose community centroids match this embedding
            relevant = self.hologram_index.find_relevant_adapters(context_embedding, top_k=3)
            
            for adapter_name, sim_score in relevant:
                if adapter_name in queried_adapters:
                    continue
                if sim_score < 0.7: # Relevance threshold
                    continue
                
                # Perform a 'find_similar' on the remote adapter
                remote_adapter = self.adapters[adapter_name]
                similar_entities = remote_adapter.find_similar(context_embedding, top_k=3)
                
                for ent in similar_entities:
                    if ent.id in seen_targets:
                        continue
                        
                    # Materialize a virtual WORMHOLE edge
                    # Weight is scaled by similarity score
                    all_edges.append(Edge(
                        source_id=entity_id,
                        target_id=ent.id,
                        relation_type="WORMHOLE",
                        weight=sim_score,
                        confidence=sim_score,
                        provenance=f"hologram:{adapter_name}"
                    ))
                    seen_targets.add(ent.id)
                    
        return all_edges[:max_neighbors]

    def find_similar(
        self, 
        embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[Entity]:
        """Aggregate similar entities from all relevant adapters."""
        results: List[Entity] = []
        seen_ids: Set[str] = set()
        
        # Use hologram to narrow down adapters
        relevant = self.hologram_index.find_relevant_adapters(embedding, top_k=3)
        
        for name, _ in relevant:
            adapter = self.adapters[name]
            for entity in adapter.find_similar(embedding, top_k):
                if entity.id not in seen_ids:
                    results.append(entity)
                    seen_ids.add(entity.id)
                    
        return results[:top_k]

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """Aggregate search results from relevant adapters."""
        results: List[Entity] = []
        seen_ids: Set[str] = set()
        
        # 1. Targeted search: if ID is unique, probe Bloom Filters
        targeted_adapters = self.hologram_index.probe_entity(query)
        adapters_to_query = targeted_adapters if targeted_adapters else list(self.adapters.keys())

        for name in adapters_to_query:
            adapter = self.adapters[name]
            for entity in adapter.find_entities(query, top_k):
                if entity.id not in seen_ids:
                    results.append(entity)
                    seen_ids.add(entity.id)
                    
        return results[:top_k]

    def find_entities_masked(self, query: str, top_k: int = 10) -> List[Dict]:
        """Aggregate masked search results from relevant adapters."""
        results: List[Dict] = []
        seen_ids: Set[str] = set()
        
        targeted_adapters = self.hologram_index.probe_entity(query)
        adapters_to_query = targeted_adapters if targeted_adapters else list(self.adapters.keys())

        for name in adapters_to_query:
            adapter = self.adapters[name]
            for r in adapter.find_entities_masked(query, top_k):
                if r["id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["id"])
                    
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
            # Check aliases in case it's known under a different ID
            # This handles 'wormhole' entities from other graphs
            aliases = self.alignment.resolve_aliases("", entity_id)
            for adapter_name, alias_id in aliases:
                if adapter_name in self.adapters:
                    owner_name = adapter_name
                    entity_id = alias_id
                    break
            
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
        
        # Check if it's a Remote adapter (uses get_community method)
        try:
            local_cid = adapter.get_community(entity_id)
            if local_cid == -1:
                return -1
            return hash(owner_name) % 1000000 + local_cid
        except Exception:
            return -1

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding from the owner adapter, applying alignment rotation if registered."""
        owner_name = self._resolve_adapter(entity_id)
        if not owner_name:
            return None

        adapter = self.adapters[owner_name]
        emb = None
        if hasattr(adapter, "embeddings"):
            emb = adapter.embeddings.get(entity_id)

        if emb is None:
            return None

        R = self._alignment_rotations.get(owner_name)
        if R is not None:
            emb = emb @ R
        return emb

    def align_embeddings(
        self,
        primary_name: str,
        secondary_name: str,
        min_anchors: int = 5,
    ) -> int:
        """
        Compute and register a Procrustes rotation matrix to align the
        secondary adapter's embedding space into the primary adapter's space.

        Uses shared entity IDs (entities present in both adapters) as anchors.
        Requires at least ``min_anchors`` shared entities to produce a
        meaningful alignment; returns 0 and does nothing otherwise.

        The rotation is cached in ``_alignment_rotations[secondary_name]``
        and applied automatically by ``get_embedding()`` for all entities
        owned by the secondary adapter.

        Returns the number of anchor entities used for alignment.
        """
        if primary_name not in self.adapters or secondary_name not in self.adapters:
            return 0

        primary_adapter   = self.adapters[primary_name]
        secondary_adapter = self.adapters[secondary_name]

        # Collect embeddings from both adapters for shared entity IDs
        primary_embs:   List[np.ndarray] = []
        secondary_embs: List[np.ndarray] = []

        # Gather all entity IDs in the secondary adapter
        secondary_ids: Set[str] = set()
        if hasattr(secondary_adapter, "embeddings"):
            secondary_ids = set(secondary_adapter.embeddings.keys())
        elif hasattr(secondary_adapter, "_G"):
            secondary_ids = set(secondary_adapter._G.nodes())

        for eid in secondary_ids:
            p_emb = None
            if hasattr(primary_adapter, "embeddings"):
                p_emb = primary_adapter.embeddings.get(eid)
            if p_emb is None:
                continue
            s_emb = None
            if hasattr(secondary_adapter, "embeddings"):
                s_emb = secondary_adapter.embeddings.get(eid)
            if s_emb is None:
                continue
            primary_embs.append(p_emb)
            secondary_embs.append(s_emb)

        if len(primary_embs) < min_anchors:
            return 0

        # Orthogonal Procrustes: find R = argmin ||A - B R||_F  s.t. R^T R = I
        # Closed-form SVD solution: R = U @ V^T  where  A^T B = U S V^T
        A = np.stack(primary_embs, axis=0).astype(np.float64)
        B = np.stack(secondary_embs, axis=0).astype(np.float64)
        M = A.T @ B
        U, _s, Vt = np.linalg.svd(M)
        R = (U @ Vt).astype(np.float32)
        self._alignment_rotations[secondary_name] = R
        return len(primary_embs)

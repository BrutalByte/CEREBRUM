"""
FederatedGraphRegistry - Cross-domain reasoning engine.

Manages multiple independent graph backends and resolves cross-domain entity aliases
during beam traversal. Allows CEREBRUM to seamlessly hop between KGs.
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from adapters.mmap_adapter import MmapAdapter
from core.graph_adapter import GraphAdapter

logger = logging.getLogger("cerebrum.federated")

class FederatedGraphRegistry:
    """
    Registry for multiple Knowledge Graphs.
    Handles 'Cross-Domain Bridges' (aliases) to hop between graphs.
    """
    def __init__(self, governor=None):
        self.graphs: Dict[str, GraphAdapter] = {}
        self.bridges: Dict[str, str] = {}  # Source Entity -> Target Entity (in another KG)
        self.governor = governor
        self.active_graph: Optional[str] = None

    def register_graph(self, domain_id: str, data_path: str):
        """Register a new graph domain."""
        logger.info("Registering domain: %s at %s", domain_id, data_path)
        self.graphs[domain_id] = MmapAdapter(data_path, governor=self.governor)
        
    def add_bridge(self, src_entity: str, dst_entity: str):
        """Link an entity in one domain to another."""
        self.bridges[src_entity] = dst_entity

    def get_graph_for_entity(self, entity_id: str) -> Optional[str]:
        """Find which domain an entity belongs to."""
        for domain, adapter in self.graphs.items():
            if entity_id in adapter.id_map:
                return domain
        return None

    def get_adapter(self, domain_id: str) -> GraphAdapter:
        return self.graphs[domain_id]

    def resolve_alias(self, entity_id: str) -> str:
        """If this entity has a bridge, return the cross-domain alias."""
        return self.bridges.get(entity_id, entity_id)

    def close_all(self):
        for adapter in self.graphs.values():
            adapter.close()

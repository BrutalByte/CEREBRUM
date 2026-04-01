"""
Discovery Engine for Federated CEREBRUM (Phase 32).

Automates the discovery, handshake, and registration of remote reasoning nodes.
Uses the Holographic Index to identify relevant peers for specific queries.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
import requests
import numpy as np

from adapters.remote_adapter import RemoteCerebrumAdapter
from adapters.federated_adapter import FederatedAdapter

log = logging.getLogger("cerebrum.discovery")

@dataclass
class NodeMetadata:
    """Metadata for a discovered remote CEREBRUM node."""
    url: str
    node_id: str
    node_count: int = 0
    community_count: int = 0
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    trust_score: float = 1.0

class DiscoveryEngine:
    """
    Automates the discovery and health-checking of remote reasoning nodes.
    """
    def __init__(self, federated_adapter: FederatedAdapter):
        self.federated = federated_adapter
        self.known_nodes: Dict[str, NodeMetadata] = {}
        self.blacklist: Set[str] = set()

    def discover_node(self, url: str, api_key: Optional[str] = None) -> bool:
        """
        Perform handshake with a new remote node and register it if healthy.
        """
        if url in self.blacklist:
            return False
            
        try:
            # 1. Handshake
            adapter = RemoteCerebrumAdapter(url, token=api_key)
            if not adapter.validate_connection():
                log.warning("Discovery failed: Could not validate connection to %s", url)
                return False
                
            # 2. Fetch Metadata
            # (Assuming RemoteCerebrumAdapter has a way to get these, 
            # or we call /health /handshake directly)
            # For now, we probe basics
            n_count = adapter.node_count()
            
            # 3. Create metadata
            # We use the URL as a temporary node_id until we get a real one
            node_id = url 
            metadata = NodeMetadata(
                url=url,
                node_id=node_id,
                node_count=n_count,
                capabilities=["query", "traverse", "hologram"]
            )
            
            # 4. Register in FederatedAdapter
            # We might need a more unique name than just the URL
            adapter_name = f"remote_{len(self.known_nodes)}"
            self.federated.adapters[adapter_name] = adapter
            self.known_nodes[adapter_name] = metadata
            
            # 5. Refresh Holograms
            if hasattr(adapter, "get_hologram"):
                self.federated.refresh_holograms()
                
            log.info("Discovered and registered remote node: %s as %s", url, adapter_name)
            return True
            
        except Exception as e:
            log.error("Error discovering node %s: %s", url, e)
            self.blacklist.add(url)
            return False

    def health_check_all(self):
        """Verify all registered remote nodes and prune inactive ones."""
        to_remove = []
        for name, metadata in self.known_nodes.items():
            adapter = self.federated.adapters.get(name)
            if not adapter:
                to_remove.append(name)
                continue
                
            if not adapter.validate_connection():
                log.warning("Node %s is unhealthy. Pruning.", metadata.url)
                to_remove.append(name)
                
        for name in to_remove:
            del self.known_nodes[name]
            if name in self.federated.adapters:
                del self.federated.adapters[name]

    def find_relevant_nodes(self, query_embedding: np.ndarray) -> List[str]:
        """
        Use Holographic Index to find which registered nodes are 
        most relevant to a query.
        """
        relevant = self.federated.hologram_index.find_relevant_adapters(query_embedding, top_k=3)
        return [name for name, score in relevant if score > 0.5]

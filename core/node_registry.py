"""
NodeRegistry — Phase 172.5.

Persistent registry of trusted federated peers, their public keys, and metadata.
"""
import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger("cerebrum.node_registry")

class NodeRegistry:
    def __init__(self, registry_path: str = "data/cerebrum/node_registry.json"):
        self.registry_path = registry_path
        self.peers: Dict[str, str] = {}  # node_id: public_key_pem
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    self.peers = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load node registry: {e}")

    def _save(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.peers, f, indent=2)

    def register_peer(self, node_id: str, public_key_pem: str):
        self.peers[node_id] = public_key_pem
        self._save()
        logger.info(f"Peer {node_id} registered.")

    def get_public_key(self, node_id: str) -> Optional[str]:
        return self.peers.get(node_id)

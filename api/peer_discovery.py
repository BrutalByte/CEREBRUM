"""
Peer Discovery Daemon for CEREBRUM Swarm.

Periodically broadcasts local HolographicIndex signatures to known peers.
"""
import logging
import threading
import time
import secrets
from typing import Dict, Optional
from core.neural_coupling_engine import NeuralCouplingEngine
from core.holographic_index import build_signatures
from core.security import FederatedAuth
from core.node_registry import NodeRegistry

log = logging.getLogger("cerebrum.discovery")

class PeerDiscovery:
    """
    Background daemon that manages peer-to-peer coupling cycles with handshake verification.
    """
    def __init__(self, coupling_engine: NeuralCouplingEngine, adapter, community_map, embeddings, registry: NodeRegistry, interval: float = 60.0):
        self.engine = coupling_engine
        self.adapter = adapter
        self.community_map = community_map
        self.embeddings = embeddings
        self.registry = registry
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nonces: Dict[str, bytes] = {}  # peer_id: nonce

    def initiate_handshake(self, peer_id: str) -> bytes:
        """Create a challenge nonce for a peer."""
        nonce = secrets.token_bytes(32)
        self._nonces[peer_id] = nonce
        return nonce

    def verify_handshake(self, peer_id: str, signature: bytes, nonce: bytes) -> bool:
        """Verify peer identity via nonce signature."""
        if self._nonces.get(peer_id) != nonce:
            return False
        
        pub_key = self.registry.get_public_key(peer_id)
        if not pub_key:
            return False
            
        return FederatedAuth.verify_signature(pub_key, signature, nonce)

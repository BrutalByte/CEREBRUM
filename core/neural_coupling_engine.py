"""
Neural Coupling Engine for CEREBRUM Federated Swarm Intelligence.

Handles peer-to-peer exchange of HolographicIndex sketches and community centroids.
"""
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Any
import requests
from core.holographic_index import HolographicIndex, CommunitySignature

log = logging.getLogger("cerebrum.coupling")

@dataclass
class CouplingMessage:
    """Schema for coupling-based index exchange."""
    source_url: str
    timestamp: float
    signatures: List[Dict[str, Any]]  # List of CommunitySignature.to_dict()
    version: int = 1

class NeuralCouplingEngine:
    """
    Engine for decentralized exchange of structural graph signatures.
    """
    def __init__(self, local_url: str, holographic_index: HolographicIndex):
        self.local_url = local_url
        self.index = holographic_index
        self.peers: Set[str] = set()
        self.lock = threading.Lock()
        self._seen_messages: Dict[str, float] = {}  # source_url -> last_timestamp

    def add_peer(self, peer_url: str):
        with self.lock:
            if peer_url != self.local_url:
                self.peers.add(peer_url)

    def remove_peer(self, peer_url: str):
        with self.lock:
            self.peers.discard(peer_url)

    def receive_coupling(self, message: CouplingMessage) -> bool:
        """Process an incoming coupling message."""
        with self.lock:
            last_ts = self._seen_messages.get(message.source_url, 0)
            if message.timestamp <= last_ts:
                return False  # Already seen or older than current

            self._seen_messages[message.source_url] = message.timestamp

            # Update HolographicIndex
            sigs = [CommunitySignature.from_dict(s) for s in message.signatures]
            self.index.add_adapter_signatures(message.source_url, sigs)
            
            # Record peer if new
            if message.source_url != self.local_url:
                self.peers.add(message.source_url)
            
            return True

    def generate_coupling(self, local_signatures: List[CommunitySignature]) -> CouplingMessage:
        """Create a new coupling message from local state."""
        return CouplingMessage(
            source_url=self.local_url,
            timestamp=time.time(),
            signatures=[s.to_dict() for s in local_signatures]
        )

    def broadcast_coupling(self, message: CouplingMessage):
        """Send coupling to all known peers."""
        with self.lock:
            target_peers = list(self.peers)

        for peer in target_peers:
            try:
                # In a real swarm, this would be an async POST to /api/v1/coupling
                requests.post(
                    f"{peer}/api/v1/coupling",
                    json=asdict(message),
                    timeout=2
                )
            except Exception as e:
                log.warning(f"Failed to coupling to {peer}: {e}")

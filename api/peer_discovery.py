"""
Peer Discovery Daemon for CEREBRUM Swarm.

Periodically broadcasts local HolographicIndex signatures to known peers.
"""
import logging
import threading
import time
from typing import Optional
from core.neural_coupling_engine import NeuralCouplingEngine
from core.holographic_index import build_signatures

log = logging.getLogger("cerebrum.discovery")

class PeerDiscovery:
    """
    Background daemon that manages peer-to-peer coupling cycles.
    """
    def __init__(self, coupling_engine: NeuralCouplingEngine, adapter, community_map, embeddings, interval: float = 60.0):
        self.engine = coupling_engine
        self.adapter = adapter
        self.community_map = community_map
        self.embeddings = embeddings
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"PeerDiscovery started with interval {self.interval}s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        log.info("PeerDiscovery stopped")

    def _run(self):
        while self._running:
            try:
                # 1. Generate local signatures (the "Hologram")
                local_sigs = build_signatures(self.adapter, self.community_map, self.embeddings)
                
                # 2. Create coupling message
                msg = self.engine.generate_coupling(local_sigs)
                
                # 3. Broadcast to known peers
                self.engine.broadcast_coupling(msg)
                
            except Exception as e:
                log.error(f"Error in PeerDiscovery cycle: {e}")
            
            time.sleep(self.interval)

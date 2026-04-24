import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("cerebrum.gws")

@dataclass
class CommunitySignal:
    path: List[str]
    community_id: int
    novelty_score: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

class GlobalWorkspace:
    def __init__(self):
        self.blackboard: asyncio.Queue = asyncio.Queue()
        self.active_signals: Dict[int, CommunitySignal] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.ttl = 5.0 # Seconds

    async def post(self, topic: str, data: Any, node_id: str):
        """Post a reasoning proposal or bid to the blackboard."""
        entry = {"topic": topic, "data": data, "node_id": node_id, "timestamp": time.time()}
        await self.blackboard.put(entry)
        
        # Trigger subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(topic, entry)

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.info(f"GWS: Subscribed to {topic}")

    def get_top_signals(self, limit: int = 5) -> List[CommunitySignal]:
        # Bidding logic: weight = novelty * confidence
        sorted_signals = sorted(
            self.active_signals.values(),
            key=lambda s: s.novelty_score * s.confidence,
            reverse=True
        )
        return sorted_signals[:limit]

    async def cleanup_daemon(self):
        while True:
            now = time.time()
            keys_to_remove = [cid for cid, s in self.active_signals.items() if now - s.timestamp > self.ttl]
            for cid in keys_to_remove:
                del self.active_signals[cid]
            await asyncio.sleep(1.0)

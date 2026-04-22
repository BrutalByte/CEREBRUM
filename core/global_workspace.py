import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

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
        self.ttl = 5.0 # Seconds

    async def broadcast(self, signal: CommunitySignal):
        await self.blackboard.put(signal)
        self.active_signals[signal.community_id] = signal

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

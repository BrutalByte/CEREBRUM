from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Deque

@dataclass
class BridgeTask:
    node_id: str
    source_community: int
    dest_community: int
    path_score: Optional[float] = None
    path: Optional[Any] = None # TraversalPath, but avoid circular import
    is_twin_use: bool = False
    u_node: Optional[str] = None # For InsightEngine: source node of the crossing
    v_node: Optional[str] = None # For InsightEngine: target node of the crossing

class TaskQueue:
    def __init__(self):
        self._queue: Deque[BridgeTask] = deque()

    def enqueue(self, task: BridgeTask):
        self._queue.append(task)

    def dequeue(self) -> Optional[BridgeTask]:
        if self._queue:
            return self._queue.popleft()
        return None

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

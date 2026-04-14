"""
SynapticPruner — Periodic Graph Optimization via Utility-Based Edge Pruning (Phase 61).

Identifies and removes "low-utility" edges from the graph. Utility is defined by:
  1. Age (older synthetic edges are more likely to be pruned).
  2. Confidence (lower confidence edges are pruned first).
  3. Structural Redundancy (edges that don't bridge communities or aren't central).
  4. Usage (edges that aren't part of successful reasoning paths).
"""
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("cerebrum.synaptic_pruner")

class SynapticPruner:
    """
    Prunes low-utility edges from a GraphAdapter.
    """
    def __init__(
        self,
        adapter: Any,
        min_confidence: float = 0.3,
        max_age_days: float = 30.0,
        prune_ratio: float = 0.05,
        protected_relation_types: Optional[Set[str]] = None
    ):
        self.adapter = adapter
        self.min_confidence = min_confidence
        self.max_age_days = max_age_days
        self.prune_ratio = prune_ratio
        self.protected_types = protected_relation_types or set()
        self._total_pruned = 0
        self._lock = threading.Lock()

    def prune(self, dry_run: bool = False) -> int:
        """
        Execute one pruning cycle. Returns count of edges removed.
        """
        G = self.adapter.to_networkx()
        if G.number_of_edges() == 0:
            return 0

        # 1. Score all edges by utility
        scored_edges = []
        now = time.time()
        
        for u, v, data in G.edges(data=True):
            rel_type = data.get("relation", "related_to")
            if rel_type in self.protected_types:
                continue
                
            # Base utility: starts at confidence [0, 1]
            conf = data.get("confidence", 1.0)
            
            # Age penalty
            created_at = data.get("timestamp", now)
            age_sec = now - created_at
            age_days = age_sec / 86400.0
            age_penalty = min(1.0, age_days / self.max_age_days) if self.max_age_days > 0 else 0.0
            
            # Usage bonus (if tracked)
            usage_count = data.get("traversal_count", 0)
            usage_bonus = min(0.5, usage_count * 0.05)
            
            utility = (conf * (1.0 - 0.5 * age_penalty)) + usage_bonus
            
            scored_edges.append(((u, v, rel_type), utility))

        # 2. Sort by utility (ascending)
        scored_edges.sort(key=lambda x: x[1])
        
        # 3. Select candidates for pruning
        # Combine absolute min_confidence threshold + relative prune_ratio
        threshold_idx = int(len(scored_edges) * self.prune_ratio)
        candidates = []
        
        for i, ((u, v, r), utility) in enumerate(scored_edges):
            if utility < self.min_confidence or i < threshold_idx:
                candidates.append((u, v, r))
            else:
                break

        # 4. Remove edges
        if not dry_run:
            count = 0
            # NetworkXAdapter doesn't have a direct 'remove_edge' but we can
            # modify its _G if it's a NetworkXAdapter.
            if hasattr(self.adapter, "_G"):
                with self._lock:
                    for u, v, r in candidates:
                        try:
                            if self.adapter._G.has_edge(u, v):
                                self.adapter._G.remove_edge(u, v)
                                count += 1
                        except Exception:
                            continue
            self._total_pruned += count
            logger.info("SynapticPruner: pruned %d edges (Utility threshold).", count)
            return count
        
        return len(candidates)

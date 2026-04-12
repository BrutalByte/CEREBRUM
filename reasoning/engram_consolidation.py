"""
Engram consolidation and promotion logic (Phase 64).

This module provides the logic to promote high-utility reasoning patterns
from the dynamic Engram cache into permanent 'Canonical Engrams'.
"""
import logging
from typing import Dict, List, Set, Optional
from core.persistence import QueryLog

logger = logging.getLogger("cerebrum.engram")

class EngramConsolidator:
    """
    Analyzes Engram usage and promotes high-affinity patterns.
    """
    def __init__(self, engram_cache, min_success_threshold: int = 5):
        self.cache = engram_cache
        self.min_success_threshold = min_success_threshold
        self.canonical_patterns: Set[tuple] = set()

    def consolidate(self) -> int:
        """
        Promotes sequences that exceed the success threshold.
        Returns the number of patterns promoted.
        """
        promoted_count = 0
        # Access raw counts from the cache
        # Assumption: Engram has a _counts dict: {rel_tuple -> int}
        for rel_seq, count in self.cache._counts.items():
            if count >= self.min_success_threshold:
                if rel_seq not in self.canonical_patterns:
                    self.canonical_patterns.add(rel_seq)
                    promoted_count += 1
        
        logger.info(f"Consolidated Engram: promoted {promoted_count} new canonical patterns.")
        return promoted_count

    def get_canonical_patterns(self) -> List[tuple]:
        return list(self.canonical_patterns)

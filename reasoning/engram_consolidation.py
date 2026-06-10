"""
Engram consolidation and promotion logic (Phase 64).

This module provides the logic to promote high-utility reasoning patterns
from the dynamic Engram cache into permanent 'Canonical Engrams'.

Sparse selective consolidation (Phase 241) is inspired by the sparse selective
readback mechanism in Behrouz et al. (2026) "Memory Caching: RNNs with Growing
Memory" (arXiv:2602.24281).  Instead of promoting every pattern that exceeds a
fixed count threshold, sparse_consolidate() ranks all candidate patterns by a
composite score (usage × confidence) and keeps only the top-k — preventing
unbounded canonical-pattern growth and prioritising the most reliably useful
relation sequences over merely frequent ones.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple
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
        for rel_seq, count in self.cache._counts.items():
            if count >= self.min_success_threshold:
                if rel_seq not in self.canonical_patterns:
                    self.canonical_patterns.add(rel_seq)
                    promoted_count += 1

        logger.info(f"Consolidated Engram: promoted {promoted_count} new canonical patterns.")
        return promoted_count

    def sparse_consolidate(self, top_k: int = 100) -> int:
        """
        Sparse selective promotion — retain only the top-k patterns by composite score.

        Composite score = usage_count × mean_path_confidence, where
        mean_path_confidence is drawn from cache._confidence if available,
        otherwise falls back to usage_count alone.

        Inspired by the sparse selective readback mechanism in:
          Behrouz, A., Li, Z., Deng, Y., Zhong, P., Razaviyayn, M., & Mirrokni, V.
          (2026). Memory Caching: RNNs with Growing Memory. arXiv:2602.24281.

        Unlike consolidate(), which promotes all patterns above a threshold,
        sparse_consolidate() enforces a hard cap on canonical-pattern count.
        This prevents memory growth proportional to graph size and ensures that
        the consolidation budget is spent on the most reliable patterns.

        Parameters
        ----------
        top_k : int
            Maximum number of canonical patterns to retain after this call.

        Returns
        -------
        int
            Number of net-new patterns added to canonical_patterns.
        """
        counts: Dict[tuple, int] = getattr(self.cache, "_counts", {})
        confidences: Dict[tuple, float] = getattr(self.cache, "_confidence", {})

        scored: List[Tuple[float, tuple]] = []
        for rel_seq, count in counts.items():
            conf = confidences.get(rel_seq, 1.0)
            scored.append((count * conf, rel_seq))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = {seq for _, seq in scored[:top_k]}

        previously_canonical = len(self.canonical_patterns)
        self.canonical_patterns = selected
        promoted = len(self.canonical_patterns) - previously_canonical

        logger.info(
            "sparse_consolidate: top_k=%d  selected=%d  net_new=%d",
            top_k, len(self.canonical_patterns), promoted,
        )
        return max(promoted, 0)

    def get_canonical_patterns(self) -> List[tuple]:
        return list(self.canonical_patterns)

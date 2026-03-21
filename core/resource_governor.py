"""
Resource Governor for Parallax.

Provides dynamic, process-aware resource management to prevent OOM
without being premature. Tracks memory pressure and computational energy.
"""
import os
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger("parallax.resource_governor")

class ResourceGovernor:
    """
    Monitors system and process resources to provide a scalable 'Energy Budget'.
    """
    def __init__(self, memory_threshold_pct: float = 85.0, safety_buffer_mb: int = 500):
        """
        Args:
            memory_threshold_pct: Stop expansions if system RAM exceeds this %.
            safety_buffer_mb: Minimum free RAM to maintain for process stability.
        """
        self.process = psutil.Process(os.getpid())
        self.threshold = memory_threshold_pct
        self.buffer_bytes = safety_buffer_mb * 1024 * 1024

    def get_current_stats(self) -> Dict[str, Any]:
        """Return real-time resource consumption stats."""
        mem = psutil.virtual_memory()
        proc_mem = self.process.memory_info().rss
        return {
            "system_ram_pct": mem.percent,
            "system_ram_free_mb": mem.available // (1024 * 1024),
            "process_rss_mb": proc_mem // (1024 * 1024),
        }

    def can_expand(self, current_expansions: int, max_budget: int) -> bool:
        """
        Check if we have enough 'Energy' and 'Memory' to continue.
        
        This is the core logic for scalable capping:
        1. Energy check (Soft Cap): The user-defined expansion count.
        2. Memory check (Hard Cap): Real-time process/system health.
        """
        # 1. Hard Expansion Cap (User defined)
        if current_expansions >= max_budget:
            logger.debug(f"Governor: Hit energy cap ({max_budget})")
            return False

        # 2. Dynamic Memory Pressure Check
        mem = psutil.virtual_memory()
        
        # If system is under high pressure (> 85%), stop immediately
        if mem.percent > self.threshold:
            logger.warning(f"Governor: High system memory pressure ({mem.percent}%)")
            return False

        # If available memory falls below our safety buffer, stop
        if mem.available < self.buffer_bytes:
            logger.warning(f"Governor: Available RAM ({mem.available // 1024**2}MB) below safety buffer")
            return False

        return True

    def estimate_path_capacity(self, avg_path_bytes: int = 1024) -> int:
        """
        Scalable Estimation: How many paths can we safely store in the beam?
        Used to prevent premature capping on large machines.
        """
        mem = psutil.virtual_memory()
        safe_mem = mem.available - self.buffer_bytes
        if safe_mem <= 0:
            return 0
        return safe_mem // avg_path_bytes

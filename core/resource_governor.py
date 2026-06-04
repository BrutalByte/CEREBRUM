"""
Resource Governor for CEREBRUM.

Provides dynamic, process-aware resource management to prevent OOM
without being premature. Tracks both system RAM and GPU VRAM, and
exposes an energy-budget API used by BeamTraversal to cap expansion.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import psutil

logger = logging.getLogger("cerebrum.resource_governor")


class ResourceGovernor:
    """
    Monitors system RAM and GPU VRAM to provide a scalable 'Energy Budget'.

    On machines with no GPU the VRAM methods return safe defaults so
    callers need not branch on hardware availability.

    Parameters
    ----------
    memory_threshold_pct : float
        Stop expansions when system RAM usage exceeds this percentage (default 85%).
    safety_buffer_mb : int
        Minimum free system RAM to maintain for process stability (default 500 MB).
    vram_safety_buffer_mb : int
        Minimum free VRAM to maintain before reporting GPU as usable (default 256 MB).
    """

    def __init__(
        self,
        memory_threshold_pct: float = 95.0,
        safety_buffer_mb: int = 200,
        vram_safety_buffer_mb: int = 256,
        max_ram_gb: Optional[float] = None,
        max_vram_gb: Optional[float] = None,
    ):
        self.process = psutil.Process(os.getpid())
        self.threshold = memory_threshold_pct
        self.buffer_bytes = safety_buffer_mb * 1024 * 1024
        self.vram_buffer_mb = vram_safety_buffer_mb
        self.max_ram_bytes = (max_ram_gb * 1024**3) if max_ram_gb else None
        self.max_vram_bytes = (max_vram_gb * 1024**3) if max_vram_gb else None

    # ------------------------------------------------------------------
    # System RAM
    # ------------------------------------------------------------------

    def get_current_stats(self) -> Dict[str, Any]:
        """Return real-time system RAM consumption stats."""
        mem = psutil.virtual_memory()
        proc_mem = self.process.memory_info().rss
        return {
            "system_ram_pct": mem.percent,
            "system_ram_free_mb": mem.available // (1024 * 1024),
            "process_rss_mb": proc_mem // (1024 * 1024),
            "ram_limit_gb": self.max_ram_bytes / (1024**3) if self.max_ram_bytes else None
        }

    def can_expand(self, current_expansions: int, max_budget: int) -> bool:
        """
        Check if there is enough energy and RAM to continue beam expansion.

        Two checks:
        1. **Energy cap** (soft): user-defined expansion count.
        2. **Memory pressure** (hard): real-time system RAM health.
        """
        if current_expansions >= max_budget:
            logger.debug("Governor: hit energy cap (%d)", max_budget)
            return False

        mem = psutil.virtual_memory()
        
        # Check relative threshold
        if mem.percent > self.threshold:
            logger.warning(
                "Governor: high system RAM pressure (%.1f%%)", mem.percent
            )
            return False
            
        # Check absolute limit (Phase 168)
        proc_mem = self.process.memory_info().rss
        if self.max_ram_bytes and (proc_mem + self.buffer_bytes >= self.max_ram_bytes):
            logger.warning(
                "Governor: process RSS (%d MB) exceeds max_ram_gb limit (%d MB)",
                proc_mem // (1024**2), self.max_ram_bytes // (1024**2)
            )
            return False

        if mem.available < self.buffer_bytes:
            logger.warning(
                "Governor: available RAM (%d MB) below safety buffer",
                mem.available // (1024 ** 2),
            )
            return False

        return True

    def estimate_path_capacity(self, avg_path_bytes: int = 1024) -> int:
        """
        Estimate how many paths can safely be stored in the beam given
        current available system RAM.  Scales automatically on large machines.
        """
        mem = psutil.virtual_memory()
        safe_mem = mem.available - self.buffer_bytes
        if safe_mem <= 0:
            return 0
        return safe_mem // avg_path_bytes

    # ------------------------------------------------------------------
    # GPU VRAM
    # ------------------------------------------------------------------

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Return real-time GPU VRAM stats for the best available CUDA device.

        Returns a dict with ``gpu_available=False`` when no CUDA device
        is present (e.g., CPU-only, MPS, HPU, or XLA environments).
        On Jetson the VRAM pool is the same as system RAM; ``is_jetson``
        is flagged so callers can interpret the numbers accordingly.
        """
        from core.hardware import HAS_CUDA, IS_JETSON, get_best_cuda_device, get_gpu_vram_mb

        if not HAS_CUDA:
            return {"gpu_available": False, "is_jetson": IS_JETSON}

        idx = get_best_cuda_device()
        free_mb, total_mb = get_gpu_vram_mb(idx)
        used_mb = total_mb - free_mb
        used_pct = round(used_mb / max(total_mb, 1) * 100, 1)

        return {
            "gpu_available": True,
            "device_index": idx,
            "vram_free_mb": free_mb,
            "vram_total_mb": total_mb,
            "vram_used_mb": used_mb,
            "vram_used_pct": used_pct,
            "is_jetson": IS_JETSON,
        }

    def can_use_gpu(self, required_mb: int = 256) -> bool:
        """
        Return True if a CUDA device has enough free VRAM AND we are within
        the governor's absolute VRAM budget (max_vram_gb).
        """
        from core.hardware import HAS_CUDA, get_best_cuda_device, get_gpu_vram_mb
        import torch

        if not HAS_CUDA:
            return False
            
        # Check absolute process budget (Phase 168)
        if self.max_vram_bytes:
            used_bytes = torch.cuda.memory_allocated()
            if used_bytes + (required_mb * 1024**2) + self.vram_buffer_mb * 1024**2 >= self.max_vram_bytes:
                logger.warning(
                    "Governor: process VRAM (%d MB) exceeds max_vram_gb limit (%d MB)",
                    used_bytes // (1024**2), self.max_vram_bytes // (1024**2)
                )
                return False

        idx = get_best_cuda_device()
        free_mb, _ = get_gpu_vram_mb(idx)
        needed = required_mb + self.vram_buffer_mb
        if free_mb < needed:
            logger.warning(
                "Governor: insufficient physical VRAM â€” %d MB free, %d MB needed "
                "(requested %d + %d buffer)",
                free_mb, needed, required_mb, self.vram_buffer_mb,
            )
            return False
        return True

    def get_combined_stats(self) -> Dict[str, Any]:
        """Return a merged dict of both system RAM and GPU stats."""
        stats = self.get_current_stats()
        stats.update(self.get_gpu_stats())
        return stats

    def check_constraints(self, cpu_limit_pct: float = 80.0) -> bool:
        """Production check: enforce CPU and Memory usage constraints."""
        stats = self.get_current_stats()
        cpu = psutil.cpu_percent()
        if stats["system_ram_pct"] > self.threshold:
            logger.warning(f"Memory limit reached: {stats['system_ram_pct']}%")
            return False
        if cpu > cpu_limit_pct:
            logger.warning(f"CPU limit reached: {cpu}%")
            return False
        return True

"""
MemoryGovernor: Hardware-aware resource management for Hybrid-Memory CEREBRUM.
Controls RAM/VRAM resource budgets and triggers Mmap spill-over.
"""
import os
import psutil
import torch
from typing import Optional, Dict

class MemoryGovernor:
    def __init__(
        self,
        max_ram_gb: Optional[float] = None,
        max_vram_gb: Optional[float] = None,
        safety_buffer_mb: int = 512
    ):
        self.max_ram_bytes = (max_ram_gb * 1024**3) if max_ram_gb else None
        self.max_vram_bytes = (max_vram_gb * 1024**3) if max_vram_gb else None
        self.safety_buffer_bytes = safety_buffer_mb * 1024**2
        
    def get_stats(self) -> Dict[str, float]:
        """Return current memory utilization and governor limits."""
        ram = psutil.virtual_memory()
        stats = {
            "ram_used_bytes": ram.used,
            "ram_total_bytes": ram.total,
            "ram_limit_bytes": self.max_ram_bytes if self.max_ram_bytes else ram.total
        }
        
        if torch.cuda.is_available():
            stats["vram_used_bytes"] = torch.cuda.memory_allocated()
            stats["vram_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            stats["vram_limit_bytes"] = self.max_vram_bytes if self.max_vram_bytes else stats["vram_total_bytes"]
            
        return stats

    def is_spill_needed(self) -> bool:
        """Check if memory usage exceeds configured Governor limits."""
        stats = self.get_stats()
        
        # Check RAM
        if self.max_ram_bytes and (stats["ram_used_bytes"] + self.safety_buffer_bytes >= self.max_ram_bytes):
            return True
            
        # Check VRAM if GPU is active
        if "vram_used_bytes" in stats and self.max_vram_bytes:
            if stats["vram_used_bytes"] + self.safety_buffer_bytes >= self.max_vram_bytes:
                return True
                
        return False

"""
MmapAdapter: Memory-mapped graph storage providing disk-backed reasoning.
Configurable via CEREBRUM_USE_MMAP environment flag.
"""
import mmap
import os
import struct
from pathlib import Path
from typing import List, Tuple, Dict, Any
from core.hardware_governor import MemoryGovernor

class MmapAdapter:
    def __init__(self, data_dir: str, governor: Optional[MemoryGovernor] = None):
        self.data_dir = Path(data_dir)
        self.governor = governor
        
        # Open binary files
        self.f_a = open(self.data_dir / "graph.a", "rb")
        self.f_e = open(self.data_dir / "graph.e", "rb")
        
        # Mmap files
        self.a_map = mmap.mmap(self.f_a.fileno(), 0, access=mmap.ACCESS_READ)
        self.e_map = mmap.mmap(self.f_e.fileno(), 0, access=mmap.ACCESS_READ)

    def get_neighbors(self, node_idx: int) -> List[Tuple[int, float]]:
        """Fetch neighbors using direct pointer arithmetic on mmap blocks."""
        # A-File record size is 32 bytes
        record_start = node_idx * 32
        
        # Unpack: Q (id), I (degree), Q (offset), H (community)
        _, degree, offset, _ = struct.unpack("<QIQH", self.a_map[record_start:record_start+22])
        
        neighbors = []
        # Edge size is 12 bytes
        edge_start = offset
        for _ in range(degree):
            target_id, _, conf = struct.unpack("<QH e", self.e_map[edge_start:edge_start+12])
            neighbors.append((target_id, float(conf)))
            edge_start += 12
            
        return neighbors

    def close(self):
        self.a_map.close()
        self.e_map.close()
        self.f_a.close()
        self.f_e.close()

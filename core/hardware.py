"""
Hardware Abstraction Layer for CEREBRUM.

Probes all available compute backends at import time and exposes a unified
API for device selection, VRAM introspection, and platform detection.

Supported backends (in priority order for 'auto' selection):
  1. NVIDIA CUDA       — torch.cuda (also covers AMD ROCm via same API)
  2. Apple MPS         — torch.backends.mps  (Apple Silicon M1/M2/M3/M4)
  3. Intel Gaudi / HPU — habana_frameworks or torch.hpu  (Gaudi 2/3)
  4. XLA               — torch_xla  (Google TPU v4/v5, AWS Trainium/Inferentia)
  5. CPU               — always available (ARM64 and Jetson variants noted)

RAPIDS (cuGraph/cuDF) and CuPy are probed separately for graph-specific
GPU acceleration and are NVIDIA-only.

Usage::

    from core.hardware import device_info, get_best_cuda_device, get_gpu_vram_mb
    print(device_info())
"""
from __future__ import annotations

import logging
import os
import platform
import psutil
import time
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import torch

logger = logging.getLogger("cerebrum.hardware")

# ---------------------------------------------------------------------------
# 1. RAPIDS (cuGraph, cuDF) — NVIDIA only
# ---------------------------------------------------------------------------
try:
    import cugraph  # type: ignore
    import cudf  # type: ignore
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False

# ---------------------------------------------------------------------------
# 2. CuPy — NVIDIA only
# ---------------------------------------------------------------------------
try:
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ---------------------------------------------------------------------------
# 3. PyTorch — probe all device backends
# ---------------------------------------------------------------------------
HAS_TORCH = True

# --- NVIDIA CUDA / AMD ROCm -------------------------------------------
HAS_CUDA = torch.cuda.is_available()
HAS_ROCM = HAS_CUDA and (getattr(torch.version, "hip", None) is not None)

# --- Apple Silicon MPS ------------------------------------------------
HAS_MPS = (
    getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
)

# --- Intel Gaudi / HPU -----------------------------------------------
try:
    import habana_frameworks.torch.hpu as _hpu_mod  # type: ignore
    HAS_HPU = _hpu_mod.is_available()
except ImportError:
    HAS_HPU = hasattr(torch, "hpu") and callable(
        getattr(getattr(torch, "hpu", None), "is_available", None)
    ) and torch.hpu.is_available()  # type: ignore[attr-defined]

# --- Google TPU / AWS Trainium & Inferentia ---------------------------
try:
    import torch_xla.core.xla_model as _xm  # type: ignore
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# ---------------------------------------------------------------------------
# 4. Platform / system characteristics
# ---------------------------------------------------------------------------

IS_ARM64: bool = platform.machine().lower() in ("aarch64", "arm64")
IS_JETSON: bool = os.path.exists("/etc/nv_tegra_release")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def device_info() -> dict:
    n_cuda = torch.cuda.device_count() if HAS_CUDA else 0
    if HAS_CUDA and not HAS_ROCM:
        backend = f"GPU — NVIDIA CUDA ({n_cuda}× device)"
    elif HAS_ROCM:
        backend = f"GPU — AMD ROCm ({n_cuda}× device)"
    elif HAS_MPS:
        backend = "GPU — Apple Metal/MPS (unified memory)"
    elif HAS_HPU:
        backend = "Accelerator — Intel Gaudi/HPU"
    elif HAS_XLA:
        backend = "XLA — Google TPU / AWS Trainium / Inferentia"
    else:
        parts = ["CPU"]
        if IS_ARM64: parts.append("ARM64")
        if IS_JETSON: parts.append("Jetson unified-memory")
        backend = " — ".join(parts)

    return {
        "gpu_available": any([HAS_CUDA, HAS_MPS, HAS_HPU, HAS_XLA]),
        "rapids_enabled": HAS_RAPIDS,
        "cupy_enabled": HAS_CUPY,
        "cuda_enabled": HAS_CUDA,
        "rocm_enabled": HAS_ROCM,
        "mps_enabled": HAS_MPS,
        "hpu_enabled": HAS_HPU,
        "xla_enabled": HAS_XLA,
        "is_arm64": IS_ARM64,
        "is_jetson": IS_JETSON,
        "cuda_device_count": n_cuda,
        "backend": backend,
    }

def get_best_cuda_device() -> int:
    if not HAS_CUDA:
        return 0
    best_idx, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free, best_idx = free, i
        except Exception:
            pass
    return best_idx

def get_gpu_vram_mb(device_idx: int = 0) -> Tuple[int, int]:
    if not HAS_CUDA:
        return (0, 0)
    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        return (free // (1024 ** 2), total // (1024 ** 2))
    except Exception:
        return (0, 0)

def resolve_torch_device(preference: str = "auto"):
    if preference != "auto":
        return torch.device(preference)
    if HAS_CUDA: return torch.device(f"cuda:{get_best_cuda_device()}")
    if HAS_MPS: return torch.device("mps")
    if HAS_HPU: return torch.device("hpu")
    if HAS_XLA:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device("cpu")

def to_gpu_graph(G_nx):
    if not HAS_RAPIDS:
        return G_nx, False
    try:
        df = nx_to_pandas_edgelist(G_nx)
        gdf = cudf.from_pandas(df)
        G_cuda = cugraph.Graph(directed=G_nx.is_directed())
        G_cuda.from_cudf_edgelist(
            gdf, source="source", destination="target",
            edge_attr="weight" if "weight" in df.columns else None,
            renumber=True,
        )
        return G_cuda, True
    except Exception as e:
        logger.warning("Failed to move graph to GPU: %s. Falling back to CPU.", e)
        return G_nx, False

def nx_to_pandas_edgelist(G):
    import pandas as pd
    edges = []
    for u, v, data in G.edges(data=True):
        edge = {"source": u, "target": v}
        edge.update(data)
        edges.append(edge)
    return pd.DataFrame(edges)

def get_xp():
    if HAS_CUPY: return cp
    import numpy as np
    return np

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
        ram = psutil.virtual_memory()
        stats = {
            "ram_used_bytes": ram.used,
            "ram_total_bytes": ram.total,
            "ram_limit_bytes": self.max_ram_bytes if self.max_ram_bytes else ram.total
        }
        if HAS_CUDA:
            stats["vram_used_bytes"] = torch.cuda.memory_allocated()
            stats["vram_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            stats["vram_limit_bytes"] = self.max_vram_bytes if self.max_vram_bytes else stats["vram_total_bytes"]
        return stats

    def is_spill_needed(self) -> bool:
        stats = self.get_stats()
        if self.max_ram_bytes and (stats["ram_used_bytes"] + self.safety_buffer_bytes >= self.max_ram_bytes):
            return True
        if "vram_used_bytes" in stats and self.max_vram_bytes:
            if stats["vram_used_bytes"] + self.safety_buffer_bytes >= self.max_vram_bytes:
                return True
        return False

class HardwareManager:
    """
    HardwareManager: OS-agnostic utility for SSD discovery, formatting, 
    and performance telemetry.
    """
    @staticmethod
    def list_drives() -> List[Dict[str, Any]]:
        drives = []
        for part in psutil.disk_partitions():
            if 'cdrom' in part.opts or not part.device: continue
            if part.mountpoint in ["/", "C:\\"]: continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
                drives.append({
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "total_gb": usage.total // (1024**3),
                    "free_gb": usage.free // (1024**3),
                    "percent": usage.percent
                })
            except PermissionError: continue
        return drives

    @staticmethod
    def initialize_drive(mountpoint: str) -> str:
        path = Path(mountpoint) / "cerebrum_data"
        if not path.exists():
            path.mkdir(parents=True)
        return str(path)

    @staticmethod
    def get_io_stats(path: str) -> Dict[str, float]:
        io_start = psutil.disk_io_counters()
        time.sleep(0.1)
        io_end = psutil.disk_io_counters()
        read_mb = (io_end.read_bytes - io_start.read_bytes) / (1024**2 * 0.1)
        write_mb = (io_end.write_bytes - io_start.write_bytes) / (1024**2 * 0.1)
        return {"read_mb_s": read_mb, "write_mb_s": write_mb}

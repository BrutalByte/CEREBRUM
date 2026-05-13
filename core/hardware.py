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
from typing import Tuple, Optional, Dict

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
try:
    import torch

    HAS_TORCH = True

    # --- NVIDIA CUDA / AMD ROCm -------------------------------------------
    # Both appear as torch.cuda; ROCm sets torch.version.hip to a HIP version
    # string so we can distinguish NVIDIA vs AMD without extra dependencies.
    HAS_CUDA = torch.cuda.is_available()
    HAS_ROCM = HAS_CUDA and (getattr(torch.version, "hip", None) is not None)

    # --- Apple Silicon MPS ------------------------------------------------
    HAS_MPS = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )

    # --- Intel Gaudi / HPU -----------------------------------------------
    # Two detection paths: habana_frameworks (official SDK) and torch.hpu
    # which is natively integrated in PyTorch ≥ 2.3.
    try:
        import habana_frameworks.torch.hpu as _hpu_mod  # type: ignore
        HAS_HPU = _hpu_mod.is_available()
    except ImportError:
        HAS_HPU = hasattr(torch, "hpu") and callable(
            getattr(getattr(torch, "hpu", None), "is_available", None)
        ) and torch.hpu.is_available()  # type: ignore[attr-defined]

    # --- Google TPU / AWS Trainium & Inferentia ---------------------------
    # torch-xla provides a unified XLA device abstraction for Cloud TPU
    # (v4/v5p), AWS Trainium (Trn1), and AWS Inferentia 2 (Inf2).
    try:
        import torch_xla.core.xla_model as _xm  # type: ignore
        HAS_XLA = True
    except ImportError:
        HAS_XLA = False

except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False
    HAS_CUDA = False
    HAS_ROCM = False
    HAS_MPS = False
    HAS_HPU = False
    HAS_XLA = False

# ---------------------------------------------------------------------------
# 4. Platform / system characteristics
# ---------------------------------------------------------------------------

#: True on AWS Graviton, Ampere Altra, Apple M-series running Linux, RPi, etc.
IS_ARM64: bool = platform.machine().lower() in ("aarch64", "arm64")

#: True on NVIDIA Jetson devices (Orin, Xavier, Nano, TX2 …).
#: Jetson uses unified memory — VRAM and system RAM are the same pool.
IS_JETSON: bool = os.path.exists("/etc/nv_tegra_release")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def device_info() -> dict:
    """
    Return a dictionary describing all detected acceleration hardware.

    Keys
    ----
    gpu_available   : bool  — any GPU/accelerator detected
    rapids_enabled  : bool  — NVIDIA RAPIDS (cuGraph/cuDF)
    cupy_enabled    : bool  — CuPy array library
    cuda_enabled    : bool  — NVIDIA CUDA or AMD ROCm via torch.cuda
    rocm_enabled    : bool  — AMD ROCm specifically
    mps_enabled     : bool  — Apple Silicon MPS
    hpu_enabled     : bool  — Intel Gaudi HPU
    xla_enabled     : bool  — Google TPU / AWS Trainium / Inferentia
    is_arm64        : bool  — ARM64 CPU architecture
    is_jetson       : bool  — NVIDIA Jetson (unified memory)
    cuda_device_count: int  — number of CUDA/ROCm devices visible
    backend         : str   — human-readable primary backend label
    """
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
        if IS_ARM64:
            parts.append("ARM64")
        if IS_JETSON:
            parts.append("Jetson unified-memory")
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
    """
    Return the CUDA/ROCm device index with the most free VRAM.

    On multi-GPU systems (DGX H100 ×8, MI300X cluster, etc.) this avoids
    always allocating to GPU 0 and instead picks the least-loaded card.
    Falls back to index 0 if VRAM info is unavailable.
    """
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
    """
    Return ``(free_mb, total_mb)`` for a CUDA/ROCm device.

    Returns ``(0, 0)`` when CUDA is unavailable or the query fails
    (e.g., on Jetson where ``mem_get_info`` may not be exposed).
    """
    if not HAS_CUDA:
        return (0, 0)
    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        return (free // (1024 ** 2), total // (1024 ** 2))
    except Exception:
        return (0, 0)


def resolve_torch_device(preference: str = "auto"):
    """
    Return the best available ``torch.device`` following the priority chain:
    CUDA (best card) → MPS → HPU → XLA → CPU.

    Parameters
    ----------
    preference : str
        ``"auto"``   — pick best available (default)
        ``"cuda"``   — force CUDA (raises if unavailable)
        ``"mps"``    — force MPS
        ``"hpu"``    — force HPU
        ``"xla"``    — force XLA
        ``"cpu"``    — force CPU
    Any other string is passed directly to ``torch.device()``.
    """
    if not HAS_TORCH:
        return "cpu"

    if preference != "auto":
        return torch.device(preference)

    if HAS_CUDA:
        return torch.device(f"cuda:{get_best_cuda_device()}")
    if HAS_MPS:
        return torch.device("mps")
    if HAS_HPU:
        return torch.device("hpu")
    if HAS_XLA:
        import torch_xla.core.xla_model as xm  # type: ignore
        return xm.xla_device()
    return torch.device("cpu")


def to_gpu_graph(G_nx):
    """
    Convert a NetworkX graph to a cuGraph object if RAPIDS is available.
    Returns ``(graph, is_gpu)``.
    """
    if not HAS_RAPIDS:
        return G_nx, False
    try:
        df = nx_to_pandas_edgelist(G_nx)
        gdf = cudf.from_pandas(df)
        G_cuda = cugraph.Graph(directed=G_nx.is_directed())
        G_cuda.from_cudf_edgelist(
            gdf,
            source="source",
            destination="target",
            edge_attr="weight" if "weight" in df.columns else None,
            renumber=True,
        )
        return G_cuda, True
    except Exception as e:
        logger.warning("Failed to move graph to GPU: %s. Falling back to CPU.", e)
        return G_nx, False


def nx_to_pandas_edgelist(G):
    """Helper: NetworkX graph → Pandas edge-list DataFrame."""
    import pandas as pd
    edges = []
    for u, v, data in G.edges(data=True):
        edge = {"source": u, "target": v}
        edge.update(data)
        edges.append(edge)
    return pd.DataFrame(edges)


def get_xp():
    """Return ``cupy`` if available, otherwise ``numpy``."""
    if HAS_CUPY:
        return cp
    import numpy as np
    return np


class MemoryGovernor:
    """
    Hardware-aware resource management for Hybrid-Memory CEREBRUM.
    Controls RAM/VRAM resource budgets and triggers Mmap spill-over.
    """
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
        
        if HAS_CUDA:
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

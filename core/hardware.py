"""
Hardware Abstraction Layer for Parallax.

Handles agnostic dispatch between CPU (NetworkX/NumPy) and GPU (RAPIDS/CuPy/Torch).
Ensures scalability across diverse hardware environments.
"""
import logging
import functools
from typing import Optional, Any

logger = logging.getLogger("parallax.hardware")

# 1. Check for RAPIDS (cuGraph, cuDF)
try:
    import cugraph
    import cudf
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False

# 2. Check for CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# 3. Check for PyTorch/CUDA
try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

def device_info():
    """Returns a dictionary of available acceleration hardware."""
    return {
        "gpu_available": HAS_CUDA or HAS_RAPIDS,
        "rapids_enabled": HAS_RAPIDS,
        "cupy_enabled": HAS_CUPY,
        "cuda_enabled": HAS_CUDA,
        "backend": "GPU (RAPIDS/CUDA)" if (HAS_RAPIDS or HAS_CUDA) else "CPU"
    }

def to_gpu_graph(G_nx):
    """
    Convert a NetworkX graph to a cuGraph object if RAPIDS is available.
    Returns (graph, is_gpu).
    """
    if not HAS_RAPIDS:
        return G_nx, False
    
    try:
        # Convert NetworkX to cuGraph via cuDF edgelist
        import pandas as pd
        df = nx_to_pandas_edgelist(G_nx)
        gdf = cudf.from_pandas(df)
        
        G_cuda = cugraph.Graph(directed=G_nx.is_directed())
        G_cuda.from_cudf_edgelist(
            gdf, 
            source='source', 
            destination='target', 
            edge_attr='weight' if 'weight' in df.columns else None,
            renumber=True
        )
        return G_cuda, True
    except Exception as e:
        logger.warning(f"Failed to move graph to GPU: {e}. Falling back to CPU.")
        return G_nx, False

def nx_to_pandas_edgelist(G):
    """Helper for NetworkX -> Pandas conversion."""
    import pandas as pd
    edges = []
    for u, v, data in G.edges(data=True):
        edge = {'source': u, 'target': v}
        edge.update(data)
        edges.append(edge)
    return pd.DataFrame(edges)

def get_xp():
    """Returns cupy if available, otherwise numpy."""
    if HAS_CUPY:
        return cp
    import numpy as np
    return np

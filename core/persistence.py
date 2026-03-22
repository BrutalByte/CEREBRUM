"""
Persistence layer for Parallax — save/load graph state, communities, and metadata.

Parallax states (especially DSCF communities and structural encodings) can
be expensive to compute on large graphs. This module provides a unified
mechanism to serialize the current state to disk and reload it instantly.
"""
import pickle
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Security: Define the root for all persistent data. 
# In production, this should be configurable but restricted.
SAFE_DATA_DIR = Path(os.getenv("PARALLAX_DATA_DIR", "data/parallax")).absolute()

def _resolve_safe_path(file_path: str) -> Path:
    """
    Ensure the file_path is within the SAFE_DATA_DIR to prevent path traversal.
    """
    requested_path = Path(file_path)
    # If the user passed an absolute path, we only allow it if it starts with SAFE_DATA_DIR
    if requested_path.is_absolute():
        final_path = requested_path.resolve()
    else:
        # Join relative path to our sandbox
        final_path = (SAFE_DATA_DIR / requested_path).resolve()

    if not str(final_path).startswith(str(SAFE_DATA_DIR)):
        raise PermissionError(f"Security: Path traversal attempt blocked: {file_path}")
    
    return final_path

def save_state(
    file_path: str,
    adapter: Any,
    community_map: Dict[str, int],
    embeddings: Dict[str, np.ndarray],
    csa_metadata: Dict[str, Any],
    default_edge_type_weights: Optional[Dict[str, float]] = None,
    hologram: Optional[Any] = None,
) -> None:
    """
    Serialize the entire Parallax state to a binary pickle file.
    Only allows paths within the SAFE_DATA_DIR sandbox.
    """
    path = _resolve_safe_path(file_path)
    
    state = {
        "version": "0.2.0",
        "timestamp": time.time(),
        "adapter": adapter,
        "community_map": community_map,
        "embeddings": embeddings,
        "csa_metadata": csa_metadata,
        "default_edge_type_weights": default_edge_type_weights,
        "hologram": hologram,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(state, f)
    
    print(f"  [Persistence] State saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_state(file_path: str) -> Dict[str, Any]:
    """
    Load a Parallax state from a pickle file.
    Only allows paths within the SAFE_DATA_DIR sandbox.
    """
    path = _resolve_safe_path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Parallax state file not found: {path}")
    
    t0 = time.time()
    with open(path, "rb") as f:
        state = pickle.load(f)
    
    print(f"  [Persistence] State loaded from {path} in {time.time() - t0:.2f}s")
    return state


def is_state_cached(file_path: str) -> bool:
    """Return True if the state file exists and is within sandbox."""
    try:
        path = _resolve_safe_path(file_path)
        return path.exists()
    except PermissionError:
        return False




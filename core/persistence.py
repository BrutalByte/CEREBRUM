"""
Persistence layer for Parallax — save/load graph state, communities, and metadata.

Parallax states (especially DSCF communities and structural encodings) can
be expensive to compute on large graphs. This module provides a unified
mechanism to serialize the current state to disk and reload it instantly.
"""
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def save_state(
    file_path: str,
    adapter: Any,
    community_map: Dict[str, int],
    embeddings: Dict[str, np.ndarray],
    csa_metadata: Dict[str, Any],
    default_edge_type_weights: Optional[Dict[str, float]] = None,
) -> None:
    """
    Serialize the entire Parallax state to a binary pickle file.

    Parameters
    ----------
    file_path        : destination path (e.g. 'data/state.pkl')
    adapter          : the GraphAdapter instance (must be pickleable)
    community_map    : {node_id -> community_id}
    embeddings       : {node_id -> float32 vector}
    csa_metadata     : {"distances": dict, "adjacent_pairs": set}
    default_edge_type_weights : optional weights
    """
    state = {
        "version": "0.1.0",
        "timestamp": time.time(),
        "adapter": adapter,
        "community_map": community_map,
        "embeddings": embeddings,
        "csa_metadata": csa_metadata,
        "default_edge_type_weights": default_edge_type_weights,
    }
    
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(state, f)
    
    print(f"  [Persistence] State saved to {file_path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_state(file_path: str) -> Dict[str, Any]:
    """
    Load a Parallax state from a pickle file.

    Returns
    -------
    Dict containing the keys used in save_state.
    Raises FileNotFoundError if file is missing.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parallax state file not found: {file_path}")
    
    t0 = time.time()
    with open(path, "rb") as f:
        state = pickle.load(f)
    
    print(f"  [Persistence] State loaded from {file_path} in {time.time() - t0:.2f}s")
    return state


def is_state_cached(file_path: str) -> bool:
    """Return True if the state file exists."""
    return Path(file_path).exists()




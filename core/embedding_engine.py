"""
Entity embedding engines for CEREBRUM.

The EmbeddingEngine ABC defines the interface. Three implementations ship:
  - RandomEngine  : seeded random unit vectors — always available, used in tests
  - SentenceEngine: sentence-transformers, zero training required (default for production)
  - GraphSAGELayer: one-pass neighborhood mean aggregation — smooths any base embeddings
                    using graph structure, making the CSA alpha (semantic) term richer.

Add TransEEngine / RotatEEngine (via pykeen) for graph-structure-aware embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx


class EmbeddingEngine(ABC):
    """Abstract entity embedding engine."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of produced embedding vectors."""
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into an [N x dim] float32 matrix.
        Vectors should be L2-normalized (unit norm) for cosine similarity to work.
        """
        ...

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience: encode a single text string."""
        return self.encode([text])[0]

    def encode_entities(self, entities: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Encode a {entity_id -> label} dict, returning {entity_id -> vector}.
        This is the primary entry point from the forward pass.
        """
        if not entities:
            return {}
        ids   = list(entities.keys())
        texts = [entities[i] for i in ids]
        vecs  = self.encode(texts)
        return {entity_id: vecs[j] for j, entity_id in enumerate(ids)}


# ---------------------------------------------------------------------------
# RandomEngine — baseline / testing
# ---------------------------------------------------------------------------

class RandomEngine(EmbeddingEngine):
    """
    Random unit-vector embeddings, seeded deterministically by label hash.

    Semantically meaningless — used for unit tests and as a sanity-check
    baseline. DSCF community membership (beta term) carries full attention
    weight when embeddings are random (alpha term becomes noise).
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        result = []
        for text in texts:
            # Deterministic seed from label hash — same label = same vector
            seed = abs(hash(text)) % (2 ** 32)
            rng  = np.random.default_rng(seed)
            v    = rng.normal(size=self._dim).astype(np.float32)
            v    = v / (np.linalg.norm(v) + 1e-8)
            result.append(v.astype(np.float16))
        return np.array(result, dtype=np.float16)


# ---------------------------------------------------------------------------
# SentenceEngine — production default (Option B from Section 8.1)
# ---------------------------------------------------------------------------

class SentenceEngine(EmbeddingEngine):
    """
    sentence-transformers based engine. No graph-specific training required.

    Encodes entity labels and descriptions using a pre-trained language model.
    Recommended default for zero-shot deployment (Section 8.1, Option B).

    Requires: pip install sentence-transformers
    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: Optional[str] = None):
        try:
            from core.hardware import (
                HAS_CUDA, HAS_MPS, HAS_HPU, HAS_XLA,
                IS_ARM64, IS_JETSON,
                get_best_cuda_device,
            )
            from sentence_transformers import SentenceTransformer

            if device is None:
                if HAS_CUDA:
                    device = f"cuda:{get_best_cuda_device()}"
                elif HAS_MPS:
                    device = "mps"
                elif HAS_HPU:
                    device = "hpu"
                elif HAS_XLA:
                    device = "xla"
                else:
                    device = "cpu"
                    if IS_ARM64 and not IS_JETSON:
                        import logging as _log
                        _log.getLogger("cerebrum.embedding").info(
                            "ARM64 CPU detected. sentence-transformers will use "
                            "the native CPU path. For faster inference install "
                            "PyTorch with ARM64 optimisations or use a CUDA/MPS device."
                        )

            # BGE models typically require a query instruction prefix
            self._model = SentenceTransformer(model_name, device=device)
            # BGE instructions (v1.5)
            self._instruction = "Represent this sentence for searching relevant passages: "
            self._dim   = self._model.get_sentence_embedding_dimension()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceEngine. "
                "Install with: pip install sentence-transformers"
            ) from e

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        # Backward compatibility: no instruction for base encoding
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs.astype(np.float16)

    def encode_query(self, texts: List[str]) -> np.ndarray:
        # BGE models perform better with instructions for queries
        input_texts = [self._instruction + t for t in texts]
        vecs = self._model.encode(
            input_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs.astype(np.float16)


# ---------------------------------------------------------------------------
# GraphSAGE one-pass neighborhood smoother
# ---------------------------------------------------------------------------

def smooth_with_graphsage(
    embeddings: Dict[str, np.ndarray],
    G: "nx.Graph",
    *,
    self_weight: float = 0.5,
    neighbor_weight: float = 0.5,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Apply a single GraphSAGE mean-aggregation pass over ``embeddings``.

    For each node v:
        h_v' = normalize(self_weight * h_v
                         + neighbor_weight * mean({h_u : u ∈ N(v)}))

    Nodes with no neighbours (degree-0) are returned unchanged.

    Parameters
    ----------
    embeddings      : {node_id -> float32/float16 vector}  — existing embeddings
    G               : NetworkX graph providing the adjacency structure
    self_weight     : weight of the node's own embedding (default 0.5)
    neighbor_weight : weight of the aggregated neighbour mean (default 0.5)
    normalize       : L2-normalise the output vectors (default True)

    Returns
    -------
    New dict with the same keys, smoothed embeddings as float32 arrays.
    """
    smoothed: Dict[str, np.ndarray] = {}

    for node in G.nodes():
        node_str = str(node)
        if node_str not in embeddings:
            continue

        h_self = embeddings[node_str].astype(np.float32)

        # Collect neighbour embeddings (neighbours for undirected; both
        # predecessors and successors for directed graphs)
        neighbour_vecs = []
        if hasattr(G, "predecessors"):
            nbrs = set(G.predecessors(node)) | set(G.successors(node))
        else:
            nbrs = set(G.neighbors(node))
        for nbr in nbrs:
            nbr_str = str(nbr)
            if nbr_str in embeddings and nbr_str != node_str:
                neighbour_vecs.append(embeddings[nbr_str].astype(np.float32))

        if neighbour_vecs:
            h_nbr = np.mean(neighbour_vecs, axis=0)
            h_new = self_weight * h_self + neighbor_weight * h_nbr
        else:
            h_new = h_self.copy()

        if normalize:
            norm = np.linalg.norm(h_new)
            if norm > 1e-8:
                h_new = h_new / norm

        smoothed[node_str] = h_new.astype(np.float32)

    return smoothed

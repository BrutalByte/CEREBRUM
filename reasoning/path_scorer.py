"""
Multi-signal path scoring (Section 5, STEP 4).

Combines:
  - Attention weight product (coverage of the reasoning chain)
  - Community coherence (domain consistency across hops)
  - Semantic alignment to the query (optional)
"""
import math
from typing import List, Optional

import numpy as np


def community_coherence(community_sequence: List[int]) -> float:
    """
    Score a community traversal sequence for domain consistency.

      intra-community step  : +1.0
      cross-community step  : +0.5

    A fully intra-community path scores 1.0 (tight local reasoning).
    A single community leap scores ~0.75 (one conceptual bridge).
    Penalizes paths that jump incoherently across unrelated domains.

    Parameters
    ----------
    community_sequence : list of community IDs for each entity node visited,
                         in traversal order. Unknown community = -1 (ignored).

    Returns
    -------
    float in [0.5, 1.0]
    """
    known = [c for c in community_sequence if c >= 0]
    if len(known) <= 1:
        return 1.0

    total = len(known) - 1
    intra = sum(1 for i in range(total) if known[i] == known[i + 1])
    cross = total - intra
    return (intra * 1.0 + cross * 0.5) / total


def score_path(
    path,
    query_embedding: Optional[np.ndarray] = None,
    weight_attention: float = 0.4,
    weight_community: float = 0.3,
    weight_semantic: float  = 0.3,
) -> float:
    """
    Final path score combining attention, community coherence, and semantic alignment.

    score(P) = weight_attention * prod(attention_weights)
             + weight_community * community_coherence(P)
             + weight_semantic  * semantic_alignment(h_final, query_emb)

    When query_embedding is None, semantic weight is redistributed to the
    other two signals proportionally.

    Parameters
    ----------
    path             : TraversalPath object
    query_embedding  : optional query vector for semantic alignment check
    weight_*         : linear combination weights (should sum to 1.0)

    Returns
    -------
    float in [0, 1]
    """
    if not path.attention_weights:
        # Depth-0 seed path — neutral score
        return 0.5

    # Attention: product of weights along the path
    attention_score = math.prod(max(w, 1e-9) for w in path.attention_weights)

    # Community coherence
    coh = community_coherence(path.community_sequence)

    # Semantic alignment to query
    if query_embedding is not None and path.embedding is not None:
        qn = float(np.linalg.norm(query_embedding))
        pn = float(np.linalg.norm(path.embedding))
        if qn > 0 and pn > 0:
            raw_sim  = float(np.dot(query_embedding, path.embedding) / (qn * pn))
            semantic = (raw_sim + 1.0) / 2.0   # map [-1, 1] -> [0, 1]
        else:
            semantic = 0.5
        return (
            weight_attention * attention_score
            + weight_community * coh
            + weight_semantic  * semantic
        )
    else:
        # No query embedding — redistribute semantic weight
        total = weight_attention + weight_community
        if total == 0:
            return 0.0
        wa = weight_attention / total
        wc = weight_community / total
        return wa * attention_score + wc * coh

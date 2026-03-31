"""
Multi-signal path scoring (Section 5, STEP 4).

Combines:
  - Attention weight product (coverage of the reasoning chain)
  - Community coherence (domain consistency across hops)
  - Semantic alignment to the query (optional)
  - Relation path prior (optional) — frequency bonus for productive sequences
"""
import math
from typing import Any, List, Optional

import numpy as np


def path_confidence(path) -> float:
    """
    Minimum edge confidence along the path (weakest-link rule).

    A reasoning chain is only as confident as its least-certain edge.
    Paths with no confidence data (legacy graphs) return 1.0.

    Parameters
    ----------
    path : TraversalPath

    Returns
    -------
    float in [0, 1]
    """
    confs = getattr(path, "edge_confidences", [])
    if not confs:
        return 1.0
    return min(confs)


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
    relation_prior: Optional[Any] = None,
    weight_prior: float = 0.15,
) -> float:
    """
    Final path score combining attention, community coherence, semantic
    alignment, and an optional relation path frequency prior.

    score(P) = w_attn  * prod(attention_weights)
             + w_comm  * community_coherence(P)
             + w_sem   * semantic_alignment(h_final, query_emb)   [optional]
             + w_prior * relation_prior.score(P)                  [optional]

    When query_embedding is None, its weight is redistributed to the other
    active signals proportionally.  When relation_prior is None, its weight
    is similarly redistributed.

    Parameters
    ----------
    path             : TraversalPath object
    query_embedding  : optional query vector for semantic alignment check
    weight_*         : linear combination weights
    relation_prior   : optional RelationPathPrior or GraphRelationPrior
    weight_prior     : weight given to the relation prior term (default 0.15)

    Returns
    -------
    float in [0, 1]
    """
    if not path.attention_weights:
        return 0.5

    # Attention: product of weights along the path
    attention_score = math.prod(max(w, 1e-9) for w in path.attention_weights)

    # Community coherence
    coh = community_coherence(path.community_sequence)

    # Determine which optional signals are active
    has_semantic = query_embedding is not None and path.embedding is not None
    has_prior    = relation_prior is not None

    # Compute active signal scores
    w_attn = weight_attention
    w_comm = weight_community
    w_sem  = weight_semantic if has_semantic else 0.0
    w_pri  = weight_prior    if has_prior    else 0.0

    score = w_attn * attention_score + w_comm * coh

    if has_semantic:
        qn = float(np.linalg.norm(query_embedding))
        pn = float(np.linalg.norm(path.embedding))
        if qn > 0 and pn > 0:
            raw_sim  = float(np.dot(query_embedding, path.embedding) / (qn * pn))
            semantic = (raw_sim + 1.0) / 2.0
        else:
            semantic = 0.5
        score += w_sem * semantic

    if has_prior:
        prior_s = relation_prior.score_with_prefix(path) if hasattr(
            relation_prior, "score_with_prefix"
        ) else relation_prior.score(path)
        score += w_pri * prior_s

    # Normalise by total active weight
    total_w = w_attn + w_comm + w_sem + w_pri
    if total_w <= 0:
        return 0.0
    return float(np.clip(score / total_w, 0.0, 1.0))




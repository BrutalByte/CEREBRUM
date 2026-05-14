"""
Multi-signal path scoring (Section 5, STEP 4).

Combines:
  - Attention weight product (coverage of the reasoning chain)
  - Community coherence (domain consistency across hops)
  - Semantic alignment to the query (optional)
  - Relation path prior (optional) — frequency bonus for productive sequences
  - Path Specificity Score / PSS (Phase 179) — inverse relation fan-out penalty
    for hub-like traversals; favours narrow, specific reasoning chains
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


def grounding_score(path) -> float:
    """
    Score the evidential grounding of a path based on edge confidence and provenance.

    - Edge confidence: combined via product (probabilistic intersection).
    - Provenance: penalty applied if an edge lacks a specific source.

    Returns
    -------
    float in [0, 1]
    """
    confs = getattr(path, "edge_confidences", [])
    provs = getattr(path, "edge_provenances", [])
    if not confs:
        return 1.0

    # 1. Combined confidence
    score = 1.0
    for c in confs:
        score *= c

    # 2. Provenance penalty: -0.1 per edge with empty/generic provenance
    for p in provs:
        if not p or p in ["triples", "manual", "unknown"]:
            score *= 0.9  # 10% penalty for ungrounded hop

    return float(np.clip(score, 0.0, 1.0))


def path_specificity_score(path, fan_out: "Dict[str, Dict[str, int]]") -> float:
    """
    Phase 179 — Path Specificity Score (PSS).

    For each hop (entity u) → (relation r) → (entity v), compute the inverse
    fan-out of relation r from u: 1 / |{targets of r from u}|.

    Paths that traverse narrow, specific edges score near 1.0.
    Paths that pass through hub entities with hundreds of r-neighbours score
    near 0.0 — these are likely generic hub-traversals, not the intended chain.

    PSS = geometric mean of per-hop inverse fan-outs, clamped to [0, 1].

    Parameters
    ----------
    path     : TraversalPath — nodes alternate entity/relation/entity/...
    fan_out  : {entity_id: {relation: edge_count}} precomputed from KB

    Returns
    -------
    float in (0, 1]  (never 0: unknown fan-out treated as 1)
    """
    nodes = getattr(path, "nodes", [])
    # Need at least one hop: [entity, relation, entity]
    if len(nodes) < 3:
        return 1.0

    log_sum = 0.0
    n_hops = 0
    for i in range(1, len(nodes) - 1, 2):
        entity = nodes[i - 1]
        relation = nodes[i]
        count = fan_out.get(entity, {}).get(relation, 1)
        log_sum += math.log(1.0 / max(count, 1))
        n_hops += 1

    if n_hops == 0:
        return 1.0
    return float(math.exp(log_sum / n_hops))


def score_path(
    path,
    query_embedding: Optional[np.ndarray] = None,
    weight_attention: float = 0.35,
    weight_community: float = 0.25,
    weight_semantic: float  = 0.2,
    weight_grounding: float = 0.2,
    relation_prior: Optional[Any] = None,
    weight_prior: float = 0.15,
) -> float:
    """
    Final path score combining attention, community coherence, semantic
    alignment, grounding (Phase 35), and an optional relation path frequency prior.

    score(P) = w_attn  * attention_score
             + w_comm  * community_coherence(P)
             + w_sem   * semantic_alignment(h_final, query_emb)
             + w_grnd  * grounding_score(P)
             + w_prior * relation_prior.score(P)

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

    # Attention: geometric mean of weights along the path.
    # Using geometric mean (= prod^(1/n)) instead of raw product ensures paths
    # of different hop depths are scored on the same scale.  A raw product
    # systematically penalises deeper paths (0.7^3 = 0.343 vs 0.7^1 = 0.7),
    # causing short wrong-answer paths to rank above long correct-answer paths.
    n = len(path.attention_weights)
    log_sum = sum(math.log(max(w, 1e-9)) for w in path.attention_weights)
    attention_score = math.exp(log_sum / n)

    # Community coherence
    coh = community_coherence(path.community_sequence)

    # Determine which optional signals are active
    has_semantic = query_embedding is not None and path.embedding is not None
    has_prior    = relation_prior is not None

    # Compute active signal scores
    w_attn = weight_attention
    w_comm = weight_community
    w_sem  = weight_semantic  if has_semantic else 0.0
    w_grnd = weight_grounding
    w_pri  = weight_prior     if has_prior    else 0.0

    score = w_attn * attention_score + w_comm * coh + w_grnd * grounding_score(path)

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
    total_w = w_attn + w_comm + w_sem + w_grnd + w_pri
    if total_w <= 0:
        return 0.0
    return float(np.clip(score / total_w, 0.0, 1.0))




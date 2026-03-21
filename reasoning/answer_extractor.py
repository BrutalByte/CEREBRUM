"""
Extract and rank top-K answers from a set of traversal paths (Section 5, STEP 5).

An "answer" is the terminal entity of a path. Paths are re-scored using
score_path() and deduplicated by terminal entity (best score wins).
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np

from reasoning.path_scorer import score_path, community_coherence


@dataclass
class Answer:
    """A single ranked answer extracted from a traversal."""

    entity_id: str
    """Terminal entity (the answer entity)."""

    score: float
    """Final ranked score."""

    best_path: Any
    """The TraversalPath that produced this answer (highest score)."""

    score_breakdown: Dict[str, float] = field(default_factory=dict)
    """Decomposition: {attention, community, semantic}."""

    community_trace: List[int] = field(default_factory=list)
    """Community ID sequence for the best path — shows conceptual trajectory."""

    def __repr__(self) -> str:
        return f"Answer(entity={self.entity_id!r}, score={self.score:.4f})"


def extract(
    paths: List[Any],
    top_k: int = 5,
    query_embedding: Optional[np.ndarray] = None,
    min_hop: int = 1,
    weight_attention: float = 0.4,
    weight_community: float = 0.3,
    weight_semantic: float  = 0.3,
) -> List[Answer]:
    """
    Extract the top-K answer entities from a list of TraversalPaths.

    Steps:
      1. Filter out depth-0 seed paths (no hops taken)
      2. Re-score all paths using score_path()
      3. Deduplicate by terminal entity (keep highest score per entity)
      4. Return top-K sorted by score descending

    Parameters
    ----------
    paths          : list of TraversalPath objects from BeamTraversal.traverse()
    top_k          : number of answers to return
    query_embedding: optional query vector for semantic alignment re-scoring
    min_hop        : minimum hop depth to consider (default 1 — excludes seeds)

    Returns
    -------
    List of Answer objects, sorted by score descending.
    """
    # Filter to paths with at least min_hop hops
    valid = [p for p in paths if p.hop_depth >= min_hop]
    if not valid:
        return []

    # Score all valid paths
    scored = [
        (
            p,
            score_path(
                p,
                query_embedding=query_embedding,
                weight_attention=weight_attention,
                weight_community=weight_community,
                weight_semantic=weight_semantic,
            ),
        )
        for p in valid
    ]

    # Deduplicate: keep best-scoring path per terminal entity
    best: Dict[str, tuple] = {}
    for path, s in scored:
        entity = path.tail
        if entity not in best or s > best[entity][1]:
            best[entity] = (path, s)

    # Sort and return top-K
    ranked = sorted(best.values(), key=lambda t: t[1], reverse=True)[:top_k]

    answers = []
    for path, s in ranked:
        # Build score breakdown
        import math
        attn_score = math.prod(max(w, 1e-9) for w in path.attention_weights) if path.attention_weights else 0.0
        coh        = community_coherence(path.community_sequence)

        if query_embedding is not None and path.embedding is not None:
            qn = float(np.linalg.norm(query_embedding))
            pn = float(np.linalg.norm(path.embedding))
            if qn > 0 and pn > 0:
                sem = (float(np.dot(query_embedding, path.embedding) / (qn * pn)) + 1.0) / 2.0
            else:
                sem = 0.5
        else:
            sem = None

        breakdown = {
            "attention":  round(attn_score, 4),
            "community":  round(coh, 4),
        }
        if sem is not None:
            breakdown["semantic"] = round(sem, 4)

        answers.append(
            Answer(
                entity_id=path.tail,
                score=round(s, 6),
                best_path=path,
                score_breakdown=breakdown,
                community_trace=path.community_sequence,
            )
        )

    return answers




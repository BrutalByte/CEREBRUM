"""
Community-Structured Attention (CSA) engine.

Implements the attention weight formula from Section 4.1:

  a(u, v, k) = sigmoid(
      alpha   * cosine_sim(emb(u), emb(v))
    + beta    * community_score(u, v)
    + gamma   * edge_type_weight(type(u->v))
    - delta   * normalized_distance(u, v)
    + epsilon * hop_decay(k)
  )

Default parameters (zero-shot deployment):
  alpha=0.4, beta=0.4, gamma=0.1, delta=0.05, epsilon=0.05

Precompute community_distances and adjacent_pairs once after DSCF converges
using build_community_distance_matrix() and adjacent_community_pairs() from
structural_encoder. Then call set_community_graph() before traversal.
"""
import math
from typing import Dict, Optional, Set, Tuple

import numpy as np


class CSAEngine:
    """
    Community-Structured Attention weight calculator.

    Usage:
        # After running DSCF and building community graph metadata
        engine = CSAEngine(communities=community_map, embeddings=entity_embeddings)
        engine.set_community_graph(distances, adjacent_pairs)

        # During beam traversal
        w = engine.compute_weight(u, v, hop=k, edge_type="INFLUENCED")
    """

    def __init__(
        self,
        communities: Dict[str, int],
        embeddings: Dict[str, np.ndarray],
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.1,
        delta: float = 0.05,
        epsilon: float = 0.05,
        lambda_decay: float = 0.5,
        edge_type_weights: Optional[Dict[str, float]] = None,
        external_community_scores: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        """
        Parameters
        ----------
        communities       : {node_id -> community_id} mapping from DSCF
        embeddings        : {node_id -> float32 vector} from EmbeddingEngine
        alpha             : weight for cosine embedding similarity
        beta              : weight for community membership score
        gamma             : weight for edge type relevance
        delta             : penalty for normalized graph distance
        epsilon           : weight for hop-depth decay
        lambda_decay      : decay rate for cross-community distance (community_score)
        edge_type_weights : {relation_type -> weight} mapping for "Bridge Bonus"
        external_community_scores : { (cid_u, cid_v) -> score } for cross-graph links
        """
        self.communities       = communities
        self.embeddings        = embeddings
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = gamma
        self.delta             = delta
        self.epsilon           = epsilon
        self.lambda_decay      = lambda_decay
        self.edge_type_weights = edge_type_weights or {}
        self.external_community_scores = external_community_scores or {}

        # Populated by set_community_graph()
        self._community_distances: Dict[Tuple[int, int], float] = {}
        self._adjacent_pairs: Set[Tuple[int, int]]              = set()

    def set_community_graph(
        self,
        community_distances: Dict[Tuple[int, int], float],
        adjacent_pairs: Set[Tuple[int, int]],
    ) -> None:
        """
        Load precomputed community-level graph metadata.

        Call once after DSCF converges, using:
            from core.structural_encoder import (
                build_community_distance_matrix, adjacent_community_pairs
            )
            distances = build_community_distance_matrix(G, community_map)
            adj       = adjacent_community_pairs(G, community_map)
            engine.set_community_graph(distances, adj)
        """
        self._community_distances = community_distances
        self._adjacent_pairs      = adjacent_pairs

    # ------------------------------------------------------------------
    # community_score
    # ------------------------------------------------------------------

    def community_score(self, u: str, v: str) -> float:
        """
        Structural community membership score for the edge u -> v.

          same community   : 1.0
          adjacent comms   : 0.5
          distant comms    : exp(-lambda * hop_distance)
          external/unknown : use external_community_scores or 0.5 fallback
        """
        cu = self.communities.get(u)
        cv = self.communities.get(v)

        # Handle federated/external IDs
        if cu is None or cv is None:
            # Try to find a cross-graph community score
            # We use 0 as a default ID for 'unknown/remote' if specific IDs aren't provided
            # in external_community_scores.
            score = self.external_community_scores.get((cu, cv))
            if score is not None:
                return score
            return 0.5

        if cu == cv:
            return 1.0

        if (cu, cv) in self._adjacent_pairs or (cv, cu) in self._adjacent_pairs:
            return 0.5

        d = self._community_distances.get(
            (cu, cv),
            self._community_distances.get((cv, cu), 5.0),
        )
        return math.exp(-self.lambda_decay * d)

    # ------------------------------------------------------------------
    # compute_weight — the full CSA formula
    # ------------------------------------------------------------------

    def compute_weight(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
    ) -> float:
        """
        Compute the CSA attention weight for the directed edge u -> v
        at traversal hop k.

        Parameters
        ----------
        u, v                 : entity IDs
        hop                  : current BFS hop depth (1-indexed)
        edge_type            : relation type label (e.g. "INFLUENCED")
        edge_type_weights    : optional {relation_type -> weight} override
                               (takes precedence over self.edge_type_weights)
        normalized_distance  : pre-normalized shortest-path distance (0..1)
                               Pass 0 when not available (no penalty applied).

        Returns
        -------
        float in (0, 1) — sigmoid-transformed attention weight
        """
        # 1. Embedding cosine similarity
        eu = self.embeddings.get(u)
        ev = self.embeddings.get(v)
        if eu is not None and ev is not None:
            sim = _cosine_sim(eu, ev)
        else:
            sim = 0.0

        # 2. Community membership score
        cs = self.community_score(u, v)

        # 3. Edge type weight (domain-specific; 0.0 = neutral)
        weights = edge_type_weights if edge_type_weights is not None else self.edge_type_weights
        if edge_type in weights:
            etw = weights[edge_type]
        else:
            etw = 0.0

        # 4. Hop decay: 1 / (1 + k)
        hop_decay = 1.0 / (1.0 + hop)

        # 5. Assemble and sigmoid
        raw = (
            self.alpha   * sim
            + self.beta  * cs
            + self.gamma * etw
            - self.delta * normalized_distance
            + self.epsilon * hop_decay
        )
        return _sigmoid(raw)


# ------------------------------------------------------------------
# Pure-function helpers
# ------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class UniformCSAEngine(CSAEngine):
    """
    CSA engine that returns a constant weight for every edge.

    This makes BeamTraversal equivalent to BFS — no preference is given
    to any edge over any other. Used as the 'no-attention' ablation baseline.

    Weight = sigmoid(0) = 0.5 for all edges (all coefficients zeroed out).
    """

    def compute_weight(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
    ) -> float:
        return 0.5   # sigmoid(0) — perfectly neutral




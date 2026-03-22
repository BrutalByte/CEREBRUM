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
        engine = CSAEngine(adapter=adapter)
        engine.set_community_graph(distances, adjacent_pairs)

        # During beam traversal
        w = engine.compute_weight(u, v, hop=k, edge_type="INFLUENCED")
    """

    def __init__(
        self,
        adapter: GraphAdapter,
        communities: Optional[Dict[str, int]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.1,
        delta: float = 0.05,
        epsilon: float = 0.05,
        zeta: float = 0.1,
        lambda_decay: float = 0.5,
        edge_type_weights: Optional[Dict[str, float]] = None,
        external_community_scores: Optional[Dict[Tuple[int, int], float]] = None,
        pagerank: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        adapter           : GraphAdapter providing get_community() and get_embedding()
        communities       : deprecated — use adapter instead
        embeddings        : deprecated — use adapter instead
        alpha             : weight for cosine embedding similarity
        beta              : weight for community membership score
        gamma             : weight for edge type relevance
        delta             : penalty for normalized graph distance
        epsilon           : weight for hop-depth decay
        zeta              : weight for global PageRank prior (global authority signal)
        lambda_decay      : decay rate for cross-community distance (community_score)
        edge_type_weights : {relation_type -> weight} mapping for "Bridge Bonus"
        external_community_scores : { (cid_u, cid_v) -> score } for cross-graph links
        pagerank          : precomputed {node -> pagerank_score} dict; enables zeta term
        """
        self.adapter           = adapter
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = gamma
        self.delta             = delta
        self.epsilon           = epsilon
        self.zeta              = zeta
        self.lambda_decay      = lambda_decay
        self.edge_type_weights = edge_type_weights or {}
        self.external_community_scores = external_community_scores or {}

        # PageRank prior: normalize by max so the term is always in [0, 1]
        self._pagerank: Dict[str, float] = pagerank or {}
        self._max_pr: float = max(self._pagerank.values()) if self._pagerank else 1.0

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
        """
        self._community_distances = community_distances
        self._adjacent_pairs      = adjacent_pairs

    def community_score(self, u: str, v: str) -> float:
        """
        Structural community membership score for the edge u -> v.
        
        Handles:
          - Same community: 1.0
          - Adjacent communities: 0.5
          - Distant communities: exp(-lambda * d)
          - Federated/Remote communities: uses external_community_scores map
        """
        cu = self.adapter.get_community(u)
        cv = self.adapter.get_community(v)

        # Handle federated/external IDs via the external_community_scores map
        # Check specific pair first, then wildcards (-1)
        if (cu, cv) in self.external_community_scores:
            return self.external_community_scores[(cu, cv)]
        
        # Wildcard: score for any link from cu to "somewhere remote"
        if (cu, -1) in self.external_community_scores:
            return self.external_community_scores[(cu, -1)]
            
        # Wildcard: score for any link from "somewhere remote" to cv
        if (-1, cv) in self.external_community_scores:
            return self.external_community_scores[(-1, cv)]

        # Global fallback for any unknown-to-unknown or unknown-to-known link
        if cu == -1 or cv == -1:
            return self.external_community_scores.get((-1, -1), 0.5)

        if cu == cv:
            return 1.0

        if (cu, cv) in self._adjacent_pairs or (cv, cu) in self._adjacent_pairs:
            return 0.5

        d = self._community_distances.get(
            (cu, cv),
            self._community_distances.get((cv, cu), 5.0),
        )
        return math.exp(-self.lambda_decay * d)

    def compute_weight(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
    ) -> float:
        # Bridge twin shortcut: crossing to your structural relay is free.
        # The twin has the same embedding (sim=1.0) and is explicitly placed
        # in the destination community (cs=1.0), so the full formula would
        # already score this highly — but we short-circuit for clarity and
        # to avoid any floating-point drift from separate lookups.
        if edge_type == "BRIDGE_TWIN":
            hop_decay = 1.0 / (1.0 + hop)
            raw = self.alpha * 1.0 + self.beta * 1.0 + self.gamma * 1.0 + self.epsilon * hop_decay
            return _sigmoid(raw)

        # 1. Embedding cosine similarity
        eu = self.adapter.get_embedding(u)
        ev = self.adapter.get_embedding(v)
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

        # 5. Global PageRank prior: normalized authority score for destination node.
        # Gives the beam a gravity signal toward structurally important nodes,
        # closing the recall gap vs. PPR at deep hops without running random walks.
        pr_v = self._pagerank.get(v, 0.0) / self._max_pr if self._pagerank else 0.0

        # 6. Assemble and sigmoid
        raw = (
            self.alpha   * sim
            + self.beta  * cs
            + self.gamma * etw
            - self.delta * normalized_distance
            + self.epsilon * hop_decay
            + self.zeta  * pr_v
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




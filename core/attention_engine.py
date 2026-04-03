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
from core.graph_adapter import GraphAdapter
from core.reasoning_logit import ReasoningLogit # New: Unified logit framework

# Return type for compute_weight_with_features:
# (weight, sim, cs, etw, normalized_distance, hop_decay, pr_v, temporal_decay)
WeightFeatures = Tuple[float, float, float, float, float, float, float, float]

# Default exponential decay rates (λ) for temporal reasoning (SPEC_015).
# Weight = weight * exp(-λ * elapsed_time)
RELATION_DECAY_DEFAULTS = {
    "CURRENT_PRICE": 0.693,  # Half-life: 1 second
    "REPORTED_AS":   0.010,  # Half-life: 69 seconds
    "AFFILIATED_WITH": 0.001, # Half-life: ~11.5 minutes
    "BORN_IN":       0.0,    # Never decays
}

# Type alias for soft membership: {node_id: {community_id: probability}}
SoftMemberships = Dict[str, Dict[int, float]]


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
        node_recency: Optional[Dict[str, float]] = None,
        soft_memberships: Optional["SoftMemberships"] = None,
        community_params: Optional[Dict[int, Tuple[float, float, float, float, float]]] = None,
        use_temporal_decay: bool = False,
        eta: float = 0.1,
        iota: float = 0.05,
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
        node_recency      : precomputed {node -> recency_score} dict; enables iota term
        soft_memberships  : output of compute_soft_memberships() — when provided,
                            community_score() uses dot-product of membership vectors
                            instead of hard same/adjacent/distant classification.
                            Enables multi-domain entities to score well across all
                            communities they partially belong to.
        use_temporal_decay: If True, apply exponential decay based on edge validity windows.
        eta               : weight (multiplier) for the temporal decay component (edge).
        iota              : weight (multiplier) for the node recency component.
        """
        self.adapter           = adapter
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = gamma
        self.delta             = delta
        self.epsilon           = epsilon
        self.zeta              = zeta
        self.eta               = eta
        self.iota              = iota
        self.use_temporal_decay = use_temporal_decay
        self.lambda_decay      = lambda_decay
        self.edge_type_weights = edge_type_weights or {}
        self.external_community_scores = external_community_scores or {}

        # Hole 2 — Homogeneity Trap: per-community CSA parameter overrides.
        # {community_id: (alpha, beta, gamma, delta, epsilon)}
        # When set, compute_weight() uses the source node's community params
        # instead of the global defaults, allowing domain-specific attention tuning.
        self._community_params: Dict[int, Tuple[float, float, float, float, float]] = (
            community_params or {}
        )

        # Soft community memberships (optional): {node -> {community_id -> probability}}
        self.soft_memberships: Optional["SoftMemberships"] = soft_memberships

        # PageRank prior: normalize by max so the term is always in [0, 1]
        self._pagerank: Dict[str, float] = pagerank or {}
        self._max_pr: float = max(self._pagerank.values()) if self._pagerank else 1.0

        # Node recency: pre-normalized [0, 1]
        self._node_recency: Dict[str, float] = node_recency or {}

        # Populated by set_community_graph()
        self._community_distances: Dict[Tuple[int, int], float] = {}
        self._adjacent_pairs: Set[Tuple[int, int]]              = set()
        self._community_graph = None   # optional nx.Graph for lazy distance lookup

        # Milestone 4: Adaptive Parameter Learning
        self.meta_learner = None

        # Hole 1 — Mid-Flight Community Swap: per-query community map snapshot.
        # Set by BeamTraversal.traverse() at query start; cleared on completion.
        # When set, community lookups use this frozen snapshot instead of the
        # live adapter.community_map, preventing CSA inconsistency mid-query.
        self._query_snapshot: Optional[Dict[str, int]] = None

        # Per-query temporal filtering
        self._query_time: Optional[float] = None

        # Per-query community_score cache: reset each query via set/clear snapshot.
        # Eliminates duplicate soft-membership dot-product recomputation when the
        # same (u, v) pair is evaluated more than once (double-call from traversal
        # + compute_weight, and repeated frontier convergence at deep hops).
        self._cs_cache: Dict[Tuple[str, str], float] = {}

    def set_meta_learner(self, learner) -> None:
        """Attach a MetaParameterLearner for community-specific tuning."""
        self.meta_learner = learner

    def set_pagerank(self, pagerank: Dict[str, float]) -> None:
        """Update the PageRank prior map (used for the zeta term)."""
        self._pagerank = pagerank
        self._max_pr = max(pagerank.values()) if pagerank else 1.0

    def set_query_time(self, query_time: Optional[float]) -> None:
        """Set the reference time for temporal reasoning in the current query."""
        self._query_time = query_time

    # ------------------------------------------------------------------
    # Hole 1 — Mid-Flight Community Swap: Query Snapshot Isolation
    # ------------------------------------------------------------------

    def set_query_snapshot(self, community_map: Dict[str, int]) -> None:
        """
        Freeze a copy of community_map for the duration of a single query.

        Called by BeamTraversal.traverse() / traverse_stream() at query start.
        All community lookups within this query will use the snapshot,
        preventing in-flight CSA inconsistency if GlobalRebalancer commits a
        new partition mid-query.
        """
        self._query_snapshot = community_map
        self._cs_cache = {}  # reset per-query community_score cache

    def clear_query_snapshot(self) -> None:
        """Release the per-query snapshot and community_score cache."""
        self._query_snapshot = None
        self._cs_cache = {}

    def _get_community(self, node_id: str) -> int:
        """Community lookup that respects the per-query snapshot when set."""
        if self._query_snapshot is not None:
            return self._query_snapshot.get(node_id, -1)
        return self.adapter.get_community(node_id)

    def set_community_graph(
        self,
        community_distances: Dict[Tuple[int, int], float],
        adjacent_pairs: Set[Tuple[int, int]],
        community_graph=None,
    ) -> None:
        """
        Load precomputed community-level graph metadata.

        Parameters
        ----------
        community_graph : optional nx.Graph
            Community-level graph (one node per community, edges where
            cross-community edges exist).  When provided, community distances
            are computed lazily on first access and cached, so
            build_community_distance_matrix() need not be called upfront.
        """
        self._community_distances = community_distances
        self._adjacent_pairs      = adjacent_pairs
        self._community_graph     = community_graph

    def community_score(self, u: str, v: str) -> float:
        """
        Structural community membership score for the edge u -> v.

        Handles:
          - Soft memberships (when set): dot-product of membership vectors
          - Same community: 1.0
          - Adjacent communities: 0.5
          - Distant communities: exp(-lambda * d)
          - Federated/Remote communities: uses external_community_scores map

        Results are memoized per query in _cs_cache (reset by set/clear_query_snapshot).
        This eliminates redundant recomputation when the same (u, v) pair is
        evaluated multiple times within a single beam traversal query.
        """
        cached = self._cs_cache.get((u, v))
        if cached is not None:
            return cached
        score = self._community_score_uncached(u, v)
        self._cs_cache[(u, v)] = score
        return score

    def _community_score_uncached(self, u: str, v: str) -> float:
        """Inner community score computation (no cache)."""
        # Soft membership path: dot-product over shared communities
        if self.soft_memberships is not None:
            mem_u = self.soft_memberships.get(u, {})
            mem_v = self.soft_memberships.get(v, {})
            score = sum(mem_u.get(cid, 0.0) * prob_v
                        for cid, prob_v in mem_v.items())
            return min(score, 1.0)

        cu = self._get_community(u)
        cv = self._get_community(v)

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

        d = self._community_distances.get((cu, cv),
            self._community_distances.get((cv, cu), None))
        if d is None:
            if self._community_graph is not None:
                try:
                    import networkx as _nx
                    d = float(_nx.shortest_path_length(
                        self._community_graph, cu, cv))
                except Exception:
                    d = 5.0
                # cache both directions so each pair is only computed once
                self._community_distances[(cu, cv)] = d
                self._community_distances[(cv, cu)] = d
            else:
                d = 5.0
        return math.exp(-self.lambda_decay * d)

    def compute_weight(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
        valid_from: Optional[float] = None, # New parameter
        valid_to: Optional[float] = None,   # New parameter
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

        # Hole 2 — Homogeneity Trap: resolve per-community parameter overrides.
        # If the source node's community has a registered parameter vector, use it
        # instead of the global defaults, enabling domain-specific attention tuning
        # (e.g., high gamma for dense causal communities, high delta for temporal).
        cu = self._get_community(u)
        
        if self.meta_learner is not None:
            alpha, beta, gamma, delta, epsilon = self.meta_learner.get_params(cu)
        elif self._community_params and cu in self._community_params:
            alpha, beta, gamma, delta, epsilon = self._community_params[cu]
        else:
            alpha, beta, gamma, delta, epsilon = (
                self.alpha, self.beta, self.gamma, self.delta, self.epsilon
            )

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

        # 5b. Node recency prior: normalized recency score for destination node [0, 1].
        # Gives the beam a bias toward recently active nodes.
        nr_v = self._node_recency.get(v, 0.0)

        # 6. Temporal Decay (Phase 33)
        temporal_decay_term = 0.0
        if self.use_temporal_decay and self._query_time is not None and valid_to is not None:
            # Calculate elapsed time since edge was valid
            time_elapsed = self._query_time - valid_to
            if time_elapsed > 0:
                # Use edge-type specific decay rate if available, otherwise a default
                decay_rate = RELATION_DECAY_DEFAULTS.get(edge_type, self.lambda_decay)
                # Recency bias: newer edges (small time_elapsed) get higher score
                temporal_decay_term = self.eta * math.exp(-decay_rate * time_elapsed)
        
        # 7. Assemble and sigmoid (using resolved alpha/beta/gamma/delta/epsilon)
        raw = (
            alpha   * sim
            + beta  * cs
            + gamma * etw
            - delta * normalized_distance
            + epsilon * hop_decay
            + self.zeta  * pr_v
            + self.iota  * nr_v
            + temporal_decay_term # Add the temporal decay term
        )
        return _sigmoid(raw)

    def compute_weight_with_features(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        eu: Optional[np.ndarray] = None,
        ev: Optional[np.ndarray] = None,
    ) -> ReasoningLogit:
        """
        Compute CSA attention weight and all feature components in a single pass.
        Returns a ReasoningLogit object.
        """
        if edge_type == "BRIDGE_TWIN":
            hd = 1.0 / (1.0 + hop)
            return ReasoningLogit(sim=1.0, cs=1.0, etw=1.0, nd=normalized_distance, hd=hd)

        cu = self._get_community(u)

        if self.meta_learner is not None:
            a, b, g, d, e = self.meta_learner.get_params(cu)
        elif self._community_params and cu in self._community_params:
            a, b, g, d, e = self._community_params[cu]
        else:
            a, b, g, d, e = (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)

        if eu is None: eu = self.adapter.get_embedding(u)
        if ev is None: ev = self.adapter.get_embedding(v)
        sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0

        cs = self.community_score(u, v)

        weights = edge_type_weights if edge_type_weights is not None else self.edge_type_weights
        etw = weights.get(edge_type, 0.0)

        hd = 1.0 / (1.0 + hop)
        pr_v = self._pagerank.get(v, 0.0) / self._max_pr if self._pagerank else 0.0
        nr_v = self._node_recency.get(v, 0.0)

        td = 0.0
        if self.use_temporal_decay and self._query_time is not None and valid_to is not None:
            time_elapsed = self._query_time - valid_to
            if time_elapsed > 0:
                decay_rate = RELATION_DECAY_DEFAULTS.get(edge_type, self.lambda_decay)
                # Raw feature: exp(-λ*t) in [0, 1]. Recency bias = +eta * td
                td = math.exp(-decay_rate * time_elapsed)

        return ReasoningLogit(sim, cs, etw, normalized_distance, hd, pr_v, td, nr_v, 1.0)

    def get_current_params(self, u: Optional[str] = None) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """
        Return the 9-element parameter vector for the current query/community context.
        Order: (alpha, beta, gamma, delta, epsilon, zeta, eta, iota, theta)
        """
        cu = self._get_community(u) if u is not None else -1
        
        if self.meta_learner is not None and cu >= 0:
            a, b, g, d, e = self.meta_learner.get_params(cu)
        elif self._community_params and cu in self._community_params:
            a, b, g, d, e = self._community_params[cu]
        else:
            a, b, g, d, e = (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)
            
        return (a, b, g, d, e, self.zeta, self.eta, self.iota, 1.0) # theta=1.0 by default



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

    def compute_weight_with_features(
        self,
        u: str,
        v: str,
        hop: int,
        edge_type: str = "",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        eu: Optional[np.ndarray] = None,
        ev: Optional[np.ndarray] = None,
    ) -> ReasoningLogit:
        hd = 1.0 / (1.0 + hop)
        return ReasoningLogit(cs=0.5, nd=normalized_distance, hd=hd, grounding=1.0)





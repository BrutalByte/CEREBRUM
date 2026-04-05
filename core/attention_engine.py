"""
Community-Structured Attention (CSA) engine.

Implements the attention weight formula from Section 4.1, updated for Phase 43:

  a(u, v, k) = sigmoid(
      alpha   * sim
    + beta    * cs
    + gamma   * etw
    - delta   * nd
    + epsilon * hd
    + zeta    * pr_v
    + eta     * td
    + iota    * nr_v
    - mu      * sd
    + theta   * grounding
  )
"""
import math
from typing import Dict, Optional, Set, Tuple

import numpy as np
from core.graph_adapter import GraphAdapter
from core.reasoning_logit import ReasoningLogit

# Default exponential decay rates (λ) for temporal reasoning.
RELATION_DECAY_DEFAULTS = {
    "CURRENT_PRICE": 0.693,
    "REPORTED_AS":   0.010,
    "AFFILIATED_WITH": 0.001,
    "BORN_IN":       0.0,
}

SoftMemberships = Dict[str, Dict[int, float]]

class CSAEngine:
    def __init__(
        self,
        adapter: GraphAdapter,
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
        temporal_window_size: Optional[float] = None,
        eta: float = 0.1,
        iota: float = 0.05,
        mu: float = 0.1,
        theta: float = 1.0,
        **kwargs # Forward compatibility
    ):
        self.adapter = adapter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.iota = iota
        self.mu = mu
        self.theta = theta
        self.use_temporal_decay = use_temporal_decay
        self.temporal_window_size = temporal_window_size
        self.lambda_decay = lambda_decay
        self.edge_type_weights = edge_type_weights or {}
        self.external_community_scores = external_community_scores or {}
        self._community_params = community_params or {}
        self.soft_memberships = soft_memberships
        self._pagerank = pagerank or {}
        self._max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
        self._node_recency = node_recency or {}
        self._community_distances: Dict[Tuple[int, int], float] = {}
        self._adjacent_pairs: Set[Tuple[int, int]] = set()
        self._community_graph = None
        self.meta_learner = None
        self._query_snapshot: Optional[Dict[str, int]] = None
        self._query_time: Optional[float] = None
        self._cs_cache: Dict[Tuple[str, str], float] = {}

    def set_meta_learner(self, learner) -> None: self.meta_learner = learner
    def set_pagerank(self, pagerank: Dict[str, float]) -> None:
        self._pagerank = pagerank
        self._max_pr = max(pagerank.values()) if pagerank else 1.0
    def set_query_time(self, query_time: Optional[float]) -> None: self._query_time = query_time
    def set_query_snapshot(self, community_map: Dict[str, int]) -> None:
        self._query_snapshot = community_map
        self._cs_cache = {}
    def clear_query_snapshot(self) -> None:
        self._query_snapshot = None
        self._cs_cache = {}

    def _get_community(self, node_id: str) -> int:
        if self._query_snapshot is not None: return self._query_snapshot.get(node_id, -1)
        return self.adapter.get_community(node_id)

    def set_community_graph(self, *args, **kwargs) -> None:
        """
        Accepts:
          (dists, pairs, graph=None)
          (community_distances=dists, adjacent_pairs=pairs, community_graph=graph)
        """
        if args:
            self._community_distances = args[0]
            if len(args) > 1: self._adjacent_pairs = args[1]
            if len(args) > 2: self._community_graph = args[2]
        
        if 'community_distances' in kwargs: self._community_distances = kwargs['community_distances']
        if 'dists' in kwargs: self._community_distances = kwargs['dists']
        if 'adjacent_pairs' in kwargs: self._adjacent_pairs = kwargs['adjacent_pairs']
        if 'pairs' in kwargs: self._adjacent_pairs = kwargs['pairs']
        if 'community_graph' in kwargs: self._community_graph = kwargs['community_graph']
        if 'graph' in kwargs: self._community_graph = kwargs['graph']

    def community_score(self, u: str, v: str) -> float:
        cached = self._cs_cache.get((u, v))
        if cached is not None: return cached
        
        if self.soft_memberships is not None:
            mem_u = self.soft_memberships.get(u, {})
            mem_v = self.soft_memberships.get(v, {})
            score = sum(mem_u.get(cid, 0.0) * prob_v for cid, prob_v in mem_v.items())
            score = min(score, 1.0)
        else:
            cu, cv = self._get_community(u), self._get_community(v)
            if (cu, cv) in self.external_community_scores: score = self.external_community_scores[(cu, cv)]
            elif (cu, -1) in self.external_community_scores: score = self.external_community_scores[(cu, -1)]
            elif (-1, cv) in self.external_community_scores: score = self.external_community_scores[(-1, cv)]
            elif cu == -1 or cv == -1: score = self.external_community_scores.get((-1, -1), 0.5)
            elif cu == cv: score = 1.0
            elif (cu, cv) in self._adjacent_pairs or (cv, cu) in self._adjacent_pairs: score = 0.5
            else:
                d = self._community_distances.get((cu, cv), self._community_distances.get((cv, cu), 5.0))
                score = math.exp(-self.lambda_decay * d)
        
        self._cs_cache[(u, v)] = score
        return score

    def compute_weight(self, u, v, hop, edge_type="", normalized_distance=0.0, valid_from=None, valid_to=None, **kwargs) -> float:
        # Use kwargs to allow dynamic parameter overrides (like alpha, beta, gamma etc)
        params = self.get_current_params(u)
        
        # If we have dynamic overrides in kwargs, use them to build a temporary param tuple
        keys = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'iota', 'mu', 'theta']
        if any(k in kwargs for k in keys):
            p_list = list(params)
            for i, key in enumerate(keys):
                if key in kwargs: p_list[i] = kwargs[key]
            params = tuple(p_list)

        logit = self.compute_weight_with_features(
            u, v, hop, edge_type, 
            edge_type_weights=kwargs.get('edge_type_weights'),
            normalized_distance=normalized_distance, 
            valid_from=valid_from, 
            valid_to=valid_to
        )
        return logit.score(params)

    def compute_weight_with_features(self, u, v, hop, edge_type="", edge_type_weights=None, normalized_distance=0.0, valid_from=None, valid_to=None, eu=None, ev=None) -> ReasoningLogit:
        if edge_type == "BRIDGE_TWIN":
            # Force high score for structural relay
            return ReasoningLogit(sim=1.0, cs=1.0, etw=1.0, hd=10.0, grounding=1.0)

        if eu is None: eu = self.adapter.get_embedding(u)
        if ev is None: ev = self.adapter.get_embedding(v)
        sim = _cosine_sim(eu, ev) if (eu is not None and ev is not None) else 0.0
        cs = self.community_score(u, v)
        
        # Priority: explicit per-call weights > global instance weights
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
                td = math.exp(-decay_rate * time_elapsed)
                if self.temporal_window_size and time_elapsed > self.temporal_window_size:
                    td *= 0.1 # Window penalty

        sd = 1.0 if "rem_synthesized" in edge_type else 0.0
        grounding = 1.0 # Default

        return ReasoningLogit(sim, cs, etw, normalized_distance, hd, pr_v, td, nr_v, sd, grounding)

    def get_current_params(self, u: Optional[str] = None) -> Tuple[float, ...]:
        cu = self._get_community(u) if u is not None else -1
        if self.meta_learner is not None and cu >= 0:
            p = self.meta_learner.get_params(cu)
            # MetaParameterLearner now returns 10 params; fall back to engine
            # values for any params beyond what the learner manages.
            a   = p[0] if len(p) > 0 else self.alpha
            b   = p[1] if len(p) > 1 else self.beta
            g   = p[2] if len(p) > 2 else self.gamma
            d   = p[3] if len(p) > 3 else self.delta
            e   = p[4] if len(p) > 4 else self.epsilon
            zeta    = p[5] if len(p) > 5 else self.zeta
            eta     = p[6] if len(p) > 6 else self.eta
            iota    = p[7] if len(p) > 7 else self.iota
            mu      = p[8] if len(p) > 8 else self.mu
            theta   = p[9] if len(p) > 9 else self.theta
        elif cu in self._community_params:
            a, b, g, d, e = self._community_params[cu]
            zeta, eta, iota, mu, theta = (
                self.zeta, self.eta, self.iota, self.mu, self.theta
            )
        else:
            a, b, g, d, e = (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)
            zeta, eta, iota, mu, theta = (
                self.zeta, self.eta, self.iota, self.mu, self.theta
            )

        return (a, b, g, d, e, zeta, eta, iota, mu, theta)

def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)

class UniformCSAEngine(CSAEngine):
    def compute_weight(self, *args, **kwargs): return 0.5
    def compute_weight_with_features(self, u, v, hop, **kwargs):
        return ReasoningLogit(cs=0.5, hd=1.0/(1.0+hop))

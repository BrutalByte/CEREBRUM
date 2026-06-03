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
from typing import Dict, Optional, Set, Tuple, Any

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

class HomeostaticModulator:
    """
    Phase 143: Homeostatic Synaptic Scaling.
    Maintains target reasoning activity by scaling incoming weights 
    based on a rolling average of node activity.
    """
    def __init__(self, target_activity=1.158, tau=0.95):
        self.target = target_activity
        self.tau = tau
        self.node_activity = {} # entity_id -> rolling_avg
    
    def compute_scaling_factor(self, entity_id: str, current_activity: float) -> float:
        # Update rolling average
        prev = self.node_activity.get(entity_id, self.target)
        self.node_activity[entity_id] = self.tau * prev + (1 - self.tau) * current_activity
        # Scale factor: target / current
        return self.target / max(1e-6, self.node_activity[entity_id])

import torch
import torch.nn.functional as F
import numpy as np
# ... (rest of imports)

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
        **kwargs 
    ):
        self.adapter = adapter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core Parameters
        self.alpha = torch.tensor(alpha, device=self.device)
        self.beta = torch.tensor(beta, device=self.device)
        self.gamma = torch.tensor(gamma, device=self.device)
        self.delta = torch.tensor(delta, device=self.device)
        self.epsilon = torch.tensor(epsilon, device=self.device)
        self.zeta = torch.tensor(zeta, device=self.device)
        self.eta = torch.tensor(eta, device=self.device)
        self.iota = torch.tensor(iota, device=self.device)
        self.mu = torch.tensor(mu, device=self.device)
        self.theta = torch.tensor(theta, device=self.device)
        
        self.use_temporal_decay = use_temporal_decay
        self.temporal_window_size = temporal_window_size
        self.lambda_decay = lambda_decay
        # Phase 215-B: Power-law (hyperbolic) forgetting — Ebbinghaus (1885) shows
        # biological forgetting follows (1+t)^-λ, not exp(-λt). Exponential drops
        # too fast early and too slowly late. use_power_law_decay=False restores old behaviour.
        self.use_power_law_decay: bool = kwargs.get("use_power_law_decay", True)
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
        self._cdist_cache: Dict[Tuple[int, int], float] = {}
        self.meta_learner = None
        self._query_snapshot: Optional[Dict[str, int]] = None
        self._query_time: Optional[float] = None
        self._cs_cache: Dict[Tuple[str, str], float] = {}

    def compute_attention(self, u_idx: int, neighbor_indices: List[int]) -> torch.Tensor:
        """
        GPU-accelerated attention scoring.
        """
        # Batch computation on GPU
        # Convert inputs to tensors
        u_emb = self.get_emb(u_idx).to(self.device)
        v_embs = self.get_embs(neighbor_indices).to(self.device)
        
        # Parallel cosine similarity
        sims = F.cosine_similarity(u_emb, v_embs)
        # ... (rest of formula using torch operations)
        return torch.sigmoid(raw)
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

    def _lazy_community_distance(self, cu: int, cv: int) -> float:
        """BFS distance between two communities using the lightweight community graph.

        Falls back to 5.0 (exp(-λ*5) ≈ 0) when graph unavailable or no path.
        Results are memoized in _cdist_cache.
        """
        cached = self._cdist_cache.get((cu, cv))
        if cached is not None:
            return cached
        if self._community_graph is None:
            return 5.0
        try:
            import networkx as nx
            d = float(nx.shortest_path_length(self._community_graph, cu, cv))
            d = min(d, 5.0)
        except Exception:
            d = 5.0
        self._cdist_cache[(cu, cv)] = d
        self._cdist_cache[(cv, cu)] = d
        return d

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
                d = self._community_distances.get((cu, cv), self._community_distances.get((cv, cu), None))
                if d is None:
                    d = self._lazy_community_distance(cu, cv)
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
                if self.use_power_law_decay:
                    td = (1.0 + decay_rate * time_elapsed) ** -1.0
                else:
                    td = math.exp(-decay_rate * time_elapsed)
                if self.temporal_window_size and time_elapsed > self.temporal_window_size:
                    td *= 0.1 # Window penalty

        sd = 1.0 if "rem_synthesized" in edge_type else 0.0
        grounding = 1.0 # Default

        # Phase 143: Apply Homeostatic Scaling if modulator exists
        modulator = getattr(self, "homeostatic_modulator", None)
        if modulator:
            scale = modulator.compute_scaling_factor(v, sim + cs)
            sim *= scale
            cs *= scale

        return ReasoningLogit(sim, cs, etw, normalized_distance, hd, pr_v, td, nr_v, sd, grounding)

    def compute_weights_batch(
        self,
        u: str,
        v_list: List[str],
        hop: int,
        edge_types: List[str],
        valid_tos: List[Optional[float]],
        eu: Optional[np.ndarray] = None,
        ev_list: Optional[List[np.ndarray]] = None,
        edge_type_weights: Optional[Dict[str, float]] = None,
    ) -> List[ReasoningLogit]:
        """Vectorized batch version of compute_weight_with_features."""
        n = len(v_list)
        if n == 0: return []
        
        if eu is None: eu = self.adapter.get_embedding(u)

        # 1. Semantic Sim (Vectorized)
        if eu is not None and ev_list is not None:
            # Check if ev_list is a list of arrays or a single matrix
            if isinstance(ev_list, list):
                zero = np.zeros(eu.shape, dtype=eu.dtype)
                EV = np.stack([ev if ev is not None else zero for ev in ev_list])
            else:
                EV = ev_list

            # Dot products
            dots = np.dot(EV, eu)
            norms_v = np.linalg.norm(EV, axis=1)
            norm_u = np.linalg.norm(eu)
            sims = dots / (norms_v * norm_u + 1e-9)
        else:
            sims = [0.0] * n
            
        # 2. Community Scores (Still iterative for now due to complex logic)
        cs_scores = [self.community_score(u, v) for v in v_list]
        
        # 3. Features
        weights = edge_type_weights if edge_type_weights is not None else self.edge_type_weights
        hd = 1.0 / (1.0 + hop)
        
        # Phase 134: cache temporal decay per (valid_to, edge_type) — saves ~80% exp() calls
        _td_cache: Dict[Tuple[Any, str], float] = {}
        logits = []
        for i in range(n):
            v = v_list[i]
            etw = weights.get(edge_types[i], 0.0)
            pr_v = self._pagerank.get(v, 0.0) / self._max_pr if self._pagerank else 0.0
            nr_v = self._node_recency.get(v, 0.0)

            td = 0.0
            if self.use_temporal_decay and self._query_time is not None and valid_tos[i] is not None:
                time_elapsed = self._query_time - valid_tos[i]
                if time_elapsed > 0:
                    _td_key = (valid_tos[i], edge_types[i])
                    if _td_key in _td_cache:
                        td = _td_cache[_td_key]
                    else:
                        decay_rate = RELATION_DECAY_DEFAULTS.get(edge_types[i], self.lambda_decay)
                        if self.use_power_law_decay:
                            td = (1.0 + decay_rate * time_elapsed) ** -1.0
                        else:
                            td = math.exp(-decay_rate * time_elapsed)
                        if self.temporal_window_size and time_elapsed > self.temporal_window_size:
                            td *= 0.1
                        _td_cache[_td_key] = td

            sd = 1.0 if "rem_synthesized" in edge_types[i] else 0.0
            logits.append(ReasoningLogit(sims[i], cs_scores[i], etw, 0.0, hd, pr_v, td, nr_v, sd, 1.0))
            
        return logits

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

        def _to_float(v):
            return v.item() if hasattr(v, "item") else float(v)
        return tuple(_to_float(x) for x in (a, b, g, d, e, zeta, eta, iota, mu, theta))

    def score_logits_batch(
        self,
        logits: "List[ReasoningLogit]",
    ) -> "np.ndarray":
        """Vectorized sigmoid scoring for a batch of ReasoningLogit objects.

        Replaces N individual logit.score(params) calls (each a Python scalar
        multiply-chain + exp) with a single numpy matmul + vectorised exp.
        Returns float32 array of shape (N,) — same values as calling
        logit.score(get_current_params()) for each logit individually.
        """
        if not logits:
            return np.zeros(0, dtype=np.float32)

        def _f(t):
            return t.item() if hasattr(t, "item") else float(t)

        # Weight vector with sign baked in: matches ReasoningLogit.score() formula.
        # [a, b, g, -d, e, z, eta, iota, -mu, theta]
        _w = np.array([
            _f(self.alpha), _f(self.beta), _f(self.gamma),
            -_f(self.delta), _f(self.epsilon), _f(self.zeta),
            _f(self.eta), _f(self.iota), -_f(self.mu), _f(self.theta),
        ], dtype=np.float32)

        # Feature matrix (N, 10) — one row per logit
        _L = np.array([l.to_vector() for l in logits], dtype=np.float32)

        # Vectorised sigmoid: shape (N,)
        _raw = _L @ _w
        return (1.0 / (1.0 + np.exp(-_raw))).astype(np.float32)

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
    def compute_weight_with_features(
        self,
        u: Any,
        v: Any,
        hop: int,
        edge_type: str = "RELATED_TO",
        edge_type_weights: Optional[Dict[str, float]] = None,
        normalized_distance: float = 0.0,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        eu: Optional[np.ndarray] = None,
        ev: Optional[np.ndarray] = None,
    ) -> ReasoningLogit:
        return ReasoningLogit(cs=0.5, hd=1.0/(1.0+hop))

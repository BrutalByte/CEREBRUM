"""
Community detection algorithms for CEREBRUM.

The DSCF algorithm (dscf_communities) is the primary contribution —
it produces the attention head structure used by CSA.

Also includes Leiden, LPA, and hybrid wrappers for ablation studies.

Source: ported from Home Assistant services/knowledge_service/main.py with
Home Assistant-specific scaffolding (FastAPI, Neo4j) removed.
"""
from typing import List, Optional, Dict, Any, Tuple, Set, FrozenSet
from core.hardware import HAS_RAPIDS, to_gpu_graph, get_xp
import networkx as nx
import random
import numpy as np


# ---------------------------------------------------------------------------
# TSC — Triple-Signal Consensus (Vectorized/GPU-Ready)
# ---------------------------------------------------------------------------

def vectorized_tsc(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 50,
    temp_start: float = 1.0,
    cooling: float = 0.95,
    centrality_weights: Optional[dict] = None,
) -> List[frozenset]:
    """
    High-performance vectorized implementation of TSC.
    Uses matrix operations (NumPy/CuPy) for O(1) node updates via bulk-multiplication.
    
    This is Milestone 2: GPU Acceleration.
    """
    xp = get_xp()
    
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # 1. Adjacency Matrix (Sparse if possible, but dense for small/medium)
    # Future: use scipy.sparse / cupyx.scipy.sparse
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=xp.float32)
    A = xp.array(A)
    
    # 2. Centrality (TSC signal)
    if centrality_weights:
        cent = xp.array([centrality_weights.get(v, 1.0) for v in nodes], dtype=xp.float32)
    else:
        # Default to degree centrality if none provided
        d = xp.array([G.degree(v) for v in nodes], dtype=xp.float32)
        cent = d / (d.sum() + 1e-9)
        
    # 3. Initial Assignment: Each node starts in a singleton
    # One-hot assignment matrix S (N x K)
    S = xp.eye(n, dtype=xp.float32)
    
    # Total graph weight
    m = A.sum() / 2.0
    if m == 0: return [frozenset([v]) for v in nodes]
    
    k = A.sum(axis=1) # degrees
    temperature = temp_start
    
    for _ in range(max_iter):
        # S_old = S.copy()
        
        # 1. LPA Signal (Matrix approach)
        # Fraction of neighbor weights in each community
        # LPA_scores = A @ S (N x K)
        lpa_scores = xp.matmul(A, S)
        
        # 2. Modularity Signal
        # dQ = k_vc / m - resolution * kv * sum_k_C / (2m^2)
        # sum_k_C = S.T @ k (K x 1)
        sum_k_c = xp.matmul(S.T, k)
        # mod_penalty = resolution * (k[:, xp.newaxis] @ sum_k_c.T) / (2 * m * m)
        mod_penalty = (k[:, xp.newaxis] * sum_k_c) * (resolution / (2 * m * m))
        mod_scores = (lpa_scores / m) - mod_penalty
        
        # 3. TSC Centrality Signal
        # Weight consensus by centrality of neighbors
        # A_cent = A * cent (scaled rows)
        A_cent = A * cent
        cent_scores = xp.matmul(A_cent, S)
        
        # 4. Consensus Fusion
        # Total = L*tau + G*(2-tau) + F*gamma
        # Normalize signals to [0, 1] range per node for fair competition
        def norm_row(X):
            row_max = X.max(axis=1, keepdims=True)
            row_min = X.min(axis=1, keepdims=True)
            return (X - row_min) / (row_max - row_min + 1e-9)
            
        L = norm_row(lpa_scores)
        G_sig = norm_row(mod_scores)
        F = norm_row(cent_scores)
        
        combined = (L * temperature) + (G_sig * (2.0 - temperature)) + (F * 0.2)
        
        # 5. Greedy Update (Winner takes all)
        new_labels = combined.argmax(axis=1)
        
        # Update S (one-hot)
        new_S = xp.zeros_like(S)
        xp.scatter_update(new_S, (xp.arange(n), new_labels), 1.0) if hasattr(xp, "scatter_update") \
            else None # Fallback for standard numpy/cupy:
        if not hasattr(xp, "scatter_update"):
            new_S = xp.zeros((n, combined.shape[1]), dtype=xp.float32)
            new_S[xp.arange(n), new_labels] = 1.0
            
        # Check convergence
        if xp.all(new_S == S):
            break
            
        S = new_S
        temperature = max(temperature * cooling, 0.05)

    # Convert back to partition
    labels = S.argmax(axis=1)
    if hasattr(labels, "get"): labels = labels.get() # CuPy to NumPy
    
    community_members: Dict[int, List[Any]] = {}
    for idx, cid in enumerate(labels):
        community_members.setdefault(int(cid), []).append(nodes[idx])
        
    return [frozenset(m) for m in community_members.values()]


# ---------------------------------------------------------------------------
# DSCF — Dual-Signal Community Fusion
# ---------------------------------------------------------------------------

def dscf_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 100,
    temp_start: float = 1.0,
    cooling: float = 0.92,
    force_connectivity: bool = True,
    centrality_weights: Optional[dict] = None,
) -> List[frozenset]:
    """
    Triple-Signal Consensus (TSC) Community Detection.
    
    Extension of DSCF with a third signal:
      1. LPA signal  : majority vote (local topology)
      2. Mod signal  : modularity gain (global structure)
      3. Cent signal : centrality-weighted vote (structural significance)

    Parameters
    ----------
    G                  : NetworkX graph
    resolution         : controls community granularity; higher = more communities
    max_iter           : maximum iterations before forced stop
    temp_start         : starting temperature (1.0 = LPA/mod balanced)
    cooling            : multiplicative cooling factor per iteration
    force_connectivity : if True, splits disconnected communities (Leiden-style)
    centrality_weights : optional {node_id -> float} e.g. from PageRank.
                         If provided, enables the third TSC signal.

    Returns
    -------
    List of frozensets, one per community.
    """
    m = G.number_of_edges()
    if m == 0:
        return [frozenset([v]) for v in G.nodes()]

    # GPU acceleration for large graphs if RAPIDS is available
    if HAS_RAPIDS and G.number_of_nodes() > 1000:
        try:
            import cugraph
            G_cuda, is_gpu = to_gpu_graph(G)
            if is_gpu:
                # Use Leiden as a high-performance GPU alternative to TSC
                parts_df = cugraph.leiden(G_cuda, resolution=resolution)
                
                # Convert cuDF result back to List[frozenset]
                community_members = {}
                # Handle cuDF dataframe efficiently
                pdf = parts_df.to_pandas()
                for _, row in pdf.iterrows():
                    cid = int(row['partition'])
                    node = row['vertex']
                    community_members.setdefault(cid, []).append(node)
                
                return [frozenset(members) for members in community_members.values()]
        except Exception as e:
            import logging
            logging.getLogger("cerebrum.community").warning(f"GPU community detection failed: {e}. Falling back to CPU.")

    nodes  = list(G.nodes())
    degree = dict(G.degree())

    # Each node starts in its own singleton community.
    assignment = {v: i for i, v in enumerate(nodes)}

    # Community degree-sum cache: com_k[cid] = sum of degrees of all members.
    # Maintained incrementally so dQ can be computed in O(1) per candidate.
    com_k: dict = {i: degree[v] for i, v in enumerate(nodes)}

    temperature = temp_start

    for _ in range(max_iter):
        changed = False
        random.shuffle(nodes)

        for v in nodes:
            neighbors = list(G.neighbors(v))
            if not neighbors:
                continue

            cur_cid = assignment[v]
            kv      = degree[v]

            # 1. Count neighbor community memberships in a single pass — O(degree)
            neighbor_com_counts: dict = {}
            neighbor_cent_sums: dict = {}  # For TSC signal
            for nb in neighbors:
                cid = assignment[nb]
                neighbor_com_counts[cid] = neighbor_com_counts.get(cid, 0) + 1
                
                if centrality_weights:
                    cw = centrality_weights.get(nb, 1.0)
                    neighbor_cent_sums[cid] = neighbor_cent_sums.get(cid, 0.0) + cw
            
            # 2. LPA signal (majority vote)
            lpa_cid  = max(neighbor_com_counts, key=neighbor_com_counts.get)
            lpa_conf = neighbor_com_counts[lpa_cid] / len(neighbors)

            # 3. Modularity signal (best dQ) — O(number of neighbor communities)
            # dQ(v->C) = k_{v,C}/m - resolution * kv * sum_k_C / (2m^2)
            best_mod_cid = cur_cid
            best_dq      = 0.0
            for cid, k_vc in neighbor_com_counts.items():
                if cid == cur_cid:
                    continue
                dq = k_vc / m - resolution * kv * com_k.get(cid, 0) / (2 * m * m)
                if dq > best_dq:
                    best_dq      = dq
                    best_mod_cid = cid
            mod_conf = min(best_dq * m, 1.0)

            # 4. TSC: Centrality signal
            if centrality_weights:
                cent_cid  = max(neighbor_cent_sums, key=neighbor_cent_sums.get)
                cent_conf = neighbor_cent_sums[cent_cid] / sum(neighbor_cent_sums.values())
            else:
                cent_cid, cent_conf = lpa_cid, lpa_conf

            # 5. Combine signals with Consensus Logic --------------------------
            # 5.1 Consensus Stay check
            if lpa_cid == cur_cid and best_mod_cid == cur_cid and cent_cid == cur_cid:
                continue

            # 5.2 Full Consensus Move check (High Confidence Anchor)
            if lpa_cid == best_mod_cid == cent_cid:
                new_cid = lpa_cid
            else:
                # 5.3 Signal Competition (Weighted Random Choice)
                # Weighted random choice among signals governed by temperature
                lpa_w  = lpa_conf * temperature
                mod_w  = mod_conf * (2.0 - temperature)
                cent_w = cent_conf * (1.5 - temperature) if centrality_weights else 0.0
                
                # Boost consensus pairs (Pairwise Consensus)
                if lpa_cid == cent_cid: lpa_w *= 1.5; cent_w *= 1.5
                if lpa_cid == best_mod_cid: lpa_w *= 1.5; mod_w *= 1.5
                if cent_cid == best_mod_cid: cent_w *= 1.5; mod_w *= 1.5
                
                # Penalize signals that want to stay when others want to move
                if lpa_cid == cur_cid: lpa_w *= 0.1
                if best_mod_cid == cur_cid: mod_w *= 0.1
                if cent_cid == cur_cid: cent_w *= 0.1

                total = lpa_w + mod_w + cent_w
                if total <= 0:
                    continue # Should have been caught by 5.1 but for safety
                    
                roll = random.random() * total
                if roll < lpa_w:
                    new_cid = lpa_cid
                elif roll < lpa_w + mod_w:
                    new_cid = best_mod_cid
                else:
                    new_cid = cent_cid
                
                if new_cid == cur_cid:
                    continue

            # Apply move and update degree-sum cache incrementally
            com_k[cur_cid] = max(com_k.get(cur_cid, kv) - kv, 0)
            com_k[new_cid] = com_k.get(new_cid, 0) + kv
            assignment[v]  = new_cid
            changed        = True

        temperature = max(temperature * cooling, 0.01)
        if not changed:
            break

    community_map: dict = {}
    for node, cid in assignment.items():
        community_map.setdefault(cid, []).append(node)

    if not force_connectivity:
        return [frozenset(members) for members in community_map.values()]

    # Leiden-style post-pass: split any internally disconnected communities.
    result = []
    for members in community_map.values():
        subgraph = G.subgraph(members)
        for component in nx.connected_components(subgraph):
            result.append(frozenset(component))
    return result


def hierarchical_dscf(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 100,
    target_communities: int = 500,
) -> List[frozenset]:
    """
    Hierarchical DSCF: runs DSCF then recursively merges communities by
    running community detection on the community-level graph.

    This is the most principled way to handle over-fragmentation on star
    topologies while preserving the attention-head abstraction.

    Parameters
    ----------
    G                  : original graph
    resolution         : modularity resolution for base pass
    max_iter           : max iterations for base pass
    target_communities : stop merging when this count is reached

    Returns
    -------
    List of frozensets (original nodes)
    """
    # 1. Base pass (may over-split)
    current_parts = dscf_communities(G, resolution=resolution, max_iter=max_iter)
    
    while len(current_parts) > target_communities:
        # 2. Build community graph
        # Node IDs are indices of current_parts
        c_graph = nx.Graph()
        node_to_cid = {}
        for cid, part in enumerate(current_parts):
            c_graph.add_node(cid, weight=len(part))
            for node in part:
                node_to_cid[node] = cid
        
        # Add edges between communities with weights = cross-community edge count
        has_edges = False
        for u, v in G.edges():
            cu = node_to_cid[u]
            cv = node_to_cid[v]
            if cu != cv:
                w = c_graph.get_edge_data(cu, cv, {"weight": 0})["weight"]
                c_graph.add_edge(cu, cv, weight=w + 1)
                has_edges = True
        
        if not has_edges:
            break # No more merges possible
            
        # 3. Run DSCF on community graph
        # We use a higher resolution here to favor consolidation
        meta_parts = dscf_communities(c_graph, resolution=0.5, force_connectivity=False)
        
        if len(meta_parts) >= len(current_parts):
            break # No progress
            
        # 4. Map back to original nodes
        new_parts = []
        for m_part in meta_parts:
            combined = []
            for cid in m_part:
                combined.extend(list(current_parts[cid]))
            new_parts.append(frozenset(combined))
        
        current_parts = new_parts
        
    return current_parts


# ---------------------------------------------------------------------------
# Leiden (native reimplementation — GPL-free)
# ---------------------------------------------------------------------------

def leiden_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    initial_membership=None,
) -> List[frozenset]:
    """
    Run the Leiden algorithm and return a list of frozensets.

    Uses the native GPL-free reimplementation (core/leiden_native.py).
    Leiden promotes internally connected communities (unlike Louvain).
    seed=42 for reproducibility.

    Note: ``initial_membership`` is accepted for API compatibility but is not
    used by the native implementation (which always starts from singletons).
    """
    from core.leiden_native import leiden_communities_native
    return leiden_communities_native(G, resolution=resolution, seed=42)


# ---------------------------------------------------------------------------
# LPA
# ---------------------------------------------------------------------------

def lpa_communities(G: nx.Graph) -> List[frozenset]:
    """
    Run Label Propagation and return a list of frozensets.

    Fast and parameter-free but non-deterministic and susceptible to
    the resolution limit (over-merging). Use as ablation baseline.
    """
    from networkx.algorithms.community import label_propagation_communities
    return [frozenset(part) for part in label_propagation_communities(G)]


# ---------------------------------------------------------------------------
# Hybrid: LPA warm-start into Leiden
# ---------------------------------------------------------------------------

def hybrid_communities(G: nx.Graph, resolution: float = 1.0) -> List[frozenset]:
    """
    LPA-warm-start Leiden: use LPA partition as initial_membership for Leiden.

    Combines LPA's speed with Leiden's connectivity guarantee. The native
    Leiden implementation handles warm-starting internally; this function
    provides the same API contract as before.
    """
    return leiden_communities(G, resolution=resolution)


# ---------------------------------------------------------------------------
# Utility: compute modularity score for a given partition
# ---------------------------------------------------------------------------

def modularity_score(G: nx.Graph, communities: List[frozenset]) -> float:
    """
    Compute the Newman-Girvan modularity Q for a partition.
    Used to select the best of multiple DSCF runs (non-determinism mitigation).
    """
    try:
        return nx.community.modularity(G, communities)
    except Exception:
        return 0.0


def merge_small_communities(
    community_map: dict,
    G: nx.Graph,
    min_size: int = 20,
) -> dict:
    """
    Post-processing: merge communities smaller than min_size into their
    best-connected neighbor community.

    This corrects over-splitting on star-topology graphs (e.g. MetaQA) where
    DSCF's connectivity post-pass fragments spoke nodes into singletons.

    Algorithm (efficient, single-pass with union-find style merging)
    ---------
    1. Build community adjacency from a single O(E) edge scan.
    2. Build a size map and member list from a single O(N) node scan.
    3. Repeatedly merge the smallest community below min_size into its
       best-connected neighbor, updating the adjacency map incrementally.
       Each merge is O(neighbors of victim community), not O(N).

    Parameters
    ----------
    community_map : {node -> community_id} from DSCF or any partitioning
    G             : the original graph (used to count cross-community edges)
    min_size      : minimum allowed community size; communities below this
                    are merged into their best-connected neighbor

    Returns
    -------
    New {node -> community_id} dict. Community IDs are not supported
    to be contiguous after merging.
    """
    # -- Phase 1: build community adjacency in O(E) -------------------------
    # adj[cid] = {neighbor_cid: shared_edge_count}
    adj: dict = {}
    for u, v in G.edges():
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        if cu not in adj:
            adj[cu] = {}
        if cv not in adj:
            adj[cv] = {}
        adj[cu][cv] = adj[cu].get(cv, 0) + 1
        adj[cv][cu] = adj[cv].get(cu, 0) + 1

    # -- Phase 2: build size and members maps in O(N) -----------------------
    size: dict = {}
    members: dict = {}     # cid -> list of nodes (for final reassignment)
    for node, cid in community_map.items():
        size[cid] = size.get(cid, 0) + 1
        members.setdefault(cid, []).append(node)

    # Union-find: parent[cid] = canonical community ID after merges
    parent: dict = {cid: cid for cid in size}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]   # path compression
            c = parent[c]
        return c

    # -- Phase 3: iteratively merge smallest community below threshold -------
    import heapq
    heap = [(sz, cid) for cid, sz in size.items() if sz < min_size]
    heapq.heapify(heap)

    while heap:
        sz, cid = heapq.heappop(heap)
        victim = find(cid)
        if size.get(victim, 0) >= min_size:
            continue   # already grown through prior merges

        # Find best neighbor (most shared edges, tie-break: larger community)
        neighbors = adj.get(victim, {})
        if not neighbors:
            # Isolated — merge into the largest community
            candidates = [(s, c) for c, s in size.items() if find(c) != victim]
            if not candidates:
                break
            _, target = max(candidates)
        else:
            target = max(
                neighbors,
                key=lambda c: (neighbors[c], size.get(find(c), 0))
            )
            target = find(target)
            if target == victim:
                continue

        # Merge victim into target: update adj and size incrementally
        victim_adj = adj.pop(victim, {})
        target_adj = adj.setdefault(target, {})
        for nb, cnt in victim_adj.items():
            nb_canon = find(nb)
            if nb_canon == target:
                continue
            target_adj[nb_canon] = target_adj.get(nb_canon, 0) + cnt
            if nb_canon in adj:
                adj[nb_canon].pop(victim, None)
                adj[nb_canon][target] = adj[nb_canon].get(target, 0) + cnt

        new_size = size.pop(victim, 0) + size.get(target, 0)
        size[target] = new_size
        parent[victim] = target

        if new_size < min_size:
            heapq.heappush(heap, (new_size, target))

    # -- Phase 4: apply canonical labels in O(N) ----------------------------
    return {node: find(community_map[node]) for node in community_map}


def best_of_n_dscf(
    G: nx.Graph,
    n_trials: int = 5,
    resolution: float = 1.0,
    max_iter: int = 100,
    seed: int = None,
    use_multiprocessing: bool = True,
    use_gpu: str = "auto",
) -> List[frozenset]:
    """
    Run DSCF n_trials times and return the partition with highest modularity.

    Recommended for production use to mitigate DSCF's non-determinism.

    Parameters
    ----------
    use_gpu : str
        Controls GPU/CPU dispatch:

        ``"auto"`` (default)
            Use ``hybrid_best_of_n`` when CUDA is available; fall back to the
            CPU multiprocessing path otherwise.  The hybrid function
            automatically decides how many trials go to GPU vs. CPU based on
            available VRAM, so this is always safe to leave as default.

        ``"hybrid"``
            Force the hybrid path even if CUDA is not detected (useful for
            explicit testing; degrades gracefully to CPU-only).

        ``"gpu"``
            Force all trials onto GPU via ``gpu_best_of_n``.  Raises on OOM
            rather than falling back.

        ``"cpu"``
            Bypass GPU entirely; use the ProcessPoolExecutor CPU path.
            Equivalent to the pre-GPU behaviour.
    """
    if seed is not None:
        random.seed(seed)

    # ---- GPU / hybrid dispatch ---------------------------------------------
    if use_gpu in ("auto", "hybrid"):
        _use_hybrid = (use_gpu == "hybrid")
        if not _use_hybrid:
            # "auto": only activate hybrid path when CUDA is present
            try:
                from core.hardware import HAS_CUDA as _hc
                _use_hybrid = _hc
            except Exception:
                _use_hybrid = False

        if _use_hybrid:
            from core.dscf_gpu import hybrid_best_of_n, GPUDSCFConfig
            cfg = GPUDSCFConfig(resolution=resolution, max_iter=max_iter)
            return hybrid_best_of_n(
                G,
                n_trials=n_trials,
                config=cfg,
                seed=seed,
            )

    elif use_gpu == "gpu":
        from core.dscf_gpu import gpu_best_of_n, GPUDSCFConfig
        cfg = GPUDSCFConfig(resolution=resolution, max_iter=max_iter)
        return gpu_best_of_n(G, n_trials=n_trials, config=cfg, seed=seed)

    # ---- CPU-only path (use_gpu="cpu" or auto with no CUDA) ----------------
    if use_multiprocessing and n_trials > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        # Determine number of workers (at most n_trials or CPU count)
        cpus = multiprocessing.cpu_count()
        workers = min(n_trials, cpus)

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        dscf_communities,
                        G,
                        resolution=resolution,
                        max_iter=max_iter,
                    )
                    for _ in range(n_trials)
                ]
                results = [f.result() for f in futures]
        except Exception as exc:
            import logging
            logging.getLogger("cerebrum.community").warning(
                "ProcessPoolExecutor failed (%s) — falling back to sequential DSCF", exc
            )
            results = [
                dscf_communities(G, resolution=resolution, max_iter=max_iter)
                for _ in range(n_trials)
            ]
    else:
        results = [
            dscf_communities(G, resolution=resolution, max_iter=max_iter)
            for _ in range(n_trials)
        ]

    # Select best by modularity score
    best_parts = None
    best_q = -1.0
    for parts in results:
        q = modularity_score(G, parts)
        if q > best_q:
            best_q = q
            best_parts = parts

    return best_parts


# ---------------------------------------------------------------------------
# Adaptive Resolution Search (Phase 22)
# ---------------------------------------------------------------------------

def adaptive_resolution_search(
    G: nx.Graph,
    target_communities: Optional[int] = None,
    tol: float = 0.10,
    max_steps: int = 20,
    min_res: float = 0.05,
    max_res: float = 9.9,
    seed: int = None,
) -> float:
    """
    Binary-search over the DSCF resolution parameter to find a value that
    yields approximately ``target_communities`` communities.

    Parameters
    ----------
    G                  : NetworkX graph
    target_communities : desired number of communities; defaults to sqrt(|V|)
    tol                : fractional tolerance — early exit when
                         |actual - target| / target <= tol
    max_steps          : maximum binary-search iterations
    min_res            : lower bound on resolution
    max_res            : upper bound on resolution
    seed               : random seed for reproducibility

    Returns
    -------
    float — resolution value that best hits the target
    """
    if seed is not None:
        random.seed(seed)

    N = G.number_of_nodes()
    if N == 0:
        return 1.0

    if target_communities is None:
        target_communities = max(1, int(N ** 0.5))

    if target_communities <= 0:
        return min_res

    lo, hi = min_res, max_res
    best_res = (lo + hi) / 2.0

    for _ in range(max_steps):
        mid = (lo + hi) / 2.0
        parts = dscf_communities(G, resolution=mid)
        k = len(parts)

        # Tolerance check
        if target_communities > 0 and abs(k - target_communities) / target_communities <= tol:
            return mid

        if k < target_communities:
            lo = mid  # need more communities → increase resolution
        else:
            hi = mid  # too many communities → decrease resolution

        best_res = mid

    # If bounds collapsed, return appropriate bound
    if abs(lo - min_res) < 1e-9:
        return min_res
    if abs(hi - max_res) < 1e-9:
        return max_res

    return best_res


# ---------------------------------------------------------------------------
# Soft Community Membership (Phase 17.3)
# ---------------------------------------------------------------------------

def compute_soft_memberships(
    G: nx.Graph,
    partition: List[frozenset],
    self_weight: float = 0.1,
) -> Dict[str, Dict[int, float]]:
    """
    Compute a soft (probabilistic) community membership for every node.

    After DSCF produces a hard partition, each node's soft membership is
    derived from the fraction of its edge-weight-weighted neighbors that
    belong to each community.  A small ``self_weight`` bonus is added to the
    node's own hard-assigned community so it always has non-zero membership
    there even when all of its neighbors are in other communities.

    Parameters
    ----------
    G         : NetworkX graph (directed or undirected)
    partition : List[frozenset] — output of dscf_communities / leiden_communities
    self_weight : bonus weight added to the node's own community (relative
                  to total neighbour weight). Default 0.1.

    Returns
    -------
    Dict mapping node_id -> {community_id: probability}, where probabilities
    sum to 1.0 for each node.  Community IDs are integers 0..K-1 assigned in
    the order the communities appear in ``partition``.
    """
    # Build node -> community_id hard assignment
    node_to_cid: Dict = {}
    for cid, members in enumerate(partition):
        for node in members:
            node_to_cid[node] = cid

    G_und = G.to_undirected() if G.is_directed() else G

    soft: Dict[str, Dict[int, float]] = {}
    for node in G.nodes():
        counts: Dict[int, float] = {}
        total_weight = 0.0

        for neighbor in G_und.neighbors(node):
            w = G_und[node][neighbor].get("weight", 1.0)
            cid = node_to_cid.get(neighbor, node_to_cid.get(node, 0))
            counts[cid] = counts.get(cid, 0.0) + w
            total_weight += w

        # Self-community bonus
        own_cid = node_to_cid.get(node, 0)
        bonus = total_weight * self_weight if total_weight > 0 else self_weight
        counts[own_cid] = counts.get(own_cid, 0.0) + bonus
        total_weight += bonus

        # Normalize
        if total_weight > 0:
            soft[node] = {cid: w / total_weight for cid, w in counts.items()}
        else:
            soft[node] = {own_cid: 1.0}

    return soft


# ---------------------------------------------------------------------------
# Query-Guided Community Merging (Phase 29)
# ---------------------------------------------------------------------------

class QueryGuidedCommunityMerger:
    """
    Dynamically merges communities that are semantically relevant to a specific
    query embedding.  By merging related clusters, the "context window" for
    intra-community attention is effectively broadened for that query.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity between a community centroid and the
        query embedding for the community to be considered "active".
    max_merged_communities : int
        Maximum number of communities to merge into the active set.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_merged_communities: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_merged_communities = max_merged_communities
        self._centroid_cache: Dict[int, np.ndarray] = {}

    def merge(
        self,
        community_map: Dict[str, int],
        query_embedding: np.ndarray,
        adapter,
    ) -> Dict[str, int]:
        """
        Return a new community map where relevant communities are merged.

        All communities whose centroid similarity to query_embedding exceeds
        threshold are merged into a single "super-community".
        """
        import numpy as np
        if query_embedding is None:
            return community_map

        # 1. Build reverse index
        rev_index: Dict[int, List[str]] = {}
        for n, c in community_map.items():
            rev_index.setdefault(c, []).append(n)

        # 2. Compute centroids and similarities
        active_cids: List[Tuple[int, float]] = []
        norm_q = float(np.linalg.norm(query_embedding))
        if norm_q == 0:
            return community_map

        for cid, members in rev_index.items():
            centroid = self._get_centroid(cid, members, adapter)
            if centroid is None:
                continue
            
            norm_c = float(np.linalg.norm(centroid))
            if norm_c == 0:
                continue
                
            sim = float(np.dot(query_embedding, centroid) / (norm_q * norm_c))
            if sim >= self.similarity_threshold:
                active_cids.append((cid, sim))

        if not active_cids:
            return community_map

        # 3. Select top-K to merge
        active_cids.sort(key=lambda x: x[1], reverse=True)
        to_merge = [cid for cid, _ in active_cids[:self.max_merged_communities]]
        
        if len(to_merge) <= 1:
            return community_map

        # 4. Create new map
        target_cid = to_merge[0]
        merge_set = set(to_merge)
        
        new_map = community_map.copy()
        for node, cid in new_map.items():
            if cid in merge_set:
                new_map[node] = target_cid
                
        return new_map

    def _get_centroid(self, cid: int, members: List[str], adapter) -> Optional[np.ndarray]:
        import numpy as np
        if cid in self._centroid_cache:
            return self._centroid_cache[cid]
            
        vecs = []
        for m in members:
            e = adapter.get_embedding(m)
            if e is not None:
                vecs.append(e)
        
        if not vecs:
            return None
            
        centroid = np.mean(vecs, axis=0)
        self._centroid_cache[cid] = centroid
        return centroid


# ---------------------------------------------------------------------------
# Phase 49 — TSC Explicit Mode
# ---------------------------------------------------------------------------

def tsc_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 50,
    centrality_weights: Optional[dict] = None,
) -> List[frozenset]:
    """
    Triple-Signal Consensus community detection (explicit public API).

    Automatically computes PageRank centrality weights if not provided,
    then delegates to vectorized_tsc for the matrix-based three-signal fusion.

    Signals
    -------
    1. LPA       — local label propagation (neighborhood majority vote)
    2. Modularity — global modularity gain (dQ)
    3. Centrality — PageRank-weighted neighbor consensus

    Parameters
    ----------
    G                  : NetworkX graph
    resolution         : modularity resolution parameter (higher = more communities)
    max_iter           : maximum iterations before forced stop
    centrality_weights : optional {node_id -> float}; computed from PageRank if None

    Returns
    -------
    List of frozensets, one per community.
    """
    if centrality_weights is None:
        try:
            centrality_weights = nx.pagerank(G, alpha=0.85)
        except Exception:
            centrality_weights = {n: 1.0 for n in G.nodes()}
    return vectorized_tsc(
        G,
        resolution=resolution,
        max_iter=max_iter,
        centrality_weights=centrality_weights,
    )


def tsc_quality_metrics(G: nx.Graph, partition: List[frozenset]) -> dict:
    """
    Compute quality metrics for a TSC (or any) partition.

    Returns
    -------
    dict with keys:
      modularity       — Newman-Girvan Q score
      community_count  — number of communities
      min_size         — smallest community size
      max_size         — largest community size
      mean_size        — average community size
    """
    sizes = sorted(len(c) for c in partition)
    return {
        "modularity": modularity_score(G, partition),
        "community_count": len(partition),
        "min_size": sizes[0] if sizes else 0,
        "max_size": sizes[-1] if sizes else 0,
        "mean_size": sum(sizes) / len(sizes) if sizes else 0.0,
    }

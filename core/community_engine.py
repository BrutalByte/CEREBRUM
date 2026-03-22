"""
Community detection algorithms for Parallax.

The DSCF algorithm (dscf_communities) is the primary contribution —
it produces the attention head structure used by CSA.

Also includes Leiden, LPA, and hybrid wrappers for ablation studies.

Source: ported from Home Assistant services/knowledge_service/main.py with
Home Assistant-specific scaffolding (FastAPI, Neo4j) removed.
"""
from typing import List, Optional
from core.hardware import HAS_RAPIDS, to_gpu_graph
import networkx as nx
import random


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
            logging.getLogger("parallax.community").warning(f"GPU community detection failed: {e}. Falling back to CPU.")

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
# Support: igraph conversion (shared by Leiden and hybrid)
# ---------------------------------------------------------------------------

def _build_igraph(G: nx.Graph):
    """Convert a networkx Graph to igraph format for use with leidenalg."""
    import igraph as ig
    nodes_list  = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    ig_edges    = [(node_to_idx[s], node_to_idx[t]) for s, t in G.edges()]
    ig_G        = ig.Graph(n=len(nodes_list), edges=ig_edges)
    return ig_G, nodes_list, node_to_idx, idx_to_node


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------

def leiden_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    initial_membership=None,
) -> List[frozenset]:
    """
    Run the Leiden algorithm and return a list of frozensets.

    Leiden promotes internally connected communities (unlike Louvain).
    seed=42 for reproducibility. Pass initial_membership for warm-start.
    """
    import leidenalg
    ig_G, nodes_list, node_to_idx, idx_to_node = _build_igraph(G)
    kwargs = dict(resolution_parameter=resolution, seed=42)
    if initial_membership is not None:
        kwargs["initial_membership"] = initial_membership
    partition = leidenalg.find_partition(
        ig_G, leidenalg.RBConfigurationVertexPartition, **kwargs
    )
    return [frozenset(idx_to_node[i] for i in part) for part in partition]


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

    Combines LPA's speed with Leiden's connectivity promotes. Faster
    than cold-start Leiden on large graphs.
    """
    ig_G, nodes_list, node_to_idx, idx_to_node = _build_igraph(G)
    lpa_parts   = lpa_communities(G)
    membership  = [0] * len(nodes_list)
    for cid, members in enumerate(lpa_parts):
        for n in members:
            if n in node_to_idx:
                membership[node_to_idx[n]] = cid
    return leiden_communities(G, resolution=resolution, initial_membership=membership)


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
) -> List[frozenset]:
    """
    Run DSCF n_trials times and return the partition with highest modularity.

    Recommended for production use to mitigate DSCF's non-determinism.
    Uses multiprocessing by default for faster execution.
    """
    if seed is not None:
        random.seed(seed)

    if use_multiprocessing and n_trials > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        # Determine number of workers (at most n_trials or CPU count)
        cpus = multiprocessing.cpu_count()
        workers = min(n_trials, cpus)
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # We must pass arguments to dscf_communities. 
            # Note: G must be pickleable (NetworkX graphs are).
            futures = [
                executor.submit(
                    dscf_communities, 
                    G, 
                    resolution=resolution, 
                    max_iter=max_iter,
                    # Each process will have its own random state.
                )
                for _ in range(n_trials)
            ]
            results = [f.result() for f in futures]
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




"""
Community detection algorithms for Parallax.

The DSCF algorithm (dscf_communities) is the primary contribution —
it produces the attention head structure used by CSA.

Also includes Leiden, LPA, and hybrid wrappers for ablation studies.

Source: ported from AURA services/knowledge_service/main.py with
AURA-specific scaffolding (FastAPI, Neo4j) removed.
"""
import random
from typing import List

import networkx as nx


# ---------------------------------------------------------------------------
# DSCF — Dual-Signal Community Fusion
# ---------------------------------------------------------------------------

def dscf_communities(
    G: nx.Graph,
    resolution: float = 1.0,
    max_iter: int = 100,
    temp_start: float = 1.0,
    cooling: float = 0.92,
) -> List[frozenset]:
    """
    Dual-Signal Community Fusion (DSCF) — novel algorithm.

    At each node update step, two signals are computed simultaneously:
      1. LPA signal  : majority vote among neighbor labels  (local topology)
      2. Mod signal  : best modularity gain dQ move         (global structure)

    A temperature schedule (temp_start -> 0.01) governs the balance:
      high temperature  = LPA-heavy  (broad exploration)
      low  temperature  = mod-heavy  (precise exploitation)

    When both signals agree on a move, the node is assigned immediately as a
    high-confidence "anchor point." When they disagree, a weighted probabilistic
    choice is made governed by temperature.

    After convergence, a Leiden-style post-pass splits any internally
    disconnected communities. The first connected component of each split
    community retains the original ID (important for community_score caching).

    This combination of simultaneously evaluated LPA + modularity signals
    at each node update has not been published in the community detection
    literature as of the author's knowledge cutoff (August 2025).

    Parameters
    ----------
    G          : NetworkX graph
    resolution : controls community granularity; higher = more communities
    max_iter   : maximum iterations before forced stop
    temp_start : starting temperature (1.0 = LPA/mod balanced)
    cooling    : multiplicative cooling factor per iteration

    Returns
    -------
    List of frozensets, one per community.
    """
    m = G.number_of_edges()
    if m == 0:
        return [frozenset([v]) for v in G.nodes()]

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

            # -- LPA signal: majority vote among neighbor labels ---------------
            vote: dict = {}
            for nb in neighbors:
                c = assignment[nb]
                vote[c] = vote.get(c, 0) + 1
            lpa_cid  = max(vote, key=vote.get)
            lpa_conf = vote[lpa_cid] / len(neighbors)   # [0, 1]

            # -- Modularity signal: best dQ across candidate communities -------
            # dQ(v->C) = k_{v,C}/m - resolution * kv * sum_k_C / (2m^2)
            candidate_cids = set(assignment[nb] for nb in neighbors) - {cur_cid}
            best_mod_cid   = cur_cid
            best_dq        = 0.0
            for cid in candidate_cids:
                k_vc = sum(1 for nb in neighbors if assignment[nb] == cid)
                dq   = k_vc / m - resolution * kv * com_k.get(cid, 0) / (2 * m * m)
                if dq > best_dq:
                    best_dq      = dq
                    best_mod_cid = cid
            mod_conf = min(best_dq * m, 1.0)            # [0, 1]

            # -- Combine signals -----------------------------------------------
            if lpa_cid == best_mod_cid and lpa_cid != cur_cid:
                # Consensus: both agree -> anchor point, assign immediately
                new_cid = lpa_cid
            elif best_mod_cid == cur_cid and lpa_cid == cur_cid:
                continue   # both say stay
            elif best_mod_cid == cur_cid:
                # Only LPA wants to move
                if random.random() >= lpa_conf * temperature:
                    continue
                new_cid = lpa_cid
            elif lpa_cid == cur_cid:
                # Only modularity wants to move; gate grows as temperature falls
                if random.random() >= mod_conf * (1.0 + (1.0 - temperature)):
                    continue
                new_cid = best_mod_cid
            else:
                # Both want to move but to different communities.
                # Early (high temp): LPA-heavy.  Late (low temp): mod-heavy.
                lpa_w = lpa_conf * temperature
                mod_w = mod_conf * (2.0 - temperature)
                total = lpa_w + mod_w
                if total == 0:
                    continue
                new_cid = lpa_cid if random.random() < lpa_w / total else best_mod_cid

            # Apply move and update degree-sum cache incrementally
            com_k[cur_cid] = max(com_k.get(cur_cid, kv) - kv, 0)
            com_k[new_cid] = com_k.get(new_cid, 0) + kv
            assignment[v]  = new_cid
            changed        = True

        temperature = max(temperature * cooling, 0.01)
        if not changed:
            break

    # Leiden-style post-pass: split any internally disconnected communities.
    # The first component of each split retains the original ID for cache stability.
    community_map: dict = {}
    for node, cid in assignment.items():
        community_map.setdefault(cid, []).append(node)

    result = []
    for members in community_map.values():
        subgraph = G.subgraph(members)
        for component in nx.connected_components(subgraph):
            result.append(frozenset(component))
    return result


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

    Leiden guarantees internally connected communities (unlike Louvain).
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

    Combines LPA's speed with Leiden's connectivity guarantees. Faster
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
    New {node -> community_id} dict. Community IDs are not guaranteed
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
) -> List[frozenset]:
    """
    Run DSCF n_trials times and return the partition with highest modularity.

    Recommended for production use to mitigate DSCF's non-determinism.
    For reproducibility, set seed (seeds Python's random module).
    """
    if seed is not None:
        random.seed(seed)

    best_parts  = None
    best_q      = -1.0
    for _ in range(n_trials):
        parts = dscf_communities(G, resolution=resolution, max_iter=max_iter)
        q     = modularity_score(G, parts)
        if q > best_q:
            best_q     = q
            best_parts = parts

    return best_parts

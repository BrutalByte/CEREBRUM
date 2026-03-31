"""
Native Leiden Community Detection — Clean Reimplementation
==========================================================
Implements the Leiden algorithm (Traag, Waltman & van Eck, 2019) using only
NumPy and NetworkX.  No igraph.  No leidenalg.  No GPL dependencies.

Reference
---------
V.A. Traag, L. Waltman, N.J. van Eck. "From Louvain to Leiden: guaranteeing
well-connected communities." Scientific Reports 9, 5233 (2019).
https://doi.org/10.1038/s41598-019-41695-z

API
---
leiden_communities_native(G, resolution=1.0, n_iterations=10, seed=42)
    -> List[frozenset]

Drop-in replacement for leiden_communities() in core/community_engine.py.
Returns List[frozenset] of original node IDs — identical output contract.

Design note on degree tracking
-------------------------------
When the algorithm aggregates communities into super-nodes, internal edges
become self-loops and disappear from the adjacency list.  Without correction,
the super-node's degree would under-count, making modularity gains spuriously
positive and incorrectly merging well-separated communities.

Solution: ``node_degree[v]`` tracks the *effective* degree through all
aggregation levels — initially the adj-based degree, then the accumulated sum
across constituent nodes after each aggregation.  ``m`` (total edge weight in
the original graph) is kept constant throughout.  The ΔQ formula always uses
``node_degree`` (not adj-based degree) and the constant ``m``.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Lightweight internal graph
# ---------------------------------------------------------------------------

class _Graph:
    """
    Weighted undirected graph for Leiden internals.
    Integer nodes 0..n-1.  Edge weights stored in ``adj``.
    Self-loops are excluded from ``adj``.
    """
    __slots__ = ("n", "adj", "_total_weight")

    def __init__(self, n: int):
        self.n = n
        self.adj: List[Dict[int, float]] = [{} for _ in range(n)]
        self._total_weight: Optional[float] = None

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        if u == v:
            return
        self.adj[u][v] = self.adj[u].get(v, 0.0) + w
        self.adj[v][u] = self.adj[v].get(u, 0.0) + w
        self._total_weight = None

    @property
    def total_weight(self) -> float:
        if self._total_weight is None:
            self._total_weight = sum(sum(ws.values()) for ws in self.adj) / 2.0
        return self._total_weight

    def adj_degree(self, v: int) -> float:
        """Sum of adj edge weights (does NOT include self-loops or absorbed internal edges)."""
        return sum(self.adj[v].values())

    @staticmethod
    def from_networkx(G: nx.Graph) -> Tuple["_Graph", List, Dict]:
        nodes_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        g = _Graph(len(nodes_list))
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            g.add_edge(node_to_idx[u], node_to_idx[v], w)
        return g, nodes_list, node_to_idx


# ---------------------------------------------------------------------------
# Fast Local Moving (FLM)
# ---------------------------------------------------------------------------

def _fast_local_moving(
    g: _Graph,
    assignment: List[int],
    com_degree: Dict[int, float],
    node_degree: List[float],
    m: float,
    gamma: float,
    rng: random.Random,
) -> bool:
    """
    Queue-based fast local moving.

    Parameters
    ----------
    g           : current graph
    assignment  : mutable list[int], assignment[v] = community ID of node v
    com_degree  : mutable dict, com_degree[cid] = sum of node_degree for all v in cid
    node_degree : list[float], effective degree of each node (constant at this level)
    m           : total original edge weight (constant throughout)
    gamma       : resolution parameter

    Returns True if any node moved.
    """
    if m == 0:
        return False

    in_queue = [True] * g.n
    order = list(range(g.n))
    rng.shuffle(order)
    queue: deque = deque(order)
    any_moved = False

    while queue:
        v = queue.popleft()
        in_queue[v] = False

        cur_cid = assignment[v]
        kv = node_degree[v]
        kC = com_degree.get(cur_cid, 0.0)

        # Edges from v to its current community (C \ {v})
        e_to_C = 0.0
        # Collect candidate communities (neighbor communities only)
        candidate_e: Dict[int, float] = {}
        for nb, w in g.adj[v].items():
            nb_cid = assignment[nb]
            if nb_cid == cur_cid:
                e_to_C += w
            else:
                candidate_e[nb_cid] = candidate_e.get(nb_cid, 0.0) + w

        if not candidate_e:
            continue

        # Find best neighbor community by modularity gain
        best_cid = cur_cid
        best_dq = 0.0  # only move if strictly positive gain

        for cid, e_to_D in candidate_e.items():
            kD = com_degree.get(cid, 0.0)
            # ΔQ = [e(v,D) - e(v,C\{v})] / m  −  γ·kv·(kD − kC + kv) / (2m²)
            dq = (e_to_D - e_to_C) / m - gamma * kv * (kD - kC + kv) / (2.0 * m * m)
            if dq > best_dq:
                best_dq = dq
                best_cid = cid

        if best_cid == cur_cid:
            continue

        # Apply move
        com_degree[cur_cid] = max(com_degree.get(cur_cid, kv) - kv, 0.0)
        com_degree[best_cid] = com_degree.get(best_cid, 0.0) + kv
        assignment[v] = best_cid
        any_moved = True

        # Re-enqueue unqueued neighbors
        for nb in g.adj[v]:
            if not in_queue[nb]:
                in_queue[nb] = True
                queue.append(nb)

    return any_moved


# ---------------------------------------------------------------------------
# Connectivity post-pass (Leiden guarantee)
# ---------------------------------------------------------------------------

def _enforce_connectivity(
    g: _Graph,
    assignment: List[int],
    com_degree: Dict[int, float],
    node_degree: List[float],
) -> None:
    """
    Split disconnected communities in-place.

    Modifies ``assignment`` and ``com_degree`` directly.
    This is the core Leiden guarantee over Louvain.
    """
    com_members: Dict[int, List[int]] = {}
    for v, cid in enumerate(assignment):
        com_members.setdefault(cid, []).append(v)

    next_cid = max(assignment) + 1

    for cid, members in com_members.items():
        if len(members) <= 1:
            continue

        member_set = set(members)
        visited: set = set()
        components: List[List[int]] = []

        for start in members:
            if start in visited:
                continue
            component: List[int] = []
            stack = [start]
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                component.append(v)
                for nb in g.adj[v]:
                    if nb in member_set and nb not in visited:
                        stack.append(nb)
            components.append(component)

        if len(components) <= 1:
            continue

        # Keep largest component in current cid; assign new IDs to others
        largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
        for i, component in enumerate(components):
            if i == largest_idx:
                continue
            new_cid = next_cid
            next_cid += 1
            split_mass = sum(node_degree[v] for v in component)
            com_degree[cid] = max(com_degree.get(cid, 0.0) - split_mass, 0.0)
            com_degree[new_cid] = split_mass
            for v in component:
                assignment[v] = new_cid


# ---------------------------------------------------------------------------
# Refinement phase
# ---------------------------------------------------------------------------

def _refine_partition(
    g: _Graph,
    assignment: List[int],
    node_degree: List[float],
    m: float,
    gamma: float,
    rng: random.Random,
) -> List[int]:
    """
    Within each FLM community, run restricted local moving to obtain
    well-connected sub-communities.

    Returns refined assignment (new integer community IDs) that may split
    FLM communities into smaller, well-connected sub-communities.
    """
    if m == 0:
        return list(range(g.n))

    # Group nodes by FLM community
    outer_communities: Dict[int, List[int]] = {}
    for v, cid in enumerate(assignment):
        outer_communities.setdefault(cid, []).append(v)

    # Start: each node is its own sub-community (use node index as sub-cid)
    refined: List[int] = list(range(g.n))

    for outer_cid, members in outer_communities.items():
        if len(members) == 1:
            continue

        members_set = set(members)

        # Sub-community degree sums (using effective node degrees)
        sub_com_degree: Dict[int, float] = {v: node_degree[v] for v in members}

        shuffled = list(members)
        rng.shuffle(shuffled)
        changed = True

        while changed:
            changed = False
            for v in shuffled:
                cur_sub = refined[v]
                kv = node_degree[v]
                kC = sub_com_degree.get(cur_sub, 0.0)

                e_to_C = 0.0
                candidate_e: Dict[int, float] = {}
                for nb, w in g.adj[v].items():
                    if nb not in members_set:
                        continue
                    nb_sub = refined[nb]
                    if nb_sub == cur_sub:
                        e_to_C += w
                    else:
                        candidate_e[nb_sub] = candidate_e.get(nb_sub, 0.0) + w

                if not candidate_e:
                    continue

                best_sub = cur_sub
                best_dq = 0.0

                for sub_cid, e_to_D in candidate_e.items():
                    kD = sub_com_degree.get(sub_cid, 0.0)
                    dq = (e_to_D - e_to_C) / m - gamma * kv * (kD - kC + kv) / (2.0 * m * m)
                    if dq > best_dq:
                        best_dq = dq
                        best_sub = sub_cid

                if best_sub == cur_sub:
                    continue

                sub_com_degree[cur_sub] = max(sub_com_degree.get(cur_sub, kv) - kv, 0.0)
                sub_com_degree[best_sub] = sub_com_degree.get(best_sub, 0.0) + kv
                refined[v] = best_sub
                changed = True

    # Renumber to contiguous 0..k-1
    seen: Dict[int, int] = {}
    counter = 0
    result: List[int] = []
    for v in range(g.n):
        sid = refined[v]
        if sid not in seen:
            seen[sid] = counter
            counter += 1
        result.append(seen[sid])
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    g: _Graph,
    refined_assignment: List[int],
    node_degree: List[float],
) -> Tuple["_Graph", Dict[int, List[int]], List[float]]:
    """
    Build aggregate graph where each refined sub-community is a super-node.

    Returns
    -------
    agg_g            : new _Graph
    agg_to_orig      : agg_node_idx -> list of node indices in current graph
    new_node_degree  : effective degree for each super-node = sum of constituent degrees
    """
    unique_cids = sorted(set(refined_assignment))
    cid_to_agg = {cid: i for i, cid in enumerate(unique_cids)}
    k = len(unique_cids)

    agg_to_orig: Dict[int, List[int]] = {i: [] for i in range(k)}
    new_node_degree = [0.0] * k

    for v, cid in enumerate(refined_assignment):
        agg_v = cid_to_agg[cid]
        agg_to_orig[agg_v].append(v)
        # Key: carry forward the EFFECTIVE degree (not adj-based) so m is preserved
        new_node_degree[agg_v] += node_degree[v]

    agg_g = _Graph(k)

    # Add cross-community edges (each undirected pair is visited twice in adj)
    for v in range(g.n):
        agg_v = cid_to_agg[refined_assignment[v]]
        for nb, w in g.adj[v].items():
            agg_nb = cid_to_agg[refined_assignment[nb]]
            if agg_v < agg_nb:  # add once per undirected pair
                agg_g.add_edge(agg_v, agg_nb, w)

    return agg_g, agg_to_orig, new_node_degree


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leiden_communities_native(
    G: nx.Graph,
    resolution: float = 1.0,
    n_iterations: int = 10,
    seed: int = 42,
) -> List[frozenset]:
    """
    Run the Leiden algorithm and return a list of frozensets.

    Clean reimplementation using only NetworkX and NumPy.
    No igraph, no leidenalg, no GPL dependencies.

    Parameters
    ----------
    G            : NetworkX graph (directed graphs are symmetrized)
    resolution   : modularity resolution γ — higher = more/smaller communities
    n_iterations : maximum outer loop iterations (usually converges in 2–5)
    seed         : random seed for reproducibility

    Returns
    -------
    List[frozenset] of original NetworkX node IDs — one frozenset per community.
    """
    if G.number_of_nodes() == 0:
        return []

    if G.number_of_edges() == 0:
        return [frozenset([v]) for v in G.nodes()]

    G_und = G.to_undirected() if G.is_directed() else G

    rng = random.Random(seed)

    g, nodes_list, node_to_idx = _Graph.from_networkx(G_und)
    n = g.n

    # m is constant throughout — total weight of the ORIGINAL graph
    m = g.total_weight

    # node_degree[v] = effective degree, preserved across aggregation levels.
    # Initially = adj-based degree (= true degree in original graph).
    # After aggregation: sum of constituent node degrees (absorbs internal edge mass).
    node_degree: List[float] = [g.adj_degree(i) for i in range(n)]

    # Track which original nodes belong to each current-level node.
    # Indexed by current-level node index -> list of original node indices.
    current_to_orig: Dict[int, List[int]] = {i: [i] for i in range(n)}

    current_g = g

    # Initial assignment: each node is its own community
    assignment: List[int] = list(range(n))
    com_degree: Dict[int, float] = {i: node_degree[i] for i in range(n)}

    for _iteration in range(n_iterations):
        # Phase 1: Fast Local Moving
        _fast_local_moving(current_g, assignment, com_degree, node_degree, m, resolution, rng)

        # Phase 2: Connectivity guarantee (in-place)
        _enforce_connectivity(current_g, assignment, com_degree, node_degree)

        # Convergence check: no aggregation possible if each community = 1 node
        n_communities = len(set(assignment))
        if n_communities == current_g.n:
            break

        # Phase 3: Refinement within each community
        refined = _refine_partition(current_g, assignment, node_degree, m, resolution, rng)

        # Phase 4: Aggregate
        agg_g, agg_to_refined, new_node_degree = _aggregate(current_g, refined, node_degree)

        # Update origin mapping: agg_v -> original nodes
        new_current_to_orig: Dict[int, List[int]] = {}
        for agg_v, refined_nodes in agg_to_refined.items():
            orig = []
            for rv in refined_nodes:
                orig.extend(current_to_orig[rv])
            new_current_to_orig[agg_v] = orig

        current_to_orig = new_current_to_orig
        current_g = agg_g
        node_degree = new_node_degree
        n_agg = current_g.n

        # Next iteration: each aggregate node starts in its own community
        assignment = list(range(n_agg))
        com_degree = {i: node_degree[i] for i in range(n_agg)}

        if n_agg == 1:
            break

    # Map final assignments back to original NetworkX node IDs
    community_groups: Dict[int, List] = {}
    for agg_v, orig_nodes in current_to_orig.items():
        final_cid = assignment[agg_v]
        node_list = community_groups.setdefault(final_cid, [])
        for orig_v in orig_nodes:
            node_list.append(nodes_list[orig_v])

    return [frozenset(members) for members in community_groups.values()]

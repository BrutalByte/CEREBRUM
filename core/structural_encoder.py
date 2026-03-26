"""
Graph structural encoding — the positional encoding layer of CEREBRUM.

Computes PageRank, betweenness centrality, and degree per node, then
projects them into a d-dimensional embedding vector for use alongside
entity embeddings in the forward pass (Section 5, STEP 2).
"""
from typing import Dict, List, Optional

from core.hardware import HAS_RAPIDS, to_gpu_graph
import networkx as nx
import numpy as np


def compute_structural_features(
    G: nx.DiGraph,
    sample_limit: int = 800,
) -> Dict[str, dict]:
    """
    Compute PageRank, betweenness centrality, and degree for every node.

    GPU (RAPIDS) is used if available for O(10-100x) speedup on large graphs.
    Betweenness is O(V*E) — sampled on large graphs (>sample_limit nodes)
    using a random subset to stay tractable.

    Returns
    -------
    {node_id: {pagerank, betweenness, degree, in_degree, out_degree}}
    """
    if G.number_of_nodes() == 0:
        return {}

    # Check for GPU acceleration
    if HAS_RAPIDS:
        try:
            import cugraph
            G_cuda, is_gpu = to_gpu_graph(G)
            if is_gpu:
                # 1. PageRank
                pr_df = cugraph.pagerank(G_cuda, alpha=0.85, max_iter=100)
                pagerank = dict(zip(pr_df['vertex'], pr_df['pagerank']))
                
                # 2. Betweenness (Sampling if large)
                if G.number_of_nodes() > sample_limit:
                    import random
                    sample = random.sample(list(G.nodes()), min(400, G.number_of_nodes()))
                    # cuGraph betweenness doesn't have a direct 'subset' like NX, 
                    # but we can use 'k' for approximate or specify sources.
                    bc_df = cugraph.betweenness_centrality(G_cuda, k=len(sample), normalized=True)
                else:
                    bc_df = cugraph.betweenness_centrality(G_cuda, normalized=True)
                betweenness = dict(zip(bc_df['vertex'], bc_df['betweenness_centrality']))
                
                # 3. Degrees (Calculated on CPU via NX as it's O(V), fast enough)
                degree     = dict(G.degree())
                in_degree  = dict(G.in_degree()) if G.is_directed() else dict(G.degree())
                out_degree = dict(G.out_degree()) if G.is_directed() else dict(G.degree())
                
                return {
                    node: {
                        "pagerank":    round(pagerank.get(node, 0.0), 7),
                        "betweenness": round(betweenness.get(node, 0.0), 7),
                        "degree":      degree.get(node, 0),
                        "in_degree":   in_degree.get(node, 0),
                        "out_degree":  out_degree.get(node, 0),
                    }
                    for node in G.nodes()
                }
        except Exception as e:
            import logging
            logging.getLogger("cerebrum.structural").warning(f"GPU structural computation failed: {e}. Falling back to CPU.")

    # CPU Fallback (Existing implementation)
    pagerank   = nx.pagerank(G, alpha=0.85, max_iter=100)
    degree     = dict(G.degree())
    in_degree  = dict(G.in_degree()) if G.is_directed() else dict(G.degree())
    out_degree = dict(G.out_degree()) if G.is_directed() else dict(G.degree())

    if G.number_of_nodes() > sample_limit:
        import random
        sample      = random.sample(list(G.nodes()), min(400, G.number_of_nodes()))
        betweenness = nx.betweenness_centrality_subset(G, sample, sample, normalized=True)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)

    return {
        node: {
            "pagerank":    round(pagerank.get(node, 0.0), 7),
            "betweenness": round(betweenness.get(node, 0.0), 7),
            "degree":      degree.get(node, 0),
            "in_degree":   in_degree.get(node, 0),
            "out_degree":  out_degree.get(node, 0),
        }
        for node in G.nodes()
    }


def encode_structural_features(
    features: Dict[str, dict],
    dim: int = 64,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Project {pagerank, betweenness, log(degree+1)} per node into a
    d-dimensional float32 vector using a fixed random projection.

    Steps:
      1. Build raw [N x 3] feature matrix
      2. Normalize each column to [0, 1]
      3. Project into d-dimensions via a fixed random matrix W_pos

    The result can be added to entity embeddings in STEP 2 of the forward pass:
        h0_i = LayerNorm(h0_i + structural_features[entity_i])

    Parameters
    ----------
    features : dict from compute_structural_features
    dim      : target embedding dimension (default 64)
    seed     : seed for the fixed random projection (default 42)
    """
    if not features:
        return {}

    nodes = list(features.keys())
    raw   = np.array(
        [
            [
                features[n]["pagerank"],
                features[n]["betweenness"],
                float(np.log1p(features[n]["degree"])),
            ]
            for n in nodes
        ],
        dtype=np.float32,
    )

    # Normalize each column to [0, 1]
    col_max = raw.max(axis=0)
    col_max[col_max == 0] = 1.0
    raw = raw / col_max

    n_features = raw.shape[1]
    if dim == n_features:
        result_matrix = raw
    else:
        # Fixed random projection matrix (W_pos)
        # This mixes the 3 structural signals across all d dimensions.
        rng = np.random.default_rng(seed)
        # Use uniform distribution [0, 1] to keep results non-negative
        W_pos = rng.uniform(0, 1, (n_features, dim)).astype(np.float32)
        
        # Normalize W_pos rows to keep variance stable
        row_norms = np.linalg.norm(W_pos, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        W_pos /= row_norms
        
        result_matrix = raw @ W_pos

    return {n: result_matrix[i] for i, n in enumerate(nodes)}


def coarsen_communities(
    G: nx.Graph,
    community_map: Dict[str, int],
    target_max: int = 500,
    min_size: int = 3,
) -> Dict[str, int]:
    """
    Merge small communities into their most-connected neighbors until the
    community count falls to target_max or below.

    Motivation: when DSCF over-partitions a large graph (e.g. MetaQA produces
    14,976 communities, 74% of which are singletons), the community signal in
    CSA becomes degenerate — almost every edge scores 0.5 (adjacent) with no
    discrimination between near and far communities.  Coarsening restores a
    meaningful signal by grouping structurally related nodes together.

    Algorithm
    ---------
    Uses an inverse index (community -> members) so each merge is
    O(community_size × avg_degree) rather than O(N).  Iteratively finds
    communities below min_size (or the K - target_max smallest communities
    when K > target_max) and reassigns their members to the neighbor community
    sharing the most edges with them.  Stops when K <= target_max or no
    further merges are possible.

    Parameters
    ----------
    target_max : int
        Target upper bound on number of communities.
    min_size : int
        Communities smaller than this are always candidates for merging,
        even if K <= target_max.

    Returns
    -------
    New community_map with contiguous integer community IDs.
    Does NOT modify the input map.
    """
    import logging
    log = logging.getLogger("cerebrum.structural")

    cmap: Dict[str, int] = dict(community_map)

    # Build inverse index: cid -> list of member nodes
    members: Dict[int, list] = {}
    for node, cid in cmap.items():
        members.setdefault(cid, []).append(node)

    def _best_neighbor(cid: int) -> Optional[int]:
        """Neighbor community with the most shared edges — O(size × degree)."""
        counts: Dict[int, int] = {}
        for node in members.get(cid, []):
            for nbr in G.neighbors(node):
                nc = cmap.get(nbr)
                if nc is not None and nc != cid:
                    counts[nc] = counts.get(nc, 0) + 1
        return max(counts, key=lambda c: counts[c]) if counts else None

    def _merge(src: int, dst: int) -> None:
        """Merge community src into dst, updating cmap and members in O(size)."""
        for node in members.pop(src, []):
            cmap[node] = dst
            members[dst].append(node)

    for _pass in range(500):
        K = len(members)
        if K <= target_max and all(len(m) >= min_size for m in members.values()):
            break

        sizes = {cid: len(m) for cid, m in members.items()}

        # Phase 1: merge communities below min_size that have cross-community neighbors
        small = [cid for cid, sz in sizes.items() if sz < min_size]
        merged_any = False
        for cid in list(small):
            if cid not in members:
                continue
            best = _best_neighbor(cid)
            if best is None:
                continue  # isolated — skip
            _merge(cid, best)
            merged_any = True

        # Phase 2: if still over target, merge smallest mergeable communities
        if len(members) > target_max:
            sizes2 = {cid: len(m) for cid, m in members.items()}
            candidates = sorted(sizes2, key=lambda c: sizes2[c])
            needed = len(members) - target_max
            merged_phase2 = 0
            for cid in candidates:
                if merged_phase2 >= needed:
                    break
                if cid not in members:
                    continue
                best = _best_neighbor(cid)
                if best is None:
                    continue
                _merge(cid, best)
                merged_phase2 += 1
                merged_any = True

        if not merged_any:
            break

    final_k = len(members)
    log.info("coarsen_communities: %d -> %d communities", len(set(community_map.values())), final_k)

    # Relabel to contiguous 0-based integers
    old_ids = sorted(members.keys())
    remap = {old: new for new, old in enumerate(old_ids)}
    return {node: remap[cid] for node, cid in cmap.items()}


def build_community_distance_matrix(
    G: nx.Graph,
    community_map: Dict[str, int],
    max_communities: int = 2000,
) -> Dict[tuple, float]:
    """
    Precompute the shortest-path distance between every pair of communities
    in the community-level graph.

    The community-level graph has one node per community; two communities
    are adjacent if at least one cross-community edge exists between them.

    Used by CSAEngine.community_score() for the exp(-lambda * d) decay term.
    Returns a dict: {(cid_i, cid_j): hop_distance, ...} for all pairs.

    Parameters
    ----------
    max_communities : int
        If the number of unique communities exceeds this threshold, skip the
        all-pairs BFS and return an empty dict. CSAEngine falls back to its
        built-in default distance (d=5.0) for non-adjacent pairs.
        This avoids O(K²) blowup on large graphs.

        When this fires it usually means the community detection over-partitioned
        the graph (many singleton communities).  Call coarsen_communities() before
        this function to merge small communities and restore a meaningful signal.
    """
    communities = set(community_map.values())

    if len(communities) > max_communities:
        import logging
        logging.getLogger("cerebrum.structural").warning(
            "build_community_distance_matrix: %d communities exceeds cap of %d. "
            "Call coarsen_communities() first to restore full CSA signal.",
            len(communities), max_communities,
        )
        return {}

    # Build community-level graph
    community_graph = nx.Graph()
    community_graph.add_nodes_from(communities)

    for u, v in G.edges():
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu is not None and cv is not None and cu != cv:
            community_graph.add_edge(cu, cv)

    # BFS shortest paths between all community pairs, capped at depth 5.
    # exp(-λ*5) ≈ 0.007 for λ=1 — negligible CSA contribution beyond depth 5.
    distances: Dict[tuple, float] = {}
    for source in communities:
        lengths = nx.single_source_shortest_path_length(community_graph, source, cutoff=5)
        for target, dist in lengths.items():
            if source != target:
                distances[(source, target)] = float(dist)

    return distances


def build_community_graph(
    G: nx.Graph,
    community_map: Dict[str, int],
) -> "nx.Graph":
    """
    Build the community-level graph without computing any distances.

    Each unique community ID becomes a node; two communities are connected
    by an edge if at least one cross-community edge exists between them in G.

    This is O(|E|) and is the lightweight alternative to
    ``build_community_distance_matrix``.  Pass the result to
    ``CSAEngine.set_community_graph(..., community_graph=cg)`` to enable
    lazy on-demand distance computation instead of upfront all-pairs BFS.
    """
    communities = set(community_map.values())
    cg = nx.Graph()
    cg.add_nodes_from(communities)
    for u, v in G.edges():
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu is not None and cv is not None and cu != cv:
            cg.add_edge(cu, cv)
    return cg


def adjacent_community_pairs(
    G: nx.Graph,
    community_map: Dict[str, int],
) -> set:
    """
    Return the set of (cid_i, cid_j) pairs where the two communities share
    at least one edge in G. Used by CSAEngine for the 0.5 adjacency score.
    """
    pairs = set()
    for u, v in G.edges():
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu is not None and cv is not None and cu != cv:
            pairs.add((cu, cv))
            pairs.add((cv, cu))
    return pairs




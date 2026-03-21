"""
Graph structural encoding — the positional encoding layer of Parallax.

Computes PageRank, betweenness centrality, and degree per node, then
projects them into a d-dimensional embedding vector for use alongside
entity embeddings in the forward pass (Section 5, STEP 2).
"""
from typing import Dict, List

import networkx as nx
import numpy as np


def compute_structural_features(
    G: nx.DiGraph,
    sample_limit: int = 800,
) -> Dict[str, dict]:
    """
    Compute PageRank, betweenness centrality, and degree for every node.

    Betweenness is O(V*E) — sampled on large graphs (>sample_limit nodes)
    using a random subset to stay tractable.

    Returns
    -------
    {node_id: {pagerank, betweenness, degree, in_degree, out_degree}}
    """
    if G.number_of_nodes() == 0:
        return {}

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
        built-in default distance (d=5.0) for non-adjacent pairs, which
        evaluates to exp(-λ*5) ≈ 0.082 — a conservative penalty that still
        allows adjacent-pair (0.5) and same-community (1.0) scores to function.
        This avoids O(N²) blowup on large graphs with many communities
        (e.g. MetaQA DSCF produces ~15K communities → 2.1B BFS operations).
    """
    communities = set(community_map.values())

    if len(communities) > max_communities:
        import sys
        print(
            f"  [build_community_distance_matrix] {len(communities):,} communities "
            f"exceeds max_communities={max_communities:,}. "
            f"Skipping all-pairs BFS — CSA will use d=5.0 fallback for non-adjacent pairs.",
            file=sys.stderr,
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




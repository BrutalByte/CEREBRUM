
from typing import Counter, Set
import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock
from core.community_engine import QueryGuidedCommunityMerger
from reasoning.traversal import BeamTraversal

def _make_mock_adapter(nodes_by_community, dim=8):
    """
    Create a mock adapter with specified nodes in each community.
    Each community will have a distinct "ideal" embedding (centroid).
    """
    adapter = MagicMock()
    community_map = {}
    embeddings = {}
    
    for cid, nodes in nodes_by_community.items():
        # Create a "centroid" for this community
        # cid 0 -> [1, 0, 0, ...]
        # cid 1 -> [0, 1, 0, ...]
        base_vec = np.zeros(dim, dtype=np.float32)
        if cid < dim:
            base_vec[cid] = 1.0
            
        for i, node in enumerate(nodes):
            community_map[node] = cid
            # Add some noise to the node embedding so it's not identical to centroid
            noise = np.random.normal(0, 0.05, dim).astype(np.float32)
            embeddings[node] = (base_vec + noise) / np.linalg.norm(base_vec + noise)

    adapter.community_map = community_map
    adapter.get_embedding.side_effect = lambda n: embeddings.get(n)
    adapter.get_community.side_effect = lambda n: community_map.get(n, -1)
    
    # Minimal graph for traversal mocks
    G = nx.Graph()
    for nodes in nodes_by_community.values():
        for i in range(len(nodes)-1):
            G.add_edge(nodes[i], nodes[i+1])
    
    adapter.get_neighbors.side_effect = lambda n, **kwargs: [] # not needed for merger test
    return adapter

def test_merger_logic_basic():
    # Comm 0: "fruit" (1, 0, 0)
    # Comm 1: "space" (0, 1, 0)
    # Comm 2: "music" (0, 0, 1)
    nodes = {
        0: ["apple", "banana", "cherry"],
        1: ["mars", "jupiter", "saturn"],
        2: ["jazz", "blues", "rock"]
    }
    adapter = _make_mock_adapter(nodes, dim=3)
    merger = QueryGuidedCommunityMerger(similarity_threshold=0.8)
    
    # Query: "astronomy" (close to Comm 1)
    q_emb = np.array([0.1, 0.95, 0.0], dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb)
    
    # Since only ONE community (cid=1) will be "active" (sim > 0.8),
    # no merge should happen (to_merge length <= 1).
    new_map = merger.merge(adapter.community_map, q_emb, adapter)
    assert len(set(new_map.values())) == 3
    assert new_map == adapter.community_map

def test_merger_logic_multi_match():
    # Comm 0 & 1 are both "science" related
    # Comm 2 is "art"
    nodes = {
        0: ["physics", "math"],
        1: ["biology", "chemistry"],
        2: ["painting", "sculpture"]
    }
    # Comm 0 & 1 centroids will be made similar for this test
    adapter = MagicMock()
    c0_vec = np.array([1.0, 0.0], dtype=np.float32)
    c1_vec = np.array([0.9, 0.1], dtype=np.float32) # Very close to c0
    c2_vec = np.array([0.0, 1.0], dtype=np.float32) # Distant
    
    embeddings = {
        "physics": c0_vec, "math": c0_vec,
        "biology": c1_vec, "chemistry": c1_vec,
        "painting": c2_vec, "sculpture": c2_vec
    }
    cmap = {
        "physics": 0, "math": 0,
        "biology": 1, "chemistry": 1,
        "painting": 2, "sculpture": 2
    }
    adapter.community_map = cmap
    adapter.get_embedding.side_effect = lambda n: embeddings.get(n)
    
    # Query: "natural sciences"
    q_emb = np.array([0.95, 0.05], dtype=np.float32)
    
    merger = QueryGuidedCommunityMerger(similarity_threshold=0.8)
    new_map = merger.merge(cmap, q_emb, adapter)
    
    # Comm 0 & 1 should be merged
    unique_cids = set(new_map.values())
    assert len(unique_cids) == 2 # (0+1 merged) and (2)
    assert new_map["physics"] == new_map["biology"]
    assert new_map["physics"] != new_map["painting"]

def test_traversal_integration():
    # Verify BeamTraversal.traverse calls merger.merge
    adapter = MagicMock()
    adapter.community_map = {"a": 0, "b": 1}
    adapter.get_embedding.return_value = np.zeros(8)
    adapter.get_community.return_value = 0 # Return an int, not a Mock
    
    csa = MagicMock()
    bt = BeamTraversal(adapter, csa)
    
    merger = MagicMock()
    # Merger returns a modified map
    merger.merge.return_value = {"a": 99, "b": 99}
    
    q_emb = np.ones(8)
    
    bt.traverse(["a"], query_embedding=q_emb, community_merger=merger)
    
    # Verify merger was called with correct args
    merger.merge.assert_called_once()
    args, kwargs = merger.merge.call_args
    assert args[0] == adapter.community_map
    assert np.array_equal(args[1], q_emb)
    
    # Verify CSA snapshot used the MERGED map (99), not original (0, 1)
    csa.set_query_snapshot.assert_called_with({"a": 99, "b": 99})

def test_merger_max_limit():
    nodes = {i: [f"n{i}"] for i in range(10)}
    adapter = _make_mock_adapter(nodes, dim=10)
    
    # Query matches ALL (since centroids are orthogonal but threshold is low)
    # or let's just make q match many.
    q_emb = np.ones(10, dtype=np.float32) / np.sqrt(10)
    
    # Set threshold very low so all 10 match
    merger = QueryGuidedCommunityMerger(similarity_threshold=0.1, max_merged_communities=3)
    new_map = merger.merge(adapter.community_map, q_emb, adapter)
    
    # Should only merge 3 communities
    cids = list(new_map.values())
    # 3 nodes should have one cid, the other 7 should keep their own
    from collections import Counter
    counts = Counter(cids)
    # The merged cid will have 3 members, others will have 1
    assert 3 in counts.values()
    assert len([v for v in counts.values() if v == 1]) == 7

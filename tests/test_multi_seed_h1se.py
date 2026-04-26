
import pytest
from unittest.mock import MagicMock
from reasoning.expanded_traversal import HopExpandedTraversal
from core.graph_adapter import Edge

def test_multi_seed_intersection_bonus():
    """Verify that neighbors shared by multiple seeds are prioritized."""
    adapter = MagicMock()
    
    # Seeds: S1, S2
    # Neighbors:
    #   A (reached by S1 and S2) -> Common
    #   B (reached by S1 only)
    #   C (reached by S2 only)
    
    def get_neighbors(u, **kwargs):
        if u == "S1":
            return [Edge("S1", "A", "r", 1.0), Edge("S1", "B", "r", 1.1)] # B has higher base score
        if u == "S2":
            return [Edge("S2", "A", "r", 1.0), Edge("S2", "C", "r", 1.0)]
        if u in ["A", "B", "C"]:
            # Deep expansions
            return [Edge(u, "Result"+u, "r", 1.0)]
        return []
        
    adapter.get_neighbors.side_effect = get_neighbors
    adapter.get_embedding.return_value = None
    adapter.get_community.return_value = 0
    
    csa = MagicMock()
    csa.compute_weight.return_value = 1.0 # Base CSA weight
    
    # expansion_k=1 (Only expand the very best hop-1 candidate)
    # total hops = 3. 
    # Scan: S1,S2 -> {A,B,C}
    # Deep (2 hops): {A,B,C} -> {ResultA, ResultB, ResultC}
    het = HopExpandedTraversal(
        adapter=adapter,
        csa_engine=csa,
        expansion_k=1,
        max_hop=3
    )
    
    # Without intersection bonus, B would win (score 1.1) vs A (score 1.0).
    # With intersection bonus, A gets 1.0 * 1.2 = 1.2. A should win.
    
    results = het.traverse(["S1", "S2"])
    
    # Find paths that have 3 hops (stashed Stage 2 results)
    deep_paths = [p for p in results if len(p.nodes) > 3]
    
    # If A was prioritized, deep_paths should be ResultA
    assert len(deep_paths) > 0
    # nodes sequence: [S, r, A, r, ResultA] -> nodes[2] == 'A'
    assert deep_paths[0].nodes[2] == "A"
    
def test_multi_seed_no_seeds():
    """Verify empty seeds handled gracefully."""
    het = HopExpandedTraversal(MagicMock(), MagicMock())
    assert het.traverse([]) == []

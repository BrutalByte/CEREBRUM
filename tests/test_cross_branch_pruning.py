
import pytest
from unittest.mock import MagicMock
from reasoning.expanded_traversal import GlobalBeamBarrier, HopExpandedTraversal
from reasoning.traversal import TraversalPath

def test_global_beam_barrier_pruning_logic():
    """Verify barrier correctly identifies non-viable branches."""
    # min_guaranteed=0 so pruning logic engages immediately (unit-test mode).
    barrier = GlobalBeamBarrier(target_count=3, threshold_ratio=0.5, min_guaranteed=0)

    # First report: always True (not enough data)
    assert barrier.report(sub_id=0, top_score=100.0) is True

    # Second report: still True (max(100, 50) = 100. 50 >= 100*0.5)
    assert barrier.report(sub_id=1, top_score=50.0) is True

    # Third report: max is 100. 10.0 < 100*0.5 -> False (Prune)
    assert barrier.report(sub_id=2, top_score=10.0) is False


def test_global_beam_barrier_min_guaranteed():
    """Phase 185: top-N branches always run regardless of score."""
    barrier = GlobalBeamBarrier(target_count=20, threshold_ratio=0.5, min_guaranteed=10)

    # Seed data from high-scoring branches
    for i in range(10):
        barrier.report(sub_id=i, top_score=100.0 - i)

    # Branch 9 (last guaranteed): always True
    assert barrier.report(sub_id=9, top_score=0.001) is True

    # Branch 10 (first non-guaranteed): low score -> pruned
    assert barrier.report(sub_id=10, top_score=0.001) is False

    # Branch 11: high score -> kept
    assert barrier.report(sub_id=11, top_score=60.0) is True

def test_h1se_passes_callback_and_prunes():
    """Integration test: Verify HopExpandedTraversal prunes weak branches."""
    adapter = MagicMock()
    
    def get_neighbors(u, **kwargs):
        if u == "Seed":
            e0 = MagicMock(); e0.target_id="Branch0"; e0.relation_type="r"; e0.confidence=1.0
            e1 = MagicMock(); e1.target_id="Branch1"; e1.relation_type="r"; e1.confidence=1.0
            e2 = MagicMock(); e2.target_id="Branch2"; e2.relation_type="r"; e2.confidence=1.0
            return [e0, e1, e2]
        if u == "Branch0":
            e = MagicMock(); e.target_id="Mid0"; e.relation_type="r"; e.confidence=1.0
            return [e]
        if u == "Branch1":
            e = MagicMock(); e.target_id="Mid1"; e.relation_type="r"; e.confidence=1.0
            return [e]
        if u == "Branch2":
            e = MagicMock(); e.target_id="Mid2"; e.relation_type="r"; e.confidence=1.0
            return [e]
        if u.startswith("Mid"):
            e = MagicMock(); e.target_id="Result"+u[-1]; e.relation_type="r"; e.confidence=1.0
            return [e]
        return []
        
    adapter.get_neighbors.side_effect = get_neighbors
    adapter.get_embedding.return_value = None
    adapter.get_community.return_value = 0
    
    csa = MagicMock()
    # 3-hop query. 
    # Hop 1: Seed -> {B0, B1, B2} (Scan)
    # Deep Leg (2 hops):
    #   SubHop 1: Bx -> MidX
    #   SubHop 2: MidX -> ResultX
    def mock_weight(u, v, **kwargs):
        if u == "Seed": return 1.0
        if u == "Branch0": return 1.0      # Global Max = 1.0
        if u == "Branch1": return 0.8      # 0.8 >= 1.0*0.5 -> Keep
        if u == "Branch2": return 0.001    # 0.001 < 1.0*0.5 -> Prune
        return 1.0 # Result hops
    csa.compute_weight.side_effect = mock_weight
    
    # H1SE with 3 branches, total 3 hops
    het = HopExpandedTraversal(
        adapter=adapter,
        csa_engine=csa,
        expansion_k=3,
        max_hop=3,
        barrier_min_guaranteed=0,
    )
    
    results = het.traverse(["Seed"])
    
    # Verify Result2 is NOT in results because Branch2 was pruned during SubHop 1
    tails = [p.tail for p in results]
    assert "Result0" in tails
    assert "Result1" in tails
    assert "Result2" not in tails
    assert "Mid2" in tails # Mid-nodes from SubHop 1 are kept before pruning check


import pytest
from unittest.mock import MagicMock
from reasoning.expanded_traversal import HopExpandedTraversal

def test_expansion_k_scales_with_arousal():
    """Verify k_eff increases when arousal (frustration) is high."""
    modulator = MagicMock()
    modulator.arousal = 1.0        # High uncertainty
    modulator.reinforcement = 0.0  # Low confidence
    
    base_k = 20
    het = HopExpandedTraversal(
        adapter=MagicMock(),
        csa_engine=MagicMock(),
        expansion_k=base_k,
        modulator=modulator,
        use_adaptive_expansion=True
    )
    
    k_eff = het._get_adaptive_k()
    # scale = 1.0 + 1.0 - 0.0 = 2.0. k_eff should be 20 * 2.0 = 40.
    assert k_eff == 40
    assert k_eff > base_k

def test_expansion_k_scales_with_reinforcement():
    """Verify k_eff decreases when reinforcement (confidence) is high."""
    modulator = MagicMock()
    modulator.arousal = 0.0        # Low uncertainty
    modulator.reinforcement = 1.0  # High confidence
    
    base_k = 20
    het = HopExpandedTraversal(
        adapter=MagicMock(),
        csa_engine=MagicMock(),
        expansion_k=base_k,
        modulator=modulator,
        use_adaptive_expansion=True
    )
    
    k_eff = het._get_adaptive_k()
    # scale = 1.0 + 0.0 - 1.0 = 0.0. Clamped to 0.2.
    # k_eff should be 20 * 0.2 = 4.
    assert k_eff == 4
    assert k_eff < base_k

def test_adaptive_expansion_disabled():
    """Verify k_eff remains at base_k when adaptive expansion is disabled."""
    modulator = MagicMock()
    modulator.arousal = 1.0
    modulator.reinforcement = 0.0
    
    base_k = 20
    het = HopExpandedTraversal(
        adapter=MagicMock(),
        csa_engine=MagicMock(),
        expansion_k=base_k,
        modulator=modulator,
        use_adaptive_expansion=False
    )
    
    k_eff = het._get_adaptive_k()
    assert k_eff == base_k

def test_budget_scales_with_k_eff():
    """Verify per-traversal budget decreases as k_eff increases to maintain total budget."""
    het = HopExpandedTraversal(
        adapter=MagicMock(),
        csa_engine=MagicMock(),
        max_budget=10000,
        expansion_k=20
    )
    
    # k=4 -> budget per branch should be approx 10000 / 5 = 2000
    b_small = het._per_traversal_budget(4)
    # k=40 -> budget per branch should be approx 10000 / 41 = 243
    b_large = het._per_traversal_budget(40)
    
    assert b_small > b_large
    assert b_large >= 100 # Min floor

import pytest
import time
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from core.synaptic_pruner import SynapticPruner

def test_synaptic_pruner_logic():
    # 1. Setup graph
    G = nx.MultiDiGraph()
    now = time.time()
    
    # High confidence edge
    G.add_edge("A", "B", relation="related_to", confidence=0.9, timestamp=now)
    # Low confidence edge
    G.add_edge("A", "C", relation="related_to", confidence=0.1, timestamp=now)
    # Old low-ish confidence edge (should be pruned)
    old_ts = now - (40 * 86400) # 40 days ago
    G.add_edge("B", "D", relation="related_to", confidence=0.4, timestamp=old_ts)
    
    adapter = NetworkXAdapter(G)
    pruner = SynapticPruner(adapter, min_confidence=0.3, max_age_days=30.0, prune_ratio=0.5)
    
    # 2. Prune
    pruned_count = pruner.prune()
    
    # 3. Assertions
    # A->C (0.1 conf) and B->D (old) should be pruned
    assert pruned_count >= 2
    assert G.has_edge("A", "B")
    assert not G.has_edge("A", "C")
    assert not G.has_edge("B", "D")

def test_protected_types():
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", relation="essential", confidence=0.1) # Low conf but protected
    
    adapter = NetworkXAdapter(G)
    pruner = SynapticPruner(adapter, protected_relation_types={"essential"})
    
    count = pruner.prune()
    assert count == 0
    assert G.has_edge("A", "B")

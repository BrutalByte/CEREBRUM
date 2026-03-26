
import pytest
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from core.insight_engine import InsightEngine, INSIGHT_RELATION

def test_insight_decay_and_pruning():
    """
    Verify that insight edges decay over time and are pruned when 
    confidence falls below threshold.
    """
    adapter = NetworkXAdapter(nx.DiGraph())
    adapter._G.add_edge("A", "B", relation="grounded", confidence=1.0)
    
    # Manually add an insight edge
    adapter._G.add_edge("A", "C", relation=INSIGHT_RELATION, confidence=0.5)
    
    engine = InsightEngine(adapter)
    G = adapter.to_networkx()
    
    # 1. First decay pass
    # Default decay is 0.95
    pruned = engine._decay_existing_insights(G, decay_rate=0.9, min_conf=0.2)
    assert pruned == 0
    assert G.has_edge("A", "C")
    assert G["A"]["C"]["confidence"] == pytest.approx(0.45)
    
    # 2. Subsequent decay passes until pruning
    engine._decay_existing_insights(G, decay_rate=0.5, min_conf=0.2) # 0.45 -> 0.225
    assert G.has_edge("A", "C")
    
    pruned = engine._decay_existing_insights(G, decay_rate=0.5, min_conf=0.2) # 0.225 -> 0.1125
    assert pruned == 1
    assert not G.has_edge("A", "C")
    
    # 3. Grounded edges should NOT decay
    assert G.has_edge("A", "B")
    assert G["A"]["B"]["confidence"] == 1.0

def test_insight_decay_integration():
    """Verify decay is called during boundary scan."""
    adapter = NetworkXAdapter(nx.DiGraph())
    adapter._G.add_edge("A", "B", relation=INSIGHT_RELATION, confidence=0.1)
    
    engine = InsightEngine(adapter)
    # scan_boundaries calls _decay_existing_insights
    engine.scan_boundaries()
    
    G = adapter.to_networkx()
    assert not G.has_edge("A", "B"), "Stale insight should have been pruned during scan"

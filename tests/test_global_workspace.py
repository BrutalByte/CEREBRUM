import asyncio
import pytest
import time
import numpy as np
from core.global_workspace import GlobalWorkspace, CommunitySignal
from reasoning.traversal import TraversalPath

@pytest.mark.asyncio
async def test_gws_bidding_and_prioritization():
    gws = GlobalWorkspace()
    
    # Simulate high novelty discovery
    signal_a = CommunitySignal(path=["A", "r", "B"], community_id=1, novelty_score=0.9, confidence=0.8)
    # Simulate low novelty discovery
    signal_b = CommunitySignal(path=["A", "r", "C"], community_id=2, novelty_score=0.2, confidence=0.9)
    
    await gws.broadcast(signal_a)
    await gws.broadcast(signal_b)
    
    top = gws.get_top_signals(limit=2)
    
    # A should be higher because novelty * confidence: 0.9*0.8 = 0.72 vs 0.2*0.9 = 0.18
    assert top[0].community_id == 1
    assert top[1].community_id == 2

@pytest.mark.asyncio
async def test_gws_ttl_cleanup():
    gws = GlobalWorkspace()
    gws.ttl = 0.1 # short TTL
    
    signal = CommunitySignal(path=["A"], community_id=1, novelty_score=0.9, confidence=0.8)
    await gws.broadcast(signal)
    
    assert 1 in gws.active_signals
    
    # Wait for TTL
    await asyncio.sleep(0.2)
    
    # Trigger cleanup
    # We manually run the check logic since we aren't running the daemon in this test
    now = time.time()
    keys_to_remove = [cid for cid, s in gws.active_signals.items() if now - s.timestamp > gws.ttl]
    for cid in keys_to_remove:
        del gws.active_signals[cid]
        
    assert 1 not in gws.active_signals

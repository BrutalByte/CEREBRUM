import pytest
import asyncio
from unittest.mock import MagicMock
from core.global_workspace import GlobalWorkspace
from core.bridge_engine import BridgeTwinEngine

@pytest.mark.asyncio
async def test_bridge_consensus_propagate():
    gws = GlobalWorkspace()
    bridge_engine = BridgeTwinEngine(node_id="test-node")
    
    received_proposals = []
    def callback(topic, proposal):
        received_proposals.append(proposal)

    gws.subscribe("consensus_bid", callback)
    
    proposal = {"target": "A->C", "bid": 0.95}
    bridge_engine.propagate_consensus("test-node", proposal, gws)
    
    await asyncio.sleep(0.1) # Wait for the task to run
    
    assert len(received_proposals) == 1
    assert received_proposals[0]["data"]["bid"] == 0.95
    assert received_proposals[0]["node_id"] == "test-node"

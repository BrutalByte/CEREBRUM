import pytest
import asyncio
from unittest.mock import MagicMock
from core.bridge_engine import BridgeTwinEngine

def test_bridge_mirroring():
    # Setup
    mock_adapter = MagicMock()
    engine = BridgeTwinEngine()
    
    # Simulate a peer proposal with high valence
    proposal = {
        "path": [("A", "bridge:similar", "C")],
        "valence": 0.9
    }
    
    # Observe
    engine.observe_peer(proposal, mock_adapter)
    
    # Verify local reinforcement
    assert mock_adapter.update_edge_weight.called
    args, kwargs = mock_adapter.update_edge_weight.call_args
    assert args == ("A", "C", "bridge:similar")
    assert kwargs["delta"] == 0.01

def test_bridge_ignore_low_valence():
    mock_adapter = MagicMock()
    engine = BridgeTwinEngine()
    
    # Low valence proposal
    proposal = {
        "path": [("A", "bridge:similar", "C")],
        "valence": 0.2
    }
    
    engine.observe_peer(proposal, mock_adapter)
    
    # Verify no local reinforcement
    assert not mock_adapter.update_edge_weight.called

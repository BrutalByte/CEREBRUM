"""
Tests for Phase 37 — Calibration Layer (Self-Doubt).
"""
import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock
from adapters.networkx_adapter import NetworkXAdapter
from core.calibration_engine import CalibrationEngine
from reasoning.traversal import BeamTraversal

def test_calibration_entropy_calculation():
    ce = CalibrationEngine(entropy_threshold=0.5)
    
    # Low entropy: one clear winner
    weights_certain = [0.9, 0.05, 0.05]
    res1 = ce.calibrate_hop(weights_certain, ["A", "B", "C"])
    assert res1.is_uncertain is False
    assert res1.confidence_multiplier > 0.8
    
    # High entropy: all equal
    weights_uncertain = [0.33, 0.33, 0.33]
    res2 = ce.calibrate_hop(weights_uncertain, ["A", "B", "C"])
    assert res2.is_uncertain is True
    assert res2.confidence_multiplier < 0.5

def test_traversal_with_self_doubt():
    G = nx.DiGraph()
    G.add_edge("Start", "A", relation="LINK")
    G.add_edge("Start", "B", relation="LINK")
    G.add_edge("Start", "C", relation="LINK")
    
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
    
    # Mock CSA to return equal weights for all
    csa = MagicMock()
    csa.use_temporal_decay = False
    csa.compute_weight.return_value = 0.5
    csa.compute_weight_with_features.return_value = (0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
    
    # Use low threshold to trigger doubt
    ce = CalibrationEngine(entropy_threshold=0.1)
    
    bt = BeamTraversal(adapter, csa, max_hop=1, calibration_engine=ce)
    
    paths = bt.traverse(["Start"])
    
    # Should have a doubt log entry
    assert len(bt.uncertainty_log) > 0
    assert bt.uncertainty_log[0]["hop"] == 1
    
    # Scores should be penalized (original score 1.0 * weight 0.5 * multiplier < 0.5)
    hop_paths = [p for p in paths if p.hop_depth == 1]
    for p in hop_paths:
        assert p.score < 0.5

"""
Tests for Phase 36 — Contradiction-Aware Consensus (Path Pruning).
"""
import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock
from adapters.networkx_adapter import NetworkXAdapter
from core.contradiction_engine import ContradictionEngine
from core.symbolic_engine import SymbolicValidator
from reasoning.traversal import BeamTraversal, TraversalPath

def test_internal_path_contradiction():
    engine = ContradictionEngine()
    
    # Path: A --[ACTIVATES]--> B --[INHIBITS]--> A
    # This is a circular contradiction.
    p = TraversalPath(
        nodes=["A", "ACTIVATES", "B", "INHIBITS", "A"],
        attention_weights=[0.8, 0.8]
    )
    
    recs = engine.detect_contradictions_in_path(p)
    assert len(recs) == 1
    assert recs[0].contradiction_type == "internal_path"
    assert recs[0].relation_a == "ACTIVATES"
    assert recs[0].relation_b == "INHIBITS"

def test_traversal_prunes_contradictory_paths():
    G = nx.DiGraph()
    G.add_node("A", type="chemical")
    G.add_node("B", type="protein")
    
    # Path A -> B -> A
    G.add_edge("A", "B", relation="ACTIVATES")
    G.add_edge("B", "A", relation="INHIBITS")
    
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {"A": np.zeros(64), "B": np.zeros(64)}
    
    ce = ContradictionEngine()
    validator = SymbolicValidator(adapter, contradiction_engine=ce)
    
    # Mock CSA
    csa = MagicMock()
    csa.use_temporal_decay = False
    csa.compute_weight.return_value = 0.8
    csa.compute_weight_with_features.return_value = (0.8, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
    
    bt = BeamTraversal(adapter, csa, max_hop=2, symbolic_validator=validator)
    
    # Traversal from A
    paths = bt.traverse(["A"])
    
    # depth 0: [A]
    # depth 1: [A, ACTIVATES, B]
    # depth 2: [A, ACTIVATES, B, INHIBITS, A] -> SHOULD BE PRUNED
    
    depth_2_paths = [p for p in paths if p.hop_depth == 2]
    assert len(depth_2_paths) == 0, "Contradictory 2-hop path should have been pruned"
    
    # Check 1-hop path exists
    depth_1_paths = [p for p in paths if p.hop_depth == 1]
    assert len(depth_1_paths) == 1
    assert depth_1_paths[0].tail == "B"

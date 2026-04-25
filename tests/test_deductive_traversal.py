import pytest
import networkx as nx
import numpy as np
from adapters.networkx_adapter import NetworkXAdapter
from core.symbolic_engine import SymbolicValidator
from reasoning.deductive_traversal import DeductiveTraversal

def test_deductive_path_finding():
    # Setup graph
    G = nx.DiGraph()
    G.add_edges_from([("A", "B", {"relation": "is_located_in"}), 
                      ("B", "C", {"relation": "is_located_in"})])
    
    adapter = NetworkXAdapter(G)
    validator = SymbolicValidator(adapter)
    traversal = DeductiveTraversal(adapter, validator)
    
    # Perform deductive traversal
    proofs = traversal.traverse("A", "C")
    
    # Verify we found the logical path
    assert len(proofs) == 1
    assert proofs[0] == ["A", "is_located_in", "B", "is_located_in", "C"]


# ---------------------------------------------------------------------------
# Phase 134: deductive consensus memoization
# ---------------------------------------------------------------------------

def test_deductive_cache_hit():
    """traverse() must be called at most once per (seed, entity_id) pair."""
    from reasoning.answer_extractor import deductive_consensus_rerank, Answer

    call_log = []
    real_G = nx.DiGraph()
    real_G.add_edges_from([("S", "A", {"relation": "causes"})])
    real_adapter = NetworkXAdapter(real_G)
    real_validator = SymbolicValidator(real_adapter)
    real_dt = DeductiveTraversal(real_adapter, real_validator)

    original_traverse = real_dt.traverse

    def tracked_traverse(seed, target, **kwargs):
        call_log.append((seed, target))
        return original_traverse(seed, target, **kwargs)

    real_dt.traverse = tracked_traverse

    def _ans(eid):
        return Answer(entity_id=eid, score=0.5, best_path=None)

    # Three answers, two with duplicate entity_id "A"
    answers = [_ans("A"), _ans("B"), _ans("A")]
    deductive_consensus_rerank(answers, seed="S", deductive_traversal=real_dt)

    a_calls = [c for c in call_log if c[1] == "A"]
    assert len(a_calls) == 1, f"Expected 1 call for 'A', got {len(a_calls)}"

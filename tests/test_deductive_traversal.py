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

"""
Tests for Phase 34 — Symbolic Guardrails (Hard Logic Validation).
"""
import pytest
import numpy as np
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from core.symbolic_engine import SymbolicValidator, IntegrityConstraint, ConstraintType
from core.attention_engine import CSAEngine
from reasoning.traversal import BeamTraversal

def _make_setup():
    G = nx.DiGraph()
    # A person
    G.add_node("Einstein", label="Einstein", type="person")
    # A location
    G.add_node("Ulm", label="Ulm", type="location")
    # Another person
    G.add_node("Newton", label="Newton", type="person")
    
    # Valid edge
    G.add_edge("Einstein", "Ulm", relation="BORN_IN")
    # Invalid edge (logically)
    G.add_edge("Einstein", "Newton", relation="BORN_IN")
    
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {
        "Einstein": np.zeros(64),
        "Ulm": np.zeros(64),
        "Newton": np.zeros(64)
    }
    
    validator = SymbolicValidator(adapter)
    # Rule: BORN_IN must be person -> location
    rule = IntegrityConstraint(
        ConstraintType.TYPE_RESTRICTION,
        params={"relation": "BORN_IN", "source_type": "person", "target_type": "location"}
    )
    validator.add_constraint(rule)
    
    return adapter, validator

def test_symbolic_validator_unit():
    adapter, validator = _make_setup()
    
    # Valid step: person -> BORN_IN -> location
    assert validator.validate_step("Einstein", "BORN_IN", "Ulm") is True
    
    # Invalid step: person -> BORN_IN -> person
    assert validator.validate_step("Einstein", "BORN_IN", "Newton") is False

def test_beam_traversal_with_guardrails():
    adapter, validator = _make_setup()
    csa = CSAEngine(adapter)
    
    # Traversal WITHOUT validator
    bt_no_val = BeamTraversal(adapter, csa, max_hop=1)
    paths_no_val = bt_no_val.traverse(["Einstein"])
    tails_no_val = {p.tail for p in paths_no_val}
    assert "Ulm" in tails_no_val
    assert "Newton" in tails_no_val # Newton is reached despite being logically wrong
    
    # Traversal WITH validator
    bt_val = BeamTraversal(adapter, csa, max_hop=1, symbolic_validator=validator)
    paths_val = bt_val.traverse(["Einstein"])
    tails_val = {p.tail for p in paths_val}
    
    assert "Ulm" in tails_val
    assert "Newton" not in tails_val, "Newton should have been pruned by symbolic guardrail"

def test_must_not_have_constraint():
    G = nx.DiGraph()
    G.add_node("A", type="person", is_place=True) # Bogus node
    G.add_edge("Einstein", "A", relation="KNOWS")
    
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {"Einstein": np.zeros(64), "A": np.zeros(64)}
    
    validator = SymbolicValidator(adapter)
    # Rule: A person cannot have is_place=True
    rule = IntegrityConstraint(
        ConstraintType.MUST_NOT_HAVE,
        params={"type": "person", "illegal_props": ["is_place"]}
    )
    validator.add_constraint(rule)
    
    assert validator.validate_step("Einstein", "KNOWS", "A") is False

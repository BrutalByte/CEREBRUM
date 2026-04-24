"""Tests for CounterfactualEngine (Phase 123)."""
import pytest
from unittest.mock import MagicMock, patch

from core.counterfactual_engine import (
    CounterfactualEngine,
    InterventionalAdapter,
    Intervention,
    CounterfactualResult,
    BLOCK_NODE, BLOCK_EDGE, ADD_EDGE,
)
from core.causal_engine import CAUSAL_RELATIONS


# ---------------------------------------------------------------------------
# Minimal synthetic adapter with causal edges
# ---------------------------------------------------------------------------

class _Edge:
    def __init__(self, target_id, relation_type, causal_weight=0.8):
        self.target_id = target_id
        self.relation_type = relation_type
        self.properties = {"causal_weight": causal_weight}


def _make_adapter(edges):
    """
    edges: list of (source, target, relation, weight) tuples.
    Graph: A -CAUSES-> B -CAUSES-> C   (2-hop chain)
           A -CAUSES-> C               (direct shortcut)
           D -CAUSES-> B               (D is a confounder for A->C)
    """
    graph = {}
    for src, tgt, rel, w in edges:
        graph.setdefault(src, []).append(_Edge(tgt, rel, w))

    adapter = MagicMock()
    adapter.get_neighbors.side_effect = lambda eid: iter(graph.get(eid, []))

    import networkx as nx
    G = nx.Graph()
    for src, tgt, rel, w in edges:
        G.add_edge(src, tgt, relation=rel, causal_weight=w)
    adapter.to_networkx.return_value = G
    return adapter


CAUSAL_EDGES = [
    ("A", "B", "CAUSES", 0.9),
    ("B", "C", "CAUSES", 0.8),
    ("A", "C", "CAUSES", 0.7),   # direct shortcut
    ("D", "B", "CAUSES", 0.6),   # confounder
]

NON_CAUSAL_EDGES = [
    ("X", "Y", "PEERS", 0.5),
    ("Y", "Z", "PEERS", 0.5),
]


# ---------------------------------------------------------------------------
# InterventionalAdapter
# ---------------------------------------------------------------------------

def test_block_node_removes_neighbor():
    adapter = _make_adapter(CAUSAL_EDGES)
    iv = Intervention(type=BLOCK_NODE, node="B")
    ia = InterventionalAdapter(adapter, [iv])
    neighbors = list(ia.get_neighbors("A"))
    targets = [n.target_id for n in neighbors]
    assert "B" not in targets
    assert "C" in targets  # direct A->C still present


def test_block_edge_removes_specific_edge():
    adapter = _make_adapter(CAUSAL_EDGES)
    iv = Intervention(type=BLOCK_EDGE, source="A", target="C", relation="CAUSES")
    ia = InterventionalAdapter(adapter, [iv])
    neighbors = list(ia.get_neighbors("A"))
    targets = [n.target_id for n in neighbors]
    assert "B" in targets    # A->B still present
    assert "C" not in targets


def test_block_edge_wildcard_relation():
    adapter = _make_adapter(CAUSAL_EDGES)
    iv = Intervention(type=BLOCK_EDGE, source="A", target="C", relation=None)
    iv2 = Intervention(type=BLOCK_EDGE, source="A", target="C")
    ia = InterventionalAdapter(adapter, [iv2])
    # wildcard blocks all relations on that pair
    neighbors = list(ia.get_neighbors("A"))
    targets = [n.target_id for n in neighbors]
    assert "C" not in targets


def test_add_edge_injects_hypothetical():
    adapter = _make_adapter(CAUSAL_EDGES)
    iv = Intervention(type=ADD_EDGE, source="D", target="C", relation="CAUSES", weight=0.5)
    ia = InterventionalAdapter(adapter, [iv])
    neighbors = list(ia.get_neighbors("D"))
    targets = [n.target_id for n in neighbors]
    assert "C" in targets


def test_add_edge_injected_node_inherits_block():
    adapter = _make_adapter(CAUSAL_EDGES)
    ivs = [
        Intervention(type=BLOCK_NODE, node="C"),
        Intervention(type=ADD_EDGE, source="D", target="C", relation="CAUSES", weight=0.5),
    ]
    ia = InterventionalAdapter(adapter, ivs)
    neighbors = list(ia.get_neighbors("D"))
    targets = [n.target_id for n in neighbors]
    # C is blocked so injected edge should be suppressed
    assert "C" not in targets


def test_to_networkx_removes_blocked_nodes():
    adapter = _make_adapter(CAUSAL_EDGES)
    iv = Intervention(type=BLOCK_NODE, node="B")
    ia = InterventionalAdapter(adapter, [iv])
    G = ia.to_networkx()
    assert "B" not in G.nodes


# ---------------------------------------------------------------------------
# CounterfactualEngine — factual baseline
# ---------------------------------------------------------------------------

def test_factual_effect_found_for_direct_path():
    adapter = _make_adapter([("A", "C", "CAUSES", 0.8)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    result = engine.query("A", "C", interventions=[], max_hop=2)
    assert result.factual_effect > 0
    assert len(result.factual_paths) > 0


def test_factual_effect_zero_for_no_causal_path():
    adapter = _make_adapter(NON_CAUSAL_EDGES)
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    result = engine.query("X", "Z", interventions=[], max_hop=3)
    assert result.factual_effect == 0.0
    assert result.all_paths_blocked is False  # no paths to block


# ---------------------------------------------------------------------------
# CounterfactualEngine — block_node intervention
# ---------------------------------------------------------------------------

def test_block_mediator_eliminates_mediated_path():
    # A -CAUSES-> B -CAUSES-> C, no direct A->C
    adapter = _make_adapter([
        ("A", "B", "CAUSES", 0.9),
        ("B", "C", "CAUSES", 0.8),
    ])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=BLOCK_NODE, node="B")
    result = engine.query("A", "C", interventions=[iv], max_hop=3)
    assert result.factual_effect > 0
    assert result.counterfactual_effect == 0.0
    assert result.all_paths_blocked is True
    assert result.effect_delta < 0


def test_block_mediator_leaves_direct_path():
    adapter = _make_adapter(CAUSAL_EDGES)  # A->B->C and A->C direct
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=BLOCK_NODE, node="B")
    result = engine.query("A", "C", interventions=[iv], max_hop=3)
    assert result.factual_effect > 0
    assert result.counterfactual_effect > 0   # A->C direct survives
    assert result.all_paths_blocked is False
    assert len(result.alternative_paths) > 0


# ---------------------------------------------------------------------------
# CounterfactualEngine — block_edge intervention
# ---------------------------------------------------------------------------

def test_block_direct_edge_leaves_mediated_path():
    adapter = _make_adapter(CAUSAL_EDGES)
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=BLOCK_EDGE, source="A", target="C", relation="CAUSES")
    result = engine.query("A", "C", interventions=[iv], max_hop=3)
    assert result.factual_effect > 0
    assert result.counterfactual_effect > 0   # A->B->C still works
    assert result.all_paths_blocked is False


def test_block_all_paths_sets_all_paths_blocked():
    adapter = _make_adapter([("A", "C", "CAUSES", 0.9)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=BLOCK_EDGE, source="A", target="C", relation="CAUSES")
    result = engine.query("A", "C", interventions=[iv], max_hop=2)
    assert result.counterfactual_effect == 0.0
    assert result.all_paths_blocked is True
    assert result.paths_blocked_count == 1


# ---------------------------------------------------------------------------
# CounterfactualEngine — add_edge intervention
# ---------------------------------------------------------------------------

def test_add_edge_creates_new_path():
    # No edge between D and C in the real graph
    adapter = _make_adapter([("A", "B", "CAUSES", 0.9)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=ADD_EDGE, source="D", target="C", relation="CAUSES", weight=0.7)
    result = engine.query("D", "C", interventions=[iv], max_hop=2)
    assert result.factual_effect == 0.0
    assert result.counterfactual_effect > 0
    assert result.effect_delta > 0


# ---------------------------------------------------------------------------
# effect_delta sign
# ---------------------------------------------------------------------------

def test_effect_delta_negative_when_intervention_weakens():
    adapter = _make_adapter([("A", "C", "CAUSES", 0.9)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    iv = Intervention(type=BLOCK_NODE, node="C")
    result = engine.query("A", "C", interventions=[iv], max_hop=2)
    assert result.effect_delta <= 0


def test_effect_delta_zero_when_no_intervention_effect():
    adapter = _make_adapter([("A", "C", "CAUSES", 0.9)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    # Block an unrelated node
    iv = Intervention(type=BLOCK_NODE, node="Z")
    result = engine.query("A", "C", interventions=[iv], max_hop=2)
    assert result.factual_effect == pytest.approx(result.counterfactual_effect, abs=1e-6)
    assert result.all_paths_blocked is False


# ---------------------------------------------------------------------------
# Intervention model
# ---------------------------------------------------------------------------

def test_intervention_roundtrip():
    iv = Intervention(type=BLOCK_EDGE, source="A", target="B", relation="CAUSES", weight=0.5)
    d = iv.to_dict()
    restored = Intervention.from_dict(d)
    assert restored.type == iv.type
    assert restored.source == iv.source
    assert restored.target == iv.target
    assert restored.relation == iv.relation


def test_intervention_to_dict_excludes_none():
    iv = Intervention(type=BLOCK_NODE, node="X")
    d = iv.to_dict()
    assert "source" not in d
    assert "target" not in d
    assert d["node"] == "X"


# ---------------------------------------------------------------------------
# CounterfactualResult serialisation
# ---------------------------------------------------------------------------

def test_result_to_dict_complete():
    adapter = _make_adapter([("A", "C", "CAUSES", 0.8)])
    engine = CounterfactualEngine(adapter, causal_relations=frozenset({"CAUSES"}))
    result = engine.query("A", "C", interventions=[], max_hop=2)
    d = result.to_dict()
    required = {
        "source", "target", "interventions",
        "factual_effect", "counterfactual_effect", "effect_delta",
        "factual_paths", "alternative_paths",
        "all_paths_blocked", "paths_blocked_count",
        "factual_confounders", "counterfactual_identification_method",
    }
    assert required.issubset(d.keys())


# ---------------------------------------------------------------------------
# Toy-graph integration: newton -> maxwell via faraday
# ---------------------------------------------------------------------------

def test_toy_graph_block_mediator():
    from adapters.file_adapter import load_file_adapter
    from pathlib import Path
    from benchmarks.causal_epistemic_benchmark import LANGUAGE_GRAPH_CAUSAL_PROXIES

    graph_path = Path(__file__).parent / "fixtures" / "toy_graph.csv"
    adapter = load_file_adapter(str(graph_path))
    engine = CounterfactualEngine(adapter, causal_relations=LANGUAGE_GRAPH_CAUSAL_PROXIES)

    # Factual: newton -> INFLUENCED -> faraday -> INFLUENCED -> maxwell
    result_factual = engine.query("newton", "maxwell", interventions=[], max_hop=3)
    assert result_factual.factual_effect > 0, "newton should reach maxwell via faraday"

    # Counterfactual: block faraday — the only mediator on this chain
    iv = Intervention(type=BLOCK_NODE, node="faraday")
    result_cf = engine.query("newton", "maxwell", interventions=[iv], max_hop=3)
    assert result_cf.counterfactual_effect == 0.0, \
        "blocking faraday should eliminate newton->maxwell causal path"
    assert result_cf.all_paths_blocked is True
    assert result_cf.effect_delta < 0

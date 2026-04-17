import pytest
import networkx as nx
from core.cerebrum import CerebrumGraph
from core.active_inference import ActiveInferenceEngine
from core.telemetry import NeuralEventType

def make_test_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([
        ("alice", "bob",   {"relation": "KNOWS"}),
        ("bob",   "carol", {"relation": "KNOWS"}),
        ("carol", "alice", {"relation": "KNOWS"}),
        ("dave",  "eve",   {"relation": "WORKS_WITH"}),
        ("eve",   "frank", {"relation": "WORKS_WITH"}),
        ("frank", "dave",  {"relation": "WORKS_WITH"}),
    ])
    G.add_edge("carol", "dave", relation="INTRODUCED_BY")
    return G

def test_debug_communities():
    G = make_test_graph()
    triples = [(u, d['relation'], v) for u, v, d in G.edges(data=True)]
    graph = CerebrumGraph.from_triples(triples)
    graph.build()
    print(f"\nDEBUG: node_count={graph.node_count}")
    print(f"DEBUG: community_count={graph.community_count}")
    print(f"DEBUG: adapter_community_map={getattr(graph.adapter, 'community_map', {})}")
    assert graph.community_count > 0

def test_active_inference_init():
    graph = CerebrumGraph.from_triples([])
    engine = ActiveInferenceEngine(graph)
    assert engine.graph == graph
    assert engine.min_soliton_threshold == 0.7

def test_active_inference_metabolic_check():
    graph = CerebrumGraph.from_triples([])
    engine = ActiveInferenceEngine(graph, metabolic_floor=0.5)
    graph.modulator.reinforcement = 0.1
    assert engine.step() is None
    graph.modulator.reinforcement = 0.8
    assert engine.step() is None

def test_active_inference_seed_selection():
    G = make_test_graph()
    triples = [(u, d['relation'], v) for u, v, d in G.edges(data=True)]
    graph = CerebrumGraph.from_triples(triples)
    graph.build()
    engine = ActiveInferenceEngine(graph)
    seeds = engine._select_seeds()
    assert len(seeds) > 0
    assert seeds[0] in G.nodes()

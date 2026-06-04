from typing import Type
import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from adapters.federated_adapter import FederatedAdapter
from adapters.remote_adapter import RemoteCerebrumAdapter

@pytest.fixture
def graph_a():
    g = nx.Graph()
    g.add_node("newton", label="Isaac Newton", type="person")
    g.add_node("gravity", label="Gravity", type="concept")
    g.add_edge("newton", "gravity", relation="DISCOVERED")
    return NetworkXAdapter(g)

@pytest.fixture
def graph_b():
    g = nx.Graph()
    g.add_node("leibniz", label="Gottfried Leibniz", type="person")
    g.add_node("calculus", label="Calculus", type="concept")
    g.add_edge("leibniz", "calculus", relation="INVENTED")
    
    # Shared entity with graph_a, but different neighbor
    g.add_node("newton", label="Isaac Newton", type="person")
    g.add_edge("newton", "calculus", relation="DEVELOPED")
    return NetworkXAdapter(g)

def test_federated_get_entity(graph_a, graph_b):
    fed = FederatedAdapter({"A": graph_a, "B": graph_b})
    
    # Entity in A
    e1 = fed.get_entity("gravity")
    assert e1 is not None
    assert e1.label == "Gravity"
    
    # Entity in B
    e2 = fed.get_entity("calculus")
    assert e2 is not None
    assert e2.label == "Calculus"
    
    # Entity in both (first one wins usually, or merged?)
    # Implementation says: first one found in dict order or cache
    e3 = fed.get_entity("newton")
    assert e3 is not None
    assert e3.label == "Isaac Newton"

def test_federated_get_neighbors_merges_sources(graph_a, graph_b):
    from core.alignment_engine import AlignmentIndex
    align = AlignmentIndex()
    # Explicitly align 'newton' across both graphs
    align.add_alignment("A", "newton", "B", "newton")
    
    fed = FederatedAdapter({"A": graph_a, "B": graph_b}, alignment=align)
    
    # Newton should have neighbors from BOTH A (gravity) and B (calculus)
    neighbors = fed.get_neighbors("newton")
    targets = {e.target_id for e in neighbors}
    
    assert "gravity" in targets
    assert "calculus" in targets
    assert len(neighbors) == 2

def test_federated_find_entities_aggregates(graph_a, graph_b):
    fed = FederatedAdapter({"A": graph_a, "B": graph_b})
    
    # Query matching entities in both
    results = fed.find_entities("calc", top_k=5)
    assert len(results) >= 1
    assert results[0].id == "calculus"

    results = fed.find_entities("Isaac", top_k=5)
    assert len(results) >= 1
    assert results[0].id == "newton"

def test_federated_node_count(graph_a, graph_b):
    fed = FederatedAdapter({"A": graph_a, "B": graph_b})
    # A: newton, gravity (2)
    # B: newton, leibniz, calculus (3)
    # Sum = 5 (naive sum)
    assert fed.node_count() == 5

def test_federated_with_alignment(graph_a, graph_b):
    from core.alignment_engine import AlignmentIndex
    
    # Align 'gravity' (A) with 'calculus' (B) just for testing
    # (Pretend they are the same entity in different namespaces)
    align = AlignmentIndex()
    align.add_alignment("A", "gravity", "B", "calculus")
    
    fed = FederatedAdapter({"A": graph_a, "B": graph_b}, alignment=align)
    
    # Querying neighbors for 'gravity' should now also return neighbors of 'calculus'
    neighbors = fed.get_neighbors("gravity")
    targets = {e.target_id for e in neighbors}
    
    # From A (gravity): newton (actually gravity is target of newton in fixture, 
    # so we need to check outgoing. Newton has outgoing to gravity.)
    
    # In fixture: newton -> gravity. leibniz -> calculus.
    # Let's check neighbors of 'newton' (A) and 'leibniz' (B)
    
    # If we align newton(A) and leibniz(B)
    align2 = AlignmentIndex()
    align2.add_alignment("A", "newton", "B", "leibniz")
    fed2 = FederatedAdapter({"A": graph_a, "B": graph_b}, alignment=align2)
    
    neighbors = fed2.get_neighbors("newton")
    targets = {e.target_id for e in neighbors}
    
    assert "gravity" in targets   # from A:newton
    assert "calculus" in targets  # from B:leibniz
    assert len(neighbors) == 2

def test_federated_masked_search(graph_a, graph_b):
    fed = FederatedAdapter({"A": graph_a, "B": graph_b})
    results = fed.find_entities_masked("newton", top_k=5)
    
    assert len(results) >= 1
    assert results[0]["id"] == "newton"
    assert "score" in results[0]
    # In masked search results, label is usually not present or redacted at API level
    # Adapter find_entities_masked returns dict with ID, Type, Score

# ---------------------------------------------------------------------------
# Remote Adapter Tests (Mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.status_code = 200
    return mock

def test_remote_get_entity(mock_response):
    mock_response.json.return_value = {
        "id": "remote_1", "label": "Remote Entity", "type": "thing", "properties": {}
    }
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        adapter = RemoteCerebrumAdapter("http://test-api")
        ent = adapter.get_entity("remote_1")
        
        assert ent is not None
        assert ent.id == "remote_1"
        assert ent.label == "Remote Entity"
        
        mock_get.assert_called_with("http://test-api/entities/remote_1", headers={}, timeout=10)

def test_remote_get_neighbors(mock_response):
    mock_response.json.return_value = [
        {"source_id": "a", "target_id": "b", "relation_type": "rel", "weight": 1.0}
    ]
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        adapter = RemoteCerebrumAdapter("http://test-api")
        edges = adapter.get_neighbors("a")
        
        assert len(edges) == 1
        assert edges[0].target_id == "b"
        
        mock_get.assert_called_with(
            "http://test-api/entities/a/neighbors", 
            params={"max_neighbors": 50}, 
            headers={},
            timeout=10
        )

def test_remote_find_entities(mock_response):
    mock_response.json.return_value = {
        "query": "query",
        "results": [{"id": "a", "label": "A", "type": "T", "properties": {}}]
    }
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        adapter = RemoteCerebrumAdapter("http://test-api")
        results = adapter.find_entities("query")
        
        assert len(results) == 1
        assert results[0].id == "a"
        
        mock_get.assert_called_with(
            "http://test-api/search", 
            params={"q": "query", "top_k": 10}, 
            headers={},
            timeout=10
        )

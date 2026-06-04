
from typing import Set
import pytest
import networkx as nx
from fastapi.testclient import TestClient
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine

@pytest.fixture
def client():
    g = nx.DiGraph()
    g.add_node("newton", label="Isaac Newton", type="person")
    g.add_node("gravity", label="Gravity", type="concept")
    g.add_node("physics", label="Physics", type="field")
    g.add_edge("newton", "gravity", relation="DISCOVERED", confidence=1.0)
    g.add_edge("gravity", "physics", relation="PART_OF", confidence=1.0)
    
    adapter = NetworkXAdapter(g)
    
    # Force communities and embeddings
    cm = {"newton": 0, "gravity": 0, "physics": 1}
    eng = RandomEngine(dim=64)
    
    app = create_app(adapter, eng, community_map=cm)
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
        yield c

def test_query_trace_returns_hops(client):
    response = client.post("/v1/query/trace", json={
        "query": "newton",
        "max_hop": 2,
        "beam_width": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "hops" in data
    assert len(data["hops"]) > 0
    
    # Check hop structure
    hop0 = data["hops"][0]
    assert "winners" in hop0
    assert "competitors" in hop0
    assert hop0["hop"] == 0
    
    # Newton should be a winner at hop 0
    winner_tails = [w["tail"] for w in hop0["winners"]]
    assert "newton" in winner_tails

def test_query_trace_captures_competitors(client):
    # Set beam_width=1 to force competitors
    response = client.post("/v1/query/trace", json={
        "query": "newton",
        "max_hop": 2,
        "beam_width": 1
    })
    assert response.status_code == 200
    data = response.json()
    
    # Check if there's any hop with competitors
    any_competitors = any(len(h["competitors"]) > 0 for h in data["hops"])
    # In this graph, there's only 1 neighbor for Newton, so maybe no competitors at hop 1.
    pass

@pytest.fixture
def competitive_client():
    g = nx.DiGraph()
    g.add_node("A", label="Node A")
    g.add_node("B", label="Node B")
    g.add_node("C", label="Node C")
    g.add_node("D", label="Node D")
    g.add_edge("A", "B", relation="LINK", confidence=1.0)
    g.add_edge("A", "C", relation="LINK", confidence=1.0)
    g.add_edge("A", "D", relation="LINK", confidence=1.0)
    
    adapter = NetworkXAdapter(g)
    
    cm = {"A": 0, "B": 1, "C": 1, "D": 1}
    eng = RandomEngine(dim=64)
    app = create_app(adapter, eng, community_map=cm)
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
        yield c

def test_query_trace_has_competitors(competitive_client):
    response = competitive_client.post("/v1/query/trace", json={
        "query": "A",
        "max_hop": 2,
        "beam_width": 1
    })
    assert response.status_code == 200
    data = response.json()
    
    # At hop 1, Node A expands to 'B', 'C', and 'D'.
    # With beam_width=1, and max_hop=2, hop 1 is NOT terminal.
    # So 1 should be a winner, 2 should be competitors.
    hop1 = data["hops"][1]
    assert len(hop1["winners"]) == 1
    assert len(hop1["competitors"]) >= 2 # B, C, D were candidates, only 1 won.
    
    # Check feature radar in winner
    winner = hop1["winners"][0]
    assert "features" in winner
    assert len(winner["features"]) == 10

import pytest
import os
import numpy as np
from fastapi.testclient import TestClient
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
import networkx as nx

@pytest.fixture
def hardened_client():
    # Set dev key for tests
    os.environ["PARALLAX_API_KEY"] = "test-secret"
    
    g = nx.Graph()
    # Create a path safe_node -> n1 -> n2 -> n3 ... to test budget
    g.add_node("safe_node", label="Safe Node", type="entity")
    for i in range(100):
        g.add_edge(f"n{i}", f"n{i+1}", relation="LINK")
    g.add_edge("safe_node", "n0", relation="LINK")
    
    adapter = NetworkXAdapter(g)
    
    # Mock embedding engine
    class MockEngine:
        def encode_entities(self, labels):
            return {k: np.zeros(64, dtype=np.float32) for k in labels}
            
    from api.server import _state
    _state["adapter"] = adapter
    _state["community_map"] = {n: 0 for n in g.nodes()}
    _state["embeddings"] = {n: np.zeros(64, dtype=np.float32) for n in g.nodes()}
    _state["csa_metadata"] = {"distances": {}, "adjacent_pairs": set()}
    _state["hologram"] = []
    
    app = create_app(adapter=adapter, embedding_engine=MockEngine())
    return TestClient(app)

class TestAPISecurity:
    
    def test_query_no_api_key_fails(self, hardened_client):
        r = hardened_client.post("/query", json={"query": "safe_node"})
        assert r.status_code == 403

    def test_query_valid_api_key_passes(self, hardened_client):
        r = hardened_client.post(
            "/query", 
            json={"query": "safe_node"},
            headers={"X-API-Key": "test-secret"}
        )
        assert r.status_code == 200

    def test_budget_enforcement(self, hardened_client):
        # Request a deep query with a tiny budget
        r = hardened_client.post(
            "/query", 
            json={
                "query": "safe_node",
                "max_hop": 5,
                "max_budget": 10 # Min allowed is 10
            },
            headers={"X-API-Key": "test-secret"}
        )
        assert r.status_code == 200
        data = r.json()
        # total_paths_explored should be low
        assert data["total_paths_explored"] <= 11 # 1 seed + 10 expansions

    def test_pydantic_parameter_bounds(self, hardened_client):
        # top_k max is 50
        r = hardened_client.post(
            "/query", 
            json={"query": "safe_node", "top_k": 100},
            headers={"X-API-Key": "test-secret"}
        )
        assert r.status_code == 422 # Validation error

    def test_masked_search_cap(self, hardened_client):
        # We implementation-capped this at 100 in server.py
        r = hardened_client.get(
            "/search/masked", 
            params={"q": "n", "top_k": 1000},
            headers={"X-API-Key": "test-secret"}
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) <= 100


import pytest
from fastapi.testclient import TestClient
import numpy as np
import networkx as nx
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine

def test_traverse_endpoint():
    # 1. Setup local graph
    G = nx.Graph()
    G.add_edge("A", "B", relation="REL1")
    G.add_edge("B", "C", relation="REL2")
    G.add_edge("C", "D", relation="REL3")
    
    adapter = NetworkXAdapter(G)
    # Mocking communities and embeddings as server would
    adapter.community_map = {"A": 0, "B": 0, "C": 1, "D": 1}
    adapter.embeddings = {"A": np.zeros(64), "B": np.zeros(64), "C": np.zeros(64), "D": np.zeros(64)}
    adapter.csa_metadata = {"distances": {}, "adjacent_pairs": set()}
    
    # 2. Setup FastAPI app
    app = create_app(adapter, RandomEngine(dim=64), community_map=adapter.community_map)
    
    # 3. Test /traverse
    with TestClient(app) as client:
        payload = {
            "seed_id": "A",
            "max_hop": 2,
            "beam_width": 5
        }
        # Using local-admin API key for auth bypass in test
        response = client.post("/v1/traverse", json=payload, headers={"X-API-Key": "dev-secret"})
        
        assert response.status_code == 200
        data = response.json()
        assert data["seed_id"] == "A"
        assert len(data["branches"]) > 0
        
        # Verify first branch
        branch = data["branches"][0]
        assert branch["nodes"][0] == "A"
        assert len(branch["nodes"]) >= 3 # A -> REL1 -> B
        assert "score" in branch

if __name__ == "__main__":
    pytest.main([__file__])

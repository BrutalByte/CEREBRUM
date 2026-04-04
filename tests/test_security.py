import pytest
import os
import numpy as np
from fastapi.testclient import TestClient
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
import networkx as nx

from core.security import FederatedAuth
from typing import List

@pytest.fixture
def hardened_client():
    os.environ["PARALLAX_SHARED_SECRET"] = "a-very-secret-and-long-enough-key-for-hs256"
    
    g = nx.Graph()
    g.add_node("safe_node")
    adapter = NetworkXAdapter(g)
    
    class MockEngine:
        def encode_entities(self, labels): return {k: np.zeros(64) for k in labels}
            
    from api.server import _state
    _state["adapter"] = adapter
    adapter.community_map = {n: 0 for n in g.nodes()}
    adapter.embeddings = {n: np.zeros(64) for n in g.nodes()}
    _state["community_map"] = adapter.community_map
    _state["embeddings"] = adapter.embeddings
    _state["csa_metadata"] = {"distances": {}, "adjacent_pairs": set()}
    _state["hologram"] = []
    
    app = create_app(adapter=adapter, embedding_engine=MockEngine())
    return TestClient(app)

def _get_auth_headers(scopes: List[str]):
    token = FederatedAuth.create_token("test-node", scopes=scopes)
    return {"Authorization": f"Bearer {token}"}

class TestJWTSecurity:
    def test_no_token(self, hardened_client):
        r = hardened_client.post("/query", json={"query": "safe_node"})
        assert r.status_code == 401
    
    def test_invalid_token(self, hardened_client):
        headers = {"Authorization": "Bearer invalid-token"}
        r = hardened_client.post("/query", json={"query": "safe_node"}, headers=headers)
        assert r.status_code == 401
        
    def test_wrong_scope(self, hardened_client):
        headers = _get_auth_headers(scopes=["search"])
        r = hardened_client.post("/query", json={"query": "safe_node"}, headers=headers)
        assert r.status_code == 403
        
    def test_valid_token_and_scope(self, hardened_client):
        headers = _get_auth_headers(scopes=["query"])
        r = hardened_client.post("/query", json={"query": "safe_node"}, headers=headers)
        assert r.status_code == 200

    def test_handshake_is_unprotected(self, hardened_client):
        # Handshake should be open to allow discovery
        r = hardened_client.get("/handshake")
        assert r.status_code == 200

    def test_health_is_protected(self, hardened_client):
        r = hardened_client.get("/health")
        assert r.status_code == 401

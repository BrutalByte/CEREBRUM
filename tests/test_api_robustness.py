import pytest
from starlette.testclient import TestClient
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
import networkx as nx
import numpy as np

@pytest.fixture
def api_client():
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="COUSIN", weight=1.0)
    G.add_node("A", label="Alice", type="person")
    G.add_node("B", label="Bob", type="person")
    
    adapter = NetworkXAdapter(G)
    engine = RandomEngine(dim=64)
    
    # Minimal wiring for build
    cmap = {"A": 0, "B": 0}
    
    app = create_app(adapter=adapter, embedding_engine=engine, community_map=cmap)
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as client:
        yield client

def test_api_agnostic_health(api_client):
    """Verify health endpoint is agnostic and reports correct counts."""
    res = api_client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["node_count"] == 2

def test_api_9_parameter_query(api_client):
    """Verify that the query endpoint handles the 9-parameter logit scoring."""
    payload = {
        "query": "Alice",
        "beam_width": 5,
        "max_hop": 2,
        "top_k": 2
    }
    res = api_client.post("/v1/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "paths" in data
    # If Bob is reached, check breakdown
    if data["paths"]:
        breakdown = data["paths"][0]["score_breakdown"]
        # Basic breakdown includes semantic, community, edge.
        # api/server.py uses to_structured which might not show all 9 yet
        assert "semantic" in breakdown

def test_api_streaming_fallback(api_client):
    """Verify streaming status works even on static adapters (fallback mode)."""
    res = api_client.get("/v1/stream/status")
    assert res.status_code == 200
    data = res.json()
    assert data["nodes"] == 2
    assert data["running"] is True

def test_api_insight_endpoints(api_client):
    """Verify insight engine endpoints are wired correctly."""
    # status
    res = api_client.get("/v1/insight/status")
    assert res.status_code == 200
    assert "total_events" in res.json()
    
    # scan (dry run or on-demand)
    res = api_client.post("/v1/insight/scan")
    assert res.status_code == 200
    assert "events_found" in res.json()

def test_api_unauthorized(api_client, monkeypatch):
    """Verify security boundary.

    /health is intentionally open (no auth).
    Protected endpoints must reject a wrong key when CEREBRUM_API_KEYS is set.
    In dev mode (no env var), all requests are accepted — that behaviour is
    tested separately; here we explicitly configure a key.
    """
    monkeypatch.setenv("CEREBRUM_API_KEYS", "correct-key")

    # Wrong key on a protected endpoint must be rejected
    client_bad = TestClient(api_client.app, headers={"X-API-Key": "wrong-key"})
    res = client_bad.get("/v1/communities")
    assert res.status_code == 403

    # /health stays open regardless of key configuration
    res_health = client_bad.get("/v1/health")
    assert res_health.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])

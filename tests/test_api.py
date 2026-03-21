"""
Phase 3 tests for the Parallax FastAPI REST server (api/server.py).

Uses FastAPI's TestClient (Starlette WSGI test adapter) — no running server
process required. All tests are fully in-process and deterministic.

Implementation notes
--------------------
1. TestClient lifespan: FastAPI lifespan events (which call _load()) only fire
   inside a `with TestClient(app) as client:` block. A bare TestClient(app)
   does NOT trigger lifespan, so state is never populated. All fixtures here
   use the context-manager form.

2. Global _state: api/server.py uses a single module-level _state dict shared
   across all app instances (design assumption: one app per process). Unloaded-
   behavior tests use the `unloaded_client` fixture, which saves/restores state
   around the test so the loaded client remains intact.

Endpoints under test:
  GET  /health       — readiness and component status
  POST /query        — multi-hop KG reasoning
  GET  /communities  — community partition metadata
"""
import random
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import api.server as _server_module
from adapters.csv_adapter import load_csv_adapter
from core.community_engine import best_of_n_dscf
from core.embedding_engine import RandomEngine
from api.server import create_app

TOY_CSV = Path(__file__).parent / "fixtures" / "toy_graph.csv"
SEED    = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped loaded client.  Built once, shared across all tests in this
    file. The `with` block supports the FastAPI lifespan fires and _load() runs.
    """
    random.seed(SEED)
    adapter = load_csv_adapter(str(TOY_CSV))
    engine  = RandomEngine(dim=64)

    G     = adapter.to_networkx()
    parts = best_of_n_dscf(G, n_trials=5, seed=SEED)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}

    app = create_app(adapter=adapter, embedding_engine=engine, community_map=cmap)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def unloaded_client():
    """
    Function-scoped client backed by an empty (no-adapter) app.

    Saves and clears the global _state before the test; restores it after.
    This isolates "service not ready" behavior without contaminating other tests.
    """
    saved = {k: v for k, v in _server_module._state.items()}
    for key in _server_module._state:
        _server_module._state[key] = None

    bare_app = create_app()
    with TestClient(bare_app, raise_server_exceptions=False) as c:
        yield c

    _server_module._state.update(saved)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok_when_loaded(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_adapter_loaded(self, client):
        r = client.get("/health")
        assert r.json()["adapter_loaded"] is True

    def test_health_communities_loaded(self, client):
        r = client.get("/health")
        assert r.json()["communities_loaded"] is True

    def test_health_embeddings_loaded(self, client):
        r = client.get("/health")
        assert r.json()["embeddings_loaded"] is True

    def test_health_node_count_matches_graph(self, client):
        """Reported node_count must match the actual fixture size (21)."""
        r = client.get("/health")
        assert r.json()["node_count"] == 21

    def test_health_community_count_positive(self, client):
        """At least one community must be reported."""
        r = client.get("/health")
        assert r.json()["community_count"] >= 1

    def test_health_unloaded_returns_loading(self, unloaded_client):
        r = unloaded_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "loading"


# ---------------------------------------------------------------------------
# GET /communities
# ---------------------------------------------------------------------------

class TestCommunities:

    def test_communities_returns_200(self, client):
        r = client.get("/communities")
        assert r.status_code == 200

    def test_communities_node_count_matches(self, client):
        r = client.get("/communities")
        assert r.json()["node_count"] == 21

    def test_communities_partition_is_complete(self, client):
        """node_to_community must contain an entry for every graph node."""
        r    = client.get("/communities")
        body = r.json()
        assert len(body["node_to_community"]) == body["node_count"]

    def test_communities_count_matches_map(self, client):
        """community_count must equal the number of distinct community IDs."""
        r        = client.get("/communities")
        body     = r.json()
        distinct = len(set(body["node_to_community"].values()))
        assert body["community_count"] == distinct

    def test_communities_list_length_matches_count(self, client):
        r    = client.get("/communities")
        body = r.json()
        assert len(body["communities"]) == body["community_count"]

    def test_communities_each_entry_has_required_fields(self, client):
        r = client.get("/communities")
        for comm in r.json()["communities"]:
            assert "community_id" in comm
            assert "size" in comm
            assert "sample_members" in comm

    def test_communities_sizes_sum_to_node_count(self, client):
        r     = client.get("/communities")
        body  = r.json()
        total = sum(c["size"] for c in body["communities"])
        assert total == body["node_count"]

    def test_communities_unloaded_returns_503(self, unloaded_client):
        r = unloaded_client.get("/communities")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:

    def test_query_with_explicit_seed_returns_200(self, client):
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        assert r.status_code == 200

    def test_query_response_contains_paths(self, client):
        r    = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        body = r.json()
        assert "paths" in body
        assert len(body["paths"]) > 0

    def test_query_returns_correct_query_echo(self, client):
        r    = client.post("/query", json={"query": "Who influenced Einstein?", "seeds": ["newton"]})
        body = r.json()
        assert body["query"] == "Who influenced Einstein?"

    def test_query_seeds_used_echoed(self, client):
        r    = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        body = r.json()
        assert "newton" in body["seeds_used"]

    def test_query_total_paths_explored_positive(self, client):
        r    = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        body = r.json()
        assert body["total_paths_explored"] > 0

    def test_query_path_ranks_are_sequential(self, client):
        """Ranks must start at 1 and increment by 1."""
        r     = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 5})
        paths = r.json()["paths"]
        ranks = [p["rank"] for p in paths]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_query_path_scores_are_descending(self, client):
        r      = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 10})
        scores = [p["score"] for p in r.json()["paths"]]
        assert scores == sorted(scores, reverse=True)

    def test_query_path_score_breakdown_present(self, client):
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        for p in r.json()["paths"]:
            assert "attention" in p["score_breakdown"]
            assert "community" in p["score_breakdown"]

    def test_query_path_nodes_alternate_entity_relation(self, client):
        """
        path nodes must alternate: entity, relation, entity, ...
        Even indices are entities (type='entity'), odd are relations (type='relation').
        """
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"]})
        for p in r.json()["paths"]:
            nodes = p["path"]
            for i, node in enumerate(nodes):
                if i % 2 == 0:
                    assert node["type"] == "entity", f"Expected entity at index {i}: {node}"
                else:
                    assert node["type"] == "relation", f"Expected relation at index {i}: {node}"

    def test_query_top_k_limits_results(self, client):
        r     = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 2})
        paths = r.json()["paths"]
        assert len(paths) <= 2

    def test_query_max_hop_one_returns_direct_neighbors(self, client):
        """At max_hop=1 the path length must be exactly 3 nodes (entity-rel-entity)."""
        r = client.post("/query", json={
            "query": "newton", "seeds": ["newton"], "max_hop": 1, "top_k": 10
        })
        for p in r.json()["paths"]:
            assert len(p["path"]) == 3, (
                f"max_hop=1 path should be 3 nodes, got {len(p['path'])}: {p['path']}"
            )

    def test_query_entity_grounding_finds_newton(self, client):
        """
        Text query 'newton' without explicit seeds must resolve to the newton
        entity via find_entities and return paths from it.
        """
        r    = client.post("/query", json={"query": "newton"})
        body = r.json()
        assert r.status_code == 200
        assert "newton" in body["seeds_used"]

    def test_query_unknown_explicit_seed_returns_empty_paths(self, client):
        """
        An explicit seed that doesn't exist in the graph is trusted by the server —
        no 404 is raised. Traversal finds no neighbors and returns 200 with an empty
        paths list. This is correct: the caller supplied the seed explicitly.
        """
        r    = client.post("/query", json={
            "query": "xyzzy_unknown",
            "seeds": ["xyzzy_unknown"],
        })
        assert r.status_code == 200
        body = r.json()
        assert body["paths"] == []

    def test_query_unresolvable_text_query_returns_404(self, client):
        """
        When no explicit seeds are given and find_entities() finds no matches,
        the server must return 404. This tests the entity-grounding failure path.
        """
        r = client.post("/query", json={
            "query": "xyzzy_absolutely_unknown_entity_zzz_no_match_at_all",
        })
        assert r.status_code == 404

    def test_query_invalid_top_k_returns_422(self, client):
        """top_k=0 violates the ge=1 constraint — must return 422 Unprocessable Entity."""
        r = client.post("/query", json={"query": "newton", "top_k": 0})
        assert r.status_code == 422

    def test_query_invalid_max_hop_returns_422(self, client):
        """max_hop=0 violates ge=1 — must return 422."""
        r = client.post("/query", json={"query": "newton", "max_hop": 0})
        assert r.status_code == 422

    def test_query_unloaded_returns_503(self, unloaded_client):
        r = unloaded_client.post(
            "/query", json={"query": "newton", "seeds": ["newton"]}
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# GET /entities, /search (Low-level Graph Access)
# ---------------------------------------------------------------------------

class TestGraphAccess:

    def test_get_entity_returns_200(self, client):
        r = client.get("/entities/newton")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "newton"
        assert data["label"] == "newton"

    def test_get_entity_not_found_returns_404(self, client):
        r = client.get("/entities/nonexistent_entity")
        assert r.status_code == 404

    def test_get_neighbors_returns_list(self, client):
        r = client.get("/entities/newton/neighbors")
        assert r.status_code == 200
        edges = r.json()
        assert isinstance(edges, list)
        assert len(edges) > 0
        # Check structure
        assert "source_id" in edges[0]
        assert "target_id" in edges[0]
        assert "relation_type" in edges[0]

    def test_search_returns_results(self, client):
        r = client.get("/search", params={"q": "newton", "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "newton"
        assert len(data["results"]) > 0
        assert data["results"][0]["id"] == "newton"

    def test_search_empty_returns_empty_list(self, client):
        r = client.get("/search", params={"q": "xyzzy_unknown", "top_k": 5})
        assert r.status_code == 200
        assert len(r.json()["results"]) == 0




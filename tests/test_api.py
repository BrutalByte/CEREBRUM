"""
Phase 3 tests for the CEREBRUM FastAPI REST server (api/server.py).

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
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
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
    with TestClient(bare_app, raise_server_exceptions=False, headers={"X-API-Key": "dev-secret"}) as c:
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

    def test_search_masked_returns_redacted(self, client):
        r = client.get("/search/masked", params={"q": "newton", "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0
        res = data["results"][0]
        assert res["id"] == "newton"
        assert res["label"] == "[REDACTED]"
        assert "score" in res


# ---------------------------------------------------------------------------
# POST /feedback, /search/similar
# ---------------------------------------------------------------------------

class TestAdaptiveLearning:

    def test_search_similar_returns_results(self, client):
        # 64-dim vector for toy graph RandomEngine
        vec = [0.1] * 64
        r = client.post("/search/similar", json={"embedding": vec, "top_k": 5})
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0
        assert "query_vector" in data

    def test_query_path_includes_edge_features(self, client):
        """Phase 46: /query paths must expose edge_features for /feedback use."""
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        assert r.status_code == 200
        path = r.json()["paths"][0]
        assert "edge_features" in path
        assert isinstance(path["edge_features"], list)
        # Each feature tuple should be a non-empty list of floats
        for feat in path["edge_features"]:
            assert isinstance(feat, list)
            assert len(feat) > 0

    def test_query_path_includes_community_sequence(self, client):
        """Phase 46: /query paths must expose community_sequence for /feedback use."""
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        assert r.status_code == 200
        path = r.json()["paths"][0]
        assert "community_sequence" in path
        assert isinstance(path["community_sequence"], list)
        assert len(path["community_sequence"]) > 0

    def test_feedback_uses_query_edge_features(self, client):
        """Phase 46: /feedback round-trip using edge_features from /query response."""
        # 1. Get a real path including edge_features
        r_query = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        assert r_query.status_code == 200
        path_data = r_query.json()["paths"][0]

        # 2. Use the actual edge_features and community_sequence from the query response
        entity_nodes = [n["label"] for n in path_data["path"] if n["type"] == "entity"]
        feedback_req = {
            "path_nodes": entity_nodes,
            "edge_features": path_data["edge_features"],
            "community_sequence": path_data["community_sequence"],
            "reward": 1.0,
        }

        r_fb = client.post("/feedback", json=feedback_req)
        assert r_fb.status_code == 200
        assert r_fb.json()["status"] == "success"

    def test_params_endpoint_returns_structure(self, client):
        """Phase 46: GET /params returns 10-param vector and community overrides."""
        r = client.get("/params")
        assert r.status_code == 200
        body = r.json()
        assert "param_names" in body
        assert "global_params" in body
        assert "community_count" in body
        assert "community_overrides" in body
        assert len(body["param_names"]) == 10
        assert len(body["global_params"]) == 10
        assert body["param_names"][0] == "alpha"
        assert body["param_names"][9] == "theta"

    def test_params_community_overrides_update_after_feedback(self, client):
        """Phase 46: After /feedback, /params should show at least one community override."""
        # 1. Get a path with community_sequence
        r_query = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        path_data = r_query.json()["paths"][0]

        # 2. Send feedback for a community that exists
        cseq = path_data["community_sequence"]
        if not cseq:
            return  # skip if no community info (e.g. trivial graph)

        feedback_req = {
            "path_nodes": [n["label"] for n in path_data["path"] if n["type"] == "entity"],
            "edge_features": path_data["edge_features"],
            "community_sequence": cseq,
            "reward": 1.0,
        }
        client.post("/feedback", json=feedback_req)

        # 3. /params should now show at least one community override
        r_params = client.get("/params")
        assert r_params.status_code == 200
        body = r_params.json()
        assert body["community_count"] >= 1
        assert len(body["community_overrides"]) >= 1

    def test_feedback_unloaded_returns_501(self, unloaded_client):
        # Unloaded client doesn't have meta_learner initialized
        r = unloaded_client.post("/feedback", json={
            "path_nodes": ["A", "B"],
            "edge_features": [[0.5, 0.5, 0.5, 0.5, 0.5]],
            "community_sequence": [0, 0],
            "reward": 1.0,
        })
        assert r.status_code == 501

    def test_post_params_restores_global_prior(self, client):
        """Phase 47: POST /params replaces the global prior and returns it."""
        custom = [0.1, 0.2, 0.3, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.5]
        r = client.post("/params", json={"global_prior": custom, "community_overrides": {}})
        assert r.status_code == 200
        body = r.json()
        assert body["global_params"] == pytest.approx(custom, abs=1e-5)
        assert body["community_count"] == 0

    def test_post_params_restores_community_overrides(self, client):
        """Phase 47: POST /params restores per-community overrides."""
        custom_prior = [0.4] * 10
        custom_overrides = {"5": [0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        r = client.post("/params", json={
            "global_prior": custom_prior,
            "community_overrides": custom_overrides,
        })
        assert r.status_code == 200
        body = r.json()
        assert "5" in body["community_overrides"]
        assert body["community_overrides"]["5"] == pytest.approx(custom_overrides["5"], abs=1e-5)

    def test_post_params_export_import_roundtrip(self, client):
        """Phase 47: GET /params → POST /params produces identical state."""
        # 1. Send some feedback to create community overrides
        r_query = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        path_data = r_query.json()["paths"][0]
        client.post("/feedback", json={
            "path_nodes": [n["label"] for n in path_data["path"] if n["type"] == "entity"],
            "edge_features": path_data["edge_features"],
            "community_sequence": path_data["community_sequence"],
            "reward": 1.0,
        })

        # 2. Export
        r_export = client.get("/params")
        exported = r_export.json()

        # 3. Reset by posting default params
        from core.parameter_learner import _DEFAULT_INIT
        client.post("/params", json={"global_prior": list(_DEFAULT_INIT), "community_overrides": {}})
        assert client.get("/params").json()["community_count"] == 0

        # 4. Re-import exported state
        r_import = client.post("/params", json={
            "global_prior": exported["global_params"],
            "community_overrides": exported["community_overrides"],
        })
        assert r_import.status_code == 200
        reimported = r_import.json()
        assert reimported["community_count"] == exported["community_count"]

    def test_post_params_wrong_length_returns_422(self, client):
        """Phase 47: POST /params with wrong-length vector returns 422."""
        r = client.post("/params", json={"global_prior": [0.5, 0.5, 0.5], "community_overrides": {}})
        assert r.status_code == 422

    # ------------------------------------------------------------------
    # Phase 48: Auto-Retrain Scheduler
    # ------------------------------------------------------------------

    def _send_feedback(self, client, reward: float):
        """Helper: get a query path and send feedback with given reward."""
        r = client.post("/query", json={"query": "newton", "seeds": ["newton"], "top_k": 1})
        path_data = r.json()["paths"][0]
        client.post("/feedback", json={
            "path_nodes": [n["label"] for n in path_data["path"] if n["type"] == "entity"],
            "edge_features": path_data["edge_features"],
            "community_sequence": path_data["community_sequence"],
            "reward": reward,
        })

    def test_retrain_requires_mixed_feedback(self, client):
        """Phase 48: /retrain with only positive feedback returns 422."""
        # Reset buffer
        from core.parameter_learner import _DEFAULT_INIT
        client.post("/params", json={"global_prior": list(_DEFAULT_INIT), "community_overrides": {}})
        # Send only positive feedback
        self._send_feedback(client, reward=1.0)
        r = client.post("/retrain")
        assert r.status_code == 422

    def test_retrain_with_mixed_feedback(self, client):
        """Phase 48: /retrain succeeds when buffer has both pos and neg items."""
        self._send_feedback(client, reward=1.0)
        self._send_feedback(client, reward=-1.0)
        r = client.post("/retrain")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["pairs_used"] >= 1
        assert "learned_params" in body
        assert set(body["learned_params"].keys()) == {
            "alpha", "beta", "gamma", "delta", "epsilon",
            "zeta", "eta", "iota", "mu", "theta",
        }

    def test_retrain_updates_global_prior(self, client):
        """Phase 48: /retrain learned_params are reflected in GET /params global_params."""
        self._send_feedback(client, reward=1.0)
        self._send_feedback(client, reward=-1.0)
        r = client.post("/retrain", json={"max_iterations": 50, "learning_rate": 0.1})
        assert r.status_code == 200
        learned = r.json()["learned_params"]

        # GET /params should reflect the new global prior
        params_body = client.get("/params").json()
        for i, name in enumerate(params_body["param_names"]):
            assert params_body["global_params"][i] == pytest.approx(learned[name], abs=1e-4)

    def test_retrain_clears_buffer_by_default(self, client):
        """Phase 48: buffer_remaining == 0 after default retrain."""
        self._send_feedback(client, reward=1.0)
        self._send_feedback(client, reward=-1.0)
        r = client.post("/retrain")
        assert r.json()["buffer_remaining"] == 0

    def test_retrain_keep_buffer(self, client):
        """Phase 48: buffer_remaining > 0 when clear_buffer=False."""
        self._send_feedback(client, reward=1.0)
        self._send_feedback(client, reward=-1.0)
        r = client.post("/retrain", json={"clear_buffer": False})
        assert r.status_code == 200
        assert r.json()["buffer_remaining"] >= 2




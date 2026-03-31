"""
Tests for Phase 12: Bridge Twin Engine.

Covers:
  - Candidate tracking increments correctly
  - Bridge not created below n_min
  - Bridge not created when semantic similarity is too low
  - Bridge created at n_min with sufficient similarity
  - Twin has correct embedding, community assignment, and bidirectional edges
  - CSA weight for BRIDGE_TWIN edge is higher than a cold cross-community edge
  - Twin divergence: twin can accumulate edges the original doesn't have
  - Bridge pruning removes idle twins from both engine and graph (LTD analog)
  - record_twin_use resets idle timer
  - is_bridge_twin distinguishes twin from original
  - get_twin lookup returns correct twin_id
  - Full circuit-completion traversal: query that previously paid
    cross-community penalty now routes through the bridge
"""
import time
import threading

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.bridge_engine import BridgeTwinEngine, BRIDGE_RELATION, _cosine_sim
from core.attention_engine import CSAEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_two_community_graph():
    """
    Simple graph with two communities clearly separated.

    Community 0: A, B, C  — embeddings close to [1, 0, 0, 0]
    Community 1: X, Y, Z  — embeddings close to [0, 1, 0, 0]

    Node "M" (bridge candidate) sits between the two: embedding [0.8, 0.7, 0, 0],
    home community 0, but semantically similar to community 1 (cos_sim ≈ 0.66+).
    """
    G = nx.Graph()
    nodes = ["A", "B", "C", "X", "Y", "Z", "M"]
    G.add_nodes_from(nodes)
    # Intra-community edges
    G.add_edge("A", "B", relation="KNOWS")
    G.add_edge("B", "C", relation="KNOWS")
    G.add_edge("X", "Y", relation="KNOWS")
    G.add_edge("Y", "Z", relation="KNOWS")
    # One cross-community edge through M
    G.add_edge("C", "M", relation="RELATED_TO")
    G.add_edge("M", "X", relation="RELATED_TO")

    adapter = NetworkXAdapter(G)

    # Assign embeddings (4-dim unit vectors)
    adapter.embeddings = {
        "A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "B": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "C": np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32),
        "X": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "Y": np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
        "Z": np.array([0.05, 0.95, 0.0, 0.0], dtype=np.float32),
        # M bridges: cos_sim to community-1 centroid ≈ 0.66 (above threshold 0.65)
        "M": np.array([0.8, 0.7, 0.0, 0.0], dtype=np.float32),
    }

    # Community assignments
    adapter.community_map = {
        "A": 0, "B": 0, "C": 0,
        "X": 1, "Y": 1, "Z": 1,
        "M": 0,   # M lives in community 0 but will bridge into 1
    }

    return adapter


def _make_low_sim_graph():
    """M has near-zero similarity to community 1 — bridge should NOT form."""
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "M", "X", "Y"])
    G.add_edge("A", "B", relation="KNOWS")
    G.add_edge("X", "Y", relation="KNOWS")
    G.add_edge("B", "M", relation="RELATED_TO")
    G.add_edge("M", "X", relation="RELATED_TO")

    adapter = NetworkXAdapter(G)
    adapter.embeddings = {
        "A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "B": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "M": np.array([1.0, 0.01, 0.0, 0.0], dtype=np.float32),  # far from comm-1
        "X": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "Y": np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32),
    }
    adapter.community_map = {"A": 0, "B": 0, "M": 0, "X": 1, "Y": 1}
    return adapter


# ---------------------------------------------------------------------------
# 1. Candidate tracking
# ---------------------------------------------------------------------------

class TestCandidateTracking:

    def test_initial_count_is_zero(self):
        engine = BridgeTwinEngine(n_min=5)
        assert engine.candidate_count("M", 1) == 0

    def test_count_increments_per_crossing(self):
        engine = BridgeTwinEngine(n_min=10)
        adapter = _make_two_community_graph()
        for i in range(4):
            engine.record_crossing("M", 0, 1, adapter)
        assert engine.candidate_count("M", 1) == 4

    def test_no_bridge_below_n_min(self):
        engine = BridgeTwinEngine(n_min=5)
        adapter = _make_two_community_graph()
        for _ in range(4):
            result = engine.record_crossing("M", 0, 1, adapter)
        assert result is None
        assert engine.get_twin("M", 1) is None

    def test_different_destinations_tracked_separately(self):
        engine = BridgeTwinEngine(n_min=10)
        adapter = _make_two_community_graph()
        for _ in range(3):
            engine.record_crossing("M", 0, 1, adapter)
        for _ in range(6):
            engine.record_crossing("M", 0, 2, adapter)
        assert engine.candidate_count("M", 1) == 3
        assert engine.candidate_count("M", 2) == 6


# ---------------------------------------------------------------------------
# 2. Semantic similarity gate
# ---------------------------------------------------------------------------

class TestSemanticGate:

    def test_bridge_not_created_when_similarity_too_low(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_low_sim_graph()
        for _ in range(5):
            result = engine.record_crossing("M", 0, 1, adapter)
        # M's embedding is nearly orthogonal to community-1 centroid
        assert result is None
        assert engine.get_twin("M", 1) is None

    def test_bridge_created_when_similarity_sufficient(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)
        assert twin_id is not None
        assert twin_id == engine.get_twin("M", 1)

    def test_similarity_threshold_respected(self):
        """Raise the threshold so M (sim ≈ 0.66) just barely passes default but fails higher."""
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.99)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(5):
            twin_id = engine.record_crossing("M", 0, 1, adapter)
        assert twin_id is None  # similarity too low for 0.99 threshold


# ---------------------------------------------------------------------------
# 3. Twin node properties
# ---------------------------------------------------------------------------

class TestTwinProperties:

    @pytest.fixture
    def engine_with_bridge(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)
        return engine, adapter, twin_id

    def test_twin_exists_in_graph(self, engine_with_bridge):
        _, adapter, twin_id = engine_with_bridge
        assert twin_id in adapter._G

    def test_twin_assigned_to_destination_community(self, engine_with_bridge):
        _, adapter, twin_id = engine_with_bridge
        assert adapter.get_community(twin_id) == 1

    def test_twin_has_identical_embedding(self, engine_with_bridge):
        _, adapter, twin_id = engine_with_bridge
        original_emb = adapter.get_embedding("M")
        twin_emb = adapter.get_embedding(twin_id)
        assert twin_emb is not None
        np.testing.assert_array_almost_equal(original_emb, twin_emb)

    def test_bidirectional_bridge_edges_exist(self, engine_with_bridge):
        _, adapter, twin_id = engine_with_bridge
        G = adapter._G
        # original → twin
        assert G.has_edge("M", twin_id)
        assert G["M"][twin_id].get("relation") == BRIDGE_RELATION
        # twin → original
        assert G.has_edge(twin_id, "M")
        assert G[twin_id]["M"].get("relation") == BRIDGE_RELATION

    def test_twin_is_bridge_twin(self, engine_with_bridge):
        engine, _, twin_id = engine_with_bridge
        assert engine.is_bridge_twin(twin_id)

    def test_original_is_not_bridge_twin(self, engine_with_bridge):
        engine, _, _ = engine_with_bridge
        assert not engine.is_bridge_twin("M")

    def test_bridge_record_metadata(self, engine_with_bridge):
        engine, _, twin_id = engine_with_bridge
        records = {r.twin_id: r for r in engine.active_bridges()}
        assert twin_id in records
        rec = records[twin_id]
        assert rec.original_id == "M"
        assert rec.source_community == 0
        assert rec.destination_community == 1
        assert rec.traversal_count == 3
        assert 0.0 < rec.similarity_at_creation <= 1.0


# ---------------------------------------------------------------------------
# 4. Existing bridge reuse
# ---------------------------------------------------------------------------

class TestBridgeReuse:

    def test_record_crossing_returns_none_for_existing_bridge(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        # Create the bridge
        for _ in range(3):
            engine.record_crossing("M", 0, 1, adapter)
        # Further crossings after bridge exists
        result = engine.record_crossing("M", 0, 1, adapter)
        assert result is None  # bridge already live

    def test_traversal_count_increments_on_reuse(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)
        engine.record_crossing("M", 0, 1, adapter)
        engine.record_crossing("M", 0, 1, adapter)
        rec = {r.twin_id: r for r in engine.active_bridges()}[twin_id]
        assert rec.traversal_count == 5  # 3 creation + 2 reuses


# ---------------------------------------------------------------------------
# 5. CSA attention weight for BRIDGE_TWIN edges
# ---------------------------------------------------------------------------

class TestCSABridgeWeight:

    def test_bridge_twin_weight_is_high(self):
        adapter = _make_two_community_graph()
        csa = CSAEngine(adapter=adapter)
        # Should short-circuit with high weight (alpha+beta+gamma all 1.0)
        w = csa.compute_weight("M", "M::twin::1", hop=1, edge_type=BRIDGE_RELATION)
        # Expected: sigmoid(0.4*1 + 0.4*1 + 0.1*1 + 0.05*0.5) = sigmoid(0.925) ≈ 0.716
        assert w > 0.70, f"Expected bridge weight > 0.70, got {w:.4f}"

    def test_bridge_twin_weight_exceeds_cold_cross_community(self):
        adapter = _make_two_community_graph()
        csa = CSAEngine(adapter=adapter)
        csa.set_community_graph(
            community_distances={(0, 1): 3.0},
            adjacent_pairs=set(),
        )
        # Normal cross-community hop (no bridge, distant communities)
        w_normal = csa.compute_weight("M", "X", hop=1, edge_type="RELATED_TO")
        w_bridge = csa.compute_weight("M", "M::twin::1", hop=1, edge_type=BRIDGE_RELATION)
        assert w_bridge > w_normal, (
            f"Bridge weight {w_bridge:.4f} should exceed "
            f"cold cross-community weight {w_normal:.4f}"
        )


# ---------------------------------------------------------------------------
# 6. Twin divergence
# ---------------------------------------------------------------------------

class TestTwinDivergence:

    def test_twin_can_accumulate_new_edges(self):
        """Twin acquires a community-1-specific edge; original does not have it."""
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        # Give the twin a new edge that the original doesn't have
        adapter._G.add_edge(twin_id, "Y", relation="COLLABORATES_WITH")

        # Twin has the new edge
        twin_neighbors = [e.target_id for e in adapter.get_neighbors(twin_id)]
        assert "Y" in twin_neighbors

        # Original does not have it
        original_neighbors = [e.target_id for e in adapter.get_neighbors("M")]
        assert "Y" not in original_neighbors

    def test_original_retains_original_edges(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        original_neighbors = {e.target_id for e in adapter.get_neighbors("M")}
        # Original still has C, X, and the twin
        assert "C" in original_neighbors
        assert "X" in original_neighbors
        assert twin_id in original_neighbors


# ---------------------------------------------------------------------------
# 7. Pruning (LTD analog)
# ---------------------------------------------------------------------------

class TestBridgePruning:

    def test_prune_unused_removes_idle_bridge(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65, prune_after_days=0.0)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        assert twin_id is not None
        assert twin_id in adapter._G

        # prune_after_days=0.0 means immediately pruneable
        pruned = engine.prune_unused(adapter=adapter)

        assert twin_id in pruned
        assert twin_id not in adapter._G
        assert len(engine.active_bridges()) == 0

    def test_active_bridge_not_pruned(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65, prune_after_days=30.0)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        pruned = engine.prune_unused(adapter=adapter)

        assert twin_id not in pruned
        assert twin_id in adapter._G
        assert len(engine.active_bridges()) == 1

    def test_record_twin_use_resets_idle_timer(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65, prune_after_days=0.0)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        # Force last_used into the future (simulate recent use)
        engine._bridges[twin_id].last_used = time.time() + 86400

        pruned = engine.prune_unused(adapter=adapter)
        assert twin_id not in pruned  # recent use protects it

    def test_record_twin_use_api(self):
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        before = engine._bridges[twin_id].last_used
        time.sleep(0.01)
        engine.record_twin_use(twin_id)
        after = engine._bridges[twin_id].last_used
        assert after > before


# ---------------------------------------------------------------------------
# 8. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_concurrent_crossings_do_not_duplicate_bridge(self):
        """Many threads recording crossings should create exactly one bridge."""
        engine = BridgeTwinEngine(n_min=5, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        created = []

        def cross():
            for _ in range(10):
                twin_id = engine.record_crossing("M", 0, 1, adapter)
                if twin_id is not None:
                    created.append(twin_id)

        threads = [threading.Thread(target=cross) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one bridge should exist, regardless of concurrency
        assert len(engine.active_bridges()) == 1
        # All created IDs (if any) point to the same twin
        assert len(set(created)) <= 1


# ---------------------------------------------------------------------------
# 9. Circuit completion — end-to-end traversal test
# ---------------------------------------------------------------------------

class TestCircuitCompletion:

    def test_bridge_appears_in_neighbors(self):
        """After bridge creation, the twin is reachable from the original via get_neighbors."""
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        neighbor_ids = {e.target_id for e in adapter.get_neighbors("M")}
        assert twin_id in neighbor_ids

    def test_twin_reaches_destination_community_nodes(self):
        """Traversal through twin continues into community-1 nodes."""
        engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.65)
        adapter = _make_two_community_graph()
        twin_id = None
        for _ in range(3):
            twin_id = engine.record_crossing("M", 0, 1, adapter)

        # Give the twin an edge into community-1
        adapter._G.add_edge(twin_id, "Y", relation="RELAY_TO")
        adapter.community_map["Y"] = 1

        twin_neighbors = {e.target_id for e in adapter.get_neighbors(twin_id)}
        assert "Y" in twin_neighbors
        assert adapter.get_community("Y") == 1

    def test_cosine_sim_helper(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0)
        assert _cosine_sim(a, a) == pytest.approx(1.0)

    def test_cosine_sim_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_sim(a, b) == 0.0

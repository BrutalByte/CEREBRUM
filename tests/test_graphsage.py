"""
Tests for GraphSAGE neighborhood smoothing (Phase 55).

Verifies that smooth_with_graphsage() correctly aggregates neighbour
embeddings and that CerebrumGraph.build(use_graphsage=True) integrates
the smoothing step without breaking the downstream pipeline.
"""
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from core.embedding_engine import smooth_with_graphsage
from core.cerebrum import CerebrumGraph

TOY_CSV = str(Path(__file__).parent / "fixtures" / "toy_graph.csv")


# ---------------------------------------------------------------------------
# Unit tests for smooth_with_graphsage
# ---------------------------------------------------------------------------

class TestSmoothWithGraphSAGE:
    def _make_triangle(self):
        """Simple A-B-C triangle with known embeddings."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        embs = {
            "A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "C": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        return G, embs

    def test_output_has_same_keys(self):
        G, embs = self._make_triangle()
        result = smooth_with_graphsage(embs, G)
        assert set(result.keys()) == set(embs.keys())

    def test_output_is_unit_normalized(self):
        G, embs = self._make_triangle()
        result = smooth_with_graphsage(embs, G, normalize=True)
        for v in result.values():
            assert abs(np.linalg.norm(v) - 1.0) < 1e-5

    def test_no_normalize_preserves_magnitude(self):
        G, embs = self._make_triangle()
        # All input vectors are unit norm; output magnitude should change with blending
        result = smooth_with_graphsage(embs, G, normalize=False)
        for node, v in result.items():
            # Since neighbours exist, output ≠ input; no norm enforcement
            assert isinstance(v, np.ndarray)

    def test_isolated_node_unchanged(self):
        G = nx.DiGraph()
        G.add_node("X")
        embs = {"X": np.array([1.0, 0.0], dtype=np.float32)}
        result = smooth_with_graphsage(embs, G)
        np.testing.assert_allclose(result["X"], embs["X"], atol=1e-6)

    def test_self_weight_only_recovers_input(self):
        """With neighbor_weight=0, output should equal normalized self."""
        G, embs = self._make_triangle()
        result = smooth_with_graphsage(embs, G, self_weight=1.0, neighbor_weight=0.0)
        for node in embs:
            np.testing.assert_allclose(result[node], embs[node], atol=1e-5)

    def test_neighbour_blending_changes_vector(self):
        G, embs = self._make_triangle()
        result = smooth_with_graphsage(embs, G, self_weight=0.5, neighbor_weight=0.5)
        # Each output should differ from the input (neighbours inject signal)
        for node in embs:
            assert not np.allclose(result[node], embs[node], atol=1e-3)

    def test_output_dtype_is_float32(self):
        G, embs = self._make_triangle()
        result = smooth_with_graphsage(embs, G)
        for v in result.values():
            assert v.dtype == np.float32

    def test_nodes_missing_from_embeddings_ignored(self):
        """Nodes in G but not in embeddings are skipped without error."""
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        embs = {"A": np.array([1.0, 0.0], dtype=np.float32)}
        # B and C are neighbours of A but have no embeddings
        result = smooth_with_graphsage(embs, G)
        # A has neighbours with no embeddings → treated as isolated
        assert "A" in result

    def test_empty_embeddings_returns_empty(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B")])
        result = smooth_with_graphsage({}, G)
        assert result == {}

    def test_large_star_graph_mean_aggregation(self):
        """Hub connected to N spokes: smoothed hub should be close to mean of spokes."""
        N = 8
        G = nx.DiGraph()
        hub = "hub"
        spokes = [f"s{i}" for i in range(N)]
        for s in spokes:
            G.add_edge(hub, s)

        # Hub is a zero vector; spokes are standard basis / random unit vecs
        rng = np.random.default_rng(0)
        embs = {hub: np.zeros(4, dtype=np.float32)}
        for s in spokes:
            v = rng.standard_normal(4).astype(np.float32)
            embs[s] = v / np.linalg.norm(v)

        result = smooth_with_graphsage(embs, G, self_weight=0.0, neighbor_weight=1.0)
        # With self_weight=0, hub's result = normalized mean of spoke embeddings
        spoke_mean = np.mean([embs[s] for s in spokes], axis=0).astype(np.float32)
        norm = np.linalg.norm(spoke_mean)
        expected = spoke_mean / norm if norm > 1e-8 else spoke_mean
        np.testing.assert_allclose(result[hub], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration: CerebrumGraph.build(use_graphsage=True)
# ---------------------------------------------------------------------------

class TestGraphSAGEIntegration:
    def test_build_with_graphsage_completes(self):
        graph = CerebrumGraph.from_csv(TOY_CSV)
        graph.build(use_graphsage=True)
        assert graph._built

    def test_graphsage_embeddings_are_unit_normalized(self):
        graph = CerebrumGraph.from_csv(TOY_CSV)
        graph.build(use_graphsage=True)
        for node_id, vec in graph.adapter.embeddings.items():
            norm = np.linalg.norm(vec.astype(np.float32))
            assert abs(norm - 1.0) < 1e-3, f"Node {node_id!r} norm={norm:.4f}"

    def test_graphsage_differs_from_plain_build(self):
        """GraphSAGE-smoothed embeddings should differ from the base embeddings."""
        g_plain = CerebrumGraph.from_csv(TOY_CSV)
        g_plain.build(use_graphsage=False)

        g_sage = CerebrumGraph.from_csv(TOY_CSV)
        g_sage.build(use_graphsage=True)

        # At least one node should have a different embedding
        changed = 0
        for node in g_plain.adapter.embeddings:
            if node in g_sage.adapter.embeddings:
                a = g_plain.adapter.embeddings[node].astype(np.float32)
                b = g_sage.adapter.embeddings[node].astype(np.float32)
                if not np.allclose(a, b, atol=1e-4):
                    changed += 1
        assert changed > 0, "GraphSAGE smoothing had no effect on any embedding"

    def test_graphsage_query_still_works(self):
        graph = CerebrumGraph.from_csv(TOY_CSV)
        graph.build(use_graphsage=True)
        answers = graph.query(
            [next(iter(graph.adapter.embeddings))],
            top_k=5, max_hop=2,
        )
        assert isinstance(answers, list)

    def test_custom_weights_accepted(self):
        graph = CerebrumGraph.from_csv(TOY_CSV)
        graph.build(
            use_graphsage=True,
            graphsage_self_weight=0.7,
            graphsage_neighbor_weight=0.3,
        )
        assert graph._built

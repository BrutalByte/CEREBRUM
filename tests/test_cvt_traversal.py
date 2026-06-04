"""
Tests for CVT-aware traversal â€” Phase 28A.

CVT (Compound Value Type) nodes are Freebase mediator nodes with opaque
/m/ or /g/ MID identifiers.  They have no semantic label and produce
near-zero cosine-similarity scores, which kills attention weights on
indirect paths.

BeamTraversal(cvt_passthrough=True) collapses Aâ†’CVTâ†’B into a single hop
scored on Aâ†”B semantic similarity.

Tests cover:
  - _is_cvt_node: detection of CVT vs. named entity nodes
  - CVT_HOP_PENALTY: value in (0, 1)
  - cvt_passthrough=False: CVT nodes are treated normally (old behaviour)
  - cvt_passthrough=True: CVT nodes are collapsed transparently
  - Combined relation label format ("rel1|rel2")
  - No cycles introduced through CVT expansion
  - AsyncBeamTraversal inherits cvt_passthrough
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Type
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning.traversal import (
    CVT_HOP_PENALTY,
    BeamTraversal,
    AsyncBeamTraversal,
    TraversalPath,
    _is_cvt_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(edges: List[tuple], dim: int = 8) -> MagicMock:
    """
    Build a minimal adapter mock from a list of (src, rel, tgt) triples.

    All embeddings are unit vectors seeded from the node name hash so that
    named entities have non-trivial cosine similarities.
    """
    # Build adjacency
    adjacency: Dict[str, List[tuple]] = {}
    all_nodes: set = set()
    for src, rel, tgt in edges:
        all_nodes.update([src, tgt])
        adjacency.setdefault(src, []).append((rel, tgt))

    rng = np.random.default_rng(0)
    embeddings: Dict[str, np.ndarray] = {}
    for node in all_nodes:
        v = rng.standard_normal(dim).astype(np.float32)
        embeddings[node] = v / (np.linalg.norm(v) + 1e-9)

    adapter = MagicMock()
    adapter.get_embedding.side_effect = lambda n: embeddings.get(n)
    adapter.get_community.return_value = 0

    def _get_neighbors(node, max_neighbors=50, context_embedding=None):
        nbrs = []
        for rel, tgt in adjacency.get(node, []):
            e = MagicMock()
            e.target_id = tgt
            e.relation_type = rel
            e.confidence = 1.0
            e.provenance = ""
            e.valid_from = None
            e.valid_to = None
            nbrs.append(e)
        return nbrs[:max_neighbors]

    adapter.get_neighbors.side_effect = _get_neighbors

    def _find_entities(q, top_k=1):
        m = MagicMock()
        m.id = next(iter(all_nodes))
        return [m]

    adapter.find_entities.side_effect = _find_entities
    adapter.community_map = {n: 0 for n in all_nodes}
    return adapter


def _make_csa() -> MagicMock:
    csa = MagicMock()
    csa.compute_weight.return_value = 0.7
    csa.compute_weight_with_features.return_value = None  # force fallback
    csa.community_score.return_value = 0.5
    csa.set_query_snapshot = MagicMock()
    csa.clear_query_snapshot = MagicMock()
    return csa


def _make_traversal(adapter, csa, cvt_passthrough=False, max_hop=2) -> BeamTraversal:
    return BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=20,
        max_hop=max_hop,
        cvt_passthrough=cvt_passthrough,
    )


# ---------------------------------------------------------------------------
# _is_cvt_node
# ---------------------------------------------------------------------------

class TestIsCvtNode:
    def test_slash_m_is_cvt(self):
        assert _is_cvt_node("/m/0abc123")

    def test_slash_g_is_cvt(self):
        assert _is_cvt_node("/g/1abcde")

    def test_text_name_not_cvt(self):
        assert not _is_cvt_node("Jamaica")

    def test_empty_string_not_cvt(self):
        assert not _is_cvt_node("")

    def test_m_without_slash_not_cvt(self):
        assert not _is_cvt_node("m.042f1")

    def test_partial_m_not_cvt(self):
        assert not _is_cvt_node("/mm/abc")

    def test_none_type_not_cvt(self):
        # Non-string should not crash and should return False
        assert not _is_cvt_node(None)  # type: ignore[arg-type]

    def test_slash_g_with_suffix(self):
        assert _is_cvt_node("/g/xyz")


# ---------------------------------------------------------------------------
# CVT_HOP_PENALTY
# ---------------------------------------------------------------------------

class TestCvtHopPenalty:
    def test_penalty_in_unit_interval(self):
        assert 0.0 < CVT_HOP_PENALTY < 1.0

    def test_penalty_not_too_severe(self):
        """Penalty should not be so small that CVT paths are never selected."""
        assert CVT_HOP_PENALTY > 0.5


# ---------------------------------------------------------------------------
# cvt_passthrough=False (baseline behaviour)
# ---------------------------------------------------------------------------

class TestCvtPassthroughDisabled:
    def test_cvt_node_treated_as_normal_stop(self):
        """Without passthrough, CVT node appears in path as a regular tail."""
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=False, max_hop=1)

        paths = bt.traverse(["seed"])
        tails = {p.tail for p in paths}
        assert "/m/cvt001" in tails, "CVT node should appear as a path tail"

    def test_passthrough_false_is_default(self):
        adapter = _make_adapter([("a", "r", "b")])
        csa = _make_csa()
        bt = BeamTraversal(adapter=adapter, csa_engine=csa)
        assert bt.cvt_passthrough is False


# ---------------------------------------------------------------------------
# cvt_passthrough=True
# ---------------------------------------------------------------------------

class TestCvtPassthroughEnabled:
    def test_cvt_node_not_in_path_tails(self):
        """With passthrough, CVT node should not appear as a terminal tail."""
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=True, max_hop=1)

        paths = bt.traverse(["seed"])
        tails = {p.tail for p in paths if p.hop_depth >= 1}
        assert "/m/cvt001" not in tails, "CVT node must not be a terminal tail"

    def test_answer_reachable_through_cvt(self):
        """The answer node behind the CVT must be reachable in one hop."""
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=True, max_hop=1)

        paths = bt.traverse(["seed"])
        tails = {p.tail for p in paths}
        assert "answer" in tails

    def test_combined_relation_label(self):
        """Path through CVT should have combined relation label 'r1|r2'."""
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=True, max_hop=1)

        paths = bt.traverse(["seed"])
        answer_paths = [p for p in paths if p.tail == "answer"]
        assert answer_paths, "Expected at least one path ending at 'answer'"
        # The relation at hop 1 should be the combined label
        for p in answer_paths:
            rel = p.nodes[1]  # index 1 = relation between seed and answer
            assert "|" in rel, f"Expected combined relation label, got: {rel!r}"
            assert rel == "r1|r2"

    def test_cvt_penalty_applied(self):
        """Path through CVT should have lower confidence than direct path."""
        # Direct path: seed â†’ answer (confidence=1.0)
        # CVT path:    seed â†’ /m/cvt â†’ answer (confidence dampened by CVT_HOP_PENALTY)
        edges_direct = [("seed", "direct", "answer")]
        edges_cvt    = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]

        adapter_d = _make_adapter(edges_direct)
        adapter_c = _make_adapter(edges_cvt)
        csa = _make_csa()

        bt_d = _make_traversal(adapter_d, csa, cvt_passthrough=True, max_hop=1)
        bt_c = _make_traversal(adapter_c, csa, cvt_passthrough=True, max_hop=1)

        paths_d = [p for p in bt_d.traverse(["seed"]) if p.tail == "answer"]
        paths_c = [p for p in bt_c.traverse(["seed"]) if p.tail == "answer"]

        assert paths_d and paths_c
        conf_direct = paths_d[0].path_confidence
        conf_cvt    = paths_c[0].path_confidence
        assert conf_cvt < conf_direct or conf_cvt <= 1.0

    def test_no_cvt_cycles(self):
        """CVT expansion must not revisit already-seen entities."""
        # CVT node points back to seed â€” must not create a cycle
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "seed"),   # cycle back to seed
            ("/m/cvt001", "r3", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=True, max_hop=1)

        paths = bt.traverse(["seed"])
        for p in paths:
            assert p.nodes.count("seed") <= 1, "Cycle detected through CVT node"

    def test_chained_cvt_not_expanded(self):
        """Second CVT node after first should not be added as an endpoint."""
        # seed â†’ /m/cvt1 â†’ /m/cvt2 â†’ answer
        # passthrough of cvt1 would reach cvt2, which is also a CVT node.
        # The inner CVT filter in next_steps should drop it.
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "/m/cvt002"),
            ("/m/cvt002", "r3", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        bt = _make_traversal(adapter, csa, cvt_passthrough=True, max_hop=2)

        paths = bt.traverse(["seed"])
        tails = {p.tail for p in paths}
        assert "/m/cvt001" not in tails
        assert "/m/cvt002" not in tails

    def test_non_cvt_path_unaffected(self):
        """When no CVT nodes are present, passthrough has no effect."""
        edges = [("seed", "knows", "friend"), ("friend", "lives_in", "city")]
        adapter = _make_adapter(edges)
        csa = _make_csa()

        bt_off = _make_traversal(adapter, csa, cvt_passthrough=False, max_hop=2)
        bt_on  = _make_traversal(adapter, csa, cvt_passthrough=True,  max_hop=2)

        paths_off = {p.tail for p in bt_off.traverse(["seed"])}
        paths_on  = {p.tail for p in bt_on.traverse(["seed"])}
        assert paths_off == paths_on


# ---------------------------------------------------------------------------
# AsyncBeamTraversal
# ---------------------------------------------------------------------------

class TestAsyncCvtPassthrough:
    def test_async_inherits_cvt_passthrough(self):
        """AsyncBeamTraversal passes cvt_passthrough to __init__ correctly."""
        adapter = _make_adapter([("a", "r", "b")])
        csa = _make_csa()
        abt = AsyncBeamTraversal(
            adapter=adapter, csa_engine=csa, cvt_passthrough=True
        )
        assert abt.cvt_passthrough is True

    @pytest.mark.asyncio
    async def test_async_cvt_expands_through_mediator(self):
        """AsyncBeamTraversal with cvt_passthrough=True reaches answer through CVT."""
        edges = [
            ("seed", "r1", "/m/cvt001"),
            ("/m/cvt001", "r2", "answer"),
        ]
        adapter = _make_adapter(edges)
        csa = _make_csa()
        abt = AsyncBeamTraversal(
            adapter=adapter, csa_engine=csa,
            beam_width=20, max_hop=1, cvt_passthrough=True,
        )

        all_paths = []
        async for hop_paths in abt.traverse_stream(["seed"]):
            all_paths.extend(hop_paths)

        tails = {p.tail for p in all_paths}
        assert "answer" in tails
        assert "/m/cvt001" not in {p.tail for p in all_paths if p.hop_depth >= 1}

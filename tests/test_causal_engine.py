"""Tests for Phase 120 — Causal Inference Engine."""
import pytest
import networkx as nx
import numpy as np
from adapters.networkx_adapter import NetworkXAdapter
from core.causal_engine import CausalEngine, CausalProof, CAUSAL_RELATIONS
from core.truth_cache import TruthCache
from core.symbolic_engine import SymbolicValidator, IntegrityConstraint, ConstraintType


def _make_causal_graph():
    """Smoke graph: A -[CAUSES]-> B -[CAUSES]-> C; D -[RELATED_TO]-> C."""
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="CAUSES", weight=1.0, causal_weight=0.9)
    G.add_edge("B", "C", relation="CAUSES", weight=1.0, causal_weight=0.8)
    G.add_edge("D", "C", relation="RELATED_TO", weight=1.0)
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
    return adapter


def _make_confounder_graph():
    """X causes both A and B (X is a confounder for A->B query)."""
    G = nx.DiGraph()
    G.add_edge("X", "A", relation="CAUSES", weight=1.0, causal_weight=0.85)
    G.add_edge("X", "B", relation="CAUSES", weight=1.0, causal_weight=0.80)
    G.add_edge("A", "B", relation="CAUSES", weight=1.0, causal_weight=0.75)
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
    return adapter


def _make_temporal_graph():
    """A -> B with valid_from=100, B -> C with valid_from=50 (reverse temporal order)."""
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="CAUSES", weight=1.0, causal_weight=0.9, valid_from=100.0)
    G.add_edge("B", "C", relation="CAUSES", weight=1.0, causal_weight=0.9, valid_from=50.0)
    adapter = NetworkXAdapter(G)
    adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
    return adapter


class TestCausalRelations:
    def test_causal_relations_set(self):
        assert "CAUSES" in CAUSAL_RELATIONS
        assert "ACTIVATES" in CAUSAL_RELATIONS
        assert "RELATED_TO" not in CAUSAL_RELATIONS


class TestCausalEngine:
    def test_causal_path_found(self):
        adapter = _make_causal_graph()
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        assert proof.effect_estimate > 0.0
        assert len(proof.direct_paths) > 0
        assert proof.confidence > 0.0
        assert proof.identification_method in ("direct", "mediated", "noisy_or")

    def test_no_path_between_unconnected(self):
        adapter = _make_causal_graph()
        engine = CausalEngine(adapter)
        proof = engine.query("D", "A", max_hop=3)
        assert proof.effect_estimate == 0.0
        assert proof.identification_method == "none"
        assert proof.direct_paths == []

    def test_filters_correlational_edges(self):
        adapter = _make_causal_graph()
        engine = CausalEngine(adapter)
        # D->C is only RELATED_TO, not CAUSES — should not appear in causal paths
        proof = engine.query("D", "C", max_hop=2)
        assert proof.effect_estimate == 0.0
        for path in proof.direct_paths:
            relations = path[1::2]
            for rel in relations:
                assert rel in CAUSAL_RELATIONS

    def test_confounder_detected(self):
        adapter = _make_confounder_graph()
        engine = CausalEngine(adapter, max_backdoor_hops=2)
        proof = engine.query("A", "B", max_hop=2)
        # X is a common ancestor of both A and B
        assert "X" in proof.confounders_detected
        assert proof.is_confounded is True

    def test_confounder_reduces_confidence(self):
        adapter = _make_confounder_graph()
        engine = CausalEngine(adapter, max_backdoor_hops=2)
        proof = engine.query("A", "B", max_hop=2)
        # Confounded proof should have confidence < effect_estimate
        if proof.effect_estimate > 0:
            assert proof.confidence <= proof.effect_estimate

    def test_temporal_ordering_valid(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="CAUSES", causal_weight=0.9, valid_from=10.0)
        G.add_edge("B", "C", relation="CAUSES", causal_weight=0.9, valid_from=20.0)
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        assert proof.temporal_valid is True

    def test_temporal_ordering_violated(self):
        adapter = _make_temporal_graph()
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        # B->C has valid_from=50 which is before A->B's valid_from=100
        assert proof.temporal_valid is False

    def test_temporal_violation_reduces_confidence(self):
        adapter = _make_temporal_graph()
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        if proof.effect_estimate > 0:
            assert proof.confidence < proof.effect_estimate

    def test_direct_path_method(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="CAUSES", causal_weight=0.9)
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
        engine = CausalEngine(adapter)
        proof = engine.query("A", "B", max_hop=2)
        assert proof.identification_method == "direct"

    def test_noisy_or_multiple_paths(self):
        G = nx.DiGraph()
        G.add_edge("A", "C", relation="CAUSES", causal_weight=0.6)
        G.add_edge("A", "B", relation="CAUSES", causal_weight=0.7)
        G.add_edge("B", "C", relation="CAUSES", causal_weight=0.8)
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        # Multiple paths → effect should be higher than any single path
        assert proof.effect_estimate > 0.6

    def test_causal_proof_cached(self):
        adapter = _make_causal_graph()
        tc = TruthCache()
        engine = CausalEngine(adapter, truth_cache=tc)

        proof1 = engine.query("A", "C", max_hop=3)
        assert proof1.cached is False

        proof2 = engine.query("A", "C", max_hop=3)
        assert proof2.cached is True
        assert proof2.effect_estimate == proof1.effect_estimate

    def test_causal_proof_to_dict(self):
        adapter = _make_causal_graph()
        engine = CausalEngine(adapter)
        proof = engine.query("A", "C", max_hop=3)
        d = proof.to_dict()
        assert "source" in d
        assert "target" in d
        assert "effect_estimate" in d
        assert "is_confounded" in d
        assert "confidence" in d


class TestTruthCacheCausal:
    def test_store_and_retrieve(self):
        tc = TruthCache()
        proof = CausalProof(
            source="A", target="B",
            effect_estimate=0.8, direct_paths=[["A", "CAUSES", "B"]],
            confounders_detected=[], is_confounded=False,
            temporal_valid=True, causal_relations_used=["CAUSES"],
            confidence=0.8, identification_method="direct",
        )
        tc.store_causal_proof("A", "B", proof)
        retrieved = tc.get_causal_proof("A", "B")
        assert retrieved is not None
        assert retrieved.source == "A"
        assert retrieved.effect_estimate == 0.8

    def test_miss_returns_none(self):
        tc = TruthCache()
        assert tc.get_causal_proof("X", "Y") is None

    def test_clear_causal(self):
        tc = TruthCache()
        proof = CausalProof(
            source="A", target="B", effect_estimate=0.5, direct_paths=[],
            confounders_detected=[], is_confounded=False, temporal_valid=True,
            causal_relations_used=[], confidence=0.5, identification_method="none",
        )
        tc.store_causal_proof("A", "B", proof)
        tc.clear_causal()
        assert tc.get_causal_proof("A", "B") is None


class TestSymbolicCausalConstraints:
    def test_causal_ordering_constraint(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="CAUSES", valid_from=50.0)
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {"A": np.zeros(64), "B": np.zeros(64)}

        validator = SymbolicValidator(adapter)
        rule = IntegrityConstraint(
            ConstraintType.CAUSAL_ORDERING,
            params={"last_timestamp": 100.0}
        )
        validator.add_constraint(rule)

        # A->B has valid_from=50 but last_timestamp=100 → temporal violation
        assert validator.validate_step("A", "CAUSES", "B") is False

    def test_no_backdoor_constraint(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", relation="CAUSES")
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {"A": np.zeros(64), "B": np.zeros(64)}

        validator = SymbolicValidator(adapter)
        rule = IntegrityConstraint(
            ConstraintType.NO_BACKDOOR,
            params={"confounders": {"B"}}
        )
        validator.add_constraint(rule)

        assert validator.validate_step("A", "CAUSES", "B") is False
        assert validator.validate_step("A", "CAUSES", "C") is True


class TestDeductiveCausalOnly:
    def test_causal_only_filters_noncausal(self):
        from reasoning.deductive_traversal import DeductiveTraversal
        from core.symbolic_engine import SymbolicValidator

        G = nx.DiGraph()
        G.add_edge("A", "B", relation="CAUSES")
        G.add_edge("A", "C", relation="RELATED_TO")
        adapter = NetworkXAdapter(G)
        adapter.embeddings = {n: np.zeros(64) for n in G.nodes()}

        validator = SymbolicValidator(adapter)
        dt = DeductiveTraversal(adapter, validator)

        paths = dt.traverse("A", "C", causal_only=True)
        assert paths == []  # RELATED_TO not in CAUSAL_RELATIONS

        paths_all = dt.traverse("A", "C", causal_only=False)
        assert len(paths_all) > 0

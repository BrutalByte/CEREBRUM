"""
Tests for ProvenanceLedger (Phase 76).

Coverage:
  - record_batch: new entry, idempotent re-record, LRU eviction
  - rollback_batch: removes edges, marks rolled_back, raises on unknown
  - rollback_batch: warns but continues when single edge removal fails
  - rollback_batch: no-op on already-rolled-back batch
  - rollback_cycle: removes all batches for a given cycle
  - rollback_cycle: skips already-rolled-back batches
  - rollback_batch / rollback_cycle: raises NotImplementedError when no remove_edge
  - list_batches: ordered newest-first, respects n
  - get_batch: found and not found
  - stats: correct counts
  - ResearchAgent.set_provenance_ledger: wires correctly
  - ResearchAgent.approve: records to ledger when attached
  - ResearchAgent.approve: cycle_number forwarded to ledger
  - AutonomousDiscoveryLoop: passes cycle_number to approve
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from core.provenance_ledger import BatchRecord, EdgeRecord, ProvenanceLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(fail_edges=None):
    """Return a mock adapter; fail_edges is a set of (u,v,r) that raise on remove."""
    adapter = MagicMock()
    fail_set = set(fail_edges or [])

    def _remove(u, v, relation):
        if (u, v, relation) in fail_set:
            raise ValueError("edge not found")

    adapter.remove_edge.side_effect = _remove
    return adapter


def _edges(*triples):
    return list(triples)


# ---------------------------------------------------------------------------
# record_batch
# ---------------------------------------------------------------------------

class TestRecordBatch:
    def test_creates_record(self):
        ledger = ProvenanceLedger()
        rec = ledger.record_batch("b-1", "f-1", [("a", "b", "knows")])
        assert rec.batch_id == "b-1"
        assert rec.finding_id == "f-1"
        assert len(rec.edges) == 1
        assert rec.edges[0] == EdgeRecord("a", "b", "knows")

    def test_idempotent_re_record(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "knows")])
        rec2 = ledger.record_batch("b-1", "f-1", [("a", "b", "knows"), ("c", "d", "likes")])
        assert len(rec2.edges) == 2
        assert ledger.stats()["total_batches"] == 1

    def test_cycle_number_stored(self):
        ledger = ProvenanceLedger()
        rec = ledger.record_batch("b-1", "f-1", [("a", "b", "r")], cycle_number=7)
        assert rec.cycle_number == 7

    def test_lru_eviction(self):
        ledger = ProvenanceLedger(max_batches=3)
        for i in range(4):
            ledger.record_batch(f"b-{i}", f"f-{i}", [("a", "b", "r")])
        s = ledger.stats()
        assert s["total_batches"] == 3
        # Oldest (b-0) should be evicted
        assert ledger.get_batch("b-0") is None
        assert ledger.get_batch("b-3") is not None


# ---------------------------------------------------------------------------
# rollback_batch
# ---------------------------------------------------------------------------

class TestRollbackBatch:
    def test_removes_edges(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "knows"), ("c", "d", "likes")])
        adapter = _make_adapter()
        removed = ledger.rollback_batch("b-1", adapter)
        assert removed == 2
        assert adapter.remove_edge.call_count == 2

    def test_marks_rolled_back(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")])
        ledger.rollback_batch("b-1", _make_adapter())
        assert ledger.get_batch("b-1").rolled_back is True

    def test_no_op_on_already_rolled_back(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")])
        adapter = _make_adapter()
        ledger.rollback_batch("b-1", adapter)
        removed2 = ledger.rollback_batch("b-1", adapter)
        assert removed2 == 0
        assert adapter.remove_edge.call_count == 1  # only first rollback

    def test_raises_on_unknown_batch(self):
        ledger = ProvenanceLedger()
        with pytest.raises(ValueError, match="not found"):
            ledger.rollback_batch("nonexistent", _make_adapter())

    def test_raises_when_adapter_lacks_remove_edge(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")])
        adapter = MagicMock(spec=[])  # no remove_edge
        with pytest.raises(NotImplementedError):
            ledger.rollback_batch("b-1", adapter)

    def test_continues_on_failed_edge_removal(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r"), ("c", "d", "s")])
        # First edge removal will fail
        adapter = _make_adapter(fail_edges={("a", "b", "r")})
        removed = ledger.rollback_batch("b-1", adapter)
        # Second edge should still be attempted
        assert adapter.remove_edge.call_count == 2
        assert removed == 1  # only the non-failing edge counts


# ---------------------------------------------------------------------------
# rollback_cycle
# ---------------------------------------------------------------------------

class TestRollbackCycle:
    def test_removes_all_batches_in_cycle(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")], cycle_number=3)
        ledger.record_batch("b-2", "f-2", [("c", "d", "s")], cycle_number=3)
        ledger.record_batch("b-3", "f-3", [("e", "f", "t")], cycle_number=4)
        adapter = _make_adapter()
        removed = ledger.rollback_cycle(3, adapter)
        assert removed == 2
        assert ledger.get_batch("b-1").rolled_back is True
        assert ledger.get_batch("b-2").rolled_back is True
        assert not ledger.get_batch("b-3").rolled_back

    def test_skips_already_rolled_back(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")], cycle_number=5)
        adapter = _make_adapter()
        ledger.rollback_batch("b-1", adapter)
        removed = ledger.rollback_cycle(5, adapter)
        assert removed == 0

    def test_returns_zero_for_unknown_cycle(self):
        ledger = ProvenanceLedger()
        removed = ledger.rollback_cycle(999, _make_adapter())
        assert removed == 0

    def test_raises_when_adapter_lacks_remove_edge(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")], cycle_number=1)
        with pytest.raises(NotImplementedError):
            ledger.rollback_cycle(1, MagicMock(spec=[]))


# ---------------------------------------------------------------------------
# list_batches / get_batch
# ---------------------------------------------------------------------------

class TestInspection:
    def test_list_batches_newest_first(self):
        ledger = ProvenanceLedger()
        for i in range(5):
            ledger.record_batch(f"b-{i}", f"f-{i}", [("a", "b", "r")])
        batches = ledger.list_batches()
        assert batches[0].batch_id == "b-4"
        assert batches[-1].batch_id == "b-0"

    def test_list_batches_n_limit(self):
        ledger = ProvenanceLedger()
        for i in range(10):
            ledger.record_batch(f"b-{i}", f"f-{i}", [("a", "b", "r")])
        assert len(ledger.list_batches(n=3)) == 3

    def test_get_batch_found(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-x", "f-x", [("a", "b", "r")])
        rec = ledger.get_batch("b-x")
        assert rec is not None
        assert rec.batch_id == "b-x"

    def test_get_batch_not_found(self):
        ledger = ProvenanceLedger()
        assert ledger.get_batch("missing") is None


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_fields(self):
        ledger = ProvenanceLedger(max_batches=50)
        ledger.record_batch("b-1", "f-1", [("a", "b", "r"), ("c", "d", "s")], cycle_number=1)
        ledger.record_batch("b-2", "f-2", [("e", "f", "t")], cycle_number=2)
        s = ledger.stats()
        assert s["total_batches"] == 2
        assert s["total_edges_recorded"] == 3
        assert s["batches_rolled_back"] == 0
        assert 1 in s["cycles_seen"]
        assert 2 in s["cycles_seen"]
        assert s["max_batches"] == 50

    def test_rolled_back_count_increments(self):
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "r")])
        ledger.rollback_batch("b-1", _make_adapter())
        assert ledger.stats()["batches_rolled_back"] == 1


# ---------------------------------------------------------------------------
# ResearchAgent integration
# ---------------------------------------------------------------------------

class TestResearchAgentIntegration:
    def _make_proposal(self, src="a", tgt="b", rel="knows", conf=0.8):
        p = MagicMock()
        p.source_id = src
        p.target_id = tgt
        p.relation = rel
        p.confidence = conf
        return p

    def test_set_provenance_ledger_wires(self):
        from core.research_agent import ResearchAgent
        adapter = MagicMock()
        adapter.get_all_entities.return_value = []
        adapter.community_map = {}
        he = MagicMock()
        agent = ResearchAgent(adapter=adapter, hypothesis_engine=he)
        ledger = ProvenanceLedger()
        agent.set_provenance_ledger(ledger)
        assert agent._provenance_ledger is ledger

    def test_approve_records_to_ledger(self):
        from core.research_agent import ResearchAgent, ResearchFinding, ResearchCandidate
        adapter = MagicMock()
        adapter.get_all_entities.return_value = []
        adapter.community_map = {}
        he = MagicMock()
        he.materialize.return_value = 2
        agent = ResearchAgent(adapter=adapter, hypothesis_engine=he)
        ledger = ProvenanceLedger()
        agent.set_provenance_ledger(ledger)

        # Inject a finding into the ring buffer directly
        proposal = self._make_proposal()
        cand = ResearchCandidate("a", "b", 0.5, 0.3, 1, "embedding_scan")
        finding = ResearchFinding(finding_id="fid-1", candidate=cand, proposals=[proposal, proposal], best_confidence=0.8)
        agent._findings.append(finding)

        agent.approve(finding.finding_id)
        assert ledger.stats()["total_batches"] == 1

    def test_approve_forwards_cycle_number(self):
        from core.research_agent import ResearchAgent, ResearchFinding, ResearchCandidate
        adapter = MagicMock()
        adapter.get_all_entities.return_value = []
        adapter.community_map = {}
        he = MagicMock()
        he.materialize.return_value = 1
        agent = ResearchAgent(adapter=adapter, hypothesis_engine=he)
        ledger = ProvenanceLedger()
        agent.set_provenance_ledger(ledger)

        proposal = self._make_proposal()
        cand = ResearchCandidate("a", "b", 0.5, 0.3, 1, "embedding_scan")
        finding = ResearchFinding(finding_id="fid-2", candidate=cand, proposals=[proposal], best_confidence=0.8)
        agent._findings.append(finding)

        agent.approve(finding.finding_id, cycle_number=42)
        rec = ledger.list_batches()[0]
        assert rec.cycle_number == 42

    def test_approve_no_ledger_still_works(self):
        from core.research_agent import ResearchAgent, ResearchFinding, ResearchCandidate
        adapter = MagicMock()
        adapter.get_all_entities.return_value = []
        adapter.community_map = {}
        he = MagicMock()
        he.materialize.return_value = 1
        agent = ResearchAgent(adapter=adapter, hypothesis_engine=he)

        proposal = self._make_proposal()
        cand = ResearchCandidate("a", "b", 0.5, 0.3, 1, "embedding_scan")
        finding = ResearchFinding(finding_id="fid-3", candidate=cand, proposals=[proposal], best_confidence=0.8)
        agent._findings.append(finding)

        result = agent.approve(finding.finding_id)
        assert result == 1  # no error, no ledger


# ---------------------------------------------------------------------------
# AutonomousDiscoveryLoop passes cycle_number
# ---------------------------------------------------------------------------

class TestLoopCycleNumber:
    def test_cycle_number_passed_to_approve(self):
        from core.autonomous_loop import AutonomousDiscoveryLoop, LoopConfig
        from core.auto_approver import AutoDecision

        findings = [MagicMock()]
        findings[0].finding_id = "f-0"

        agent = MagicMock()
        agent.scan_once.return_value = findings
        agent.approve.return_value = 1
        aa = MagicMock()
        aa.decide.return_value = AutoDecision(
            action="approve", confidence=0.9, reason="test", features=[]
        )
        agent._auto_approver = aa

        loop = AutonomousDiscoveryLoop(agent=agent, config=LoopConfig())
        loop.run_cycle()
        loop.run_cycle()

        # First cycle → cycle_number=1, second → cycle_number=2
        calls = agent.approve.call_args_list
        assert calls[0] == call("f-0", cycle_number=1)
        assert calls[1] == call("f-0", cycle_number=2)


# ---------------------------------------------------------------------------
# NetworkXAdapter.remove_edge integration
# ---------------------------------------------------------------------------

class TestNetworkXAdapterRemoveEdge:
    def _make_nx_adapter(self, multigraph=False):
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        G = nx.MultiGraph() if multigraph else nx.Graph()
        G.add_edge("a", "b", relation="knows", confidence=1.0,
                   provenance="", synthetic=False)
        G.add_edge("c", "d", relation="likes", confidence=0.9,
                   provenance="", synthetic=False)
        return NetworkXAdapter(G), G

    def test_remove_edge_simple_graph(self):
        adapter, G = self._make_nx_adapter()
        adapter.remove_edge("a", "b", "knows")
        assert not G.has_edge("a", "b")

    def test_remove_edge_multigraph(self):
        adapter, G = self._make_nx_adapter(multigraph=True)
        adapter.remove_edge("a", "b", "knows")
        assert not G.has_edge("a", "b")

    def test_remove_edge_raises_wrong_relation(self):
        adapter, G = self._make_nx_adapter()
        with pytest.raises(ValueError):
            adapter.remove_edge("a", "b", "nonexistent")

    def test_remove_edge_raises_no_edge(self):
        adapter, G = self._make_nx_adapter()
        with pytest.raises(ValueError):
            adapter.remove_edge("x", "y", "r")

    def test_rollback_batch_with_real_adapter(self):
        import networkx as nx
        from adapters.networkx_adapter import NetworkXAdapter
        G = nx.Graph()
        G.add_edge("a", "b", relation="knows", confidence=1.0, provenance="", synthetic=False)
        adapter = NetworkXAdapter(G)
        ledger = ProvenanceLedger()
        ledger.record_batch("b-1", "f-1", [("a", "b", "knows")])
        removed = ledger.rollback_batch("b-1", adapter)
        assert removed == 1
        assert not G.has_edge("a", "b")

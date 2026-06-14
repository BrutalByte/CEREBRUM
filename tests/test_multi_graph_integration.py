"""
Multi-Graph Integration Test — Phase 275/280.

Demonstrates the three graph systems running together:

  1. Engineering KB  — the main CEREBRUM knowledge graph (toy_graph.csv)
  2. AuraMemory      — personal episodic KG (preferences, project context)
  3. DialogueGraph   — transformer-free conversation memory

Shows:
  - Independent KG queries on each graph
  - Dialogue acts materialized as graph nodes + traversal-based recall
  - Cross-graph entity bridging (person in KB linked to AURA context)
  - FederatedGraphRegistry combining both adapters
  - KnowledgeHarvester vetting pipeline (offline, no external calls)
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.aura_memory import AuraMemory, EpisodicFact
from core.dialogue_graph import DialogueGraph
from core.federated_registry import FederatedGraphRegistry
from core.knowledge_harvester import KnowledgeHarvester, CandidateTriple
from core.hypothesis_engine import HypothesisEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_engineering_adapter() -> NetworkXAdapter:
    """Small scientist knowledge graph (toy_graph.csv subset)."""
    G = nx.DiGraph()
    edges = [
        ("newton",   "einstein",  {"relation": "INFLUENCED",    "weight": 0.9}),
        ("newton",   "faraday",   {"relation": "INFLUENCED",    "weight": 0.85}),
        ("einstein", "bohr",      {"relation": "COLLABORATED",  "weight": 0.85}),
        ("faraday",  "maxwell",   {"relation": "INFLUENCED",    "weight": 0.9}),
        ("einstein", "curie",     {"relation": "CORRESPONDED",  "weight": 0.7}),
        ("maxwell",  "bohr",      {"relation": "PEERS",         "weight": 0.8}),
        ("darwin",   "faraday",   {"relation": "CONTEMPORARIES","weight": 0.6}),
        ("curie",    "bohr",      {"relation": "CORRESPONDED",  "weight": 0.75}),
    ]
    for src, tgt, attrs in edges:
        G.add_edge(src, tgt, **attrs)

    adapter = NetworkXAdapter(G)
    rng = np.random.default_rng(42)
    adapter.embeddings    = {n: rng.random(16).astype(np.float32) for n in G.nodes()}
    adapter.community_map = {"newton": 0, "einstein": 0, "faraday": 1,
                              "maxwell": 1, "bohr": 0, "darwin": 1, "curie": 0}
    return adapter


@pytest.fixture
def engineering_adapter():
    return _make_engineering_adapter()


@pytest.fixture
def aura(tmp_path):
    return AuraMemory(persist_path=tmp_path / "aura_memory.jsonl")


@pytest.fixture
def dialogue(aura):
    return DialogueGraph(aura)


# ---------------------------------------------------------------------------
# 1. Engineering KB — standard graph queries
# ---------------------------------------------------------------------------

class TestEngineeringKB:

    def test_direct_neighbors(self, engineering_adapter):
        neighbors = engineering_adapter.get_neighbors("newton", max_neighbors=10)
        entity_ids = [e.target_id for e in neighbors]
        assert "einstein" in entity_ids
        assert "faraday" in entity_ids

    def test_path_exists_via_community(self, engineering_adapter):
        G = engineering_adapter.to_networkx()
        assert nx.has_path(G, "newton", "bohr")

    def test_community_map_populated(self, engineering_adapter):
        assert engineering_adapter.community_map["einstein"] == 0
        assert engineering_adapter.community_map["maxwell"] == 1

    def test_add_edge_updates_graph(self, engineering_adapter):
        engineering_adapter.add_edge(
            "newton", "darwin", relation="CONTEMPORARIES", confidence=0.5
        )
        G = engineering_adapter.to_networkx()
        assert G.has_edge("newton", "darwin")


# ---------------------------------------------------------------------------
# 2. AuraMemory — personal episodic KG
# ---------------------------------------------------------------------------

class TestAuraMemory:

    def test_ingest_and_recall(self, aura):
        aura.ingest_triple("Bryan", "prefers", "dark_mode")
        aura.ingest_triple("Bryan", "is_working_on", "cerebrum_phase_275")
        aura.ingest_triple("Bryan", "dislikes", "morning_meetings")

        result = aura.recall("Bryan")
        relations = {f.relation for f in result.facts}

        print(f"\n[AuraMemory] Recalled {len(result.facts)} facts about Bryan:")
        for f in result.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        assert "prefers" in relations
        assert "is_working_on" in relations
        assert "dislikes" in relations

    def test_recall_hop2(self, aura):
        aura.ingest_triple("Bryan", "is_working_on", "cerebrum")
        aura.ingest_triple("cerebrum", "context_of", "phd_research")
        aura.ingest_triple("phd_research", "involves", "knowledge_graphs")

        result = aura.recall("Bryan", max_hops=2)
        objs = {f.obj for f in result.facts}

        print(f"\n[AuraMemory] 2-hop recall from Bryan: {objs}")
        assert "cerebrum" in objs
        assert "phd_research" in objs

    def test_fact_persistence_reload(self, tmp_path):
        path = tmp_path / "reload_test.jsonl"
        m1 = AuraMemory(persist_path=path)
        m1.ingest_triple("Bryan", "prefers", "dark_mode")
        m1.ingest_triple("Bryan", "knows_about", "phase_260")

        m2 = AuraMemory(persist_path=path)
        result = m2.recall("Bryan")
        objs = {f.obj for f in result.facts}

        print(f"\n[AuraMemory] Reloaded {len(result.facts)} facts from disk.")
        assert "dark_mode" in objs
        assert "phase_260" in objs

    def test_forget(self, aura):
        fact_id = aura.ingest_triple("Bryan", "reminded_me", "submit_paper")
        assert aura.forget(fact_id)
        result = aura.recall("Bryan")
        assert not any(f.obj == "submit_paper" for f in result.facts)

    def test_stats(self, aura):
        aura.ingest_triple("Bryan", "prefers", "vim")
        aura.ingest_triple("Bryan", "prefers", "dark_mode")
        stats = aura.stats()
        print(f"\n[AuraMemory] Stats: {stats}")
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 2
        assert "prefers" in stats["relations"]


# ---------------------------------------------------------------------------
# 3. DialogueGraph — transformer-free conversation memory
# ---------------------------------------------------------------------------

class TestDialogueGraph:

    def test_session_lifecycle(self, dialogue, aura):
        session = dialogue.open_session("Bryan")

        act1 = dialogue.record_utterance(
            session, "What phase are we on?",
            intent="question", speaker="Bryan",
        )
        resp1 = dialogue.record_response(
            act1, "We are on Phase 275.",
            speaker="AURA",
        )
        dialogue.link_entity(act1, "phase_275", transition="references")
        dialogue.link_entity(resp1, "phase_275", transition="confirms")

        history = dialogue.get_session_history(session)
        print(f"\n[DialogueGraph] Session {session}: {len(history)} acts")
        for act in history:
            print(f"  [{act.speaker}] {act.text!r}")

        assert len(history) == 2
        assert history[0].speaker == "Bryan"
        assert history[1].speaker == "AURA"

    def test_recall_about_entity(self, dialogue, aura):
        session = dialogue.open_session("Bryan")
        act = dialogue.record_utterance(
            session, "Tell me about Einstein's influence.",
            intent="question", speaker="Bryan",
            referenced_entities=["einstein"],
        )
        dialogue.link_entity(act, "einstein", transition="references")

        act_ids = dialogue.recall_about("einstein")
        print(f"\n[DialogueGraph] Acts referencing 'einstein': {act_ids}")
        assert act.act_id in act_ids

    def test_responds_to_chain(self, dialogue, aura):
        session = dialogue.open_session("Bryan")
        q = dialogue.record_utterance(session, "What is CEREBRUM?",
                                      intent="question", speaker="Bryan")
        a = dialogue.record_response(q, "A knowledge graph reasoning system.",
                                     speaker="AURA")

        result = aura.recall(a.act_id)
        relations = {f.relation for f in result.facts}
        print(f"\n[DialogueGraph] Response act relations: {relations}")
        assert "responds_to" in relations

    def test_close_session(self, dialogue):
        session = dialogue.open_session("Bryan")
        dialogue.record_utterance(session, "Hello", speaker="Bryan")
        dialogue.record_utterance(session, "World", speaker="Bryan")
        count = dialogue.close_session(session)
        assert count == 2


# ---------------------------------------------------------------------------
# 4. All three graphs running together
# ---------------------------------------------------------------------------

class TestMultiGraphIntegration:

    def test_cross_graph_entity_bridge(self, engineering_adapter, aura):
        """
        Einstein exists in the engineering KB.
        AURA stores personal context about Einstein (last_discussed).
        Both graphs can be queried independently for the same entity.
        """
        # Engineering KB: who did Einstein collaborate with?
        kb_neighbors = engineering_adapter.get_neighbors("einstein", max_neighbors=10)
        kb_collab = [e.target_id for e in kb_neighbors]

        # AURA: personal context
        aura.ingest_triple("Bryan", "last_discussed", "einstein")
        aura.ingest_triple("einstein", "context_of", "physics_conversation")
        aura_result = aura.recall("einstein")

        print("\n[Multi-Graph] Engineering KB — Einstein's connections:")
        for e in kb_neighbors:
            print(f"  {e.target_id}")

        print("\n[Multi-Graph] AURA — Einstein episodic context:")
        for f in aura_result.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        assert "bohr" in kb_collab or "curie" in kb_collab
        assert any(f.obj == "physics_conversation" for f in aura_result.facts)

    def test_federated_registry_both_graphs(self, engineering_adapter, aura, tmp_path):
        """
        FederatedGraphRegistry holds both graphs under separate domain IDs.
        """
        registry = FederatedGraphRegistry()
        registry.graphs["engineering_kb"] = engineering_adapter
        registry.graphs["aura_memory"]    = aura.adapter

        assert "engineering_kb" in registry.graphs
        assert "aura_memory"    in registry.graphs

        kb_adapter   = registry.get_adapter("engineering_kb")
        aura_adapter = registry.get_adapter("aura_memory")

        # KB query
        kb_neighbors = kb_adapter.get_neighbors("newton", max_neighbors=5)
        # AURA query (graph may be empty or have data from fixture)
        aura_G = aura_adapter._G

        print(f"\n[FederatedRegistry] KB nodes: {list(kb_adapter.to_networkx().nodes())}")
        print(f"[FederatedRegistry] AURA nodes: {list(aura_G.nodes())}")

        assert len(list(kb_adapter.to_networkx().nodes())) >= 7

    def test_dialogue_enriches_aura_and_both_queriable(
        self, engineering_adapter, aura, dialogue
    ):
        """
        A conversation about 'newton' creates dialogue nodes in AURA.
        After the conversation, both the KB (facts about newton) and AURA
        (what we discussed about newton) are independently queryable.
        """
        session = dialogue.open_session("Bryan")
        act = dialogue.record_utterance(
            session,
            "How did Newton influence Einstein?",
            intent="question",
            speaker="Bryan",
            referenced_entities=["newton", "einstein"],
        )
        dialogue.link_entity(act, "newton",   transition="references")
        dialogue.link_entity(act, "einstein", transition="references")

        # Engineering KB: does newton -> einstein edge exist?
        G = engineering_adapter.to_networkx()
        kb_has_path = nx.has_path(G, "newton", "einstein")

        # AURA: dialogue acts that referenced newton
        newton_acts = dialogue.recall_about("newton")
        einstein_acts = dialogue.recall_about("einstein")

        print(f"\n[Integration] KB: newton->einstein path exists: {kb_has_path}")
        print(f"[Integration] AURA: acts referencing newton: {newton_acts}")
        print(f"[Integration] AURA: acts referencing einstein: {einstein_acts}")

        assert kb_has_path
        assert act.act_id in newton_acts
        assert act.act_id in einstein_acts

    def test_knowledge_harvester_vetting_pipeline_offline(
        self, engineering_adapter, aura
    ):
        """
        KnowledgeHarvester vetting pipeline runs entirely offline
        (no external HTTP calls). Tests schema gate and materialization.
        """
        harvester = KnowledgeHarvester(adapter=engineering_adapter)

        # Well-formed triple: should pass schema and materialize
        good = CandidateTriple(
            triple_id   = "test-001",
            source      = "newton",
            relation    = "INFLUENCED",
            target      = "darwin",
            source_name = "Isaac Newton",
            target_name = "Charles Darwin",
            source_url  = "https://en.wikipedia.org/wiki/Isaac_Newton",
            source_tier = 2,
            confidence  = 0.80,
        )

        # Malformed triple: empty relation -> should fail schema gate
        bad = CandidateTriple(
            triple_id   = "test-002",
            source      = "newton",
            relation    = "",
            target      = "darwin",
            source_name = "Isaac Newton",
            target_name = "Charles Darwin",
            source_url  = "https://example.com",
            source_tier = 2,
            confidence  = 0.80,
        )

        ok_good, reason_good = harvester._vet(good)
        ok_bad,  reason_bad  = harvester._vet(bad)

        print(f"\n[KnowledgeHarvester] Good triple passed vetting: {ok_good}")
        print(f"[KnowledgeHarvester] Bad triple rejected: reason={reason_bad!r}")

        assert ok_good,  f"Expected good triple to pass, got: {reason_good}"
        assert not ok_bad, "Expected empty-relation triple to be rejected"

        # Materialize the good triple and verify it's in the graph
        if ok_good:
            harvester._materialize(good)
            G = engineering_adapter.to_networkx()
            assert G.has_edge("newton", "darwin"), "Edge should exist after materialization"
            print(f"[KnowledgeHarvester] newton->darwin materialized into engineering KB.")

    def test_full_pipeline_output(self, engineering_adapter, aura, dialogue):
        """
        End-to-end demonstration of all three graphs working in concert.
        Prints a summary of the combined knowledge state.
        """
        # 1. Populate AURA with personal context
        aura.ingest_triple("Bryan", "is_working_on", "cerebrum")
        aura.ingest_triple("Bryan", "prefers",        "graph_based_reasoning")
        aura.ingest_triple("cerebrum", "context_of",  "phd_research")

        # 2. Simulate a conversation about the KB
        session = dialogue.open_session("Bryan")
        q1 = dialogue.record_utterance(
            session, "Show me Einstein's scientific network.",
            intent="question", speaker="Bryan",
            referenced_entities=["einstein"],
        )
        dialogue.link_entity(q1, "einstein", transition="references")
        a1 = dialogue.record_response(
            q1,
            "Einstein collaborated with Bohr and corresponded with Curie.",
            speaker="AURA",
            referenced_entities=["bohr", "curie"],
        )
        dialogue.link_entity(a1, "bohr",  transition="references")
        dialogue.link_entity(a1, "curie", transition="references")

        # 3. Collect outputs from all three graphs
        kb_neighbors   = engineering_adapter.get_neighbors("einstein", max_neighbors=10)
        aura_about_me  = aura.recall("Bryan")
        dialogue_about_einstein = dialogue.recall_about("einstein")
        dialogue_about_bohr     = dialogue.recall_about("bohr")

        # 4. Print combined state
        print("\n" + "="*60)
        print("CEREBRUM MULTI-GRAPH STATE")
        print("="*60)

        print("\n[Engineering KB] Einstein's graph connections:")
        G = engineering_adapter.to_networkx()
        for n in kb_neighbors:
            nid = n.target_id
            rel = G[nid]["einstein"].get("relation", "?") if G.has_edge(nid, "einstein") else \
                  G["einstein"][nid].get("relation", "?") if G.has_edge("einstein", nid) else "?"
            print(f"  einstein --{rel}--> {nid}")

        print("\n[AURA Memory] Bryan's personal context:")
        for f in aura_about_me.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        print("\n[Dialogue Graph] Acts referencing 'einstein':")
        for act_id in dialogue_about_einstein:
            history = dialogue.get_session_history(session)
            for act in history:
                if act.act_id == act_id:
                    print(f"  [{act.speaker}] {act.text!r}")

        print("\n[Dialogue Graph] Acts referencing 'bohr':")
        for act_id in dialogue_about_bohr:
            history = dialogue.get_session_history(session)
            for act in history:
                if act.act_id == act_id:
                    print(f"  [{act.speaker}] {act.text!r}")

        print("="*60)

        # Assertions
        assert len(kb_neighbors) >= 2
        assert any(f.relation == "is_working_on" for f in aura_about_me.facts)
        assert q1.act_id in dialogue_about_einstein
        assert a1.act_id in dialogue_about_bohr

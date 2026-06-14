"""
Hetionet Multi-Graph Integration Test — Phase 275/280.

Same three-graph architecture as test_multi_graph_integration.py,
but running against the real Hetionet biomedical knowledge graph
(100k edges: genes, diseases, drugs, biological processes).

What Hetionet is:
  A network where nodes are biological things (genes like EGFR,
  diseases like 'lung cancer', drugs like 'Erlotinib') and edges
  are how they relate ('treats', 'associates', 'regulates', etc.).

The three graphs:
  1. Hetionet KB    — the 100k-edge biomedical knowledge graph
  2. AuraMemory     — a researcher's personal context (what they're
                      studying, which diseases they care about)
  3. DialogueGraph  — conversation history about those diseases,
                      recalled by graph traversal, no LLM

Together they simulate a biomedical researcher using AURA as a
personal AI backed by CEREBRUM's knowledge graph.
"""
import csv
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from adapters.csv_adapter import load_csv_adapter
from core.aura_memory import AuraMemory
from core.dialogue_graph import DialogueGraph
from core.federated_registry import FederatedGraphRegistry
from core.knowledge_harvester import KnowledgeHarvester, CandidateTriple

HETIONET_CSV = Path(__file__).parent / "fixtures" / "hetionet_medical_100k.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hetionet():
    """Load Hetionet once for the whole module (100k edges, ~1s)."""
    return load_csv_adapter(str(HETIONET_CSV))


@pytest.fixture
def aura(tmp_path):
    return AuraMemory(persist_path=tmp_path / "researcher_memory.jsonl")


@pytest.fixture
def dialogue(aura):
    return DialogueGraph(aura)


# ---------------------------------------------------------------------------
# 1. Hetionet KB — biomedical graph queries
# ---------------------------------------------------------------------------

class TestHetionetKB:

    def test_graph_scale(self, hetionet):
        G = hetionet.to_networkx()
        print(f"\n[Hetionet KB] {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        assert G.number_of_nodes() > 20_000
        assert G.number_of_edges() > 90_000

    def test_egfr_neighbors(self, hetionet):
        """EGFR is a gene implicated in lung cancer — richly connected."""
        neighbors = hetionet.get_neighbors("EGFR", max_neighbors=20)
        relation_types = {e.relation_type for e in neighbors}

        print(f"\n[Hetionet KB] EGFR neighbors ({len(neighbors)} shown):")
        for e in neighbors[:8]:
            print(f"  EGFR --{e.relation_type}--> {e.target_id}")
        print(f"  ...relation types: {sorted(relation_types)}")

        assert len(neighbors) > 0
        assert any(r in relation_types for r in ("interacts", "regulates", "participates"))

    def test_disease_treatment_path(self, hetionet):
        """lung cancer is treated by Erlotinib — path should exist."""
        G = hetionet.to_networkx()
        assert nx.has_path(G, "lung cancer", "Erlotinib")
        path = nx.shortest_path(G, "lung cancer", "Erlotinib")
        print(f"\n[Hetionet KB] Shortest path lung cancer -> Erlotinib:")
        print(f"  {' -> '.join(path)}")

    def test_hypertension_treatments(self, hetionet):
        """Hypertension has multiple known drug treatments in the graph."""
        neighbors = hetionet.get_neighbors("hypertension", max_neighbors=100)
        treatments = [
            e.target_id for e in neighbors
            if e.relation_type.lower() == "treats"
        ]
        print(f"\n[Hetionet KB] Drugs that treat hypertension: {treatments[:5]}")
        assert len(treatments) >= 2

    def test_gene_disease_association(self, hetionet):
        """LRRK2 gene is associated with Parkinson's disease in Hetionet."""
        neighbors = hetionet.get_neighbors("LRRK2", max_neighbors=50)
        connected = {e.target_id for e in neighbors}
        print(f"\n[Hetionet KB] LRRK2 connected entities (sample): "
              f"{list(connected)[:6]}")
        assert len(connected) > 0

    def test_melanoma_network(self, hetionet):
        """Melanoma is a disease with associations, localizations, and treatments."""
        neighbors = hetionet.get_neighbors("melanoma", max_neighbors=20)
        relations = {e.relation_type for e in neighbors}
        print(f"\n[Hetionet KB] Melanoma relation types: {sorted(relations)}")
        print(f"  Sample connections:")
        for e in neighbors[:5]:
            print(f"  melanoma --{e.relation_type}--> {e.target_id}")
        assert len(neighbors) > 0


# ---------------------------------------------------------------------------
# 2. AuraMemory — researcher's personal context
# ---------------------------------------------------------------------------

class TestResearcherMemory:

    def test_researcher_context(self, aura):
        """A researcher studying cancer stores their personal context."""
        aura.ingest_triple("Dr_Chen",  "is_working_on",  "lung_cancer_research")
        aura.ingest_triple("Dr_Chen",  "interested_in",  "EGFR_inhibitors")
        aura.ingest_triple("Dr_Chen",  "knows_about",    "Erlotinib")
        aura.ingest_triple("Dr_Chen",  "dislikes",       "side_effect_rash")
        aura.ingest_triple("lung_cancer_research", "context_of", "phd_thesis")

        result = aura.recall("Dr_Chen")
        relations = {f.relation for f in result.facts}

        print(f"\n[AuraMemory] Dr. Chen's research context:")
        for f in result.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        assert "is_working_on"  in relations
        assert "interested_in"  in relations
        assert "knows_about"    in relations

    def test_2hop_research_chain(self, aura):
        """2-hop: Dr_Chen -> lung_cancer_research -> phd_thesis."""
        aura.ingest_triple("Dr_Chen", "is_working_on", "lung_cancer_research")
        aura.ingest_triple("lung_cancer_research", "context_of", "phd_thesis")
        aura.ingest_triple("phd_thesis", "supervised_by", "Prof_Smith")

        result = aura.recall("Dr_Chen", max_hops=2)
        objs = {f.obj for f in result.facts}
        print(f"\n[AuraMemory] 2-hop from Dr_Chen: {objs}")
        assert "lung_cancer_research" in objs
        assert "phd_thesis" in objs

    def test_drug_interest_recall(self, aura):
        """Recall personal notes about a specific drug."""
        aura.ingest_triple("Erlotinib", "noted_by",   "Dr_Chen")
        aura.ingest_triple("Erlotinib", "context_of", "EGFR_inhibitor_study")

        result = aura.recall("Erlotinib")
        objs = {f.obj for f in result.facts}
        print(f"\n[AuraMemory] Personal notes on Erlotinib: {objs}")
        assert "EGFR_inhibitor_study" in objs

    def test_persistence_across_sessions(self, tmp_path):
        """Researcher context survives a server restart."""
        path = tmp_path / "researcher.jsonl"
        session1 = AuraMemory(persist_path=path)
        session1.ingest_triple("Dr_Chen", "is_working_on", "melanoma_study")
        session1.ingest_triple("Dr_Chen", "knows_about",   "PRKCE_gene")
        del session1

        session2 = AuraMemory(persist_path=path)
        result = session2.recall("Dr_Chen")
        objs = {f.obj for f in result.facts}
        print(f"\n[AuraMemory] Reloaded context: {objs}")
        assert "melanoma_study" in objs
        assert "PRKCE_gene"     in objs


# ---------------------------------------------------------------------------
# 3. DialogueGraph — biomedical conversation memory
# ---------------------------------------------------------------------------

class TestBiomedicalDialogue:

    def test_clinical_question_session(self, dialogue, aura):
        """A researcher asks AURA about a disease; conversation is stored as graph."""
        session = dialogue.open_session("Dr_Chen")

        q1 = dialogue.record_utterance(
            session,
            "What drugs are used to treat lung cancer?",
            intent="question", speaker="Dr_Chen",
            referenced_entities=["lung_cancer", "drugs"],
        )
        a1 = dialogue.record_response(
            q1,
            "Erlotinib and Methotrexate are used to treat lung cancer.",
            speaker="AURA",
            referenced_entities=["Erlotinib", "Methotrexate", "lung_cancer"],
        )
        dialogue.link_entity(q1, "lung_cancer", transition="references")
        dialogue.link_entity(a1, "Erlotinib",   transition="references")
        dialogue.link_entity(a1, "Methotrexate", transition="references")

        history = dialogue.get_session_history(session)
        print(f"\n[DialogueGraph] Clinical session ({len(history)} turns):")
        for act in history:
            print(f"  [{act.speaker}] {act.text!r}")

        assert len(history) == 2
        assert history[0].intent == "question"

    def test_recall_drug_discussions(self, dialogue, aura):
        """Find all conversation turns that mentioned Erlotinib."""
        session = dialogue.open_session("Dr_Chen")
        act = dialogue.record_utterance(
            session, "Is Erlotinib effective for EGFR-mutant lung cancer?",
            intent="question", speaker="Dr_Chen",
        )
        dialogue.link_entity(act, "Erlotinib", transition="references")
        dialogue.link_entity(act, "EGFR",      transition="references")

        erlotinib_acts = dialogue.recall_about("Erlotinib")
        egfr_acts      = dialogue.recall_about("EGFR")

        print(f"\n[DialogueGraph] Acts about Erlotinib: {len(erlotinib_acts)}")
        print(f"[DialogueGraph] Acts about EGFR:      {len(egfr_acts)}")

        assert act.act_id in erlotinib_acts
        assert act.act_id in egfr_acts

    def test_responds_to_chain_medical(self, dialogue, aura):
        """Response act links back to the question that triggered it."""
        session = dialogue.open_session("Dr_Chen")
        q = dialogue.record_utterance(
            session, "What is the mechanism of Erlotinib?",
            intent="question", speaker="Dr_Chen",
        )
        a = dialogue.record_response(
            q, "Erlotinib inhibits the EGFR tyrosine kinase.", speaker="AURA"
        )
        result = aura.recall(a.act_id)
        rels = {f.relation for f in result.facts}
        print(f"\n[DialogueGraph] Response act relations: {rels}")
        assert "responds_to" in rels


# ---------------------------------------------------------------------------
# 4. All three graphs running together — biomedical scenario
# ---------------------------------------------------------------------------

class TestHetionetMultiGraphIntegration:

    def test_gene_in_kb_and_aura(self, hetionet, aura):
        """
        EGFR exists in Hetionet (a real gene with 20+ connections).
        A researcher also has personal notes about EGFR in AURA.
        Both are queryable independently from the same interface.
        """
        kb_neighbors = hetionet.get_neighbors("EGFR", max_neighbors=10)
        kb_targets   = {e.target_id for e in kb_neighbors}

        aura.ingest_triple("Dr_Chen", "is_studying",  "EGFR")
        aura.ingest_triple("EGFR",    "noted_as",     "key_oncogene")
        aura.ingest_triple("EGFR",    "context_of",   "lung_cancer_research")
        aura_result  = aura.recall("EGFR")
        aura_objs    = {f.obj for f in aura_result.facts}

        print(f"\n[Integration] Hetionet: EGFR connects to {len(kb_targets)} entities (sample):")
        for e in kb_neighbors[:5]:
            print(f"  EGFR --{e.relation_type}--> {e.target_id}")

        print(f"\n[Integration] AURA: personal notes on EGFR:")
        for f in aura_result.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        assert len(kb_targets) >= 5
        assert "key_oncogene"         in aura_objs
        assert "lung_cancer_research" in aura_objs

    def test_federated_registry_bio(self, hetionet, aura):
        """
        Both graphs registered side-by-side in FederatedGraphRegistry.
        KB holds 28k biomedical entities; AURA holds personal research context.
        """
        registry = FederatedGraphRegistry()
        registry.graphs["hetionet"]    = hetionet
        registry.graphs["aura_memory"] = aura.adapter

        kb_adapter   = registry.get_adapter("hetionet")
        aura_adapter = registry.get_adapter("aura_memory")

        kb_node_count   = kb_adapter.to_networkx().number_of_nodes()
        aura_node_count = aura_adapter._G.number_of_nodes()

        print(f"\n[FederatedRegistry] hetionet:    {kb_node_count:,} nodes")
        print(f"[FederatedRegistry] aura_memory: {aura_node_count} nodes")

        assert kb_node_count > 20_000
        assert "hetionet"    in registry.graphs
        assert "aura_memory" in registry.graphs

    def test_dialogue_bridges_kb_and_aura(self, hetionet, aura, dialogue):
        """
        A conversation about 'melanoma' creates dialogue nodes in AURA
        AND references facts from the Hetionet KB.
        After the conversation:
          - Hetionet can answer: what does melanoma associate with?
          - AURA can answer:     what did we discuss about melanoma?
        """
        session = dialogue.open_session("Dr_Chen")

        q = dialogue.record_utterance(
            session,
            "What genes are associated with melanoma?",
            intent="question", speaker="Dr_Chen",
            referenced_entities=["melanoma"],
        )
        dialogue.link_entity(q, "melanoma", transition="references")

        a = dialogue.record_response(
            q,
            "PRKCE and other genes associate with or regulate melanoma pathways.",
            speaker="AURA",
            referenced_entities=["PRKCE", "melanoma"],
        )
        dialogue.link_entity(a, "PRKCE",    transition="references")
        dialogue.link_entity(a, "melanoma", transition="references")

        # KB: melanoma's actual connections
        kb_neighbors  = hetionet.get_neighbors("melanoma", max_neighbors=10)
        kb_relations  = {e.relation_type for e in kb_neighbors}

        # Dialogue recall: what was discussed about melanoma?
        melanoma_acts = dialogue.recall_about("melanoma")
        prkce_acts    = dialogue.recall_about("PRKCE")

        print(f"\n[Integration] Hetionet: melanoma relation types: {sorted(kb_relations)}")
        print(f"[Integration] Hetionet: melanoma sample neighbors:")
        for e in kb_neighbors[:5]:
            print(f"  melanoma --{e.relation_type}--> {e.target_id}")

        print(f"\n[Integration] AURA: {len(melanoma_acts)} dialogue act(s) referenced melanoma")
        history = dialogue.get_session_history(session)
        for act_id in melanoma_acts:
            for act in history:
                if act.act_id == act_id:
                    print(f"  [{act.speaker}] {act.text!r}")

        assert len(kb_neighbors) > 0
        assert q.act_id in melanoma_acts
        assert a.act_id in melanoma_acts
        assert a.act_id in prkce_acts

    def test_harvester_vetting_biomedical(self, hetionet):
        """
        KnowledgeHarvester vets biomedical triples offline.
        A medically plausible triple (gene regulates gene) should pass.
        A triple with a relation not matching known schema logs a warning
        but still passes stage 1 (novel relations are allowed).
        """
        harvester = KnowledgeHarvester(adapter=hetionet)

        # Well-formed triple: EGFR regulates a known gene
        well_formed = CandidateTriple(
            triple_id   = "bio-001",
            source      = "EGFR",
            relation    = "regulates",
            target      = "PRKCE",
            source_name = "EGFR gene",
            target_name = "PRKCE gene",
            source_url  = "https://www.ncbi.nlm.nih.gov/gene/1956",
            source_tier = 1,
            confidence  = 0.85,
        )

        # Malformed: empty source
        malformed = CandidateTriple(
            triple_id   = "bio-002",
            source      = "",
            relation    = "treats",
            target      = "lung_cancer",
            source_name = "",
            target_name = "lung cancer",
            source_url  = "https://example.com",
            source_tier = 2,
            confidence  = 0.5,
        )

        ok1, reason1 = harvester._vet(well_formed)
        ok2, reason2 = harvester._vet(malformed)

        print(f"\n[KnowledgeHarvester] EGFR regulates PRKCE — passed: {ok1}")
        print(f"[KnowledgeHarvester] Empty source triple  — rejected: {reason2!r}")

        assert ok1,       f"Well-formed triple should pass: {reason1}"
        assert not ok2,   "Malformed triple should be rejected"

        if ok1:
            harvester._materialize(well_formed)
            G = hetionet.to_networkx()
            assert G.has_edge("EGFR", "PRKCE")
            print(f"[KnowledgeHarvester] EGFR --regulates--> PRKCE added to Hetionet KB.")

    def test_full_biomedical_pipeline(self, hetionet, aura, dialogue):
        """
        End-to-end: a researcher asks about lung cancer treatment,
        the answer draws from Hetionet, the conversation is stored in AURA,
        and all three graphs can be queried to reconstruct the full picture.
        """
        # Researcher context
        aura.ingest_triple("Dr_Chen", "is_working_on",  "lung_cancer_treatment")
        aura.ingest_triple("Dr_Chen", "interested_in",  "EGFR_targeted_therapy")
        aura.ingest_triple("lung_cancer_treatment", "context_of", "clinical_trial_2026")

        # Dialogue
        session = dialogue.open_session("Dr_Chen")
        q = dialogue.record_utterance(
            session,
            "Which drugs target EGFR in lung cancer?",
            intent="question", speaker="Dr_Chen",
            referenced_entities=["EGFR", "lung_cancer"],
        )
        dialogue.link_entity(q, "EGFR",        transition="references")
        dialogue.link_entity(q, "lung_cancer",  transition="references")

        a = dialogue.record_response(
            q,
            "Erlotinib is an EGFR inhibitor used for lung cancer treatment.",
            speaker="AURA",
            referenced_entities=["Erlotinib", "EGFR"],
        )
        dialogue.link_entity(a, "Erlotinib", transition="references")
        dialogue.link_entity(a, "EGFR",      transition="confirms")

        # Query all three
        kb_egfr       = hetionet.get_neighbors("EGFR", max_neighbors=10)
        aura_context  = aura.recall("Dr_Chen")
        egfr_acts     = dialogue.recall_about("EGFR")
        erlotinib_acts = dialogue.recall_about("Erlotinib")

        print("\n" + "=" * 60)
        print("CEREBRUM BIOMEDICAL MULTI-GRAPH STATE")
        print("=" * 60)

        print("\n[Hetionet KB] EGFR connections (sample):")
        for e in kb_egfr[:6]:
            print(f"  EGFR --{e.relation_type}--> {e.target_id}")

        print("\n[AURA Memory] Dr. Chen's research context:")
        for f in aura_context.facts:
            print(f"  ({f.subject}, {f.relation}, {f.obj})")

        print("\n[Dialogue Graph] Acts referencing EGFR:")
        history = dialogue.get_session_history(session)
        for act_id in egfr_acts:
            for act in history:
                if act.act_id == act_id:
                    print(f"  [{act.speaker}] {act.text!r}")

        print("\n[Dialogue Graph] Acts referencing Erlotinib:")
        for act_id in erlotinib_acts:
            for act in history:
                if act.act_id == act_id:
                    print(f"  [{act.speaker}] {act.text!r}")

        print("=" * 60)

        # Assertions
        assert len(kb_egfr) >= 5
        assert any(f.relation == "is_working_on" for f in aura_context.facts)
        assert q.act_id in egfr_acts
        assert a.act_id in erlotinib_acts

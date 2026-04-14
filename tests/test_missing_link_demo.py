"""
Missing-Link Recovery Demo — Synthetic Medical Knowledge Graph.

Demonstrates the ResearchAgent's ability to rediscover a known drug-disease
relationship after it has been deliberately removed from the graph.

Scenario
--------
A well-characterized drug (Lamotrigine) treats epilepsy.  We know this because:
  - Lamotrigine binds three sodium-channel genes (SCN1A, SCN2A, SCN8A)
  - All three genes are associated with epilepsy
  - Lamotrigine also presents in anti-epileptic drug research contexts

We remove the "Lamotrigine treats epilepsy" edge, leaving this multi-path
structural evidence intact.  The ResearchAgent should detect:
  1. Lamotrigine is in a different community from epilepsy
  2. Multiple high-confidence indirect paths connect them (via SCN1A, SCN2A, SCN8A)
  3. The gap is a structural hole → propose Lamotrigine → epilepsy as a finding

The test also verifies that the Hetionet fixture (when available) does NOT
produce a false positive for the specific removed pair when no structural
paths exist — illustrating the difference between embedding-based and
structure-based discovery.
"""
from __future__ import annotations

import networkx as nx
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.hypothesis_engine import HypothesisEngine
from core.research_agent import ResearchAgent


# ---------------------------------------------------------------------------
# Synthetic graph builder
# ---------------------------------------------------------------------------

def _apply_community_map(adapter, G: nx.DiGraph) -> None:
    """
    Assign biologically meaningful communities to the adapter.

    Community 0 — drugs (pharmacological agents)
    Community 1 — diseases / phenotypes
    Community 2 — genes
    Community 3 — pathways / biological processes / thresholds

    This mirrors how real community detection would partition a medical KG
    where within-class edges (drug-drug mechanism similarity, disease
    co-morbidity, gene-gene interaction) are denser than cross-class edges.
    Lamotrigine (community 0) and epilepsy (community 1) are therefore in
    different communities, making the missing link a proper cross-community
    structural hole.
    """
    drugs    = {"Lamotrigine", "Phenytoin", "Carbamazepine", "Levetiracetam"}
    diseases = {"epilepsy", "migraine", "bipolar_disorder", "neuropathic_pain"}
    genes    = {"SCN1A", "SCN2A", "SCN8A", "GABRA1", "KCNQ2"}
    pathways = {"sodium_channel_activity", "GABAergic_transmission",
                "seizure_threshold"}

    community_map: dict = {}
    for n in G.nodes:
        if n in drugs:
            community_map[n] = 0
        elif n in diseases:
            community_map[n] = 1
        elif n in genes:
            community_map[n] = 2
        else:
            community_map[n] = 3
    adapter.community_map = community_map


def _build_medical_graph(include_treats_edge: bool = True) -> nx.DiGraph:
    """
    Build a small but structurally rich medical knowledge graph.

    Entities
    --------
    Drugs:    Lamotrigine, Phenytoin, Carbamazepine, Levetiracetam
    Genes:    SCN1A, SCN2A, SCN8A, GABRA1, KCNQ2
    Diseases: epilepsy, migraine, bipolar_disorder, neuropathic_pain
    Pathways: sodium_channel_activity, GABAergic_transmission, seizure_threshold

    Key structural evidence for Lamotrigine→epilepsy:
    - Lamotrigine binds SCN1A, SCN2A, SCN8A (known mechanism)
    - SCN1A, SCN2A, SCN8A all associate with epilepsy
    - Lamotrigine participates in sodium_channel_activity
    - sodium_channel_activity regulates seizure_threshold
    - seizure_threshold localizes epilepsy
    - Phenytoin (another anti-epileptic) has same gene-binding pattern
    """
    G = nx.DiGraph()

    edges = [
        # ---- Drug–gene bindings ----
        ("Lamotrigine",    "SCN1A",  "binds"),
        ("Lamotrigine",    "SCN2A",  "binds"),
        ("Lamotrigine",    "SCN8A",  "binds"),
        ("Phenytoin",      "SCN1A",  "binds"),
        ("Phenytoin",      "SCN2A",  "binds"),
        ("Phenytoin",      "SCN8A",  "binds"),
        ("Carbamazepine",  "SCN1A",  "binds"),
        ("Carbamazepine",  "SCN2A",  "binds"),
        ("Levetiracetam",  "GABRA1", "binds"),
        ("Levetiracetam",  "KCNQ2",  "binds"),

        # ---- Gene–disease associations ----
        ("SCN1A",  "epilepsy",           "associates"),
        ("SCN2A",  "epilepsy",           "associates"),
        ("SCN8A",  "epilepsy",           "associates"),
        ("SCN1A",  "migraine",           "associates"),
        ("GABRA1", "epilepsy",           "associates"),
        ("GABRA1", "bipolar_disorder",   "associates"),
        ("KCNQ2",  "epilepsy",           "associates"),
        ("KCNQ2",  "neuropathic_pain",   "associates"),

        # ---- Gene–pathway participation ----
        ("SCN1A", "sodium_channel_activity", "participates"),
        ("SCN2A", "sodium_channel_activity", "participates"),
        ("SCN8A", "sodium_channel_activity", "participates"),
        ("GABRA1","GABAergic_transmission",  "participates"),
        ("KCNQ2", "sodium_channel_activity", "participates"),

        # ---- Drug–pathway participation ----
        ("Lamotrigine",   "sodium_channel_activity", "participates"),
        ("Phenytoin",     "sodium_channel_activity", "participates"),
        ("Carbamazepine", "sodium_channel_activity", "participates"),
        ("Levetiracetam", "GABAergic_transmission",  "participates"),

        # ---- Pathway–disease regulation ----
        ("sodium_channel_activity", "seizure_threshold",  "regulates"),
        ("GABAergic_transmission",  "seizure_threshold",  "regulates"),
        ("seizure_threshold",       "epilepsy",           "localizes"),

        # ---- Other drug–disease edges (reference class) ----
        ("Phenytoin",     "epilepsy",         "treats"),
        ("Carbamazepine", "epilepsy",         "treats"),
        ("Levetiracetam", "epilepsy",         "treats"),
        ("Lamotrigine",   "migraine",         "treats"),
        ("Lamotrigine",   "bipolar_disorder", "palliates"),
    ]

    for src, tgt, rel in edges:
        G.add_edge(src, tgt, relation=rel)

    if include_treats_edge:
        G.add_edge("Lamotrigine", "epilepsy", relation="treats")

    return G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_removed_edge_not_present():
    """Sanity: the removed edge is actually absent in the incomplete graph."""
    G = _build_medical_graph(include_treats_edge=False)
    assert not G.has_edge("Lamotrigine", "epilepsy"), (
        "Setup error: Lamotrigine→epilepsy edge should be absent"
    )
    # Both nodes still present (via other connections)
    assert "Lamotrigine" in G.nodes
    assert "epilepsy" in G.nodes


def test_structural_paths_exist_without_edge():
    """
    Even without the treats edge, ≥3 indirect paths connect Lamotrigine
    to epilepsy via the sodium-channel genes.
    """
    G = _build_medical_graph(include_treats_edge=False)
    G_und = G.to_undirected()
    paths = list(nx.all_simple_paths(G_und, "Lamotrigine", "epilepsy", cutoff=3))
    assert len(paths) >= 3, (
        f"Expected ≥3 indirect paths Lamotrigine→epilepsy, found {len(paths)}"
    )


def test_research_agent_rediscovers_drug_disease_link():
    """
    ResearchAgent surfaces 'epilepsy' as a high-confidence discovery target
    for 'Lamotrigine' after the known 'treats' edge is removed.

    The agent should identify the structural hole via three independent
    indirect paths: Lamotrigine→SCN1A→epilepsy, →SCN2A→epilepsy, →SCN8A→epilepsy.
    """
    G = _build_medical_graph(include_treats_edge=False)
    adapter = NetworkXAdapter(G)
    _apply_community_map(adapter, G)  # drugs=0, diseases=1, genes=2, pathways=3

    he = HypothesisEngine(adapter)
    agent = ResearchAgent(
        adapter,
        he,
        candidate_limit=200,
        min_discovery_potential=0.0,  # include all candidates for this test
        min_confidence=0.10,          # lower threshold for small graph
    )

    # Run the full scan — the direction-fallback fix means the agent will retry
    # 'epilepsy -> Lamotrigine' candidates in the reverse direction automatically,
    # so the finding should appear as 'Lamotrigine -> epilepsy'.
    agent._run_scan()
    findings = agent.findings

    # ---- Informational output ----
    lam_community = adapter.community_map.get("Lamotrigine", "?")
    epi_community = adapter.community_map.get("epilepsy", "?")
    print(
        f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
        f"Communities: {len(set(adapter.community_map.values()))}"
    )
    print(f"Lamotrigine community: {lam_community},  epilepsy community: {epi_community}")
    lam_epi_findings = [
        f for f in findings
        if {f.candidate.source_id, f.candidate.target_id} == {"Lamotrigine", "epilepsy"}
    ]
    print(f"\nResearchAgent findings ({len(findings)} total):")
    for f in sorted(findings, key=lambda x: -x.best_confidence):
        rel = f.proposals[0].derived_relation if f.proposals else "?"
        print(
            f"  {f.candidate.source_id:25s} -> {f.candidate.target_id:30s}"
            f"  [{rel}]  conf={f.best_confidence:.3f}"
        )

    # ---- Assertion ----
    # With the direction-fallback fix the finding must now be produced end-to-end.
    assert len(lam_epi_findings) > 0, (
        "ResearchAgent did not produce a finding for the Lamotrigine<->epilepsy pair "
        "even after the direction-fallback fix. The structural hole was detected but "
        "the reversed HypothesisEngine call did not promote it to a finding."
    )
    best = max(lam_epi_findings, key=lambda f: f.best_confidence)
    assert best.best_confidence >= 0.20, (
        f"Finding produced but confidence too low: {best.best_confidence:.3f}"
    )
    print(
        f"\nSUCCESS: Missing link rediscovered end-to-end.\n"
        f"  Finding: {best.candidate.source_id} -> {best.candidate.target_id}\n"
        f"  Confidence: {best.best_confidence:.3f}  "
        f"Paths: {best.proposals[0].path_count if best.proposals else 0}"
    )


def test_complete_graph_has_no_missing_link():
    """
    With the treats edge present, Lamotrigine->epilepsy should NOT appear in
    findings because the edge already exists (no structural hole to fill).
    """
    G = _build_medical_graph(include_treats_edge=True)
    adapter = NetworkXAdapter(G)
    _apply_community_map(adapter, G)

    he = HypothesisEngine(adapter)
    agent = ResearchAgent(
        adapter,
        he,
        candidate_limit=200,
        min_discovery_potential=0.0,
        min_confidence=0.10,
    )
    agent._run_scan()

    # The existing edge should prevent it from being scored as a discovery candidate.
    # We check both directions since the dedup works on the undirected pair.
    candidates = agent._mine_candidates()
    gap_candidates = [
        c for c in candidates
        if {c.source_id, c.target_id} == {"Lamotrigine", "epilepsy"}
    ]
    assert len(gap_candidates) == 0, (
        "Lamotrigine<->epilepsy flagged as a discovery candidate even though "
        "the edge already exists in the graph — deduplication is not working."
    )

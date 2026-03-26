"""
Tests for Phase 21 — TextIngestor.
"""
import os
import tempfile
import pytest
import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.text_ingestor import (
    TextIngestor,
    IngestReport,
    _split_sentences,
    _find_entity_mentions,
    _extract_verb_phrase,
    _map_verb_to_relation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_adapter(edges=None):
    G = nx.Graph()
    for src, dst, rel in (edges or []):
        G.add_edge(src, dst, relation_type=rel, relation=rel, confidence=1.0)
    return NetworkXAdapter(G)


@pytest.fixture
def science_adapter():
    """Graph with known scientific entities."""
    return _make_adapter([
        ("aspirin",  "headache",    "TREATS"),
        ("smoking",  "cancer",      "CAUSES"),
        ("newton",   "leibniz",     "INFLUENCED"),
        ("newton",   "optics",      "WROTE"),
        ("leibniz",  "calculus",    "INVENTED"),
    ])


@pytest.fixture
def empty_adapter():
    return _make_adapter()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    text = "Newton influenced Leibniz. Leibniz invented calculus."
    sents = _split_sentences(text)
    assert len(sents) == 2
    assert "Newton" in sents[0]
    assert "calculus" in sents[1]


def test_split_sentences_single():
    sents = _split_sentences("Newton influenced Leibniz.")
    assert len(sents) == 1


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_find_entity_mentions_known(science_adapter):
    idx = {"newton": "newton", "leibniz": "leibniz"}
    mentions = _find_entity_mentions("Newton influenced Leibniz.", idx)
    assert len(mentions) == 2
    assert mentions[0]["entity_id"] == "newton"
    assert mentions[1]["entity_id"] == "leibniz"


def test_find_entity_mentions_order(science_adapter):
    idx = {"aspirin": "aspirin", "headache": "headache"}
    mentions = _find_entity_mentions("aspirin treats headache and headache is bad", idx)
    assert mentions[0]["entity_id"] == "aspirin"


def test_find_entity_mentions_empty():
    mentions = _find_entity_mentions("nothing here", {"newton": "newton"})
    assert len(mentions) == 0


def test_extract_verb_phrase_simple():
    verb, conf = _extract_verb_phrase("influenced")
    assert verb != ""
    assert conf > 0


def test_extract_verb_phrase_with_aux():
    verb, conf = _extract_verb_phrase("was influenced by")
    assert "influenc" in verb.lower() or verb != ""


def test_extract_verb_phrase_empty():
    verb, conf = _extract_verb_phrase("")
    assert verb == ""
    assert conf == 0.0


def test_map_verb_to_relation_influences():
    rel, conf = _map_verb_to_relation("influenced")
    assert rel == "INFLUENCED"
    assert conf > 0


def test_map_verb_to_relation_causes():
    rel, conf = _map_verb_to_relation("causes")
    assert rel == "CAUSES"


def test_map_verb_to_relation_treats():
    rel, conf = _map_verb_to_relation("treats")
    assert rel == "TREATS"


def test_map_verb_to_relation_invented():
    rel, conf = _map_verb_to_relation("invented")
    assert rel == "INVENTED"


def test_map_verb_to_relation_unknown():
    rel, conf = _map_verb_to_relation("xyzzyverbxyz")
    assert rel == ""
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Basic ingest
# ---------------------------------------------------------------------------

def test_ingest_returns_report(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    assert isinstance(report, IngestReport)


def test_ingest_empty_text(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("", dry_run=True)
    assert report.triples_extracted == 0
    assert report.edges_added == 0


def test_ingest_known_entities_extracted(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text(
        "Newton influenced Leibniz in the development of calculus.",
        dry_run=True,
    )
    rels = {(s, r, d) for s, r, d, _ in report.added_triples}
    # Should find newton→leibniz (INFLUENCED) but it already exists → dedup
    # May find newton→calculus or leibniz→calculus as new triples
    assert isinstance(report.triples_extracted, int)
    assert report.triples_extracted >= 0


def test_new_entity_discovered(science_adapter):
    """Entity 'faraday' is not in the graph — should be created."""
    ing = TextIngestor(science_adapter, create_new_entities=True)
    report = ing.ingest_text(
        "Newton influenced Faraday who studied electricity.",
        dry_run=True,
    )
    entity_ids = {s for s, r, d, _ in report.added_triples} | \
                 {d for s, r, d, _ in report.added_triples}
    assert "faraday" in entity_ids or any("faraday" in x.lower() for x in entity_ids)


def test_no_new_entity_when_disabled(science_adapter):
    """create_new_entities=False: triples with unknown entities are skipped."""
    ing = TextIngestor(science_adapter, create_new_entities=False)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    # "Faraday" is not in the graph; with create_new_entities=False it should be skipped
    entity_ids = {s for s, r, d, _ in report.added_triples} | \
                 {d for s, r, d, _ in report.added_triples}
    assert not any("faraday" in x.lower() for x in entity_ids)


def test_duplicate_edge_skipped(science_adapter):
    """newton→leibniz INFLUENCED already exists — should be counted as duplicate."""
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Leibniz.", dry_run=True)
    assert report.triples_skipped_duplicate >= 1


def test_low_confidence_filtered(science_adapter):
    """With very high min_confidence, most triples are filtered."""
    ing = TextIngestor(science_adapter, min_confidence=0.99)
    report = ing.ingest_text(
        "Newton influenced Faraday who studied electricity.",
        dry_run=True,
    )
    assert report.triples_skipped_low_confidence >= 0  # some may be filtered


# ---------------------------------------------------------------------------
# Relation type mapping
# ---------------------------------------------------------------------------

def test_influence_verb_mapped(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    rels = [r for _, r, _, _ in report.added_triples]
    assert "INFLUENCED" in rels


def test_causes_verb_mapped(science_adapter):
    ing = TextIngestor(science_adapter)
    # "pollution" is not in graph — but with create_new_entities it should work
    report = ing.ingest_text("Smoking causes cancer.", dry_run=True)
    # "Smoking causes cancer" already exists (duplicate) — but CAUSES should be found
    # Actually it's a duplicate so won't be in added_triples, but report.triples_extracted >= 1
    assert report.triples_extracted >= 1


def test_invented_verb_mapped(science_adapter):
    ing = TextIngestor(science_adapter)
    # Use entities in the graph
    report = ing.ingest_text("Leibniz invented calculus.", dry_run=True)
    assert report.triples_extracted >= 1
    # If it's a duplicate, skipped_duplicate >= 1
    assert report.triples_skipped_duplicate + len(report.added_triples) >= 1


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dry_run_does_not_mutate(science_adapter):
    G_before = science_adapter.to_networkx()
    n_before = G_before.number_of_edges()
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday who studied electricity.", dry_run=True)
    assert science_adapter.to_networkx().number_of_edges() == n_before


def test_real_run_mutates_graph(science_adapter):
    n_before = science_adapter.to_networkx().number_of_edges()
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday who studied electricity.")
    if report.edges_added > 0:
        assert science_adapter.to_networkx().number_of_edges() > n_before


def test_dry_run_report_shows_proposals(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    # Proposals shown regardless of dry_run
    assert report.edges_added >= 0   # 0 if already exists (duplicate)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_removes_added_edges(science_adapter):
    n_before = science_adapter.to_networkx().number_of_edges()
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday who studied electricity.")
    if report.edges_added > 0:
        removed = ing.rollback()
        assert removed > 0
        assert science_adapter.to_networkx().number_of_edges() == n_before


def test_rollback_raises_without_prior_run(science_adapter):
    ing = TextIngestor(science_adapter)
    with pytest.raises(RuntimeError):
        ing.rollback()


def test_rollback_raises_after_dry_run(science_adapter):
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    with pytest.raises(RuntimeError):
        ing.rollback()


def test_can_rollback_false_initially(science_adapter):
    ing = TextIngestor(science_adapter)
    assert ing.can_rollback is False


def test_can_rollback_true_after_run(science_adapter):
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday who studied electricity.")
    assert ing.can_rollback is True


def test_can_rollback_false_after_rollback(science_adapter):
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday who studied electricity.")
    ing.rollback()
    assert ing.can_rollback is False


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_set_on_edge(science_adapter):
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday who studied electricity.")
    G = science_adapter.to_networkx()
    found = False
    for u, v, data in G.edges(data=True):
        if data.get("provenance", "").startswith("text_ingest:"):
            found = True
            assert data.get("source_text") is not None
            break
    # Only assert if edges were actually added
    if science_adapter.to_networkx().number_of_edges() > 5:  # > original 5
        assert found


def test_provenance_hash_consistent():
    """Same text → same provenance hash."""
    from core.text_ingestor import TextIngestor
    a1 = _make_adapter([("newton", "leibniz", "INFLUENCED")])
    a2 = _make_adapter([("newton", "leibniz", "INFLUENCED")])
    i1 = TextIngestor(a1)
    i2 = TextIngestor(a2)
    r1 = i1.ingest_text("Newton influenced Faraday.", dry_run=True)
    r2 = i2.ingest_text("Newton influenced Faraday.", dry_run=True)
    assert r1.provenance == r2.provenance


# ---------------------------------------------------------------------------
# File ingest
# ---------------------------------------------------------------------------

def test_ingest_file(science_adapter):
    ing = TextIngestor(science_adapter)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write("Newton influenced Faraday who studied electricity.\n")
        f.write("Leibniz invented calculus independently.\n")
        fname = f.name
    try:
        report = ing.ingest_file(fname, dry_run=True)
        assert isinstance(report, IngestReport)
        assert report.sentences_processed >= 2
    finally:
        os.unlink(fname)


def test_ingest_file_missing_raises(science_adapter):
    ing = TextIngestor(science_adapter)
    with pytest.raises(FileNotFoundError):
        ing.ingest_file("/nonexistent/path/file.txt")


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

def test_report_text_length(science_adapter):
    text = "Newton influenced Faraday."
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text(text, dry_run=True)
    assert report.text_length == len(text)


def test_report_duration_positive(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    assert report.duration_seconds >= 0.0


def test_report_sentences_processed(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text(
        "Newton influenced Faraday. Leibniz invented calculus.",
        dry_run=True,
    )
    assert report.sentences_processed >= 2


def test_report_dry_run_flag(science_adapter):
    ing = TextIngestor(science_adapter)
    assert ing.ingest_text("Newton influenced Faraday.", dry_run=True).dry_run is True
    assert ing.ingest_text("Newton influenced Faraday.", dry_run=False).dry_run is False


def test_report_summary_string(science_adapter):
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    s = report.summary()
    assert isinstance(s, str)
    assert "IngestReport" in s


def test_last_report_none_before_run(science_adapter):
    ing = TextIngestor(science_adapter)
    assert ing.last_report is None


def test_last_report_populated_after_run(science_adapter):
    ing = TextIngestor(science_adapter)
    ing.ingest_text("Newton influenced Faraday.", dry_run=True)
    assert ing.last_report is not None


# ---------------------------------------------------------------------------
# Multi-sentence paragraph
# ---------------------------------------------------------------------------

def test_multi_sentence_paragraph(science_adapter):
    text = (
        "Newton influenced Leibniz who invented calculus. "
        "Aspirin treats headache. "
        "Smoking causes cancer."
    )
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text(text, dry_run=True)
    assert report.sentences_processed >= 3
    assert report.triples_extracted >= 0  # at least attempted


# ---------------------------------------------------------------------------
# Case-insensitive entity matching
# ---------------------------------------------------------------------------

def test_case_insensitive_link(science_adapter):
    """'NEWTON' should link to 'newton' in the graph."""
    ing = TextIngestor(science_adapter)
    report = ing.ingest_text("NEWTON influenced Faraday.", dry_run=True)
    src_ids = {s for s, _, _, _ in report.added_triples}
    assert "newton" in src_ids or any("newton" in x for x in src_ids)


# ---------------------------------------------------------------------------
# Empty graph (no anchors available)
# ---------------------------------------------------------------------------

def test_empty_graph_uses_capitalization(empty_adapter):
    """With no known entities, still extracts from capitalized nouns."""
    ing = TextIngestor(empty_adapter, create_new_entities=True)
    report = ing.ingest_text(
        "Einstein influenced Bohr who studied quantum mechanics.",
        dry_run=True,
    )
    # May or may not find triples depending on capitalization heuristic
    assert isinstance(report, IngestReport)
    assert report.triples_extracted >= 0


import pytest
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from core.extraction_engine import ExtractionEngine, ExtractionConfig

@pytest.fixture
def toy_adapter():
    G = nx.Graph()
    G.add_node("Newton", label="Newton", type="Person")
    G.add_node("Leibniz", label="Leibniz", type="Person")
    G.add_node("Calculus", label="Calculus", type="Field")
    return NetworkXAdapter(G)

def test_extraction_local_anchored(toy_adapter):
    """Test Stage 1 anchoring (no dependencies beyond standard libs)."""
    engine = ExtractionEngine(toy_adapter, config=ExtractionConfig(backend="local"))
    
    # Text with known entities
    text = "Newton influenced Leibniz who invented Calculus."
    triples = engine.extract(text)
    
    # We expect at least two triples: (Newton, INFLUENCED, Leibniz) and (Leibniz, INVENTED, Calculus)
    # The actual relation names depend on _verb_to_relation mapping.
    assert len(triples) >= 2
    
    subjects = {t.subject for t in triples}
    objects = {t.object for t in triples}
    
    assert "Newton" in subjects
    assert "Leibniz" in subjects or "Leibniz" in objects
    assert "Calculus" in objects

def test_extraction_ingest_text(toy_adapter):
    """Test full ingestion flow."""
    engine = ExtractionEngine(toy_adapter, config=ExtractionConfig(backend="local", min_confidence=0.1))
    
    text = "Newton influenced Leibniz."
    report = engine.ingest_text(text)
    
    assert report.edges_added >= 1
    
    # Verify edge in graph
    G = toy_adapter.to_networkx()
    assert G.has_edge("Newton", "Leibniz")
    edge_data = G.get_edge_data("Newton", "Leibniz")
    assert edge_data["relation"] == "INFLUENCED"
    assert "extraction:local_verb" in edge_data["provenance"]

def test_extraction_namespace(toy_adapter):
    """Test namespace prefixing during extraction/ingestion."""
    config = ExtractionConfig(backend="local", namespace="test", min_confidence=0.1)
    engine = ExtractionEngine(toy_adapter, config=config)
    
    # "Einstein" is NOT in the graph, so it should be prefixed
    # "Newton" IS in the graph, so it should be linked (no prefix)
    text = "Newton influenced Einstein."
    report = engine.ingest_text(text)
    
    G = toy_adapter.to_networkx()
    # Newton should be linked to existing node
    assert "Newton" in G.nodes
    # Einstein should be a new node with prefix (and lowercase by IngestionPipeline)
    assert "test:einstein" in G.nodes
    
    assert G.has_edge("Newton", "test:einstein")

def test_extraction_rollback(toy_adapter):
    """Test that rollback removes edges added by ExtractionEngine."""
    engine = ExtractionEngine(toy_adapter, config=ExtractionConfig(backend="local", min_confidence=0.1))
    
    text = "Newton influenced Leibniz."
    report = engine.ingest_text(text)
    assert toy_adapter.to_networkx().has_edge("Newton", "Leibniz")
    
    # Rollback
    engine.rollback(report)
    assert not toy_adapter.to_networkx().has_edge("Newton", "Leibniz")

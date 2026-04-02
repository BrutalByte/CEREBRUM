"""
Tests for Phase 35 — Source-Grounded Provenance.
"""
import pytest
import numpy as np
from core.thalamus import IngestionPipeline
from reasoning.traversal import TraversalPath
from reasoning.path_scorer import score_path, grounding_score

def test_grounding_score_unit():
    # 1. Path with generic/missing provenance
    p1 = TraversalPath(
        edge_confidences=[1.0, 1.0],
        edge_provenances=["triples", "manual"]
    )
    # Expected: 1.0 * 0.9 * 0.9 = 0.81
    assert pytest.approx(grounding_score(p1), rel=1e-3) == 0.81
    
    # 2. Path with verified provenance
    p2 = TraversalPath(
        edge_confidences=[1.0, 1.0],
        edge_provenances=["pubmed:123", "wikidata:Q42"]
    )
    # Expected: 1.0 * 1.0 = 1.0
    assert grounding_score(p2) == 1.0
    
    # 3. Path with low confidence but verified provenance
    p3 = TraversalPath(
        edge_confidences=[0.5, 0.8],
        edge_provenances=["pubmed:123", "wikidata:Q42"]
    )
    # Expected: 0.5 * 0.8 = 0.4
    assert pytest.approx(grounding_score(p3), rel=1e-3) == 0.4

def test_strict_provenance_ingestion():
    # Pipeline without strict enforcement
    pipe_lazy = IngestionPipeline(strict_provenance=False)
    edge_lazy = pipe_lazy.process("A", "B", "KNOWS", {})
    assert edge_lazy.confidence == 1.0
    
    # Pipeline WITH strict enforcement
    pipe_strict = IngestionPipeline(strict_provenance=True)
    edge_strict = pipe_strict.process("A", "B", "KNOWS", {})
    # No provenance provided -> penalized to 0.1
    assert edge_strict.confidence == 0.1
    
    # Provenance provided -> kept at 1.0
    edge_ok = pipe_strict.process("A", "B", "KNOWS", {"provenance": "verified_source"})
    assert edge_ok.confidence == 1.0

def test_path_scoring_with_grounding():
    # Two paths, same attention weights and coherence
    # P1 is grounded, P2 is not
    p_grounded = TraversalPath(
        attention_weights=[0.8, 0.8],
        community_sequence=[1, 1],
        edge_confidences=[1.0, 1.0],
        edge_provenances=["source1", "source2"],
        embedding=np.zeros(64)
    )
    p_ungrounded = TraversalPath(
        attention_weights=[0.8, 0.8],
        community_sequence=[1, 1],
        edge_confidences=[1.0, 1.0],
        edge_provenances=["unknown", "unknown"],
        embedding=np.zeros(64)
    )
    
    s1 = score_path(p_grounded)
    s2 = score_path(p_ungrounded)
    
    assert s1 > s2, "Grounded path should score higher than ungrounded path"

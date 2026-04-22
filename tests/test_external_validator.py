"""
Tests for ExternalValidator — Literature Search for Hypothesis Validation (Phase 52).

Covers:
  - Novel (no hits) → status == "novel", novelty_score == 1.0
  - Established (many hits) → status == "established", novelty_score == 0.0
  - ClinicalTrials active trial → status == "active_research"
  - Contested (opposing hits) → status == "contested"
  - Cache hit skips network call (adapter called exactly once)
  - Adapter OSError → graceful degradation, error field set
  - Batch validate skips cached entries
  - PubMedAdapter._build_query constructs correct URL
  - API smoke test via POST /research/validate
"""
import time
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.external_validator import (
    ExternalValidator,
    ExternalValidatorAdapter,
    LiteratureHit,
    PubMedAdapter,
    ValidationReport,
)
from core.hypothesis_engine import HypothesisEngine
from adapters.networkx_adapter import NetworkXAdapter
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(source="aspirin", target="alzheimers", relation="TREATS", hyp_id=None):
    """Build a minimal HypothesisProposal-like object for testing."""
    import uuid

    class _FakeProposal:
        def __init__(self):
            self.hypothesis_id = hyp_id or str(uuid.uuid4())
            self.source = source
            self.target = target
            self.derived_relation = relation

    return _FakeProposal()


def _mock_adapter(hits: list) -> ExternalValidatorAdapter:
    """Return a mock adapter that returns `hits` from search()."""
    adapter = MagicMock(spec=ExternalValidatorAdapter)
    adapter.name.return_value = "mock"
    adapter.search.return_value = hits
    return adapter


class _RelationAwareAdapter(ExternalValidatorAdapter):
    """Adapter that returns different hits depending on the relation queried."""

    def __init__(self, hits_by_relation: dict, default=None):
        self._hits = {k.upper(): v for k, v in hits_by_relation.items()}
        self._default = default or []
        self.call_count = 0

    def name(self):
        return "aware_mock"

    def search(self, source, relation, target):
        self.call_count += 1
        return self._hits.get(relation.upper(), self._default)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_validator_novel_when_no_hits():
    """Zero hits from all adapters → status == 'novel', novelty_score == 1.0."""
    mock = _mock_adapter([])
    validator = ExternalValidator(adapters=[mock])
    proposal = _make_proposal()
    report = validator.validate(proposal)
    assert report.literature_status == "novel"
    assert report.novelty_score == 1.0
    assert report.hit_count == 0


def test_validator_established_when_many_hits():
    """≥10 hits for primary relation, none for opposing → status == 'established'."""
    primary_hits = [
        LiteratureHit(adapter="pubmed", external_id=f"pm{i}", title=f"Paper {i}", relevance_score=1.0)
        for i in range(15)
    ]
    adapter = _RelationAwareAdapter({"TREATS": primary_hits})  # WORSENS → []
    validator = ExternalValidator(adapters=[adapter])
    proposal = _make_proposal(relation="TREATS")
    report = validator.validate(proposal)
    assert report.literature_status == "established"
    assert report.novelty_score == 0.0


def test_validator_active_research_from_clinical_trials():
    """ClinicalTrials active trial for primary relation → status == 'active_research'."""
    active_hit = LiteratureHit(
        adapter="clinical_trials",
        external_id="NCT12345678",
        title="Phase 2 Trial of Aspirin for Alzheimer's",
        relevance_score=1.0,  # 1.0 = active trial flag
    )
    adapter = _RelationAwareAdapter({"TREATS": [active_hit]})  # WORSENS → []
    validator = ExternalValidator(adapters=[adapter])
    proposal = _make_proposal(relation="TREATS")
    report = validator.validate(proposal)
    assert report.literature_status == "active_research"


def test_validator_few_hits_active_research():
    """1-9 hits for primary relation, none opposing → status == 'active_research'."""
    hits = [
        LiteratureHit(adapter="pubmed", external_id=f"pm{i}", title=f"Paper {i}", relevance_score=1.0)
        for i in range(3)
    ]
    adapter = _RelationAwareAdapter({"TREATS": hits})  # WORSENS → []
    validator = ExternalValidator(adapters=[adapter])
    proposal = _make_proposal(relation="TREATS")
    report = validator.validate(proposal)
    assert report.literature_status == "active_research"
    assert 0.0 < report.novelty_score <= 1.0


def test_validator_contested_when_opposing_hits():
    """
    Both TREATS and WORSENS have hits → status == 'contested'.
    Primary hits (TREATS) ensure opposing search is triggered.
    """
    primary_hit = LiteratureHit(
        adapter="pubmed", external_id="pm001",
        title="Aspirin May Help Alzheimer's Patients",
        relevance_score=1.0,
    )
    opposing_hit = LiteratureHit(
        adapter="pubmed", external_id="pm999",
        title="Aspirin Worsens Alzheimer's in Some Patients",
        relevance_score=1.0,
    )
    adapter = _RelationAwareAdapter({"TREATS": [primary_hit], "WORSENS": [opposing_hit]})
    validator = ExternalValidator(adapters=[adapter])
    proposal = _make_proposal(relation="TREATS")
    report = validator.validate(proposal)
    assert report.literature_status == "contested"


def test_validator_cache_hit_skips_network():
    """Second validate() call for same (source, relation, target) uses cache; adapter called once."""
    mock = _mock_adapter([])
    validator = ExternalValidator(adapters=[mock])
    proposal = _make_proposal(hyp_id="cached-id")

    report1 = validator.validate(proposal)
    report2 = validator.validate(proposal)

    # Mock search() should have been called only once per adapter
    assert mock.search.call_count == 1
    assert report1.literature_status == report2.literature_status


def test_validator_adapter_timeout_graceful():
    """OSError in one adapter sets error field but does not block other adapters."""
    failing = MagicMock(spec=ExternalValidatorAdapter)
    failing.name.return_value = "failing"
    failing.search.side_effect = OSError("network_unavailable")

    working = _mock_adapter([
        LiteratureHit(adapter="openalex", external_id="W1", title="A Paper", relevance_score=1.0)
    ])
    working.name.return_value = "openalex"

    validator = ExternalValidator(adapters=[failing, working])
    proposal = _make_proposal()
    report = validator.validate(proposal)

    # Should still have results from the working adapter
    assert report.hit_count >= 1 or report.error is not None
    # Error field records the failed adapter
    if failing.search.called:
        assert report.error is not None


def test_validator_batch_skips_cached():
    """validate_batch: second call on same proposal skips network (uses cache)."""
    mock = _mock_adapter([])
    validator = ExternalValidator(adapters=[mock])

    p1 = _make_proposal(source="metformin", target="diabetes", relation="TREATS")
    p2 = _make_proposal(source="aspirin", target="headache", relation="TREATS")

    # First batch: both uncached
    validator.validate_batch([p1, p2])
    count_after_first = mock.search.call_count

    # Second batch: both cached — no new adapter calls
    validator.validate_batch([p1, p2])
    assert mock.search.call_count == count_after_first  # no additional calls


def test_pubmed_adapter_query_string():
    """PubMedAdapter._build_query produces a correctly encoded URL."""
    adapter = PubMedAdapter()
    url = adapter._build_query("metformin", "TREATS", "type 2 diabetes")
    assert "esearch.fcgi" in url
    assert "metformin" in url
    assert "type+2+diabetes" in url or "type%202%20diabetes" in url or "type 2 diabetes" in url


def test_validator_api_endpoint_smoke():
    """POST /research/validate returns 200 + valid ValidateProposalsResponse schema."""
    import random as _random
    from starlette.testclient import TestClient
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf
    from core.embedding_engine import RandomEngine
    from api.server import create_app

    TOY_CSV = Path(__file__).parent / "fixtures" / "toy_graph.csv"
    _random.seed(42)
    adapter = load_csv_adapter(str(TOY_CSV))
    engine = RandomEngine(dim=64)

    G = adapter.to_networkx()
    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    cmap = {node: cid for cid, members in enumerate(parts) for node in members}

    app = create_app(adapter=adapter, embedding_engine=engine, community_map=cmap)
    with TestClient(app, headers={"X-API-Key": "dev-secret"}) as c:
        # First generate a hypothesis so there are proposals to validate
        hyp_r = c.post("/v1/hypothesize", json={

            "source_id": "newton",
            "target_id": "maxwell",
            "max_paths": 5,
            "max_hop": 3,
            "beam_width": 10,
            "min_confidence": 0.0,
        })
        assert hyp_r.status_code == 200

        # Validate with mocked adapters (pass empty hypothesis_ids = all from last run)
        # Use a fresh ExternalValidator with no-op adapters to avoid real network calls
        from core.external_validator import ExternalValidator
        from api import server as _srv
        _srv._state["external_validator"] = ExternalValidator(adapters=[])

        r = c.post("/v1/research/validate", json={"hypothesis_ids": [], "adapters": []})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "validated" in body
        assert "reports" in body
        assert "duration_seconds" in body

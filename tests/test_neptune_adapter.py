"""
Tests for Phase 23 — Amazon Neptune Adapter.

Verifies graph operations across a mocked Gremlin WebSocket connection.
"""
import pytest
from unittest.mock import MagicMock, patch
from adapters.neptune_adapter import NeptuneAdapter

@pytest.fixture
def mock_gremlin_g():
    g = MagicMock()
    return g

@pytest.fixture
def adapter(mock_gremlin_g):
    adp = NeptuneAdapter()
    adp.g = mock_gremlin_g
    return adp


def test_neptune_not_connected():
    adp = NeptuneAdapter()
    with pytest.raises(RuntimeError, match="Call connect()"):
        adp.get_entity("test")


def test_get_entity_success(adapter, mock_gremlin_g):
    # Mock g.V().has().valueMap().toList()
    chain = mock_gremlin_g.V.return_value.has.return_value.valueMap.return_value.toList
    chain.return_value = [{"name": ["einstein"], "type": ["scientist"]}]

    ent = adapter.get_entity("einstein")
    assert ent is not None
    assert ent.id == "einstein"
    assert ent.label == "einstein"
    assert ent.type == "scientist"


def test_get_entity_not_found(adapter, mock_gremlin_g):
    chain = mock_gremlin_g.V.return_value.has.return_value.valueMap.return_value.toList
    chain.return_value = []

    ent = adapter.get_entity("einstein")
    assert ent is None


def test_get_neighbors_success(adapter, mock_gremlin_g):
    class MockElement:
        def __init__(self, label, props=None):
            self.label = label
            self.properties = props or {}
            self.id = "mock_id"

    # path list element = [v_out, e, v_in]
    v_out = MockElement("Entity", {"name": ["einstein"]})
    e = MockElement("WORKS_AT")
    v_in = MockElement("Entity", {"name": ["ias"]})

    chain = mock_gremlin_g.V.return_value.has.return_value.outE.return_value.inV.return_value.path.return_value.limit.return_value.toList
    chain.return_value = [
        [v_out, e, v_in]
    ]

    edges = adapter.get_neighbors("einstein")
    assert len(edges) == 1
    assert edges[0].source_id == "einstein"
    assert edges[0].target_id == "ias"
    assert edges[0].relation_type == "WORKS_AT"


def test_get_neighbors_no_results(adapter, mock_gremlin_g):
    chain = mock_gremlin_g.V.return_value.has.return_value.outE.return_value.inV.return_value.path.return_value.limit.return_value.toList
    chain.return_value = []

    edges = adapter.get_neighbors("einstein")
    assert len(edges) == 0


def test_abstract_overrides(adapter):
    assert adapter.get_community("test") == -1
    assert adapter.get_embedding("test") is None
    assert adapter.find_similar(None) == []
    # to_networkx returns empty graph
    assert len(adapter.to_networkx().nodes()) == 0

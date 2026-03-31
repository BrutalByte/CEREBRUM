"""
Tests for Phase 23 — Neo4j Enterprise Bulk-Loader

Covers:
  - create_indices() executing CREATE INDEX
  - bulk_load() generating UNWIND queries for nodes
  - bulk_load() generating grouped UNWIND queries for edges by relationship type
  - batching logic properly slicing large arrays
"""
import pytest
from unittest.mock import MagicMock, call
from adapters.neo4j_adapter import Neo4jAdapter

# ---------------------------------------------------------------------------
# Mocks & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value = session
    session.__enter__.return_value = session
    return driver

@pytest.fixture
def adapter(mock_driver):
    adp = Neo4jAdapter()
    adp._driver = mock_driver
    return adp

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_indices(adapter, mock_driver):
    session = mock_driver.session.return_value
    adapter.create_indices()
    
    session.run.assert_called_once()
    cypher = session.run.call_args[0][0]
    assert "CREATE INDEX IF NOT EXISTS" in cypher
    assert "(n:Entity) ON (n.name)" in cypher


def test_bulk_load_nodes(adapter, mock_driver):
    session = mock_driver.session.return_value
    nodes = [
        {"name": "A", "type": "person", "age": 30},
        {"name": "B", "type": "person", "age": 25},
    ]
    adapter.bulk_load(nodes, [])
    
    session.run.assert_called_once()
    cypher, params = session.run.call_args[0]
    
    assert "UNWIND $batch AS row" in cypher
    assert "MERGE (n:Entity {name: row.name})" in cypher
    assert "SET n += row" in cypher
    assert params["batch"] == nodes


def test_bulk_load_edges_grouping(adapter, mock_driver):
    session = mock_driver.session.return_value
    edges = [
        {"source": "A", "target": "B", "relation": "KNOWS"},
        {"source": "A", "target": "C", "relation": "LIKES"},
        {"source": "B", "target": "C", "relation": "KNOWS"},
    ]
    adapter.bulk_load([], edges)
    
    # Needs 2 calls because we have 2 distinct relations: KNOWS, LIKES
    assert session.run.call_count == 2
    
    calls = session.run.call_args_list
    cyphers = [c[0][0] for c in calls]
    
    # Ensure both relationship types have distinct MERGE statements
    assert any("MERGE (a)-[r:KNOWS]->(b)" in c for c in cyphers)
    assert any("MERGE (a)-[r:LIKES]->(b)" in c for c in cyphers)


def test_bulk_load_batching(adapter, mock_driver):
    session = mock_driver.session.return_value
    
    # 25 nodes, batch size 10 => 3 batches (10, 10, 5)
    nodes = [{"name": f"N{i}"} for i in range(25)]
    adapter.bulk_load(nodes, [], batch_size=10)
    
    assert session.run.call_count == 3
    
    batches = [c[0][1]["batch"] for c in session.run.call_args_list]
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5

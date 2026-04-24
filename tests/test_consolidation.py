import pytest
import asyncio
from typing import List
from unittest.mock import MagicMock
from core.consolidation_engine import ConsolidationEngine

# Create a mock Trace class that matches the expected structure in ConsolidationEngine
class MockTrace:
    def __init__(self, path: List[str]):
        self.path = path

def test_consolidation_edge_creation():
    mock_adapter = MagicMock()
    mock_log = MagicMock()
    mock_graph = MagicMock()
    
    # Mock traces: 6 occurrences of path A->B->C
    mock_log.get_recent_traces.return_value = [
        MockTrace(path=["A", "B", "C"]) for _ in range(6)
    ]
    
    engine = ConsolidationEngine(mock_adapter, mock_graph, query_log=mock_log, threshold=5)
    
    # Run consolidation
    asyncio.run(engine.run_rem_cycle())
    
    # Verify edge creation
    assert mock_adapter.add_edge.called
    # Check arguments
    args, kwargs = mock_adapter.add_edge.call_args
    assert kwargs['u'] == "A"
    assert kwargs['v'] == "C"
    assert kwargs['relation'] == "REM_SHORTCUT"
    assert kwargs['synthetic'] is True
    
    # Verify telemetry
    assert mock_graph.emit.called

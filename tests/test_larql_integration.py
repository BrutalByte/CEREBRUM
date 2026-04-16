import pytest
from unittest.mock import MagicMock, patch
from core.larql_client import LarqlClient, LarqlNeuralLink
from core.external_validator import LarqlAdapter
from core.research_agent import ResearchAgent, ResearchCandidate
from core.hypothesis_engine import HypothesisProposal

@pytest.fixture
def mock_larql_client():
    client = MagicMock(spec=LarqlClient)
    return client

class TestLarqlIntegration:

    def test_larql_adapter_search(self, mock_larql_client):
        """LarqlAdapter uses LarqlClient to return literature hits."""
        mock_larql_client.query_relation.return_value = LarqlNeuralLink(
            source="A", relation="TREATS", target="B", confidence=0.85, metadata={}
        )
        
        with patch("core.larql_client.LarqlClient", return_value=mock_larql_client):
            adapter = LarqlAdapter(endpoint="http://localhost:8000")
            hits = adapter.search("A", "TREATS", "B")
            
            assert len(hits) == 1
            assert hits[0].adapter == "larql"
            assert hits[0].relevance_score == 0.85
            assert "Neural Consensus" in hits[0].title

    def test_larql_adapter_no_hit(self, mock_larql_client):
        """LarqlAdapter returns empty list if confidence is low."""
        mock_larql_client.query_relation.return_value = LarqlNeuralLink(
            source="A", relation="TREATS", target="B", confidence=0.05, metadata={}
        )
        
        with patch("core.larql_client.LarqlClient", return_value=mock_larql_client):
            adapter = LarqlAdapter(endpoint="http://localhost:8000")
            hits = adapter.search("A", "TREATS", "B")
            assert len(hits) == 0

    def test_research_agent_larql_scan(self, mock_larql_client):
        """ResearchAgent discovers candidates via Larql scan."""
        mock_larql_client.find_neural_neighbors.return_value = [
            LarqlNeuralLink(source="newton", relation="INFLUENCED", target="einstein", confidence=0.9, metadata={})
        ]
        
        # Setup minimal agent environment
        mock_adapter = MagicMock()
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["newton", "einstein"]
        mock_graph.has_edge.return_value = False
        mock_graph.successors.return_value = []
        mock_graph.predecessors.return_value = []
        mock_graph.subgraph.return_value.number_of_edges.return_value = 0
        mock_adapter.to_networkx.return_value = mock_graph
        mock_adapter.get_embedding.return_value = None
        
        mock_hypo = MagicMock()
        
        with patch("core.larql_client.LarqlClient", return_value=mock_larql_client):
            with patch("random.sample", side_effect=lambda l, k: l[:k]): # Deterministic sample
                agent = ResearchAgent(mock_adapter, mock_hypo, larql_endpoint="http://larql")
                
                # Manually trigger internal _larql_scan
                G = mock_adapter.to_networkx.return_value
                nodes = ["newton", "einstein"]
                cands = agent._larql_scan(G, nodes, {}, {}, set())
                
                assert len(cands) >= 1
                assert cands[0].seeded_by == "larql_neural_scan"
                assert cands[0].source_id == "newton"
                assert cands[0].target_id == "einstein"

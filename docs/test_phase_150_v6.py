import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath('..'))

from core.cerebrum import CerebrumGraph
from core.frontal_engine import ReasoningStrategy
from reasoning.traversal import TraversalPath

class TestPhase150(unittest.TestCase):
    @patch('core.cerebrum.BeamTraversal')
    @patch('core.cerebrum.extract')
    def test_frontal_engine_deep_strategy(self, MockExtract, MockTraversal):
        # Setup mocks
        mock_adapter = MagicMock()
        mock_adapter.to_networkx.return_value = MagicMock()
        mock_emb = MagicMock()
        
        graph = CerebrumGraph(adapter=mock_adapter, embedding_engine=mock_emb, max_hop=1)
        graph._built = True 
        
        mock_agent = MagicMock()
        graph.set_research_agent(mock_agent)
        
        # Force DEEP strategy by setting high arousal
        graph.modulator.update_arousal(0.9)
        
        # Setup mock traversal instance
        mock_traversal_instance = MockTraversal.return_value
        mock_traversal_instance.epistemic_gaps = [{"source": "A", "target": "B", "score": 0.1}]
        mock_traversal_instance._last_entropy = 3.0
        mock_traversal_instance._beam_widths = {}
        
        # Setup paths
        path_low = MagicMock(spec=TraversalPath)
        path_low.score = 0.1
        path_low.nodes = ["A", "r", "B"]
        path_low.tail = "B"
        path_low.hop_depth = 1
        
        path_high = MagicMock(spec=TraversalPath)
        path_high.score = 0.9
        path_high.nodes = ["A", "r", "B"]
        path_high.tail = "B"
        path_high.hop_depth = 1
        
        mock_traversal_instance.traverse.side_effect = [[path_low], [path_high]]
        
        # Ensure graph uses our mock instance
        graph._traversal = mock_traversal_instance
        
        # Run query
        results = graph.query(["A"], max_hop=1)
        
        # Verify
        self.assertTrue(mock_agent.push_candidate.called)
        self.assertTrue(mock_agent.scan_once.called)
        self.assertEqual(mock_traversal_instance.traverse.call_count, 2)
        print("TestPhase150: Success")

if __name__ == "__main__":
    unittest.main()

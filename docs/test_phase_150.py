import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath('..'))

from core.cerebrum import CerebrumGraph
from core.frontal_engine import ReasoningStrategy
from reasoning.traversal import TraversalPath

class TestPhase150(unittest.TestCase):
    def test_frontal_engine_deep_strategy(self):
        # Setup mocks
        mock_adapter = MagicMock()
        mock_adapter.to_networkx.return_value = MagicMock()
        mock_cid = MagicMock()
        mock_cid.__ge__.return_value = True
        mock_adapter.get_community.return_value = mock_cid
        mock_adapter.get_embedding.return_value = None
        mock_emb = MagicMock()
        
        graph = CerebrumGraph(adapter=mock_adapter, embedding_engine=mock_emb)
        graph._built = True # Phase 150: Bypass build check
        mock_agent = MagicMock()
        graph.set_research_agent(mock_agent)
        
        # Force DEEP strategy by setting high arousal
        graph.modulator.update_arousal(0.9)
        
        # Mock traversal to have epistemic gaps
        mock_traversal = MagicMock()
        mock_traversal.epistemic_gaps = [{"source": "A", "target": "B", "score": 0.1}]
        mock_traversal._last_entropy = 3.0
        
        # First call returns low-confidence paths, second call returns high-confidence
        path_low = TraversalPath(nodes=["A", "r", "B"], score=0.1)
        path_high = TraversalPath(nodes=["A", "r", "B"], score=0.9)
        
        # Mock traversal.traverse
        mock_traversal.traverse.side_effect = [[path_low], [path_high]]
        
        # Replace graph's traversal with our mock
        graph._traversal = mock_traversal
        graph._max_hop = 1  # Phase 150: Ensure needs_custom is False
        graph._beam_width = 10

        # Run query
        results = graph.query(["A"], max_hop=1)
        # Verify
        self.assertEqual(graph.frontal.determine_strategy([path_low], entropy=3.0), ReasoningStrategy.DEEP)
        self.assertTrue(mock_agent.push_candidate.called)
        self.assertTrue(mock_agent.scan_once.called)
        # Verify it retried
        self.assertEqual(mock_traversal.traverse.call_count, 2)
        print("TestPhase150: Success")

if __name__ == "__main__":
    unittest.main()

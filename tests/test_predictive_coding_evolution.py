import pytest
from unittest.mock import MagicMock
from core.predictive_coder import PredictiveCoder

def test_predictive_prior_success_weighting():
    # Setup
    mock_engram = MagicMock()
    coder = PredictiveCoder(mock_engram)
    
    # Mock engrams with different success rates
    class MockEngram:
        def __init__(self, id, freq, success):
            self.id = id
            self.frequency = freq
            self.success_rate = success
            
    engrams = [
        MockEngram("e1", 10, 1.0), # Success rate 1.0 -> weight 10
        MockEngram("e2", 10, 0.5), # Success rate 0.5 -> weight 5
    ]
    
    prior = coder.generate_prior(engrams)
    
    # Assert weighting: e1 gets 2/3, e2 gets 1/3 of prior
    assert prior["e1"] == pytest.approx(2/3)
    assert prior["e2"] == pytest.approx(1/3)

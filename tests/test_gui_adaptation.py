
import pytest
from core.gui_adaptation_engine import GUIAdaptationEngine
from core.telemetry import NeuralEvent, NeuralEventType

def test_arousal_triggers_neighborhood():
    broadcasted = []
    def mock_broadcast(evt):
        broadcasted.append(evt)
        
    engine = GUIAdaptationEngine(broadcast_fn=mock_broadcast)
    engine.threshold = 1.5 # Lower for test
    
    # Simulate high arousal (e.g. from cognitive dissonance)
    engine.process_flux({"arousal": 2.0, "reinforcement": 1.0, "cohesion": 1.0})
    
    assert len(broadcasted) == 1
    assert broadcasted[0].event_type == NeuralEventType.GUI_ADAPTATION
    assert broadcasted[0].payload["target"] == "neighborhood"

def test_reinforcement_triggers_cortical():
    broadcasted = []
    def mock_broadcast(evt):
        broadcasted.append(evt)
        
    engine = GUIAdaptationEngine(broadcast_fn=mock_broadcast)
    engine.threshold = 1.5
    
    # Simulate high reinforcement (e.g. from successful validation)
    engine.process_flux({"arousal": 1.0, "reinforcement": 2.0, "cohesion": 1.0})
    
    assert len(broadcasted) == 1
    assert broadcasted[0].payload["target"] == "cortical"

def test_cohesion_triggers_clustered():
    broadcasted = []
    def mock_broadcast(evt):
        broadcasted.append(evt)
        
    engine = GUIAdaptationEngine(broadcast_fn=mock_broadcast)
    engine.threshold = 1.5
    
    # Start in cortical, then shift back to clustered
    engine.last_layout = "cortical"
    engine.process_flux({"arousal": 1.0, "reinforcement": 1.0, "cohesion": 2.0})
    
    assert len(broadcasted) == 1
    assert broadcasted[0].payload["target"] == "clustered"

"""
GUIAdaptationEngine (Phase 94).

Monitors metabolic flux and triggers real-time 3D organization shifts
to mirror the system's "cognitive state" in the UI.
"""
import logging
from typing import Dict, Any, Optional
from core.telemetry import NeuralEvent, NeuralEventType

logger = logging.getLogger("cerebrum.adaptation")

class GUIAdaptationEngine:
    """
    Translates internal metabolic signals into GUI layout shift commands.
    
    Logic:
      - High Arousal (>1.5): Shift to 'neighborhood' mode to focus on the 
        immediate "dissonance" area.
      - High Cohesion (>1.5): Shift to 'clustered' to show the structural 
        stability of the communities.
      - High Reinforcement (>1.5): Shift to 'cortical' columns to show the
        efficient "white matter" pathways.
    """
    def __init__(self, broadcast_fn):
        self.broadcast_fn = broadcast_fn
        self.last_layout = "clustered"
        self.threshold = 1.6

    def process_flux(self, state: Dict[str, float]):
        arousal = state.get("arousal", 1.0)
        cohesion = state.get("cohesion", 1.0)
        reinforcement = state.get("reinforcement", 1.0)
        
        target_layout = None
        
        if arousal > self.threshold:
            target_layout = "neighborhood"
        elif reinforcement > self.threshold:
            target_layout = "cortical"
        elif cohesion > self.threshold:
            target_layout = "clustered"
            
        if target_layout and target_layout != self.last_layout:
            logger.info(f"Metabolic Adaptation: Shifting GUI layout to '{target_layout}'")
            self.last_layout = target_layout
            
            self.broadcast_fn(NeuralEvent(
                event_type=NeuralEventType.GUI_ADAPTATION,
                payload={
                    "action": "layout_shift",
                    "target": target_layout,
                    "note": f"Metabolic trigger: {'Arousal' if target_layout == 'neighborhood' else 'Reinforcement' if target_layout == 'cortical' else 'Cohesion'}"
                }
            ))

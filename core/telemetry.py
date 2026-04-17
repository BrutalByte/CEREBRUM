"""
Neural Telemetry System for CEREBRUM (Phase 63+).

Standardizes real-time event emission for external visualization clients 
(e.g., Unreal Engine 5).
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import time


class NeuralCommandType(str, Enum):
    """Commands sent from the client to the CEREBRUM server (Phase 90)."""
    TRIGGER_QUERY = "TRIGGER_QUERY"   # Start a traversal from a seed
    PROBE_NODE    = "PROBE_NODE"      # Get details on a node (inspect)
    METABOLIC_SYNC = "METABOLIC_SYNC" # Trigger homeostatic step
    FOCUS_SHIFT    = "FOCUS_SHIFT"    # Change the current attention focus

class NeuralEventType(str, Enum):
    """Types of events emitted by the CEREBRUM Brain Server."""
    SYNAPTIC_PULSE = "SYNAPTIC_PULSE"   # A beam expansion step (traversal)
    NEUROGENESIS   = "NEUROGENESIS"     # New node created (ResearchAgent)
    SYNAPTOGENESIS = "SYNAPTOGENESIS"   # New edge created (REM/Synthesis)
    SYNAPTIC_PRUNE = "SYNAPTIC_PRUNE"   # Edge removed (SynapticPruner)
    CORTICAL_GLOW  = "CORTICAL_GLOW"    # Community activation
    DISSONANCE     = "DISSONANCE"       # CEC dissonance event
    METABOLIC_FLUX = "METABOLIC_FLUX"   # Phase 88 metabolism update
    ENGRAM_STEER   = "ENGRAM_STEER"     # Phase 87 pattern steering

class NeuralEventHeader(BaseModel):
    """Metadata for every neural event."""
    timestamp: float = Field(default_factory=time.time)
    version: str = "2.4.0"
    layer: str = "CORTEX"

class NeuralEvent(BaseModel):
    """The standard packet for external visualization."""
    header: NeuralEventHeader = Field(default_factory=NeuralEventHeader)
    event_type: NeuralEventType
    payload: Dict[str, Any]

    @classmethod
    def pulse(
        cls, 
        source: str, 
        target: str, 
        relation: str, 
        weight: float, 
        hop: int, 
        features: Optional[List[float]] = None,
        is_wormhole: bool = False
    ) -> "NeuralEvent":
        """Helper to create a traversal pulse event."""
        return cls(
            event_type=NeuralEventType.SYNAPTIC_PULSE,
            payload={
                "source_node": source,
                "target_node": target,
                "relation": relation,
                "weight": float(weight),
                "hop_count": hop,
                "features": features or [],
                "is_wormhole": is_wormhole
            }
        )

    @classmethod
    def genesis(cls, node_id: str, label: str, node_type: str) -> "NeuralEvent":
        """Helper to create a node creation event."""
        return cls(
            event_type=NeuralEventType.NEUROGENESIS,
            payload={
                "node_id": node_id,
                "label": label,
                "type": node_type
            }
        )

    @classmethod
    def flux(cls, state: Dict[str, float], learning_rate_scale: float) -> "NeuralEvent":
        """Helper to create a metabolic flux event (Phase 88)."""
        return cls(
            event_type=NeuralEventType.METABOLIC_FLUX,
            payload={
                "state": state,
                "learning_rate_scale": learning_rate_scale
            }
        )

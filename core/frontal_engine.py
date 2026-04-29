"""
Phase 150 — Frontal Engine.

Executive control for tiered reasoning strategies: Fast (Reflex), Hybrid (Async),
and Deep (Blocking Research).

Inspired by the dorsolateral prefrontal cortex (DLPFC) and its role in
cognitive control and rule-based decision making.
"""
from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.chemical_modulator import ChemicalModulator

logger = logging.getLogger("cerebrum.frontal")

class ReasoningStrategy(str, Enum):
    FAST      = "fast"       # Reflex: standard traversal, no retries
    HYBRID    = "hybrid"     # Curiosity: return best result, async research gap
    DEEP      = "deep"       # Obsession: suspend, research gap, materialize, retry

@dataclass
class EpistemicGap:
    """Detected logical gap during reasoning."""
    source_id: str
    target_id: str
    confidence_deficit: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class FrontalEngine:
    """
    Executive controller that determines the reasoning strategy based on
    metabolic state and query outcomes.
    """

    def __init__(
        self,
        modulator: Optional[ChemicalModulator] = None,
        strategy_thresholds: Optional[Dict[str, float]] = None
    ):
        self.modulator = modulator
        # Default thresholds for strategy escalation
        self.thresholds = strategy_thresholds or {
            "deep_arousal": 0.8,    # Escalates to DEEP if arousal > 0.8
            "hybrid_novelty": 0.5,  # Escalates to HYBRID if novelty > 0.5
            "gap_entropy": 2.5      # Conflict entropy trigger
        }

    def determine_strategy(
        self,
        paths: List[Any],
        entropy: float = 0.0,
        manual_override: Optional[ReasoningStrategy] = None
    ) -> ReasoningStrategy:
        """
        Determine the appropriate reasoning strategy for the current context.
        """
        if manual_override:
            return manual_override

        # Get metabolic state
        arousal = 0.0
        novelty = 0.0
        if self.modulator:
            state = self.modulator.state
            arousal = state.get("arousal", 0.0)
            novelty = state.get("novelty", 0.0)

        # Strategy Escalation Logic
        if arousal > self.thresholds["deep_arousal"]:
            logger.info("FrontalEngine: Escalating to DEEP reasoning (Arousal=%.3f)", arousal)
            return ReasoningStrategy.DEEP
        
        if novelty > self.thresholds["hybrid_novelty"] or entropy > self.thresholds["gap_entropy"]:
            logger.info("FrontalEngine: Escalating to HYBRID reasoning (Novelty=%.3f, Entropy=%.3f)", novelty, entropy)
            return ReasoningStrategy.HYBRID

        return ReasoningStrategy.FAST

    def detect_gaps(self, paths: List[Any]) -> List[EpistemicGap]:
        """
        Analyze paths to identify missing logical links.
        (Placeholder for Step 2 logic)
        """
        gaps = []
        if not paths:
            return gaps
        
        # Simple heuristic: if top path score is low (< 0.1) but it has many nodes
        # it might be a weak chain that needs a "bridge".
        top_path = paths[0]
        if hasattr(top_path, 'score') and top_path.score < 0.1 and len(getattr(top_path, 'nodes', [])) > 3:
            gaps.append(EpistemicGap(
                source_id=top_path.nodes[0],
                target_id=top_path.tail,
                confidence_deficit=1.0 - top_path.score,
                description="Low-confidence long-chain path."
            ))
            
        return gaps

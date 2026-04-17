"""
ActiveInferenceEngine - Phase 93: Daydreaming Mode.
"""
from __future__ import annotations
import logging
import random
import time
from typing import Any, Dict, List, Optional
from core.telemetry import NeuralEvent, NeuralEventType

logger = logging.getLogger("cerebrum.active_inference")

class ActiveInferenceEngine:
    def __init__(self, graph: Any, min_soliton_threshold: float = 0.7, metabolic_floor: float = 0.2) -> None:
        self.graph = graph
        self.min_soliton_threshold = min_soliton_threshold
        self.metabolic_floor = metabolic_floor

    def step(self) -> Optional[Dict[str, Any]]:
        reinforcement = getattr(self.graph.modulator, "reinforcement", 1.0)
        if reinforcement < self.metabolic_floor: return None
        seeds = self._select_seeds()
        if not seeds: return None
        t0 = time.time()
        try:
            reason = "high_pe" if self._is_dissonant(seeds) else "exploration"
            self.graph.emit(NeuralEvent.inference_pulse(seeds=seeds, reason=reason, state=self.graph.modulator.state))
            answers = self.graph.query(seeds=seeds, top_k=5, max_hop=2, beam_width=5, max_loops=2)
            return {"seeds": seeds, "answers_found": len(answers), "duration": time.time() - t0}
        except Exception as e:
            logger.error(f"ActiveInference step error: {e}")
            return None

    def _select_seeds(self) -> List[str]:
        # 1. Predictive Coding
        if self.graph.predictive_coder:
            try:
                stats = self.graph.predictive_coder.soliton_stats()
                dissonant = [k for k, v in stats.items() if v < self.min_soliton_threshold]
                if dissonant: return min(dissonant, key=lambda k: stats[k]).split("|")
            except Exception: pass

        # 2. Random selection via adapter (DEBUGGING: Ensure access)
        try:
            all_nodes = list(self.graph.adapter.to_networkx().nodes())
            if all_nodes: return [random.choice(all_nodes)]
        except Exception as e: 
            logger.error(f"Seed selection error: {e}")
            pass
        return []

    def _is_dissonant(self, seeds: List[str]) -> bool:
        if not self.graph.predictive_coder: return False
        key = "|".join(sorted(seeds))
        stats = self.graph.predictive_coder.soliton_stats()
        return stats.get(key, 1.0) < self.min_soliton_threshold

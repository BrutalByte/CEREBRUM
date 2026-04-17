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

    def step(self, context_seeds: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Run one idle inference cycle.

        Args:
            context_seeds: Goal-directed seeds from GoalEvaluator. When provided
                these are merged with high-PE seeds before querying.
        """
        reinforcement = getattr(self.graph.modulator, "reinforcement", 1.0)
        if reinforcement < self.metabolic_floor:
            return None
        seeds = self._select_seeds(context_seeds=context_seeds)
        if not seeds:
            return None
        t0 = time.time()
        try:
            reason = "goal_directed" if context_seeds else ("high_pe" if self._is_dissonant(seeds) else "exploration")
            self.graph.emit(NeuralEvent.inference_pulse(seeds=seeds, reason=reason, state=self.graph.modulator.state))
            answers = self.graph.query(seeds=seeds, top_k=5, max_hop=2, beam_width=5, max_loops=2)
            return {"seeds": seeds, "answers_found": len(answers), "duration": time.time() - t0}
        except Exception as e:
            logger.error(f"ActiveInference step error: {e}")
            return None

    def _select_seeds(self, context_seeds: Optional[List[str]] = None) -> List[str]:
        """Select seeds, prioritising goal-directed context over high-PE nodes."""
        _MAX_SEEDS = 3
        result: List[str] = list(context_seeds or [])[:_MAX_SEEDS]

        # Augment with high-PE nodes up to _MAX_SEEDS total
        if len(result) < _MAX_SEEDS and self.graph.predictive_coder:
            try:
                stats = self.graph.predictive_coder.soliton_stats()
                dissonant = [k for k, v in stats.items() if v < self.min_soliton_threshold]
                if dissonant:
                    best = min(dissonant, key=lambda k: stats[k]).split("|")
                    for s in best:
                        if s not in result and len(result) < _MAX_SEEDS:
                            result.append(s)
            except Exception:
                pass

        if result:
            return result

        # Fallback: random node
        try:
            all_nodes = list(self.graph.adapter.to_networkx().nodes())
            if all_nodes:
                return [random.choice(all_nodes)]
        except Exception as e:
            logger.error(f"Seed selection error: {e}")
        return []

    def _is_dissonant(self, seeds: List[str]) -> bool:
        if not self.graph.predictive_coder: return False
        key = "|".join(sorted(seeds))
        stats = self.graph.predictive_coder.soliton_stats()
        return stats.get(key, 1.0) < self.min_soliton_threshold

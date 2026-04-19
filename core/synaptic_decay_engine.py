"""
SynapticDecayEngine — Phase 97: Synaptic Decay (LTD / Synaptic Homeostasis).

Counterpart to Phase 96's Hebbian LTP. Periodically decays all edge weights
toward a baseline, with resistance proportional to recent traversal frequency.
Edges used often resist decay; unused edges fade back to baseline.

This is the KG analogue of the Synaptic Homeostasis Hypothesis (Tononi & Cirelli):
during sleep, all synapses downscale proportionally, then LTP selectively restores
the important ones — preventing saturation and preserving discriminability.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger("cerebrum.synaptic_decay")

# Source-based frequency weight multipliers (Phase 98 Gap 8).
# Higher multiplier → higher traversal_freq → higher resistance → SLOWER decay.
# Dissonance (failed paths) should decay FASTER → low multiplier (0.5).
# Approval/insight (successful paths) should decay SLOWER → high multiplier.
_DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "dissonance": 0.5,   # low freq contribution → less resistance → faster decay
    "approval":   3.0,   # high freq contribution → more resistance → slower decay
    "insight":    2.0,   # high freq contribution → more resistance → slower decay
}


@dataclass
class DecayResult:
    """Outcome summary for a single synaptic decay pass."""
    edges_processed: int
    edges_decayed: int       # weight moved away from baseline
    edges_resisted: int      # high-frequency, barely changed
    mean_delta: float        # average |delta| applied
    duration: float


class SynapticDecayEngine:
    """Hebbian LTD engine — decays edge weights toward baseline.

    Parameters
    ----------
    adapter        : graph adapter (must implement update_edge_weight with min_weight)
    graph          : CerebrumGraph — for telemetry emit()
    baseline_weight: target weight (default 1.0 — natural resting weight)
    decay_rate     : fraction of gap closed per pass (default 0.01)
    min_weight     : absolute floor — never decay below this (default 0.5)
    resistance_k   : traversal count at which resistance = 50% (default 5.0)
    source_weights : per-source frequency multipliers (Phase 98 Gap 8)
    """

    def __init__(
        self,
        adapter: Any,
        graph: Any,
        baseline_weight: float = 1.0,
        decay_rate: float = 0.01,
        min_weight: float = 0.5,
        resistance_k: float = 5.0,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.adapter = adapter
        self.graph = graph
        self.baseline_weight = baseline_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.resistance_k = resistance_k
        self.source_weights = source_weights if source_weights is not None else dict(_DEFAULT_SOURCE_WEIGHTS)

    def decay(self, wm: Optional["WorkingMemoryBuffer"] = None) -> DecayResult:
        """Apply one synaptic homeostasis pass over all graph edges.

        Traversal frequency is estimated from wm.recent(100).path_edges.
        High-frequency edges receive less decay; unused edges decay at full rate.
        """
        t0 = time.time()
        freq = self._build_frequency_map(wm)

        G = self.adapter.to_networkx()
        edges_processed = 0
        edges_decayed = 0
        edges_resisted = 0
        total_abs_delta = 0.0

        for u, v, data in G.edges(data=True):
            relation = data.get("relation", "")
            current_weight = data.get("weight", 1.0)
            triple = (u, relation, v)

            traversal_count = freq.get(triple, 0)
            # Sigmoid-like resistance in [0, 1): freq / (freq + k)
            resistance = traversal_count / (traversal_count + self.resistance_k)
            effective_rate = self.decay_rate * (1.0 - resistance)
            delta = effective_rate * (self.baseline_weight - current_weight)

            if abs(delta) < 1e-9:
                edges_processed += 1
                continue

            n = self.adapter.update_edge_weight(
                u, v, relation,
                delta=delta,
                max_weight=max(current_weight, self.baseline_weight) + 0.01,
                min_weight=self.min_weight,
            )
            if n:
                edges_processed += 1
                total_abs_delta += abs(delta)
                if resistance > 0.3:
                    edges_resisted += 1
                else:
                    edges_decayed += 1
            else:
                edges_processed += 1

        mean_delta = (total_abs_delta / max(1, edges_decayed + edges_resisted))
        result = DecayResult(
            edges_processed=edges_processed,
            edges_decayed=edges_decayed,
            edges_resisted=edges_resisted,
            mean_delta=mean_delta,
            duration=time.time() - t0,
        )
        self._emit(result)
        logger.debug(
            "SynapticDecay: processed=%d decayed=%d resisted=%d mean_delta=%.5f",
            result.edges_processed, result.edges_decayed,
            result.edges_resisted, result.mean_delta,
        )
        return result

    def _build_frequency_map(
        self, wm: Optional["WorkingMemoryBuffer"]
    ) -> Counter:
        """Count how many times each edge triple appears in recent WM entries.

        Source-specific multipliers from self.source_weights amplify or
        dampen frequency contributions (Phase 98 Gap 8).
        """
        freq: Counter = Counter()
        if wm is None:
            return freq
        for entry in wm.recent(100):
            multiplier = self.source_weights.get(entry.source, 1.0)
            for triple in entry.path_edges:
                freq[triple] += multiplier
        return freq

    def _emit(self, result: DecayResult) -> None:
        try:
            from core.telemetry import NeuralEvent
            self.graph.emit(NeuralEvent.synaptic_decay(
                edges_processed=result.edges_processed,
                edges_decayed=result.edges_decayed,
                edges_resisted=result.edges_resisted,
                mean_delta=result.mean_delta,
            ))
        except Exception as exc:
            logger.debug("SYNAPTIC_DECAY emit failed: %s", exc)

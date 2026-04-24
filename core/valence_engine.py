"""
ValenceEngine — Phase 101: Emotional Valence (Amygdala).

Records aversive and appetitive memories as edge-level "valence" attributes
(separate from edge weight). Negative valence on frequently-rejected paths
reduces their traversal attractiveness; positive valence on approved paths
subtly amplifies them.

Biologically, this is the amygdala's fear-conditioning mechanism: paths that
repeatedly lead to bad outcomes become aversive, paths that lead to rewards
become appetitive.
"""
from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger("cerebrum.valence")

_PathEdge = Tuple[str, str, str]   # (source_id, relation, target_id)


@dataclass
class ValenceResult:
    """Outcome summary for a single valence update pass."""
    edges_updated: int
    outcome_score: float
    mean_delta: float


class ValenceEngine:
    """Amygdala-inspired valence learner.

    Records outcome_score onto the edges of a reasoning path.
    Negative outcome (rejection, contradiction) → valence decreases.
    Positive outcome (approval, high answer score) → valence increases.

    Parameters
    ----------
    adapter        : graph adapter (must implement update_edge_valence)
    graph          : CerebrumGraph — for telemetry emit()
    learning_rate  : fraction of outcome_score applied as delta (default 0.05)
    valence_weight : weight applied during traversal expansion (stored, used externally)
    """

    def __init__(
        self,
        adapter: Any,
        graph: Any,
        learning_rate: float = 0.05,
        valence_weight: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.graph = graph
        self.learning_rate = learning_rate
        self.valence_weight = valence_weight

    def _emit(self, result: ValenceResult) -> None:
        try:
            from core.telemetry import NeuralEvent
            self.graph.emit(NeuralEvent.valence_update(
                path_edges=result.edges_updated,
                outcome_score=result.outcome_score,
                mean_delta=result.mean_delta,
            ))
        except Exception as exc:
            logger.debug("VALENCE_UPDATE emit failed: %s", exc)

    def record_outcome(
        self,
        path_edges: List[_PathEdge],
        outcome_score: float,
    ) -> ValenceResult:
        """Apply valence delta to all edges on the given path.

        Parameters
        ----------
        path_edges    : list of (source_id, relation, target_id) triples
        outcome_score : signed score — negative for aversive, positive for appetitive
        """
        if not path_edges:
            return ValenceResult(edges_updated=0, outcome_score=outcome_score, mean_delta=0.0)

        delta = self.learning_rate * outcome_score
        updated = 0
        for (u, rel, v) in path_edges:
            try:
                n = self.adapter.update_edge_valence(u, v, rel, delta=delta)
                updated += n
            except Exception as exc:
                logger.debug("update_edge_valence failed (%s,%s,%s): %s", u, rel, v, exc)

        result = ValenceResult(
            edges_updated=updated,
            outcome_score=outcome_score,
            mean_delta=delta if updated else 0.0,
        )
        self._emit(result)
        return result

    def get_valence(self, u: str, v: str, relation: str) -> float:
        """Return the current valence of an edge (0.0 = neutral)."""
        try:
            return self.adapter.get_edge_valence(u, v, relation)
        except Exception:
            return 0.0

    def broadcast_learning(self, path_edges: List[_PathEdge], outcome_score: float, gws: Any, node_id: str):
        """Broadcast high-valence reasoning success to the GWS."""
        if outcome_score > 0.5:  # Only broadcast high-confidence successes
            proposal = {
                "path": path_edges,
                "valence": outcome_score
            }
            logger.info(f"ValenceEngine: Broadcasting learning success for node {node_id}")
            asyncio.run(gws.post("learning_channel", proposal, node_id))

"""
DefaultModeEngine — Phase 102: Default Mode Network (Self-Referential Idle Reasoning).

During idle periods (no active queries), the DMN audits the graph's own structure
and knowledge gaps, generating hypotheses about what it doesn't know. Four audits:

  ISOLATION  — communities with zero cross-community edges (structural islands)
  DEAD_ZONE  — nodes absent from all recent WM entries (knowledge that is forgotten)
  UNANSWERED — WM entries with top_score == 0 (queries the system failed to answer)
  FRONTIER   — DiscoveryCalibrator communities with weight > threshold (unexplored)

Each audit produces DMNInsights that can optionally be auto-pushed as Goals.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.working_memory import WorkingMemoryBuffer

logger = logging.getLogger("cerebrum.default_mode")


@dataclass
class DMNInsight:
    """A single self-referential insight produced by the Default Mode Network."""
    type: str                          # "isolation" | "dead_zone" | "unanswered" | "frontier"
    description: str
    context_seeds: List[str] = field(default_factory=list)
    priority: int = 5
    auto_push_goal: bool = True


class DefaultModeEngine:
    """Self-referential idle reasoning engine.

    Parameters
    ----------
    adapter            : graph adapter (must implement to_networkx, get_all_entities)
    graph              : CerebrumGraph — for telemetry emit()
    max_insights       : cap on total insights returned per scan (default 3)
    frontier_threshold : calibrator weight above which a community is a frontier (default 2.0)
    """

    def __init__(
        self,
        adapter: Any,
        graph: Any,
        max_insights: int = 3,
        frontier_threshold: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.graph = graph
        self.max_insights = max_insights
        self.frontier_threshold = frontier_threshold

    def idle_scan(
        self,
        wm: Optional["WorkingMemoryBuffer"] = None,
        goal_stack: Optional[Any] = None,
        calibrator: Optional[Any] = None,
    ) -> List[DMNInsight]:
        """Run all four self-referential audits and return insights.

        Parameters
        ----------
        wm          : WorkingMemoryBuffer (used by dead_zone and unanswered audits)
        goal_stack  : GoalStack (insights may be auto-pushed as Goals)
        calibrator  : DiscoveryCalibrator (used by frontier audit)
        """
        insights: List[DMNInsight] = []

        try:
            G = self.adapter.to_networkx()
        except Exception:
            logger.debug("DefaultModeEngine: to_networkx() failed — skipping.")
            return insights

        cmap: Dict[str, int] = getattr(self.adapter, "community_map", {}) or {}

        insights.extend(self._isolation_audit(G, cmap))
        insights.extend(self._dead_zone_audit(G, wm))
        insights.extend(self._unanswered_audit(wm))
        insights.extend(self._frontier_audit(calibrator))

        # Cap
        insights = insights[:self.max_insights]

        # Auto-push to GoalStack
        if goal_stack is not None:
            for ins in insights:
                if ins.auto_push_goal:
                    try:
                        from core.goal_system import make_goal
                        goal_stack.push(make_goal(
                            description=ins.description,
                            metric_type="approval_rate",
                            target_value=0.1,
                            priority=ins.priority,
                        ))
                    except Exception:
                        pass

        self._emit(insights)
        logger.debug("DefaultModeEngine: %d insights generated.", len(insights))
        return insights

    # ------------------------------------------------------------------
    # Audits
    # ------------------------------------------------------------------

    def _isolation_audit(self, G, cmap: Dict[str, int]) -> List[DMNInsight]:
        """Find communities with no cross-community edges (structural islands)."""
        insights: List[DMNInsight] = []
        # Build cross-community edge counts per community
        cross: Dict[int, int] = {}
        for cid in set(cmap.values()):
            cross[cid] = 0
        for u, v in G.edges():
            cu = cmap.get(u, -1)
            cv = cmap.get(v, -1)
            if cu != cv and cu >= 0 and cv >= 0:
                cross[cu] = cross.get(cu, 0) + 1
                cross[cv] = cross.get(cv, 0) + 1

        isolated_cids = [cid for cid, cnt in cross.items() if cnt == 0 and cid >= 0]
        for cid in isolated_cids[:2]:
            seeds = [n for n, c in cmap.items() if c == cid][:3]
            insights.append(DMNInsight(
                type="isolation",
                description=f"isolated_community_{cid}",
                context_seeds=seeds,
                priority=7,
            ))
        return insights

    def _dead_zone_audit(
        self, G, wm: Optional["WorkingMemoryBuffer"]
    ) -> List[DMNInsight]:
        """Find graph nodes absent from all recent WM entries."""
        if wm is None:
            return []
        recent_nodes: set = set()
        for entry in wm.recent(50):
            recent_nodes.update(entry.seeds)
            recent_nodes.update(entry.answers)
            for (u, _, v) in entry.path_edges:
                recent_nodes.add(u)
                recent_nodes.add(v)

        all_nodes = set(G.nodes())
        dead = all_nodes - recent_nodes
        if not dead:
            return []
        # Pick a few representative dead nodes
        sample = list(dead)[:3]
        return [DMNInsight(
            type="dead_zone",
            description=f"unexplored_nodes_{len(dead)}",
            context_seeds=sample,
            priority=5,
        )]

    def _unanswered_audit(self, wm: Optional["WorkingMemoryBuffer"]) -> List[DMNInsight]:
        """Find WM entries where the system produced no answer (top_score == 0)."""
        if wm is None:
            return []
        failed_seeds: List[str] = []
        for entry in wm.recent(50):
            if entry.top_score < 1e-6 and entry.seeds:
                failed_seeds.extend(entry.seeds[:2])
        if not failed_seeds:
            return []
        # Deduplicate
        seen: set = set()
        unique: List[str] = []
        for s in failed_seeds:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return [DMNInsight(
            type="unanswered",
            description=f"reinvestigate_{unique[0]}" if unique else "reinvestigate",
            context_seeds=unique[:3],
            priority=6,
        )]

    def _frontier_audit(self, calibrator: Optional[Any]) -> List[DMNInsight]:
        """Find calibrator communities with weight > frontier_threshold."""
        if calibrator is None:
            return []
        try:
            stats = calibrator.stats()
            insights: List[DMNInsight] = []
            for cid, info in stats.get("communities", {}).items():
                if info.get("weight", 1.0) > self.frontier_threshold:
                    insights.append(DMNInsight(
                        type="frontier",
                        description=f"frontier_community_{cid}",
                        context_seeds=[],
                        priority=6,
                    ))
            return insights[:2]
        except Exception:
            return []

    def _emit(self, insights: List[DMNInsight]) -> None:
        try:
            from core.telemetry import NeuralEvent
            self.graph.emit(NeuralEvent.default_mode_pulse(
                insight_count=len(insights),
                types=[i.type for i in insights],
            ))
        except Exception as exc:
            logger.debug("DEFAULT_MODE_PULSE emit failed: %s", exc)

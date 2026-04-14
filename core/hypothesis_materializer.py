"""
Hypothesis materialization logic for ResearchAgent (Phase 65).

Materializes candidate links discovered by the ResearchAgent into the graph
with formal provenance and Noisy-OR confidence aggregation.
"""
import logging
from typing import Dict, Any, Optional
from core.graph_adapter import GraphAdapter
from core.hypothesis_engine import HypothesisEngine

logger = logging.getLogger("cerebrum.research")

class HypothesisMaterializer:
    """
    Promotes ResearchAgent candidates to permanent graph edges.
    """
    def __init__(self, adapter: GraphAdapter, hypothesis_engine: HypothesisEngine):
        self.adapter = adapter
        self.engine = hypothesis_engine

    def materialize(self, candidate: Dict[str, Any], confidence: float = 0.85) -> bool:
        """
        Materializes a candidate link into the graph.
        
        Args:
            candidate: Dict containing {'source': str, 'target': str, 'relation': str}
            confidence: The Noisy-OR confidence score to assign.
        """
        source = candidate.get('source')
        target = candidate.get('target')
        relation = candidate.get('relation')

        if not all([source, target, relation]):
            logger.error("Materialization failed: Missing candidate fields.")
            return False

        # Add edge with formal provenance
        provenance = f"hypothesized_by_ResearchAgent:confidence={confidence}"

        # Guard against duplicate edges on MultiGraph adapters.
        # Use the underlying NetworkX graph if accessible; skip check otherwise.
        if hasattr(self.adapter, "_G"):
            G = self.adapter._G
            if G.has_edge(source, target):
                ed = G.get_edge_data(source, target) or {}
                existing = (
                    {d.get("relation") for d in ed.values()}
                    if G.is_multigraph() else {ed.get("relation")}
                )
                if relation in existing:
                    logger.debug("Edge already exists: %s -[%s]-> %s — skipping.", source, relation, target)
                    return True

        try:
            self.adapter.add_edge(
                u=source,
                v=target,
                relation=relation,
                confidence=confidence,
                provenance=provenance,
                synthetic=True
            )
            logger.info(f"Materialized link: {source} -[{relation}]-> {target} (conf: {confidence})")
            return True
        except Exception as e:
            logger.error(f"Materialization failed for {source}->{target}: {e}")
            return False

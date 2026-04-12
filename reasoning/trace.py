"""
Explainable Reasoning Trace (ERT) for CEREBRUM (Phase 62).

Captures the decision-making process at each hop of the beam search,
including winners, top-K rejected competitors, and their feature radars.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class HopTrace:
    """Detailed record of a single hop's selection process."""
    hop: int
    winners: List[Dict[str, Any]] = field(default_factory=list)
    competitors: List[Dict[str, Any]] = field(default_factory=list)
    total_candidates: int = 0
    beam_width: int = 0

@dataclass
class ReasoningTrace:
    """Full trace of a reasoning query across all hops."""
    query: str
    seeds: List[str]
    hops: List[HopTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_hop(
        self, 
        hop: int, 
        winners: List[Any], 
        competitors: List[Any], 
        total_count: int,
        beam_width: int
    ):
        """Add a hop trace using TraversalPath objects."""
        hop_winners = [self._path_to_dict(p) for p in winners]
        hop_competitors = [self._path_to_dict(p) for p in competitors[:5]] # Top 5 rejected
        
        self.hops.append(HopTrace(
            hop=hop,
            winners=hop_winners,
            competitors=hop_competitors,
            total_candidates=total_count,
            beam_width=beam_width
        ))

    def _path_to_dict(self, path: Any) -> Dict[str, Any]:
        """Convert a TraversalPath to a serializable dict for the trace."""
        return {
            "tail": path.tail,
            "score": float(path.score),
            "relation_sequence": [path.nodes[i] for i in range(1, len(path.nodes), 2)],
            "nodes": path.nodes,
            "features": [float(x) for x in path.edge_features[-1]] if path.edge_features else [],
            "attention_weight": float(path.attention_weights[-1]) if path.attention_weights else 1.0
        }

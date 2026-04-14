"""
Consensus Scorer for Federated CEREBRUM (Phase 32).

Aggregates reasoning paths from multiple agents, resolving discrepancies
in scores and entity alignment. 

Uses:
  - Agent reliability weights (Trust score)
  - Path overlap (Consensus confirmation)
  - Procrustes-aligned embedding similarity
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from reasoning.traversal import TraversalPath

log = logging.getLogger("cerebrum.consensus")

from enum import IntEnum

class ConsensusLevel(IntEnum):
    """
    Consensus Hierarchy Levels (Phase 60).
    Higher levels represent more rigorous/expensive verification steps.
    """
    L1_LOCAL = 1      # Multi-strategy voting within a single instance
    L2_FEDERATED = 2  # Cross-node validation in a federated cluster
    L3_GOLD = 3       # Verification against a high-trust "Gold Standard" node

@dataclass
class PathConsensus:
    """Aggregated result for a specific reasoning path."""
    path: TraversalPath
    agents: List[str]            # Names of agents that found/confirmed this path
    consensus_score: float       # Aggregated confidence score [0, 1]
    level: ConsensusLevel = ConsensusLevel.L1_LOCAL
    variance: float = 0.0        # Disagreement between agents
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConsensusScorer:
    """
    Coordinates score aggregation across a federated set of paths.
    """
    def __init__(self, agent_trust_scores: Optional[Dict[str, float]] = None):
        # agent_name -> weight in [0, 1]. Default 1.0.
        self.trust_scores = agent_trust_scores or {}

    def aggregate(
        self, 
        agent_paths: Dict[str, List[TraversalPath]],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[PathConsensus]:
        """
        Merge paths from multiple agents into a unified consensus list.
        
        Parameters:
            agent_paths: {agent_name: [list of paths]}
            query_embedding: Optional vector for re-scoring semantic fit
            
        Returns:
            List[PathConsensus] ranked by consensus_score.
        """
        if not agent_paths:
            return []

        # 1. Group identical paths
        # Key is tuple of node IDs
        grouped: Dict[Tuple[str, ...], List[Tuple[str, TraversalPath]]] = {}
        
        for agent_name, paths in agent_paths.items():
            for p in paths:
                key = tuple(p.nodes)
                grouped.setdefault(key, []).append((agent_name, p))

        results: List[PathConsensus] = []

        # 2. Score each group
        for key, members in grouped.items():
            # Initial path from first member
            base_agent, base_path = members[0]
            
            # Simple average weighted by agent trust
            total_score = 0.0
            total_weight = 0.0
            confirming_agents = []
            
            for agent_name, path in members:
                weight = self.trust_scores.get(agent_name, 1.0)
                total_score += path.score * weight
                total_weight += weight
                confirming_agents.append(agent_name)
            
            # Consensus Bonus: paths found by multiple agents get a boost
            # (Independent verification increases confidence)
            n_agents = len(confirming_agents)
            bonus = 0.1 * (n_agents - 1) # +10% for each additional agent
            
            final_score = (total_score / total_weight if total_weight > 0 else 0.0) + bonus
            
            # Final clip
            final_score = min(1.0, max(0.0, final_score))
            
            results.append(PathConsensus(
                path=base_path,
                agents=confirming_agents,
                consensus_score=final_score,
                metadata={"n_confirmations": n_agents}
            ))

        # 3. Sort by consensus score
        results.sort(key=lambda x: x.consensus_score, reverse=True)
        
        return results

    def resolve_discrepancies(self, consensus_list: List[PathConsensus]) -> List[PathConsensus]:
        """
        Identify and flag paths that have high score variance between agents.
        (Contradiction detection at the path level).

        Uses per-agent trust scores as a proxy for individual path scores —
        agents with low trust who confirmed a path are treated as potential
        sources of conflict. Flags paths where trust-weighted score spread
        exceeds 0.3 as conflicted.
        """
        for cp in consensus_list:
            if len(cp.agents) <= 1:
                cp.metadata["variance"] = 0.0
                cp.metadata["conflict"] = False
                continue

            # Proxy: treat each agent's trust score as their effective path score.
            # Real per-path-per-agent scores are not stored in PathConsensus;
            # trust spread is the best available signal.
            agent_scores = [
                self.trust_scores.get(a, 1.0) for a in cp.agents
            ]
            variance = float(np.var(agent_scores)) if len(agent_scores) > 1 else 0.0
            cp.metadata["variance"] = round(variance, 4)
            cp.metadata["conflict"] = variance > 0.09  # spread > 0.3 std-dev

        return consensus_list

"""
MultiStrategyConsensus — L1 Local Consensus (Phase 60).

Runs multiple traversal strategies against the same query and aggregates
results to find the most robust reasoning paths.
"""
import time
import logging
from typing import List, Dict, Optional, Any

from reasoning.traversal import BeamTraversal
from reasoning.engram_traversal import EngramTraversal
from reasoning.consensus_scorer import ConsensusScorer, PathConsensus, ConsensusLevel
from core.graph_adapter import GraphAdapter

logger = logging.getLogger("cerebrum.mach.l1")

class MultiStrategyConsensus:
    """
    Coordinates multiple local traversal strategies for MACH L1 consensus.
    """
    def __init__(
        self, 
        adapter: GraphAdapter,
        engram: Optional[Any] = None,
        agent_name: str = "local_node"
    ):
        self.adapter = adapter
        self.engram = engram
        self.agent_name = agent_name
        self.scorer = ConsensusScorer()

    async def run_consensus_query(
        self,
        seeds: List[str],
        strategies: List[str] = ["standard", "bayesian", "engram"],
        top_k: int = 5,
        max_hop: int = 3,
        beam_width: int = 10,
        max_budget: int = 1000,
        edge_type_weights: Optional[Dict[str, float]] = None,
    ) -> List[PathConsensus]:
        """
        Execute multiple strategies and aggregate results.
        """
        if not strategies:
            strategies = ["standard"]

        agent_results = {}
        total_explored = 0

        for strategy in strategies:
            logger.debug("Running L1 strategy: %s", strategy)
            
            # 1. Setup CSA Engine
            from core.attention_engine import CSAEngine
            csa_meta = getattr(self.adapter, "_csa_metadata", {}) or {}
            csa = CSAEngine(
                adapter=self.adapter,
                edge_type_weights=edge_type_weights or {}
            )
            # Pre-load community distances if available
            if "distances" in csa_meta:
                csa.set_community_graph(csa_meta["distances"], csa_meta.get("adjacent_pairs", set()))

            # 2. Setup traversal based on strategy
            if strategy == "engram" and self.engram is not None:
                traversal = EngramTraversal(
                    self.adapter, 
                    csa_engine=csa,
                    cache=self.engram,
                    beam_width=beam_width,
                    max_hop=max_hop,
                    max_budget=max_budget,
                    edge_type_weights=edge_type_weights
                )
            elif strategy == "bayesian":
                traversal = BeamTraversal(
                    self.adapter,
                    csa_engine=csa,
                    beam_width=beam_width,
                    max_hop=max_hop,
                    max_budget=max_budget,
                    probabilistic=True,
                    edge_type_weights=edge_type_weights
                )
            else:
                # "standard"
                traversal = BeamTraversal(
                    self.adapter,
                    csa_engine=csa,
                    beam_width=beam_width,
                    max_hop=max_hop,
                    max_budget=max_budget,
                    probabilistic=False,
                    edge_type_weights=edge_type_weights
                )

            # 2. Execute
            # Resolve query_embedding from first seed if possible
            q_emb = self.adapter.get_embedding(seeds[0]) if seeds else None
            
            paths = traversal.traverse(
                seeds, 
                query_embedding=q_emb
            )
            
            # Store results under a strategy-specific agent name
            strategy_agent = f"{self.agent_name}_{strategy}"
            agent_results[strategy_agent] = paths
            total_explored += len(paths)

        # 3. Aggregate via ConsensusScorer
        # We use L1_LOCAL level here
        consensus_list = self.scorer.aggregate(agent_results)
        
        # Mark all as L1
        for cp in consensus_list:
            cp.level = ConsensusLevel.L1_LOCAL
            
        return consensus_list, total_explored

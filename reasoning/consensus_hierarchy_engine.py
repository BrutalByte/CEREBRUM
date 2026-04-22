"""
ConsensusHierarchyEngine — MACH L2 & L3 (Phase 60).

Coordinates multi-level consensus across local strategies (L1), 
federated nodes (L2), and high-trust "Gold" sources (L3).

Phase 110: Integrated with GlobalWorkspace for novelty-driven pre-emption.
"""
import time
import logging
import asyncio
from typing import List, Dict, Optional, Any

from reasoning.multi_strategy_consensus import MultiStrategyConsensus
from reasoning.consensus_scorer import ConsensusScorer, PathConsensus, ConsensusLevel
from adapters.federated_adapter import FederatedAdapter
from core.global_workspace import GlobalWorkspace, CommunitySignal

logger = logging.getLogger("cerebrum.mach.hierarchy")

class ConsensusHierarchyEngine:
    """
    Orchestrates the MACH hierarchy, now augmented with GWS pre-emption.
    """
    def __init__(
        self, 
        adapter: Any,
        engram: Optional[Any] = None,
        research_agent: Optional[Any] = None,
        gws: Optional[GlobalWorkspace] = None,
        agent_name: str = "coordinator"
    ):
        self.adapter = adapter
        self.engram = engram
        self.research_agent = research_agent
        self.gws = gws
        self.agent_name = agent_name
        self.l1_engine = MultiStrategyConsensus(adapter, engram=engram, agent_name=agent_name)
        self.scorer = ConsensusScorer()

    async def query_with_consensus(
        self,
        query: str,
        seeds: List[str],
        level: ConsensusLevel = ConsensusLevel.L1_LOCAL,
        strategies: List[str] = ["standard", "bayesian"],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute query and escalate consensus to the requested level.
        Phase 110: GWS-enabled pre-emption.
        """
        t0 = time.perf_counter()
        
        # 1. Level 1: Local Multi-Strategy Consensus
        l1_results, total_explored = await self.l1_engine.run_consensus_query(
            seeds=seeds,
            strategies=strategies,
            **kwargs
        )
        
        # Phase 110: GWS Pre-emption Logic
        # Check if any path in L1 results was 'broadcast' via GWS with extreme surprise
        current_results = l1_results
        if self.gws is not None:
            gws_top = self.gws.get_top_signals(limit=3)
            # Boost scores for paths corroborated by GWS signals
            for cand in current_results:
                for signal in gws_top:
                    if cand.path.tail == signal.path[-1]:
                        # Signal corroborated; boost confidence based on novelty
                        cand.consensus_score = min(1.0, cand.consensus_score * (1.0 + signal.novelty_score))
                        cand.agents.append("gws_boost")
        
        # 2. Level 2: Federated Consensus (Escalation)
        if level >= ConsensusLevel.L2_FEDERATED and isinstance(self.adapter, FederatedAdapter):
            current_results = await self._escalate_to_l2(current_results, seeds, **kwargs)

        # 3. Level 3: Gold Standard (Verify against external high-trust node/literature)
        # Phase 110: High-surprise GWS paths can skip waiting for L2 if validated
        if level >= ConsensusLevel.L3_GOLD and self.research_agent is not None:
            current_results = await self._escalate_to_l3(current_results, seeds, **kwargs)

        dt = time.perf_counter() - t0
        return {
            "results": current_results,
            "total_explored": total_explored,
            "duration": dt,
            "level_reached": level
        }


    async def _escalate_to_l3(
        self,
        l2_results: List[PathConsensus],
        seeds: List[str],
        **kwargs
    ) -> List[PathConsensus]:
        """
        MACH L3: Verify top results against external literature via ResearchAgent.
        """
        if not l2_results:
            return []

        # Limit L3 to top 3 (expensive literature search)
        top_candidates = l2_results[:3]
        
        for cand in top_candidates:
            source_id = seeds[0] if seeds else None
            target_id = cand.path.tail
            
            if not source_id:
                continue

            try:
                # We ask ResearchAgent to validate the connection (source -> relation -> target)
                # We'll use the LAST relation in the path as the predicate.
                # Path nodes are [E, R, E, R, E...]
                predicate = cand.path.nodes[-2] if len(cand.path.nodes) > 1 else "related_to"
                
                # Use synchronous literature search (if available in ResearchAgent)
                # In Phase 52, ResearchAgent was integrated with ExternalValidator.
                from core.hypothesis_engine import HypothesisProposal
                prop = HypothesisProposal(
                    source=source_id,
                    target=target_id,
                    derived_relation=predicate,
                    confidence=cand.consensus_score,
                    supporting_paths=[cand.path]
                )
                
                # Check if ResearchAgent can validate this proposal
                if hasattr(self.research_agent, "_validator") and self.research_agent._validator:
                    report = self.research_agent._validator.validate(prop)
                    if report and report.hit_count > 0:
                        cand.level = ConsensusLevel.L3_GOLD
                        cand.consensus_score = (cand.consensus_score + 1.0) / 2.0
                        cand.agents.append(f"gold_literature_({report.hit_count}_hits)")
                        cand.metadata["literature_report"] = report.__dict__
            except Exception as exc:
                logger.debug("L3 Escalation failed for path %s: %s", cand.path, exc)
                continue

        return sorted(l2_results, key=lambda x: x.consensus_score, reverse=True)

    async def _escalate_to_l2(
        self, 
        l1_results: List[PathConsensus], 
        seeds: List[str],
        **kwargs
    ) -> List[PathConsensus]:
        """
        MACH L2: Verify L1 winners against other nodes in the federation.
        """
        if not l1_results:
            return []

        # We take the top L1 candidates and ask other adapters if they can confirm them
        top_candidates = l1_results[:5] # Limit escalation to top 5
        
        tasks = []
        for cand in top_candidates:
            # For each candidate path, we ask OTHER nodes if they can verify it.
            # In a real FederatedAdapter, some adapters are remote nodes.
            tasks.append(self._verify_path_federated(cand, seeds))
            
        verified_results = await asyncio.gather(*tasks)
        
        # Merge L2 verification back into consensus scores
        # (Simplified: if verified by L2, boost score and mark as L2)
        for cand, is_verified in zip(top_candidates, verified_results):
            if is_verified:
                cand.level = ConsensusLevel.L2_FEDERATED
                cand.consensus_score = (cand.consensus_score + 1.0) / 2.0
                cand.agents.append("federated_verification")
        
        return sorted(l1_results, key=lambda x: x.consensus_score, reverse=True)

    async def _verify_path_federated(self, consensus: PathConsensus, seeds: List[str]) -> bool:
        """
        Check if other adapters in the FederatedAdapter can corroborate the tail.
        """
        target_id = consensus.path.tail
        source_id = seeds[0] if seeds else None
        
        if not source_id:
            return False

        # Query other adapters for the same (source -> target) connection
        confirmations = 0
        for name, sub_adapter in self.adapter.adapters.items():
            if name == self.agent_name:
                continue # Skip the one that already found it
            
            try:
                # We use a shallow search on the other adapter to see if the target 
                # is reachable from the source within the same max_hop.
                # In production, this calls the /reason endpoint on the remote node.
                # For now, we'll check if the adapter has the entity.
                if sub_adapter.get_entity(target_id):
                    confirmations += 1
            except Exception:
                continue
                
        return confirmations > 0

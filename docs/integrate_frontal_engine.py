import os
import re

def modify_cerebrum():
    path = '../core/cerebrum.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add FrontalEngine import and initialization
    if 'from core.frontal_engine import FrontalEngine' not in content:
        content = content.replace('from __future__ import annotations', 'from __future__ import annotations\nfrom core.frontal_engine import FrontalEngine, ReasoningStrategy')
        content = content.replace('self.modulator = ChemicalModulator()', 'self.modulator = ChemicalModulator()\n        self.frontal = FrontalEngine(modulator=self.modulator)\n        self._research_agent = None')

    # 2. Add set_research_agent method
    if 'def set_research_agent' not in content:
        method = """
    def set_research_agent(self, agent: Any) -> None:
        \"\"\"Phase 150: Attach a ResearchAgent for targeted investigations.\"\"\"
        self._research_agent = agent
"""
        # Inject before query()
        content = content.replace('    def query(', method + '\n    def query(')

    # 3. Update query() with Tiered Reasoning
    if 'Phase 150: Frontal Engine' not in content:
        # We need to find the point after initial paths are gathered
        old_ cingulate = """            # Phase 149: Cingulate Engine recursive refinement
            if ProvenanceValidator.is_hub_flooded(paths):
                logger.info('CingulateEngine: Retrying with strict constraints...')
                # Retry once with half beam width
                traversal.beam_width = max(2, traversal.beam_width // 2)
                paths = traversal.traverse(seeds, query_embedding=query_embedding, node_priming=_priming_map)"""

        new_executive = """
            # Phase 150: Frontal Engine executive strategy
            entropy = getattr(traversal, "_last_entropy", 0.0)
            strategy = self.frontal.determine_strategy(paths, entropy=entropy)

            if strategy == ReasoningStrategy.DEEP and self._research_agent and getattr(traversal, "epistemic_gaps", []):
                logger.info("FrontalEngine: Suspending for DEEP research...")
                for gap in traversal.epistemic_gaps[:2]:
                    # Step 4: Targeted ResearchAgent coupling
                    if hasattr(self._research_agent, "push_candidate"):
                        from core.research_agent import ResearchCandidate
                        self._research_agent.push_candidate(ResearchCandidate(
                            source_id=gap["source"],
                            target_id=gap["target"],
                            discovery_potential=0.9,
                            gap_score=1.0 - gap["score"],
                            community_distance=1,
                            seeded_by="frontal_engine"
                        ))
                
                # Trigger immediate scan
                self._research_agent.scan_once()
                # Retry with new knowledge
                logger.info("FrontalEngine: Resuming query after research.")
                paths = traversal.traverse(seeds, query_embedding=query_embedding, node_priming=_priming_map)

            elif strategy == ReasoningStrategy.HYBRID and self._research_agent and getattr(traversal, "epistemic_gaps", []):
                logger.info("FrontalEngine: Dispatching HYBRID async research.")
                for gap in traversal.epistemic_gaps[:2]:
                    if hasattr(self._research_agent, "push_candidate"):
                        from core.research_agent import ResearchCandidate
                        self._research_agent.push_candidate(ResearchCandidate(
                            source_id=gap["source"],
                            target_id=gap["target"],
                            discovery_potential=0.7,
                            gap_score=1.0 - gap["score"],
                            community_distance=1,
                            seeded_by="frontal_engine"
                        ))
                # Agent will pick these up in its next background cycle

            # Phase 149: Cingulate Engine recursive refinement (Legacy fallback)
            if strategy == ReasoningStrategy.FAST and ProvenanceValidator.is_hub_flooded(paths):
                logger.info('CingulateEngine: Retrying with strict constraints...')
                traversal.beam_width = max(2, traversal.beam_width // 2)
                paths = traversal.traverse(seeds, query_embedding=query_embedding, node_priming=_priming_map)
"""
        content = content.replace(old_cingulate, new_executive)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_cerebrum()
    print("Cerebrum executive logic integrated.")

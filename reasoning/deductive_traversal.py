"""
DeductiveTraversal — Phase 118: Deterministic Proof Traversal.

A complete state-space search engine. Replaces the probabilistic BeamTraversal
to provide 100% logical consistency via axiomatic verification.
"""
import logging
from typing import List, Dict, Any, Optional
from core.causal_engine import CAUSAL_RELATIONS

logger = logging.getLogger("cerebrum.deductive")

class DeductiveTraversal:
    """
    Exhaustive deductive path searcher.
    
    Explores graph branches as a state-machine of logical propositions.
    Validates every hop through the SymbolicValidator before materialization.
    """
    def __init__(self, adapter, validator):
        self.adapter = adapter
        self.validator = validator

    def traverse(self, seed: str, target: str,
                 causal_only: bool = False) -> List[List[str]]:
        """Perform an exhaustive search for a verified proof."""
        logger.info(f"DeductiveTraversal: Searching for logical proof {seed} -> {target}")
        
        # State queue: [current_path_list]
        queue = [[seed]]
        proofs = []

        while queue:
            path = queue.pop(0)
            u = path[-1]

            if u == target:
                if self.validator.validate(path):
                    proofs.append(path)
                continue

            # Expand neighbors
            neighbors = list(self.adapter.get_neighbors(u))
            logger.debug(f"Neighbors of {u}: {neighbors}")
            for neighbor in neighbors:
                # Handle Edge object from NetworkXAdapter
                if hasattr(neighbor, "target_id"):
                    v = neighbor.target_id
                    rel = getattr(neighbor, "relation_type", "unknown")
                elif isinstance(neighbor, tuple):
                    if len(neighbor) == 2 and isinstance(neighbor[1], str):
                        v, rel = neighbor
                    elif len(neighbor) == 2 and isinstance(neighbor[1], dict):
                        v, rel = neighbor[0], neighbor[1].get("relation", "unknown")
                    else:
                        v, rel = neighbor[0], "unknown"
                else:
                    v = neighbor
                    rel = "unknown"
                
                if causal_only and rel not in CAUSAL_RELATIONS:
                    continue

                new_path = path + [rel, v]
                logger.debug(f"Exploring path: {new_path}")

                # Prune logically impossible paths early (The Prefrontal Bridge)
                if self.validator.validate(new_path):
                    queue.append(new_path)
                    
        return proofs

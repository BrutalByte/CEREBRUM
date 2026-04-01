"""
Distributed Beam Traversal for Federated CEREBRUM.

Extends BeamTraversal to support delegated reasoning. When the traversal
reaches a node that exists in multiple graphs (aliases) or a WORMHOLE node,
it can request a 'Reasoning Branch' from the remote node rather than
fetching neighbors one-by-one over the network.
"""
from typing import List, Optional, Dict
import numpy as np
from reasoning.traversal import BeamTraversal, TraversalPath
from adapters.federated_adapter import FederatedAdapter


class DistributedBeamTraversal(BeamTraversal):
    """
    BeamTraversal with federated delegation support.
    """

    def _traverse_inner(
        self,
        seeds: List[str],
        emb_dim: int,
        query_time: Optional[float],
    ) -> List[TraversalPath]:
        # 1. Initialize beam
        beam: List[TraversalPath] = []
        for seed in seeds:
            emb = self.adapter.get_embedding(seed)
            if emb is None:
                emb = np.zeros(emb_dim, dtype=np.float32)
            
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            cid = self.adapter.get_community(seed)
            beam.append(
                TraversalPath(
                    nodes=[seed],
                    seen_entities={seed},
                    embedding=emb.copy(),
                    score=1.0,
                    community_sequence=[cid] if cid >= 0 else [],
                )
            )

        # 2. Initial Delegation (Optional)
        # If seeds are remote, we might want to fetch branches immediately
        if isinstance(self.adapter, FederatedAdapter):
            delegated_paths: List[TraversalPath] = []
            for seed in seeds:
                branches = self.adapter.get_reasoning_branches(
                    seed, 
                    max_hop=self.max_hop, 
                    beam_width=self.beam_width
                )
                for b_dict in branches:
                    p = TraversalPath.from_dict(b_dict)
                    # Ensure seed is correct
                    if p.nodes[0] == seed:
                        delegated_paths.append(p)
            
            # Merge delegated paths into initial beam
            # (Limit to beam_width)
            beam.extend(delegated_paths)
            beam = sorted(beam, key=lambda p: p.score, reverse=True)[:self.beam_width]

        all_paths: List[TraversalPath] = list(beam)

        # 3. Standard Hopping with Boundary Delegation
        for hop in range(1, self.max_hop + 1):
            candidates: List[TraversalPath] = []
            
            # Use base class expansion logic for the bulk of the work
            # But we could override specific parts if needed.
            # For Phase 32, we focus on the initial and boundary delegation.
            
            # ... (Rest of standard BeamTraversal logic would go here)
            # To keep it DRY, I'll call the parent's _traverse_inner for one hop at a time
            # But parent's _traverse_inner runs the whole loop.
            
            # Re-implementing the loop here to allow per-hop delegation.
            # (Simplified version for now)
            
            # For brevity in this prototype, we'll just use the base class
            # and rely on FederatedAdapter.get_neighbors which already handles 
            # basic neighbor-level federation.
            
            # True DistributedBeamTraversal would use get_reasoning_branches 
            # at every node that looks like a community bridge or a remote node.
            
            # For now, I'll just delegate to parent to ensure it still works.
            # In a full Phase 32 implementation, I'd move the expansion logic 
            # to a helper method that can be shared.
            
            return super()._traverse_inner(seeds, emb_dim, query_time)
            
        return all_paths

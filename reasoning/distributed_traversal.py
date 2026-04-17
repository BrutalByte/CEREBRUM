"""
Distributed Beam Traversal for Federated CEREBRUM (Phase 32).

Extends BeamTraversal to support delegated reasoning. When the traversal
reaches a node that exists in multiple graphs (aliases) or a SynapticBridge node,
it can request a 'Reasoning Branch' from the remote node rather than
fetching neighbors one-by-one over the network.
"""
from typing import List, Optional, Dict, Set
import numpy as np
import logging
from reasoning.traversal import BeamTraversal, TraversalPath
from adapters.federated_adapter import FederatedAdapter

log = logging.getLogger("cerebrum.distributed_traversal")

class DistributedBeamTraversal(BeamTraversal):
    """
    BeamTraversal with federated delegation support.
    """

    def _traverse_inner(
        self,
        seeds: List[str],
        emb_dim: int,
        query_time: Optional[float],
        trace_info: Optional["ReasoningTrace"] = None,
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
                    q_score=255, # Added for Phase 61
                    quantized=self.quantized, # Added for Phase 61
                    community_sequence=[cid] if cid >= 0 else [],
                )
            )

        # Phase 62: Record hop 0 (seeds) in trace
        if trace_info:
            trace_info.add_hop(hop=0, winners=beam, competitors=[], total_count=len(beam), beam_width=len(beam))

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
            beam.extend(delegated_paths)
            beam = sorted(beam, key=lambda p: p.score, reverse=True)[:self.beam_width]

        all_paths: List[TraversalPath] = list(beam)
        
        # Track which nodes we have already delegated for to prevent loops
        delegated_nodes: Set[str] = set()

        # 3. Standard Hopping with Boundary Delegation
        for hop in range(1, self.max_hop + 1):
            candidates: List[TraversalPath] = []
            
            for path in beam:
                current = path.nodes[-1]
                
                # Check for boundary delegation (aliases in other graphs)
                # But only if we have enough hop budget left.
                remaining_hops = self.max_hop - hop + 1
                
                if isinstance(self.adapter, FederatedAdapter) and remaining_hops >= 1:
                    # We check if this node belongs to other adapters too
                    # (Boundary detection)
                    owner_name = self.adapter._resolve_adapter(current)
                    aliases = self.adapter.alignment.resolve_aliases(owner_name or "", current)
                    
                    # If it has more than 1 alias (meaning it exists elsewhere), delegate
                    if len(aliases) > 1 and current not in delegated_nodes:
                        delegated_nodes.add(current)
                        log.debug("Delegating boundary branch at node: %s", current)
                        
                        branches = self.adapter.get_reasoning_branches(
                            current,
                            context_embedding=path.embedding,
                            max_hop=remaining_hops,
                            beam_width=self.beam_width
                        )
                        for b_dict in branches:
                            branch_p = TraversalPath.from_dict(b_dict)
                            # Graft the remote branch onto the current path
                            # branch_p.nodes[0] is 'current', so we skip it to avoid duplication
                            if len(branch_p.nodes) > 1:
                                grafted = TraversalPath(
                                    nodes=path.nodes + branch_p.nodes[1:],
                                    score=path.score * branch_p.score,
                                    embedding=branch_p.embedding,
                                    attention_weights=path.attention_weights + branch_p.attention_weights,
                                    community_sequence=path.community_sequence + branch_p.community_sequence[1:],
                                    seen_entities=path.seen_entities.union(set(branch_p.nodes))
                                )
                                candidates.append(grafted)

                # 4. Standard Neighbor Expansion (Local/Federated-Pull)
                neighbors = self.adapter.get_neighbors(
                    current, 
                    context_embedding=path.embedding,
                    max_neighbors=50
                )
                
                for edge in neighbors:
                    if edge.target_id in path.seen_entities:
                        continue
                        
                    # Calculate CSA score (simplified for this layer)
                    # In a full implementation, we'd use the CSAEngine here.
                    # For now, we use the edge weight as a proxy.
                    w = edge.weight if hasattr(edge, 'weight') else 0.5
                    
                    target_emb = self.adapter.get_embedding(edge.target_id)
                    if target_emb is None:
                        target_emb = path.embedding # fallback
                        
                    cid = self.adapter.get_community(edge.target_id)
                    
                    new_path = TraversalPath(
                        nodes=path.nodes + [edge.relation_type, edge.target_id],
                        score=path.score * w,
                        embedding=target_emb,
                        attention_weights=path.attention_weights + [w],
                        community_sequence=path.community_sequence + [cid],
                        seen_entities=path.seen_entities.union({edge.target_id})
                    )
                    candidates.append(new_path)

            # 5. Prune and keep top-B
            if not candidates:
                break
            
            beam = sorted(candidates, key=lambda p: p.score, reverse=True)[:self.beam_width]

            # Phase 62: Record hop trace
            if trace_info:
                winners_set = set(id(p) for p in beam)
                competitors = [p for p in candidates if id(p) not in winners_set]
                competitors.sort(key=lambda p: p.score, reverse=True)
                trace_info.add_hop(hop, beam, competitors, len(candidates), self.beam_width)

            all_paths.extend(beam)
            
        return all_paths

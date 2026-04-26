"""
Phase 137: Hop-1 Intermediate Seed Expansion (H1SE).

Eliminates cross-branch beam competition at high-degree hub nodes by giving
each hop-1 entity its own independent deep traversal instead of competing in
a shared beam pool.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set

from reasoning.traversal import BeamTraversal, TraversalPath


def _funnel_beam_widths(max_hop: int, bw: int, factor: float) -> Dict[int, int]:
    """Phase 136 funnel logic applied to the deep leg of H1SE."""
    if max_hop <= 1:
        return {}
    non_terminal = list(range(1, max_hop))
    n = len(non_terminal)
    widths: Dict[int, int] = {}
    for i, hop in enumerate(non_terminal):
        mult = factor if n == 1 else 1.0 + (factor - 1.0) * (i / (n - 1))
        widths[hop] = max(bw, int(bw * mult))
    return widths


def _stitch(parent: TraversalPath, child: TraversalPath) -> TraversalPath:
    """
    Concatenate parent (seed→hop1) with child (hop1→answer) into a single
    full-depth path (seed→hop1→answer).

    parent.nodes = [seed, r1, hop1]
    child.nodes  = [hop1, r2, answer, ...]
    result.nodes = [seed, r1, hop1, r2, answer, ...]
    """
    stitched_nodes = parent.nodes + child.nodes[1:]
    return TraversalPath(
        nodes=stitched_nodes,
        seen_entities=parent.seen_entities | child.seen_entities,
        embedding=child.embedding,
        score=parent.score * child.score,
        attention_weights=parent.attention_weights + child.attention_weights,
        community_sequence=parent.community_sequence + child.community_sequence,
        edge_confidences=parent.edge_confidences + child.edge_confidences,
        edge_provenances=parent.edge_provenances + child.edge_provenances,
        edge_features=parent.edge_features + child.edge_features,
        beta_alpha=child.beta_alpha,
        beta_beta=child.beta_beta,
    )


class HopExpandedTraversal:
    """
    Phase 137: Hop-1 Intermediate Seed Expansion.

    Stage 1: 1-hop scan (terminal — no pruning) returns all hop-1 neighbors.
    Stage 2: each hop-1 entity runs an independent (max_hop-1)-hop
             BeamTraversal with its own beam, eliminating cross-branch
             competition for beam slots.

    Paths from stage 2 are stitched with their stage-1 parent so that
    hop_depth reflects the full depth from the original seed. This ensures
    min_hop filtering in extract() behaves correctly.

    Budget control: max_budget is divided evenly across the scan traversal
    and all K deep traversals so total expansion cost is bounded at
    max_budget regardless of expansion_k.

    Drop-in replacement for BeamTraversal inside CerebrumGraph.query() when
    hop_expand=True.  Compatible with LoopedBeamTraversal as the inner
    traversal.
    """

    def __init__(
        self,
        adapter,
        csa_engine,
        beam_width: int = 10,
        max_hop: int = 3,
        max_neighbors: int = 100,
        expansion_k: int = 20,
        beam_profile_factor: float = 3.0,
        max_budget: int = 10_000,
        governor=None,
        probabilistic: bool = False,
        warm_start_strength: float = 0.0,
        **traversal_kwargs,
    ):
        self.adapter = adapter
        self.csa = csa_engine
        self.beam_width = beam_width
        self.max_hop = max_hop
        self.max_neighbors = max_neighbors
        self.expansion_k = expansion_k
        self.max_budget = max_budget
        self.governor = governor
        self.probabilistic = probabilistic
        self.warm_start_strength = warm_start_strength
        self._traversal_kwargs = traversal_kwargs
        self._deep_beam_widths = _funnel_beam_widths(
            max_hop - 1, beam_width, beam_profile_factor
        )
        # Phase 124: causal index stamped by CerebrumGraph after construction
        self._causal_edge_index: set = set()
        self.causal_bonus: float = float(traversal_kwargs.get("causal_bonus", 0.3))

    def _per_traversal_budget(self) -> int:
        """
        Divide max_budget evenly: 1 slot for the scan + expansion_k deep slots.
        Guarantees total expansions <= max_budget regardless of expansion_k.
        """
        return max(100, self.max_budget // (self.expansion_k + 1))

    def _make_traversal(
        self,
        max_hop: int,
        beam_widths: Optional[Dict[int, int]] = None,
        per_budget: Optional[int] = None,
    ) -> BeamTraversal:
        t = BeamTraversal(
            adapter=self.adapter,
            csa_engine=self.csa,
            beam_width=self.beam_width,
            max_hop=max_hop,
            max_neighbors=self.max_neighbors,
            max_budget=per_budget if per_budget is not None else self.max_budget,
            governor=self.governor,
            probabilistic=self.probabilistic,
            warm_start_strength=self.warm_start_strength,
            beam_widths=beam_widths or {},
            **self._traversal_kwargs,
        )
        t._causal_edge_index = self._causal_edge_index
        t.causal_bonus = self.causal_bonus
        return t

    def traverse(
        self,
        seeds: List[str],
        query_time=None,
        query_embedding=None,
        community_merger=None,
        trace_info=None,
        node_priming=None,
    ) -> List[TraversalPath]:
        per_budget = self._per_traversal_budget()

        # Stage 1: 1-hop terminal scan — BeamTraversal never prunes the final
        # hop, so all hop-1 neighbors are returned regardless of beam_width.
        scan = self._make_traversal(max_hop=1, per_budget=per_budget)
        scan_paths = scan.traverse(
            seeds,
            query_embedding=query_embedding,
            trace_info=None,
            node_priming=node_priming,
        )

        if self.max_hop <= 1 or not scan_paths:
            return scan_paths

        # Build best parent path per hop-1 entity (for stitching).
        seed_set: Set[str] = set(seeds)
        parent_map: Dict[str, TraversalPath] = {}
        for p in scan_paths:
            eid = p.tail
            if eid not in seed_set:
                if eid not in parent_map or p.score > parent_map[eid].score:
                    parent_map[eid] = p

        # Rank hop-1 entities by scan score, cap at expansion_k.
        hop1_entities: List[str] = []
        for p in sorted(parent_map.values(), key=lambda x: x.score, reverse=True):
            hop1_entities.append(p.tail)
            if len(hop1_entities) >= self.expansion_k:
                break

        # Stage 2: independent deep traversal per hop-1 entity.
        # Stitch each deep path with its parent so hop_depth = full depth from seed.
        deep_hop = self.max_hop - 1
        all_paths: List[TraversalPath] = list(scan_paths)
        for entity in hop1_entities:
            parent = parent_map[entity]
            deep = self._make_traversal(
                max_hop=deep_hop,
                beam_widths=self._deep_beam_widths,
                per_budget=per_budget,
            )
            deep_paths = deep.traverse(
                [entity],
                query_embedding=query_embedding,
                trace_info=None,
                node_priming=node_priming,
            )
            for dp in deep_paths:
                all_paths.append(_stitch(parent, dp))

        return all_paths

"""
Phase 137: Hop-1 Intermediate Seed Expansion (H1SE).

Eliminates cross-branch beam competition at high-degree hub nodes by giving
each hop-1 entity its own independent deep traversal instead of competing in
a shared beam pool.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Set, Any, Callable
import logging

_log = logging.getLogger("cerebrum.traversal")

from reasoning.traversal import BeamTraversal, TraversalPath

# ---------------------------------------------------------------------------
# Phase 184: diagnostic hook — set by phase184_hop1_audit.py only.
# Called with (sorted_neighbors, score_map, k_eff) after stage-1 ranking.
# None in all production paths.
# ---------------------------------------------------------------------------
_hop1_audit_cb = None

def set_hop1_audit_cb(cb):
    global _hop1_audit_cb
    _hop1_audit_cb = cb

def clear_hop1_audit_cb():
    global _hop1_audit_cb
    _hop1_audit_cb = None


class GlobalBeamBarrier:
    """
    Phase 139: Cross-Branch Pruning Barrier.
    Allows independent H1SE traversals to synchronize and prune branches
    that fall behind the global top score.
    """

    def __init__(self, target_count: int, threshold_ratio: float = 0.5):
        self.target_count = target_count
        self.threshold_ratio = threshold_ratio
        self.best_scores: Dict[int, float] = {}  # sub_id -> score

    def report(self, sub_id: int, top_score: float) -> bool:
        """
        Register the current top score for a branch.
        Returns True if the branch should continue, False if it should be pruned.
        """
        self.best_scores[sub_id] = top_score
        
        # Don't prune until we have data from at least 1/3 of the expected branches
        if len(self.best_scores) < max(2, self.target_count // 3):
            return True

        global_max = max(self.best_scores.values())
        if global_max <= 0:
            return True

        # Prune if this branch's best is significantly worse than the global best
        is_viable = top_score >= (global_max * self.threshold_ratio)
        _log.debug("Barrier report sub_id=%d score=%.4f global_max=%.4f viable=%s", 
                   sub_id, top_score, global_max, is_viable)
        return is_viable


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
        modulator=None,
        use_adaptive_expansion: bool = True,
        min_diversity_target: int = 15,
        residual_k: int = 10,
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
        self.modulator = modulator
        self.use_adaptive_expansion = use_adaptive_expansion
        self.min_diversity_target = min_diversity_target
        self.residual_k = residual_k
        # Phase 151: PenultimateGate — hop-1 branch score-gap filter
        self.penultimate_decay: float = float(traversal_kwargs.get("penultimate_decay", 0.0))
        # Phase 172: Stage-1 anchor — biases which hop-1 entities receive deep traversals.
        # Entities in the anchor set get a score bonus during stage-1 ranking.
        self._stage1_anchor = traversal_kwargs.get("stage1_anchor_hint", None)
        self._traversal_kwargs = traversal_kwargs
        self._deep_beam_widths = _funnel_beam_widths(
            max_hop - 1, beam_width, beam_profile_factor
        )
        # Phase 124: causal index stamped by CerebrumGraph after construction
        self._causal_edge_index: set = set()
        self.causal_bonus: float = float(traversal_kwargs.get("causal_bonus", 0.3))

    def _get_adaptive_k(self) -> int:
        """
        Phase 138: Scale expansion_k based on metabolic signals.
        High arousal/uncertainty -> more expansion.
        High reinforcement/confidence -> less expansion.
        """
        if not self.use_adaptive_expansion or self.modulator is None:
            return self.expansion_k

        # Metabolic signals: 0.0 to 1.0
        arousal = getattr(self.modulator, "arousal", 0.0)
        reinforcement = getattr(self.modulator, "reinforcement", 0.0)

        # Formula: more expansion when aroused, less when reinforced.
        # Scale ranges from 0.2 (confident) to 2.0 (exploring).
        scale = 1.0 + arousal - reinforcement
        k_eff = int(self.expansion_k * max(0.2, min(2.0, scale)))
        return max(1, k_eff)

    def _per_traversal_budget(self, k: int) -> int:
        """
        Divide max_budget evenly: 1 slot for the scan + k deep slots.
        Guarantees total expansions <= max_budget regardless of k.
        """
        return max(100, self.max_budget // (k + 1))

    def _make_traversal(
        self,
        max_hop: int,
        beam_widths: Optional[Dict[int, int]] = None,
        per_budget: Optional[int] = None,
    ) -> BeamTraversal:
        # Ensure we don't pass duplicate beam_widths or other explicit args via kwargs
        clean_kwargs = self._traversal_kwargs.copy()
        clean_kwargs.pop('beam_widths', None)
        clean_kwargs.pop('stage1_anchor_hint', None)  # Phase 172: not needed in BeamTraversal
        # relation boost is explicitly handled below if not in clean_kwargs
        trb = clean_kwargs.pop('terminal_relation_boost', None)
        prb = clean_kwargs.pop('penultimate_relation_boost', None)  # Phase 156

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
            terminal_relation_boost=trb,
            penultimate_relation_boost=prb,  # Phase 156
            **clean_kwargs,
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
        if not seeds:
            return []

        k_eff = self._get_adaptive_k()
        per_budget = self._per_traversal_budget(k_eff)

        # Stage 1: 1-hop terminal scan for ALL seeds.
        # This captures the union of all immediate neighbors.
        scan = self._make_traversal(max_hop=1, per_budget=per_budget)
        scan_paths = scan.traverse(
            seeds,
            query_embedding=query_embedding,
            trace_info=trace_info,
            node_priming=node_priming,
        )

        if self.max_hop <= 1 or not scan_paths:
            return scan_paths

        # Build best parent path per hop-1 entity (for stitching).
        # Phase 140: Track seed-to-neighbor counts for Intersection Bonus.
        seed_set: Set[str] = set(seeds)
        parent_map: Dict[str, TraversalPath] = {}
        neighbor_seed_counts: Dict[str, int] = {} # entity -> num seeds that reached it
        
        for p in scan_paths:
            eid = p.tail
            if eid not in seed_set:
                # Track best path to this neighbor
                if eid not in parent_map or p.score > parent_map[eid].score:
                    parent_map[eid] = p
                # Increment intersection count
                neighbor_seed_counts[eid] = neighbor_seed_counts.get(eid, 0) + 1

        # Rank hop-1 entities.
        # Multi-seed logic: boost score of entities reached by >1 seed.
        # Phase 172: Stage-1 anchor bonus for entities that can lead to the answer type.
        _s1_anchor = self._stage1_anchor  # Optional Tuple[Set[str], float]
        def _rank_key(eid: str) -> float:
            base_score = parent_map[eid].score
            # Intersection bonus: +20% per additional seed
            bonus = 1.0 + (0.2 * (neighbor_seed_counts[eid] - 1))
            # Phase 172: anchor bonus — prefer hop-1 entities in anchor set
            if _s1_anchor and eid in _s1_anchor[0]:
                bonus *= _s1_anchor[1]
            return base_score * bonus

        # sorted_neighbors is kept in scope for the Phase 145 residual sweep.
        sorted_neighbors = sorted(parent_map.keys(), key=_rank_key, reverse=True)
        if _hop1_audit_cb is not None:
            _hop1_audit_cb(sorted_neighbors, {e: _rank_key(e) for e in sorted_neighbors}, k_eff)
        hop1_entities: List[str] = []
        for eid in sorted_neighbors:
            hop1_entities.append(eid)
            if len(hop1_entities) >= k_eff:
                break

        # Phase 151: PenultimateGate — drop hop-1 branches below decay_factor * best_score.
        # Prevents weak intermediate branches from polluting Stage 2 final-hop expansion.
        if self.penultimate_decay > 0.0 and hop1_entities:
            best_h1 = max(_rank_key(e) for e in hop1_entities)
            floor_h1 = self.penultimate_decay * best_h1
            hop1_entities = [e for e in hop1_entities if _rank_key(e) >= floor_h1]

        # Stage 2: independent deep traversal per hop-1 entity.
        # Phase 139: Synchronization barrier for cross-branch pruning.
        barrier = GlobalBeamBarrier(target_count=len(hop1_entities), threshold_ratio=0.5)

        deep_hop = self.max_hop - 1
        # all_paths: List[TraversalPath] = list(scan_paths)
        # FIX: scan_paths already contains parent_map entries.
        # We start with scan_paths but ensure Stage 2 doesn't duplicate seeds.
        all_paths: List[TraversalPath] = list(scan_paths)

        # Phase 142 (Hotfix): Cycle prevention. Don't allow deep traversals
        # to go back to the original seeds.
        forbidden_seeds = set(seeds)

        for i, entity in enumerate(hop1_entities):
            parent = parent_map[entity]
            deep = self._make_traversal(
                max_hop=deep_hop,
                beam_widths=self._deep_beam_widths,
                per_budget=per_budget,
            )

            # Phase 184: Barrier evaluates sub-traversal quality independently
            # of hop-1 entity score. Prior: parent.score * top_score caused the
            # barrier to terminate sub-traversals for low-scoring hop-1 entities
            # (avg score ratio 0.227) before they could reach the answer at hop-2.
            def _prune_cb(top_score: float, sub_id=i) -> bool:
                return barrier.report(sub_id, top_score)

            # FIX: We need to prime the sub-traversal to avoid the forbidden seeds.
            # BeamTraversal doesn't have a direct forbidden_nodes list, but we can
            # use node_priming with a negative boost or just filter results.
            # Filtering is safer and easier.

            deep_paths = deep.traverse(
                [entity],
                query_embedding=query_embedding,
                trace_info=trace_info,
                node_priming=node_priming,
                pruning_callback=_prune_cb,
            )
            for dp in deep_paths:
                # Cycle prevention: if the deep path leads back to original seeds,
                # discard it as a redundant answer.
                if dp.tail in forbidden_seeds:
                    continue
                all_paths.append(_stitch(parent, dp))

        # Phase 145: Residual Hop-1 Expansion (RHE).
        # If unique terminal-depth answers are below min_diversity_target AND
        # unexplored hop-1 entities remain, run a reduced-budget residual sweep
        # to improve coverage of long-tail hop-1 branches.
        if deep_hop >= 2 and self.residual_k > 0:
            explored_set = set(hop1_entities[:k_eff])
            remaining_neighbors = [
                e for e in sorted_neighbors if e not in explored_set
            ]
            current_terminals = {
                p.tail for p in all_paths if p.hop_depth >= self.max_hop
            }
            if len(current_terminals) < self.min_diversity_target and remaining_neighbors:
                residual_budget = max(50, per_budget // 2)
                for entity in remaining_neighbors[:self.residual_k]:
                    parent = parent_map.get(entity)
                    if parent is None:
                        continue
                    deep = self._make_traversal(
                        max_hop=deep_hop,
                        beam_widths=self._deep_beam_widths,
                        per_budget=residual_budget,
                    )
                    deep_paths = deep.traverse(
                        [entity],
                        query_embedding=query_embedding,
                        trace_info=trace_info,
                        node_priming=node_priming,
                    )
                    for dp in deep_paths:
                        if dp.tail not in forbidden_seeds:
                            all_paths.append(_stitch(parent, dp))
                    _log.debug(
                        "Phase 145 RHE: residual entity=%s added %d paths",
                        entity, len(deep_paths),
                    )

        return all_paths

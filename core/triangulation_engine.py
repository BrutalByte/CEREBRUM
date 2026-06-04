"""
TriangulationEngine â€” Multi-Perspective Discovery Validation (Phase 72).

Examines each ResearchCandidate from four independent angles after the primary
HypothesisEngine run, producing four numeric signals that extend the AutoApprover
feature vector from 12 â†’ 16.

Design invariant: absence of structural precedent is NOT evidence of impossibility.
A relation type that has never crossed a given community distance is exactly what
the ResearchAgent is designed to find. This engine never penalises novelty â€”
it amplifies uncertainty for borderline cases and confirms confidence for strong ones.

Four perspectives
-----------------
P1  reverse_confidence      â€” HypothesisEngine run in the Bâ†’A direction
P2  strategy_agreement      â€” fraction of 3 strategy configs that find proposals
P3  mean_path_independence   â€” mean Jaccard independence from primary proposals (free)
P4  semantic_type_score      â€” relation-type entity-class consistency index
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger("cerebrum.triangulation")

# ---------------------------------------------------------------------------
# Additional strategy configs for P2.
# The primary run (3-hop, 8-beam, 300-budget) counts as one of the three.
# TriangulationEngine runs these two additional configs.
# ---------------------------------------------------------------------------
_EXTRA_STRATEGIES: List[Dict[str, int]] = [
    {"max_hop": 2, "beam_width": 5,  "max_budget": 100},   # conservative
    {"max_hop": 4, "beam_width": 15, "max_budget": 500},   # exploratory
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriangulationReport:
    """
    Four-perspective validation report attached to a ResearchFinding.
    Stored in ``finding.metadata["triangulation"]``.
    """

    reverse_confidence: float
    """P1: best Noisy-OR confidence of Bâ†’A hypothesis run. 0.0 if no reverse path."""

    strategy_agreement: float
    """P2: fraction of 3 strategy configs (conservative/standard/exploratory) that
    returned â‰¥1 valid proposal. 0.33 = only one, 1.0 = all three."""

    mean_path_independence: float
    """P3: mean Jaccard independence across primary proposals' independence_scores.
    0.5 if no proposals or no independence_scores."""

    semantic_type_score: float
    """P4: entity-type / relation-type consistency against the graph's existing edge
    index. 0.5 for completely novel relations (neutral), 1.0 for exact type match."""

    is_SynapticBridge_candidate: bool
    """Diagnostic flag â€” True when all four conditions for a genuine SynapticBridge are met.
    NOT used as a classifier feature (derivable from the four scores above)."""


# ---------------------------------------------------------------------------
# TriangulationEngine
# ---------------------------------------------------------------------------

class TriangulationEngine:
    """
    Runs four independent validation perspectives on a ResearchCandidate.

    Parameters
    ----------
    adapter
        GraphAdapter providing entity type lookups and graph access.
    hypothesis_engine
        Pre-constructed HypothesisEngine.  Shared with ResearchAgent â€” all
        calls are protected by the engine's internal RLock.
    min_confidence
        Minimum confidence for a proposal to count as "found" in P1/P2.
    run_bidirectional
        If False, skip the reverse-direction run (P1 always returns 0.0).
    run_multi_strategy
        If False, skip the two extra strategy runs.  P2 returns 1.0 if the
        primary produced proposals, 0.0 otherwise.
    """

    def __init__(
        self,
        adapter: Any,
        hypothesis_engine: Any,
        min_confidence: float = 0.20,
        run_bidirectional: bool = True,
        run_multi_strategy: bool = True,
    ) -> None:
        self._adapter = adapter
        self._hyp = hypothesis_engine
        self._min_confidence = min_confidence
        self._run_bidirectional = run_bidirectional
        self._run_multi_strategy = run_multi_strategy

        # Type index cache â€” invalidated when graph signature changes
        self._type_index: Optional[Dict[str, Set[Tuple[str, str]]]] = None
        self._type_index_sig: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: Any,
        primary_proposals: List[Any],
    ) -> TriangulationReport:
        """
        Evaluate a candidate from four perspectives.

        Parameters
        ----------
        candidate
            The ``ResearchCandidate`` that was evaluated.
        primary_proposals
            The ``List[HypothesisProposal]`` returned by the primary
            HypothesisEngine run (source â†’ target, standard config).
        """
        source_id = candidate.source_id
        target_id = candidate.target_id

        # Derive relation from best primary proposal (for P4)
        best_prop = (
            max(primary_proposals, key=lambda p: getattr(p, "confidence", 0.0))
            if primary_proposals else None
        )
        derived_relation = getattr(best_prop, "derived_relation", "") if best_prop else ""

        # P1 â€” bidirectional
        rev_conf = self._reverse_confidence(source_id, target_id)

        # P2 â€” multi-strategy agreement
        strat_agree = self._strategy_agreement(source_id, target_id, primary_proposals)

        # P3 â€” path independence (free from primary proposals)
        path_indep = self._mean_path_independence(primary_proposals)

        # P4 â€” semantic type consistency
        sem_score = self._semantic_type_score(source_id, target_id, derived_relation)

        # SynapticBridge diagnostic
        community_distance = getattr(candidate, "community_distance", 0)
        is_SynapticBridge = (
            rev_conf > 0.0
            and community_distance > 2
            and sem_score > 0.3
            and path_indep > 0.4
        )

        report = TriangulationReport(
            reverse_confidence=rev_conf,
            strategy_agreement=strat_agree,
            mean_path_independence=path_indep,
            semantic_type_score=sem_score,
            is_SynapticBridge_candidate=is_SynapticBridge,
        )

        logger.debug(
            "Triangulation %sâ†’%s: rev=%.3f agree=%.3f indep=%.3f sem=%.3f SynapticBridge=%s",
            source_id, target_id, rev_conf, strat_agree, path_indep, sem_score, is_SynapticBridge,
        )
        return report

    # ------------------------------------------------------------------
    # Perspective implementations
    # ------------------------------------------------------------------

    def _reverse_confidence(self, source_id: str, target_id: str) -> float:
        """P1: Run hypothesis engine in the Bâ†’A direction."""
        if not self._run_bidirectional:
            return 0.0
        try:
            rev_proposals = self._hyp.generate(
                source_id=target_id,
                target_id=source_id,
            )
            valid = [p for p in rev_proposals if getattr(p, "confidence", 0.0) >= self._min_confidence]
            if not valid:
                return 0.0
            return float(max(p.confidence for p in valid))
        except Exception as exc:
            logger.debug("_reverse_confidence error for %sâ†’%s: %s", target_id, source_id, exc)
            return 0.0

    def _strategy_agreement(
        self,
        source_id: str,
        target_id: str,
        primary_proposals: List[Any],
    ) -> float:
        """P2: Fraction of 3 strategy configs that found valid proposals."""
        # Primary (standard) counts as strategy #1
        primary_found = any(
            getattr(p, "confidence", 0.0) >= self._min_confidence
            for p in primary_proposals
        )
        strategies_found = 1 if primary_found else 0
        total_strategies = 1

        if self._run_multi_strategy:
            for config in _EXTRA_STRATEGIES:
                total_strategies += 1
                try:
                    proposals = self._hyp.generate(
                        source_id=source_id,
                        target_id=target_id,
                        **config,
                    )
                    if any(getattr(p, "confidence", 0.0) >= self._min_confidence for p in proposals):
                        strategies_found += 1
                except Exception as exc:
                    logger.debug("_strategy_agreement error (config=%s): %s", config, exc)
                    # Count as "not found" â€” conservative

        return strategies_found / total_strategies

    def _mean_path_independence(self, primary_proposals: List[Any]) -> float:
        """P3: Mean Jaccard independence across all primary proposals (free â€” already computed)."""
        all_scores: List[float] = []
        for prop in primary_proposals:
            scores = getattr(prop, "independence_scores", None) or []
            all_scores.extend(float(s) for s in scores)
        if not all_scores:
            return 0.5  # neutral default
        return sum(all_scores) / len(all_scores)

    def _semantic_type_score(
        self,
        source_id: str,
        target_id: str,
        derived_relation: str,
    ) -> float:
        """
        P4: Entity-type / relation-type consistency.

        Score table
        -----------
        - Relation not in index (novel)                 â†’ 0.5  (neutral, never penalise)
        - Exact (src_type, tgt_type) pair in index      â†’ 1.0
        - Both types appear in index (not as a pair)    â†’ 0.65
        - One type appears in index                     â†’ 0.65
        - Relation known, neither type appears          â†’ 0.30
        """
        if not derived_relation:
            return 0.5

        index = self._get_type_index()
        if derived_relation not in index:
            return 0.5  # novel relation â€” neutral, not penalised

        known_pairs = index[derived_relation]

        try:
            src_ent = self._adapter.get_entity(source_id)
            src_type = getattr(src_ent, "type", "entity") or "entity"
        except Exception:
            src_type = "entity"

        try:
            tgt_ent = self._adapter.get_entity(target_id)
            tgt_type = getattr(tgt_ent, "type", "entity") or "entity"
        except Exception:
            tgt_type = "entity"

        # Exact pair match
        if (src_type, tgt_type) in known_pairs:
            return 1.0

        # Partial matches
        src_match = any(pair[0] == src_type for pair in known_pairs)
        tgt_match = any(pair[1] == tgt_type for pair in known_pairs)
        score = 0.30 + 0.35 * int(src_match) + 0.35 * int(tgt_match)
        return round(score, 4)

    # ------------------------------------------------------------------
    # Type index â€” lazy build + graph-signature cache invalidation
    # ------------------------------------------------------------------

    def _get_type_index(self) -> Dict[str, Set[Tuple[str, str]]]:
        """
        Return the relation-type â†’ {(src_type, tgt_type)} index.
        Rebuilt whenever the graph grows (node_count or edge_count changes).
        """
        try:
            G = self._adapter.to_networkx()
            sig = (G.number_of_nodes(), G.number_of_edges())
        except Exception:
            sig = None

        if self._type_index is not None and sig == self._type_index_sig:
            return self._type_index

        # Build / rebuild
        index: Dict[str, Set[Tuple[str, str]]] = {}
        if G is not None:
            for u, v, data in G.edges(data=True):
                relation = data.get("relation", "related_to") or "related_to"
                src_type = self._safe_entity_type(u)
                tgt_type = self._safe_entity_type(v)
                if relation not in index:
                    index[relation] = set()
                index[relation].add((src_type, tgt_type))

        self._type_index = index
        self._type_index_sig = sig
        logger.debug(
            "TriangulationEngine: rebuilt type index (%d relation types, graph sig=%s)",
            len(index), sig,
        )
        return index

    def _safe_entity_type(self, entity_id: str) -> str:
        """Return entity type string, falling back to 'entity' on any error."""
        try:
            ent = self._adapter.get_entity(entity_id)
            return getattr(ent, "type", "entity") or "entity"
        except Exception:
            return "entity"

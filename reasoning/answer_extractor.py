"""
Extract and rank top-K answers from a set of traversal paths (Section 5, STEP 5).

An "answer" is the terminal entity of a path. Paths are re-scored using
score_path() and deduplicated by terminal entity (best score wins).

Contradiction surfacing: after ranking, detect_answer_contradictions() inspects
each pair of answers for cross-path contradictions (Type 2). When found, a
ContradictionFlag is attached to each conflicting Answer. Contradictions are
surfaced as research signals — they are never filtered from the result.
"""
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np

from reasoning.path_scorer import score_path, community_coherence, path_confidence, path_specificity_score
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reasoning.relation_path_prior import RelationPathPrior, GraphRelationPrior

# Lazy import to avoid circular dependency. Falls back gracefully if unavailable.
try:
    from core.contradiction_engine import relations_contradict
    _HAS_CONTRADICTION_ENGINE = True
except ImportError:
    _HAS_CONTRADICTION_ENGINE = False
    def relations_contradict(a: str, b: str) -> bool:  # type: ignore[misc]
        return False


@dataclass
class ContradictionFlag:
    """
    A detected contradiction between this answer and another in the result set.

    Contradictions are research signals: the conflicting claim may represent an
    emerging discovery, an unsettled debate, or a claim that has been superseded.
    The presence of a flag means both answers should be investigated, not that
    either should be discarded.
    """

    contradiction_type: str
    """Type of contradiction detected: "cross_path" for query-time detections."""

    conflicting_entity: str
    """Terminal entity of the answer that conflicts with this one."""

    conflicting_path: Any
    """The TraversalPath of the conflicting answer (for citation / inspection)."""

    this_confidence: float
    """path_confidence of this answer's path (weakest-link rule)."""

    conflicting_confidence: float
    """path_confidence of the conflicting answer's path."""

    contradicting_relations: Tuple[str, str]
    """The (rel_a, rel_b) pair that triggered the flag."""

    resolution_status: str = "unresolved"
    resolved_path: Any = None
    """The preferred path when resolution_status == 'resolved_by_credibility'."""
    note: str = ""

    def __repr__(self) -> str:
        a, b = self.contradicting_relations
        return (
            f"ContradictionFlag(type={self.contradiction_type!r}, "
            f"conflicts_with={self.conflicting_entity!r}, "
            f"relations=({a!r} vs {b!r}))"
        )


@dataclass
class Answer:
    """A single ranked answer extracted from a traversal."""

    entity_id: str
    """Terminal entity (the answer entity)."""

    score: float
    """Final ranked score."""

    best_path: Any
    """The TraversalPath that produced this answer (highest score)."""

    score_breakdown: Dict[str, float] = field(default_factory=dict)
    """Decomposition: {attention, community, semantic}."""

    community_trace: List[int] = field(default_factory=list)
    """Community ID sequence for the best path — shows conceptual trajectory."""

    path_confidence: float = 1.0
    """
    Weakest-link confidence along the best path's edges.
    1.0 when all edges are fully certain; lower values indicate one or more
    speculative edges (REM-synthesized, low-confidence ingest, etc.) on the
    reasoning chain. Use this to communicate answer credence to callers.
    """

    contradiction_flags: List[ContradictionFlag] = field(default_factory=list)
    """
    Any cross-path contradictions detected among the extracted answers.
    Non-empty means another answer in the same result set reached a conflicting
    conclusion via a contradicting relation. Inspect both answers and their paths.
    """

    score_uncertainty: float = 0.0
    """
    Beta-distribution variance of the best path at extraction time.
    Higher values indicate wider uncertainty over the path score (noisier evidence).
    0.0 when no Beta data is present (deterministic traversal).
    """

    path_score: float = 0.0
    """
    Raw score of the single best path to this entity (before consensus voting).
    """

    consensus_score: float = 0.0
    """
    Normalized weighted vote sum [0, 1] across all paths reaching this entity.
    """

    branch_count: int = 0
    """
    Phase 144: Number of distinct first-intermediate nodes (hop-1 branch sources)
    across all paths reaching this entity. branch_count > 1 means independent
    corroborating evidence from separate reasoning branches.
    """

    def __repr__(self) -> str:
        flags = f", flags={len(self.contradiction_flags)}" if self.contradiction_flags else ""
        return f"Answer(entity={self.entity_id!r}, score={self.score:.4f}{flags})"


def detect_answer_contradictions(answers: List[Answer]) -> None:
    """
    Detect Type 2 (cross-path) contradictions among extracted answers in-place.

    For each pair of answers, compares the intermediate edge relations in their
    best paths. When a pair of relations is found to be contradictory (per
    CONTRADICTION_PAIRS), a ContradictionFlag is appended to both answers.

    This operation is O(K^2 * max_path_len^2) where K = number of answers — fast
    for the typical K <= 10 result set.

    Modifies answer.contradiction_flags in-place; returns None.
    """
    if not _HAS_CONTRADICTION_ENGINE or len(answers) < 2:
        return

    for i, ans_a in enumerate(answers):
        for ans_b in answers[i + 1:]:
            path_a = ans_a.best_path
            path_b = ans_b.best_path
            if path_a is None or path_b is None:
                continue

            # Extract edge relations from path.nodes (odd-index elements)
            rels_a = [path_a.nodes[k] for k in range(1, len(path_a.nodes), 2)]
            rels_b = [path_b.nodes[k] for k in range(1, len(path_b.nodes), 2)]

            for ra in rels_a:
                for rb in rels_b:
                    if ra == rb:
                        continue
                    if relations_contradict(ra, rb):
                        conf_a = path_confidence(path_a)
                        conf_b = path_confidence(path_b)
                        flag_for_a = ContradictionFlag(
                            contradiction_type="cross_path",
                            conflicting_entity=ans_b.entity_id,
                            conflicting_path=path_b,
                            this_confidence=conf_a,
                            conflicting_confidence=conf_b,
                            contradicting_relations=(ra, rb),
                            note=(
                                f"Path to {ans_a.entity_id!r} uses {ra!r}; "
                                f"path to {ans_b.entity_id!r} uses {rb!r}"
                            ),
                        )
                        flag_for_b = ContradictionFlag(
                            contradiction_type="cross_path",
                            conflicting_entity=ans_a.entity_id,
                            conflicting_path=path_a,
                            this_confidence=conf_b,
                            conflicting_confidence=conf_a,
                            contradicting_relations=(rb, ra),
                            note=(
                                f"Path to {ans_b.entity_id!r} uses {rb!r}; "
                                f"path to {ans_a.entity_id!r} uses {ra!r}"
                            ),
                        )
                        # Phase 221-C: resolve via grounding (credibility baked in via Phase 216).
                        # Mean edge grounding (index 8) reflects source credibility.
                        def _mean_grounding(p: Any) -> float:
                            feats = getattr(p, "edge_features", None) or []
                            gs = [f[8] for f in feats if len(f) > 8]
                            return sum(gs) / len(gs) if gs else 0.5
                        g_a = _mean_grounding(path_a)
                        g_b = _mean_grounding(path_b)
                        if abs(g_a - g_b) > 0.05:
                            winner = path_a if g_a >= g_b else path_b
                            flag_for_a.resolution_status = "resolved_by_credibility"
                            flag_for_a.resolved_path = winner
                            flag_for_b.resolution_status = "resolved_by_credibility"
                            flag_for_b.resolved_path = winner

                        ans_a.contradiction_flags.append(flag_for_a)
                        ans_b.contradiction_flags.append(flag_for_b)
                        # One flag per pair is enough
                        break
                else:
                    continue
                break


def extract(
    paths: List[Any],
    top_k: int = 5,
    query_embedding: Optional[np.ndarray] = None,
    min_hop: int = 1,
    weight_attention: float = 0.4,
    weight_community: float = 0.3,
    weight_semantic: float  = 0.3,
    vote_weight: float = 0.45,
    relation_prior: Optional[Any] = None,
    weight_prior: float = 0.15,
    branch_bonus_weight: float = 0.0,
    degree_penalty_weight: float = 0.0,
    adapter: Optional[Any] = None,
    fan_out: Optional[Dict] = None,
    weight_specificity: float = 0.0,
) -> List[Answer]:
    """
    Extract the top-K answer entities from a list of TraversalPaths.

    Steps:
      1. Filter out depth-0 seed paths (no hops taken)
      2. Re-score all paths using score_path()
      3. Deduplicate by terminal entity, accumulating path convergence vote count
      4. Final score = (1 - vote_weight) * path_score + vote_weight * normalised_vote_count
      5. Return top-K sorted by combined score descending

    Parameters
    ----------
    paths          : list of TraversalPath objects from BeamTraversal.traverse()
    top_k          : number of answers to return
    query_embedding: optional query vector for semantic alignment re-scoring.
                     Pass the seed entity's embedding to activate the semantic
                     alignment term in score_path().
    min_hop        : minimum hop depth to consider (default 1 — excludes seeds)
    vote_weight    : weight given to path-convergence vote count vs path score.
                     Higher values prefer entities reached by more distinct paths
                     (ensemble-like voting). Default 0.3.
                     Set to 0.0 to restore legacy behaviour (path score only).
    branch_bonus_weight : Phase 144 — multiplicative bonus scaling for answers
                     reached via multiple distinct first-intermediate-node branches.
                     0.25 means a 2-branch answer scores ~17% higher (log1p(1)*0.25);
                     a 4-branch answer scores ~35% higher. Set to 0.0 to disable.

    Returns
    -------
    List of Answer objects, sorted by combined score descending.
    """
    # Filter to paths with at least min_hop hops
    valid = [p for p in paths if p.hop_depth >= min_hop]
    if not valid:
        return []

    # Score all valid paths
    scored = [
        (
            p,
            score_path(
                p,
                query_embedding=query_embedding,
                weight_attention=weight_attention,
                weight_community=weight_community,
                weight_semantic=weight_semantic,
                relation_prior=relation_prior,
                weight_prior=weight_prior,
            ),
        )
        for p in valid
    ]

    # Deduplicate: keep best-scoring path per terminal entity AND accumulate
    # score-weighted votes for path convergence ranking.
    # Each path contributes its score as a vote weight — high-confidence paths
    # count more than low-confidence paths toward the entity's vote total.
    # More/stronger independent reasoning chains converging = stronger signal.
    #
    # Phase 144: Also track distinct first-intermediate nodes (branch sources).
    # Paths sharing nodes[2] came from the same hop-1 branch; paths with different
    # nodes[2] represent independent corroborating evidence.
    best: Dict[str, tuple] = {}        # entity -> (best_path, best_score)
    vote_weight_sum: Dict[str, float] = {}  # entity -> sum of path scores
    branch_sets: Dict[str, set] = {}   # entity -> set of distinct branch keys
    for path, s in scored:
        entity = path.tail
        vote_weight_sum[entity] = vote_weight_sum.get(entity, 0.0) + s
        if entity not in best or s > best[entity][1]:
            best[entity] = (path, s)
        # Branch key: nodes[2] for multi-hop paths (the hop-1 intermediary);
        # nodes[0] (seed) for 1-hop paths where no branching applies.
        branch_id = path.nodes[2] if len(path.nodes) >= 5 else path.nodes[0]
        branch_sets.setdefault(entity, set()).add(branch_id)

    # Normalise weighted vote sums to [0, 1] relative to the highest-voted entity
    max_vote_sum = max(vote_weight_sum.values()) if vote_weight_sum else 1.0

    # Phase 179: Precompute PSS for each candidate's best path and normalise.
    _pss_raw: Dict[str, float] = {}
    if fan_out is not None and weight_specificity > 0.0:
        for eid, (path, _) in best.items():
            _pss_raw[eid] = path_specificity_score(path, fan_out)
        _max_pss = max(_pss_raw.values()) if _pss_raw else 1.0
        _norm_pss: Dict[str, float] = {
            eid: s / _max_pss for eid, s in _pss_raw.items()
        }
    else:
        _norm_pss = {}

    # Combine path score with normalised weighted vote sum.
    # Phase 144: Apply branch-diversity multiplier — log-scale bonus for answers
    # reached via multiple independent hop-1 branches.
    import math as _math
    def combined(entity_id: str, path_score: float) -> float:
        if vote_weight <= 0.0:
            base = path_score
        else:
            norm_votes = vote_weight_sum[entity_id] / max_vote_sum
            base = (1.0 - vote_weight) * path_score + vote_weight * norm_votes
        if branch_bonus_weight > 0.0:
            n_br = len(branch_sets.get(entity_id, set()))
            branch_factor = 1.0 + branch_bonus_weight * _math.log1p(n_br - 1) if n_br > 1 else 1.0
            base *= branch_factor
        if degree_penalty_weight > 0.0 and adapter is not None:
            # Phase 148: Degree Penalty to suppress hub flooding
            deg = adapter.get_degree(entity_id)
            penalty = 1.0 / (1.0 + degree_penalty_weight * _math.log1p(deg))
            base *= penalty
        # Phase 179: Path Specificity Score — penalise hub-like traversals
        if _norm_pss:
            pss = _norm_pss.get(entity_id, 0.0)
            base = (1.0 - weight_specificity) * base + weight_specificity * pss
        return base

    # Sort and return top-K
    top_candidates = sorted(
        best.items(),
        key=lambda kv: combined(kv[0], kv[1][1]),
        reverse=True,
    )[:top_k]
    # Unpack to (path, combined_score) for answer construction below
    # Clamp scores to [0, 1] for display. Ranking itself uses un-clamped combined()
    # so the branch multiplier can exceed 1.0 during sorting without breaking callers.
    ranked_answers = [(v[0], min(1.0, combined(k, v[1]))) for k, v in top_candidates]

    answers = []
    for path, s in ranked_answers:
        # Build score breakdown
        import math
        attn_score = math.prod(max(w, 1e-9) for w in path.attention_weights) if path.attention_weights else 0.0
        coh        = community_coherence(path.community_sequence)

        if query_embedding is not None and path.embedding is not None:
            qn = float(np.linalg.norm(query_embedding))
            pn = float(np.linalg.norm(path.embedding))
            if qn > 0 and pn > 0:
                sem = (float(np.dot(query_embedding, path.embedding) / (qn * pn)) + 1.0) / 2.0
            else:
                sem = 0.5
        else:
            sem = None

        breakdown = {
            "attention":  round(attn_score, 4),
            "community":  round(coh, 4),
        }
        if sem is not None:
            breakdown["semantic"] = round(sem, 4)

        # Consensus score calculation
        ent_id = path.tail
        raw_path_score = best[ent_id][1]
        norm_votes = (vote_weight_sum[ent_id] / max_vote_sum) if max_vote_sum > 0 else 0.0

        answers.append(
            Answer(
                entity_id=ent_id,
                score=round(s, 6),
                best_path=path,
                score_breakdown=breakdown,
                community_trace=path.community_sequence,
                path_confidence=path_confidence(path),
                score_uncertainty=path.score_variance,
                path_score=round(raw_path_score, 6),
                consensus_score=round(norm_votes, 6),
                branch_count=len(branch_sets.get(ent_id, set())),
            )
        )

    # Surface any cross-path contradictions among the result set.
    # Contradictions are research signals — they are attached as flags,
    # never used to filter or reorder results.
    detect_answer_contradictions(answers)

    return answers


def counterfactual_rerank(
    answers: List[Answer],
    seeds: List[str],
    adapter: Any,
    robustness_weight: float = 0.2,
    causal_relations: Optional[Any] = None,
) -> List[Answer]:
    """
    Phase 126: Re-rank answers by counterfactual robustness.

    For each answer, block its intermediate path node(s) and measure how much
    the causal effect drops. Answers that survive intervention (small effect_delta)
    are more robust — they rank higher.

    formula: final_score = base_score * (1 + robustness_weight * (1 - |effect_delta|))

    Parameters
    ----------
    answers           : ranked answers from extract()
    seeds             : original query seeds (used as causal source)
    adapter           : GraphAdapter — passed to CounterfactualEngine
    robustness_weight : how much robustness boosts score (default 0.2)
    causal_relations  : optional frozenset; defaults to CAUSAL_RELATIONS

    Returns
    -------
    Re-ranked List[Answer], sorted by adjusted score descending.
    """
    if not answers or not seeds:
        return answers

    try:
        from core.counterfactual_engine import CounterfactualEngine, Intervention, BLOCK_NODE
    except ImportError:
        return answers

    if causal_relations is None:
        try:
            from core.causal_engine import CAUSAL_RELATIONS
            causal_relations = CAUSAL_RELATIONS
        except ImportError:
            causal_relations = frozenset()

    engine = CounterfactualEngine(adapter=adapter, causal_relations=causal_relations)
    seed = seeds[0]

    reranked = []
    for ans in answers:
        path = ans.best_path
        entity_nodes = getattr(path, "entity_nodes", None) if path else None
        # Collect intermediate nodes (exclude seed and answer endpoints)
        intermediates = (
            [n for n in entity_nodes[1:-1] if n != seed and n != ans.entity_id]
            if entity_nodes and len(entity_nodes) > 2
            else []
        )

        if not intermediates:
            reranked.append((ans, ans.score))
            continue

        try:
            ivs = [Intervention(type=BLOCK_NODE, node=n) for n in intermediates]
            cf = engine.query(seed, ans.entity_id, interventions=ivs, max_hop=4)
            robustness = 1.0 - abs(cf.effect_delta)
            adjusted = ans.score * (1.0 + robustness_weight * robustness)
        except Exception:
            adjusted = ans.score

        reranked.append((ans, adjusted))

    reranked.sort(key=lambda x: x[1], reverse=True)
    # Update scores in place and return
    for ans, new_score in reranked:
        ans.score = round(new_score, 6)
    return [ans for ans, _ in reranked]





def deductive_consensus_rerank(
    answers: List[Answer],
    seed: str,
    deductive_traversal: Any,
    boost: float = 1.3,
    penalty: float = 0.9,
) -> List[Answer]:
    """Phase 132: re-rank answers by deductive BFS consensus.

    For each answer entity, run DeductiveTraversal.traverse(seed, target, causal_only=True).
    A non-empty proof list → agreement (multiply score by boost).
    Empty proof list → no causal chain found (multiply score by penalty).
    """
    # Phase 134: cache proof results — avoids redundant O(V+E) BFS for duplicate entity_ids
    _proof_cache: Dict[Tuple[str, str], bool] = {}
    for ans in answers:
        try:
            _key = (seed, ans.entity_id)
            if _key not in _proof_cache:
                _proof_cache[_key] = bool(
                    deductive_traversal.traverse(seed, ans.entity_id, causal_only=True)
                )
            if _proof_cache[_key]:
                ans.score = round(ans.score * boost, 6)
            else:
                ans.score = round(ans.score * penalty, 6)
        except Exception:
            pass
    answers.sort(key=lambda a: a.score, reverse=True)
    return answers

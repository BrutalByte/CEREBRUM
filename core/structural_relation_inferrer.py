"""
StructuralRelationInferrer (SRI) — Phase 161

Infers candidate terminal relations from graph topology alone.
No domain knowledge, no keyword lists, no LLM.

Structural insight: in well-structured knowledge graphs, answer entities
tend to be less central than intermediate hub entities. A 3-hop path reads
  seed → (hub) → (hub) → specific_answer
Relations whose target entities are low-degree and high-diversity are more
likely to be terminal (answer-type) relations for a given query.

Usage:
    sri = StructuralRelationInferrer()
    sri.build(adapter)                  # once, after graph load
    boost = sri.to_boost_dict(seed_id, n_hops, adapter)
    # boost is Dict[str, float] in the same format as terminal_relation_boost
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter


class StructuralRelationInferrer:
    """
    Precomputes per-relation structural statistics at build() time.
    At query() time, ranks candidate terminal relations by structural
    specificity — no domain knowledge, no LLM, no question text.
    """

    def __init__(self) -> None:
        self._rel_freq: Dict[str, int] = {}
        self._rel_n_unique_targets: Dict[str, int] = {}
        self._rel_n_unique_sources: Dict[str, int] = {}
        self._rel_target_degree_sum: Dict[str, float] = {}
        self._specificity: Dict[str, float] = {}
        self._built: bool = False
        # Phase 162: community fingerprints
        self._community_dominant_rel: Dict[int, str] = {}
        self._community_purity: Dict[int, float] = {}
        self._community_rel_counts: Dict[int, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Build-time: one O(E) pass via adapter.get_relation_statistics()
    # ------------------------------------------------------------------

    def build(self, adapter: "GraphAdapter") -> None:
        """Compute per-relation structural stats. Call once after graph load."""
        stats = adapter.get_relation_statistics()
        for rel, s in stats.items():
            self._rel_freq[rel] = s["freq"]
            self._rel_n_unique_targets[rel] = s["n_unique_targets"]
            self._rel_n_unique_sources[rel] = s["n_unique_sources"]
            self._rel_target_degree_sum[rel] = s["target_degree_sum"]
        self._specificity = self._compute_specificity()
        self._built = True

    def _compute_specificity(self) -> Dict[str, float]:
        """
        specificity(r) = target_diversity(r) / (1 + log1p(mean_target_degree(r)))

        target_diversity = unique_targets / freq  ∈ [0, 1]
          High → each edge reaches a different entity (specific answers)
          Low  → many edges converge to the same few entities (hub-attracting)

        mean_target_degree: penalises relations whose targets are structural hubs.
        """
        out: Dict[str, float] = {}
        for rel, f in self._rel_freq.items():
            if f == 0:
                continue
            target_diversity = self._rel_n_unique_targets.get(rel, 0) / f
            mean_deg = self._rel_target_degree_sum.get(rel, 0.0) / f
            out[rel] = target_diversity / (1.0 + math.log1p(mean_deg))
        return out

    # ------------------------------------------------------------------
    # Query-time: rank candidate terminal relations
    # ------------------------------------------------------------------

    def rank_relations(
        self,
        seed_id: str,
        n_hops: int,
        adapter: "GraphAdapter",
        top_k: int = 3,
        min_confidence_ratio: float = 1.5,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate terminal relations for the given seed and hop depth.

        Combines global specificity (70%) with seed-local complement (30%).
        For n_hops >= 2: relations absent from seed's hop-1 neighbourhood are
        favoured as terminal candidates (they are not the "entry" relation).

        Returns [] when top-1 / top-2 score ratio < min_confidence_ratio,
        signalling that the graph is too symmetric to make a reliable choice.
        Callers should treat [] as "apply no TRB".
        """
        if not self._built or not self._specificity:
            return []

        scores: Dict[str, float] = dict(self._specificity)

        # Seed-local complement: down-weight relations dominant at hop-1
        if n_hops >= 2:
            try:
                seed_edges = adapter.get_neighbors(seed_id, max_neighbors=200)
                seed_rel_counts: Dict[str, int] = defaultdict(int)
                for e in seed_edges:
                    seed_rel_counts[e.relation_type] += 1
                total = max(1, len(seed_edges))
                for rel in list(scores.keys()):
                    local_rate = seed_rel_counts.get(rel, 0) / total
                    scores[rel] = 0.7 * scores[rel] + 0.3 * (1.0 - local_rate)
            except Exception:
                pass  # safe fallback — use global specificity only

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return ranked[:top_k]

        top_score = ranked[0][1]
        second_score = ranked[1][1]
        if second_score <= 0 or (top_score / max(second_score, 1e-9)) < min_confidence_ratio:
            return []  # too ambiguous — caller should skip TRB

        return ranked[:top_k]

    def to_boost_dict(
        self,
        seed_id: str,
        n_hops: int,
        adapter: "GraphAdapter",
        min_boost: float = 1.0,
        max_boost: float = 5.0,
        min_confidence_ratio: float = 1.5,
        hard_select: bool = False,
        hard_select_factor: float = 5.0,
    ) -> Dict[str, float]:
        """
        Convert structural relation scores to a TRB-compatible dict.

        Soft mode (default, hard_select=False):
            All known relations receive a proportional boost in [min_boost, max_boost].
            High-specificity relations are favoured; no relation is hard-blocked.
            Safe for arbitrary graphs — never crashes H@1 below no-TRB baseline.

        Hard-select mode (hard_select=True):
            Top-1 relation receives hard_select_factor boost; all others are absent
            from the dict (traversal defaults them to 0.01×).
            Higher upside when inference is correct; larger penalty when wrong.
            Gated by min_confidence_ratio — returns {} if ambiguous.

        Returns {} on any error or when confidence is insufficient.
        """
        if not self._built or not self._specificity:
            return {}

        if hard_select:
            ranked = self.rank_relations(
                seed_id, n_hops, adapter,
                top_k=1,
                min_confidence_ratio=min_confidence_ratio,
            )
            if not ranked:
                return {}
            return {ranked[0][0]: hard_select_factor}

        # Soft mode: normalise global specificity scores to [min_boost, max_boost]
        scores = self._specificity
        lo = min(scores.values())
        hi = max(scores.values())
        rng = hi - lo if hi > lo else 1.0
        return {
            rel: min_boost + ((s - lo) / rng) * (max_boost - min_boost)
            for rel, s in scores.items()
        }

    # ------------------------------------------------------------------
    # Phase 162: community fingerprinting (build-time)
    # ------------------------------------------------------------------

    def build_community_fingerprints(self, adapter: "GraphAdapter") -> None:
        """
        O(E) scan: for each edge (u→v, relation), tally relation type at the
        target entity's community. Compute dominant relation and purity per
        community.

        Purity = max_rel_count / total_edge_count for that community.
        High purity (>0.5): community is entity-type-pure — reliable inference.
        Low purity (<0.3): mixed community — inference skipped.
        """
        community_map = getattr(adapter, "community_map", {})
        if not community_map:
            return
        G = adapter.to_networkx()
        cid_rels: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            # Count only incoming edges for each community — terminal entities
            # are the *targets* of the final-hop relation. Counting source-side
            # would inflate movie communities with every relation type.
            v_cid = community_map.get(v, -1)
            if v_cid >= 0:
                cid_rels[v_cid][rel] += 1
        for cid, rel_counts in cid_rels.items():
            total = sum(rel_counts.values())
            dominant = max(rel_counts, key=rel_counts.__getitem__)
            self._community_dominant_rel[cid] = dominant
            self._community_purity[cid] = rel_counts[dominant] / total if total else 0.0
            self._community_rel_counts[cid] = dict(rel_counts)

    # ------------------------------------------------------------------
    # Phase 162: post-traversal community consensus boost (query-time)
    # ------------------------------------------------------------------

    def community_consensus_boost(
        self,
        answers_obj: List,
        adapter: "GraphAdapter",
        boost_factor: float = 3.0,
        penalty_factor: float = 1.0,
        min_purity: float = 0.50,
        min_consensus_fraction: float = 0.65,
        top_k_candidates: int = 50,
    ) -> bool:
        """
        Post-traversal re-ranking using path-based terminal relation consensus.

        Primary: vote on the actual terminal relation in each answer's traversal
        path (best_path.nodes[-2]). Confidence = winning_rel_weight / total.
        When consensus is strong enough:
          - boost_factor applied to answers matching consensus relation
          - penalty_factor applied to answers NOT matching consensus relation
        The combination of boost+penalty is more discriminative than boost alone.

        Fallback (no path info): community-based voting using DSCF fingerprints.

        Returns True if re-ranking was applied, False if confidence insufficient.
        """
        rel_weights: Dict[str, float] = defaultdict(float)
        has_path_info: bool = False

        # Determine max path depth in top-K candidates.
        # Only vote from the deepest paths — short paths (2-hop when expecting 3-hop)
        # would pollute the vote with 1-hop-away relation types (genres, tags, years)
        # that happen to score well due to hop_decay favouring short paths.
        max_len = 0
        for ans in answers_obj[:top_k_candidates]:
            path = getattr(ans, "best_path", None)
            if path is not None and path.nodes:
                max_len = max(max_len, len(path.nodes))
        # Accept paths within 2 nodes of the maximum depth (one hop shorter)
        min_len = max(3, max_len - 2)

        # Primary: path-based vote (deep paths only)
        for ans in answers_obj[:top_k_candidates]:
            path = getattr(ans, "best_path", None)
            if path is not None and path.nodes and len(path.nodes) >= min_len:
                terminal_rel = path.nodes[-2]
                if isinstance(terminal_rel, str):
                    has_path_info = True
                    rel_weights[terminal_rel] += ans.score

        # Fallback: community-based vote when path info is absent
        if not rel_weights and self._community_dominant_rel:
            community_map = getattr(adapter, "community_map", {})
            for ans in answers_obj[:top_k_candidates]:
                cid = community_map.get(ans.entity_id, -1)
                if cid < 0:
                    continue
                purity = self._community_purity.get(cid, 0.0)
                if purity < min_purity:
                    continue
                dom_rel = self._community_dominant_rel.get(cid)
                if dom_rel:
                    rel_weights[dom_rel] += ans.score * purity

        if not rel_weights:
            return False
        total = sum(rel_weights.values())
        consensus_rel = max(rel_weights, key=rel_weights.__getitem__)
        confidence = rel_weights[consensus_rel] / total
        if confidence < min_consensus_fraction:
            return False

        # Scale boost/penalty by confidence: stronger consensus → sharper re-ranking
        eff_boost = 1.0 + (boost_factor - 1.0) * confidence
        eff_penalty = penalty_factor + (1.0 - penalty_factor) * (1.0 - confidence)

        changed = False
        for ans in answers_obj:
            if has_path_info:
                path = getattr(ans, "best_path", None)
                if path is not None and path.nodes and len(path.nodes) >= min_len:
                    if path.nodes[-2] == consensus_rel:
                        ans.score *= eff_boost
                    else:
                        ans.score *= eff_penalty
                    changed = True
            else:
                community_map = getattr(adapter, "community_map", {})
                cid = community_map.get(ans.entity_id, -1)
                if cid < 0:
                    continue
                if (self._community_dominant_rel.get(cid) == consensus_rel
                        and self._community_purity.get(cid, 0.0) >= min_purity):
                    ans.score *= eff_boost
                else:
                    ans.score *= eff_penalty
                changed = True

        if changed:
            answers_obj.sort(key=lambda a: a.score, reverse=True)
        return changed

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> List[Tuple[str, float, int, float]]:
        """Return [(relation, specificity, freq, mean_target_degree)] sorted by specificity desc."""
        rows = []
        for rel, spec in sorted(self._specificity.items(), key=lambda x: x[1], reverse=True):
            f = self._rel_freq.get(rel, 0)
            mean_deg = (self._rel_target_degree_sum.get(rel, 0.0) / f) if f else 0.0
            rows.append((rel, spec, f, mean_deg))
        return rows

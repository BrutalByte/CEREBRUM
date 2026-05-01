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

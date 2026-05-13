"""
Phase 172: GraphProfiler — Automatic Query Strategy Selection.

Computes structural signals from the loaded graph at build time and produces
a QueryProfile that drives per-query defaults for hop_expand, auto_infer_terminal_relation,
and anchor_bonus. Eliminates the need for manual per-graph configuration.

Signals (all O(E) or O(V)):
  hub_score          — fraction of edges incident to top-1% degree nodes.
                       High (>0.30) = hub-heavy (MetaQA-like) → H1SE helps.
  degree_cv          — coefficient of variation of the degree distribution.
                       High (>2.0) = heavy-tailed degree → hub bottleneck exists.
  mean_rel_coverage  — mean over all relation types of |source_nodes(R)| / |nodes|.
                       High (>0.60) = all entities are sources of most relations
                       (homogeneous graph) → TRB cannot discriminate.
  min_rel_coverage   — minimum coverage across relation types.
                       Low (<0.10) = at least one relation is typed-specific
                       (strict source subset) → TRB will discriminate.

Regime classification:
  "hub_homogeneous"    hub_score > HUB_THRESH  AND  mean_rel_coverage > COV_THRESH
                       Example: MetaQA — all hop-2 entities are movies.
                       → recommended_hop_expand=True, recommended_trb_auto=False

  "typed_heterogeneous" hub_score <= HUB_THRESH  AND  min_rel_coverage < MIN_COV_THRESH
                       Example: Hetionet — disease, gene, compound communities.
                       → recommended_hop_expand=False, recommended_trb_auto=True

  "mixed"              everything else
                       → recommended_hop_expand=True, recommended_trb_auto=True (safe fallback)

Usage:
    profile = GraphProfiler.profile(adapter, anchor_sources)
    graph._query_profile = profile
    print(profile.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Tunable thresholds derived from MetaQA vs Hetionet empirical analysis.
HUB_THRESH     = 0.30   # fraction of edges on top-1% nodes that flags hub-heavy
DEGREE_CV_THRESH = 2.0  # CV above this also indicates hub structure
COV_THRESH     = 0.60   # mean relation coverage above this = homogeneous
MIN_COV_THRESH = 0.10   # min relation coverage below this = has typed relations


@dataclass
class QueryProfile:
    """
    Structural fingerprint of a loaded graph and recommended query defaults.
    Stored as graph._query_profile after build().
    """
    # --- raw signals ---
    hub_score:         float                  # fraction of edges on top-1% nodes
    degree_cv:         float                  # coefficient of variation of degree dist
    mean_rel_coverage: float                  # mean |sources(R)| / |nodes|
    min_rel_coverage:  float                  # min coverage over all relations
    typed_relations:   List[str]              # relations with coverage < MIN_COV_THRESH
    n_nodes:           int
    n_edges:           int
    n_relation_types:  int

    # --- recommendations ---
    recommended_hop_expand:   bool
    recommended_trb_auto:     bool            # set auto_infer_terminal_relation=True
    recommended_anchor_bonus: Optional[float] # None = disable TAB
    regime:                   str             # "hub_homogeneous" | "typed_heterogeneous" | "mixed"

    def summary(self) -> str:
        lines = [
            f"  GraphProfile ({self.regime})",
            f"    Nodes: {self.n_nodes:,}  Edges: {self.n_edges:,}  "
            f"Relation types: {self.n_relation_types}",
            f"    hub_score={self.hub_score:.3f}  degree_cv={self.degree_cv:.3f}  "
            f"mean_rel_coverage={self.mean_rel_coverage:.3f}  "
            f"min_rel_coverage={self.min_rel_coverage:.3f}",
            f"    Typed relations (<{MIN_COV_THRESH*100:.0f}% node coverage): "
            f"{len(self.typed_relations)} "
            + (f"({', '.join(self.typed_relations[:3])}" +
               (f", +{len(self.typed_relations)-3} more" if len(self.typed_relations) > 3 else "")
               + ")" if self.typed_relations else "(none)"),
            f"    Recommended: hop_expand={self.recommended_hop_expand}  "
            f"trb_auto={self.recommended_trb_auto}  "
            f"anchor_bonus={self.recommended_anchor_bonus}",
        ]
        return "\n".join(lines)


class GraphProfiler:
    """Stateless profiler — call GraphProfiler.profile(adapter, anchor_sources)."""

    @staticmethod
    def profile(
        adapter,
        anchor_sources: Dict[str, Set[str]],
        hub_percentile: float = 0.99,
    ) -> QueryProfile:
        """
        Compute structural signals and return a QueryProfile.

        Parameters
        ----------
        adapter        : any GraphAdapter with to_networkx()
        anchor_sources : dict from CerebrumGraph._anchor_sources (built in Phase 172)
        hub_percentile : top fraction of nodes considered "hubs" (default 1%)
        """
        G = adapter.to_networkx()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes == 0:
            return QueryProfile(
                hub_score=0.0, degree_cv=0.0,
                mean_rel_coverage=0.0, min_rel_coverage=0.0,
                typed_relations=[], n_nodes=0, n_edges=0, n_relation_types=0,
                recommended_hop_expand=False, recommended_trb_auto=False,
                recommended_anchor_bonus=None, regime="mixed",
            )

        # ----------------------------------------------------------------
        # Hub score: fraction of edges incident to top-p% degree nodes
        # ----------------------------------------------------------------
        degrees = sorted((d for _, d in G.degree()), reverse=True)
        hub_count = max(1, int(math.ceil(n_nodes * (1.0 - hub_percentile))))
        hub_degree_sum = sum(degrees[:hub_count])
        # Each edge contributes 2 to total degree sum in undirected graph
        total_degree = sum(degrees)
        hub_score = hub_degree_sum / total_degree if total_degree > 0 else 0.0

        # ----------------------------------------------------------------
        # Degree CV
        # ----------------------------------------------------------------
        mean_deg = total_degree / n_nodes if n_nodes > 0 else 0.0
        variance = sum((d - mean_deg) ** 2 for d in degrees) / n_nodes if n_nodes > 0 else 0.0
        degree_cv = math.sqrt(variance) / mean_deg if mean_deg > 0 else 0.0

        # ----------------------------------------------------------------
        # Relation coverage — O(E) pass on the graph directly.
        # anchor_sources (Phase 172) only adds _u per edge, correct for
        # directed graphs. For undirected graphs both endpoints can be
        # traversed from, so we recount here using both u and v.
        # This gives the correct "what fraction of nodes can be a source
        # of this relation type in a traversal?" semantics.
        # ----------------------------------------------------------------
        is_directed = G.is_directed()
        _rel_sources: Dict[str, Set[str]] = {}
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", "")
            if not rel:
                continue
            _rel_sources.setdefault(rel, set()).add(u)
            if not is_directed:
                _rel_sources.setdefault(rel, set()).add(v)

        n_relation_types = len(_rel_sources)

        if _rel_sources and n_nodes > 0:
            coverages = [len(s) / n_nodes for s in _rel_sources.values()]
            mean_rel_coverage = sum(coverages) / len(coverages)
            min_rel_coverage  = min(coverages)
            typed_relations   = [
                rel for rel, s in _rel_sources.items()
                if len(s) / n_nodes < MIN_COV_THRESH
            ]
        else:
            mean_rel_coverage = 0.0
            min_rel_coverage  = 0.0
            typed_relations   = []

        # ----------------------------------------------------------------
        # Regime classification
        # ----------------------------------------------------------------
        # Use hub_score (edge-fraction) as the primary hub signal.
        # degree_cv is informative but fires spuriously on typed heterogeneous graphs
        # whose high-degree nodes are biologically meaningful, not structural bottlenecks
        # (e.g. Hetionet gene nodes have hundreds of disease connections but H1SE hurts).
        # hub_score measures what fraction of graph traversal flows through a small set
        # of nodes — a direct proxy for "will hub competition starve the beam?"
        is_hub_heavy   = hub_score > HUB_THRESH
        is_homogeneous = mean_rel_coverage > COV_THRESH
        has_typed_rels = min_rel_coverage < MIN_COV_THRESH

        if is_hub_heavy and not has_typed_rels:
            # Pure hub graph: high-degree hubs, but relations aren't type-restricted.
            # Example: MetaQA — all nodes can be sources of all relations.
            regime = "hub_homogeneous"
            recommended_hop_expand = True
            recommended_trb_auto   = False
            recommended_anchor_bonus = None
        elif (not is_hub_heavy) and has_typed_rels:
            # Typed heterogeneous graph: entities are typed, relations are selective.
            # Example: Hetionet — only Compounds treat Diseases.
            regime = "typed_heterogeneous"
            recommended_hop_expand = False
            recommended_trb_auto   = True
            recommended_anchor_bonus = 2.0
        elif is_hub_heavy and has_typed_rels:
            # Hub graph WITH typed relations: use both H1SE and TRB.
            regime = "mixed"
            recommended_hop_expand = True
            recommended_trb_auto   = True
            recommended_anchor_bonus = 2.0
        else:
            # Low-degree, homogeneous: safe fallback.
            regime = "mixed"
            recommended_hop_expand = False
            recommended_trb_auto   = True
            recommended_anchor_bonus = None

        return QueryProfile(
            hub_score=hub_score,
            degree_cv=degree_cv,
            mean_rel_coverage=mean_rel_coverage,
            min_rel_coverage=min_rel_coverage,
            typed_relations=sorted(typed_relations),
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_relation_types=n_relation_types,
            recommended_hop_expand=recommended_hop_expand,
            recommended_trb_auto=recommended_trb_auto,
            recommended_anchor_bonus=recommended_anchor_bonus,
            regime=regime,
        )

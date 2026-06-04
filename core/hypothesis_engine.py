"""
HypothesisEngine â€” Multi-Path Abductive Reasoning (Phase 50).

Given two nodes with no direct edge, finds all multi-hop paths between them,
composes each path's relation chain, combines independent evidence using the
Noisy-OR model (equifinality), identifies intersection hub nodes
(intersectionality), and proposes a new direct edge with full derivation
provenance.

Key concepts
------------
Equifinality
    Multiple independent paths converging on the same derived relation
    compound confidence via Noisy-OR:
        P(link | P1, P2, ...) = 1 - âˆ�(1 - score_i)

Intersectionality
    Intermediate nodes shared across â‰¥2 independent paths are structural hubs
    â€” the highest-value candidates for follow-up study or new edge creation.

Domain-agnostic
    The composition table is user-configurable (pass ``composition_table``).
    Defaults to the 50+ rules from TransitiveInferenceEngine.

Usage
-----
    from adapters.networkx_adapter import NetworkXAdapter
    from core.hypothesis_engine import HypothesisEngine

    adapter = NetworkXAdapter.from_csv("hetionet.csv")
    engine  = HypothesisEngine(adapter)

    proposals = engine.generate("Compound::aspirin", "Disease::alzheimers")
    for p in proposals:
        print(p.derived_relation, p.confidence, p.derivation_text)

    engine.materialize(proposals)   # write to graph
    engine.rollback()               # undo if desired
"""
from __future__ import annotations

import time
import uuid
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Counter, Dict, List, Optional, Set, Tuple

import networkx as nx  # noqa: F401  (used indirectly via adapter.to_networkx())


# ---------------------------------------------------------------------------
# Opposing relation map â€” for contradiction detection
# ---------------------------------------------------------------------------

_OPPOSING_RELATIONS: Dict[str, str] = {
    "CAUSES":                "PREVENTS",
    "PREVENTS":              "CAUSES",
    "ACTIVATES":             "INHIBITS",
    "INHIBITS":              "ACTIVATES",
    "PROMOTES":              "SUPPRESSES",
    "SUPPRESSES":            "PROMOTES",
    "INCREASES":             "DECREASES",
    "DECREASES":             "INCREASES",
    "TREATS":                "WORSENS",
    "WORSENS":               "TREATS",
    "UPREGULATES":           "DOWNREGULATES",
    "DOWNREGULATES":         "UPREGULATES",
    "INDIRECTLY_CAUSES":     "INDIRECTLY_PREVENTS",
    "INDIRECTLY_PREVENTS":   "INDIRECTLY_CAUSES",
    "MAY_PREVENT":           "MAY_CAUSE_SIDE_EFFECT",
    "MAY_CAUSE_SIDE_EFFECT": "MAY_PREVENT",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HypothesisProposal:
    """A single proposed edge derived from multi-path abductive reasoning."""

    hypothesis_id: str
    """UUID4 for targeted materialization / rollback."""

    source: str
    """Source entity ID."""

    target: str
    """Target entity ID."""

    derived_relation: str
    """Relation type inferred by composing the supporting path chains."""

    confidence: float
    """Noisy-OR combined confidence across independent supporting paths (0â€“1)."""

    path_count: int
    """Number of independent supporting paths contributing to this proposal."""

    independence_scores: List[float]
    """Per-path Jaccard independence vs. the highest-scoring other path."""

    contradiction_score: float
    """0.0 = no counter-evidence found; approaches 1.0 as counter-paths grow stronger."""

    derivation_text: str
    """Human-readable summary: paths, composed relations, and confidence values."""

    supporting_paths: list
    """TraversalPath objects used as evidence (serialized separately for the API)."""

    intersection_nodes: List[str]
    """Intermediate nodes appearing in â‰¥2 independent paths (equifinality hubs)."""


# ---------------------------------------------------------------------------
# HypothesisEngine
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """
    Multi-path abductive reasoning engine.

    Finds all multi-hop paths between a (source, target) pair, composes each
    path's relation chain into a single derived relation using the inference
    rule index, groups convergent paths by derived relation, and returns
    proposals backed by Noisy-OR combined evidence.

    Parameters
    ----------
    adapter
        GraphAdapter with populated community_map and embeddings.
    csa_metadata
        Precomputed CSA metadata dict with keys ``distances`` and
        ``adjacent_pairs``.  If None, a minimal CSAEngine is built.
    min_confidence
        Proposals below this combined Noisy-OR confidence are discarded.
    min_path_independence
        Minimum Jaccard distance required between two paths before the second
        is counted as independent evidence.  Range [0, 1].
    composition_table
        Optional custom {(rel_a, rel_b) -> derived_relation} dict.  If None,
        the 50+ rules from TransitiveInferenceEngine are used.
    """

    def __init__(
        self,
        adapter,
        csa_metadata: Optional[dict] = None,
        min_confidence: float = 0.30,
        min_path_independence: float = 0.20,
        composition_table: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        self._adapter            = adapter
        self._csa_metadata       = csa_metadata
        self._min_confidence     = min_confidence
        self._min_path_independence = min_path_independence
        self._lock               = threading.RLock()
        self._snapshot: Optional[List[dict]] = None
        self._last_proposals: Optional[List[HypothesisProposal]] = None
        self._materialized_count: int = 0
        self._last_source: Optional[str] = None
        self._last_target: Optional[str] = None

        # Build composition rule index
        if composition_table is not None:
            self._rule_index: Dict[Tuple[str, str], str] = composition_table
        else:
            from core.inference_engine import INFERENCE_RULES, _build_rule_index
            idx = _build_rule_index(INFERENCE_RULES)
            self._rule_index = {k: r.derived for k, r in idx.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        source_id: str,
        target_id: str,
        max_paths: int = 10,
        max_hop: int = 3,
        beam_width: int = 10,
        max_budget: int = 500,
    ) -> List[HypothesisProposal]:
        """
        Find all paths from source_id to target_id and return proposals.

        Each proposal represents a derived relation backed by one or more
        independent paths.  Returns an empty list when no paths exist or
        all combined confidences fall below min_confidence.

        Parameters
        ----------
        source_id, target_id
            Entity IDs in the loaded graph.
        max_paths
            Cap on the number of target-reaching paths to consider (highest
            score first).
        max_hop
            Maximum hop depth for BeamTraversal.
        beam_width
            Beam width for traversal.
        max_budget
            Maximum neighbor expansions for traversal.
        """
        with self._lock:
            self._last_source = source_id
            self._last_target = target_id

            # 1. Run BeamTraversal from source, filter paths reaching target
            traversal = self._build_traversal(beam_width, max_hop, max_budget)
            all_paths = traversal.traverse([source_id])
            target_paths = [p for p in all_paths if p.tail == target_id]

            if not target_paths:
                self._last_proposals = []
                return []

            # Cap to max_paths (highest scoring first)
            target_paths = sorted(target_paths, key=lambda p: p.score, reverse=True)
            target_paths = target_paths[:max_paths]

            # 2. Compose relation chain for each path
            path_relations: List[Optional[str]] = [
                self._compose_chain(p) for p in target_paths
            ]

            # 3. Group paths by derived relation
            groups: Dict[str, List[Tuple[int, object]]] = {}
            for i, rel in enumerate(path_relations):
                if rel is None:
                    continue
                groups.setdefault(rel, []).append((i, target_paths[i]))

            # 4. Build a proposal for each relation group
            proposals: List[HypothesisProposal] = []
            for derived_rel, indexed_paths in groups.items():
                paths_in_group = [p for _, p in indexed_paths]

                # Select independent subset (greedy, highest score first)
                independent = self._select_independent_paths(paths_in_group)

                # Independence scores (vs. the best other path in the set)
                indep_scores: List[float] = []
                for i, p in enumerate(independent):
                    if i == 0:
                        indep_scores.append(1.0)
                    else:
                        indep_scores.append(
                            min(self._independence(p, s) for s in independent[:i])
                        )

                # Noisy-OR confidence across independent paths
                raw_confidence = self._noisy_or([p.score for p in independent])

                # Contradiction check: counter-paths reduce confidence
                c_score = self._contradiction_score(
                    derived_rel, target_paths, path_relations
                )

                final_confidence = raw_confidence * (1.0 - c_score)

                if final_confidence < self._min_confidence:
                    continue

                # Intersection hub nodes (shared intermediaries across paths)
                hubs = self._find_intersection_nodes(
                    independent, source_id, target_id
                )

                # Human-readable derivation text
                parts = []
                for i, p in enumerate(independent):
                    chain = self._chain_string(p)
                    parts.append(
                        f"P{i + 1}: {source_id}â†’{'â†’'.join(p.entity_nodes[1:-1])}â†’{target_id}"
                        f" [{chain}] score={p.score:.3f}"
                    )
                derivation = "; ".join(parts)
                if c_score > 0:
                    derivation += f"; contradiction={c_score:.3f}"

                proposals.append(
                    HypothesisProposal(
                        hypothesis_id=str(uuid.uuid4()),
                        source=source_id,
                        target=target_id,
                        derived_relation=derived_rel,
                        confidence=final_confidence,
                        path_count=len(independent),
                        independence_scores=indep_scores,
                        contradiction_score=c_score,
                        derivation_text=derivation,
                        supporting_paths=independent,
                        intersection_nodes=hubs,
                    )
                )

            proposals.sort(key=lambda p: p.confidence, reverse=True)
            self._last_proposals = proposals
            return proposals

    def materialize(self, proposals: List[HypothesisProposal]) -> int:
        """
        Add proposed edges to the graph with full provenance metadata.

        Returns the number of new edges added.  Edges that already exist
        with the same relation are skipped.  The added edges are snapshotted
        for rollback.
        """
        with self._lock:
            G = self._adapter.to_networkx()
            added = 0
            new_snapshot: List[dict] = []

            for prop in proposals:
                # Skip if relation already exists (non-multigraph)
                if not G.is_multigraph() and G.has_edge(prop.source, prop.target):
                    existing_rel = (
                        G[prop.source][prop.target].get("relation_type") or
                        G[prop.source][prop.target].get("relation", "")
                    ).upper()
                    if existing_rel == prop.derived_relation:
                        continue

                attrs = {
                    "relation_type":    prop.derived_relation,
                    "relation":         prop.derived_relation,
                    "confidence":       round(prop.confidence, 6),
                    "weight":           prop.confidence,
                    "provenance":       "hypothesis_engine",
                    "hypothesis_id":    prop.hypothesis_id,
                    "derivation":       prop.derivation_text,
                    "intersection_nodes": ",".join(prop.intersection_nodes),
                    "path_count":       prop.path_count,
                    "inferred_at":      time.time(),
                }
                G.add_edge(prop.source, prop.target, **attrs)
                new_snapshot.append(
                    {
                        "source":        prop.source,
                        "target":        prop.target,
                        "relation":      prop.derived_relation,
                        "hypothesis_id": prop.hypothesis_id,
                    }
                )
                added += 1

            if self._snapshot is None:
                self._snapshot = new_snapshot
            else:
                self._snapshot.extend(new_snapshot)

            self._materialized_count += added
            return added

    def rollback(self) -> int:
        """
        Remove all edges added by the most recent materialize() call(s).

        Returns the number of edges removed.

        Raises
        ------
        RuntimeError
            If materialize() has not been called yet.
        """
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(
                    "Nothing to roll back. Call materialize() first."
                )
            G = self._adapter.to_networkx()
            removed = 0

            for entry in self._snapshot:
                u, v = entry["source"], entry["target"]
                rel = entry["relation"]
                hid = entry["hypothesis_id"]

                if G.is_multigraph():
                    if G.has_edge(u, v):
                        keys = [
                            k for k, d in G[u][v].items()
                            if d.get("hypothesis_id") == hid
                        ]
                        for k in keys:
                            G.remove_edge(u, v, key=k)
                            removed += 1
                else:
                    if G.has_edge(u, v):
                        d = G[u][v]
                        if (d.get("relation_type") or d.get("relation", "")).upper() == rel:
                            G.remove_edge(u, v)
                            removed += 1

            self._snapshot = None
            self._materialized_count = 0
            return removed

    @property
    def can_rollback(self) -> bool:
        """True if a prior materialize() run can be undone."""
        return self._snapshot is not None

    @property
    def last_proposals(self) -> Optional[List[HypothesisProposal]]:
        """The proposals returned by the most recent generate() call."""
        return self._last_proposals

    @property
    def materialized_count(self) -> int:
        """Number of edges currently materialized (reset on rollback)."""
        return self._materialized_count

    @property
    def last_source(self) -> Optional[str]:
        return self._last_source

    @property
    def last_target(self) -> Optional[str]:
        return self._last_target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_traversal(self, beam_width: int, max_hop: int, max_budget: int):
        """Construct a minimal BeamTraversal from the adapter."""
        from core.attention_engine import CSAEngine
        from reasoning.traversal import BeamTraversal

        csa = CSAEngine(adapter=self._adapter)
        if self._csa_metadata:
            csa.set_community_graph(
                self._csa_metadata["distances"],
                self._csa_metadata["adjacent_pairs"],
            )

        return BeamTraversal(
            adapter=self._adapter,
            csa_engine=csa,
            beam_width=beam_width,
            max_hop=max_hop,
            max_budget=max_budget,
        )

    def _compose_chain(self, path) -> Optional[str]:
        """
        Compose a multi-hop relation chain into a single derived relation.

        Extracts the relation labels from ``path.nodes[1::2]``, then
        iteratively applies the composition rule index.  Falls back to
        ``"RELATED_TO"`` whenever no rule matches at a given step.
        """
        rels = path.nodes[1::2]
        if not rels:
            return None

        composed = rels[0].upper()
        for next_rel in rels[1:]:
            nr = next_rel.upper()
            result = self._rule_index.get((composed, nr))
            composed = result if result else "RELATED_TO"

        return composed

    def _independence(self, p1, p2) -> float:
        """
        Jaccard distance on intermediate node sets (head and tail excluded).

        Returns 1.0 when either path has no intermediate nodes (trivially
        independent) or the intersection is empty.
        """
        inter1: Set[str] = set(p1.entity_nodes[1:-1])
        inter2: Set[str] = set(p2.entity_nodes[1:-1])
        union = inter1 | inter2
        if not union:
            return 1.0
        return 1.0 - len(inter1 & inter2) / len(union)

    def _noisy_or(self, scores: List[float]) -> float:
        """
        Noisy-OR combination: 1 - âˆ�(1 - s_i).

        Models independent causal chains â€” each path is an independent
        ``cause'' of the proposed link.
        """
        if not scores:
            return 0.0
        product = 1.0
        for s in scores:
            product *= 1.0 - max(0.0, min(1.0, s))
        return 1.0 - product

    def _contradiction_score(
        self,
        derived_rel: str,
        all_paths: list,
        path_relations: List[Optional[str]],
    ) -> float:
        """
        Score of the strongest counter-path composing to the opposing relation.

        Returns 0.0 when no opposing paths exist.
        """
        opposing = _OPPOSING_RELATIONS.get(derived_rel)
        if not opposing:
            return 0.0
        counter_scores = [
            p.score
            for p, rel in zip(all_paths, path_relations)
            if rel == opposing
        ]
        return max(counter_scores, default=0.0)

    def _select_independent_paths(self, paths: list) -> list:
        """
        Greedy selection of independent paths.

        Start with the highest-scoring path; add the next only when its
        Jaccard independence vs. all already-selected paths exceeds
        ``min_path_independence``.
        """
        sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
        if not sorted_paths:
            return []
        selected = [sorted_paths[0]]
        for p in sorted_paths[1:]:
            if all(
                self._independence(p, s) >= self._min_path_independence
                for s in selected
            ):
                selected.append(p)
        return selected

    def _find_intersection_nodes(
        self, paths: list, source_id: str, target_id: str
    ) -> List[str]:
        """
        Identify intermediate nodes appearing in â‰¥2 independent paths.

        These are equifinality hubs â€” structural bottlenecks through which
        multiple reasoning routes converge.
        """
        if len(paths) < 2:
            return []
        counts: Counter = Counter()
        excluded = {source_id, target_id}
        for p in paths:
            intermediates = set(p.entity_nodes[1:-1]) - excluded
            for node in intermediates:
                counts[node] += 1
        return [node for node, cnt in counts.items() if cnt >= 2]

    def _chain_string(self, path) -> str:
        """Human-readable relation chain: REL1â†’REL2â†’REL3."""
        rels = path.nodes[1::2]
        return "â†’".join(rels) if rels else "?"

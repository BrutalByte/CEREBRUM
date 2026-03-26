"""
TransitiveInferenceEngine — CEREBRUM discovery layer (Phase 19).

Discovers new knowledge by composing existing graph relationships
according to domain-specific inference rules.  Every derived edge is
fully traceable: source path, intermediate node, rule applied, and
confidence are all stored on the edge and in the report.

How it works
------------
For every intermediate node M in the graph, the engine checks every
(in-edge → out-edge) pair at M.  If the two relation types match a
known inference rule, a new edge is proposed between the far endpoints.

Example
-------
    aspirin  --[INHIBITS]-->  COX2
    COX2     --[PROMOTES]-->  inflammation
    ────────────────────────────────────────
    RULE: INHIBITS + PROMOTES → INDIRECTLY_REDUCES  (factor=0.7)
    ────────────────────────────────────────
    PROPOSED: aspirin --[INDIRECTLY_REDUCES]--> inflammation
              confidence = conf(aspirin→COX2) × conf(COX2→inflammation) × 0.7
              provenance = "transitive_inference"
              via = "COX2"
              derivation = "aspirin-[INHIBITS]->COX2-[PROMOTES]->inflammation"

Usage
-----
    from adapters.networkx_adapter import NetworkXAdapter
    from core.inference_engine import TransitiveInferenceEngine

    adapter = NetworkXAdapter.from_csv("...")
    engine  = TransitiveInferenceEngine(adapter)

    report  = engine.run(dry_run=True)   # inspect without mutating
    print(report.proposal_count, "discoveries found")
    for p in report.proposals[:5]:
        print(p.derivation_str)

    report  = engine.run()               # apply to graph
    engine.rollback()                    # undo if needed
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Inference rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceRule:
    """A single transitive composition rule."""

    rel_a: str
    """First relation in the chain (u → mid)."""

    rel_b: str
    """Second relation in the chain (mid → v)."""

    derived: str
    """Relation type inferred for the new edge (u → v)."""

    factor: float
    """Confidence multiplier applied to the product of edge confidences (0–1)."""

    domain: str
    """Rule domain tag: 'general' | 'biology' | 'academic' | 'social' | 'causal' | 'film'."""

    note: str
    """Human-readable justification shown in derivation strings."""


# ---------------------------------------------------------------------------
# Comprehensive rule table — 50+ rules across six domains
# ---------------------------------------------------------------------------

INFERENCE_RULES: List[InferenceRule] = [

    # ── Causal chains ────────────────────────────────────────────────────
    InferenceRule("CAUSES",        "CAUSES",        "INDIRECTLY_CAUSES",        0.80, "causal",   "A causes B causes C → A indirectly causes C"),
    InferenceRule("CAUSES",        "TREATS",        "MAY_TREAT",                0.55, "causal",   "A causes B which treats C → A may treat C via that pathway"),
    InferenceRule("CAUSES",        "PREVENTS",      "MAY_PREVENT",              0.55, "causal",   "A causes B which prevents C → A may prevent C"),
    InferenceRule("PREVENTS",      "CAUSES",        "INDIRECTLY_REDUCES_RISK",  0.60, "causal",   "A prevents B which causes C → A may reduce risk of C"),
    InferenceRule("ACTIVATES",     "CAUSES",        "INDIRECTLY_CAUSES",        0.70, "causal",   "A activates B which causes C → A indirectly causes C"),
    InferenceRule("ACTIVATES",     "PREVENTS",      "INDIRECTLY_PREVENTS",      0.65, "causal",   "Activation leading to prevention"),
    InferenceRule("INHIBITS",      "CAUSES",        "MAY_PREVENT",              0.65, "causal",   "A inhibits B which causes C → A may prevent C"),
    InferenceRule("INHIBITS",      "PREVENTS",      "INDIRECTLY_CAUSES",        0.55, "causal",   "A inhibits B which prevents C → A may indirectly cause C"),

    # ── Biology / medicine ───────────────────────────────────────────────
    InferenceRule("INHIBITS",      "PROMOTES",      "INDIRECTLY_REDUCES",       0.70, "biology",  "A inhibits B which promotes C → A reduces C"),
    InferenceRule("PROMOTES",      "INHIBITS",      "INDIRECTLY_REDUCES",       0.70, "biology",  "A promotes B which inhibits C → A reduces C"),
    InferenceRule("ACTIVATES",     "ACTIVATES",     "INDIRECTLY_ACTIVATES",     0.70, "biology",  "Activation cascade"),
    InferenceRule("INHIBITS",      "INHIBITS",      "INDIRECTLY_ACTIVATES",     0.60, "biology",  "Double inhibition → net activation"),
    InferenceRule("PROMOTES",      "PROMOTES",      "INDIRECTLY_PROMOTES",      0.70, "biology",  "Promotion cascade"),
    InferenceRule("UPREGULATES",   "UPREGULATES",   "INDIRECTLY_UPREGULATES",   0.70, "biology",  "Upregulation cascade"),
    InferenceRule("DOWNREGULATES", "DOWNREGULATES", "INDIRECTLY_DOWNREGULATES", 0.70, "biology",  "Downregulation cascade"),
    InferenceRule("UPREGULATES",   "DOWNREGULATES", "INDIRECTLY_DOWNREGULATES", 0.60, "biology",  "Mixed regulation → net down"),
    InferenceRule("DOWNREGULATES", "UPREGULATES",   "INDIRECTLY_DOWNREGULATES", 0.60, "biology",  "Mixed regulation → net down"),
    InferenceRule("ENCODES",       "BINDS",         "GENE_PRODUCT_BINDS",       0.80, "biology",  "Gene encodes protein that binds target"),
    InferenceRule("ENCODES",       "INHIBITS",      "GENE_INHIBITS",            0.80, "biology",  "Gene encodes protein that inhibits target"),
    InferenceRule("ENCODES",       "ACTIVATES",     "GENE_ACTIVATES",           0.80, "biology",  "Gene encodes protein that activates target"),
    InferenceRule("ENCODES",       "PROMOTES",      "GENE_PROMOTES",            0.80, "biology",  "Gene encodes protein that promotes target"),
    InferenceRule("BINDS",         "INHIBITS",      "BLOCKS_VIA_BINDING",       0.75, "biology",  "A binds B which inhibits C → A blocks C via B"),
    InferenceRule("TREATS",        "CAUSES",        "MAY_CAUSE_SIDE_EFFECT",    0.45, "biology",  "Drug treats A which causes B → drug may cause B as side effect"),
    InferenceRule("EXPRESSED_IN",  "CAUSES",        "TISSUE_IMPLICATED_IN",     0.65, "biology",  "Gene expressed in tissue which causes disease → tissue implicated"),
    InferenceRule("INTERACTS_WITH","CAUSES",        "IMPLICATED_IN",            0.60, "biology",  "Interaction chain leading to disease"),
    InferenceRule("INTERACTS_WITH","INHIBITS",      "INDIRECT_INHIBITION",      0.60, "biology",  "Interaction leading to inhibition"),

    # ── Academic / intellectual ──────────────────────────────────────────
    InferenceRule("INFLUENCED",    "INFLUENCED",    "INDIRECTLY_INFLUENCED",    0.80, "academic", "A influenced B who influenced C → A indirectly influenced C"),
    InferenceRule("INFLUENCED",    "INVENTED",      "INSPIRED_INVENTION_OF",    0.70, "academic", "Influence chain leading to invention"),
    InferenceRule("INFLUENCED",    "DISCOVERED",    "INSPIRED_DISCOVERY_OF",    0.70, "academic", "Influence chain leading to discovery"),
    InferenceRule("INFLUENCED",    "PROVED",        "INSPIRED_PROOF_OF",        0.70, "academic", "Influence leading to mathematical proof"),
    InferenceRule("STUDENT_OF",    "STUDENT_OF",    "ACADEMIC_DESCENDANT_OF",   0.90, "academic", "Academic lineage — grandstudent relationship"),
    InferenceRule("STUDENT_OF",    "INVENTED",      "TRAINED_IN_TRADITION_OF",  0.70, "academic", "Student of inventor inherits intellectual tradition"),
    InferenceRule("STUDENT_OF",    "DISCOVERED",    "TRAINED_IN_TRADITION_OF",  0.70, "academic", "Student of discoverer inherits intellectual tradition"),
    InferenceRule("STUDENT_OF",    "INFLUENCED",    "INDIRECTLY_INFLUENCED_BY", 0.75, "academic", "Advisor's influences become student's indirect influences"),
    InferenceRule("MENTORED",      "MENTORED",      "ACADEMIC_GRANDCHILD_OF",   0.90, "academic", "Mentorship lineage"),
    InferenceRule("MENTORED",      "INVENTED",      "MENTORSHIP_LED_TO",        0.70, "academic", "Mentor's student invented something"),
    InferenceRule("AUTHORED",      "CITED",         "WORK_CITED_BY",            0.85, "academic", "A authored B which was cited by C"),
    InferenceRule("CITED",         "INFLUENCED",    "ACADEMIC_INFLUENCE_VIA",   0.70, "academic", "Citation leading to broader influence"),
    InferenceRule("STUDIED",       "DISCOVERED",    "CONTRIBUTED_TO",           0.70, "academic", "Study of a subject led to discovery"),
    InferenceRule("WROTE",         "INFLUENCED",    "INTELLECTUAL_INFLUENCE_ON", 0.75, "academic", "Writing that influenced others"),
    InferenceRule("INVENTED",      "INFLUENCED",    "INVENTION_INFLUENCED",     0.75, "academic", "Invention that influenced others"),

    # ── Social / organizational ──────────────────────────────────────────
    InferenceRule("KNOWS",         "KNOWS",         "CONNECTED_VIA",            0.55, "social",   "Friend-of-friend connection"),
    InferenceRule("WORKED_WITH",   "FOUNDED",       "INVOLVED_IN_FOUNDING",     0.70, "social",   "Collaborator of founder involved in founding"),
    InferenceRule("WORKED_WITH",   "INVENTED",      "CONTRIBUTED_TO",           0.70, "social",   "Collaborator on invention"),
    InferenceRule("WORKED_WITH",   "DISCOVERED",    "CONTRIBUTED_TO",           0.70, "social",   "Collaborator on discovery"),
    InferenceRule("EMPLOYED_BY",   "FOUNDED",       "ASSOCIATED_WITH_FOUNDING", 0.65, "social",   "Employee of founder associated with founding"),
    InferenceRule("MEMBER_OF",     "FOUNDED",       "LINKED_TO_FOUNDING",       0.65, "social",   "Member of organization that was founded"),
    InferenceRule("COLLABORATED_WITH", "INVENTED",  "CONTRIBUTED_TO",           0.70, "social",   "Collaborator on invention"),
    InferenceRule("COLLABORATED_WITH", "DISCOVERED","CONTRIBUTED_TO",           0.70, "social",   "Collaborator on discovery"),
    InferenceRule("COLLABORATED_WITH", "COLLABORATED_WITH", "NETWORK_PEER",     0.60, "social",   "Shared collaboration network"),
    InferenceRule("OPPOSED",       "SUPPORTED",     "INDIRECT_CONFLICT",        0.60, "social",   "A opposed B which supported C → A indirectly conflicted with C"),

    # ── Structural / spatial ─────────────────────────────────────────────
    InferenceRule("PART_OF",       "PART_OF",       "PART_OF",                  0.90, "general",  "Transitivity of part-of"),
    InferenceRule("LOCATED_IN",    "LOCATED_IN",    "LOCATED_IN",               0.90, "general",  "Transitivity of location"),
    InferenceRule("PART_OF",       "LOCATED_IN",    "LOCATED_IN",               0.85, "general",  "Part inherits container's location"),
    InferenceRule("MEMBER_OF",     "MEMBER_OF",     "TRANSITIVELY_MEMBER_OF",   0.80, "general",  "Transitivity of membership"),
    InferenceRule("PRECEDES",      "PRECEDES",      "LONG_BEFORE",              0.90, "general",  "Temporal chain — A precedes B precedes C → A long before C"),
    InferenceRule("FOLLOWS",       "FOLLOWS",       "LONG_AFTER",               0.90, "general",  "Temporal chain — A follows B follows C → A long after C"),
    InferenceRule("PRECEDES",      "CAUSED",        "HISTORICALLY_LED_TO",      0.70, "general",  "A precedes B which caused C → A historically led to C"),

    # ── Film / entertainment ─────────────────────────────────────────────
    InferenceRule("STARRED_IN",    "DIRECTED",      "WORKED_UNDER",             0.80, "film",     "Actor starred in film directed by director → worked under that director"),
    InferenceRule("STARRED_IN",    "HAS_GENRE",     "KNOWN_FOR_GENRE",          0.70, "film",     "Actor in film of genre → known for that genre"),
    InferenceRule("DIRECTED",      "HAS_GENRE",     "KNOWN_FOR_GENRE",          0.80, "film",     "Director of film with genre → known for that genre"),
    InferenceRule("DIRECTED",      "STARRED_IN",    "REGULAR_COLLABORATOR",     0.75, "film",     "Director's film starred actor → regular collaborator"),
    InferenceRule("PRODUCED_BY",   "DIRECTED",      "PRODUCER_DIRECTOR_PAIR",   0.75, "film",     "Film produced by A and directed by B → A-B are a producing pair"),
    InferenceRule("PRODUCED_BY",   "STARRED_IN",    "PRODUCER_ACTOR_PAIR",      0.70, "film",     "Film produced by A and starred B → A-B working pair"),
]


def _build_rule_index(rules: List[InferenceRule]) -> Dict[Tuple[str, str], InferenceRule]:
    """Build a fast (rel_a, rel_b) → rule lookup dict."""
    return {(r.rel_a, r.rel_b): r for r in rules}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InferenceProposal:
    """A single discovered edge proposed by transitive composition."""

    source: str
    """Entity at the start of the derived edge."""

    via: str
    """Intermediate entity that bridges the two input edges."""

    target: str
    """Entity at the end of the derived edge."""

    derived_relation: str
    """Relation type of the proposed new edge."""

    confidence: float
    """conf_a × conf_b × rule.factor — weakest-link propagation."""

    rule: InferenceRule
    """The inference rule that produced this proposal."""

    confidence_a: float
    """Confidence of the edge (source → via)."""

    confidence_b: float
    """Confidence of the edge (via → target)."""

    @property
    def derivation_str(self) -> str:
        """Human-readable derivation chain."""
        return (
            f"{self.source}-[{self.rule.rel_a}]->{self.via}"
            f"-[{self.rule.rel_b}]->{self.target}"
            f"  =>  {self.derived_relation}"
            f"  (conf={self.confidence:.3f}, rule: {self.rule.note})"
        )


@dataclass
class InferenceReport:
    """Summary of one TransitiveInferenceEngine run."""

    proposals: List[InferenceProposal]
    """All proposed new edges, sorted by confidence descending."""

    materialized: int
    """Edges actually added to the graph (0 when dry_run=True)."""

    rules_applied: Dict[str, int]
    """Count of proposals per derived relation type."""

    skipped_existing: int
    """Edges skipped because the relation already existed."""

    duration_seconds: float
    dry_run: bool
    timestamp: float = field(default_factory=time.time)

    @property
    def proposal_count(self) -> int:
        return len(self.proposals)

    def summary(self) -> str:
        lines = [
            f"InferenceReport ({'dry-run' if self.dry_run else 'applied'})",
            f"  Proposals    : {self.proposal_count}",
            f"  Materialized : {self.materialized}",
            f"  Skipped      : {self.skipped_existing} (edge already existed)",
            f"  Duration     : {self.duration_seconds:.3f}s",
            f"  Top derived relations:",
        ]
        for rel, count in sorted(self.rules_applied.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {rel:40s}  {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TransitiveInferenceEngine
# ---------------------------------------------------------------------------

class TransitiveInferenceEngine:
    """
    Discovers new knowledge by composing existing graph relations.

    Parameters
    ----------
    adapter : GraphAdapter
        The loaded knowledge graph.  Must expose ``to_networkx()``.
    max_proposals : int
        Maximum number of proposed edges per run.  Prevents combinatorial
        explosion on dense graphs.  Proposals are sorted by confidence
        before the cap is applied, so the highest-confidence discoveries
        are always included.
    min_confidence : float
        Proposals below this confidence are discarded.
    enabled_domains : set of str, optional
        Restrict to rules whose domain is in this set.  ``'general'``
        rules are always included.  Pass ``None`` to enable all domains.
    custom_rules : list of InferenceRule, optional
        Additional rules merged with the built-in table.
    """

    def __init__(
        self,
        adapter,
        max_proposals: int = 200,
        min_confidence: float = 0.10,
        enabled_domains: Optional[Set[str]] = None,
        custom_rules: Optional[List[InferenceRule]] = None,
    ) -> None:
        self._adapter       = adapter
        self._max_proposals = max_proposals
        self._min_confidence = min_confidence
        self._lock          = threading.RLock()
        self._last_report: Optional[InferenceReport] = None
        self._snapshot: Optional[List[dict]] = None   # for rollback

        # Build active rule index
        rules = list(INFERENCE_RULES)
        if custom_rules:
            rules.extend(custom_rules)
        if enabled_domains is not None:
            rules = [r for r in rules
                     if r.domain == "general" or r.domain in enabled_domains]
        self._rule_index = _build_rule_index(rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> InferenceReport:
        """
        Execute one inference cycle.

        Parameters
        ----------
        dry_run : bool
            If True, discover but do not mutate the graph.
            The returned report shows exactly what a real run would do.

        Returns
        -------
        InferenceReport
        """
        with self._lock:
            t0 = time.time()
            G  = self._adapter.to_networkx()

            proposals, skipped = self._discover(G)

            materialized = 0
            if not dry_run:
                # Snapshot for rollback before mutating
                self._snapshot = [
                    {
                        "source": p.source,
                        "target": p.target,
                        "relation": p.derived_relation,
                    }
                    for p in proposals
                ]
                materialized = self._materialize(G, proposals)

            rules_applied: Dict[str, int] = {}
            for p in proposals:
                rules_applied[p.derived_relation] = rules_applied.get(p.derived_relation, 0) + 1

            report = InferenceReport(
                proposals=proposals,
                materialized=materialized,
                rules_applied=rules_applied,
                skipped_existing=skipped,
                duration_seconds=time.time() - t0,
                dry_run=dry_run,
            )
            self._last_report = report
            return report

    def rollback(self) -> int:
        """
        Remove all edges added by the most recent non-dry-run cycle.

        Returns
        -------
        int
            Number of edges removed.

        Raises
        ------
        RuntimeError
            If no prior non-dry-run cycle exists to roll back.
        """
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(
                    "No prior inference run to roll back. "
                    "Call run(dry_run=False) first."
                )
            G = self._adapter.to_networkx()
            removed = 0
            for entry in self._snapshot:
                u, v, rel = entry["source"], entry["target"], entry["relation"]
                if G.has_edge(u, v):
                    if G.is_multigraph():
                        keys_to_remove = [
                            k for k, d in G[u][v].items()
                            if (d.get("relation_type") or d.get("relation", "")).upper() == rel
                        ]
                        for k in keys_to_remove:
                            G.remove_edge(u, v, key=k)
                            removed += 1
                    else:
                        d = G[u][v]
                        if (d.get("relation_type") or d.get("relation", "")).upper() == rel:
                            G.remove_edge(u, v)
                            removed += 1
            self._snapshot = None
            return removed

    @property
    def last_report(self) -> Optional[InferenceReport]:
        """The most recent InferenceReport, or None before the first run."""
        return self._last_report

    @property
    def can_rollback(self) -> bool:
        """True if the last run was non-dry and can be undone."""
        return self._snapshot is not None

    def rule_count(self) -> int:
        """Number of active inference rules."""
        return len(self._rule_index)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover(
        self,
        G: nx.Graph,
    ) -> Tuple[List[InferenceProposal], int]:
        """
        Enumerate composable (in-edge, out-edge) pairs at every node.

        Returns (proposals sorted by confidence desc, skipped_existing_count).
        """
        proposals: List[InferenceProposal] = []
        skipped = 0
        seen: Set[Tuple[str, str, str]] = set()  # (source, target, derived_rel)
        is_directed = G.is_directed()
        is_multi    = G.is_multigraph()

        for mid in G.nodes():
            # Collect in-edges and out-edges at this intermediate node
            if is_directed:
                in_edges  = list(G.in_edges(mid, data=True))   # (u, mid, data)
                out_edges = list(G.out_edges(mid, data=True))  # (mid, v, data)
            else:
                # Undirected: every neighbour can be either side
                neighbours = [(mid, nbr, data) for nbr, data in G[mid].items()]
                in_edges  = neighbours
                out_edges = neighbours

            for u, _m1, data_a in in_edges:
                if u == mid:
                    continue
                rel_a  = _rel(data_a)
                if not rel_a:
                    continue
                conf_a = data_a.get("confidence", 1.0)

                for _m2, v, data_b in out_edges:
                    if v == mid or v == u:
                        continue
                    rel_b  = _rel(data_b)
                    if not rel_b:
                        continue

                    rule = self._rule_index.get((rel_a, rel_b))
                    if rule is None:
                        continue

                    conf_b = data_b.get("confidence", 1.0)
                    derived_conf = conf_a * conf_b * rule.factor
                    if derived_conf < self._min_confidence:
                        continue

                    key = (u, v, rule.derived)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Skip if this relation already exists on the graph
                    if _edge_relation_exists(G, u, v, rule.derived, is_multi):
                        skipped += 1
                        continue

                    proposals.append(InferenceProposal(
                        source=u,
                        via=mid,
                        target=v,
                        derived_relation=rule.derived,
                        confidence=derived_conf,
                        rule=rule,
                        confidence_a=conf_a,
                        confidence_b=conf_b,
                    ))

        # Sort by confidence descending, then cap
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals[:self._max_proposals], skipped

    def _materialize(
        self,
        G: nx.Graph,
        proposals: List[InferenceProposal],
    ) -> int:
        """Add proposed edges to the graph with full provenance metadata."""
        added = 0
        for p in proposals:
            attrs = {
                "relation_type":  p.derived_relation,
                "relation":       p.derived_relation,
                "confidence":     round(p.confidence, 6),
                "weight":         p.confidence,
                "provenance":     "transitive_inference",
                "via":            p.via,
                "derivation":     (
                    f"{p.source}-[{p.rule.rel_a}]->{p.via}"
                    f"-[{p.rule.rel_b}]->{p.target}"
                ),
                "rule_domain":    p.rule.domain,
                "rule_note":      p.rule.note,
                "inferred_at":    time.time(),
            }
            if G.is_multigraph():
                G.add_edge(p.source, p.target, **attrs)
            else:
                G.add_edge(p.source, p.target, **attrs)
            added += 1
        return added


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(data: dict) -> str:
    """Extract and normalise relation type from edge data dict."""
    return (data.get("relation_type") or data.get("relation", "")).upper()


def _edge_relation_exists(
    G: nx.Graph,
    u: str,
    v: str,
    rel: str,
    is_multi: bool,
) -> bool:
    """Return True if edge u→v already carries relation type ``rel``."""
    if not G.has_edge(u, v):
        return False
    if is_multi:
        return any(
            _rel(d) == rel
            for d in G[u][v].values()
        )
    return _rel(G[u][v]) == rel

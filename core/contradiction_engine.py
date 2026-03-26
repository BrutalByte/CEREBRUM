"""
Contradiction detection and materialization for CEREBRUM.

Contradictions in a knowledge graph are research signals, not errors.
A lone contradicting edge may represent an emerging discovery, an unsettled
scientific debate, or a claim that has been superseded. Surfacing it allows
researchers to investigate whether it can be disproven or whether it opens an
entirely new reasoning path.

Design principle: **surface contradictions, never suppress them**.

Five contradiction types:
  Type 1 — Direct         : same (subject, object), contradictory predicates
  Type 2 — Cross-path     : contradiction only visible when multi-hop paths combine
  Type 3 — Temporal       : both claims true but at different time periods
  Type 4 — Provenance     : different sources disagree on the same fact
  Type 5 — Semantic       : circular causation / logical impossibility

Detection moments:
  Ingest/index time — Types 1, 3, 4 (ContradictionEngine.scan + materialize)
  Query time        — Type 2 (ContradictionEngine.detect_cross_path, called from extractor)

CONTRADICTS edges are first-class graph citizens with structured metadata
(detected_at, resolution_status, authority_delta, human_reviewed) so that
contradictions are queryable and auditable without separate bookkeeping.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRADICTS_RELATION = "CONTRADICTS"

# Known contradictory relation pairs.
# Checked at ingest time via the rule engine.
# Keys are uppercase; lookup is case-insensitive (see relations_contradict()).
CONTRADICTION_PAIRS: Dict[str, Set[str]] = {
    # Biomedical causal
    "ACTIVATES":       {"INHIBITS", "SUPPRESSES", "BLOCKS", "DOWNREGULATES"},
    "INHIBITS":        {"ACTIVATES", "PROMOTES", "STIMULATES", "UPREGULATES"},
    "CAUSES":          {"PREVENTS", "TREATS", "CURES"},
    "PREVENTS":        {"CAUSES", "INDUCES", "PROMOTES"},
    "TREATS":          {"WORSENS", "EXACERBATES", "CAUSES"},
    "WORSENS":         {"TREATS", "IMPROVES", "CURES"},
    "IMPROVES":        {"WORSENS", "EXACERBATES"},
    "UPREGULATES":     {"DOWNREGULATES"},
    "DOWNREGULATES":   {"UPREGULATES"},
    "INCREASES":       {"DECREASES", "REDUCES"},
    "DECREASES":       {"INCREASES", "AMPLIFIES"},
    "AMPLIFIES":       {"DECREASES", "REDUCES"},
    "PROMOTES":        {"INHIBITS", "SUPPRESSES", "PREVENTS"},
    "SUPPRESSES":      {"ACTIVATES", "PROMOTES", "STIMULATES"},
    "STIMULATES":      {"INHIBITS", "SUPPRESSES", "BLOCKS"},
    "BLOCKS":          {"ACTIVATES", "STIMULATES"},
    "INDUCES":         {"PREVENTS"},
    "CURES":           {"CAUSES", "WORSENS"},
    # Epistemic / logical
    "SUPPORTS":        {"REFUTES", "CONTRADICTS", "DISPROVES"},
    "REFUTES":         {"SUPPORTS", "CONFIRMS", "PROVES"},
    "CONFIRMS":        {"REFUTES", "DISPROVES"},
    "PROVES":          {"DISPROVES"},
    "DISPROVES":       {"PROVES", "CONFIRMS"},
    "CONTRADICTS":     {"SUPPORTS"},
    # Identity
    "IS_A":            {"IS_NOT_A"},
    "IS_NOT_A":        {"IS_A"},
    # Temporal ordering
    "PRECEDES":        {"FOLLOWS"},
    "FOLLOWS":         {"PRECEDES"},
    "BEFORE":          {"AFTER"},
    "AFTER":           {"BEFORE"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_at(edge, query_time: Optional[float]) -> bool:
    """
    Return True if the edge is temporally active at query_time.

    - query_time is None  → no temporal filter → always True
    - edge has no valid_from/valid_to → always valid
    - query_time < valid_from → not yet valid → False
    - query_time > valid_to   → expired       → False
    """
    if query_time is None:
        return True
    vf = getattr(edge, "valid_from", None)
    vt = getattr(edge, "valid_to", None)
    if vf is not None and query_time < vf:
        return False
    if vt is not None and query_time > vt:
        return False
    return True


def relations_contradict(rel_a: str, rel_b: str) -> bool:
    """
    Return True if rel_a and rel_b are known contradictory relation types.

    Checks CONTRADICTION_PAIRS in both directions (bidirectional by construction).
    Comparison is case-insensitive.
    """
    a_up = rel_a.upper()
    b_up = rel_b.upper()
    return (
        b_up in CONTRADICTION_PAIRS.get(a_up, set())
        or a_up in CONTRADICTION_PAIRS.get(b_up, set())
    )


# ---------------------------------------------------------------------------
# ContradictionRecord
# ---------------------------------------------------------------------------

@dataclass
class ContradictionRecord:
    """
    A detected contradiction between two claims in the knowledge graph.

    resolution_status values:
      "unresolved"   — contradiction exists; cause unknown
      "temporal"     — both claims correct but at different time periods
      "source_bias"  — claims differ by source methodology or credibility
      "resolved"     — human review determined which claim is correct
    """

    node_a: str           # Subject (or left-hand) entity
    node_b: str           # Object (or right-hand) entity
    relation_a: str       # First claim's relation type
    relation_b: str       # Contradicting relation type
    contradiction_type: str  # "direct" | "cross_path" | "temporal" | "provenance" | "semantic"

    provenance_a: str = ""
    provenance_b: str = ""
    confidence_a: float = 1.0
    confidence_b: float = 1.0

    detected_at: float = field(default_factory=time.time)
    resolution_status: str = "unresolved"
    authority_delta: float = 0.0   # confidence_a - confidence_b (auto-set in __post_init__)
    human_reviewed: bool = False
    note: str = ""

    def __post_init__(self):
        self.authority_delta = round(self.confidence_a - self.confidence_b, 6)


# ---------------------------------------------------------------------------
# ContradictionEngine
# ---------------------------------------------------------------------------

class ContradictionEngine:
    """
    Detects and materializes contradictions in a knowledge graph.

    Index-time usage (run once after graph load):
        engine = ContradictionEngine()
        records = engine.scan(G)
        n_added = engine.materialize(G, records)

    Query-time usage (called from answer extractor):
        records = engine.detect_cross_path(paths_a, paths_b)
    """

    # ------------------------------------------------------------------
    # Type 1 — Direct contradictions
    # ------------------------------------------------------------------

    def detect_direct(
        self,
        G: nx.Graph,
        community_map: Optional[Dict[str, int]] = None,
    ) -> List[ContradictionRecord]:
        """
        Type 1: same (subject, object) pair connected by contradictory predicates.

        For directed graphs: checks u→v (rel_a) and v→u (rel_b) pairs.
        For multigraphs: also checks parallel edges on the same (u, v).

        Complexity: O(|E| · avg_degree) — suitable for index-time use.
        """
        records: List[ContradictionRecord] = []
        seen: Set[Tuple[str, str, str, str]] = set()  # deduplicate symmetric pairs

        for u in G.nodes():
            out_neighbors = (
                list(G.successors(u)) if G.is_directed() else list(G.neighbors(u))
            )

            for v in out_neighbors:
                # --- Edges u→v ---
                if G.is_multigraph():
                    uv_edges = [G[u][v][k] for k in G[u][v]]
                else:
                    ed = G.get_edge_data(u, v)
                    uv_edges = [ed] if ed else []

                for ed_a in uv_edges:
                    rel_a = ed_a.get("relation", "RELATED_TO")
                    if rel_a == CONTRADICTS_RELATION:
                        continue
                    conf_a = float(ed_a.get("confidence", 1.0))
                    prov_a = ed_a.get("provenance", "")

                    # Check reverse edge v→u (directed) for contradiction
                    if G.is_directed() and G.has_edge(v, u):
                        if G.is_multigraph():
                            vu_edges = [G[v][u][k] for k in G[v][u]]
                        else:
                            rev = G.get_edge_data(v, u)
                            vu_edges = [rev] if rev else []

                        for ed_b in vu_edges:
                            rel_b = ed_b.get("relation", "RELATED_TO")
                            if rel_b == CONTRADICTS_RELATION:
                                continue
                            if relations_contradict(rel_a, rel_b):
                                key = tuple(sorted([(u, rel_a), (v, rel_b)]))
                                sig = (key[0][0], key[0][1], key[1][0], key[1][1])
                                if sig not in seen:
                                    seen.add(sig)
                                    records.append(ContradictionRecord(
                                        node_a=u,
                                        node_b=v,
                                        relation_a=rel_a,
                                        relation_b=rel_b,
                                        contradiction_type="direct",
                                        provenance_a=prov_a,
                                        provenance_b=ed_b.get("provenance", ""),
                                        confidence_a=conf_a,
                                        confidence_b=float(ed_b.get("confidence", 1.0)),
                                    ))

                    # Check parallel edges on the same (u, v) in multigraphs
                    if G.is_multigraph():
                        for ed_b in uv_edges:
                            rel_b = ed_b.get("relation", "RELATED_TO")
                            if rel_b == rel_a or rel_b == CONTRADICTS_RELATION:
                                continue
                            if relations_contradict(rel_a, rel_b):
                                sig = tuple(sorted([rel_a, rel_b]) + sorted([u, v]))
                                sig_key = (sig[0], sig[1], sig[2], sig[3])
                                if sig_key not in seen:
                                    seen.add(sig_key)
                                    records.append(ContradictionRecord(
                                        node_a=u,
                                        node_b=v,
                                        relation_a=rel_a,
                                        relation_b=rel_b,
                                        contradiction_type="direct",
                                        provenance_a=prov_a,
                                        provenance_b=ed_b.get("provenance", ""),
                                        confidence_a=conf_a,
                                        confidence_b=float(ed_b.get("confidence", 1.0)),
                                    ))

        return records

    # ------------------------------------------------------------------
    # Type 3 — Temporal contradictions
    # ------------------------------------------------------------------

    def detect_temporal(
        self,
        G: nx.Graph,
    ) -> List[ContradictionRecord]:
        """
        Type 3: two edges with contradicting relations that have non-overlapping
        valid_from/valid_to windows. Both claims are correct — just at different
        times. Emitted with resolution_status="temporal".

        Only edges that carry at least one temporal bound are considered.
        """
        records: List[ContradictionRecord] = []
        seen: Set[Tuple] = set()

        edges = list(G.edges(data=True))
        for i, (u, v, da) in enumerate(edges):
            rel_a = da.get("relation", "RELATED_TO")
            if rel_a == CONTRADICTS_RELATION:
                continue
            vfa = da.get("valid_from")
            vta = da.get("valid_to")
            if vfa is None and vta is None:
                continue  # no temporal data — skip

            conf_a = float(da.get("confidence", 1.0))
            prov_a = da.get("provenance", "")

            # Compare against all subsequent edges from the same source u
            for j, (u2, w, db) in enumerate(edges):
                if j <= i or u2 != u or w == v:
                    continue
                rel_b = db.get("relation", "RELATED_TO")
                if rel_b == CONTRADICTS_RELATION:
                    continue
                if not relations_contradict(rel_a, rel_b):
                    continue

                vfb = db.get("valid_from")
                vtb = db.get("valid_to")
                if vfb is None and vtb is None:
                    continue

                # Compute overlap
                start_a = vfa if vfa is not None else 0.0
                end_a   = vta if vta is not None else float("inf")
                start_b = vfb if vfb is not None else 0.0
                end_b   = vtb if vtb is not None else float("inf")
                no_overlap = end_a < start_b or end_b < start_a

                sig = (u, min(v, w), max(v, w), rel_a, rel_b)
                if sig in seen:
                    continue
                seen.add(sig)

                records.append(ContradictionRecord(
                    node_a=u,
                    node_b=w,
                    relation_a=rel_a,
                    relation_b=rel_b,
                    contradiction_type="temporal",
                    provenance_a=prov_a,
                    provenance_b=db.get("provenance", ""),
                    confidence_a=conf_a,
                    confidence_b=float(db.get("confidence", 1.0)),
                    resolution_status="temporal" if no_overlap else "unresolved",
                    note=f"Time windows: [{vfa},{vta}] vs [{vfb},{vtb}]",
                ))

        return records

    # ------------------------------------------------------------------
    # Type 4 — Provenance contradictions (multigraph only)
    # ------------------------------------------------------------------

    def detect_provenance(
        self,
        G: nx.Graph,
    ) -> List[ContradictionRecord]:
        """
        Type 4: multiple sources assert the same (u, v, relation) fact but with
        meaningfully different confidence values (spread > 0.1), indicating active
        scientific or factual disagreement.

        Only applies to multigraphs where parallel edges can carry different
        provenances for the same relation.
        """
        if not G.is_multigraph():
            return []

        records: List[ContradictionRecord] = []
        seen: Set[Tuple] = set()

        for u in G.nodes():
            neighbors = (
                list(G.successors(u)) if G.is_directed() else list(G.neighbors(u))
            )
            for v in neighbors:
                if not G.has_edge(u, v):
                    continue
                edges_uv = [G[u][v][k] for k in G[u][v]]

                # Group by relation
                by_rel: Dict[str, List[dict]] = {}
                for ed in edges_uv:
                    rel = ed.get("relation", "RELATED_TO")
                    if rel != CONTRADICTS_RELATION:
                        by_rel.setdefault(rel, []).append(ed)

                for rel, eds in by_rel.items():
                    if len(eds) < 2:
                        continue
                    provenances = [ed.get("provenance", "") for ed in eds]
                    unique_provs = set(provenances)
                    if len(unique_provs) < 2:
                        continue  # same source — not a provenance conflict

                    conf_values = [float(ed.get("confidence", 1.0)) for ed in eds]
                    conf_spread = max(conf_values) - min(conf_values)
                    if conf_spread <= 0.1:
                        continue  # sources agree closely enough

                    sig = (min(u, v), max(u, v), rel)
                    if sig in seen:
                        continue
                    seen.add(sig)

                    best_idx  = conf_values.index(max(conf_values))
                    worst_idx = conf_values.index(min(conf_values))
                    records.append(ContradictionRecord(
                        node_a=u,
                        node_b=v,
                        relation_a=rel,
                        relation_b=rel,
                        contradiction_type="provenance",
                        provenance_a=provenances[best_idx],
                        provenance_b=provenances[worst_idx],
                        confidence_a=conf_values[best_idx],
                        confidence_b=conf_values[worst_idx],
                        resolution_status="source_bias",
                        note=f"{len(unique_provs)} sources disagree on {rel!r} "
                             f"(confidence spread {conf_spread:.2f})",
                    ))

        return records

    # ------------------------------------------------------------------
    # Type 2 — Cross-path contradictions (query time)
    # ------------------------------------------------------------------

    def detect_cross_path(
        self,
        paths_a: List,
        paths_b: List,
    ) -> List[ContradictionRecord]:
        """
        Type 2: contradiction only visible when two multi-hop paths are combined.

        Compares each path in paths_a against each path in paths_b. For paths
        that share the same terminal entity but travel through contradicting
        intermediate relations, emits a cross_path record.

        Parameters
        ----------
        paths_a, paths_b : lists of TraversalPath objects
        """
        records: List[ContradictionRecord] = []
        seen: Set[Tuple] = set()

        # Index paths_a by terminal entity
        by_terminal: Dict[str, List] = {}
        for p in paths_a:
            by_terminal.setdefault(p.tail, []).append(p)

        for p_b in paths_b:
            for p_a in by_terminal.get(p_b.tail, []):
                # Extract edge relations from each path (odd indices in nodes list)
                rels_a = [p_a.nodes[i] for i in range(1, len(p_a.nodes), 2)]
                rels_b = [p_b.nodes[i] for i in range(1, len(p_b.nodes), 2)]

                for ra in rels_a:
                    for rb in rels_b:
                        if ra == rb:
                            continue
                        if relations_contradict(ra, rb):
                            sig = (p_b.tail, tuple(sorted([ra, rb])))
                            if sig in seen:
                                continue
                            seen.add(sig)
                            records.append(ContradictionRecord(
                                node_a=p_a.head,
                                node_b=p_b.head,
                                relation_a=ra,
                                relation_b=rb,
                                contradiction_type="cross_path",
                                confidence_a=getattr(p_a, "path_confidence", 1.0),
                                confidence_b=getattr(p_b, "path_confidence", 1.0),
                                note=(
                                    f"Paths reach terminal {p_b.tail!r} via "
                                    f"contradicting relations {ra!r} vs {rb!r}"
                                ),
                            ))

        return records

    # ------------------------------------------------------------------
    # Full scan (index time)
    # ------------------------------------------------------------------

    def scan(
        self,
        G: nx.Graph,
        community_map: Optional[Dict[str, int]] = None,
    ) -> List[ContradictionRecord]:
        """
        Run all applicable index-time detectors.

        Covers Types 1, 3, and 4.
        Type 2 is query-time only (detect_cross_path).
        Type 5 (circular causation) is not yet automated.

        Returns combined, deduplicated list of ContradictionRecords.
        """
        records: List[ContradictionRecord] = []
        records.extend(self.detect_direct(G, community_map))
        records.extend(self.detect_temporal(G))
        records.extend(self.detect_provenance(G))
        return records

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize(
        self,
        G: nx.Graph,
        records: List[ContradictionRecord],
    ) -> int:
        """
        Add CONTRADICTS edges to the graph for each ContradictionRecord.

        CONTRADICTS edges carry structured metadata:
          relation, contradiction_type, relation_a, relation_b,
          provenance_a, provenance_b, confidence_a, confidence_b,
          authority_delta, resolution_status, human_reviewed,
          detected_at, note

        Edge weight is set to 0.1 — low enough that the beam does not follow
        CONTRADICTS edges blindly during normal traversal.

        On undirected graphs, only one edge is added per (a, b) pair.
        On directed graphs, edges are added in the direction of the record.
        Existing CONTRADICTS edges between the same pair are not duplicated.

        Returns the count of new edges added.
        """
        added = 0

        for rec in records:
            edge_data = {
                "relation":           CONTRADICTS_RELATION,
                "contradiction_type": rec.contradiction_type,
                "relation_a":         rec.relation_a,
                "relation_b":         rec.relation_b,
                "provenance_a":       rec.provenance_a,
                "provenance_b":       rec.provenance_b,
                "confidence_a":       rec.confidence_a,
                "confidence_b":       rec.confidence_b,
                "authority_delta":    rec.authority_delta,
                "resolution_status":  rec.resolution_status,
                "human_reviewed":     rec.human_reviewed,
                "detected_at":        rec.detected_at,
                "note":               rec.note,
                "weight":             0.1,
                "confidence":         0.5,
            }

            if G.is_multigraph():
                # Multigraphs can hold multiple edges on the same (u, v) pair.
                # Only skip if a CONTRADICTS edge already exists there.
                if G.has_edge(rec.node_a, rec.node_b):
                    existing_rels = {
                        G[rec.node_a][rec.node_b][k].get("relation")
                        for k in G[rec.node_a][rec.node_b]
                    }
                    if CONTRADICTS_RELATION in existing_rels:
                        continue
                G.add_edge(rec.node_a, rec.node_b, **edge_data)
                added += 1
            else:
                # Simple graphs: adding to an occupied (u, v) would overwrite
                # the existing edge data, destroying the original relation.
                # Strategy: try forward direction, then reverse direction.
                placed = False
                for u, v in [(rec.node_a, rec.node_b), (rec.node_b, rec.node_a)]:
                    if not G.has_edge(u, v):
                        # Direction is free — safe to add
                        G.add_edge(u, v, **edge_data)
                        added += 1
                        placed = True
                        break
                    existing = G.get_edge_data(u, v) or {}
                    if existing.get("relation") == CONTRADICTS_RELATION:
                        # CONTRADICTS already materialized in this direction
                        placed = True  # already present — not an error
                        break
                # If both directions are occupied by non-CONTRADICTS edges,
                # the contradiction is still recorded in the ContradictionRecord
                # (returned by scan/detect_direct) but cannot be materialized
                # as a new edge in a simple graph without overwriting real data.
                # In that case, placed=False and we silently skip.

        return added

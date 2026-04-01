"""
Provable graph completion rules for CEREBRUM.

Rules add synthetic edges to the graph with full provenance metadata.
Every synthetic edge cites the specific rule and backing evidence that
produced it.  No statistical predictions are made — only logical deductions.

Available rules
---------------
InverseRule
    Add B -R_inv-> A whenever A -R-> B exists, for a known symmetric or
    inverse-relation pair.  Backed by the source edge.

CompositionRule
    Add A -R_composed-> C whenever A -R1-> B -R2-> C exists, for a known
    composable relation pair.  Backed by the two-hop path (A,B,C).

All synthetic edges carry:
  relation    : the logical relation name
  synthetic   : True
  confidence  : min(conf(backing edges))   [weakest-link rule]
  provenance  : structured string citing the rule and backing evidence
  weight      : min(weight(backing edges))

Provenenance format
-------------------
  InverseRule:
    "rule:inverse:{R}→{R_inv}|source:{src_id}→{tgt_id}"

  CompositionRule:
    "rule:compose:{R1}+{R2}→{Rc}|path:{a}→{b}→{c}"
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter

logger = logging.getLogger("cerebrum.graph_completion")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CompletionRule:
    """Abstract base for provable graph completion rules."""

    def apply(self, adapter: NetworkXAdapter) -> int:
        """
        Apply this rule to the adapter's underlying graph.

        Returns the number of new edges added.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Human-readable description of what this rule does."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InverseRule
# ---------------------------------------------------------------------------

class InverseRule(CompletionRule):
    """
    Materialize the inverse (or symmetric twin) of a known relation.

    For symmetric relations (e.g. MARRIED_TO):
        InverseRule("MARRIED_TO")
        → adds B -MARRIED_TO-> A for every A -MARRIED_TO-> B

    For asymmetric inverse pairs (e.g. directed_by / director_of):
        InverseRule("directed_by", "director_of")
        → adds B -director_of-> A for every A -directed_by-> B

    Logical basis
    -------------
    The rule encodes a domain fact about the relation's semantics — not a
    statistical guess.  The caller is asserting that the relation is known
    to be symmetric (or to have a named inverse), which is a statement about
    the ontology, not about the data.

    Provenance
    ----------
    Every synthetic edge records:
        provenance = "rule:inverse:{relation}→{inverse}|source:{src}→{tgt}"
        confidence = source edge confidence
        synthetic  = True
    """

    def __init__(self, relation: str, inverse_relation: Optional[str] = None):
        """
        Parameters
        ----------
        relation         : the forward relation to match
        inverse_relation : the relation name for the reverse edge.
                           Defaults to ``relation`` (symmetric case).
        """
        self.relation         = relation
        self.inverse_relation = inverse_relation or relation

    def describe(self) -> str:
        if self.inverse_relation == self.relation:
            return f"InverseRule: {self.relation!r} is symmetric → add reverse edges"
        return (
            f"InverseRule: {self.relation!r} → "
            f"inverse is {self.inverse_relation!r}"
        )

    def apply(self, adapter: NetworkXAdapter) -> int:
        G   = adapter._G
        new = 0

        # Collect edges to add (don't mutate while iterating)
        to_add: List[Tuple] = []
        for src, tgt, data in list(G.edges(data=True)):
            if data.get("relation") != self.relation:
                continue
            # Skip if the inverse already exists
            if G.has_edge(tgt, src):
                existing = G.get_edge_data(tgt, src) or {}
                if existing.get("relation") == self.inverse_relation:
                    continue

            provenance = (
                f"rule:inverse:{self.relation}→{self.inverse_relation}"
                f"|source:{src}→{tgt}"
            )
            to_add.append((
                tgt,
                src,
                {
                    "relation":   self.inverse_relation,
                    "weight":     float(data.get("weight", 1.0)),
                    "confidence": float(data.get("confidence", 1.0)),
                    "provenance": provenance,
                    "synthetic":  True,
                    "valid_from": data.get("valid_from"),
                    "valid_to":   data.get("valid_to"),
                },
            ))

        for tgt, src, attrs in to_add:
            G.add_edge(tgt, src, **attrs)
            new += 1

        if new:
            logger.debug(
                "InverseRule(%r → %r): added %d edges",
                self.relation, self.inverse_relation, new,
            )
            # Invalidate n-gram index if present
            if hasattr(adapter, "_ngram_index"):
                del adapter._ngram_index

        return new


# ---------------------------------------------------------------------------
# CompositionRule
# ---------------------------------------------------------------------------

class CompositionRule(CompletionRule):
    """
    Materialize a composed relation A -Rc-> C whenever A -R1-> B -R2-> C exists.

    Example: if R1="directed_by" and R2="born_in", then composing them
    produces A -directed_by:born_in-> C meaning "the director of A was born
    in C".  Give it a meaningful name via ``composed_relation``.

    Logical basis
    -------------
    The synthetic edge is not a guess — it is a shortcut backed by the
    two-hop path (A, B, C) that already exists in the graph.  The path is
    cited in the provenance of every added edge.  If the backing path is
    ever removed, the synthetic edge loses its citation but can be recomputed.

    This is identical to what a query engine would compute on-the-fly for a
    two-hop SPARQL join; materializing it makes it traversable in one hop.

    Parameters
    ----------
    r1                : first relation in the chain
    r2                : second relation in the chain
    composed_relation : name for the materialized shortcut edge
    min_occurrences   : only materialize if this (A, C) pair is reached by
                        at least ``min_occurrences`` distinct intermediate B
                        nodes.  Default 1 (add on any single evidence path).
                        Raise to 2+ to require corroborating evidence.
    max_edges         : safety cap on total edges added (default 500_000)

    Provenance
    ----------
    Every synthetic edge records:
        provenance = "rule:compose:{R1}+{R2}→{Rc}|path:{a}→{b}→{c}"
                     (first supporting path only; confidence = weakest-link)
        synthetic  = True
    """

    def __init__(
        self,
        r1:                str,
        r2:                str,
        composed_relation: str,
        min_occurrences:   int = 1,
        max_edges:         int = 500_000,
    ):
        self.r1                = r1
        self.r2                = r2
        self.composed_relation = composed_relation
        self.min_occurrences   = min_occurrences
        self.max_edges         = max_edges

    def describe(self) -> str:
        return (
            f"CompositionRule: {self.r1!r} + {self.r2!r} → "
            f"{self.composed_relation!r}  "
            f"(min_occurrences={self.min_occurrences})"
        )

    def apply(self, adapter: NetworkXAdapter) -> int:
        G   = adapter._G
        new = 0

        # Index: A → {B nodes reachable via R1}
        # Then for each B, follow R2 to find C nodes.
        # Group by (A, C) to count distinct intermediates.

        # intermediate_data[(a, c)] = list of (b, conf_ab, conf_bc, prov_path)
        from collections import defaultdict
        evidence: Dict[
            Tuple[str, str],
            List[Tuple[str, float, float, str]]
        ] = defaultdict(list)

        for a, b, d_ab in G.edges(data=True):
            if d_ab.get("relation") != self.r1:
                continue
            conf_ab = float(d_ab.get("confidence", 1.0))

            for _b, c, d_bc in G.out_edges(b, data=True) if G.is_directed() \
                    else G.edges(b, data=True):
                if _b != b:
                    continue
                if d_bc.get("relation") != self.r2:
                    continue
                if c == a:          # skip trivial cycles
                    continue
                conf_bc = float(d_bc.get("confidence", 1.0))
                prov    = (
                    f"rule:compose:{self.r1}+{self.r2}→{self.composed_relation}"
                    f"|path:{a}→{b}→{c}"
                )
                evidence[(a, c)].append((b, conf_ab, conf_bc, prov))

        to_add: List[Tuple] = []
        for (a, c), paths in evidence.items():
            if len(paths) < self.min_occurrences:
                continue
            if G.has_edge(a, c):
                existing = G.get_edge_data(a, c) or {}
                if existing.get("relation") == self.composed_relation:
                    continue

            # Use the highest-confidence path as the primary citation
            best = max(paths, key=lambda x: min(x[1], x[2]))
            b_best, conf_ab, conf_bc, prov = best
            confidence = min(conf_ab, conf_bc)

            to_add.append((
                a,
                c,
                {
                    "relation":   self.composed_relation,
                    "weight":     confidence,
                    "confidence": confidence,
                    "provenance": prov,
                    "synthetic":  True,
                },
            ))

            if len(to_add) >= self.max_edges:
                logger.warning(
                    "CompositionRule(%r+%r): hit max_edges cap %d",
                    self.r1, self.r2, self.max_edges,
                )
                break

        for a, c, attrs in to_add:
            G.add_edge(a, c, **attrs)
            new += 1

        if new:
            logger.debug(
                "CompositionRule(%r+%r → %r): added %d edges",
                self.r1, self.r2, self.composed_relation, new,
            )
            if hasattr(adapter, "_ngram_index"):
                del adapter._ngram_index

        return new


# ---------------------------------------------------------------------------
# GeneralizedCompositionRule
# ---------------------------------------------------------------------------

class GeneralizedCompositionRule(CompletionRule):
    """
    Auto-discover and materialize ALL composable 2-hop relation pairs in the
    graph without requiring manual (R1, R2) specification.

    For every intermediate node B that has at least one incoming edge with
    relation R1 and at least one outgoing edge with relation R2, materialize
    a shortcut A -"R1|R2"-> C for every matching (A, B, C) triple.

    Only relation pairs that appear in at least ``min_occurrences`` distinct
    (A, C) pairs are materialized (corroborating-evidence filter).

    This is a *generalization* of CompositionRule: it is equivalent to running
    CompositionRule for every (R1, R2) pair in the graph, but does so in a
    single efficient pass without needing explicit enumeration.

    Logical basis
    -------------
    Every synthetic edge is backed by a concrete 2-hop path (A, B, C) that
    already exists in the graph.  The composed relation name encodes exactly
    which two relations were chained, preserving full traceability.

    Parameters
    ----------
    min_occurrences : minimum number of distinct (A, C) pairs that must be
                      reachable via a given (R1, R2) pair before edges for
                      that pair are materialized (default 2)
    max_edges       : safety cap on total edges added (default 1_000_000)

    Provenance
    ----------
    Every synthetic edge records:
        provenance  = "rule:compose_auto:{R1}+{R2}|path:{A}→{B}→{C}"
        relation    = "{R1}|{R2}"
        synthetic   = True
        confidence  = min(conf_ab, conf_bc)
    """

    def __init__(
        self,
        min_occurrences: int = 2,
        max_edges:       int = 1_000_000,
    ):
        self.min_occurrences = min_occurrences
        self.max_edges       = max_edges

    def describe(self) -> str:
        return (
            f"GeneralizedCompositionRule: auto-discover all R1+R2 pairs "
            f"(min_occurrences={self.min_occurrences}, "
            f"max_edges={self.max_edges:,})"
        )

    def apply(self, adapter: NetworkXAdapter) -> int:  # noqa: C901
        from collections import defaultdict

        G   = adapter._G
        new = 0

        # pair_evidence[(R1, R2)][(A, C)] = (best_b, best_conf, best_prov)
        # We accumulate evidence per (R1,R2) pair, then filter by min_occurrences.
        pair_evidence: Dict[
            Tuple[str, str],
            Dict[Tuple[str, str], Tuple[str, float, str]],
        ] = defaultdict(dict)

        directed = G.is_directed()

        for b in G.nodes():
            # Incoming edges A → B
            in_edges  = list(G.in_edges(b, data=True)) if directed \
                        else [(nb, b, d) for nb, d in G[b].items()
                              for d in ([d] if isinstance(d, dict) else d.values())]
            # Outgoing edges B → C
            out_edges = list(G.out_edges(b, data=True)) if directed \
                        else [(b, nb, d) for nb, d in G[b].items()
                              for d in ([d] if isinstance(d, dict) else d.values())]

            if not in_edges or not out_edges:
                continue

            for a, _b, d_ab in in_edges:
                r1 = d_ab.get("relation")
                if not r1:
                    continue
                conf_ab = float(d_ab.get("confidence", 1.0))

                for _b2, c, d_bc in out_edges:
                    if c == a:          # skip trivial self-loops
                        continue
                    r2 = d_bc.get("relation")
                    if not r2:
                        continue
                    conf_bc = float(d_bc.get("confidence", 1.0))
                    conf    = min(conf_ab, conf_bc)
                    key     = (r1, r2)
                    ac      = (a, c)
                    prov    = (
                        f"rule:compose_auto:{r1}+{r2}"
                        f"|path:{a}\u2192{b}\u2192{c}"
                    )

                    existing = pair_evidence[key].get(ac)
                    if existing is None or conf > existing[1]:
                        pair_evidence[key][ac] = (b, conf, prov)

        # Materialize edges for qualifying pairs
        to_add: List[Tuple] = []
        for (r1, r2), ac_map in pair_evidence.items():
            if len(ac_map) < self.min_occurrences:
                continue
            composed_rel = f"{r1}|{r2}"
            for (a, c), (_b, conf, prov) in ac_map.items():
                if G.has_edge(a, c):
                    existing = G.get_edge_data(a, c) or {}
                    if existing.get("relation") == composed_rel:
                        continue
                to_add.append((
                    a,
                    c,
                    {
                        "relation":   composed_rel,
                        "weight":     conf,
                        "confidence": conf,
                        "provenance": prov,
                        "synthetic":  True,
                    },
                ))
                if len(to_add) >= self.max_edges:
                    break
            if len(to_add) >= self.max_edges:
                logger.warning(
                    "GeneralizedCompositionRule: hit max_edges cap %d",
                    self.max_edges,
                )
                break

        for a, c, attrs in to_add:
            G.add_edge(a, c, **attrs)
            new += 1

        if new:
            logger.debug("GeneralizedCompositionRule: added %d edges", new)
            if hasattr(adapter, "_ngram_index"):
                del adapter._ngram_index

        return new

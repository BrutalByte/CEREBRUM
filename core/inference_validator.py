"""
Phase 22 — InferenceValidator: hold-out causal precision/recall harness.

Validates the confidence calibration of TransitiveInferenceEngine by treating
known graph edges as ground-truth and measuring how often inference rediscovers
them when they are held out.

Usage
-----
    from core.inference_validator import InferenceValidator

    validator = InferenceValidator(adapter, hold_out_fraction=0.2, seed=42)
    report = validator.validate(dry_run=True)
    print(report.summary())

Algorithm
---------
1. Sample ``hold_out_fraction`` of edges that *could* be inferred
   (i.e. edges whose (source, target) pair would be reachable via at least
   one two-hop path through any intermediate node with a matching rule).
2. Remove those edges from a working copy of the graph.
3. Run TransitiveInferenceEngine on the modified graph.
4. Measure: what fraction of the held-out edges appear in the proposals?

This is entirely self-contained — no external labels, no LLM.  The graph's own
topology provides the ground truth.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    held_out_edges: int
    """Total edges withheld from the graph for this validation run."""

    proposals_generated: int
    """Total inference proposals produced on the modified graph."""

    recovered: int
    """How many held-out edges appeared in the proposals (true positives)."""

    precision: float
    """recovered / proposals_generated  (what fraction of proposals are correct)."""

    recall: float
    """recovered / held_out_edges  (what fraction of held-out edges were found)."""

    f1: float
    """Harmonic mean of precision and recall."""

    confidence_calibration: Dict[str, float]
    """
    Per-rule-domain mean confidence of correct vs incorrect proposals:
    {"domain.correct": mean_conf, "domain.incorrect": mean_conf, ...}
    """

    duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            "InferenceValidator Report",
            f"  Held-out edges   : {self.held_out_edges}",
            f"  Proposals        : {self.proposals_generated}",
            f"  Recovered        : {self.recovered}",
            f"  Precision        : {self.precision:.3f}",
            f"  Recall           : {self.recall:.3f}",
            f"  F1               : {self.f1:.3f}",
            f"  Duration         : {self.duration_seconds:.2f}s",
        ]
        if self.confidence_calibration:
            lines.append("  Confidence calibration:")
            for k, v in sorted(self.confidence_calibration.items()):
                lines.append(f"    {k}: {v:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class InferenceValidator:
    """
    Hold-out precision/recall harness for TransitiveInferenceEngine.

    Parameters
    ----------
    adapter
        GraphAdapter whose underlying graph provides the ground truth.
        Must expose a NetworkX graph via ``to_networkx()``.
    hold_out_fraction : float
        Fraction of "inferable" edges to withhold.  Default 0.20 (20 %).
    seed : int
        Random seed for reproducible hold-out sampling.
    min_confidence : float
        Minimum confidence threshold passed to TransitiveInferenceEngine.
    max_proposals : int
        Maximum proposals passed to TransitiveInferenceEngine.
    enabled_domains : optional list
        Restrict inference to these domains.
    """

    def __init__(
        self,
        adapter,
        hold_out_fraction: float = 0.20,
        seed: int = 42,
        min_confidence: float = 0.10,
        max_proposals: int = 500,
        enabled_domains: Optional[List[str]] = None,
        path_preserving: bool = True,
    ) -> None:
        self._adapter          = adapter
        self._hold_out_frac    = hold_out_fraction
        self._seed             = seed
        self._min_confidence   = min_confidence
        self._max_proposals    = max_proposals
        self._enabled_domains  = enabled_domains
        # Hole 4 — Sparse-Graph Validation Bias: Path-Preserving Hold-out.
        # When True (default), an edge (u,v) is only held out if at least one
        # alternative multi-hop path exists between u and v after removal.
        # Prevents false-zero recall on sparse graphs where 20% removal can
        # shatter connectivity, making ground-truth paths unreachable.
        self._path_preserving  = path_preserving

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, dry_run: bool = False) -> ValidationReport:
        """
        Run one validation pass.

        Parameters
        ----------
        dry_run : bool
            When True, the graph is never modified (held-out edges are
            identified but not removed).  The report will have
            ``recovered=0`` and ``recall=0`` because no inference is run.
            Useful for inspecting which edges would be held out.
        """
        t0 = time.time()

        G = self._adapter.to_networkx()
        if G.number_of_nodes() == 0:
            return ValidationReport(
                held_out_edges=0, proposals_generated=0, recovered=0,
                precision=0.0, recall=0.0, f1=0.0,
                confidence_calibration={}, duration_seconds=0.0,
            )

        # 1. Find hold-out candidates
        candidates = self._find_inferable_edges(G)
        rng = random.Random(self._seed)
        rng.shuffle(candidates)

        # Hole 4 — Path-Preserving Hold-out: filter candidates to only those
        # edges (u, v) where removing the edge leaves at least one alternative
        # multi-hop path from u to v, ensuring the transitive inference task
        # is actually possible and recall can never be a false negative caused
        # by graph shatter rather than reasoning failure.
        if self._path_preserving:
            candidates = [
                (u, v, rel) for u, v, rel in candidates
                if self._has_alternative_path(G, u, v)
            ]

        n_hold = max(1, int(len(candidates) * self._hold_out_frac))
        held_out: List[Tuple[str, str, str]] = candidates[:n_hold]  # (u, v, rel)

        if dry_run or not held_out:
            return ValidationReport(
                held_out_edges=len(held_out),
                proposals_generated=0,
                recovered=0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                confidence_calibration={},
                duration_seconds=time.time() - t0,
            )

        # 2. Build modified graph (deep copy, remove held-out edges)
        G_mod = G.copy()
        for u, v, rel in held_out:
            if G_mod.has_edge(u, v):
                G_mod.remove_edge(u, v)
            if not G_mod.is_directed() or True:
                if G_mod.has_edge(v, u):
                    G_mod.remove_edge(v, u)

        # 3. Wrap in a temporary adapter and run inference
        from adapters.networkx_adapter import NetworkXAdapter
        from core.inference_engine import TransitiveInferenceEngine

        tmp_adapter = NetworkXAdapter(G_mod)
        engine = TransitiveInferenceEngine(
            tmp_adapter,
            max_proposals=self._max_proposals,
            min_confidence=self._min_confidence,
            enabled_domains=self._enabled_domains,
        )
        report = engine.run(dry_run=True)

        # 4. Score proposals against held-out set
        held_out_keys: Set[Tuple[str, str]] = {(u, v) for u, v, _ in held_out}
        # Also accept (v, u) for undirected graphs
        if not G.is_directed():
            held_out_keys |= {(v, u) for u, v, _ in held_out}

        recovered = 0
        cal_correct: Dict[str, List[float]] = {}
        cal_incorrect: Dict[str, List[float]] = {}

        for p in report.proposals:
            domain = p.rule.domain
            if (p.source, p.target) in held_out_keys:
                recovered += 1
                cal_correct.setdefault(domain, []).append(p.confidence)
            else:
                cal_incorrect.setdefault(domain, []).append(p.confidence)

        n_proposals = len(report.proposals)
        precision = recovered / n_proposals if n_proposals > 0 else 0.0
        recall    = recovered / len(held_out) if held_out else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Build calibration dict
        calib: Dict[str, float] = {}
        for domain, confs in cal_correct.items():
            calib[f"{domain}.correct"] = sum(confs) / len(confs)
        for domain, confs in cal_incorrect.items():
            calib[f"{domain}.incorrect"] = sum(confs) / len(confs)

        return ValidationReport(
            held_out_edges=len(held_out),
            proposals_generated=n_proposals,
            recovered=recovered,
            precision=precision,
            recall=recall,
            f1=f1,
            confidence_calibration=calib,
            duration_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_alternative_path(self, G: nx.Graph, u: str, v: str) -> bool:
        """
        Return True if removing edge (u, v) leaves at least one alternative
        path from u to v of length >= 2.

        Uses a temporary graph copy so the original is never mutated.
        """
        G_tmp = G.copy()
        if G_tmp.has_edge(u, v):
            G_tmp.remove_edge(u, v)
        if not G_tmp.is_directed() and G_tmp.has_edge(v, u):
            G_tmp.remove_edge(v, u)
        try:
            return nx.has_path(G_tmp, u, v)
        except nx.NodeNotFound:
            return False

    def _find_inferable_edges(
        self, G: nx.Graph
    ) -> List[Tuple[str, str, str]]:
        """
        Find edges in G that *could* be re-derived by transitive inference
        (i.e. there exists a two-hop path A->M->B with a matching rule, and
        a direct A->B edge already exists that confirms the derivation).

        Returns list of (source, target, relation_type) tuples.
        """
        from core.inference_engine import INFERENCE_RULES

        # Build rule index: (rel_a, rel_b) -> derived_relation
        rule_index: Dict[Tuple[str, str], str] = {}
        for rule in INFERENCE_RULES:
            rule_index[(rule.rel_a, rule.rel_b)] = rule.derived

        def _rel(data: dict) -> str:
            return (data.get("relation_type") or
                    data.get("relation") or "").upper()

        candidates: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        nodes = list(G.nodes())
        for mid in nodes:
            in_edges  = list(G.in_edges(mid, data=True)) if G.is_directed() \
                        else [(nbr, mid, G[nbr][mid]) for nbr in G.neighbors(mid)]
            out_edges = list(G.out_edges(mid, data=True)) if G.is_directed() \
                        else [(mid, nbr, G[mid][nbr]) for nbr in G.neighbors(mid)]

            for u, _m1, data_a in in_edges:
                rel_a = _rel(data_a)
                for _m2, v, data_b in out_edges:
                    if v == u:
                        continue
                    rel_b = _rel(data_b)
                    derived = rule_index.get((rel_a, rel_b))
                    if derived is None:
                        continue
                    # Does a direct u->v edge exist?
                    if G.has_edge(u, v) or (not G.is_directed() and G.has_edge(v, u)):
                        key = (u, v) if (u, v) not in seen else None
                        if key and key not in seen:
                            seen.add(key)
                            seen.add((v, u))
                            direct_rel = _rel(G[u][v] if G.has_edge(u, v) else G[v][u])
                            candidates.append((u, v, direct_rel or derived))

        return candidates

"""
RelationBoostDeriver — KB-agnostic per-relation r2_boost weights (Phase 202/203).

Phase 202: single scale factor gamma.
    boost(r) = gamma * fan_out(r)

Phase 203: two-parameter power-law formula.
    boost(r) = gamma * fan_out(r)^beta

fan_out(r) = total_triples_with_r / unique_head_entities_for_r

beta controls the shape of the fan_out → boost mapping:
  beta=1.0  linear (Phase 202 behaviour — preserved by default)
  beta>1.0  amplifies high-fan_out relations disproportionately,
            reproducing the asymmetry seen in hand-tuned per-relation params.
  beta<1.0  compresses differences — useful for near-uniform KBs.

Both parameters are fully KB-agnostic: computed from the graph's own structure
at load time with no domain knowledge required.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple


class RelationBoostDeriver:
    """Derives per-relation r2_boost weights from KB fan-out statistics."""

    def __init__(self) -> None:
        self._fan_out: Dict[str, float] = {}
        self._unique_heads: Dict[str, int] = {}
        self._unique_tails: Dict[str, int] = {}
        self._triple_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_file(self, kb_path: str, sep: str = "|") -> None:
        """Compute fan-out statistics by reading a sep-delimited triples file."""
        rel_heads: Dict[str, set] = defaultdict(set)
        rel_tails: Dict[str, set] = defaultdict(set)
        rel_count: Dict[str, int] = defaultdict(int)
        with open(kb_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) == 3:
                    h, r, t = parts
                    rel_heads[r].add(h)
                    rel_tails[r].add(t)
                    rel_count[r] += 1
        self._finalise(rel_heads, rel_tails, rel_count)

    def build_from_triples(
        self, triples: Iterable[Tuple[str, str, str]]
    ) -> None:
        """Compute fan-out statistics from an iterable of (head, relation, tail)."""
        rel_heads: Dict[str, set] = defaultdict(set)
        rel_tails: Dict[str, set] = defaultdict(set)
        rel_count: Dict[str, int] = defaultdict(int)
        for h, r, t in triples:
            rel_heads[r].add(h)
            rel_tails[r].add(t)
            rel_count[r] += 1
        self._finalise(rel_heads, rel_tails, rel_count)

    def _finalise(
        self,
        rel_heads: Dict[str, set],
        rel_tails: Dict[str, set],
        rel_count: Dict[str, int],
    ) -> None:
        self._fan_out = {
            r: rel_count[r] / max(len(rel_heads[r]), 1) for r in rel_count
        }
        self._unique_heads = {r: len(rel_heads[r]) for r in rel_count}
        self._unique_tails = {r: len(rel_tails[r]) for r in rel_count}
        self._triple_count = dict(rel_count)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return bool(self._fan_out)

    def fan_out(self, relation: str) -> float:
        """Return fan_out for a relation, or 1.0 if not in KB."""
        return self._fan_out.get(relation, 1.0)

    def boost_map(self, gamma: float, beta: float = 1.0) -> Optional[Dict[str, float]]:
        """
        Return {relation: gamma * fan_out(r)^beta} for all KB relations.

        beta=1.0 (default) reproduces Phase 202 linear behaviour.
        beta>1.0 amplifies high-fan_out relations disproportionately.
        Returns None if not built or gamma <= 0.
        """
        if not self.is_built or gamma <= 0.0:
            return None
        if beta == 1.0:
            return {r: gamma * fo for r, fo in self._fan_out.items()}
        return {r: gamma * (fo ** beta) for r, fo in self._fan_out.items()}

    def fan_out_stats(self):
        """Return (max_fan_out, mean_fan_out, harmonic_mean_fan_out, n_relations).

        Used by ParameterInitializer to derive gamma analytically.
        Returns (1.0, 1.0, 1.0, 0) if not built.
        """
        if not self.is_built:
            return 1.0, 1.0, 1.0, 0
        values = list(self._fan_out.values())
        n = len(values)
        max_fo = max(values)
        mean_fo = sum(values) / n
        harmonic_fo = n / sum(1.0 / v for v in values)
        return max_fo, mean_fo, harmonic_fo, n

    def relation_stats(self) -> Dict[str, Dict]:
        """Full per-relation statistics for logging and analysis."""
        return {
            r: {
                "fan_out":      self._fan_out[r],
                "unique_heads": self._unique_heads[r],
                "unique_tails": self._unique_tails[r],
                "triple_count": self._triple_count[r],
            }
            for r in self._fan_out
        }

    def summary(self) -> str:
        if not self.is_built:
            return "RelationBoostDeriver(not built)"
        lines = ["RelationBoostDeriver fan-out:"]
        for r, fo in sorted(self._fan_out.items(), key=lambda x: -x[1]):
            lines.append(f"  {r:32s}  {fo:.4f}")
        return "\n".join(lines)

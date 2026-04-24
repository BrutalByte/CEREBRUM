"""
CounterfactualEngine — Phase 123: Interventional & Counterfactual Reasoning.

Extends CausalEngine (Phase 120) with do-calculus style interventions.
Answers queries of the form:

  "If we remove node X from the causal graph, does A still cause B?"
  "If we add a hypothetical edge X→Y, does the effect estimate change?"
  "Was node X necessary for A to cause B?"

Implementation follows Pearl's do-calculus intervention model:

  do(X): cut all incoming causal edges to X (block_node).
          Simulates "forcing X to a state regardless of its causes."
  do(X→Y): remove a specific directed causal edge (block_edge).
  add(X→Y): inject a hypothetical causal edge with given weight (add_edge).

The engine never mutates the real adapter. It wraps it in a lightweight
InterventionalAdapter that filters get_neighbors() calls according to the
declared interventions. The CausalEngine is then run twice — once on the
real adapter (factual world) and once on the interventional adapter
(counterfactual world) — and the results are compared.

Key outputs
-----------
  factual_effect        Noisy-OR effect in the real graph [0, 1]
  counterfactual_effect Noisy-OR effect after intervention [0, 1]
  effect_delta          counterfactual_effect - factual_effect
  all_paths_blocked     True when every factual causal path was blocked
                        (the intervened-on structure was necessary)
  alternative_paths     Causal paths that survive the intervention
                        (evidence of redundant causal mechanisms)

Terminology note
----------------
"Necessary cause": removing it eliminates the effect → all_paths_blocked=True
"Sufficient cause": it alone produces the effect even after other paths are
                    blocked → alternative_paths is non-empty AND factual_effect>0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from core.causal_engine import CausalEngine, CausalProof, CAUSAL_RELATIONS

if TYPE_CHECKING:
    from core.truth_cache import TruthCache

logger = logging.getLogger("cerebrum.counterfactual")

# ---------------------------------------------------------------------------
# Intervention model
# ---------------------------------------------------------------------------

BLOCK_NODE = "block_node"  # remove all causal edges through this node
BLOCK_EDGE = "block_edge"  # remove a specific (source, target, relation) edge
ADD_EDGE   = "add_edge"    # inject a hypothetical causal edge


@dataclass
class Intervention:
    """A single do-calculus intervention."""

    type: str
    """One of: 'block_node', 'block_edge', 'add_edge'."""

    node: Optional[str] = None
    """Node to block (block_node only)."""

    source: Optional[str] = None
    """Source entity (block_edge / add_edge)."""

    target: Optional[str] = None
    """Target entity (block_edge / add_edge)."""

    relation: Optional[str] = None
    """Relation type (block_edge / add_edge)."""

    weight: float = 0.5
    """Causal weight for injected edges (add_edge only)."""

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "type": self.type,
            "node": self.node,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Intervention":
        return cls(**{k: v for k, v in d.items() if k in {
            "type", "node", "source", "target", "relation", "weight"
        }})


# ---------------------------------------------------------------------------
# Interventional adapter
# ---------------------------------------------------------------------------

class _InjectedEdge:
    """Minimal neighbor object for add_edge hypothetical edges."""

    def __init__(self, target_id: str, relation_type: str, weight: float) -> None:
        self.target_id = target_id
        self.relation_type = relation_type
        self.properties = {"causal_weight": weight}


class InterventionalAdapter:
    """
    Wraps a real GraphAdapter and applies declared interventions.

    Never mutates the underlying adapter — all filtering is done in memory.
    """

    def __init__(
        self,
        base_adapter: Any,
        interventions: List[Intervention],
        causal_relations: Optional[frozenset] = None,
    ) -> None:
        self._base = base_adapter
        self._blocked_nodes: Set[str] = set()
        self._blocked_edges: Set[Tuple[str, str, str]] = set()
        self._added_edges: Dict[str, List[_InjectedEdge]] = {}
        self._causal_rels = causal_relations if causal_relations is not None else CAUSAL_RELATIONS

        for iv in interventions:
            if iv.type == BLOCK_NODE and iv.node:
                self._blocked_nodes.add(iv.node)
            elif iv.type == BLOCK_EDGE and iv.source and iv.target:
                rel = iv.relation or "*"
                self._blocked_edges.add((iv.source, iv.target, rel))
            elif iv.type == ADD_EDGE and iv.source and iv.target and iv.relation:
                inj = _InjectedEdge(iv.target, iv.relation, iv.weight)
                self._added_edges.setdefault(iv.source, []).append(inj)

    # Delegate everything except get_neighbors / to_networkx to base adapter
    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def get_neighbors(self, entity_id: str):
        for neighbor in self._base.get_neighbors(entity_id):
            v, rel = self._extract_vr(neighbor)
            if v in self._blocked_nodes:
                continue
            if self._is_blocked_edge(entity_id, v, rel):
                continue
            yield neighbor
        # Inject hypothetical edges added by add_edge interventions
        for inj in self._added_edges.get(entity_id, []):
            if inj.target_id not in self._blocked_nodes:
                yield inj

    def to_networkx(self):
        G = self._base.to_networkx().copy()
        for node in self._blocked_nodes:
            if node in G:
                G.remove_node(node)
        for u, v, rel in self._blocked_edges:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        for src, injections in self._added_edges.items():
            for inj in injections:
                if inj.target_id in self._blocked_nodes:
                    continue
                # Add nodes unconditionally — hypothetical edges may introduce
                # entities not present in the base graph.
                if src not in G:
                    G.add_node(src)
                if inj.target_id not in G:
                    G.add_node(inj.target_id)
                G.add_edge(src, inj.target_id, relation=inj.relation_type,
                           causal_weight=inj.weight)
        return G

    def _extract_vr(self, neighbor: Any) -> Tuple[str, str]:
        if hasattr(neighbor, "target_id"):
            return neighbor.target_id, getattr(neighbor, "relation_type", "UNKNOWN")
        if isinstance(neighbor, tuple):
            if len(neighbor) == 2:
                if isinstance(neighbor[1], str):
                    return neighbor[0], neighbor[1]
                if isinstance(neighbor[1], dict):
                    return neighbor[0], neighbor[1].get("relation", "UNKNOWN")
        return str(neighbor), "UNKNOWN"

    def _is_blocked_edge(self, u: str, v: str, rel: str) -> bool:
        if (u, v, rel) in self._blocked_edges:
            return True
        # Wildcard match: (u, v, "*") blocks all relations on that pair
        return (u, v, "*") in self._blocked_edges


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""

    source: str
    target: str
    interventions: List[dict]

    factual_effect: float
    """Noisy-OR causal effect in the unmodified graph."""

    counterfactual_effect: float
    """Noisy-OR causal effect after applying interventions."""

    effect_delta: float
    """counterfactual_effect - factual_effect.  Negative = intervention weakened causation."""

    factual_paths: List[List[str]]
    """Causal paths found in the real graph."""

    alternative_paths: List[List[str]]
    """Factual causal paths that survive the intervention (redundant mechanisms)."""

    all_paths_blocked: bool
    """True when every factual path was eliminated — the intervened structure was necessary."""

    paths_blocked_count: int
    """Number of factual paths blocked by the intervention."""

    factual_confounders: List[str]
    """Confounders detected in the real graph."""

    counterfactual_identification_method: str
    """Identification method in the counterfactual world."""

    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "interventions": self.interventions,
            "factual_effect": round(self.factual_effect, 4),
            "counterfactual_effect": round(self.counterfactual_effect, 4),
            "effect_delta": round(self.effect_delta, 4),
            "factual_paths": self.factual_paths,
            "alternative_paths": self.alternative_paths,
            "all_paths_blocked": self.all_paths_blocked,
            "paths_blocked_count": self.paths_blocked_count,
            "factual_confounders": self.factual_confounders,
            "counterfactual_identification_method": self.counterfactual_identification_method,
            "cached": self.cached,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """
    Interventional reasoning engine for CEREBRUM knowledge graphs.

    Parameters
    ----------
    adapter           : GraphAdapter — the real (factual) graph
    truth_cache       : TruthCache (optional) — caches factual proofs only
    max_backdoor_hops : depth for confounder search (passed to CausalEngine)
    causal_relations  : relation set to treat as causal (defaults to CAUSAL_RELATIONS)
    """

    def __init__(
        self,
        adapter: Any,
        truth_cache: Optional["TruthCache"] = None,
        max_backdoor_hops: int = 3,
        causal_relations: Optional[frozenset] = None,
    ) -> None:
        self._adapter = adapter
        self._truth_cache = truth_cache
        self._max_backdoor_hops = max_backdoor_hops
        self._causal_relations = causal_relations if causal_relations is not None else CAUSAL_RELATIONS
        # Factual engine operates on the real adapter
        self._factual_engine = CausalEngine(
            adapter=adapter,
            truth_cache=truth_cache,
            max_backdoor_hops=max_backdoor_hops,
            causal_relations=self._causal_relations,
        )

    def query(
        self,
        source: str,
        target: str,
        interventions: List[Intervention],
        max_hop: int = 4,
        beam_width: int = 10,
    ) -> CounterfactualResult:
        """
        Compute factual and counterfactual causal effects between source and target.

        Parameters
        ----------
        source        : seed entity
        target        : target entity
        interventions : list of Intervention objects to apply
        max_hop       : maximum causal path depth
        beam_width    : beam width passed to CausalEngine BFS

        Returns
        -------
        CounterfactualResult with both factual and counterfactual proofs.
        """
        # ── Factual proof (real graph) ────────────────────────────────
        factual: CausalProof = self._factual_engine.query(
            source, target, max_hop=max_hop, beam_width=beam_width
        )

        # ── Counterfactual proof (interventional graph) ───────────────
        iv_adapter = InterventionalAdapter(
            self._adapter, interventions, self._causal_relations
        )
        cf_engine = CausalEngine(
            adapter=iv_adapter,
            truth_cache=None,           # no caching for counterfactual worlds
            max_backdoor_hops=self._max_backdoor_hops,
            causal_relations=self._causal_relations,
        )
        counterfactual: CausalProof = cf_engine.query(
            source, target, max_hop=max_hop, beam_width=beam_width
        )

        # ── Compute which factual paths survived ──────────────────────
        factual_path_set = {tuple(p) for p in factual.direct_paths}
        surviving_paths = [
            list(p) for p in
            {tuple(p) for p in counterfactual.direct_paths} & factual_path_set
        ]
        # Also include new counterfactual-only paths (from add_edge interventions)
        new_cf_paths = [
            p for p in counterfactual.direct_paths
            if tuple(p) not in factual_path_set
        ]
        alternative_paths = surviving_paths + new_cf_paths

        paths_blocked = len(factual.direct_paths) - len(surviving_paths)
        all_blocked = (
            len(factual.direct_paths) > 0 and len(surviving_paths) == 0
            and len(new_cf_paths) == 0
        )

        iv_dicts = [iv.to_dict() for iv in interventions]

        result = CounterfactualResult(
            source=source,
            target=target,
            interventions=iv_dicts,
            factual_effect=factual.effect_estimate,
            counterfactual_effect=counterfactual.effect_estimate,
            effect_delta=counterfactual.effect_estimate - factual.effect_estimate,
            factual_paths=factual.direct_paths,
            alternative_paths=alternative_paths,
            all_paths_blocked=all_blocked,
            paths_blocked_count=paths_blocked,
            factual_confounders=factual.confounders_detected,
            counterfactual_identification_method=counterfactual.identification_method,
        )

        logger.info(
            "Counterfactual %s->%s interventions=%d "
            "factual=%.3f cf=%.3f delta=%+.3f blocked=%d/%d",
            source, target, len(interventions),
            factual.effect_estimate, counterfactual.effect_estimate,
            result.effect_delta, paths_blocked, len(factual.direct_paths),
        )
        return result

"""
CausalEngine — Phase 120: Causal Inference.

Promotes KG traversal from correlational to causal reasoning by:
  - Filtering to causal relation types (CAUSES, ACTIVATES, etc.)
  - Using STDP-generated causal_weight metadata already on edges
  - Enforcing temporal ordering (cause must precede effect via valid_from)
  - Detecting confounding paths (common ancestors = backdoor nodes)
  - Aggregating multi-path causal evidence via Noisy-OR

Extends the existing SymbolicValidator and TruthCache infrastructure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.truth_cache import TruthCache

logger = logging.getLogger("cerebrum.causal")

# Relations that carry directional causal semantics
CAUSAL_RELATIONS: frozenset = frozenset({
    "CAUSES", "ACTIVATES", "INDIRECTLY_CAUSES", "PROMOTES",
    "INDUCES", "TRIGGERS", "STIMULATES", "MAY_CAUSE",
    "LEADS_TO", "DRIVES",
})

# Relations that negate or reverse causation (used for contradiction checks)
ANTI_CAUSAL_RELATIONS: frozenset = frozenset({
    "PREVENTS", "INHIBITS", "BLOCKS", "SUPPRESSES",
    "DOWNREGULATES", "COUNTERS", "OPPOSES",
})


@dataclass
class CausalProof:
    """Result of a causal query between two entities."""
    source: str
    target: str
    effect_estimate: float              # Noisy-OR across independent causal paths [0,1]
    direct_paths: List[List[str]]       # Paths using only causal relations
    confounders_detected: List[str]     # Common ancestors (potential backdoor nodes)
    is_confounded: bool                 # Backdoor path exists and is unblocked
    temporal_valid: bool                # All edges respect valid_from ordering
    causal_relations_used: List[str]    # Distinct causal relation types in result
    confidence: float                   # Final adjusted confidence
    identification_method: str          # "direct" | "mediated" | "noisy_or" | "none"
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "effect_estimate": round(self.effect_estimate, 4),
            "direct_paths": self.direct_paths,
            "confounders_detected": self.confounders_detected,
            "is_confounded": self.is_confounded,
            "temporal_valid": self.temporal_valid,
            "causal_relations_used": self.causal_relations_used,
            "confidence": round(self.confidence, 4),
            "identification_method": self.identification_method,
            "cached": self.cached,
        }


class CausalEngine:
    """
    Causal inference engine for CEREBRUM knowledge graphs.

    Parameters
    ----------
    adapter       : GraphAdapter — must implement get_neighbors(), get_entity()
    truth_cache   : TruthCache (optional) — caches proved causal chains
    max_backdoor_hops : depth for confounder search (default 3)
    """

    def __init__(
        self,
        adapter: Any,
        truth_cache: Optional["TruthCache"] = None,
        max_backdoor_hops: int = 3,
    ) -> None:
        self.adapter = adapter
        self.truth_cache = truth_cache
        self.max_backdoor_hops = max_backdoor_hops

    def query(
        self,
        source: str,
        target: str,
        max_hop: int = 4,
        beam_width: int = 10,
    ) -> CausalProof:
        """
        Find causal paths from source to target; detect confounders.

        Returns a CausalProof with effect estimate, confounder list,
        temporal validity, and adjusted confidence.
        """
        # Cache lookup
        if self.truth_cache is not None:
            cached = self.truth_cache.get_causal_proof(source, target)
            if cached is not None:
                cached.cached = True
                return cached

        # BFS over causal edges only
        causal_paths = self._find_causal_paths(source, target, max_hop)

        if not causal_paths:
            proof = CausalProof(
                source=source, target=target,
                effect_estimate=0.0, direct_paths=[],
                confounders_detected=[], is_confounded=False,
                temporal_valid=True, causal_relations_used=[],
                confidence=0.0, identification_method="none",
            )
            if self.truth_cache is not None:
                self.truth_cache.store_causal_proof(source, target, proof)
            return proof

        # Temporal ordering check
        temporal_valid = all(self._check_temporal_ordering(p) for p in causal_paths)

        # Confounder detection (backdoor paths)
        confounders = self._detect_confounders(source, target, causal_paths)
        is_confounded = len(confounders) > 0

        # Noisy-OR effect aggregation weighted by causal_weight edge metadata
        effect = self._aggregate_effect(causal_paths)

        # Penalty for confounding and temporal violations
        confidence = effect
        if is_confounded:
            confidence *= 0.75
        if not temporal_valid:
            confidence *= 0.85

        # Identify method
        method = self._identify_method(causal_paths)

        # Collect relation types used
        relations_used = list({
            rel for path in causal_paths
            for rel in self._extract_relations(path)
        })

        # Format paths as list-of-strings for serialization
        serialized = [path for path in causal_paths]

        proof = CausalProof(
            source=source, target=target,
            effect_estimate=effect,
            direct_paths=serialized,
            confounders_detected=confounders,
            is_confounded=is_confounded,
            temporal_valid=temporal_valid,
            causal_relations_used=relations_used,
            confidence=confidence,
            identification_method=method,
        )

        if self.truth_cache is not None:
            self.truth_cache.store_causal_proof(source, target, proof)

        logger.info(
            "CausalEngine: %s→%s paths=%d effect=%.3f confounded=%s temporal=%s",
            source, target, len(causal_paths), effect, is_confounded, temporal_valid,
        )
        return proof

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def _find_causal_paths(
        self, source: str, target: str, max_hop: int
    ) -> List[List[str]]:
        """BFS over causal-relation edges only."""
        found: List[List[str]] = []
        # Queue entries: [entity_id, ...] interleaved: [src, rel, node, rel, node, ...]
        queue: List[Tuple[str, List[str]]] = [(source, [source])]
        visited_states: Set[Tuple[str, ...]] = set()

        while queue:
            current, path = queue.pop(0)
            hops = (len(path) - 1) // 2  # each hop = 1 entity + 1 relation

            if hops > max_hop:
                continue

            state = tuple(path)
            if state in visited_states:
                continue
            visited_states.add(state)

            if current == target and hops > 0:
                found.append(path)
                continue

            try:
                neighbors = list(self.adapter.get_neighbors(current))
            except Exception:
                continue

            for neighbor in neighbors:
                v, rel, edge_data = self._unpack_neighbor(neighbor)
                if rel not in CAUSAL_RELATIONS:
                    continue
                if v in [path[i] for i in range(0, len(path), 2)]:
                    continue  # avoid cycles
                queue.append((v, path + [rel, v]))

        return found

    def _unpack_neighbor(self, neighbor: Any) -> Tuple[str, str, dict]:
        """Extract (target_id, relation, edge_data) from various neighbor formats."""
        if hasattr(neighbor, "target_id"):
            return (
                neighbor.target_id,
                getattr(neighbor, "relation_type", "UNKNOWN"),
                getattr(neighbor, "properties", {}),
            )
        if isinstance(neighbor, tuple):
            if len(neighbor) == 2 and isinstance(neighbor[1], str):
                return neighbor[0], neighbor[1], {}
            if len(neighbor) == 2 and isinstance(neighbor[1], dict):
                return neighbor[0], neighbor[1].get("relation", "UNKNOWN"), neighbor[1]
        return str(neighbor), "UNKNOWN", {}

    # ------------------------------------------------------------------
    # Confounder detection (backdoor criterion)
    # ------------------------------------------------------------------

    def _detect_confounders(
        self, source: str, target: str, causal_paths: List[List[str]]
    ) -> List[str]:
        """
        Find common ancestors of source and target that are not on any
        causal path (potential unblocked backdoor paths).
        """
        source_ancestors = self._ancestors(source, self.max_backdoor_hops)
        target_ancestors = self._ancestors(target, self.max_backdoor_hops)
        common = source_ancestors & target_ancestors

        # Nodes already on causal paths are "blocked" mediators, not confounders
        path_nodes: Set[str] = {
            node for path in causal_paths
            for node in path[::2]  # every 2nd element = entity
        }
        return [n for n in common if n not in path_nodes and n != source and n != target]

    def _ancestors(self, node_id: str, max_depth: int) -> Set[str]:
        """Collect all nodes with a directed edge TO node_id up to max_depth."""
        ancestors: Set[str] = set()
        try:
            G = self.adapter.to_networkx()
        except Exception:
            return ancestors

        frontier = {node_id}
        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for n in frontier:
                for pred in G.predecessors(n):
                    if pred not in ancestors:
                        ancestors.add(pred)
                        next_frontier.add(pred)
            frontier = next_frontier
            if not frontier:
                break
        return ancestors

    # ------------------------------------------------------------------
    # Temporal ordering
    # ------------------------------------------------------------------

    def _check_temporal_ordering(self, path: List[str]) -> bool:
        """
        Verify that valid_from timestamps increase monotonically along the path.
        Edges without timestamps are considered valid (no constraint).
        """
        entities = path[::2]
        relations = path[1::2]

        last_ts: Optional[float] = None
        for i, rel in enumerate(relations):
            src = entities[i]
            tgt = entities[i + 1]
            edge_ts = self._get_edge_timestamp(src, tgt, rel)
            if edge_ts is None:
                continue
            if last_ts is not None and edge_ts < last_ts:
                return False
            last_ts = edge_ts
        return True

    def _get_edge_timestamp(self, u: str, v: str, relation: str) -> Optional[float]:
        """Return valid_from timestamp for an edge, or None if unavailable."""
        try:
            G = self.adapter.to_networkx()
            data = G.get_edge_data(u, v) or {}
            return data.get("valid_from")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Effect aggregation (Noisy-OR)
    # ------------------------------------------------------------------

    def _aggregate_effect(self, causal_paths: List[List[str]]) -> float:
        """
        Noisy-OR: P(effect | P1, P2, ...) = 1 - ∏(1 - score_i)
        Each path score is weighted by the minimum causal_weight on the path.
        """
        if not causal_paths:
            return 0.0

        scores = []
        for path in causal_paths:
            entities = path[::2]
            relations = path[1::2]
            causal_weights = []
            for i, rel in enumerate(relations):
                cw = self._get_causal_weight(entities[i], entities[i + 1], rel)
                causal_weights.append(cw)
            path_score = min(causal_weights) if causal_weights else 0.5
            scores.append(path_score)

        # Noisy-OR aggregation
        not_effect = 1.0
        for s in scores:
            not_effect *= (1.0 - s)
        return 1.0 - not_effect

    def _get_causal_weight(self, u: str, v: str, relation: str) -> float:
        """Return STDP causal_weight from edge properties, defaulting to 0.5."""
        try:
            G = self.adapter.to_networkx()
            data = G.get_edge_data(u, v) or {}
            # STDP stores causal_weight directly or in properties dict
            cw = data.get("causal_weight", data.get("properties", {}).get("causal_weight"))
            return float(cw) if cw is not None else 0.5
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _identify_method(self, causal_paths: List[List[str]]) -> str:
        if not causal_paths:
            return "none"
        lengths = [len(p) for p in causal_paths]
        if any(l == 3 for l in lengths):  # [src, rel, tgt]
            return "direct"
        if len(causal_paths) > 1:
            return "noisy_or"
        return "mediated"

    def _extract_relations(self, path: List[str]) -> List[str]:
        return path[1::2]

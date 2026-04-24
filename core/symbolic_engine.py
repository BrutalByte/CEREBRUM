"""
SymbolicEngine — Phase 117: The Prefrontal Bridge.

Provides a formal verification layer to validate reasoning paths against
graph-global logical axioms (e.g., transitivity, symmetry, inverse relations).
"""
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

class ConstraintType(Enum):
    TRANSITIVITY = "transitivity"
    SYMMETRY = "symmetry"
    INVERSE = "inverse"
    TYPE_RESTRICTION = "type_restriction"
    MUST_NOT_HAVE = "must_not_have"
    CAUSAL_ORDERING = "causal_ordering"   # enforce valid_from temporal order
    NO_BACKDOOR = "no_backdoor"           # reject edges to known confounder nodes

class IntegrityConstraint:
    def __init__(self, constraint_type: ConstraintType, params: Dict[str, Any] = None,
                 rel_a: str = "", rel_b: str = "", result: str = ""):
        self.constraint_type = constraint_type
        self.params = params or {}
        self.rel_a = rel_a
        self.rel_b = rel_b
        self.result = result

logger = logging.getLogger("cerebrum.symbolic")

class SymbolicValidator:
    """
    Validates reasoning traces against formal axioms extracted from the KG.
    Acts as the 'Prefrontal Bridge' filtering System 1 (probabilistic) 
    traversal results.
    """
    def __init__(self, adapter, contradiction_engine=None):
        self.adapter = adapter
        self.contradiction_engine = contradiction_engine
        self.constraints: List[IntegrityConstraint] = []
        self.axioms = {
            ("is_located_in", "is_located_in"): "is_located_in",
            ("member_of", "part_of"): "member_of"
        }

    def add_constraint(self, constraint: "IntegrityConstraint") -> None:
        self.constraints.append(constraint)

    def validate_step(self, u: str, relation: str, v: str, path=None) -> bool:
        """Per-hop validation against IntegrityConstraints and contradiction_engine."""
        for c in self.constraints:
            if c.constraint_type == ConstraintType.TYPE_RESTRICTION:
                p = c.params
                if p.get("relation") == relation:
                    u_type = self._node_type(u)
                    v_type = self._node_type(v)
                    if u_type != p.get("source_type") or v_type != p.get("target_type"):
                        return False
            elif c.constraint_type == ConstraintType.MUST_NOT_HAVE:
                p = c.params
                v_type = self._node_type(v)
                if v_type == p.get("type"):
                    attrs = self._node_attrs(v)
                    for prop in p.get("illegal_props", []):
                        if attrs.get(prop):
                            return False
            elif c.constraint_type == ConstraintType.CAUSAL_ORDERING:
                # Enforce that valid_from on this edge >= any prior edge timestamp
                ts = self._get_edge_timestamp(u, v, relation)
                prior_ts = c.params.get("last_timestamp")
                if ts is not None and prior_ts is not None and ts < prior_ts:
                    return False
            elif c.constraint_type == ConstraintType.NO_BACKDOOR:
                # Reject edges to/from known confounder nodes
                confounders = c.params.get("confounders", set())
                if u in confounders or v in confounders:
                    return False

        if self.contradiction_engine is not None and path is not None:
            nodes = list(getattr(path, "nodes", path))
            candidate = nodes + [relation, v] if nodes else [u, relation, v]
            from reasoning.traversal import TraversalPath
            tp = TraversalPath(nodes=candidate, attention_weights=[])
            recs = self.contradiction_engine.detect_contradictions_in_path(tp)
            if recs:
                return False
        return True

    def _get_edge_timestamp(self, u: str, v: str, relation: str) -> Optional[float]:
        try:
            g = getattr(self.adapter, "_G", None) or getattr(self.adapter, "graph", None)
            if g is None:
                return None
            data = g.get_edge_data(u, v) or {}
            return data.get("valid_from")
        except Exception:
            return None

    def _node_type(self, node_id: str) -> Optional[str]:
        try:
            g = getattr(self.adapter, "_G", None) or getattr(self.adapter, "graph", None)
            return g.nodes[node_id].get("type") if g is not None else None
        except Exception:
            return None

    def _node_attrs(self, node_id: str) -> Dict[str, Any]:
        try:
            g = getattr(self.adapter, "_G", None) or getattr(self.adapter, "graph", None)
            return dict(g.nodes[node_id]) if g is not None else {}
        except Exception:
            return {}

    def validate(self, path: List[str]) -> bool:
        """
        Multi-Angle Deductive Convergence (The Triple-Lens Validator):
        1. Transitive Consistency (Lens A)
        2. Symmetry/Inverse Verification (Lens B)
        3. Community Modularity (Lens C)
        """
        # Lens A: Logical Transitivity (Basic Example)
        if not self._check_transitivity(path):
            return False

        # Lens B: Symmetry/Inverse Verification
        if not self._check_symmetry(path):
            return False
            
        # Lens C: Community Integrity
        if not self._check_community_consistency(path):
            return False
            
        return True

    def _check_transitivity(self, path: List[str]) -> bool:
        # Placeholder for axiom-driven transitivity derivation
        return True

    def _check_symmetry(self, path: List[str]) -> bool:
        # Placeholder for relation inverse consistency check
        return True

    def _check_community_consistency(self, path: List[str]) -> bool:
        # Placeholder for modularity boundary traversal validation
        return True

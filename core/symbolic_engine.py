"""
Phase 34 — Symbolic Guardrails (Hard Logic Validation).

Implements hard-logic integrity constraints that reasoning paths must satisfy.
Paths that violate these symbolic rules are pruned during traversal, ensuring
that CEREBRUM never produces logically impossible answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter
    from reasoning.traversal import TraversalPath
    from core.contradiction_engine import ContradictionEngine


class ConstraintType(Enum):
    MUST_NOT_HAVE = auto()  # Node cannot have certain attribute combinations
    TYPE_RESTRICTION = auto() # Relation must connect specific types
    SYMMETRIC = auto()      # If A-R-B, then B-R-A must be possible
    TRANSITIVE = auto()     # If A-R-B and B-R-C, then A-R-C is expected (or implied)
    PATH_EXCLUSION = auto() # Specific relation sequences that are invalid


@dataclass(frozen=True)
class IntegrityConstraint:
    """A symbolic rule that must be satisfied by a reasoning path."""
    
    type: ConstraintType
    domain: str = "general"
    
    # MUST_NOT_HAVE: { "type": "person", "illegal_props": ["is_inanimate"] }
    # TYPE_RESTRICTION: { "relation": "BORN_IN", "source_type": "person", "target_type": "location" }
    params: Dict[str, Any] = field(default_factory=dict)
    
    note: str = ""


class SymbolicValidator:
    """
    Validates TraversalPaths against a set of IntegrityConstraints.
    """

    def __init__(self, adapter: GraphAdapter, contradiction_engine: Optional[ContradictionEngine] = None):
        self.adapter = adapter
        self.contradiction_engine = contradiction_engine
        self.constraints: List[IntegrityConstraint] = []
        self._type_cache: Dict[str, str] = {}

    def add_constraint(self, constraint: IntegrityConstraint) -> None:
        self.constraints.append(constraint)

    def validate_step(self, u: str, rel: str, v: str, path: Optional[TraversalPath] = None) -> bool:
        """
        Check if adding the step (u)-[rel]->(v) violates any constraints.
        Returns True if valid, False if it violates a hard rule.
        """
        u_type = self._get_type(u)
        v_type = self._get_type(v)

        # 1. Integrity Constraints (Type/Must Not Have)
        for c in self.constraints:
            if c.type == ConstraintType.TYPE_RESTRICTION:
                # Enforce source/target type compatibility
                if c.params.get("relation") == rel:
                    req_src = c.params.get("source_type")
                    req_tgt = c.params.get("target_type")
                    
                    if req_src and u_type != req_src:
                        return False
                    if req_tgt and v_type != req_tgt:
                        return False
            
            elif c.type == ConstraintType.MUST_NOT_HAVE:
                # Check for illegal property/type combinations at the target node
                target_type = c.params.get("type")
                if v_type == target_type:
                    illegal_props = c.params.get("illegal_props", [])
                    entity = self.adapter.get_entity(v)
                    if entity:
                        for p in illegal_props:
                            if entity.properties.get(p):
                                return False

            elif c.type == ConstraintType.SYMMETRIC:
                # If rel is symmetric, v-[rel]->u MUST exist in the graph.
                # This ensures we don't follow "one-way" edges for relations that must be mutual.
                if rel in c.params.get("relations", []):
                    neighbors = self.adapter.get_neighbors(v)
                    if not any(n.target_id == u and n.relation_type == rel for n in neighbors):
                        return False

            elif c.type == ConstraintType.PATH_EXCLUSION:
                # Disallow specific relation sequences (e.g., [r1, r2]).
                if path and len(path.nodes) >= 2:
                    last_rel = path.nodes[-2] # e.g. nodes = [e1, r1, e2] -> last_rel is r1
                    for seq in c.params.get("sequences", []):
                        if len(seq) == 2 and seq[0] == last_rel and seq[1] == rel:
                            return False
                        # Support for longer sequences could be added here

            elif c.type == ConstraintType.TRANSITIVE:
                # If A-rel-B and B-rel-C, then A-rel-C must also exist.
                # Prunes paths where a transitive shortcut is missing, enforcing consistency.
                if rel in c.params.get("relations", []):
                    if path and len(path.nodes) >= 3:
                        # path.nodes = [..., start_node, r_prev, u]
                        # Current step is u -[rel]-> v
                        r_prev = path.nodes[-2]
                        start_node = path.nodes[-3]
                        if r_prev == rel:
                            # Check if start_node -[rel]-> v exists
                            neighbors_start = self.adapter.get_neighbors(start_node)
                            if not any(n.target_id == v and n.relation_type == rel for n in neighbors_start):
                                return False

        # 2. Contradiction Checking (Internal Path)
        if self.contradiction_engine is not None and path is not None:
            # Create a 'hypothetical' path to scan
            # We don't want to copy everything, just nodes for the check
            # We can optimize this by checking the new edge against existing edges in path
            nodes = path.nodes
            path_edges = []
            for i in range(0, len(nodes) - 2, 2):
                path_edges.append((nodes[i], nodes[i+1], nodes[i+2]))
            
            from core.contradiction_engine import relations_contradict
            for (u_prev, rel_prev, v_prev) in path_edges:
                # Check if current (u, rel, v) contradicts any (u_prev, rel_prev, v_prev)
                if {u, v} == {u_prev, v_prev}:
                    if relations_contradict(rel, rel_prev):
                        return False

        return True

    def validate_path(self, path: TraversalPath) -> bool:
        """
        Full path validation. Checks all hops.
        """
        # 1. Check integrity constraints (step-by-step)
        nodes = path.nodes
        # nodes is [e1, r1, e2, r2, e3, ...]
        for i in range(0, len(nodes) - 2, 2):
            u = nodes[i]
            rel = nodes[i+1]
            v = nodes[i+2]
            if not self.validate_step(u, rel, v):
                return False
        
        # 2. Check for internal contradictions (Phase 36)
        if self.contradiction_engine is not None:
            recs = self.contradiction_engine.detect_contradictions_in_path(path)
            if recs:
                return False

        return True

    def _get_type(self, node_id: str) -> str:
        if node_id in self._type_cache:
            return self._type_cache[node_id]
        
        ent = self.adapter.get_entity(node_id)
        t = ent.type if ent else "unknown"
        self._type_cache[node_id] = t
        return t


# ---------------------------------------------------------------------------
# Default Global Guardrails
# ---------------------------------------------------------------------------

DEFAULT_GUARDRAILS: List[IntegrityConstraint] = [
    IntegrityConstraint(
        ConstraintType.TYPE_RESTRICTION,
        domain="biology",
        params={"relation": "INHIBITS", "source_type": "chemical", "target_type": "protein"},
        note="Only chemicals can inhibit proteins in this domain model."
    ),
    IntegrityConstraint(
        ConstraintType.TYPE_RESTRICTION,
        domain="general",
        params={"relation": "BORN_IN", "source_type": "person", "target_type": "location"},
        note="Only people can be born in locations."
    ),
    IntegrityConstraint(
        ConstraintType.MUST_NOT_HAVE,
        domain="general",
        params={"type": "person", "illegal_props": ["is_place"]},
        note="A person entity cannot also be a geographical place."
    )
]

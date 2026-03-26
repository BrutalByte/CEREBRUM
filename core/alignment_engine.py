"""
Entity Alignment Engine for CEREBRUM.

Handles the mapping and resolution of entities across multiple GraphAdapters.
Provides an explicit alignment index to unify disparate entity IDs into
a single canonical reference.
"""
from typing import Dict, List, Optional, Set, Tuple
from core.graph_adapter import Entity


class AlignmentIndex:
    """
    Maintains mappings between (adapter_name, entity_id) and a canonical_id.
    
    Attributes:
        canonical_map (Dict[Tuple[str, str], str]): (adapter, id) -> canonical_id
        reverse_map (Dict[str, Set[Tuple[str, str]]]): canonical_id -> {(adapter, id), ...}
    """

    def __init__(self):
        self.canonical_map: Dict[Tuple[str, str], str] = {}
        self.reverse_map: Dict[str, Set[Tuple[str, str]]] = {}

    def add_alignment(self, adapter_a: str, id_a: str, adapter_b: str, id_b: str, canonical_id: Optional[str] = None):
        """
        Register that id_a in adapter_a and id_b in adapter_b are the same entity.
        If canonical_id is not provided, it defaults to adapter_a:id_a.
        """
        # 1. Determine or create canonical_id
        if not canonical_id:
            # Check if either already has a canonical_id
            can_a = self.get_canonical(adapter_a, id_a)
            can_b = self.get_canonical(adapter_b, id_b)
            canonical_id = can_a or can_b or f"{adapter_a}:{id_a}"
            
        # 2. Update mappings
        for adapter, eid in [(adapter_a, id_a), (adapter_b, id_b)]:
            self.canonical_map[(adapter, eid)] = canonical_id
            self.reverse_map.setdefault(canonical_id, set()).add((adapter, eid))

    def get_canonical(self, adapter_name: str, entity_id: str) -> Optional[str]:
        """Return the canonical ID for a specific (adapter, entity) pair."""
        return self.canonical_map.get((adapter_name, entity_id))

    def get_aliases(self, canonical_id: str) -> Set[Tuple[str, str]]:
        """Return all (adapter, id) pairs associated with this canonical ID."""
        return self.reverse_map.get(canonical_id, set())

    def resolve_aliases(self, adapter_name: str, entity_id: str) -> Set[Tuple[str, str]]:
        """
        Given a specific entity, find all its aliases in other adapters
        (including itself).
        """
        can_id = self.get_canonical(adapter_name, entity_id)
        if not can_id:
            return {(adapter_name, entity_id)}
        return self.get_aliases(can_id)


def auto_align_by_label(adapters: Dict[str, "GraphAdapter"], threshold: float = 1.0) -> AlignmentIndex:
    """
    Automatically align entities across adapters if their labels match.
    Currently uses exact match (threshold=1.0). 
    Future: Use fuzzy matching or embedding-based alignment.
    """
    index = AlignmentIndex()
    adapter_names = list(adapters.keys())
    
    # This is O(N^2) over the sum of entities, but can be optimized with an 
    # inverted label index.
    label_to_entities: Dict[str, List[Tuple[str, str]]] = {}
    
    for name, adapter in adapters.items():
        try:
            # Only works if adapter supports get_all_entities
            for entity in adapter.get_all_entities():
                label_to_entities.setdefault(entity.label, []).append((name, entity.id))
        except NotImplementedError:
            continue
            
    for label, entities in label_to_entities.items():
        if len(entities) > 1:
            # Align all entities with the same label to the first one's canonical ID
            first_adapter, first_id = entities[0]
            for other_adapter, other_id in entities[1:]:
                index.add_alignment(first_adapter, first_id, other_adapter, other_id)
                
    return index

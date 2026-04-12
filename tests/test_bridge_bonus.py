import numpy as np
from typing import Optional
from core.attention_engine import CSAEngine

from core.graph_adapter import GraphAdapter

class MockAdapter(GraphAdapter):
    def __init__(self, communities, embeddings):
        self.community_map = communities
        self.embeddings = embeddings
    def get_community(self, node: str) -> int:
        return self.community_map.get(node, -1)
    def get_embedding(self, node: str) -> Optional[np.ndarray]:
        return self.embeddings.get(node)
    def add_edge(self, u, v, relation, confidence=1.0, provenance="", synthetic=False):
        pass
    def get_neighbors(self, *args, **kwargs) -> list: return []
    def find_similar(self, *args, **kwargs) -> list: return []
    def find_entities(self, *args, **kwargs) -> list: return []
    def get_entity(self, entity_id: str): return None
    def to_networkx(self): return None

def test_bridge_bonus_influence():
    # Simple setup
    communities = {"A": 1, "B": 2}
    embeddings = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0])}
    adapter = MockAdapter(communities, embeddings)
    
    # 1. No bonus
    engine_no_bonus = CSAEngine(adapter=adapter, gamma=1.0)
    w_no_bonus = engine_no_bonus.compute_weight("A", "B", hop=1, edge_type="treats")
    
    # 2. Positive bonus for "treats"
    # gamma * etw = 1.0 * 0.5 = 0.5
    engine_bonus = CSAEngine(
        adapter=adapter,
        gamma=1.0,
        edge_type_weights={"treats": 0.5}
    )
    w_bonus = engine_bonus.compute_weight("A", "B", hop=1, edge_type="treats")
    
    print(f"Weight without bonus: {w_no_bonus}")
    print(f"Weight with bonus: {w_bonus}")
    
    assert w_bonus > w_no_bonus
    
    # 3. Dynamic override
    w_dynamic = engine_bonus.compute_weight(
        "A", "B", hop=1, edge_type="treats", edge_type_weights={"treats": 2.0}
    )
    print(f"Weight with dynamic bonus: {w_dynamic}")
    assert w_dynamic > w_bonus

if __name__ == "__main__":
    test_bridge_bonus_influence()


import numpy as np
import pytest
from core.attention_engine import CSAEngine

def test_bridge_bonus_influence():
    # Simple setup
    communities = {"A": 1, "B": 2}
    embeddings = {"A": np.array([1.0, 0.0]), "B": np.array([1.0, 0.0])}
    
    # 1. No bonus
    engine_no_bonus = CSAEngine(communities=communities, embeddings=embeddings, gamma=1.0)
    w_no_bonus = engine_no_bonus.compute_weight("A", "B", hop=1, edge_type="treats")
    
    # 2. Positive bonus for "treats"
    # gamma * etw = 1.0 * 0.5 = 0.5
    engine_bonus = CSAEngine(
        communities=communities, 
        embeddings=embeddings, 
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




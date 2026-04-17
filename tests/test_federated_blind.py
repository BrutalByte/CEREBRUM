
import numpy as np
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from adapters.federated_adapter import FederatedAdapter
from core.holographic_index import build_signatures

def test_federated_SynapticBridge_discovery():
    """
    Verify that FederatedAdapter can discover semantically similar 
    entities in remote adapters via SynapticBridge edges.
    """
    # 1. Setup Graph A (Local)
    adapter_a = NetworkXAdapter(nx.DiGraph())
    adapter_a._G.add_edge("text:Apple", "text:Company", relation="is_a")
    adapter_a.embeddings = {"text:Apple": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    adapter_a.community_map = {"text:Apple": 0, "text:Company": 0}
    
    # 2. Setup Graph B (Remote)
    adapter_b = NetworkXAdapter(nx.DiGraph())
    adapter_b._G.add_edge("text:Fruit", "text:Food", relation="is_a")
    # 'Fruit' is semantically close to 'Apple'
    adapter_b.embeddings = {"text:Fruit": np.array([0.9, 0.1, 0.0], dtype=np.float32)}
    adapter_b.community_map = {"text:Fruit": 1, "text:Food": 1}
    
    # 3. Create Federated Adapter
    fed = FederatedAdapter({"local": adapter_a, "remote": adapter_b})
    
    # 4. Initialize Holograms
    sigs_b = build_signatures(adapter_b, adapter_b.community_map, adapter_b.embeddings)
    fed.hologram_index.add_adapter_signatures("remote", sigs_b)
    
    # 5. Verify Discovery
    emb_apple = adapter_a.get_embedding("text:Apple")
    edges = fed.get_neighbors("text:Apple", context_embedding=emb_apple)
    
    targets = [e.target_id for e in edges]
    relations = [e.relation_type for e in edges]
    
    assert "text:Company" in targets
    assert "text:Fruit" in targets 
    assert "SynapticBridge" in relations
    
    # Verify weight/confidence is high for close match
    SynapticBridge = next(e for e in edges if e.target_id == "text:Fruit")
    assert SynapticBridge.weight >= 0.7
    assert SynapticBridge.provenance == "hologram:remote"

def test_federated_SynapticBridge_no_match():
    """Verify that dissimilar entities do not trigger SynapticBridges."""
    adapter_a = NetworkXAdapter(nx.DiGraph())
    adapter_a._G.add_node("text:Tesla")
    adapter_a.embeddings = {"text:Tesla": np.array([0.0, 0.0, 1.0], dtype=np.float32)}
    adapter_a.community_map = {"text:Tesla": 0}
    
    adapter_b = NetworkXAdapter(nx.DiGraph())
    adapter_b._G.add_node("text:Fruit")
    adapter_b.embeddings = {"text:Fruit": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    adapter_b.community_map = {"text:Fruit": 1}
    
    fed = FederatedAdapter({"local": adapter_a, "remote": adapter_b})
    sigs_b = build_signatures(adapter_b, adapter_b.community_map, adapter_b.embeddings)
    fed.hologram_index.add_adapter_signatures("remote", sigs_b)
    
    # 'Tesla' (0,0,1) is orthogonal to 'Fruit' (1,0,0)
    emb_tesla = adapter_a.get_embedding("text:Tesla")
    edges = fed.get_neighbors("text:Tesla", context_embedding=emb_tesla)
    
    relations = [e.relation_type for e in edges]
    assert "SynapticBridge" not in relations

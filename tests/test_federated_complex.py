
import pytest
from fastapi.testclient import TestClient
import numpy as np
import networkx as nx
from api.server import create_app
from adapters.networkx_adapter import NetworkXAdapter
from adapters.remote_adapter import RemoteCerebrumAdapter
from adapters.federated_adapter import FederatedAdapter
from core.embedding_engine import RandomEngine
from core.alignment_engine import AlignmentIndex
from reasoning.distributed_traversal import DistributedBeamTraversal

def test_distributed_boundary_delegation():
    # 1. Setup Node A (Primary)
    # A -> B
    ga = nx.Graph()
    ga.add_edge("A", "B", relation="LOCAL_REL")
    adapter_a = NetworkXAdapter(ga)
    adapter_a.community_map = {"A": 0, "B": 0}
    adapter_a.embeddings = {"A": np.random.rand(64), "B": np.random.rand(64)}
    
    # 2. Setup Node B (Secondary)
    # B -> C -> D
    gb = nx.Graph()
    gb.add_edge("B", "C", relation="REMOTE_REL")
    gb.add_edge("C", "D", relation="DEEP_REL")
    adapter_b = NetworkXAdapter(gb)
    adapter_b.community_map = {"B": 1, "C": 1, "D": 1}
    adapter_b.embeddings = {"B": np.random.rand(64), "C": np.random.rand(64), "D": np.random.rand(64)}
    
    # 3. Create FastAPI app for Node B
    app_b = create_app(adapter_b, RandomEngine(dim=64), community_map=adapter_b.community_map)
    
    with TestClient(app_b) as client_b:
        # Mock RemoteCerebrumAdapter to call client_b
        class MockRemoteAdapter(RemoteCerebrumAdapter):
            def get_reasoning_branches(self, seed_id, context_embedding=None, **kwargs):
                payload = {
                    "seed_id": seed_id, 
                    "context_embedding": context_embedding.tolist() if context_embedding is not None else None,
                    **kwargs
                }
                resp = client_b.post("/v1/traverse", json=payload, headers={"X-API-Key": "dev-secret"})
                return resp.json()["branches"]
            def get_entity(self, eid):
                return adapter_b.get_entity(eid)
            def get_neighbors(self, eid, edge_types=None, max_neighbors=50, context_embedding=None):
                return adapter_b.get_neighbors(eid, edge_types, max_neighbors)
            def get_community(self, eid):
                return adapter_b.get_community(eid)
            def get_embedding(self, eid):
                return adapter_b.get_embedding(eid)
            def validate_connection(self):
                return True

        remote_b = MockRemoteAdapter("http://node-b", token="dev-secret")
        
        # 4. Federated View on Node A
        alignment = AlignmentIndex()
        alignment.add_alignment("node_a", "B", "node_b", "B") # B exists in both
        
        fed = FederatedAdapter({"node_a": adapter_a, "node_b": remote_b}, alignment=alignment)
        
        # 5. Run Distributed Traversal from A
        from core.attention_engine import CSAEngine
        csa = CSAEngine(fed)
        traversal = DistributedBeamTraversal(fed, csa_engine=csa, max_hop=3, beam_width=5)
        paths = traversal.traverse(["A"])
        
        # 6. Verify results
        # Should find path A -> LOCAL_REL -> B -> REMOTE_REL -> C -> DEEP_REL -> D
        path_nodes = [p.nodes for p in paths]
        found_deep = False
        for p in path_nodes:
            if "D" in p:
                found_deep = True
                assert "LOCAL_REL" in p
                assert "REMOTE_REL" in p
                assert "DEEP_REL" in p
        
        assert found_deep, "Should have delegated to node_b to find entity D"

if __name__ == "__main__":
    pytest.main([__file__])

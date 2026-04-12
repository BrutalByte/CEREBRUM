import pytest
import networkx as nx
from adapters.networkx_adapter import NetworkXAdapter
from adapters.federated_adapter import FederatedAdapter
from core.embedding_engine import RandomEngine
from reasoning.consensus_hierarchy_engine import ConsensusHierarchyEngine
from reasoning.consensus_scorer import ConsensusLevel

@pytest.fixture
def federated_mach_setup():
    # 1. Create two sub-adapters
    G1 = nx.Graph()
    G1.add_node("A", label="Seed", type="entity")
    G1.add_node("C", label="Target", type="entity")
    G1.add_edge("A", "C", relation="related_to", weight=1.0)
    
    G2 = nx.Graph()
    G2.add_node("A", label="Seed", type="entity")
    G2.add_node("C", label="Target", type="entity")
    G2.add_edge("A", "C", relation="confirmed_by_other", weight=1.0)
    
    a1 = NetworkXAdapter(G1)
    a2 = NetworkXAdapter(G2)
    
    # Setup metadata for CSA
    a1.community_map = {"A": 0, "C": 0}
    a2.community_map = {"A": 0, "C": 0}
    
    eng = RandomEngine(dim=16)
    emb = eng.encode_entities({"A": "Seed", "C": "Target"})
    a1.embeddings = emb
    a2.embeddings = emb
    
    a1._csa_metadata = {"distances": {0: {0: 0.0}}, "adjacent_pairs": set()}
    a2._csa_metadata = {"distances": {0: {0: 0.0}}, "adjacent_pairs": set()}

    # 2. Create FederatedAdapter
    fed = FederatedAdapter(adapters={
        "node_1": a1,
        "node_2": a2
    })
    
    return fed

@pytest.mark.asyncio
async def test_mach_l2_federated_escalation(federated_mach_setup):
    fed = federated_mach_setup
    # We use agent_name="node_1" so node_2 is the 'other' node
    che = ConsensusHierarchyEngine(fed, agent_name="node_1")
    
    # Run query with L2 escalation requested
    resp = await che.query_with_consensus(
        query="A",
        seeds=["A"],
        level=ConsensusLevel.L2_FEDERATED,
        strategies=["standard"],
        max_hop=1
    )
    
    results = resp["results"]
    assert len(results) > 0
    
    # The path A -> C should be verified by node_2 because node_2 HAS entity C
    top = results[0]
    assert top.level == ConsensusLevel.L2_FEDERATED
    assert "federated_verification" in top.agents
    assert top.consensus_score > 0.5

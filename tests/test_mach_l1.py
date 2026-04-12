import pytest
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from reasoning.multi_strategy_consensus import MultiStrategyConsensus
from reasoning.consensus_scorer import ConsensusLevel

@pytest.fixture
def mach_setup():
    import networkx as nx
    G = nx.Graph()
    G.add_node("A", label="Seed", type="entity")
    G.add_node("C", label="Target", type="entity")
    G.add_edge("A", "C", relation="related_to", weight=1.0)
    
    adapter = NetworkXAdapter(G)
    
    # Needs community map for CSA
    adapter.community_map = {"A": 0, "C": 0}
    
    # Needs embeddings
    eng = RandomEngine(dim=16)
    adapter.embeddings = eng.encode_entities({"A": "Seed", "C": "Target"})
    
    # Mock CSA metadata
    from core.structural_encoder import build_community_distance_matrix
    import networkx as nx
    G = adapter.to_networkx()
    adapter._csa_metadata = {
        "distances": {0: {0: 0.0}},
        "adjacent_pairs": set()
    }
    
    return adapter

@pytest.mark.asyncio
async def test_mach_l1_multi_strategy(mach_setup):
    adapter = mach_setup
    ce = MultiStrategyConsensus(adapter)
    
    # Run consensus query
    results, explored = await ce.run_consensus_query(
        seeds=["A"],
        strategies=["standard", "bayesian"],
        max_hop=1
    )
    
    assert len(results) > 0
    assert explored >= 2 # At least one path per strategy
    
    # Check first result
    top = results[0]
    assert top.level == ConsensusLevel.L1_LOCAL
    assert len(top.agents) >= 1
    assert "local_node_standard" in top.agents or "local_node_bayesian" in top.agents

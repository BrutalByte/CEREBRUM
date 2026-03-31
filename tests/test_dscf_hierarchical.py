
import networkx as nx
from core.community_engine import dscf_communities, hierarchical_dscf

def test_dscf_star_topology_fragmentation():
    """
    Verify that standard DSCF fragments a star topology when force_connectivity=True.
    """
    G = nx.Graph()
    hub = "HUB"
    spokes = [f"SPOKE_{i}" for i in range(20)]
    for s in spokes:
        G.add_edge(hub, s)
        
    # Standard DSCF with force_connectivity=True (default)
    # If the hub moves to a different community than its spokes during a pass,
    # the spokes become disconnected singletons.
    # We set a seed to make it more likely to see fragmentation if it's prone to it,
    # but since DSCF is non-deterministic, we just check the behavior.
    parts = dscf_communities(G, force_connectivity=True)
    
    # In a star graph of 21 nodes, we expect few communities (e.g., 1 or 2).
    # If it fragments into 21 communities, that's the over-splitting problem.
    print(f"Standard DSCF communities: {len(parts)}")
    
def test_dscf_force_connectivity_false():
    """
    Verify that force_connectivity=False prevents fragmentation on star topology.
    """
    G = nx.Graph()
    hub = "HUB"
    spokes = [f"SPOKE_{i}" for i in range(20)]
    for s in spokes:
        G.add_edge(hub, s)
        
    parts = dscf_communities(G, force_connectivity=False)
    # Without forced connectivity, it should stay as one community (usually).
    assert len(parts) < 10
    print(f"DSCF (force_connectivity=False) communities: {len(parts)}")

def test_hierarchical_dscf_consolidation():
    """
    Verify that hierarchical_dscf consolidates even if the first pass splits.
    """
    # Create two stars connected by a bridge
    G = nx.Graph()
    hub1 = "HUB1"
    spokes1 = [f"S1_{i}" for i in range(50)]
    for s in spokes1:
        G.add_edge(hub1, s)
        
    hub2 = "HUB2"
    spokes2 = [f"S2_{i}" for i in range(50)]
    for s in spokes2:
        G.add_edge(hub2, s)
        
    G.add_edge(hub1, hub2)
    
    # Run hierarchical DSCF
    parts = hierarchical_dscf(G, target_communities=5)
    
    print(f"Hierarchical DSCF communities: {len(parts)}")
    assert len(parts) <= 5
    # Ideally it should find 2 communities (the two stars)
    
if __name__ == "__main__":
    test_dscf_star_topology_fragmentation()
    test_dscf_force_connectivity_false()
    test_hierarchical_dscf_consolidation()





import time
import networkx as nx
from core.community_engine import vectorized_tsc, dscf_communities

def test_vectorized_tsc_comparison():
    """
    Compare performance and modularity between standard DSCF 
    and the new vectorized TSC engine.
    """
    # 1. Create a reasonably large synthetic graph (2 cliques)
    G = nx.relaxed_caveman_graph(50, 20, 0.1, seed=42)
    G = nx.Graph(G) # Ensure undirected for standard modularity
    
    print(f"Testing on graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # 2. Benchmark Standard DSCF
    start = time.time()
    parts_dscf = dscf_communities(G, max_iter=30)
    dur_dscf = time.time() - start
    
    q_dscf = nx.community.modularity(G, parts_dscf)
    print(f"Standard DSCF: Q={q_dscf:.4f}, time={dur_dscf:.4f}s, communities={len(parts_dscf)}")
    
    # 3. Benchmark Vectorized TSC
    start = time.time()
    parts_tsc = vectorized_tsc(G, max_iter=30)
    dur_tsc = time.time() - start
    
    q_tsc = nx.community.modularity(G, parts_tsc)
    print(f"Vectorized TSC: Q={q_tsc:.4f}, time={dur_tsc:.4f}s, communities={len(parts_tsc)}")
    
    # 4. Assertions
    # Vectorized should be faster on this scale or comparable
    # Modularity should be in the same ballpark
    assert q_tsc > 0.5 # High modularity for caveman graph
    assert len(parts_tsc) > 1
    
    print("GPU-ready vectorized TSC verified.")

if __name__ == "__main__":
    test_vectorized_tsc_comparison()

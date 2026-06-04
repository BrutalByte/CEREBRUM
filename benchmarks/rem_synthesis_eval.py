from typing import Set
import networkx as nx
import numpy as np
import time
from adapters.networkx_adapter import NetworkXAdapter
from core.rem_engine import REMEngine
from core.attention_engine import CSAEngine
from core.cerebrum import CerebrumGraph

def run_rem_synthesis_benchmark():
    print("=== REM Synthesis Evaluation (IKGWQ-S) ===")
    
    # 1. Create a "disconnected component" graph where reasoning requires a bridge
    # Component 1: A -> B -> C
    # Component 2: D -> E -> F
    # Similarity: C is very similar to D semantically.
    
    G = nx.DiGraph()
    G.add_edge("A", "B", relation="related")
    G.add_edge("B", "C", relation="related")
    G.add_edge("D", "E", relation="related")
    G.add_edge("E", "F", relation="related")
    
    adapter = NetworkXAdapter(G)
    
    # Set embeddings such that C and D are near each other
    emb_size = 384
    common_vec = np.random.rand(emb_size)
    noise = np.random.rand(emb_size) * 0.001 # Even less noise
    
    adapter.embeddings["A"] = np.random.rand(emb_size)
    adapter.embeddings["B"] = np.random.rand(emb_size)
    adapter.embeddings["C"] = common_vec
    adapter.embeddings["D"] = common_vec + noise
    adapter.embeddings["E"] = np.random.rand(emb_size)
    adapter.embeddings["F"] = np.random.rand(emb_size)

    # Add confidence to edges so they don't get pruned if we ran a full cycle
    for u, v in G.edges():
        G[u][v]['confidence'] = 1.0
        G[u][v]['relation'] = 'related'

    # 2. Before REM: Can we find a path from A to F?
    cerebrum = CerebrumGraph(adapter)
    cerebrum.build()
    
    print("Pre-REM Traversal (A -> F)...")
    results_pre = cerebrum.query(["A"], beam_width=10, max_hop=10)
    print(f"Pre-REM Results: {len(results_pre)}")
    for res in results_pre[:3]:
        print(f"  Result: {res.entity_id} (score {res.score:.4f})")
    has_path_pre = any(res.entity_id == "F" for res in results_pre)
    print(f"Path A->F found pre-REM: {has_path_pre}")
    
    # 3. Run REM Synthesis
    print("Running REM Synthesis...")
    # Use a lower threshold for debugging to ensure we get something
    rem = REMEngine(adapter, synthesis_similarity_threshold=0.7, cross_component_similarity_threshold=0.7)
    report = rem.run(dry_run=False)
    print(f"REM Report: {report}")
    print(f"Synthesized edges: {report.synthesized_edge_list}")
    
    # 5. Verify the edge exists in the adapter
    print(f"Edge C->D in adapter: {adapter._G.has_edge('C', 'D') or adapter._G.has_edge('D', 'C')}")
    if adapter._G.has_edge('C', 'D'):
        print(f"  Attributes C->D: {adapter._G.get_edge_data('C', 'D')}")
    if adapter._G.has_edge('D', 'C'):
        print(f"  Attributes D->C: {adapter._G.get_edge_data('D', 'C')}")

    # 4. After REM: Can we find the path now?
    # Re-initialize Cerebrum to pick up new edges and community map
    cerebrum_post = CerebrumGraph(adapter)
    cerebrum_post.build()
    
    print("Post-REM Traversal (A -> F)...")
    results_post = cerebrum_post.query(["A"], beam_width=10, max_hop=10)
    print(f"Post-REM Results: {len(results_post)}")
    for res in results_post[:5]:
        print(f"  Result: {res.entity_id} (score {res.score:.4f})")
    has_path_post = any(res.entity_id == "F" for res in results_post)
    print(f"Path A->F found post-REM: {has_path_post}")
    
    if has_path_post and not has_path_pre:
        print("SUCCESS: REM Synthesis successfully bridged the disconnected components!")
    else:
        print("FAILURE: Bridge not found or already existed.")

if __name__ == "__main__":
    run_rem_synthesis_benchmark()

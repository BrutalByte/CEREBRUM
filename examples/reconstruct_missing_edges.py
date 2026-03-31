"""
Example: Reconstructing Missing Edges without "Cheating".

This example demonstrates CEREBRUM's ability to recover missing edges in an 
incomplete graph using graph-native structural signals and embedding 
similarities, rather than relying on LLM-style "hallucination" or 
pre-trained memory.

We simulate a "hidden" fact, remove it from the graph, and use the 
REMEngine and IncompletenessRepairEngine to hypothesize its existence
while maintaining a verifiable "path_confidence" score.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.rem_engine import REMEngine
from core.repair_engine import IncompletenessRepairEngine
from reasoning.traversal import BeamTraversal
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs

def main():
    print("=" * 70)
    print(" CEREBRUM: Reconstructing Missing Edges (Anti-Hallucination)")
    print("=" * 70)

    # 1. Setup a small graph with a missing link
    # Fact: Newton influenced Einstein.
    # Shared Context: Both influenced Faraday.
    G = nx.Graph()
    G.add_edge("newton", "faraday", relation="INFLUENCED")
    G.add_edge("einstein", "faraday", relation="INFLUENCED") # This is the "missing" link's witness
    G.add_edge("einstein", "bohr", relation="COLLABORATED")
    G.add_edge("faraday", "maxwell", relation="INFLUENCED")
    
    # Target edge we will "forget": newton -> einstein
    # In reality, Newton influenced Einstein's work indirectly.
    
    print(f"Original Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Missing Fact: Newton -> Einstein (INFLUENCED)")

    # 2. Setup CEREBRUM components
    adapter = NetworkXAdapter(G)
    
    # Use random embeddings (no pre-trained memory!) to show structural recovery
    # In a real system, SentenceEngine would provide semantic signal.
    embedding_engine = RandomEngine(dim=32)
    labels = {n: n for n in G.nodes()}
    adapter.embeddings = embedding_engine.encode_entities(labels)
    
    # Communities (Attention Heads)
    # For this tiny graph, we'll manually assign or use a simple split
    adapter.community_map = {"newton": 0, "faraday": 0, "einstein": 1, "bohr": 1, "maxwell": 0}
    
    # 3. Try to query WITHOUT reconstruction
    print("\n[Trial 1] Querying: 'Who did Newton influence?' (Without Repair)")
    dist = build_community_distance_matrix(G, adapter.community_map)
    adj  = adjacent_community_pairs(G, adapter.community_map)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)
    
    traversal = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=5)
    paths = traversal.traverse(["newton"])
    
    results = [p.tail for p in paths if p.hop_depth > 0]
    print(f"  Directly influenced: {results}")
    print(f"  Is 'einstein' found? {'Yes' if 'einstein' in results else 'No'}")

    # 4. Use REMEngine for Global Synthesis (Structural Proximity)
    print("\n[Trial 2] Running REMEngine (Global Synthesis)...")
    rem = REMEngine(adapter=adapter, synthesis_similarity_threshold=0.5)
    report = rem.run(dry_run=False) # Mutates the adapter's graph
    
    print(f"  REM Report: {report}")
    for u, v, rel in report.synthesized_edge_list:
        print(f"  Hypothesized Link: {u} --({rel})--> {v}")

    # 5. Use IncompletenessRepairEngine for Query-Guided Repair
    print("\n[Trial 3] Running IncompletenessRepairEngine (Query-Guided)...")
    # We re-run traversal to get "dead-ends"
    paths = traversal.traverse(["newton"])
    repair = IncompletenessRepairEngine(adapter, min_path_score=0.01)
    G_repaired, n_synth = repair.repair(paths, G)
    
    print(f"  Repair Engine added {n_synth} edges.")
    for u, v, d in G_repaired.edges(data=True):
        if d.get("synthesized"):
            print(f"  Repaired Link: {u} --({d['relation']})--> {v} (conf={d['confidence']:.2f})")

    # 6. Final Verification
    print("\n[Conclusion]")
    print("CEREBRUM uses shared neighbors (Faraday) and structural proximity")
    print("to hypothesize missing links. It doesn't 'hallucinate' facts from")
    print("hidden weights, but proposes 'research hypotheses' based on")
    print("observable graph topology. Every proposed edge is verifiable.")
    print("=" * 70)

if __name__ == "__main__":
    main()

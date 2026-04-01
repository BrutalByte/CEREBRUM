import os
import csv
import random

# CEREBRUM: GPU Stress Test Graph Generator (v1.4.0)
# Creates a large synthetic graph to benchmark GPU-accelerated embeddings and community detection.

NODES_COUNT = 5000
EDGES_COUNT = 15000
OUTPUT_FILE = 'tests/fixtures/stress_test_5k.csv'

def generate():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 1. Create Nodes (varying lengths for embedding pressure)
    prefix = ["entity", "concept", "signal", "node", "object", "agent", "process", "state", "event", "property"]
    suffix = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    
    nodes = []
    for i in range(NODES_COUNT):
        p = random.choice(prefix)
        s = random.choice(suffix)
        # Unique names to force new embeddings
        nodes.append(f"{p}_{s}_{i}")
        
    # 2. Create Edges
    relations = ["RELATES_TO", "CONNECTS_TO", "INFLUENCES", "PART_OF", "CAUSES", "SIMILAR_TO"]
    
    print(f"Generating {NODES_COUNT} nodes and {EDGES_COUNT} edges...")
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relation"])
        
        for _ in range(EDGES_COUNT):
            u = random.choice(nodes)
            v = random.choice(nodes)
            while u == v:
                v = random.choice(nodes)
            
            rel = random.choice(relations)
            writer.writerow([u, v, rel])
            
    print(f"✅ Success: Saved to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

if __name__ == "__main__":
    generate()

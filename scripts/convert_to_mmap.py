"""
Converter Utility: Serializes CEREBRUM graph state to binary A-File/E-Block format.
Optimized for mmap memory alignment (32-byte record padding).
"""
import struct
import numpy as np
import networkx as nx
from pathlib import Path

# Spec: 32-byte padded A-File records
# 8 (id) + 4 (deg) + 8 (offset) + 2 (comm) + 10 (reserved) = 32 bytes
A_FILE_FORMAT = "<QIQH10x" 

# Spec: 12-byte E-Block records
# 8 (target_id) + 2 (rel_id) + 2 (conf as float16) = 12 bytes
E_BLOCK_FORMAT = "<QH e"

def convert_to_mmap(adapter, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    a_file = output_path / "graph.a"
    e_block = output_path / "graph.e"
    
    G = adapter.to_networkx()
    nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    with open(e_block, "wb") as f_e, open(a_file, "wb") as f_a:
        current_offset = 0
        for i, node in enumerate(nodes):
            neighbors = list(G[node])
            degree = len(neighbors)
            community = G.nodes[node].get("community", 0)
            
            # Write A-File record
            # Padding: 10 bytes added by '10x' in struct format
            f_a.write(struct.pack(A_FILE_FORMAT, i, degree, current_offset, community))
            
            # Write E-Block records
            for neighbor in neighbors:
                data = G[node][neighbor]
                target_idx = node_to_idx[neighbor]
                rel_id = hash(data.get("relation", "RELATED_TO")) % 65535
                conf = float(data.get("confidence", 1.0))
                
                f_e.write(struct.pack(E_BLOCK_FORMAT, target_idx, rel_id, conf))
                current_offset += 12
                
    print(f"Graph converted successfully.")
    print(f"A-File: {a_file} ({a_file.stat().st_size} bytes)")
    print(f"E-Block: {e_block} ({e_block.stat().st_size} bytes)")

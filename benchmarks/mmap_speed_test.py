import time
import os
import shutil
import tempfile
import networkx as nx
import numpy as np
from core.cerebrum import CerebrumGraph
from adapters.networkx_adapter import NetworkXAdapter
from adapters.mmap_adapter import MmapAdapter

def benchmark_speed_decrease():
    print("Generating synthetic graph for benchmark...")
    # Create a larger graph: 10,000 nodes, ~100,000 edges
    G = nx.fast_gnp_random_graph(10000, 0.002, seed=42)
    for u, v in G.edges():
        G[u][v]['relation'] = "LINKED"
        G[u][v]['confidence'] = 0.9
    
    adapter = NetworkXAdapter(G)
    
    # 1. In-Memory Baseline
    print("\n--- Phase 1: In-Memory Baseline ---")
    graph_mem = CerebrumGraph(adapter)
    graph_mem.build()
    
    seeds = [str(i) for i in range(10)]
    
    # Warmup
    graph_mem.query(seeds, top_k=5, max_hop=2)
    
    start_time = time.perf_counter()
    for _ in range(10):
        graph_mem.query(seeds, top_k=5, max_hop=2)
    mem_duration = (time.perf_counter() - start_time) / 10
    print(f"In-Memory Avg Query Time: {mem_duration:.4f}s")

    # 2. Mmap (Disk-Backed)
    print("\n--- Phase 2: Mmap (Disk-Backed) ---")
    # Force spill by setting a tiny RAM limit
    graph_mmap = CerebrumGraph(adapter, max_ram_gb=0.001)
    graph_mmap.build()
    
    assert isinstance(graph_mmap.adapter, MmapAdapter)
    
    # Warmup
    graph_mmap.query(seeds, top_k=5, max_hop=2)
    
    start_time = time.perf_counter()
    for _ in range(5):
        graph_mmap.query(seeds, top_k=5, max_hop=2)
    mmap_duration = (time.perf_counter() - start_time) / 5
    print(f"Mmap Avg Query Time: {mmap_duration:.4f}s")

    # Results
    decrease = (mmap_duration / mem_duration) if mem_duration > 0 else 0
    print(f"\nLatency Increase: {decrease:.2f}x")
    print(f"Speed Decrease: {((mmap_duration - mem_duration) / mem_duration * 100):.1f}%")

    # Cleanup
    if isinstance(graph_mmap.adapter, MmapAdapter):
        data_dir = graph_mmap.adapter.data_dir
        graph_mmap.adapter.close()
        shutil.rmtree(data_dir)

if __name__ == "__main__":
    benchmark_speed_decrease()

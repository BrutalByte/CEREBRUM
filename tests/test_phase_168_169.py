from typing import Set
import pytest
import os
import tempfile
import shutil
from core.cerebrum import CerebrumGraph
from adapters.networkx_adapter import NetworkXAdapter
from adapters.mmap_adapter import MmapAdapter
import networkx as nx

def test_memory_governor_spill():
    # 1. Create a toy graph
    G = nx.Graph()
    G.add_edge("A", "B", relation="FRIEND", confidence=0.9)
    G.add_edge("B", "C", relation="COLLEAGUE", confidence=0.8)
    adapter = NetworkXAdapter(G)
    
    # 2. Create CerebrumGraph with a very low RAM limit to force spill
    # Set max_ram_gb to a very small value (e.g., 0.001 GB = 1 MB)
    # Since the process RSS will definitely be > 1 MB, it should trigger spill.
    graph = CerebrumGraph(adapter, max_ram_gb=0.001, max_vram_gb=1.0)
    
    # 3. Build the graph
    # This should trigger _spill_to_mmap() at the start of build()
    graph.build()
    
    # 4. Verify that the adapter has been swapped to MmapAdapter
    assert isinstance(graph.adapter, MmapAdapter)
    
    # 5. Verify that reasoning still works
    answers = graph.query(["A"], top_k=5, max_hop=2)
    assert len(answers) > 0
    assert any(a.entity_id == "C" for a in answers)
    
    # Cleanup
    if isinstance(graph.adapter, MmapAdapter):
        data_dir = graph.adapter.data_dir
        graph.adapter.close()
        shutil.rmtree(data_dir)

if __name__ == "__main__":
    test_memory_governor_spill()

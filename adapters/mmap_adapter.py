"""
MmapAdapter: Memory-mapped graph storage providing disk-backed reasoning.
Optimized for NVME storage via zero-copy NumPy structured arrays.
"""
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx
import numpy as np

from core.graph_adapter import GraphAdapter, Entity, Edge
from core.hardware import MemoryGovernor

logger = logging.getLogger("cerebrum.mmap")

# Phase 171: NVME-Optimized Vectorized Mmap Architecture
A_DTYPE = np.dtype([
    ('global_idx', 'u8'),
    ('degree', 'u4'),
    ('offset', 'u8'),    # Byte offset into E-file
    ('community', 'u2'),
    ('padding', 'V10')   # 32 bytes total
])

E_DTYPE = np.dtype([
    ('target_idx', 'u8'),
    ('rel_id', 'u2'),
    ('confidence', 'f2') # 12 bytes total
])

class MmapAdapter(GraphAdapter):
    """
    MmapAdapter: Memory-mapped graph storage providing disk-backed reasoning.
    Optimized for massive graphs that exceed available RAM.
    """
    def __init__(self, data_dir: str, governor: Optional[MemoryGovernor] = None):
        self.data_dir = Path(data_dir)
        self.governor = governor
        
        # Binary files via NumPy Memmap (Zero-Copy)
        self.a_data = None
        self.e_data = None
        
        a_path = self.data_dir / "graph.a"
        e_path = self.data_dir / "graph.e"
        
        if a_path.exists():
            self.a_data = np.memmap(a_path, dtype=A_DTYPE, mode='r')
            self.e_data = np.memmap(e_path, dtype=E_DTYPE, mode='r')

        # RAM-resident metadata
        self.id_map: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.community_map: Dict[str, int] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        self._load_metadata()

    def _load_metadata(self):
        """Load node mapping and community assignments."""
        map_path = self.data_dir / "nodes.map"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split("\t")
                    node_id = parts[0]
                    self.id_map[node_id] = i
                    self.idx_to_id[i] = node_id
                    if len(parts) > 1:
                        self.community_map[node_id] = int(parts[1])

    @classmethod
    def from_adapter(cls, source_adapter, output_dir: str):
        """Serialize a source adapter's topology to binary format."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        G = source_adapter.to_networkx()
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # 1. Write nodes.map
        with open(out / "nodes.map", "w", encoding="utf-8") as f:
            for node in nodes:
                cid = source_adapter.get_community(node)
                f.write(f"{node}\t{cid}\n")
        
        # 2. Pre-allocate memmaps to final size
        n_nodes = len(nodes)
        # Total records in E-file = sum of all node degrees
        n_edge_records = sum(dict(G.degree()).values())
        
        a_path = out / "graph.a"
        e_path = out / "graph.e"
        
        # Create empty files of correct size
        with open(a_path, 'wb') as f:
            f.seek(n_nodes * A_DTYPE.itemsize - 1)
            f.write(b'\0')
        with open(e_path, 'wb') as f:
            f.seek(n_edge_records * E_DTYPE.itemsize - 1)
            f.write(b'\0')

        a_mmap = np.memmap(a_path, dtype=A_DTYPE, mode='r+', shape=(n_nodes,))
        e_mmap = np.memmap(e_path, dtype=E_DTYPE, mode='r+', shape=(n_edge_records,))
        
        e_ptr = 0
        for i, node in enumerate(nodes):
            neighbors = G.adj[node]
            deg = len(neighbors)
            cid = source_adapter.get_community(node)
            if cid < 0: cid = 0
            
            # Write A-File Record
            a_mmap[i] = (i, deg, e_ptr * E_DTYPE.itemsize, cid, b'\x00'*10)
            
            # Write E-Block Records
            for neighbor, data in neighbors.items():
                target_idx = node_to_idx[neighbor]
                rel_id = 0 
                conf = data.get("confidence", 1.0)
                e_mmap[e_ptr] = (target_idx, rel_id, conf)
                e_ptr += 1
        
        # Flush and close
        a_mmap.flush()
        e_mmap.flush()
        del a_mmap
        del e_mmap
        
        return cls(output_dir)

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
        context_embedding: Optional[np.ndarray] = None,
    ) -> List[Edge]:
        """Fetch neighbors using zero-copy NumPy indexing."""
        if self.a_data is None or entity_id not in self.id_map:
            return []
            
        idx = self.id_map[entity_id]
        if idx >= len(self.a_data):
            return []

        record = self.a_data[idx]
        degree = record['degree']
        if degree == 0:
            return []
            
        offset_bytes = record['offset']
        start_idx = offset_bytes // E_DTYPE.itemsize
        count = min(degree, max_neighbors)
        
        # Zero-copy slice (returns a view)
        neighbor_block = self.e_data[start_idx : start_idx + count]
        
        edges = []
        for row in neighbor_block:
            target_id = self.idx_to_id.get(int(row['target_idx']))
            if target_id:
                edges.append(Edge(
                    source_id=entity_id,
                    target_id=target_id,
                    relation_type="LINKED",
                    confidence=float(row['confidence']),
                    weight=float(row['confidence'])
                ))
        return edges

    def get_neighbors_batch(self, entity_ids: List[str], max_neighbors: int = 50) -> Dict[str, List[Edge]]:
        """Phase 171: Batch neighbor retrieval."""
        # Simple loop for now, but NumPy optimized
        return {eid: self.get_neighbors(eid, max_neighbors=max_neighbors) for eid in entity_ids}

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if entity_id not in self.id_map:
            return None
        return Entity(id=entity_id, label=entity_id)

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        results = []
        q_lower = query.lower()
        for eid in self.id_map:
            if q_lower in eid.lower():
                results.append(Entity(id=eid, label=eid))
                if len(results) >= top_k:
                    break
        return results

    def get_community(self, entity_id: str) -> int:
        if entity_id in self.community_map:
            return self.community_map[entity_id]
        return -1

    def get_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        return self.embeddings.get(entity_id)

    def get_degree(self, entity_id: str) -> int:
        if self.a_data is None or entity_id not in self.id_map:
            return 0
        idx = self.id_map[entity_id]
        if idx < len(self.a_data):
            return int(self.a_data[idx]['degree'])
        return 0

    def add_edge(self, u, v, relation, confidence=1.0, provenance="", synthetic=False):
        pass

    def find_similar(self, embedding, top_k=10):
        if not self.embeddings:
            return []
        items = list(self.embeddings.items())
        ids = [it[0] for it in items]
        embs = np.stack([it[1] for it in items])
        sims = embs @ embedding
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [Entity(id=ids[i], label=ids[i]) for i in top_idx]

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        if self.a_data is None: return G
        for i in range(len(self.a_data)):
            node_id = self.idx_to_id.get(i)
            if not node_id: continue
            
            record = self.a_data[i]
            G.add_node(node_id, community=int(record['community']))
            
            start_idx = record['offset'] // E_DTYPE.itemsize
            for j in range(record['degree']):
                row = self.e_data[start_idx + j]
                target_id = self.idx_to_id.get(int(row['target_idx']))
                if target_id:
                    G.add_edge(node_id, target_id, confidence=float(row['confidence']))
        return G

    def node_count(self) -> int:
        return len(self.id_map)

    def close(self):
        self.a_data = None
        self.e_data = None

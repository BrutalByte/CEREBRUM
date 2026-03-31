"""
Holographic Index for CEREBRUM Federated Discovery.

Provides compressed, privacy-preserving structural signatures of graphs.
Uses Bloom Filters for membership and Centroid Embeddings for semantic relevance.
"""
import hashlib
import math
from typing import Dict, List, Tuple

import numpy as np
from core.graph_adapter import GraphAdapter


class SimpleBloomFilter:
    """A lightweight Bloom Filter implementation using MD5 hashes."""
    def __init__(self, capacity: int = 1000, error_rate: float = 0.05):
        self.capacity = capacity
        self.error_rate = error_rate
        self.size = int(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))
        self.hash_count = int((self.size / capacity) * math.log(2))
        self.bit_array = 0  # Using a large integer as a bit array

    def _hashes(self, item: str) -> List[int]:
        hashes = []
        for i in range(self.hash_count):
            h = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes

    def add(self, item: str):
        for h in self._hashes(item):
            self.bit_array |= (1 << h)

    def __contains__(self, item: str) -> bool:
        for h in self._hashes(item):
            if not (self.bit_array & (1 << h)):
                return False
        return True

    def to_hex(self) -> str:
        return hex(self.bit_array)

    @classmethod
    def from_hex(cls, hex_str: str, capacity: int, error_rate: float):
        bf = cls(capacity, error_rate)
        bf.bit_array = int(hex_str, 16)
        return bf


class CommunitySignature:
    """
    Compressed signature of a graph community.
    
    Attributes:
        community_id (int): Local community ID.
        centroid (np.ndarray): Mean embedding vector of members.
        bloom (SimpleBloomFilter): Set of entity IDs in this community.
        size (int): Number of members.
    """
    def __init__(self, community_id: int, centroid: np.ndarray, bloom: SimpleBloomFilter, size: int):
        self.community_id = community_id
        self.centroid = centroid
        self.bloom = bloom
        self.size = size

    def to_dict(self) -> Dict:
        return {
            "community_id": self.community_id,
            "centroid": self.centroid.tolist(),
            "bloom_hex": self.bloom.to_hex(),
            "bloom_cap": self.bloom.capacity,
            "bloom_err": self.bloom.error_rate,
            "size": self.size
        }

    @classmethod
    def from_dict(cls, d: Dict):
        bf = SimpleBloomFilter.from_hex(d["bloom_hex"], d["bloom_cap"], d["bloom_err"])
        return cls(
            community_id=d["community_id"],
            centroid=np.array(d["centroid"], dtype=np.float32),
            bloom=bf,
            size=d["size"]
        )


class HolographicIndex:
    """
    Aggregates signatures from multiple CEREBRUM instances.
    Used by FederatedAdapter to decide which remote instance to query.
    """
    def __init__(self):
        # adapter_name -> List[CommunitySignature]
        self.signatures: Dict[str, List[CommunitySignature]] = {}

    def add_adapter_signatures(self, adapter_name: str, sigs: List[CommunitySignature]):
        self.signatures[adapter_name] = sigs

    def find_relevant_adapters(self, query_emb: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Rank adapters by how well the query matches any of their community centroids.
        """
        scores: Dict[str, float] = {}
        
        for name, sigs in self.signatures.items():
            max_sim = -1.0
            for s in sigs:
                # Cosine similarity with centroid
                norm_q = np.linalg.norm(query_emb)
                norm_c = np.linalg.norm(s.centroid)
                if norm_q > 0 and norm_c > 0:
                    sim = float(np.dot(query_emb, s.centroid) / (norm_q * norm_c))
                    max_sim = max(max_sim, sim)
            scores[name] = max_sim

        sorted_adapters = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_adapters[:top_k]

    def probe_entity(self, entity_id: str) -> List[str]:
        """
        Identify which adapters likely contain this entity_id via Bloom Filters.
        Returns list of adapter names.
        """
        hits = []
        for name, sigs in self.signatures.items():
            for s in sigs:
                if entity_id in s.bloom:
                    hits.append(name)
                    break
        return hits


def build_signatures(
    adapter: GraphAdapter, 
    community_map: Dict[str, int], 
    embeddings: Dict[str, np.ndarray]
) -> List[CommunitySignature]:
    """
    Compute signatures for all communities in a graph.
    """
    # 1. Group by community
    com_members: Dict[int, List[str]] = {}
    for node, cid in community_map.items():
        com_members.setdefault(cid, []).append(node)

    sigs = []
    for cid, members in com_members.items():
        # 2. Compute Centroid
        vectors = [embeddings[m] for m in members if m in embeddings]
        if not vectors:
            continue
        centroid = np.mean(vectors, axis=0)

        # 3. Build Bloom Filter
        bf = SimpleBloomFilter(capacity=max(100, len(members) * 2))
        for m in members:
            bf.add(m)

        sigs.append(CommunitySignature(
            community_id=cid,
            centroid=centroid,
            bloom=bf,
            size=len(members)
        ))
    
    return sigs

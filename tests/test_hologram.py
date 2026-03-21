import numpy as np
import pytest
from core.holographic_index import (
    SimpleBloomFilter, CommunitySignature, HolographicIndex, build_signatures
)
from adapters.networkx_adapter import NetworkXAdapter
import networkx as nx

def test_bloom_filter():
    bf = SimpleBloomFilter(capacity=100, error_rate=0.01)
    bf.add("apple")
    bf.add("banana")
    
    assert "apple" in bf
    assert "banana" in bf
    assert "cherry" not in bf
    
    # Test serialization
    hex_str = bf.to_hex()
    bf2 = SimpleBloomFilter.from_hex(hex_str, 100, 0.01)
    assert "apple" in bf2
    assert "banana" in bf2
    assert "cherry" not in bf2

def test_community_signature_serialization():
    bf = SimpleBloomFilter(capacity=10)
    bf.add("n1")
    sig = CommunitySignature(
        community_id=1,
        centroid=np.array([1.0, 0.0], dtype=np.float32),
        bloom=bf,
        size=5
    )
    
    d = sig.to_dict()
    sig2 = CommunitySignature.from_dict(d)
    
    assert sig2.community_id == 1
    assert sig2.size == 5
    assert np.allclose(sig2.centroid, sig.centroid)
    assert "n1" in sig2.bloom

def test_holographic_index_discovery():
    idx = HolographicIndex()
    
    # Adapter A: Fruit Graph
    bf_a = SimpleBloomFilter(capacity=10)
    bf_a.add("apple")
    sig_a = CommunitySignature(
        community_id=0,
        centroid=np.array([1.0, 0.0], dtype=np.float32), # "fruit" vector
        bloom=bf_a,
        size=1
    )
    idx.add_adapter_signatures("fruits", [sig_a])
    
    # Adapter B: Space Graph
    bf_b = SimpleBloomFilter(capacity=10)
    bf_b.add("mars")
    sig_b = CommunitySignature(
        community_id=0,
        centroid=np.array([0.0, 1.0], dtype=np.float32), # "planet" vector
        bloom=bf_b,
        size=1
    )
    idx.add_adapter_signatures("space", [sig_b])
    
    # Test probing by ID (Bloom Filter)
    assert idx.probe_entity("apple") == ["fruits"]
    assert idx.probe_entity("mars") == ["space"]
    assert idx.probe_entity("earth") == []
    
    # Test relevance by embedding (Centroid)
    query_fruit = np.array([0.9, 0.1], dtype=np.float32)
    results = idx.find_relevant_adapters(query_fruit)
    assert results[0][0] == "fruits"
    
    query_planet = np.array([0.1, 0.9], dtype=np.float32)
    results = idx.find_relevant_adapters(query_planet)
    assert results[0][0] == "space"

def test_build_signatures():
    g = nx.Graph()
    g.add_node("n1")
    g.add_node("n2")
    adapter = NetworkXAdapter(g)
    
    cmap = {"n1": 0, "n2": 0}
    embs = {
        "n1": np.array([1.0, 0.0], dtype=np.float32),
        "n2": np.array([0.0, 1.0], dtype=np.float32)
    }
    
    sigs = build_signatures(adapter, cmap, embs)
    assert len(sigs) == 1
    assert sigs[0].size == 2
    # Centroid should be mean
    assert np.allclose(sigs[0].centroid, [0.5, 0.5])
    assert "n1" in sigs[0].bloom
    assert "n2" in sigs[0].bloom

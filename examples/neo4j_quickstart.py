"""
Quickstart: connect to Neo4j and run a query.

Requires:
  pip install neo4j
  A running Neo4j instance (default: bolt://localhost:7687)

Set environment variables:
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password

Usage:
    python examples/neo4j_quickstart.py
"""
from typing import Set
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.neo4j_adapter import Neo4jAdapter
from core.community_engine import best_of_n_dscf
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract


def main():
    with Neo4jAdapter.from_env() as adapter:
        G = adapter.to_networkx()
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        parts = best_of_n_dscf(G, n_trials=3, resolution=adapter.adaptive_resolution(), seed=42)
        community_map = {node: cid for cid, members in enumerate(parts) for node in members}
        print(f"Communities: {len(parts)}")

        engine    = RandomEngine(dim=64)
        labels    = {n: n for n in G.nodes()}
        embeddings = engine.encode_entities(labels)

        # Update adapter with metadata for CSAEngine to use
        adapter.community_map = community_map
        adapter.embeddings = embeddings

        dist = build_community_distance_matrix(G, community_map)
        adj  = adjacent_community_pairs(G, community_map)
        csa  = CSAEngine(adapter=adapter)
        csa.set_community_graph(dist, adj)

        query   = "einstein"
        seeds   = [e.id for e in adapter.find_entities(query, top_k=3) if e]
        print(f"Seeds for {query!r}: {seeds}")

        traversal = BeamTraversal(
            adapter=adapter, csa_engine=csa, beam_width=10, max_hop=3,
            edge_type_weights={"KNOWS": 0.4, "WORKS_AT": 0.3},
        )
        paths   = traversal.traverse(seeds)
        answers = extract(paths, top_k=5)

        print("\nTop answers:")
        for ans in answers:
            print(f"  {ans.entity_id:30s}  score={ans.score:.4f}")


if __name__ == "__main__":
    main()




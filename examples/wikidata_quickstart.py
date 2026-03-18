"""
Quickstart: query Wikidata via SPARQL.

Requires:
  pip install SPARQLWrapper

NOTE: The Wikidata SPARQL endpoint is rate-limited. For serious use,
download a Wikidata dump and run a local Fuseki or Virtuoso instance.

Usage:
    python examples/wikidata_quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.rdf_adapter import RDFAdapter, WIKIDATA_PREFIXES
from core.community_engine import best_of_n_dscf
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract


def main():
    adapter = RDFAdapter(
        endpoint="https://query.wikidata.org/sparql",
        prefixes=WIKIDATA_PREFIXES,
        timeout=30,
    )

    # Search for Einstein's entity
    print("Finding Albert Einstein in Wikidata...")
    entities = adapter.find_entities("Albert Einstein", top_k=3)
    if not entities:
        print("No entities found. Check your SPARQL connection.")
        return

    einstein = entities[0]
    print(f"Found: {einstein.label} ({einstein.id})")

    # Get 1-hop neighborhood
    neighbors = adapter.get_neighbors(einstein.id, max_neighbors=20)
    print(f"Neighbors: {len(neighbors)}")

    # Build a local NetworkX subgraph from the 1-hop neighborhood
    import networkx as nx
    G = nx.DiGraph()
    G.add_node(einstein.id)
    for edge in neighbors:
        G.add_edge(edge.source_id, edge.target_id, relation=edge.relation_type)

    # Community detection on local subgraph
    parts         = best_of_n_dscf(nx.Graph(G), n_trials=3, seed=42)
    community_map = {node: cid for cid, members in enumerate(parts) for node in members}

    # Embeddings
    engine    = RandomEngine(dim=64)
    labels    = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    # CSA + traversal on local graph
    from adapters.networkx_adapter import NetworkXAdapter
    local_adapter = NetworkXAdapter(nx.Graph(G))
    dist = build_community_distance_matrix(nx.Graph(G), community_map)
    adj  = adjacent_community_pairs(nx.Graph(G), community_map)
    csa  = CSAEngine(communities=community_map, embeddings=embeddings)
    csa.set_community_graph(dist, adj)

    traversal = BeamTraversal(
        adapter=local_adapter, csa_engine=csa, embeddings=embeddings,
        communities=community_map, beam_width=5, max_hop=2,
    )
    paths   = traversal.traverse([einstein.id])
    answers = extract(paths, top_k=5)

    print(f"\nTop answers from Einstein's neighborhood:")
    for ans in answers:
        print(f"  {ans.entity_id:50s}  score={ans.score:.4f}")


if __name__ == "__main__":
    main()

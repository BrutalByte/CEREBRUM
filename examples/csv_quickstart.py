"""
Quickstart: load a graph from CSV and run a query.

The simplest possible Parallax session — no external dependencies
beyond networkx, igraph, and leidenalg.

Usage:
    python examples/csv_quickstart.py
"""
import random
import sys
from pathlib import Path

# Make sure parallax root is on path when running from examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.csv_adapter import load_csv_adapter
from core.community_engine import best_of_n_dscf, modularity_score
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from llm_bridge.context_formatter import to_prompt


def main():
    # 1. Load graph
    csv_path = Path(__file__).parent.parent / "tests" / "fixtures" / "toy_graph.csv"
    adapter  = load_csv_adapter(str(csv_path))
    G        = adapter.to_networkx()
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Detect communities (DSCF — best of 5 trials)
    parts = best_of_n_dscf(G, n_trials=5, seed=42)
    q     = modularity_score(G, parts)
    print(f"Communities: {len(parts)}  |  Modularity Q: {q:.4f}")

    community_map = {}
    for cid, members in enumerate(parts):
        for node in members:
            community_map[node] = cid

    # 3. Embed entities (RandomEngine — no sentence-transformers needed)
    engine    = RandomEngine(dim=64)
    labels    = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    # 4. Build CSA engine
    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(communities=community_map, embeddings=embeddings)
    csa.set_community_graph(dist, adj)

    # 5. Traverse from "newton"
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        embeddings=embeddings,
        communities=community_map,
        beam_width=10,
        max_hop=3,
    )

    print("\nQuery: What is connected to 'newton'?")
    paths   = traversal.traverse(["newton"])
    answers = extract(paths, top_k=5)

    print(f"Paths explored: {len(paths)}")
    print("\nTop-5 answers:")
    for ans in answers:
        path_str = " -> ".join(ans.best_path.nodes)
        print(f"  {ans.entity_id:20s}  score={ans.score:.4f}  path: {path_str}")

    # 6. Format as LLM prompt (optional)
    prompt = to_prompt(answers, query="What is connected to newton?")
    print("\n--- LLM prompt ---")
    print(prompt)


if __name__ == "__main__":
    main()

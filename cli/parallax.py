"""
Parallax command-line interface.

Usage:
    python -m cli.parallax query --csv tests/fixtures/toy_graph.csv "newton"
    python -m cli.parallax communities --csv tests/fixtures/toy_graph.csv
    python -m cli.parallax serve --csv tests/fixtures/toy_graph.csv --port 8200
"""
import argparse
import json
import sys


def cmd_query(args):
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf
    from core.embedding_engine import RandomEngine
    from core.attention_engine import CSAEngine
    from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
    from reasoning.traversal import BeamTraversal
    from reasoning.answer_extractor import extract
    from llm_bridge.context_formatter import to_structured

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Communities
    parts        = best_of_n_dscf(G, n_trials=3, seed=42)
    community_map = {}
    for cid, members in enumerate(parts):
        for node in members:
            community_map[node] = cid
    print(f"Communities: {len(parts)}")

    # Embeddings
    engine = RandomEngine(dim=64)
    labels = {}
    for node in G.nodes():
        e = adapter.get_entity(node)
        labels[node] = e.label if e else node
    embeddings = engine.encode_entities(labels)

    # CSA
    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(communities=community_map, embeddings=embeddings)
    csa.set_community_graph(dist, adj)

    # Traversal
    seeds = [args.query] if args.query in G else [
        e.id for e in adapter.find_entities(args.query, top_k=3) if e
    ]
    if not seeds:
        print(f"No entities found for: {args.query!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Seeds: {seeds}")

    traversal = BeamTraversal(
        adapter=adapter, csa_engine=csa, embeddings=embeddings,
        communities=community_map, beam_width=args.beam_width, max_hop=args.max_hop,
    )
    paths   = traversal.traverse(seeds)
    answers = extract(paths, top_k=args.top_k)

    if args.json:
        result = to_structured(answers, query=args.query, adapter=adapter)
        print(json.dumps(result, indent=2))
    else:
        print(f"\nTop-{args.top_k} answers for: {args.query!r}")
        print("-" * 60)
        for ans in answers:
            print(f"  {ans.entity_id:20s}  score={ans.score:.4f}  {ans.score_breakdown}")
            path_str = " -> ".join(ans.best_path.nodes)
            print(f"    path: {path_str}")


def cmd_communities(args):
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf, modularity_score

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()
    parts   = best_of_n_dscf(G, n_trials=5, seed=42)
    q       = modularity_score(G, parts)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Communities: {len(parts)}  |  Modularity Q: {q:.4f}")
    print()

    for i, members in enumerate(sorted(parts, key=len, reverse=True)):
        sample = sorted(members)[:8]
        ellipsis = " ..." if len(members) > 8 else ""
        print(f"  Community {i:3d}  ({len(members):3d} nodes): {', '.join(sample)}{ellipsis}")


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print("uvicorn required: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    from adapters.csv_adapter import load_csv_adapter
    from core.embedding_engine import RandomEngine
    from api.server import create_app

    adapter = load_csv_adapter(args.csv)
    engine  = RandomEngine(dim=64)
    app     = create_app(adapter=adapter, embedding_engine=engine)

    print(f"Serving Parallax API on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="parallax",
        description="Parallax — Community-Structured Graph Attention for KG Reasoning",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # query
    q = sub.add_parser("query", help="Run a multi-hop reasoning query")
    q.add_argument("query", help="Query string or entity ID")
    q.add_argument("--csv", required=True, help="Path to edge-list CSV")
    q.add_argument("--top-k", type=int, default=5, dest="top_k")
    q.add_argument("--max-hop", type=int, default=3, dest="max_hop")
    q.add_argument("--beam-width", type=int, default=10, dest="beam_width")
    q.add_argument("--json", action="store_true", help="Output as JSON")

    # communities
    c = sub.add_parser("communities", help="Show detected communities")
    c.add_argument("--csv", required=True, help="Path to edge-list CSV")

    # serve
    s = sub.add_parser("serve", help="Start the REST API server")
    s.add_argument("--csv", required=True, help="Path to edge-list CSV")
    s.add_argument("--port", type=int, default=8200)

    args = parser.parse_args()

    if args.command == "query":
        cmd_query(args)
    elif args.command == "communities":
        cmd_communities(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()

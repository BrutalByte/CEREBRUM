"""
CEREBRUM command-line interface.

Usage:
    python -m cli.cerebrum query --csv tests/fixtures/toy_graph.csv "newton"
    python -m cli.cerebrum communities --csv tests/fixtures/toy_graph.csv
    python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200
    python -m cli.cerebrum ask --csv tests/fixtures/toy_graph.csv "What did newton influence?"
"""
import argparse
import json
import sys


def cmd_query(args):
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf, hierarchical_dscf
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
    if args.hierarchical:
        parts = hierarchical_dscf(G, target_communities=args.target_communities)
    else:
        parts = best_of_n_dscf(G, n_trials=3, seed=42)

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

    # Bridge Bonus
    bridge_weights = {}
    if args.bridge_bonus:
        try:
            bridge_weights = json.loads(args.bridge_bonus)
        except json.JSONDecodeError:
            # Fallback to key:val,key:val
            for pair in args.bridge_bonus.split(","):
                if ":" in pair:
                    k, v = pair.split(":")
                    bridge_weights[k] = float(v)

    # Attach to adapter for CSAEngine
    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    # CSA
    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(
        adapter=adapter,
        edge_type_weights=bridge_weights
    )
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
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=args.max_hop,
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
    from core.community_engine import best_of_n_dscf, hierarchical_dscf, modularity_score

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    if args.hierarchical:
        parts = hierarchical_dscf(G, target_communities=args.target_communities)
    else:
        parts = best_of_n_dscf(G, n_trials=5, seed=42)

    q = modularity_score(G, parts)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Communities: {len(parts)}  |  Modularity Q: {q:.4f}")
    print()

    for i, members in enumerate(sorted(parts, key=len, reverse=True)):
        sample = sorted(members)[:8]
        ellipsis = " ..." if len(members) > 8 else ""
        print(f"  Community {i:3d}  ({len(members):3d} nodes): {', '.join(sample)}{ellipsis}")


def cmd_ingest(args):
    """Extract triples from plain text and add them to the graph."""
    import csv as csv_mod
    from adapters.csv_adapter import load_csv_adapter
    from core.text_ingestor import TextIngestor

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    if not args.quiet:
        print(f"Graph before: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Get text from --text or --file
    if args.text:
        text = args.text
    elif args.file:
        from pathlib import Path
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        print("Provide --text TEXT or --file PATH", file=sys.stderr)
        sys.exit(1)

    ingestor = TextIngestor(
        adapter,
        min_confidence=args.min_confidence,
        create_new_entities=not args.no_new_entities,
    )

    report = ingestor.ingest_text(text, dry_run=args.dry_run)

    print(report.summary())

    if not args.dry_run and report.edges_added > 0 and args.output:
        G2 = adapter.to_networkx()
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.writer(fh)
            writer.writerow(["source", "target", "relation", "confidence"])
            for u, v, d in G2.edges(data=True):
                rel  = d.get("relation_type") or d.get("relation", "")
                conf = d.get("confidence", 1.0)
                writer.writerow([u, v, rel, conf])
        print(f"\nSaved updated graph → {args.output}")
    elif not args.dry_run and report.edges_added > 0:
        print(f"\nGraph updated in memory. Use --output PATH to persist.")


def cmd_chat(args):
    """Interactive multi-turn conversation loop."""
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf
    from core.embedding_engine import RandomEngine
    from core.attention_engine import CSAEngine
    from core.structural_encoder import (
        build_community_distance_matrix,
        adjacent_community_pairs,
        coarsen_communities,
    )
    from core.conversation import ConversationManager
    from reasoning.traversal import BeamTraversal

    # ── Load graph + pipeline ────────────────────────────────────────
    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    print(f"CEREBRUM  |  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    community_map = {}
    for cid, members in enumerate(parts):
        for node in members:
            community_map[node] = cid
    if len(parts) > 500:
        community_map = coarsen_communities(G, community_map, target_max=500)

    engine = RandomEngine(dim=64)
    labels = {n: (adapter.get_entity(n).label if adapter.get_entity(n) else n)
              for n in G.nodes()}
    embeddings = engine.encode_entities(labels)
    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)

    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=args.max_hop,
    )

    manager = ConversationManager(
        adapter=adapter,
        embedding_engine=engine,
        csa_engine=csa,
        beam_traversal=traversal,
        top_k=args.top_k,
    )
    session = manager.new_session()

    # ── Enable readline history if available ────────────────────────
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print(f"Communities: {len(set(community_map.values()))}  |  Type 'help' for commands\n")

    # ── REPL ────────────────────────────────────────────────────────
    while True:
        try:
            raw = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        # Special commands
        if cmd in ("quit", "exit", "q", "bye"):
            print("Goodbye.")
            break

        if cmd == "reset":
            session.reset()
            print("[Session cleared]\n")
            continue

        if cmd == "history":
            if not session.turns:
                print("[No turns yet]\n")
            else:
                for t in session.turns:
                    print(f"  [{t.turn_number}] {t.raw_question!r}")
                    if t.resolved_question != t.raw_question:
                        print(f"       → resolved: {t.resolved_question!r}")
            print()
            continue

        if cmd == "focus":
            if session.focus_entity:
                print(f"[Focus: {session.focus_entity_label!r} ({session.focus_entity})]")
            else:
                print("[No focus entity set]")
            print()
            continue

        if cmd == "summary":
            print(manager.session_summary(session))
            print()
            continue

        if cmd == "help":
            print(
                "  Commands: quit | reset | history | focus | summary | help\n"
                "  Just type a question to query the graph.\n"
                "  Pronouns (he/she/it/they/there) resolve to recent entities.\n"
                "  Follow-up phrases (and/what else/also) continue from current focus.\n"
            )
            continue

        # ── Process question ────────────────────────────────────────
        turn = manager.process(raw, session)

        print()
        print(f"CEREBRUM > {turn.answer_text}")
        if turn.is_followup:
            print(f"  [follow-up on {session.focus_entity_label!r}]")
        if turn.focus_shift:
            print(f"  [topic shift → {turn.seed_entity_label!r}]")
        if turn.resolved_question != turn.raw_question:
            print(f"  [resolved: {turn.resolved_question!r}]")
        print()


def cmd_infer(args):
    """Run the transitive inference engine and print discoveries."""
    from adapters.csv_adapter import load_csv_adapter
    from core.inference_engine import TransitiveInferenceEngine, InferenceRule

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    if not args.quiet:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    enabled = set(args.domains.split(",")) if args.domains else None

    custom_rules = []
    if args.rule:
        for rule_str in args.rule:
            # format: "REL_A:REL_B:DERIVED:factor:domain"
            parts = rule_str.split(":")
            if len(parts) != 5:
                print(f"Invalid --rule format (expected A:B:DERIVED:factor:domain): {rule_str!r}",
                      file=sys.stderr)
                sys.exit(1)
            custom_rules.append(InferenceRule(
                rel_a=parts[0].upper(), rel_b=parts[1].upper(),
                derived=parts[2].upper(), factor=float(parts[3]),
                domain=parts[4], note="user-defined",
            ))

    engine = TransitiveInferenceEngine(
        adapter,
        max_proposals=args.max_proposals,
        min_confidence=args.min_confidence,
        enabled_domains=enabled,
        custom_rules=custom_rules or None,
    )

    if not args.quiet:
        print(f"Active rules: {engine.rule_count()}")
        print(f"Mode        : {'dry-run' if args.dry_run else 'apply'}")
        print()

    report = engine.run(dry_run=args.dry_run)

    print(report.summary())
    print()

    if report.proposals:
        print(f"Top discoveries (showing up to {args.show}):")
        print("-" * 70)
        for p in report.proposals[:args.show]:
            print(f"  {p.derivation_str}")
    else:
        print("No new discoveries found.")


def cmd_ask(args):
    """Full NL→Graph→NL pipeline: parse question, traverse, verbalize."""
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf
    from core.embedding_engine import RandomEngine
    from core.attention_engine import CSAEngine
    from core.structural_encoder import (
        build_community_distance_matrix,
        adjacent_community_pairs,
        coarsen_communities,
    )
    from core.query_parser import QueryParser
    from core.verbalizer import PathVerbalizer
    from reasoning.traversal import BeamTraversal
    from reasoning.answer_extractor import extract

    adapter = load_csv_adapter(args.csv)
    G       = adapter.to_networkx()

    if not args.quiet:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Communities
    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    community_map = {}
    for cid, members in enumerate(parts):
        for node in members:
            community_map[node] = cid

    # Coarsen if over-partitioned
    if len(parts) > 500:
        community_map = coarsen_communities(G, community_map, target_max=500)

    if not args.quiet:
        print(f"Communities: {len(set(community_map.values()))}")

    # Embeddings
    engine = RandomEngine(dim=64)
    labels = {}
    for node in G.nodes():
        e = adapter.get_entity(node)
        labels[node] = e.label if e else node
    embeddings = engine.encode_entities(labels)

    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    # Parse the natural language question
    qparser = QueryParser(adapter, engine)
    parsed  = qparser.parse(args.question)

    if parsed.seed_entity_id is None:
        print(f"Could not identify a graph entity in: {args.question!r}", file=sys.stderr)
        print("Try rephrasing with an entity name that appears in the graph.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Seed entity : {parsed.seed_entity_id!r}  (score={parsed.seed_entity_score:.3f})")
        print(f"Relation hints: {parsed.relation_hints}")
        print(f"Hop hint    : {parsed.hop_hint}")
        print()

    # CSA + traversal
    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)

    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=parsed.hop_hint,
    )
    paths   = traversal.traverse([parsed.seed_entity_id])
    answers = extract(paths, top_k=args.top_k)

    # Verbalize
    verb = PathVerbalizer()
    output = verb.verbalize_answers(answers, adapter=adapter, question=args.question, top_k=args.top_k)
    print(output)


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

    # Bridge Bonus
    bridge_weights = {}
    if args.bridge_bonus:
        try:
            bridge_weights = json.loads(args.bridge_bonus)
        except json.JSONDecodeError:
            for pair in args.bridge_bonus.split(","):
                if ":" in pair:
                    k, v = pair.split(":")
                    bridge_weights[k] = float(v)

    app     = create_app(
        adapter=adapter,
        embedding_engine=engine,
        hierarchical_dscf_enabled=args.hierarchical,
        target_communities=args.target_communities,
        default_edge_type_weights=bridge_weights,
    )

    print(f"Serving CEREBRUM API on http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="cerebrum",
        description="CEREBRUM — Community-Structured Graph Attention for KG Reasoning",
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
    q.add_argument("--hierarchical", action="store_true", help="Use hierarchical DSCF")
    q.add_argument("--target-communities", type=int, default=500, dest="target_communities")
    q.add_argument("--bridge-bonus", help="JSON or 'key:val,key:val' for edge weights")

    # communities
    c = sub.add_parser("communities", help="Show detected communities")
    c.add_argument("--csv", required=True, help="Path to edge-list CSV")
    c.add_argument("--hierarchical", action="store_true", help="Use hierarchical DSCF")
    c.add_argument("--target-communities", type=int, default=500, dest="target_communities")

    # ingest
    ig = sub.add_parser("ingest", help="Extract triples from text and add to graph")
    ig.add_argument("--csv",    required=True, help="Path to edge-list CSV")
    ig.add_argument("--text",   default=None,  help="Text to ingest (quoted string)")
    ig.add_argument("--file",   default=None,  help="Path to text file to ingest")
    ig.add_argument("--output", default=None,  help="Save updated graph to this CSV path")
    ig.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Show discoveries without modifying the graph")
    ig.add_argument("--min-confidence", type=float, default=0.30, dest="min_confidence")
    ig.add_argument("--no-new-entities", action="store_true", dest="no_new_entities",
                    help="Only add edges between existing graph entities")
    ig.add_argument("--quiet", action="store_true")

    # chat
    ch = sub.add_parser("chat", help="Interactive multi-turn conversation with the graph")
    ch.add_argument("--csv", required=True, help="Path to edge-list CSV")
    ch.add_argument("--top-k", type=int, default=5, dest="top_k")
    ch.add_argument("--max-hop", type=int, default=3, dest="max_hop")
    ch.add_argument("--beam-width", type=int, default=10, dest="beam_width")

    # infer
    inf = sub.add_parser("infer", help="Discover new knowledge via transitive inference")
    inf.add_argument("--csv", required=True, help="Path to edge-list CSV")
    inf.add_argument("--dry-run", action="store_true", dest="dry_run",
                     help="Show discoveries without modifying the graph")
    inf.add_argument("--max-proposals", type=int, default=200, dest="max_proposals",
                     help="Maximum number of inferred edges to propose")
    inf.add_argument("--min-confidence", type=float, default=0.10, dest="min_confidence",
                     help="Minimum confidence threshold for proposals")
    inf.add_argument("--domains", default=None,
                     help="Comma-separated domain filter: biology,academic,social,causal,film,general")
    inf.add_argument("--rule", action="append", metavar="A:B:DERIVED:factor:domain",
                     help="Add a custom inference rule (repeatable)")
    inf.add_argument("--show", type=int, default=20,
                     help="Number of discoveries to print")
    inf.add_argument("--quiet", action="store_true", help="Suppress diagnostic headers")

    # ask
    a = sub.add_parser("ask", help="Answer a natural language question using the graph")
    a.add_argument("question", help="Natural language question, e.g. 'What did newton influence?'")
    a.add_argument("--csv", required=True, help="Path to edge-list CSV")
    a.add_argument("--top-k", type=int, default=5, dest="top_k")
    a.add_argument("--beam-width", type=int, default=10, dest="beam_width")
    a.add_argument("--quiet", action="store_true", help="Suppress diagnostic output, print only the answer")

    # serve
    s = sub.add_parser("serve", help="Start the REST API server")
    s.add_argument("--csv", required=True, help="Path to edge-list CSV")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument("--hierarchical", action="store_true", help="Use hierarchical DSCF for API server")
    s.add_argument("--target-communities", type=int, default=500, dest="target_communities", help="Target communities for hierarchical DSCF")
    s.add_argument("--bridge-bonus", help="JSON or 'key:val,key:val' for default API edge weights")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "communities":
        cmd_communities(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()

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
    from core.resource_governor import ResourceGovernor

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

    governor = ResourceGovernor(max_ram_gb=args.max_ram_gb, max_vram_gb=args.max_vram_gb)
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=args.max_hop,
        governor=governor,
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
        print("\nGraph updated in memory. Use --output PATH to persist.")


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
    from core.resource_governor import ResourceGovernor

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
    labels = {}
    for n in G.nodes():
        ent = adapter.get_entity(n)
        labels[n] = ent.label if ent else n
    embeddings = engine.encode_entities(labels)
    adapter.community_map = community_map
    adapter.embeddings    = embeddings

    dist = build_community_distance_matrix(G, community_map)
    adj  = adjacent_community_pairs(G, community_map)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)

    governor = ResourceGovernor(max_ram_gb=args.max_ram_gb, max_vram_gb=args.max_vram_gb)
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=args.max_hop,
        governor=governor,
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
    from core.resource_governor import ResourceGovernor

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

    governor = ResourceGovernor(max_ram_gb=args.max_ram_gb, max_vram_gb=args.max_vram_gb)
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=args.beam_width,
        max_hop=parsed.hop_hint,
        governor=governor,
    )
    paths   = traversal.traverse([parsed.seed_entity_id])

    answers = extract(paths, top_k=args.top_k)

    # Verbalize
    verb = PathVerbalizer()
    output = verb.verbalize_answers(answers, adapter=adapter, question=args.question, top_k=args.top_k)
    print(output)


def cmd_init(args):
    """Quickstart wizard: load a KB and optionally launch Studio."""
    import os
    import webbrowser
    from pathlib import Path

    # ── Resolve CSV path ────────────────────────────────────────────────────
    if args.demo:
        csv_path = str(Path(__file__).parent.parent / "tests" / "fixtures" / "toy_graph.csv")
        print("CEREBRUM Quickstart — demo mode (toy_graph.csv)")
    elif args.from_csv:
        csv_path = args.from_csv
        if not Path(csv_path).exists():
            print(f"Error: file not found: {csv_path!r}", file=sys.stderr)
            sys.exit(1)
        print(f"CEREBRUM Quickstart — loading {csv_path!r}")
    else:
        print("Provide --from-csv PATH or --demo", file=sys.stderr)
        sys.exit(1)

    # ── Load graph ───────────────────────────────────────────────────────────
    from adapters.csv_adapter import load_csv_adapter
    from core.community_engine import best_of_n_dscf

    print("  Loading knowledge base...", end=" ", flush=True)
    adapter = load_csv_adapter(csv_path)
    G = adapter.to_networkx()
    print("done.")

    # ── Detect communities ───────────────────────────────────────────────────
    print("  Detecting communities...", end=" ", flush=True)
    parts = best_of_n_dscf(G, n_trials=3, seed=42)
    rel_types = {d.get("relation_type") or d.get("relation", "—") for _, _, d in G.edges(data=True)}
    print("done.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  Knowledge Base Ready")
    print("=" * 50)
    print(f"  Entities       : {G.number_of_nodes():,}")
    print(f"  Relations      : {G.number_of_edges():,}")
    print(f"  Relation types : {len(rel_types)}")
    print(f"  Communities    : {len(parts)}")
    print()
    print("  Try it now:")
    print(f'    cerebrum ask --csv "{csv_path}" "Your question here"')
    print(f'    cerebrum chat --csv "{csv_path}"')
    print(f'    cerebrum serve --csv "{csv_path}" --port 8200')
    print()

    # ── Optionally launch Studio ──────────────────────────────────────────────
    if args.serve or args.open:
        try:
            import uvicorn
        except ImportError:
            print("uvicorn required to launch server: pip install uvicorn", file=sys.stderr)
            sys.exit(1)
        from core.embedding_engine import RandomEngine
        from api.server import create_app

        engine = RandomEngine(dim=64)
        app = create_app(
            adapter=adapter,
            embedding_engine=engine,
            max_ram_gb=None,
            max_vram_gb=None,
        )
        port = args.port
        url = f"http://localhost:{port}/v1/docs"
        print(f"  Starting API server at http://localhost:{port}/v1/")
        print(f"  Swagger docs: {url}")
        if args.open:
            webbrowser.open(url)
        uvicorn.run(app, host="0.0.0.0", port=port)


def cmd_tune(args):
    """Launch the live hyperparameter tuner (Optuna + Rich dashboard)."""
    try:
        from benchmarks.cerebrum_tuner import run_tuner
    except ImportError as exc:
        print(
            f"cerebrum_tuner import failed: {exc}\n"
            "Install dependencies with:  pip install 'cerebrum-kg[tuning]'",
            file=sys.stderr,
        )
        sys.exit(1)
    run_tuner(
        n_trials=args.n_trials,
        sample=args.sample,
        study_name=args.study_name,
        validate=args.validate,
        seed=args.seed,
    )


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

    app = create_app(
        adapter=adapter,
        embedding_engine=engine,
        hierarchical_dscf_enabled=args.hierarchical,
        target_communities=args.target_communities,
        default_edge_type_weights=bridge_weights,
        ws_port=getattr(args, "ws_port", None),
        max_ram_gb=args.max_ram_gb,
        max_vram_gb=args.max_vram_gb,
        compliance=getattr(args, "compliance", False),
        audit_log_file=getattr(args, "audit_log", None),
    )

    # Restore learned parameters from file if provided
    params_file = getattr(args, "params_file", None)
    if params_file:
        import json as _json
        from core.parameter_learner import MetaParameterLearner
        from api.server import _state
        try:
            with open(params_file) as fh:
                data = _json.load(fh)
            _state["meta_learner"] = MetaParameterLearner.from_dict(data)
            n_overrides = len(_state["meta_learner"].community_overrides)
            print(f"  [CLI] Loaded params from {params_file} ({n_overrides} community overrides)")
        except Exception as exc:
            print(f"  [CLI] Warning: could not load params from {params_file}: {exc}", file=sys.stderr)

    print(f"Serving CEREBRUM API on http://localhost:{args.port}/v1/")
    print(f"  Swagger UI: http://localhost:{args.port}/v1/docs")
    ws_port = getattr(args, "ws_port", None)
    if ws_port:
        print(f"  Neural telemetry WebSocket: ws://localhost:{ws_port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="cerebrum",
        description="CEREBRUM — Community-Structured Graph Attention for KG Reasoning",
    )
    parser.add_argument("--max-ram-gb", type=float, default=None, help="Maximum system RAM (GB) before spilling to disk")
    parser.add_argument("--max-vram-gb", type=float, default=None, help="Maximum GPU VRAM (GB) before falling back to CPU")
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

    # init
    ini = sub.add_parser("init", help="Quickstart wizard: load a KB and explore it immediately")
    ini_src = ini.add_mutually_exclusive_group()
    ini_src.add_argument("--from-csv", metavar="PATH", dest="from_csv", help="Path to edge-list CSV")
    ini_src.add_argument("--demo", action="store_true", help="Use the built-in toy KB for instant demo")
    ini.add_argument("--serve", action="store_true", help="Start the REST API server after loading")
    ini.add_argument("--open", action="store_true", help="Start server and open Swagger UI in browser")
    ini.add_argument("--port", type=int, default=8200)

    # ask
    a = sub.add_parser("ask", help="Answer a natural language question using the graph")
    a.add_argument("question", help="Natural language question, e.g. 'What did newton influence?'")
    a.add_argument("--csv", required=True, help="Path to edge-list CSV")
    a.add_argument("--top-k", type=int, default=5, dest="top_k")
    a.add_argument("--beam-width", type=int, default=10, dest="beam_width")
    a.add_argument("--quiet", action="store_true", help="Suppress diagnostic output, print only the answer")

    # tune
    tn = sub.add_parser("tune", help="Live hyperparameter tuner with Rich dashboard (Optuna TPE)")
    tn.add_argument("--n-trials",   type=int, default=100,              dest="n_trials",
                    help="Optuna trials to run (default 100)")
    tn.add_argument("--sample",     type=int, default=500,
                    help="Questions per trial (default 500 ≈ 30s; use 2000 for stability)")
    tn.add_argument("--validate",   type=int, default=0,
                    help="After tuning, validate best on N questions (0=skip; 14274=full dataset)")
    tn.add_argument("--study-name", type=str, default="cerebrum-tuner", dest="study_name",
                    help="Optuna study name shown in dashboard header")
    tn.add_argument("--seed",       type=int, default=42)

    # serve
    s = sub.add_parser("serve", help="Start the REST API server")
    s.add_argument("--csv", required=True, help="Path to edge-list CSV")
    s.add_argument("--port", type=int, default=8200)
    s.add_argument(
        "--ws-port",
        dest="ws_port",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "Start the neural telemetry WebSocket bridge on this port "
            "(e.g. 8765). UE5 CerebrumBrain connects here for real-time "
            "SYNAPTIC_PULSE, NEUROGENESIS, and SYNAPTIC_PRUNE events. "
            "Omit to disable the bridge."
        ),
    )
    s.add_argument("--hierarchical", action="store_true", help="Use hierarchical DSCF for API server")
    s.add_argument("--target-communities", type=int, default=500, dest="target_communities", help="Target communities for hierarchical DSCF")
    s.add_argument("--bridge-bonus", help="JSON or 'key:val,key:val' for default API edge weights")
    s.add_argument("--compliance", action="store_true",
                   help="Enable compliance mode: log every query + full trace to audit file")
    s.add_argument("--audit-log", dest="audit_log", default=None, metavar="FILE",
                   help="Path for JSONL audit log (default: cerebrum_audit.jsonl)")
    s.add_argument(
        "--params-file",
        dest="params_file",
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON file containing learned CSA parameters "
            "(produced by GET /params). Restores the MetaParameterLearner state "
            "at startup so online-learned parameters survive server restarts."
        ),
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "ingest":
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
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()

"""
Hetionet production server — Phase 207/208.

Loads Hetionet with sentence-transformers embeddings, Phase 207 calibrated
params, full community resolution (no 2000-cap coarsening), GraphSAGE
smoothing, and auto-starts the AutonomousDiscoveryLoop for self-repair.

Usage:
    python scripts/hetionet_server.py [--port 8200] [--ws-port 8765]
                                      [--dry-run] [--no-loop]

The autonomous loop will discover missing Compound-binds-Gene edges over time,
lifting the compound_gene_disease H@1 ceiling above 32%.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Phase 207 typed_heterogeneous + sentence-transformers params
# (update after tuner run completes)
# ---------------------------------------------------------------------------
PHASE207_PARAMS = dict(
    trb_factor    = 19.769,
    gamma         = 4.9679,
    beta          = 0.7770,
    r2_boost      = 3.224,
    vote_weight   = 0.6047,
    beam_width    = 8,
    idf_weight    = 0.039,
    branch_bonus  = 0.308,
    fhrb_factor   = 4.658,
)

CACHE_DIR = ROOT / "benchmarks" / "data" / "hetionet" / "cache" / "cerebrum"


def build_graph(embeddings: str = "sentence") -> tuple:
    from benchmarks.hetionet_eval import load_hetionet, download_hetionet
    from core.cerebrum import CerebrumGraph
    from core.embedding_engine import SentenceEngine, RandomEngine

    print("=== Hetionet Server Startup ===\n")
    print("Downloading/loading Hetionet graph...")
    download_hetionet()
    adapter, node_type_map = load_hetionet(use_graph_cache=True)
    print(f"  {adapter.G.number_of_nodes():,} nodes  {adapter.G.number_of_edges():,} edges\n")

    if embeddings == "sentence":
        try:
            engine = SentenceEngine()
            print("  Embeddings: SentenceEngine (384-dim)")
        except ImportError:
            engine = RandomEngine(dim=64)
            print("  Embeddings: RandomEngine (64-dim) — sentence-transformers not installed")
    else:
        from core.embedding_engine import RandomEngine
        engine = RandomEngine(dim=64)
        print("  Embeddings: RandomEngine (64-dim)")

    print("\nBuilding CerebrumGraph (full community resolution)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    graph = CerebrumGraph(adapter=adapter, embedding_engine=engine)
    graph.build(
        cache_dir        = str(CACHE_DIR),
        n_trials         = 1,
        seed             = 42,
        use_graphsage    = True,    # Phase 130: 2-layer neighborhood smoothing
        coarsen_target   = 5568,    # Preserve all Hetionet communities (vs default 2000 cap)
    )
    print(f"  CerebrumGraph built.\n")
    return adapter, graph, node_type_map


def start_server(port: int, ws_port: int, dry_run: bool, no_loop: bool,
                 embeddings: str):
    adapter, graph, node_type_map = build_graph(embeddings=embeddings)

    from api.server import create_app
    from core.autonomous_loop import LoopConfig
    import uvicorn

    app = create_app(
        adapter    = adapter,
        ws_port    = ws_port,
    )

    # Inject the pre-built CerebrumGraph into server state so queries use
    # Phase 207 params immediately without a cold rebuild.
    from api import server as _srv
    _srv._state["graph"]   = graph
    _srv._state["adapter"] = adapter

    if not no_loop:
        # Wire and start the AutonomousDiscoveryLoop.
        # Conservative config: low materialization cap, auto-rollback on circuit trip.
        loop_cfg = LoopConfig(
            cycle_interval                 = 300,    # scan every 5 min
            max_materializations_per_cycle = 5,      # conservative — verify before scaling
            min_approval_rate              = 0.30,   # circuit breaker at 30% approval
            circuit_breaker_window         = 10,
            auto_rollback_on_trip          = True,
            adaptive_tuning                = True,
            dry_run                        = dry_run,
        )
        from core.autonomous_loop import AutonomousDiscoveryLoop
        from core.research_agent import ResearchAgent
        from core.hypothesis_engine import HypothesisEngine

        hyp = HypothesisEngine(adapter=adapter)
        agent = ResearchAgent(adapter=adapter, hypothesis_engine=hyp)
        loop  = AutonomousDiscoveryLoop(agent=agent, config=loop_cfg)

        _srv._state["research_agent"]  = agent
        _srv._state["autonomous_loop"] = loop

        loop.start()
        mode = "DRY RUN" if dry_run else "LIVE"
        print(f"  AutonomousDiscoveryLoop started ({mode})")
        print(f"  Scan interval: {loop_cfg.cycle_interval}s  "
              f"Cap: {loop_cfg.max_materializations_per_cycle}/cycle\n")
    else:
        print("  AutonomousDiscoveryLoop disabled (--no-loop)\n")

    print(f"Server starting on http://0.0.0.0:{port}")
    print(f"  Loop status:  GET  http://localhost:{port}/research/loop/status")
    print(f"  Query:        POST http://localhost:{port}/query\n")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hetionet CEREBRUM server")
    parser.add_argument("--port",       type=int, default=8200)
    parser.add_argument("--ws-port",    type=int, default=8765)
    parser.add_argument("--embeddings", choices=["sentence", "random"],
                        default="sentence")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run loop without materializing edges (safe for testing)")
    parser.add_argument("--no-loop",    action="store_true",
                        help="Disable AutonomousDiscoveryLoop entirely")
    args = parser.parse_args()

    start_server(
        port       = args.port,
        ws_port    = args.ws_port,
        dry_run    = args.dry_run,
        no_loop    = args.no_loop,
        embeddings = args.embeddings,
    )

"""
Feature Impact Benchmark (Phase 77).

Measures the cumulative impact of Phases 55–74 feature additions on KG
reasoning quality.  Uses the toy_graph.csv fixture (21 nodes / 30 edges) for
CI-safe runs, or any CSV graph passed via --graph.

Configurations tested
---------------------
  baseline    : vanilla BeamTraversal, no Engram, no loops
  +engram     : SpeedTalkEngramTraversal (Phase 58) on top of baseline
  +looped     : LoopedBeamTraversal (Phase 70), max_loops=3
  +full       : Engram + LoopedBeam + SpeedTalk together

Metrics per configuration
--------------------------
  Hits@1   — correct answer is rank-1 result
  Hits@5   — correct answer is in top-5
  MRR      — mean reciprocal rank

Query generation
----------------
For each graph entity, existing graph edges are used as the "gold answer":
  seed  = source entity
  query = target entity reached by 1-hop traversal
  gold  = [target entity]

This tests whether the reasoning engine can rediscover existing edges — a
clean lower-bound on traversal quality that requires no external dataset.

Usage
-----
  # Quick run (toy graph, 50 queries):
  python -m benchmarks.feature_impact_benchmark

  # Custom graph, all queries:
  python -m benchmarks.feature_impact_benchmark --graph path/to/graph.csv --sample 0

  # With sentence-transformer embeddings:
  python -m benchmarks.feature_impact_benchmark --embeddings sentence

  # JSON output for CI integration:
  python -m benchmarks.feature_impact_benchmark --json

Notes
-----
  - With random embeddings (default), CSA semantic (alpha) term is noise;
    differences between configs reflect structural path quality.
  - With sentence embeddings, semantic similarity improves all configs but
    the relative deltas are more meaningful.
  - The benchmark is deterministic given the same seed and sample size.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cerebrum import CerebrumGraph

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

TOY_GRAPH = Path(__file__).parent.parent / "tests" / "fixtures" / "toy_graph.csv"


# ---------------------------------------------------------------------------
# Query generation from edge set
# ---------------------------------------------------------------------------

def build_queries(
    adapter,
    sample: Optional[int] = 50,
    seed: int = 42,
    max_hop: int = 1,
) -> List[Tuple[str, str]]:
    """
    Build (seed_entity_id, gold_answer_entity_id) pairs from existing edges.

    For each edge (u, v), the seed is u and the gold answer is v.
    This tests 1-hop rediscovery — the simplest verifiable task.
    """
    # NetworkXAdapter exposes the underlying graph as ._G
    G = getattr(adapter, "_G", None)
    if G is None:
        raise RuntimeError(
            f"Adapter {type(adapter).__name__} does not expose ._G "
            "— cannot enumerate edges for query generation."
        )
    all_edges = [(u, v) for u, v in G.edges()]
    random.seed(seed)
    if sample and sample > 0:
        all_edges = random.sample(all_edges, min(sample, len(all_edges)))
    return [(u, v) for u, v in all_edges if u and v]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    graph: CerebrumGraph,
    queries: List[Tuple[str, str]],
    top_k: int = 10,
    max_hop: int = 2,
    beam_width: int = 10,
    max_loops: int = 1,
) -> Dict[str, float]:
    """
    Run all queries against the graph and return Hits@1, Hits@5, MRR.
    """
    hits1 = 0
    hits5 = 0
    rr_sum = 0.0
    answered = 0

    for seed_id, gold_id in queries:
        try:
            answers = graph.query(
                [seed_id],
                top_k=top_k,
                max_hop=max_hop,
                beam_width=beam_width,
                max_loops=max_loops,
            )
        except Exception:
            continue

        answered += 1
        ids = [a.entity_id for a in answers]

        if gold_id in ids:
            rank = ids.index(gold_id) + 1
            rr_sum += 1.0 / rank
            if rank == 1:
                hits1 += 1
            if rank <= 5:
                hits5 += 1

    if answered == 0:
        return {"hits@1": 0.0, "hits@5": 0.0, "mrr": 0.0, "answered": 0}

    return {
        "hits@1": hits1 / answered,
        "hits@5": hits5 / answered,
        "mrr": rr_sum / answered,
        "answered": answered,
    }


# ---------------------------------------------------------------------------
# Configuration runner
# ---------------------------------------------------------------------------

def run_configuration(
    graph_path: str,
    embedding_type: str,
    config_name: str,
    queries: List[Tuple[str, str]],
    top_k: int = 10,
    max_hop: int = 2,
    beam_width: int = 10,
) -> Dict[str, Any]:
    """Build a CerebrumGraph with the given feature set and evaluate it."""
    use_engram = "+engram" in config_name or "+full" in config_name
    use_looped = "+looped" in config_name or "+full" in config_name
    max_loops = 3 if use_looped else 1

    emb_mode = "sentence" if embedding_type == "sentence" else "random"
    t0 = time.time()

    graph = CerebrumGraph.from_kb(graph_path, embeddings=emb_mode)
    graph.build(seed=42)

    if use_engram:
        try:
            from reasoning.speedtalk_cache import SpeedTalkEngram, SpeedTalkEngramTraversal
            engram = SpeedTalkEngram.from_graph_adapter(graph.adapter)
            graph.attach_engram(engram)
        except Exception:
            pass  # SpeedTalk not available — fall back to no engram

    build_time = time.time() - t0

    t_eval = time.time()
    metrics = compute_metrics(
        graph=graph,
        queries=queries,
        top_k=top_k,
        max_hop=max_hop,
        beam_width=beam_width,
        max_loops=max_loops,
    )
    eval_time = time.time() - t_eval

    return {
        "config": config_name,
        "build_seconds": round(build_time, 2),
        "eval_seconds": round(eval_time, 2),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_table(results: List[Dict[str, Any]]) -> None:
    header = f"{'Config':<14} {'Hits@1':>8} {'Hits@5':>8} {'MRR':>8} {'Answered':>10} {'Eval(s)':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    baseline = None
    for r in results:
        if r["config"] == "baseline":
            baseline = r
        h1 = r["hits@1"]
        h5 = r["hits@5"]
        mrr = r["mrr"]
        delta = ""
        if baseline and r["config"] != "baseline":
            d = mrr - baseline["mrr"]
            delta = f" ({d:+.3f})"
        print(
            f"{r['config']:<14} {h1:>8.3f} {h5:>8.3f} {mrr:>8.3f}{delta:<12} "
            f"{r['answered']:>10} {r['eval_seconds']:>8.2f}s"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CEREBRUM Feature Impact Benchmark (Phase 77)")
    parser.add_argument(
        "--graph",
        default=str(TOY_GRAPH),
        help="Path to graph CSV (default: toy_graph.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of queries to sample (0 = all edges, default: 50)",
    )
    parser.add_argument(
        "--embeddings",
        choices=["random", "sentence"],
        default="random",
        help="Embedding type (default: random)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        help="Beam width for traversal (default: 10)",
    )
    parser.add_argument(
        "--max-hop",
        type=int,
        default=2,
        help="Maximum hop depth (default: 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K answers to retrieve (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a table",
    )
    args = parser.parse_args()

    graph_path = args.graph
    if not Path(graph_path).exists():
        print(f"[ERROR] Graph file not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Phase 77] Feature Impact Benchmark")
    print(f"  Graph     : {graph_path}")
    print(f"  Embeddings: {args.embeddings}")
    print(f"  Sample    : {args.sample if args.sample > 0 else 'all'}")
    print(f"  Beam width: {args.beam_width}  Max hop: {args.max_hop}  Top-K: {args.top_k}")
    print()

    # Build query set once from the base graph (shared across all configs)
    print("Building query set from edge list...", end=" ", flush=True)
    from adapters.file_adapter import load_file_adapter
    base_adapter = load_file_adapter(graph_path)
    queries = build_queries(
        base_adapter,
        sample=args.sample if args.sample > 0 else None,
    )
    print(f"{len(queries)} queries.")

    if not queries:
        print("[ERROR] No queries generated — check graph file format.", file=sys.stderr)
        sys.exit(1)

    configs = ["baseline", "+engram", "+looped", "+full"]
    results = []

    for cfg in configs:
        print(f"  Running config: {cfg} ...", end=" ", flush=True)
        result = run_configuration(
            graph_path=graph_path,
            embedding_type=args.embeddings,
            config_name=cfg,
            queries=queries,
            top_k=args.top_k,
            max_hop=args.max_hop,
            beam_width=args.beam_width,
        )
        results.append(result)
        print(f"Hits@1={result['hits@1']:.3f}  MRR={result['mrr']:.3f}  ({result['eval_seconds']:.2f}s)")

    print()
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)

    # Summary delta line
    if len(results) >= 2:
        baseline_mrr = results[0]["mrr"]
        best = max(results, key=lambda r: r["mrr"])
        delta = best["mrr"] - baseline_mrr
        if delta > 0:
            print(f"\nBest config: {best['config']}  MRR delta vs baseline: +{delta:.3f}")
        else:
            print(f"\nBaseline matched or exceeded all feature configs (delta {delta:+.3f})")


if __name__ == "__main__":
    main()

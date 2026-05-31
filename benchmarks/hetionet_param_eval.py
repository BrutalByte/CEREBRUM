"""
Phase 206: Parametric Hetionet benchmark — tuner-compatible CLI.

Accepts all 9 CEREBRUM tunable parameters via CLI and reports per-hop results
in the exact format parsed by cerebrum_tuner.py:

  {hop}-hop  {n:>7,} {h1:>8.4f} {h10:>9.4f} {mrr:>8.4f}

Performance optimizations vs. hetionet_cerebrum_eval:
  - max_neighbors cap (default 50): Hetionet Gene nodes have 3K+ neighbors;
    capping at 50 cuts per-query time 10-60x with minor H@1 loss.
  - --min-eval-hop 2: skip 1-hop templates (near-ceiling, low tuning signal);
    halves query count per trial.
  - Tiered over-fetch: 1-hop=2x, 2-hop=3x, 3-hop=5x top_k.
  - --workers N: per-question multiprocessing pool; each worker loads its own
    CerebrumGraph from cache.

Param wiring:
  trb_factor   -> terminal_relation_boost (known per-template terminal rel)
  gamma + beta -> SDRB: boost_map scales r2 path-consistency boost amplitude
  vote_weight  -> CerebrumGraph.query(vote_weight=...)
  branch_bonus -> CerebrumGraph.query(branch_bonus_weight=...)
  r2_boost     -> penultimate-chain-relation path-consistency multiplier
  idf_weight   -> post-query IDF frequency penalty on answer entities
  fhrb_factor  -> initial_relation_boost for chain[0] (2+ hop only)
  beam_width   -> traversal beam width

Usage:
  # Tuner-mode (fast): skip 1-hop, 50 questions, max_neighbors=50, 8 workers
  python -u benchmarks/hetionet_param_eval.py \\
      --n-questions 50 --workers 8 --min-eval-hop 2 --max-neighbors 50 \\
      --gamma 2.3 --beta 1.0

  # Full validation (accurate): all hops, 200 questions, max_neighbors=200
  python -u benchmarks/hetionet_param_eval.py \\
      --n-questions 200 --workers 8 --max-neighbors 200
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Over-fetch ratio by hop: enough candidates to survive type-filtering
# without over-fetching on cheap 1-hop queries.
# ---------------------------------------------------------------------------
_OVERFETCH = {1: 2, 2: 3, 3: 5}


# ---------------------------------------------------------------------------
# Worker-side globals (populated by _worker_init in each spawn'd process)
# ---------------------------------------------------------------------------

_W_GRAPH = None


def _worker_init(graph_args: dict) -> None:
    """Load CerebrumGraph from cache inside each worker process."""
    global _W_GRAPH
    from benchmarks.hetionet_eval import load_hetionet
    from benchmarks.hetionet_cerebrum_eval import CACHE_DIR
    from core.cerebrum import CerebrumGraph
    from core.embedding_engine import RandomEngine, SentenceEngine

    adapter, node_type_map = load_hetionet(use_graph_cache=True)
    G = adapter.to_networkx()
    for node, ntype in node_type_map.items():
        if node in G.nodes:
            G.nodes[node]["type"] = ntype

    if graph_args.get("embeddings") == "sentence":
        try:
            engine = SentenceEngine()
        except ImportError:
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)

    graph = CerebrumGraph(
        adapter         = adapter,
        embedding_engine= engine,
        beam_width      = graph_args.get("beam_width", 12),
        max_hop         = 3,
        max_neighbors   = graph_args.get("max_neighbors", 50),
    )
    cache_dir = str(CACHE_DIR / "cerebrum") if graph_args.get("use_cache", True) else None
    graph.build(cache_dir=cache_dir, n_trials=1, seed=graph_args.get("seed", 42),
                community_engine="dscf")
    _W_GRAPH = graph


def _worker_process_question(task: tuple) -> tuple:
    """Process one question. Returns (q_idx, template, h1, h10, mrr, answered, n_typed)."""
    (q_idx, template, seed, correct_answers,
     hop, answer_type, trb, fhrb,
     penultimate_rel, penultimate_idx,
     boost_map, r2_boost, idf_weight, freq_ctr_items,
     vote_weight, branch_bonus, top_k) = task

    if _W_GRAPH is None:
        return (q_idx, template, 0.0, 0.0, 0.0, False, 0)

    freq_ctr  = dict(freq_ctr_items) if freq_ctr_items else {}
    overfetch = _OVERFETCH.get(hop, 5) * top_k

    try:
        answers_obj = _W_GRAPH.query(
            seeds                   = [seed],
            top_k                   = overfetch,
            min_hop                 = hop,
            max_hop                 = hop,
            terminal_relation_boost = trb,
            vote_weight             = vote_weight,
            branch_bonus_weight     = branch_bonus,
            hop_expand              = (hop >= 2),
            initial_relation_boost  = fhrb,
        )
    except Exception:
        return (q_idx, template, 0.0, 0.0, 0.0, False, 0)

    answers_obj = [a for a in answers_obj if a.entity_id.startswith(f"{answer_type}::")]
    n_typed = len(answers_obj)
    if not answers_obj:
        return (q_idx, template, 0.0, 0.0, 0.0, False, 0)

    # r2 path-consistency: boost answers whose penultimate hop uses the
    # expected chain relation; scale boost by SDRB amplitude for that relation.
    if penultimate_rel is not None:
        eff_r2 = boost_map.get(penultimate_rel, r2_boost) if boost_map else r2_boost
        changed = False
        for a in answers_obj:
            bp = a.best_path
            if bp and len(bp.nodes) > penultimate_idx:
                if bp.nodes[penultimate_idx] == penultimate_rel:
                    a.score *= (1.0 + eff_r2)
                    changed = True
        if changed:
            answers_obj.sort(key=lambda a: a.score, reverse=True)

    # IDF hub-entity penalty
    if idf_weight > 0.0 and freq_ctr:
        for a in answers_obj:
            freq = freq_ctr.get(a.entity_id, 1)
            a.score *= 1.0 / (1.0 + idf_weight * math.log1p(freq))
        answers_obj.sort(key=lambda a: a.score, reverse=True)

    pred = [a.entity_id for a in answers_obj][:top_k]

    from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
    return (
        q_idx, template,
        float(hits_at_k(pred, correct_answers, k=1)),
        float(hits_at_k(pred, correct_answers, k=10)),
        float(reciprocal_rank(pred, correct_answers)),
        True,
        n_typed,
    )


# ---------------------------------------------------------------------------
# Template chain configuration
# ---------------------------------------------------------------------------

def _get_template_meta() -> Tuple[Dict, Dict, Dict, Dict]:
    """Return (QA_TEMPLATES_FIXED, TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP, chains)."""
    from benchmarks.hetionet_cerebrum_eval import QA_TEMPLATES_FIXED, TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP
    chains = {name: list(chain) for name, (_, _, _, chain) in QA_TEMPLATES_FIXED.items()}
    return QA_TEMPLATES_FIXED, TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP, chains


# ---------------------------------------------------------------------------
# Answer frequency index (for IDF penalty)
# ---------------------------------------------------------------------------

def _build_answer_freq(
    qa_by_template: Dict[str, List[Tuple]],
    chains: Dict[str, List[str]],
) -> Dict[str, Counter]:
    freq: Dict[str, Counter] = defaultdict(Counter)
    for name, qa_pairs in qa_by_template.items():
        terminal_rel = chains[name][-1]
        for _, correct_answers in qa_pairs:
            for ans in correct_answers:
                freq[terminal_rel][ans] += 1
    return freq


# ---------------------------------------------------------------------------
# Local graph builder (exposes max_neighbors unlike build_cerebrum_graph)
# ---------------------------------------------------------------------------

def _build_graph(adapter, *, beam_width: int, max_neighbors: int,
                 embeddings: str, cache_dir: Optional[Path], seed: int):
    from core.cerebrum import CerebrumGraph
    from core.embedding_engine import RandomEngine, SentenceEngine

    if embeddings == "sentence":
        try:
            engine = SentenceEngine()
            print(f"  Using SentenceEngine ({engine.dim}-dim)")
        except ImportError:
            print("  sentence-transformers not installed — falling back to RandomEngine")
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)
        print(f"  Using RandomEngine (64-dim)")

    graph = CerebrumGraph(
        adapter          = adapter,
        embedding_engine = engine,
        beam_width       = beam_width,
        max_hop          = 3,
        max_neighbors    = max_neighbors,
    )
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    graph.build(
        cache_dir      = str(cache_dir) if cache_dir else None,
        n_trials       = 1,
        seed           = seed,
        community_engine = "dscf",
    )
    print(f"  CerebrumGraph built in {time.time()-t0:.1f}s")
    return graph


# ---------------------------------------------------------------------------
# Serial per-template evaluation (workers=1 path)
# ---------------------------------------------------------------------------

def _eval_template_serial(
    graph, template: str, qa_pairs: List[Tuple],
    boost_map: dict, answer_freq: Dict[str, Counter],
    chains: Dict[str, List[str]],
    TEMPLATE_ANSWER_TYPE: Dict, TEMPLATE_HOP: Dict,
    *, trb_factor: float, r2_boost: float, idf_weight: float,
    vote_weight: float, branch_bonus: float, fhrb_factor: float, top_k: int,
) -> Dict:
    from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank

    hop          = TEMPLATE_HOP[template]
    answer_type  = TEMPLATE_ANSWER_TYPE[template]
    chain        = chains[template]
    terminal_rel = chain[-1]
    trb          = {terminal_rel: trb_factor}
    fhrb         = ({chain[0]: fhrb_factor} if hop >= 2 and fhrb_factor > 0.0 else None)
    penultimate_rel = chain[-2] if hop >= 2 else None
    penultimate_idx = (hop - 1) * 2 - 1 if hop >= 2 else -1
    freq_ctr     = answer_freq.get(terminal_rel)
    overfetch    = _OVERFETCH.get(hop, 5) * top_k

    h1 = h10 = 0
    mrr_sum = 0.0
    n_answered = 0
    n_typed_total = 0
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
            print(f"    [{template}] {i+1}/{len(qa_pairs)} ({time.time()-t0:.1f}s)",
                  end="\r", flush=True)

        answers_obj = graph.query(
            seeds                   = [seed],
            top_k                   = overfetch,
            min_hop                 = hop,
            max_hop                 = hop,
            terminal_relation_boost = trb,
            vote_weight             = vote_weight,
            branch_bonus_weight     = branch_bonus,
            hop_expand              = (hop >= 2),
            initial_relation_boost  = fhrb,
        )
        answers_obj = [a for a in answers_obj if a.entity_id.startswith(f"{answer_type}::")]
        n_typed_total += len(answers_obj)
        if not answers_obj:
            continue

        if penultimate_rel is not None:
            eff_r2 = boost_map.get(terminal_rel, r2_boost) if boost_map else r2_boost
            changed = False
            for a in answers_obj:
                bp = a.best_path
                if bp and len(bp.nodes) > penultimate_idx:
                    if bp.nodes[penultimate_idx] == penultimate_rel:
                        a.score *= (1.0 + eff_r2)
                        changed = True
            if changed:
                answers_obj.sort(key=lambda a: a.score, reverse=True)

        if idf_weight > 0.0 and freq_ctr:
            for a in answers_obj:
                freq = freq_ctr.get(a.entity_id, 1)
                a.score *= 1.0 / (1.0 + idf_weight * math.log1p(freq))
            answers_obj.sort(key=lambda a: a.score, reverse=True)

        pred = [a.entity_id for a in answers_obj][:top_k]
        n_answered += 1
        h1  += hits_at_k(pred, correct_answers, k=1)
        h10 += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    print()
    n = len(qa_pairs)
    return {
        "template": template, "hop": hop,
        "n_total": n, "n_answered": n_answered,
        "hits_1":  h1 / n if n else 0.0,
        "hits_10": h10 / n if n else 0.0,
        "mrr":     mrr_sum / n if n else 0.0,
        "elapsed_s": time.time() - t0,
        "avg_typed": n_typed_total / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 206: Parametric Hetionet benchmark (tuner-compatible)",
    )
    # Run control
    parser.add_argument("--n-questions",   type=int, default=100,
                        help="Questions per template (default 100)")
    parser.add_argument("--top-k",         type=int, default=10)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--embeddings",    choices=["random", "sentence"], default="random")
    parser.add_argument("--use-cache",     action="store_true", default=True)
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--no-download",   action="store_true")
    parser.add_argument("--template",      type=str, default=None,
                        help="Single template to evaluate (default: all)")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Worker processes (default 1). Each loads graph from cache.")
    parser.add_argument("--min-eval-hop",  type=int, default=1,
                        help="Skip templates below this hop depth (default 1 = all). "
                             "Use 2 for tuner runs to skip near-ceiling 1-hop templates.")
    parser.add_argument("--max-neighbors", type=int, default=50,
                        help="Per-node neighbor cap in traversal (default 50). "
                             "Hetionet Gene nodes have 3K+ neighbors — cap cuts query "
                             "time 10-60x. Use 200 for final validation.")
    # All 9 tunable params
    parser.add_argument("--beam-width",    type=int,   default=12)
    parser.add_argument("--trb-factor",    type=float, default=3.0)
    parser.add_argument("--gamma",         type=float, default=2.30)
    parser.add_argument("--beta",          type=float, default=1.0)
    parser.add_argument("--vote-weight",   type=float, default=0.79)
    parser.add_argument("--branch-bonus",  type=float, default=0.17)
    parser.add_argument("--r2-boost",      type=float, default=3.0)
    parser.add_argument("--idf-weight",    type=float, default=0.04)
    parser.add_argument("--fhrb-factor",   type=float, default=1.5)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    print("\n=== CEREBRUM Phase 206: Parametric Hetionet Benchmark ===\n")
    print(f"  beam_width={args.beam_width}  max_neighbors={args.max_neighbors}  "
          f"trb_factor={args.trb_factor}")
    print(f"  gamma={args.gamma}  beta={args.beta}  r2_boost={args.r2_boost}")
    print(f"  vote_weight={args.vote_weight}  branch_bonus={args.branch_bonus}  "
          f"idf_weight={args.idf_weight}  fhrb_factor={args.fhrb_factor}")
    print(f"  n_questions={args.n_questions}  min_eval_hop={args.min_eval_hop}  "
          f"workers={args.workers}  embeddings={args.embeddings}\n")

    from benchmarks.hetionet_eval import (
        JSON_FILE, download_hetionet, load_hetionet, generate_hetionet_qa,
    )
    from benchmarks.hetionet_cerebrum_eval import CACHE_DIR
    from core.relation_boost_deriver import RelationBoostDeriver
    import benchmarks.hetionet_eval as _heval

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not JSON_FILE.exists():
        if args.no_download:
            print("ERROR: hetionet JSON not found and --no-download specified.")
            sys.exit(1)
        download_hetionet()

    print("Loading Hetionet graph...")
    adapter, node_type_map = load_hetionet(use_graph_cache=args.use_cache)
    G = adapter.to_networkx()
    for node, ntype in node_type_map.items():
        if node in G.nodes:
            G.nodes[node]["type"] = ntype
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")

    # ------------------------------------------------------------------
    # Build main-process CerebrumGraph (warms cache for workers)
    # ------------------------------------------------------------------
    print(f"Building CerebrumGraph (max_neighbors={args.max_neighbors})...")
    cache_dir = CACHE_DIR / "cerebrum" if args.use_cache else None
    graph = _build_graph(
        adapter,
        beam_width    = args.beam_width,
        max_neighbors = args.max_neighbors,
        embeddings    = args.embeddings,
        cache_dir     = cache_dir,
        seed          = args.seed,
    )
    print()

    # ------------------------------------------------------------------
    # RelationBoostDeriver
    # ------------------------------------------------------------------
    print("Building RelationBoostDeriver...")
    deriver = RelationBoostDeriver()
    deriver.build_from_triples((u, d["relation"], v) for u, v, d in G.edges(data=True))
    max_fo, mean_fo, _, n_rels = deriver.fan_out_stats()
    boost_map = deriver.boost_map(args.gamma, args.beta) if deriver.is_built else {}
    print(f"  {n_rels} relations  mean_fo={mean_fo:.2f}  max_fo={max_fo:.2f}\n")

    # ------------------------------------------------------------------
    # Template metadata
    # ------------------------------------------------------------------
    QA_TEMPLATES_FIXED, TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP, chains = _get_template_meta()

    if args.template:
        if args.template not in QA_TEMPLATES_FIXED:
            print(f"ERROR: unknown template '{args.template}'. Options: "
                  f"{', '.join(QA_TEMPLATES_FIXED.keys())}")
            sys.exit(1)
        templates = [args.template]
    else:
        templates = [
            name for name in QA_TEMPLATES_FIXED
            if TEMPLATE_HOP[name] >= args.min_eval_hop
        ]

    print(f"Templates to evaluate ({len(templates)}): {', '.join(templates)}\n")

    # ------------------------------------------------------------------
    # Generate QA pairs
    # ------------------------------------------------------------------
    _orig = _heval.QA_TEMPLATES
    _heval.QA_TEMPLATES = QA_TEMPLATES_FIXED
    qa_by_template: Dict[str, List] = {}
    print("Generating QA pairs...")
    for name in templates:
        qa_by_template[name] = generate_hetionet_qa(
            G=G, template=name, n_questions=args.n_questions, seed=args.seed,
        )
        print(f"  {name}: {len(qa_by_template[name])} pairs")
    _heval.QA_TEMPLATES = _orig
    print()

    answer_freq = _build_answer_freq(qa_by_template, chains)

    # ------------------------------------------------------------------
    # Evaluate — parallel or serial
    # ------------------------------------------------------------------
    all_results: List[Dict] = []
    t0_eval = time.time()

    if args.workers > 1:
        # Build per-question task list
        tasks = []
        for name in templates:
            qa_pairs = qa_by_template.get(name, [])
            if not qa_pairs:
                continue
            hop          = TEMPLATE_HOP[name]
            answer_type  = TEMPLATE_ANSWER_TYPE[name]
            chain        = chains[name]
            terminal_rel = chain[-1]
            trb          = {terminal_rel: args.trb_factor}
            fhrb         = ({chain[0]: args.fhrb_factor}
                            if hop >= 2 and args.fhrb_factor > 0.0 else None)
            penultimate_rel = chain[-2] if hop >= 2 else None
            penultimate_idx = (hop - 1) * 2 - 1 if hop >= 2 else -1
            freq_items      = list(answer_freq.get(terminal_rel, {}).items())
            for q_idx, (seed, correct_answers) in enumerate(qa_pairs):
                tasks.append((
                    q_idx, name, seed, correct_answers,
                    hop, answer_type, trb, fhrb,
                    penultimate_rel, penultimate_idx,
                    boost_map, args.r2_boost, args.idf_weight, freq_items,
                    args.vote_weight, args.branch_bonus, args.top_k,
                ))

        graph_args = {
            "use_cache": args.use_cache, "embeddings": args.embeddings,
            "seed": args.seed, "beam_width": args.beam_width,
            "max_neighbors": args.max_neighbors,
        }
        print(f"  Launching {args.workers} workers ({len(tasks)} tasks)...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes   = args.workers,
            initializer = _worker_init,
            initargs    = (graph_args,),
        ) as pool:
            raw = pool.map(_worker_process_question, tasks, chunksize=4)
        print(f"  Workers done in {time.time()-t0_eval:.1f}s\n")

        buckets: Dict[str, Dict] = {
            name: {"h1": 0.0, "h10": 0.0, "mrr": 0.0, "n_answered": 0,
                   "n_typed_total": 0,
                   "n_total": len(qa_by_template[name]), "hop": TEMPLATE_HOP[name]}
            for name in templates if qa_by_template.get(name)
        }
        for (_, tname, h1, h10, mrr, answered, n_typed) in raw:
            if tname in buckets:
                b = buckets[tname]
                b["h1"] += h1; b["h10"] += h10; b["mrr"] += mrr
                b["n_typed_total"] += n_typed
                if answered:
                    b["n_answered"] += 1

        for name, b in buckets.items():
            n = b["n_total"]
            all_results.append({
                "template": name, "hop": b["hop"],
                "n_total": n, "n_answered": b["n_answered"],
                "hits_1":  b["h1"]  / n if n else 0.0,
                "hits_10": b["h10"] / n if n else 0.0,
                "mrr":     b["mrr"] / n if n else 0.0,
                "elapsed_s": time.time() - t0_eval,
                "avg_typed": b["n_typed_total"] / n if n else 0.0,
            })

    else:
        for name in templates:
            qa_pairs = qa_by_template.get(name, [])
            if not qa_pairs:
                print(f"  Skipping {name}: no QA pairs")
                continue
            print(f"  Evaluating {name} ({TEMPLATE_HOP[name]}-hop, "
                  f"{len(qa_pairs)} questions)...")
            all_results.append(_eval_template_serial(
                graph, name, qa_pairs, boost_map, answer_freq, chains,
                TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP,
                trb_factor  = args.trb_factor,
                r2_boost    = args.r2_boost,
                idf_weight  = args.idf_weight,
                vote_weight = args.vote_weight,
                branch_bonus= args.branch_bonus,
                fhrb_factor = args.fhrb_factor,
                top_k       = args.top_k,
            ))

    # ------------------------------------------------------------------
    # Aggregate by hop level
    # ------------------------------------------------------------------
    hop_buckets: Dict[int, List] = defaultdict(list)
    for r in all_results:
        hop_buckets[r["hop"]].append(r)

    hop_metrics: Dict[int, Dict] = {}
    for hop, rows in sorted(hop_buckets.items()):
        total_n = sum(r["n_total"] for r in rows)
        if not total_n:
            continue
        hop_metrics[hop] = {
            "hop":     hop,
            "n_total": total_n,
            "hits_1":  sum(r["hits_1"]  * r["n_total"] for r in rows) / total_n,
            "hits_10": sum(r["hits_10"] * r["n_total"] for r in rows) / total_n,
            "mrr":     sum(r["mrr"]     * r["n_total"] for r in rows) / total_n,
        }

    # ------------------------------------------------------------------
    # Summary — tuner-parseable output
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t0_eval
    print(f"\n=== Results Summary  ({total_elapsed:.0f}s eval) ===\n")
    print(f"  {'Hop':<6} {'N':>7} {'Hits@1':>8} {'Hits@10':>9} {'MRR':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")
    for hop in sorted(hop_metrics):
        m = hop_metrics[hop]
        print(
            f"  {m['hop']}-hop  {m['n_total']:>7,} "
            f"{m['hits_1']:>8.4f} {m['hits_10']:>9.4f} {m['mrr']:>8.4f}"
        )

    if len(templates) > 1:
        print("\n  Per-template breakdown:")
        print(f"  {'Template':<32} {'Hop':>3} {'N':>5} {'H@1':>7} {'H@10':>7} {'MRR':>7} {'AvgTyped':>9}")
        print(f"  {'-'*32} {'-'*3} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")
        for r in sorted(all_results, key=lambda x: (x["hop"], x["template"])):
            avg_t = r.get("avg_typed", 0.0)
            print(
                f"  {r['template']:<32} {r['hop']:>3} {r['n_total']:>5,} "
                f"{r['hits_1']*100:>6.1f}% {r['hits_10']*100:>6.1f}% "
                f"{r['mrr']*100:>6.1f}% {avg_t:>8.1f}"
            )
        # Warn if avg typed candidates is very low (likely cause of H@1≈H@10)
        overall_avg_typed = sum(r.get("avg_typed", 0) * r["n_total"] for r in all_results) / max(sum(r["n_total"] for r in all_results), 1)
        if overall_avg_typed < 2.0:
            print(f"\n  WARNING: avg typed candidates after filter = {overall_avg_typed:.2f} "
                  f"(< 2.0) — H@1≈H@10 is expected. "
                  f"Try --max-neighbors 200 for a more diverse ranked list.")


if __name__ == "__main__":
    mp.freeze_support()
    main()

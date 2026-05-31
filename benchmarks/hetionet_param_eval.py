"""
Phase 206: Parametric Hetionet benchmark — tuner-compatible CLI.

Accepts all 9 CEREBRUM tunable parameters via CLI and reports per-hop results
in the exact format parsed by cerebrum_tuner.py:

  {hop}-hop  {n:>7,} {h1:>8.4f} {h10:>9.4f} {mrr:>8.4f}

The 3-hop line is the objective function the tuner reads.

Param wiring:
  trb_factor   -> terminal_relation_boost weight (known per-template terminal rel)
  gamma + beta -> SDRB: deriver.boost_map(gamma, beta) scales r2 path-consistency boost
  vote_weight  -> CerebrumGraph.query(vote_weight=...)
  branch_bonus -> CerebrumGraph.query(branch_bonus_weight=...)
  r2_boost     -> path-consistency boost for multi-hop (penultimate chain relation check)
  idf_weight   -> post-query IDF frequency penalty on answer entities
  fhrb_factor  -> initial_relation_boost for first chain relation (2+ hop)
  beam_width   -> traversal beam width

Usage:
  python -u benchmarks/hetionet_param_eval.py \\
      --beam-width 12 --trb-factor 3.0 --gamma 2.3 --beta 1.0 \\
      --vote-weight 0.79 --idf-weight 0.04 --branch-bonus 0.17 \\
      --fhrb-factor 1.5 --r2-boost 3.0 --n-questions 100 --workers 8
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
# Worker-side globals (populated by _worker_init)
# ---------------------------------------------------------------------------

_W_GRAPH    = None
_W_ADAPTER  = None
_W_ARGS     = None   # (cache_dir_str, beam_width, seed, embeddings)


def _worker_init(graph_args: dict) -> None:
    """Load CerebrumGraph from cache in each worker process."""
    global _W_GRAPH, _W_ADAPTER, _W_ARGS
    _W_ARGS = graph_args

    from benchmarks.hetionet_eval import load_hetionet
    from benchmarks.hetionet_cerebrum_eval import build_cerebrum_graph, CACHE_DIR

    adapter, node_type_map = load_hetionet(use_graph_cache=True)
    G = adapter.to_networkx()
    for node, ntype in node_type_map.items():
        if node in G.nodes:
            G.nodes[node]["type"] = ntype

    cache_dir = CACHE_DIR / "cerebrum" if graph_args.get("use_cache", True) else None
    _W_GRAPH = build_cerebrum_graph(
        adapter        = adapter,
        embedding_mode = graph_args.get("embeddings", "random"),
        cache_dir      = cache_dir,
        n_trials       = 1,
        seed           = graph_args.get("seed", 42),
        beam_width     = graph_args.get("beam_width", 12),
        max_hop        = 3,
    )
    _W_ADAPTER = adapter


def _worker_process_question(task: tuple) -> tuple:
    """
    Process one question.  Returns (q_idx, template, h1, h10, mrr, answered).
    """
    (q_idx, template, seed, correct_answers,
     hop, answer_type, trb, fhrb, penultimate_rel, penultimate_idx,
     boost_map, r2_boost, idf_weight, freq_ctr_items,
     vote_weight, branch_bonus, top_k) = task

    graph = _W_GRAPH
    if graph is None:
        return (q_idx, template, 0.0, 0.0, 0.0, False)

    # Reconstruct freq_ctr from passed items (Counter not always picklable on old Python)
    freq_ctr = dict(freq_ctr_items) if freq_ctr_items else {}

    try:
        answers_obj = graph.query(
            seeds                   = [seed],
            top_k                   = top_k * 5,
            min_hop                 = hop,
            max_hop                 = hop,
            terminal_relation_boost = trb,
            vote_weight             = vote_weight,
            branch_bonus_weight     = branch_bonus,
            hop_expand              = (hop >= 2),
            initial_relation_boost  = fhrb,
        )
    except Exception:
        return (q_idx, template, 0.0, 0.0, 0.0, False)

    answers_obj = [
        a for a in answers_obj
        if a.entity_id.startswith(f"{answer_type}::")
    ]
    if not answers_obj:
        return (q_idx, template, 0.0, 0.0, 0.0, False)

    # r2 path-consistency + SDRB
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

    # IDF penalty
    if idf_weight > 0.0 and freq_ctr:
        for a in answers_obj:
            freq = freq_ctr.get(a.entity_id, 1)
            a.score *= 1.0 / (1.0 + idf_weight * math.log1p(freq))
        answers_obj.sort(key=lambda a: a.score, reverse=True)

    pred = [a.entity_id for a in answers_obj][:top_k]

    from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
    h1  = float(hits_at_k(pred, correct_answers, k=1))
    h10 = float(hits_at_k(pred, correct_answers, k=10))
    mrr = float(reciprocal_rank(pred, correct_answers))
    return (q_idx, template, h1, h10, mrr, True)


# ---------------------------------------------------------------------------
# Template chain configuration
# ---------------------------------------------------------------------------

def _get_template_chains() -> Dict[str, List[str]]:
    from benchmarks.hetionet_cerebrum_eval import QA_TEMPLATES_FIXED
    return {
        name: list(chain)
        for name, (_, _, _, chain) in QA_TEMPLATES_FIXED.items()
    }


# ---------------------------------------------------------------------------
# Answer frequency index (for IDF penalty)
# ---------------------------------------------------------------------------

def _build_answer_freq(
    qa_by_template: Dict[str, List[Tuple[str, List[str]]]],
    template_chains: Dict[str, List[str]],
) -> Dict[str, Counter]:
    freq: Dict[str, Counter] = defaultdict(Counter)
    for name, qa_pairs in qa_by_template.items():
        terminal_rel = template_chains[name][-1]
        for _, correct_answers in qa_pairs:
            for ans in correct_answers:
                freq[terminal_rel][ans] += 1
    return freq


# ---------------------------------------------------------------------------
# Serial per-template evaluation (workers=1 path)
# ---------------------------------------------------------------------------

def _evaluate_template_serial(
    graph,
    template: str,
    qa_pairs: List[Tuple[str, List[str]]],
    boost_map: dict,
    answer_freq: Dict[str, Counter],
    template_chains: Dict[str, List[str]],
    *,
    trb_factor: float,
    r2_boost: float,
    idf_weight: float,
    vote_weight: float,
    branch_bonus: float,
    top_k: int,
) -> Dict:
    from benchmarks.hetionet_cerebrum_eval import TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP
    from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank

    hop          = TEMPLATE_HOP[template]
    answer_type  = TEMPLATE_ANSWER_TYPE[template]
    chain        = template_chains[template]
    terminal_rel = chain[-1]
    trb          = {terminal_rel: trb_factor}
    fhrb         = {chain[0]: 1.0} if hop >= 2 else None  # fhrb embedded in boost_map path
    penultimate_rel = chain[-2] if hop >= 2 else None
    penultimate_idx = (hop - 1) * 2 - 1 if hop >= 2 else -1
    freq_ctr = answer_freq.get(terminal_rel)

    h1 = h10 = 0
    mrr_sum = 0.0
    n_answered = 0
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
            print(f"    [{template}] {i+1}/{len(qa_pairs)} ({time.time()-t0:.1f}s)",
                  end="\r", flush=True)

        answers_obj = graph.query(
            seeds                   = [seed],
            top_k                   = top_k * 5,
            min_hop                 = hop,
            max_hop                 = hop,
            terminal_relation_boost = trb,
            vote_weight             = vote_weight,
            branch_bonus_weight     = branch_bonus,
            hop_expand              = (hop >= 2),
            initial_relation_boost  = fhrb,
        )
        answers_obj = [a for a in answers_obj if a.entity_id.startswith(f"{answer_type}::")]
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
        h1         += hits_at_k(pred, correct_answers, k=1)
        h10        += hits_at_k(pred, correct_answers, k=10)
        mrr_sum    += reciprocal_rank(pred, correct_answers)

    print()
    n = len(qa_pairs)
    return {
        "template":  template, "hop": hop,
        "n_total":   n, "n_answered": n_answered,
        "hits_1":    h1 / n if n else 0.0,
        "hits_10":   h10 / n if n else 0.0,
        "mrr":       mrr_sum / n if n else 0.0,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 206: Parametric Hetionet benchmark (tuner-compatible)",
    )
    parser.add_argument("--n-questions", type=int,  default=100,
                        help="Questions per template (default 100)")
    parser.add_argument("--top-k",       type=int,  default=10)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--embeddings",  choices=["random", "sentence"], default="random")
    parser.add_argument("--use-cache",   action="store_true", default=True)
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--template",    type=str,  default=None,
                        help="Evaluate a single template (default: all 6)")
    parser.add_argument("--workers",     type=int,  default=1,
                        help="Parallel worker processes (default 1). Each worker loads "
                             "its own CerebrumGraph from cache.")
    # All 9 tunable params
    parser.add_argument("--beam-width",   type=int,   default=12)
    parser.add_argument("--trb-factor",   type=float, default=3.0)
    parser.add_argument("--gamma",        type=float, default=2.30)
    parser.add_argument("--beta",         type=float, default=1.0)
    parser.add_argument("--vote-weight",  type=float, default=0.79)
    parser.add_argument("--branch-bonus", type=float, default=0.17)
    parser.add_argument("--r2-boost",     type=float, default=3.0)
    parser.add_argument("--idf-weight",   type=float, default=0.04)
    parser.add_argument("--fhrb-factor",  type=float, default=1.5)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    print("\n=== CEREBRUM Phase 206: Parametric Hetionet Benchmark ===\n")
    print(f"  beam_width={args.beam_width}  trb_factor={args.trb_factor}  "
          f"gamma={args.gamma}  beta={args.beta}")
    print(f"  vote_weight={args.vote_weight}  branch_bonus={args.branch_bonus}  "
          f"r2_boost={args.r2_boost}")
    print(f"  idf_weight={args.idf_weight}  fhrb_factor={args.fhrb_factor}")
    print(f"  n_questions={args.n_questions}  workers={args.workers}  "
          f"embeddings={args.embeddings}\n")

    from benchmarks.hetionet_eval import JSON_FILE, download_hetionet, load_hetionet, generate_hetionet_qa
    from benchmarks.hetionet_cerebrum_eval import (
        QA_TEMPLATES_FIXED, TEMPLATE_ANSWER_TYPE, TEMPLATE_HOP, CACHE_DIR, build_cerebrum_graph,
    )
    from core.relation_boost_deriver import RelationBoostDeriver
    import benchmarks.hetionet_eval as _heval

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not JSON_FILE.exists():
        if args.no_download:
            print(f"ERROR: {JSON_FILE} not found and --no-download specified.")
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
    # Main-process CerebrumGraph (used for serial path or graph cache warmup)
    # ------------------------------------------------------------------
    print("Building CerebrumGraph (main process)...")
    cache_dir = CACHE_DIR / "cerebrum" if args.use_cache else None
    graph = build_cerebrum_graph(
        adapter        = adapter,
        embedding_mode = args.embeddings,
        cache_dir      = cache_dir,
        n_trials       = 1,
        seed           = args.seed,
        beam_width     = args.beam_width,
        max_hop        = 3,
    )
    print()

    # ------------------------------------------------------------------
    # RelationBoostDeriver (SDRB)
    # ------------------------------------------------------------------
    print("Building RelationBoostDeriver...")
    deriver = RelationBoostDeriver()
    deriver.build_from_triples(
        (u, d["relation"], v) for u, v, d in G.edges(data=True)
    )
    max_fo, mean_fo, _, n_rels = deriver.fan_out_stats()
    boost_map = deriver.boost_map(args.gamma, args.beta) if deriver.is_built else {}
    print(f"  {n_rels} relations  mean_fo={mean_fo:.2f}  max_fo={max_fo:.2f}\n")

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------
    template_chains = _get_template_chains()

    if args.template:
        if args.template not in QA_TEMPLATES_FIXED:
            print(f"ERROR: unknown template '{args.template}'")
            sys.exit(1)
        templates = [args.template]
    else:
        templates = list(QA_TEMPLATES_FIXED.keys())

    # ------------------------------------------------------------------
    # Generate QA pairs
    # ------------------------------------------------------------------
    _orig_templates = _heval.QA_TEMPLATES
    _heval.QA_TEMPLATES = QA_TEMPLATES_FIXED
    qa_by_template: Dict[str, List[Tuple[str, List[str]]]] = {}
    print("Generating QA pairs...")
    for name in templates:
        qa_by_template[name] = generate_hetionet_qa(
            G=G, template=name, n_questions=args.n_questions, seed=args.seed,
        )
        print(f"  {name}: {len(qa_by_template[name])} pairs")
    _heval.QA_TEMPLATES = _orig_templates
    print()

    answer_freq = _build_answer_freq(qa_by_template, template_chains)

    # ------------------------------------------------------------------
    # Evaluate — parallel or serial
    # ------------------------------------------------------------------
    all_results: List[Dict] = []

    if args.workers > 1:
        # Build task list across all templates
        tasks = []
        for name in templates:
            qa_pairs = qa_by_template[name]
            if not qa_pairs:
                continue
            hop          = TEMPLATE_HOP[name]
            answer_type  = TEMPLATE_ANSWER_TYPE[name]
            chain        = template_chains[name]
            terminal_rel = chain[-1]
            trb          = {terminal_rel: args.trb_factor}
            fhrb         = ({chain[0]: args.fhrb_factor} if hop >= 2 and args.fhrb_factor > 0 else None)
            penultimate_rel = chain[-2] if hop >= 2 else None
            penultimate_idx = (hop - 1) * 2 - 1 if hop >= 2 else -1
            freq_ctr_items  = list(answer_freq.get(terminal_rel, {}).items())

            for q_idx, (seed, correct_answers) in enumerate(qa_pairs):
                tasks.append((
                    q_idx, name, seed, correct_answers,
                    hop, answer_type, trb, fhrb,
                    penultimate_rel, penultimate_idx,
                    boost_map, args.r2_boost, args.idf_weight, freq_ctr_items,
                    args.vote_weight, args.branch_bonus, args.top_k,
                ))

        graph_args = {
            "use_cache": args.use_cache, "embeddings": args.embeddings,
            "seed": args.seed, "beam_width": args.beam_width,
        }
        print(f"  Launching {args.workers} worker processes ({len(tasks)} tasks)...")
        t0_all = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes   = args.workers,
            initializer = _worker_init,
            initargs    = (graph_args,),
        ) as pool:
            raw_results = pool.map(_worker_process_question, tasks, chunksize=4)
        print(f"  Workers done in {time.time()-t0_all:.1f}s\n")

        # Aggregate per template
        from collections import defaultdict as _dd
        buckets: Dict[str, Dict] = {}
        for name in templates:
            if qa_by_template.get(name):
                buckets[name] = {
                    "h1": 0.0, "h10": 0.0, "mrr": 0.0, "n_answered": 0,
                    "n_total": len(qa_by_template[name]),
                    "hop": TEMPLATE_HOP[name],
                }
        for (q_idx, tname, h1, h10, mrr, answered) in raw_results:
            if tname not in buckets:
                continue
            b = buckets[tname]
            b["h1"]  += h1
            b["h10"] += h10
            b["mrr"] += mrr
            if answered:
                b["n_answered"] += 1

        for name, b in buckets.items():
            n = b["n_total"]
            all_results.append({
                "template":  name,
                "hop":       b["hop"],
                "n_total":   n,
                "n_answered":b["n_answered"],
                "hits_1":    b["h1"] / n if n else 0.0,
                "hits_10":   b["h10"] / n if n else 0.0,
                "mrr":       b["mrr"] / n if n else 0.0,
                "elapsed_s": time.time() - t0_all,
            })

    else:
        # Serial path
        for name in templates:
            qa_pairs = qa_by_template[name]
            if not qa_pairs:
                print(f"  Skipping {name}: no QA pairs")
                continue
            hop = TEMPLATE_HOP[name]
            print(f"  Evaluating {name} ({hop}-hop, {len(qa_pairs)} questions)...")
            result = _evaluate_template_serial(
                graph         = graph,
                template      = name,
                qa_pairs      = qa_pairs,
                boost_map     = boost_map,
                answer_freq   = answer_freq,
                template_chains = template_chains,
                trb_factor    = args.trb_factor,
                r2_boost      = args.r2_boost,
                idf_weight    = args.idf_weight,
                vote_weight   = args.vote_weight,
                branch_bonus  = args.branch_bonus,
                top_k         = args.top_k,
            )
            all_results.append(result)

    # ------------------------------------------------------------------
    # Aggregate by hop level
    # ------------------------------------------------------------------
    hop_buckets: Dict[int, List[Dict]] = defaultdict(list)
    for r in all_results:
        hop_buckets[r["hop"]].append(r)

    hop_metrics: Dict[int, Dict] = {}
    for hop, rows in sorted(hop_buckets.items()):
        total_n = sum(r["n_total"] for r in rows)
        if total_n == 0:
            continue
        h1  = sum(r["hits_1"]  * r["n_total"] for r in rows) / total_n
        h10 = sum(r["hits_10"] * r["n_total"] for r in rows) / total_n
        mrr = sum(r["mrr"]     * r["n_total"] for r in rows) / total_n
        hop_metrics[hop] = {
            "hop": hop, "n_total": total_n,
            "hits_1": h1, "hits_10": h10, "mrr": mrr,
        }

    # ------------------------------------------------------------------
    # Summary — tuner-parseable output
    # ------------------------------------------------------------------
    print("\n=== Results Summary ===\n")
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
        print(f"  {'Template':<32} {'Hop':>3} {'N':>6} {'H@1':>7} {'H@10':>7} {'MRR':>7}")
        print(f"  {'-'*32} {'-'*3} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
        for r in sorted(all_results, key=lambda x: x["hop"]):
            print(
                f"  {r['template']:<32} {r['hop']:>3} {r['n_total']:>6,} "
                f"{r['hits_1']*100:>6.1f}% {r['hits_10']*100:>6.1f}% "
                f"{r['mrr']*100:>6.1f}%"
            )


if __name__ == "__main__":
    mp.freeze_support()   # required for Windows spawn + PyInstaller
    main()

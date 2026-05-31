"""
Phase 206: Parametric Hetionet benchmark — tuner-compatible CLI.

Accepts all 9 CEREBRUM tunable parameters via CLI and reports per-hop results
in the exact format parsed by cerebrum_tuner.py:

  {hop}-hop  {n:>7,} {h1:>8.4f} {h10:>9.4f} {mrr:>8.4f}

The 3-hop line is the objective function the tuner reads.  Averaging across all
three hop levels gives a fairer cross-hop signal when multiple hops are used.

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
      --fhrb-factor 1.5 --r2-boost 3.0 --n-questions 100
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.hetionet_eval import (
    JSON_FILE,
    DATA_DIR,
    download_hetionet,
    load_hetionet,
    generate_hetionet_qa,
)
from benchmarks.hetionet_cerebrum_eval import (
    QA_TEMPLATES_FIXED,
    TEMPLATE_ANSWER_TYPE,
    TEMPLATE_HOP,
    CACHE_DIR,
    build_cerebrum_graph,
)
from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
from core.relation_boost_deriver import RelationBoostDeriver

# ---------------------------------------------------------------------------
# Template chain configuration
# ---------------------------------------------------------------------------

# Per-template relation chain: [r1, r2?, terminal_rel]
# Used to wire fhrb (r[0]) and r2_boost (r[-2] for multi-hop).
TEMPLATE_CHAIN: Dict[str, List[str]] = {
    name: list(chain)
    for name, (_, _, _, chain) in QA_TEMPLATES_FIXED.items()
}


# ---------------------------------------------------------------------------
# Answer frequency index (for IDF penalty)
# ---------------------------------------------------------------------------

def _build_answer_freq(
    qa_pairs_by_template: Dict[str, List[Tuple[str, List[str]]]]
) -> Dict[str, Counter]:
    """
    Counts how many questions have each entity as a correct answer, keyed by
    terminal relation.  High-frequency = hub entity = penalized by IDF.
    """
    freq: Dict[str, Counter] = defaultdict(Counter)
    for name, qa_pairs in qa_pairs_by_template.items():
        terminal_rel = TEMPLATE_CHAIN[name][-1]
        for _, correct_answers in qa_pairs:
            for ans in correct_answers:
                freq[terminal_rel][ans] += 1
    return freq


# ---------------------------------------------------------------------------
# Per-template parametric evaluation
# ---------------------------------------------------------------------------

def evaluate_template(
    graph,
    template: str,
    qa_pairs: List[Tuple[str, List[str]]],
    deriver: RelationBoostDeriver,
    answer_freq: Dict[str, Counter],
    *,
    trb_factor: float,
    gamma: float,
    beta: float,
    vote_weight: float,
    branch_bonus: float,
    r2_boost: float,
    idf_weight: float,
    fhrb_factor: float,
    top_k: int = 10,
) -> Dict:
    hop          = TEMPLATE_HOP[template]
    answer_type  = TEMPLATE_ANSWER_TYPE[template]
    chain        = TEMPLATE_CHAIN[template]
    terminal_rel = chain[-1]

    # SDRB: per-relation boost = gamma * fan_out(r)^beta
    boost_map = deriver.boost_map(gamma, beta) if deriver.is_built else {}

    # TRB: boost paths ending on the known terminal relation
    trb = {terminal_rel: trb_factor}

    # FHRB: bias first hop toward the chain's opening relation (multi-hop only)
    fhrb: Optional[Dict[str, float]] = None
    if hop >= 2 and fhrb_factor > 0.0:
        fhrb = {chain[0]: fhrb_factor}

    # Path-consistency (r2): check that the penultimate hop uses the expected
    # chain relation.  For 2-hop chain [r1, r_term]: nodes[1]=r1=chain[-2], idx=1
    # For 3-hop chain [r1, r2, r_term]: nodes[3]=r2=chain[-2], idx=3
    penultimate_rel = chain[-2] if hop >= 2 else None
    penultimate_idx = (hop - 1) * 2 - 1 if hop >= 2 else -1

    freq_ctr = answer_freq.get(terminal_rel)

    h1 = h10 = 0
    mrr_sum   = 0.0
    n_answered = 0
    t0         = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
            print(
                f"    [{template}] {i+1}/{len(qa_pairs)} ({time.time()-t0:.1f}s)",
                end="\r", flush=True,
            )

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

        # Filter to expected answer entity type
        answers_obj = [
            a for a in answers_obj
            if a.entity_id.startswith(f"{answer_type}::")
        ]
        if not answers_obj:
            continue

        # r2 path-consistency + SDRB: boost answers whose penultimate hop
        # matches the expected chain relation; scale boost by SDRB amplitude.
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

        # IDF hub-entity penalty
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
        "template":   template,
        "hop":        hop,
        "n_total":    n,
        "n_answered": n_answered,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 206: Parametric Hetionet benchmark (tuner-compatible)",
    )
    # Graph / run control
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
    # All 9 tunable params
    parser.add_argument("--beam-width",  type=int,   default=12)
    parser.add_argument("--trb-factor",  type=float, default=3.0)
    parser.add_argument("--gamma",       type=float, default=2.30)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--vote-weight", type=float, default=0.79)
    parser.add_argument("--branch-bonus",type=float, default=0.17)
    parser.add_argument("--r2-boost",    type=float, default=3.0)
    parser.add_argument("--idf-weight",  type=float, default=0.04)
    parser.add_argument("--fhrb-factor", type=float, default=1.5)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    print("\n=== CEREBRUM Phase 206: Parametric Hetionet Benchmark ===\n")
    print(f"  beam_width={args.beam_width}  trb_factor={args.trb_factor}  "
          f"gamma={args.gamma}  beta={args.beta}")
    print(f"  vote_weight={args.vote_weight}  branch_bonus={args.branch_bonus}  "
          f"r2_boost={args.r2_boost}")
    print(f"  idf_weight={args.idf_weight}  fhrb_factor={args.fhrb_factor}")
    print(f"  n_questions={args.n_questions}  embeddings={args.embeddings}\n")

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
    # Build CerebrumGraph
    # ------------------------------------------------------------------
    print("Building CerebrumGraph...")
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
        (u, d["relation"], v)
        for u, v, d in G.edges(data=True)
    )
    max_fo, mean_fo, _, n_rels = deriver.fan_out_stats()
    print(f"  {n_rels} relations  mean_fo={mean_fo:.2f}  max_fo={max_fo:.2f}\n")

    # ------------------------------------------------------------------
    # Templates to run
    # ------------------------------------------------------------------
    if args.template:
        if args.template not in QA_TEMPLATES_FIXED:
            print(f"ERROR: unknown template '{args.template}'. Options: "
                  f"{', '.join(QA_TEMPLATES_FIXED.keys())}")
            sys.exit(1)
        templates = [args.template]
    else:
        templates = list(QA_TEMPLATES_FIXED.keys())

    # ------------------------------------------------------------------
    # Generate all QA pairs up front (for IDF index)
    # ------------------------------------------------------------------
    import benchmarks.hetionet_eval as _heval
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

    # Build IDF frequency index from correct answers
    answer_freq = _build_answer_freq(qa_by_template)

    # ------------------------------------------------------------------
    # Evaluate each template
    # ------------------------------------------------------------------
    all_results: List[Dict] = []
    for name in templates:
        qa_pairs = qa_by_template[name]
        if not qa_pairs:
            print(f"  Skipping {name}: no QA pairs")
            continue
        hop = TEMPLATE_HOP[name]
        print(f"  Evaluating {name} ({hop}-hop, {len(qa_pairs)} questions)...")
        result = evaluate_template(
            graph        = graph,
            template     = name,
            qa_pairs     = qa_pairs,
            deriver      = deriver,
            answer_freq  = answer_freq,
            trb_factor   = args.trb_factor,
            gamma        = args.gamma,
            beta         = args.beta,
            vote_weight  = args.vote_weight,
            branch_bonus = args.branch_bonus,
            r2_boost     = args.r2_boost,
            idf_weight   = args.idf_weight,
            fhrb_factor  = args.fhrb_factor,
            top_k        = args.top_k,
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
        # Weighted average by n_total
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

    # Per-template detail
    if len(templates) > 1:
        print("\n  Per-template breakdown:")
        print(f"  {'Template':<32} {'Hop':>3} {'N':>6} {'H@1':>7} {'H@10':>7} {'MRR':>7}")
        print(f"  {'-'*32} {'-'*3} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
        for r in all_results:
            print(
                f"  {r['template']:<32} {r['hop']:>3} {r['n_total']:>6,} "
                f"{r['hits_1']*100:>6.1f}% {r['hits_10']*100:>6.1f}% "
                f"{r['mrr']*100:>6.1f}%"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation study for the CEREBRUM flagship paper — MetaQA 3-hop H@1.

Runs 5 configurations on MetaQA 3-hop test set and measures Hits@1.
Results replace the $^\star$ placeholder rows in:
  research/papers/01-flagship/cerebrum-flagship.tex  (ablation table, lines 271-276)

Configurations (all: sentence embeddings, beam_width=20, hop=3, no TRB/PRB):
  full          TSC + GraphSAGE + BridgeTwinEngine + Bayesian beam
  no_bridge     Same, BridgeTwinEngine removed
  no_adaptive   Same, Bayesian beam disabled (standard beam)
  dscf          DSCF community detection instead of TSC (other features as full)
  no_graphsage  TSC, no GraphSAGE (other features as full)

The "Full CEREBRUM" row in the ablation table is re-measured by this script.
The Phase 53 canonical 12.5% (used in the comparison-against-SOTA table) is
measured WITHOUT GraphSAGE and without BridgeTwinEngine — it is NOT the same
as "Full CEREBRUM" in the ablation table. This is consistent: the ablation
table shows component contributions within v2.51.1; the comparison table shows
training-free performance against supervised SOTA.

Usage:
    # Quick sanity run (500 questions, ~3 min)
    python scripts/run_ablation_metaqa.py --sample 500

    # Full run (14,274 questions per config, ~90 min on RTX 5090)
    python scripts/run_ablation_metaqa.py

    # Full run + write values into paper files
    python scripts/run_ablation_metaqa.py --update-paper

    # Run only specific configs
    python scripts/run_ablation_metaqa.py --configs full no_graphsage
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.metaqa_eval import load_qa, hits_at_k, reciprocal_rank
from core.cerebrum import CerebrumGraph

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"
KB_FILE   = DATA_DIR / "kb.txt"

FLAGSHIP_TEX_PATHS = [
    Path(__file__).resolve().parent.parent / "research" / "papers" / "01-flagship" / "cerebrum-flagship.tex",
    Path(__file__).resolve().parent.parent / "research" / "papers" / "01-flagship" / "arxiv_submission" / "cerebrum-flagship.tex",
]

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "full": {
        "label":       "Full CEREBRUM",
        "community":   "tsc",
        "graphsage":   True,
        "bridge":      True,
        "bayesian":    True,
        "warm_start":  0.5,
    },
    "no_bridge": {
        "label":       "$-$ Bridge Twins",
        "community":   "tsc",
        "graphsage":   True,
        "bridge":      False,
        "bayesian":    True,
        "warm_start":  0.5,
    },
    "no_adaptive": {
        "label":       "$-$ Adaptive Beam",
        "community":   "tsc",
        "graphsage":   True,
        "bridge":      True,
        "bayesian":    False,
        "warm_start":  0.0,
    },
    "dscf": {
        "label":       "DSCF instead of TSC",
        "community":   "dscf",
        "graphsage":   True,
        "bridge":      True,
        "bayesian":    True,
        "warm_start":  0.5,
    },
    "no_graphsage": {
        "label":       "$-$ GraphSAGE smoothing",
        "community":   "tsc",
        "graphsage":   False,
        "bridge":      True,
        "bayesian":    True,
        "warm_start":  0.5,
    },
}

CONFIG_ORDER = ["full", "no_bridge", "no_adaptive", "dscf", "no_graphsage"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    cfg_name: str,
    qa_pairs: List[Tuple],
    top_k: int = 10,
    beam_width: int = 20,
    sample: Optional[int] = None,
) -> Dict:
    cfg = CONFIGS[cfg_name]
    print(f"\n  [{cfg_name}] Building graph ({cfg['community']}, "
          f"graphsage={cfg['graphsage']}, bridge={cfg['bridge']}, "
          f"bayesian={cfg['bayesian']})...")

    t_build = time.time()
    graph = CerebrumGraph.from_kb(
        KB_FILE,
        sep              = "|",
        directed         = False,
        embeddings       = "sentence",
        beam_width       = beam_width,
        max_hop          = 3,
        max_neighbors    = 100,
        probabilistic    = cfg["bayesian"],
        warm_start_strength = cfg["warm_start"],
    )

    graph.build(
        cache_dir           = CACHE_DIR,
        force_rebuild       = False,
        seed                = 42,
        use_graphsage       = cfg["graphsage"],
        community_engine    = cfg["community"],
    )
    print(f"    built in {time.time()-t_build:.1f}s  "
          f"({graph.community_count} communities)")

    # Attach BridgeTwinEngine when requested
    if cfg["bridge"] and graph._traversal is not None:
        try:
            from core.bridge_engine import BridgeTwinEngine
            bt = BridgeTwinEngine(n_min=5, similarity_threshold=0.65)
            graph._traversal.bridge_engine = bt
        except Exception as e:
            print(f"    WARN: could not attach BridgeTwinEngine: {e}")

    h1_sum = h10_sum = mrr_sum = 0
    n_answered = 0
    pairs = qa_pairs[:sample] if sample else qa_pairs
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(pairs):
        if (i + 1) % 500 == 0 or (i + 1) == len(pairs):
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(pairs):,}  H@1={h1_sum/(n_answered or 1):.3f}  "
                  f"({elapsed:.0f}s)", end="\r")

        try:
            answers = graph.query(
                seeds   = [seed],
                top_k   = top_k,
                min_hop = 1,
                max_hop = 3,
                beam_width = beam_width,
            )
        except Exception:
            continue

        if not answers:
            continue

        pred = [a.entity_id for a in answers]
        n_answered += 1
        h1_sum  += hits_at_k(pred, correct_answers, k=1)
        h10_sum += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    n_total = len(pairs)
    print()
    return {
        "config":    cfg_name,
        "label":     cfg["label"],
        "n_total":   n_total,
        "n_answered": n_answered,
        "hits_1":    h1_sum  / n_total,
        "hits_10":   h10_sum / n_total,
        "mrr":       mrr_sum / n_total,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Paper update
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v*100:.1f}\\%"


def _fmt_delta(v: float, ref: float) -> str:
    d = (v - ref) * 100
    if abs(d) < 0.05:
        return "---"
    return f"${d:+.1f}$"


def update_paper(results: List[Dict]) -> None:
    by_name = {r["config"]: r for r in results}
    if "full" not in by_name:
        print("WARN: 'full' config not in results — cannot update paper.")
        return

    full_h1 = by_name["full"]["hits_1"]

    # Build replacement table body lines
    # Order: full, no_bridge, no_adaptive, dscf, no_graphsage, bfs (unchanged)
    def row(cfg: str) -> str:
        r = by_name[cfg]
        h1_str = _fmt_pct(r["hits_1"])
        delta  = _fmt_delta(r["hits_1"], full_h1)
        label  = r["label"]
        return f"{label:<40} & {h1_str:>12} & {delta:>10} \\\\"

    bfs_row = r"BFS baseline            &  1.1\%   & $-$11.4 \\"

    lines = [
        r"Full \CEREBRUM{}        & " + _fmt_pct(full_h1) + r"   & --- \\",
    ]
    for cfg in ["no_bridge", "no_adaptive", "dscf", "no_graphsage"]:
        if cfg in by_name:
            lines.append(row(cfg))
        else:
            lines.append(f"% MISSING: {cfg}")
    lines.append(bfs_row)

    new_body = "\n".join(lines)

    # Pattern to find and replace the table midrule block
    pattern = re.compile(
        r"(\\midrule\n)"                             # \midrule
        r"(Full \\CEREBRUM\{\}.*?)"                  # everything from Full CEREBRUM
        r"(BFS baseline.*?\\\\)"                     # to BFS baseline row
        r"(\n\\bottomrule)",                         # \bottomrule
        re.DOTALL,
    )

    for tex_path in FLAGSHIP_TEX_PATHS:
        if not tex_path.exists():
            print(f"  SKIP {tex_path} (not found)")
            continue
        text = tex_path.read_text(encoding="utf-8")
        new_text, n = pattern.subn(
            r"\1" + new_body + r"\4",
            text,
        )
        if n == 0:
            print(f"  WARN: pattern not matched in {tex_path.name} — manual update needed")
            print("  Expected table body:")
            print(new_body)
        else:
            # Also remove the action-item note
            new_text = re.sub(
                r"\\noindent\\emph\{Action for camera-ready:.*?\}",
                r"\\noindent\\emph{Ablation results measured with MetaQA 3-hop test set, "
                r"sentence-transformer embeddings, beam\\_width=20, no TRB.}",
                new_text,
                flags=re.DOTALL,
            )
            tex_path.write_text(new_text, encoding="utf-8")
            print(f"  UPDATED {tex_path}")


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results: List[Dict], full_h1: float) -> None:
    header = f"  {'Config':<18} {'H@1':>7} {'H@10':>7} {'MRR':>7} {'Delta H@1':>10} {'Elapsed':>9}"
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        d = (r["hits_1"] - full_h1) * 100
        delta_str = f"{d:+.1f}%" if r["config"] != "full" else "---"
        print(
            f"  {r['config']:<18} "
            f"{r['hits_1']*100:>6.2f}% "
            f"{r['hits_10']*100:>6.2f}% "
            f"{r['mrr']*100:>6.2f}% "
            f"{delta_str:>10} "
            f"{r['elapsed_s']:>7.0f}s"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate on N questions (default: all 14,274)")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS.keys()),
                        default=CONFIG_ORDER,
                        help="Which configs to run (default: all)")
    parser.add_argument("--update-paper", action="store_true",
                        help="Write measured values into the flagship paper .tex files")
    args = parser.parse_args()

    if not KB_FILE.exists():
        print(f"ERROR: MetaQA kb.txt not found at {KB_FILE}", file=sys.stderr)
        sys.exit(1)

    print("=== CEREBRUM Ablation Study — MetaQA 3-hop ===\n")
    print(f"  Sample     : {args.sample if args.sample else 'all (14,274)'}")
    print(f"  Beam width : {args.beam_width}")
    print(f"  Configs    : {args.configs}")

    print("\nLoading MetaQA 3-hop test set...")
    qa_pairs = load_qa(hop=3, sample=args.sample, seed=42)
    print(f"  {len(qa_pairs):,} questions loaded.")

    results = []
    for cfg_name in args.configs:
        r = evaluate_config(
            cfg_name   = cfg_name,
            qa_pairs   = qa_pairs,
            top_k      = args.top_k,
            beam_width = args.beam_width,
        )
        results.append(r)
        print(f"  [{cfg_name}] H@1={r['hits_1']*100:.2f}%  "
              f"H@10={r['hits_10']*100:.2f}%  MRR={r['mrr']*100:.2f}%")

    # Find full config H@1 for delta calculations
    full_h1 = next(
        (r["hits_1"] for r in results if r["config"] == "full"),
        results[0]["hits_1"] if results else 0.0,
    )

    print(f"\n\n{'='*70}")
    print("Results:")
    print_table(results, full_h1)

    if args.update_paper:
        print("\nUpdating paper files...")
        update_paper(results)
        print("Done. Re-run publication_preflight.py to verify.")
    else:
        print("\n(Re-run with --update-paper to write values into paper files)")


if __name__ == "__main__":
    main()

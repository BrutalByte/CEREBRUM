"""
Phase 184 Diagnostic — CEREBRUM MetaQA 3-hop miss-cause analysis.

Classifies each 3-hop question into one of five categories to determine
which structural miss cause dominates before implementing fixes:

  hit_h1                   -- correct answer is top-1 (Hits@1 hit)
  hit_h10                  -- correct answer is in top-10 but not top-1
  h10_miss_beam_coverage   -- correct answer never reached in top-100 beam
  h1_miss_filter           -- correct in beam, removed by answer-type filter
  h1_miss_vote_convergence -- correct in beam, survives filter, ranked below top-1

Usage
-----
  python -m benchmarks.phase184_diagnose
  python -m benchmarks.phase184_diagnose --sample 500 --workers 8 --out phase184_diag.csv
  python -m benchmarks.phase184_diagnose --jsonl existing.jsonl --out reclassify.csv

The script invokes metaqa_eval with --diagnose-jsonl, reads the emitted JSONL,
classifies each row, writes a CSV, and prints a summary table with fix hints.

Re-classify without re-running:
  python -m benchmarks.phase184_diagnose --jsonl phase184_raw.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _truthy(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def classify_row(row: Dict) -> str:
    """
    Decision tree (priority order):

    1. final_rank == 1                                        -> hit_h1
    2. 2 <= final_rank <= 10                                  -> hit_h10
    3. in_beam_top100 == False                                -> h10_miss_beam_coverage
    4. filter_applied AND NOT filter_fell_back
       AND NOT correct_in_filtered                            -> h1_miss_filter
    5. else                                                   -> h1_miss_vote_convergence
    """
    final_rank       = int(row.get("final_rank", 0))
    in_beam          = _truthy(row.get("in_beam_top100", False))
    filter_applied   = _truthy(row.get("filter_applied", False))
    filter_fell_back = _truthy(row.get("filter_fell_back", False))
    correct_in_filt  = _truthy(row.get("correct_in_filtered", False))

    if final_rank == 1:
        return "hit_h1"
    if 2 <= final_rank <= 10:
        return "hit_h10"
    if not in_beam:
        return "h10_miss_beam_coverage"
    if filter_applied and not filter_fell_back and not correct_in_filt:
        return "h1_miss_filter"
    return "h1_miss_vote_convergence"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "q_idx",
    "question_text",
    "seed",
    "correct_answer",
    "all_correct_answers",
    "predicted_top10",
    "h1_hit",
    "h10_hit",
    "beam_rank",
    "n_candidates_before_filter",
    "n_candidates_after_filter",
    "correct_in_filtered",
    "filter_applied",
    "filter_fell_back",
    "detected_relation",
    "top1_answer",
    "top1_score",
    "correct_answer_score",
    "miss_category",
]


def build_output_row(raw: Dict, category: str) -> Dict:
    final_rank = int(raw.get("final_rank", 0))
    h1_hit  = (final_rank == 1)
    h10_hit = (1 <= final_rank <= 10)
    return {
        "q_idx":                      raw.get("q_idx", ""),
        "question_text":              raw.get("question_text", ""),
        "seed":                       raw.get("seed", ""),
        "correct_answer":             raw.get("correct_answer", ""),
        "all_correct_answers":        raw.get("all_correct_answers", ""),
        "predicted_top10":            raw.get("predicted_top10", ""),
        "h1_hit":                     h1_hit,
        "h10_hit":                    h10_hit,
        "beam_rank":                  raw.get("beam_rank", 0),
        "n_candidates_before_filter": raw.get("n_candidates_before_filter", ""),
        "n_candidates_after_filter":  raw.get("n_filtered", ""),
        "correct_in_filtered":        _truthy(raw.get("correct_in_filtered", False)),
        "filter_applied":             _truthy(raw.get("filter_applied", False)),
        "filter_fell_back":           _truthy(raw.get("filter_fell_back", False)),
        "detected_relation":          raw.get("detected_rel", ""),
        "top1_answer":                raw.get("top1_answer", ""),
        "top1_score":                 round(float(raw.get("top1_score", 0.0)), 6),
        "correct_answer_score":       round(float(raw.get("correct_answer_score", 0.0)), 6),
        "miss_category":              category,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_CATEGORY_ORDER = [
    "hit_h1",
    "hit_h10",
    "h10_miss_beam_coverage",
    "h1_miss_filter",
    "h1_miss_vote_convergence",
]

_CATEGORY_LABELS = {
    "hit_h1":                   "H@1 hit                                            ",
    "hit_h10":                  "H@10 hit (not H@1)                                 ",
    "h10_miss_beam_coverage":   "H@10 MISS -- beam coverage (correct not in top-100)",
    "h1_miss_filter":           "H@1  MISS -- type filter removed correct answer    ",
    "h1_miss_vote_convergence": "H@1  MISS -- vote convergence (ranked below top-1) ",
}

_FIX_HINTS = {
    "h10_miss_beam_coverage":   "fix: increase --expansion-k (currently drops ~60% of hop-1 entities)",
    "h1_miss_filter":           "fix: TRB mis-detection or soften --min-filter-size fallback",
    "h1_miss_vote_convergence": "fix: branch-diversity bonus scaling / penultimate-hop relation penalty",
}


def print_summary(rows: List[Dict]) -> None:
    n_total = len(rows)
    if n_total == 0:
        print("\n  No rows to summarize.")
        return

    counts: Dict[str, int] = {cat: 0 for cat in _CATEGORY_ORDER}
    for row in rows:
        cat = row["miss_category"]
        counts[cat] = counts.get(cat, 0) + 1

    print("\n" + "=" * 74)
    print("  Phase 184 Miss-Cause Distribution")
    print("=" * 74)
    print(f"  {'Category':<51} {'N':>6}  {'%':>6}")
    print(f"  {'-'*51} {'-'*6}  {'-'*6}")
    for cat in _CATEGORY_ORDER:
        n   = counts.get(cat, 0)
        pct = 100.0 * n / n_total
        print(f"  {_CATEGORY_LABELS[cat]} {n:>6,}  {pct:>5.1f}%")
    print(f"  {'-'*51} {'-'*6}  {'-'*6}")
    print(f"  {'TOTAL':<51} {n_total:>6,}  100.0%")
    print("=" * 74)

    miss_beam   = counts.get("h10_miss_beam_coverage", 0)
    miss_filter = counts.get("h1_miss_filter", 0)
    miss_vote   = counts.get("h1_miss_vote_convergence", 0)
    miss_total  = miss_beam + miss_filter + miss_vote

    if miss_total > 0:
        print("\n  Miss breakdown (relative to all misses):")
        for cat in ["h10_miss_beam_coverage", "h1_miss_filter", "h1_miss_vote_convergence"]:
            n   = counts.get(cat, 0)
            pct = 100.0 * n / miss_total
            print(f"    {pct:5.1f}%  {_FIX_HINTS[cat]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 184: CEREBRUM 3-hop miss-cause diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", type=int, default=500, metavar="N",
                        help="3-hop questions to evaluate (default: 500; 0 = full set).")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, metavar="N",
                        help="Workers passed to metaqa_eval (default: 8; 1 = serial).")
    parser.add_argument("--out",     type=str, default="phase184_diag.csv", metavar="PATH",
                        help="Output CSV path (default: phase184_diag.csv).")
    parser.add_argument("--keep-jsonl", action="store_true", default=False,
                        help="Keep the intermediate JSONL file after processing.")
    parser.add_argument("--jsonl",   type=str, default=None, metavar="PATH",
                        help="Skip evaluation and classify an existing JSONL file.")
    parser.add_argument("--python",  type=str, default=None, metavar="EXE",
                        help="Python executable for metaqa_eval subprocess.")
    # Any unrecognised args are forwarded verbatim to metaqa_eval
    args, extra_args = parser.parse_known_args()

    project_root = Path(__file__).parent.parent.resolve()
    python_exe   = args.python or sys.executable

    # ------------------------------------------------------------------
    # Step 1: obtain the JSONL
    # ------------------------------------------------------------------
    if args.jsonl:
        jsonl_path = args.jsonl
        tmp_jsonl  = None
        print(f"[phase184] Using existing JSONL: {jsonl_path}")
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="phase184_")
        os.close(tmp_fd)
        jsonl_path = tmp_path
        tmp_jsonl  = None if args.keep_jsonl else tmp_path

        sample_str = str(args.sample) if args.sample else "full test set"
        print(f"\n[phase184] Evaluating {sample_str} 3-hop questions "
              f"(seed={args.seed}, workers={args.workers})...")
        if extra_args:
            print(f"           extra args: {' '.join(extra_args)}")

        cmd = [
            python_exe, "-m", "benchmarks.metaqa_eval",
            "--hop", "3",
            "--seed", str(args.seed),
            "--workers", str(args.workers),
            "--diagnose-jsonl", jsonl_path,
        ]
        if args.sample:
            cmd += ["--sample", str(args.sample)]
        cmd += extra_args

        print(f"\n[phase184] {' '.join(cmd)}\n")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"\n[phase184] ERROR: metaqa_eval exited {result.returncode}", file=sys.stderr)
            if tmp_jsonl and os.path.exists(tmp_jsonl):
                os.unlink(tmp_jsonl)
            sys.exit(result.returncode)
        print(f"\n[phase184] Evaluation done ({time.time()-t0:.1f}s). Reading JSONL...")

    # ------------------------------------------------------------------
    # Step 2: read JSONL
    # ------------------------------------------------------------------
    if not os.path.exists(jsonl_path):
        print(f"\n[phase184] ERROR: JSONL not found: {jsonl_path}\n"
              "  Did metaqa_eval have Phase 184 --diagnose-jsonl support?", file=sys.stderr)
        sys.exit(1)

    raw_rows: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as jf:
        for lineno, line in enumerate(jf, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[phase184] WARNING: bad JSON line {lineno}: {exc}", file=sys.stderr)

    if not raw_rows:
        print("[phase184] ERROR: JSONL is empty.", file=sys.stderr)
        if tmp_jsonl and os.path.exists(tmp_jsonl):
            os.unlink(tmp_jsonl)
        sys.exit(1)

    print(f"[phase184] {len(raw_rows):,} diagnostic rows read.")

    # ------------------------------------------------------------------
    # Step 3: classify
    # ------------------------------------------------------------------
    output_rows: List[Dict] = []
    for raw in raw_rows:
        category = classify_row(raw)
        output_rows.append(build_output_row(raw, category))

    output_rows.sort(key=lambda r: int(r["q_idx"]) if str(r["q_idx"]).isdigit() else 0)

    # ------------------------------------------------------------------
    # Step 4: write CSV
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[phase184] Wrote {len(output_rows):,} rows -> {out_path}")

    # ------------------------------------------------------------------
    # Step 5: summary
    # ------------------------------------------------------------------
    print_summary(output_rows)

    if tmp_jsonl and os.path.exists(tmp_jsonl):
        try:
            os.unlink(tmp_jsonl)
        except OSError:
            pass


if __name__ == "__main__":
    main()

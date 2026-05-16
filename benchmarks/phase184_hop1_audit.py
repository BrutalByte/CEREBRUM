"""
Phase 184 Hop-1 Ranking Audit — beam-coverage miss root-cause analysis.

For each question classified as h10_miss_beam_coverage in phase184_diag.csv,
determines WHERE in the H1SE hop-1 sorted list the correct-path entity falls:

  rank <= 20   -- entity IS explored (top-k_eff), problem is in stage-2
  rank 21-25   -- would be recovered by expansion_k=25
  rank 26-30   -- would be recovered by expansion_k=30
  rank > 30    -- hop-1 scoring is fundamentally wrong for these questions
  no viable    -- correct answer unreachable in exactly 3 hops from seed

Usage
-----
  python -m benchmarks.phase184_hop1_audit
  python -m benchmarks.phase184_hop1_audit --diag phase184_diag.csv --n 71
  python -m benchmarks.phase184_hop1_audit --out hop1_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Shared constants from metaqa_eval
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.metaqa_eval import (
    KB_FILE, CACHE_DIR, QA_FILES,
    detect_target_relation,
)
from core.cerebrum import CerebrumGraph
from reasoning.expanded_traversal import set_hop1_audit_cb, clear_hop1_audit_cb

# r3 -> most-common-r2 map (from canonical Phase 182 training data scan)
_R2_FOR_R3 = {
    "directed_by":   "written_by",
    "written_by":    "written_by",
    "starred_actors":"written_by",
    "release_year":  "has_genre",
    "has_genre":     "has_genre",
    "in_language":   "written_by",
}


# ---------------------------------------------------------------------------
# 2-hop reachability check
# ---------------------------------------------------------------------------

def can_reach_2hops(adapter, start: str, correct_set: Set[str],
                    forbidden: Set[str], max_neighbors: int = 100) -> bool:
    """True if any 2-hop path from start reaches an entity in correct_set."""
    for e2 in adapter.get_neighbors(start, max_neighbors=max_neighbors):
        h2 = e2.target_id
        if h2 in forbidden or h2 == start:
            continue
        for e3 in adapter.get_neighbors(h2, max_neighbors=max_neighbors):
            if e3.target_id in correct_set:
                return True
    return False


def first_viable_rank(
    sorted_neighbors: List[str],
    correct_set: Set[str],
    adapter,
    seed: str,
    max_neighbors: int = 100,
) -> int:
    """
    Return the 1-based rank of the first hop-1 entity that can reach a
    correct answer in 2 more hops. Returns 0 if none found.
    """
    forbidden = {seed}
    for rank, entity in enumerate(sorted_neighbors, 1):
        if can_reach_2hops(adapter, entity, correct_set, forbidden, max_neighbors):
            return rank
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 184 Hop-1 Ranking Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--diag", type=str, default="phase184_diag.csv",
                        help="phase184_diagnose.py output CSV (default: phase184_diag.csv)")
    parser.add_argument("--out",  type=str, default="phase184_hop1_audit.csv",
                        help="Output CSV (default: phase184_hop1_audit.csv)")
    parser.add_argument("--n",    type=int, default=None,
                        help="Max miss questions to audit (default: all)")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--expansion-k", type=int, default=20)
    parser.add_argument("--fhrb-factor", type=float, default=3.0)
    parser.add_argument("--embeddings", type=str, default="sentence",
                        choices=["none", "sentence", "bge"])
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load beam-miss rows from diagnostic CSV
    # ------------------------------------------------------------------
    diag_path = Path(args.diag)
    if not diag_path.exists():
        print(f"ERROR: diagnostic CSV not found: {diag_path}", file=sys.stderr)
        print("  Run: python -m benchmarks.phase184_diagnose first.", file=sys.stderr)
        sys.exit(1)

    miss_rows: List[Dict] = []
    with open(diag_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("miss_category") == "h10_miss_beam_coverage":
                miss_rows.append(row)

    if not miss_rows:
        print("No h10_miss_beam_coverage rows found in diagnostic CSV.")
        sys.exit(0)

    if args.n:
        miss_rows = miss_rows[:args.n]

    print(f"\n[hop1-audit] {len(miss_rows)} beam-coverage miss questions to audit.")

    # ------------------------------------------------------------------
    # Build graph (from cache — fast)
    # ------------------------------------------------------------------
    print("[hop1-audit] Loading knowledge graph...")
    t0 = time.time()
    graph = CerebrumGraph.from_kb(
        KB_FILE,
        sep="|",
        directed=False,
        embeddings=args.embeddings,
        beam_width=args.beam_width,
        max_hop=3,
        max_neighbors=100,
    )
    graph.build(
        cache_dir=CACHE_DIR,
        force_rebuild=False,
        seed=42,
    )
    print(f"[hop1-audit] Graph ready ({time.time()-t0:.1f}s).")

    # ------------------------------------------------------------------
    # Load KB relations for TRB/FHRB detection
    # ------------------------------------------------------------------
    try:
        kb_relations: List[str] = list({
            e.relation_type
            for eid in list(graph.adapter._G.nodes())[:500]
            for e in graph.adapter.get_neighbors(eid)
        })
    except Exception:
        kb_relations = []

    # ------------------------------------------------------------------
    # Load sentence encoder for query embeddings (same as canonical run)
    # ------------------------------------------------------------------
    query_engine = None
    if args.embeddings != "none":
        try:
            from core.embeddings import SentenceEmbeddingEngine
            query_engine = SentenceEmbeddingEngine()
            print("[hop1-audit] Sentence encoder ready.")
        except Exception as exc:
            print(f"[hop1-audit] WARNING: could not load embedding engine: {exc}")

    r2_map = _R2_FOR_R3

    # ------------------------------------------------------------------
    # Audit each beam-miss question
    # ------------------------------------------------------------------
    adapter = graph.adapter
    results: List[Dict] = []
    captured: Dict = {}  # filled by the audit callback

    def _audit_cb(sorted_nbrs, score_map, k_eff):
        captured["sorted_neighbors"] = sorted_nbrs
        captured["score_map"]        = score_map
        captured["k_eff"]            = k_eff

    set_hop1_audit_cb(_audit_cb)

    print(f"[hop1-audit] Auditing {len(miss_rows)} questions...\n")

    for idx, row in enumerate(miss_rows):
        seed           = row["seed"]
        correct_raw    = row.get("all_correct_answers") or row.get("correct_answer", "")
        correct_answers= [a for a in correct_raw.split("|") if a]
        correct_set    = set(correct_answers)
        question_text  = row.get("question_text", "")

        if not seed or not correct_set:
            continue

        # Encode question for query_embedding
        query_emb = None
        if question_text and query_engine is not None:
            try:
                encode_fn = getattr(query_engine, "encode_query", None)
                if encode_fn:
                    query_emb = encode_fn([question_text])[0]
                else:
                    q_vecs = query_engine.encode_entities({"__q__": question_text})
                    query_emb = q_vecs.get("__q__")
            except Exception:
                pass

        # TRB / FHRB detection
        _trb: Dict[str, float] = {}
        if question_text and kb_relations:
            detected = detect_target_relation(question_text, kb_relations)
            if detected:
                _trb = {detected: 5.0}

        _prb: Dict[str, float] = {}
        if _trb and r2_map:
            r3 = next(iter(_trb))
            r2 = r2_map.get(r3)
            if r2:
                _prb = {r2: next(iter(_trb.values())) ** 0.5}

        _irb: Optional[Dict[str, float]] = None
        if args.fhrb_factor > 0.0 and question_text and kb_relations:
            r1 = detect_target_relation(
                question_text, kb_relations,
                exclude_relation=next(iter(_trb)) if _trb else None,
            )
            if r1:
                _irb = {r1: args.fhrb_factor}

        # Run graph.query() with audit callback active
        captured.clear()
        try:
            graph.query(
                seeds=[seed],
                top_k=100,
                min_hop=1,
                max_hop=3,
                hop_expand=True,
                query_embedding=query_emb,
                terminal_relation_boost=_trb,
                penultimate_relation_boost=_prb,
                vote_weight=0.85,
                expansion_k=args.expansion_k,
                initial_relation_boost=_irb,
            )
        except Exception as exc:
            print(f"  [warn] q_idx={row.get('q_idx')} error: {exc}")
            continue

        if "sorted_neighbors" not in captured:
            print(f"  [warn] q_idx={row.get('q_idx')} — callback not fired (hop_expand may not have triggered)")
            continue

        sorted_nbrs = captured["sorted_neighbors"]
        score_map   = captured["score_map"]
        k_eff       = captured["k_eff"]
        n_total_h1  = len(sorted_nbrs)

        # Find first viable rank
        viable_rank = first_viable_rank(sorted_nbrs, correct_set, adapter, seed)

        # Category
        if viable_rank == 0:
            category = "no_viable_h1"
        elif viable_rank <= k_eff:
            category = "in_topk_stage2_fail"
        elif viable_rank <= 25:
            category = "rank_21_25"
        elif viable_rank <= 30:
            category = "rank_26_30"
        elif viable_rank <= 50:
            category = "rank_31_50"
        else:
            category = "rank_50plus"

        # Score of first viable entity (if found)
        viable_score = score_map.get(sorted_nbrs[viable_rank - 1], 0.0) if viable_rank > 0 else 0.0
        top1_score   = score_map.get(sorted_nbrs[0], 0.0) if sorted_nbrs else 0.0

        results.append({
            "q_idx":             row.get("q_idx", idx),
            "seed":              seed,
            "question_text":     question_text,
            "correct_answer":    row.get("correct_answer", ""),
            "detected_relation": row.get("detected_relation", ""),
            "n_total_h1":        n_total_h1,
            "k_eff":             k_eff,
            "first_viable_rank": viable_rank,
            "viable_score":      round(viable_score, 6),
            "top1_score":        round(top1_score, 6),
            "score_ratio":       round(viable_score / top1_score, 4) if top1_score > 0 and viable_rank > 0 else 0.0,
            "category":          category,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(miss_rows):
            print(f"  {idx+1:3d}/{len(miss_rows)}  "
                  f"viable_rank={viable_rank or 'none':>4}  "
                  f"n_h1={n_total_h1:3d}  cat={category}")

    clear_hop1_audit_cb()

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    fields = ["q_idx", "seed", "question_text", "correct_answer",
              "detected_relation", "n_total_h1", "k_eff",
              "first_viable_rank", "viable_score", "top1_score",
              "score_ratio", "category"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n[hop1-audit] Wrote {len(results)} rows -> {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    cat_counts = Counter(r["category"] for r in results)
    n = len(results)

    cat_labels = {
        "in_topk_stage2_fail": f"rank <= {args.expansion_k} (IN top-k, stage-2 fails to find answer)",
        "rank_21_25":          "rank 21-25  (expansion_k=25 would recover)",
        "rank_26_30":          "rank 26-30  (expansion_k=30 would recover)",
        "rank_31_50":          "rank 31-50  (hop-1 scoring problem)",
        "rank_50plus":         "rank > 50   (hop-1 scoring problem — severe)",
        "no_viable_h1":        "no viable   (correct answer unreachable in 3 hops from seed)",
    }
    order = ["in_topk_stage2_fail", "rank_21_25", "rank_26_30",
             "rank_31_50", "rank_50plus", "no_viable_h1"]

    print("\n" + "=" * 72)
    print("  Phase 184 Hop-1 Ranking Audit Summary")
    print("=" * 72)
    print(f"  {'Category':<52} {'N':>5}  {'%':>6}")
    print(f"  {'-'*52} {'-'*5}  {'-'*6}")
    for cat in order:
        cnt = cat_counts.get(cat, 0)
        pct = 100.0 * cnt / n if n else 0.0
        label = cat_labels.get(cat, cat)
        print(f"  {label:<52} {cnt:>5}  {pct:>5.1f}%")
    print(f"  {'-'*52} {'-'*5}  {'-'*6}")
    print(f"  {'TOTAL':<52} {n:>5}  100.0%")
    print("=" * 72)

    # Score ratio distribution for in_topk_stage2_fail
    stage2_fails = [r for r in results if r["category"] == "in_topk_stage2_fail"]
    if stage2_fails:
        ratios = [r["score_ratio"] for r in stage2_fails]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n  Stage-2 fails: avg viable/top1 score ratio = {avg_ratio:.3f}")
        print(f"  (ratio << 1.0 means viable entity has much lower hop-1 score than top-1)")

    print()


if __name__ == "__main__":
    main()

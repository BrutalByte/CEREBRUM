"""
MetaQA 3-hop Diagnostic Analyzer

Reads --diagnose-jsonl output, classifies every miss into a bottleneck
bucket, and emits a structured report with concrete fix suggestions.

Usage
-----
    # 1. Generate the JSONL (500-question sample is fast enough):
    python -m benchmarks.metaqa_eval --hop 3 --embeddings sentence \\
        --beam-width 10 --use-prior --fhrb-factor 3.0 --workers 1 \\
        --sample 500 --diagnose-jsonl diag.jsonl

    # 2. Analyze:
    python -m benchmarks.metaqa_analyze diag.jsonl
    python -m benchmarks.metaqa_analyze diag.jsonl --top 20 --out report.txt

Fields expected in each JSONL record
-------------------------------------
    q_idx, seed, correct_answer, all_correct_answers
    in_beam_top100, beam_rank, detected_rel
    n_filtered, correct_in_filtered, filter_applied, filter_fell_back
    final_rank
    question_text, predicted_top10, correct_answer_score, top1_answer, top1_score
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Counter, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths  (mirrors metaqa_eval.py layout)
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent
DATA_DIR = _HERE / "data" / "metaqa"
KB_FILE  = DATA_DIR / "kb.txt"

_YEAR_RE = re.compile(r"^\d{4}$")

# ---------------------------------------------------------------------------
# KB helpers
# ---------------------------------------------------------------------------

def _load_relation_sets() -> Dict[str, set]:
    rel: Dict[str, set] = collections.defaultdict(set)
    if not KB_FILE.exists():
        print(f"  [warn] KB not found at {KB_FILE} â€” entity-type inference will use heuristics only",
              file=sys.stderr)
        return rel
    with open(KB_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                _, r, o = parts
                rel[r.strip()].add(o.strip())
    return dict(rel)


def _infer_type(entity: str, genres: set, langs: set) -> str:
    e = entity.strip()
    if _YEAR_RE.match(e):
        return "year"
    if e in genres:
        return "genre"
    if e in langs:
        return "language"
    return "person/other"

# ---------------------------------------------------------------------------
# Miss classification
# ---------------------------------------------------------------------------

def _classify(row: dict) -> str:
    """Return 'correct' | 'beam_miss' | 'filter_miss' | 'vote_miss'."""
    if row.get("final_rank") == 1:
        return "correct"
    if not row.get("in_beam_top100", False):
        return "beam_miss"
    if (row.get("filter_applied", False)
            and not row.get("filter_fell_back", False)
            and not row.get("correct_in_filtered", False)):
        return "filter_miss"
    return "vote_miss"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "â€”"


def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(n / total * width) if total else 0
    return "â–ˆ" * filled + "â–‘" * (width - filled)


def _section(title: str) -> str:
    return f"\n{'=' * 60}\n  {title}\n{'=' * 60}"


def _sub(title: str) -> str:
    return f"\n  â”€â”€ {title} â”€â”€"


def _extract_template(question: str, seed: str) -> str:
    if seed and seed in question:
        return question.replace(seed, "[ENTITY]")
    return question

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze(rows: List[dict], top_n: int = 15) -> str:  # noqa: C901
    out: List[str] = []

    rel_sets = _load_relation_sets()
    genres   = rel_sets.get("has_genre",     set())
    langs    = rel_sets.get("in_language",   set())

    n_total = len(rows)
    if n_total == 0:
        return "  [error] No records found in JSONL."

    # Classify every row
    classified: List[Tuple[dict, str]] = [(r, _classify(r)) for r in rows]
    by_cat: Dict[str, List[dict]] = collections.defaultdict(list)
    for row, cat in classified:
        by_cat[cat].append(row)

    correct       = by_cat["correct"]
    beam_misses   = by_cat["beam_miss"]
    filter_misses = by_cat["filter_miss"]
    vote_misses   = by_cat["vote_miss"]

    h1  = len(correct)
    h10 = sum(1 for r in rows if 1 <= (r.get("final_rank") or 0) <= 10)
    n_misses = n_total - h1

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    out.append("=" * 60)
    out.append("  MetaQA 3-hop Diagnostic Analysis")
    out.append("=" * 60)
    out.append(f"\n  Questions analyzed : {n_total:,}")
    out.append(f"  H@1               : {_pct(h1, n_total)}  ({h1:,} correct)")
    out.append(f"  H@10              : {_pct(h10, n_total)}  ({h10:,} in top-10)")
    out.append(f"  Total misses (H@1): {n_misses:,}")

    # -----------------------------------------------------------------------
    # Miss breakdown table
    # -----------------------------------------------------------------------
    out.append(_section("MISS BREAKDOWN"))
    buckets = [
        ("beam_coverage",    beam_misses,   "correct answer never in top-100 beam"),
        ("vote_convergence", vote_misses,   "in beam, ranked below top-1"),
        ("filter_miss",      filter_misses, "in beam, removed by type filter"),
    ]
    header = f"  {'Category':<22} {'N':>5}  {'% total':>7}  {'% misses':>8}  Description"
    out.append(f"\n{header}")
    out.append(f"  {'-'*22} {'-'*5}  {'-'*7}  {'-'*8}  {'-'*35}")
    for label, bucket, desc in buckets:
        n = len(bucket)
        out.append(f"  {label:<22} {n:>5}  {_pct(n, n_total):>7}  {_pct(n, n_misses):>8}  {desc}")
    out.append(f"  {'H@1 correct':<22} {h1:>5}  {_pct(h1, n_total):>7}")

    # Pre-compute per-relation totals (used in multiple sections)
    rel_total_all = collections.Counter(r.get("detected_rel", "unknown") for r in rows)

    # =======================================================================
    # 1. BEAM COVERAGE
    # =======================================================================
    out.append(_section(f"BEAM COVERAGE MISSES  (N={len(beam_misses)}, {_pct(len(beam_misses), n_total)} of total)"))
    out.append(
        "\n  The correct answer never appears in the top-100 candidate pool.\n"
        "  Traversal either missed the hop chain or the stitched score was too\n"
        "  low to survive the global top-100 cutoff window."
    )

    if beam_misses:
        # By detected relation
        rel_beam = collections.Counter(r.get("detected_rel", "unknown") for r in beam_misses)
        out.append(_sub("By detected relation"))
        for rel, cnt in rel_beam.most_common():
            rate = _pct(cnt, rel_total_all[rel])
            out.append(f"    {rel:<28} {cnt:>4} misses  ({rate} of {rel} questions)")

        # Beam rank (partial visibility: rank > 0 means it appeared somewhere in beam but below 100)
        partial = [r.get("beam_rank", 0) for r in beam_misses if (r.get("beam_rank") or 0) > 0]
        if partial:
            sorted_p = sorted(partial)
            out.append(_sub(f"Beam rank when partially visible ({len(partial)} of {len(beam_misses)} cases)"))
            out.append(f"    Min={sorted_p[0]}  Median={sorted_p[len(sorted_p)//2]}  Max={sorted_p[-1]}")
            buckets_brank = collections.Counter(
                "â‰¤50" if r <= 50 else "51-100" if r <= 100 else ">100"
                for r in partial
            )
            for label in ["â‰¤50", "51-100", ">100"]:
                if label in buckets_brank:
                    out.append(f"    rank {label:<7} : {buckets_brank[label]}")

        # Question templates
        tmpls = collections.Counter(
            _extract_template(r.get("question_text", ""), r.get("seed", ""))
            for r in beam_misses if r.get("question_text")
        )
        if tmpls:
            out.append(_sub(f"Top question templates (top {top_n})"))
            for tmpl, cnt in tmpls.most_common(top_n):
                out.append(f"    {cnt:>3}x  {tmpl}")

        # Seeds with multiple beam misses
        seed_bm = collections.Counter(r.get("seed", "") for r in beam_misses)
        hot_seeds = [(s, c) for s, c in seed_bm.most_common() if c >= 2]
        if hot_seeds:
            out.append(_sub("Seeds with 2+ beam misses"))
            for seed, cnt in hot_seeds[:10]:
                out.append(f"    {cnt}x  {seed!r}")

        out.append(_sub("SUGGESTED FIXES"))
        out.append(
            "\n    Priority: HIGH â€” every beam miss is an unrecoverable loss.\n"
            "\n    1. Increase expansion_k (default 20 â†’ 25-30):\n"
            "       More hop-1 entities enter stage-2, giving the GlobalBeamBarrier\n"
            "       more candidates to guarantee. Low risk, ~15% memory increase.\n"
            "       Code: reasoning/expanded_traversal.py â†’ _h1se_expand()\n"
            "       CLI:  --expansion-k 25\n"
            "\n    2. Widen stage-2 beam for guaranteed branches:\n"
            "       Top-10 hop-1 branches currently share the same beam=30 as the rest.\n"
            "       Give each guaranteed branch beam=50 while others keep 30.\n"
            "       Targets the in_topk_stage2_fail subcase.\n"
            "       Code: reasoning/expanded_traversal.py â†’ HopExpandedTraversal.query()\n"
            "\n    3. Raise _raw_top_k from 100 â†’ 150 (collection window):\n"
            "       Previously caused H@10 regression via filter interaction.\n"
            "       Re-test ONLY on beam_coverage cases: if they now rank 101-150,\n"
            "       the fix is valid and filter interaction was a separate bug.\n"
            "       Code: benchmarks/metaqa_eval.py â†’ _raw_top_k\n"
            "\n    4. Score floor for guaranteed branches:\n"
            "       If hop-1 score is very low (~0.001), sqrt(parent*child) is still\n"
            "       tiny. Add a minimum stitched score for barrier-guaranteed paths\n"
            "       so they can't fall below the top-100 cutoff.\n"
            "       Code: reasoning/expanded_traversal.py â†’ _stitch()"
        )
    else:
        out.append("\n  No beam coverage misses â€” traversal is reaching all correct answers.")

    # =======================================================================
    # 2. VOTE CONVERGENCE
    # =======================================================================
    out.append(_section(f"VOTE CONVERGENCE MISSES  (N={len(vote_misses)}, {_pct(len(vote_misses), n_total)} of total)"))
    out.append(
        "\n  Correct answer is in the top-100 pool but loses to a wrong answer\n"
        "  during scoring and re-ranking."
    )

    low_hanging: List[dict] = []

    if vote_misses:
        # Rank distribution
        rank_dist = collections.Counter((r.get("final_rank") or 0) for r in vote_misses)
        out.append(_sub("Final rank of correct answer"))
        for rank in sorted(rank_dist):
            label = f"rank={rank}" if rank > 0 else "not-in-top10"
            out.append(f"    {label:<14} {rank_dist[rank]:>4}  {_bar(rank_dist[rank], len(vote_misses))}")
        low_hanging = [r for r in vote_misses if (r.get("final_rank") or 0) == 2]
        out.append(f"\n    Low-hanging fruit (rank=2, one nudge away): {len(low_hanging)} questions")

        # Wrong answer type breakdown
        out.append(_sub("Wrong answer type breakdown"))
        type_counts: collections.Counter = collections.Counter()
        for row in vote_misses:
            top1    = row.get("top1_answer", "") or ""
            ca      = (row.get("all_correct_answers") or row.get("correct_answer") or "").split("|")[0]
            t1_type = _infer_type(top1, genres, langs)
            ca_type = _infer_type(ca, genres, langs)
            type_counts[f"{t1_type} beats {ca_type}"] += 1
        for pattern, cnt in type_counts.most_common():
            pct_s = _pct(cnt, len(vote_misses))
            out.append(f"    {cnt:>4}  ({pct_s:>5})  {pattern}")

        # Score gap analysis
        score_data: List[Tuple[float, float, float, dict]] = []
        for row in vote_misses:
            t1s = row.get("top1_score") or 0.0
            cas = row.get("correct_answer_score") or 0.0
            if t1s > 0:
                score_data.append((t1s - cas, t1s, cas, row))

        if score_data:
            gaps   = [g for g, *_ in score_data]
            ratios = [cas / t1s for _, t1s, cas, _ in score_data if t1s > 0]
            out.append(_sub("Score gap  (top1_score âˆ’ correct_answer_score)"))
            out.append(f"    Mean gap    : {sum(gaps)/len(gaps):.5f}")
            out.append(f"    Median gap  : {sorted(gaps)[len(gaps)//2]:.5f}")
            out.append(f"    Max gap     : {max(gaps):.5f}")
            if ratios:
                out.append(f"    Mean correct/top1 ratio : {sum(ratios)/len(ratios):.3f}")
                close  = sum(1 for r in ratios if r >= 0.75)
                medium = sum(1 for r in ratios if 0.25 <= r < 0.75)
                far    = sum(1 for r in ratios if r < 0.25)
                out.append(f"    â‰¥75% of top1 (near-tie) : {close:>4}  ({_pct(close, len(ratios))})")
                out.append(f"    25-75% of top1          : {medium:>4}  ({_pct(medium, len(ratios))})")
                out.append(f"    <25% of top1 (far miss) : {far:>4}  ({_pct(far, len(ratios))})")
                out.append(
                    "\n    Interpretation:\n"
                    f"    - Near-ties ({close}) respond well to small parameter tuning\n"
                    f"    - Far misses ({far}) need structural fixes (new penalty type or scoring term)"
                )

        # Most frequent wrong top-1 answers
        out.append(_sub(f"Most frequent wrong top-1 answers (top {top_n})"))
        wrong_top1 = collections.Counter(r.get("top1_answer", "") for r in vote_misses if r.get("top1_answer"))
        for ans, cnt in wrong_top1.most_common(top_n):
            atype = _infer_type(ans, genres, langs)
            out.append(f"    {cnt:>3}x  {ans!r:<35} [{atype}]")

        # Genre/language wins specifically
        genre_lang_wins = [
            r for r in vote_misses
            if _infer_type(r.get("top1_answer", ""), genres, langs) in ("genre", "language")
        ]
        if genre_lang_wins:
            out.append(_sub(f"Genre/language entities winning over correct answer ({len(genre_lang_wins)} cases)"))
            gl_counts = collections.Counter(r.get("top1_answer", "") for r in genre_lang_wins)
            for ent, cnt in gl_counts.most_common(10):
                etype = _infer_type(ent, genres, langs)
                out.append(f"    {cnt:>3}x  {ent!r}  [{etype}]")
            out.append(
                f"\n    â†’ Check if these are in _pure_genre in metaqa_eval.py.\n"
                f"      If not, add them to the penalty set or _FORMAT_TAG_BLOCKLIST."
            )

        # By detected relation
        out.append(_sub("Vote misses by detected relation"))
        rel_vote_c = collections.Counter(r.get("detected_rel", "unknown") for r in vote_misses)
        for rel, cnt in rel_vote_c.most_common():
            rate = _pct(cnt, rel_total_all[rel])
            out.append(f"    {rel:<28} {cnt:>4} misses  ({rate} of {rel} questions)")

        # Question templates for vote misses
        tmpls_v = collections.Counter(
            _extract_template(r.get("question_text", ""), r.get("seed", ""))
            for r in vote_misses if r.get("question_text")
        )
        if tmpls_v:
            out.append(_sub(f"Top failing question templates (top {top_n})"))
            for tmpl, cnt in tmpls_v.most_common(top_n):
                out.append(f"    {cnt:>3}x  {tmpl}")

        # Suggested fixes â€” data-driven
        out.append(_sub("SUGGESTED FIXES"))
        suggestions: List[str] = []

        genre_lang_n = (type_counts.get("genre beats person/other", 0) +
                        type_counts.get("language beats person/other", 0) +
                        type_counts.get("genre beats year", 0) +
                        type_counts.get("language beats year", 0))
        if genre_lang_n >= 3:
            suggestions.append(
                f"    1. EXTEND CROSS-TYPE PENALTY  (~{min(genre_lang_n, 8)} potential H@1 gains)\n"
                f"       {genre_lang_n} genre/language entities are beating the correct answer.\n"
                "       Check the list above: any entity NOT already in _pure_genre should\n"
                "       be added, OR add it to _FORMAT_TAG_BLOCKLIST (applies the same 0.10x\n"
                "       penalty). These entities are definitionally not people or years, so\n"
                "       penalizing them for person/year relation queries is safe.\n"
                "       Code: benchmarks/metaqa_eval.py â†’ _pure_genre, _FORMAT_TAG_BLOCKLIST"
            )

        year_vs_year = type_counts.get("year beats year", 0)
        if year_vs_year >= 3:
            suggestions.append(
                f"    2. YEAR-vs-YEAR DISAMBIGUATION  ({year_vs_year} cases)\n"
                "       Both candidates are years; the wrong year wins by accumulating more\n"
                "       undirected KB paths (popular release years like 2011 appear in\n"
                "       thousands of movies). Attempted fixes:\n"
                "         âœ— Global IDF weight â€” regressed (penalizes correct popular years too)\n"
                "       Untried approaches:\n"
                "         â†’ Release-year-specific IDF floor: apply idf_weight ONLY when\n"
                "           detected_rel == 'release_year' AND entity type is 'year'.\n"
                "           This is more targeted than the global IDF that regressed.\n"
                "         â†’ Path-length tiebreaker: when scores are within 5%, prefer the\n"
                "           year with fewer total KB occurrences.\n"
                "         â†’ Directed-path count: count only forward-direction traversals\n"
                "           (subjectâ†’release_year) instead of undirected paths."
            )

        person_vs_person = type_counts.get("person/other beats person/other", 0)
        if person_vs_person >= 3:
            suggestions.append(
                f"    3. WRONG-PERSON DISAMBIGUATION  ({person_vs_person} cases)\n"
                "       Wrong person accumulates more multi-hop paths than the correct one.\n"
                "       Current state: r2_boost=3.0 is confirmed optimal via sweep.\n"
                "       Untried approaches:\n"
                "         â†’ Relation-specific r2_boost: higher boost for written_by/directed_by\n"
                "           (currently a single global value).\n"
                "         â†’ Path uniqueness penalty: entities reachable via many DIFFERENT\n"
                "           hop-2 relations are likely hub nodes, not specific answers.\n"
                "         â†’ Anchor bonus: if seedâ†’hop1 score is high, boost hop1-rooted paths."
            )

        if low_hanging:
            close_cnt = sum(1 for _, t1s, cas, _ in score_data if cas / t1s >= 0.75) if score_data else 0
            suggestions.append(
                f"    4. NEAR-TIE TUNING  ({len(low_hanging)} rank-2 + {close_cnt} near-tie score cases)\n"
                "       Correct answer is rank=2 or scores â‰¥75% of top-1. Small parameter\n"
                "       adjustments could flip these without disrupting H@1 hits.\n"
                "         â†’ Run Optuna re-tune with a tighter search space:\n"
                "             vote_weight: [0.82, 0.88]  r2_boost: [2.5, 3.5]\n"
                "         â†’ Test branch_bonus_weight (default 0.25, currently underexplored):\n"
                "             rewards entities that appear across multiple hop-1 branches."
            )

        if not suggestions:
            suggestions.append(
                "    No dominant pattern detected. Run on a larger sample (nâ‰¥2000)\n"
                "    for clearer signal before choosing a fix direction."
            )

        out.extend([""] + suggestions)
    else:
        out.append("\n  No vote convergence misses.")

    # =======================================================================
    # 3. FILTER MISSES
    # =======================================================================
    out.append(_section(f"FILTER MISSES  (N={len(filter_misses)}, {_pct(len(filter_misses), n_total)} of total)"))
    out.append(
        "\n  Correct answer was in the top-100 beam but was removed by the\n"
        "  answer-type filter (the TRB-detected relation's valid-answer set\n"
        "  did not include the correct answer)."
    )

    if filter_misses:
        rel_filt = collections.Counter(r.get("detected_rel", "unknown") for r in filter_misses)
        out.append(_sub("By detected relation"))
        for rel, cnt in rel_filt.most_common():
            rate = _pct(cnt, rel_total_all[rel])
            out.append(f"    {rel:<28} {cnt:>4}  ({rate} of {rel} questions)")

        # n_filtered distribution for filter misses
        n_filt_vals = [r.get("n_filtered", 0) or 0 for r in filter_misses]
        if n_filt_vals:
            out.append(_sub("n_filtered (candidates surviving type filter when miss occurred)"))
            filt_dist = collections.Counter(
                "0 (empty)" if v == 0 else "1-2" if v <= 2 else "3-5" if v <= 5 else ">5"
                for v in n_filt_vals
            )
            for label in ["0 (empty)", "1-2", "3-5", ">5"]:
                if label in filt_dist:
                    out.append(f"    filtered to {label:<10} : {filt_dist[label]}")

        # Sample of filter miss questions
        sample_filt = filter_misses[:5]
        if sample_filt and sample_filt[0].get("question_text"):
            out.append(_sub("Sample filter miss questions"))
            for r in sample_filt:
                q = _extract_template(r.get("question_text", ""), r.get("seed", ""))
                ca = r.get("correct_answer", "")
                rel = r.get("detected_rel", "")
                nf = r.get("n_filtered", 0)
                out.append(f"    rel={rel!r}  n_filtered={nf}  correct={ca!r}")
                out.append(f"      Q: {q}")

        out.append(_sub("SUGGESTED FIXES"))
        zero_filt = sum(1 for v in n_filt_vals if v == 0)
        out.append(
            "\n    1. INCREASE --min-filter-size (currently 1):\n"
            "       Only apply hard filter when â‰¥N valid-type answers exist.\n"
            "       If n_filtered=0 is common above, the filter is always cutting\n"
            "       everything â€” raising to 2-3 would force fallback to unfiltered.\n"
            f"       {zero_filt} of {len(filter_misses)} filter misses had n_filtered=0.\n"
            "       CLI: --min-filter-size 3\n"
            "\n    2. CHECK TRB DETECTION ACCURACY:\n"
            "       If the wrong relation is detected, the filter uses the wrong answer set.\n"
            "       Run with --verbose to see which relation is detected per question and\n"
            "       compare against the question text manually.\n"
            "\n    3. REVIEW KB ANSWER SETS:\n"
            "       If a correct answer entity is reachable via multiple relations, it may\n"
            "       not be in the detected relation's answer set. Consider expanding the\n"
            "       answer set to include entities within 1 hop of the relation's objects."
        )
    else:
        out.append("\n  No filter misses â€” type filter is not removing correct answers.")

    # =======================================================================
    # 4. RELATION-LEVEL PERFORMANCE TABLE
    # =======================================================================
    out.append(_section("RELATION-LEVEL PERFORMANCE"))

    rel_correct_c = collections.Counter(r.get("detected_rel", "unknown") for r in correct)
    rel_h10_c     = collections.Counter(
        r.get("detected_rel", "unknown")
        for r in rows if 1 <= (r.get("final_rank") or 0) <= 10
    )
    rel_beam_c   = collections.Counter(r.get("detected_rel", "unknown") for r in beam_misses)
    rel_vote_c2  = collections.Counter(r.get("detected_rel", "unknown") for r in vote_misses)
    rel_filt_c   = collections.Counter(r.get("detected_rel", "unknown") for r in filter_misses)

    hdr = (f"\n  {'Relation':<25} {'N':>5}  {'H@1':>6}  {'H@10':>6}  "
           f"{'Beam':>6}  {'Vote':>6}  {'Filter':>6}")
    out.append(hdr)
    out.append(f"  {'-'*25} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for rel in sorted(rel_total_all, key=lambda r: -rel_total_all[r]):
        tot = rel_total_all[rel]
        out.append(
            f"  {rel:<25} {tot:>5}  "
            f"{_pct(rel_correct_c[rel], tot):>6}  "
            f"{_pct(rel_h10_c[rel], tot):>6}  "
            f"{rel_beam_c[rel]:>4}({_pct(rel_beam_c[rel], tot):>5})  "
            f"{rel_vote_c2[rel]:>4}({_pct(rel_vote_c2[rel], tot):>5})  "
            f"{rel_filt_c[rel]:>6}"
        )

    # =======================================================================
    # 5. QUESTION TEMPLATE ANALYSIS
    # =======================================================================
    out.append(_section("QUESTION TEMPLATE ANALYSIS"))

    all_tmpls = collections.Counter(
        _extract_template(r.get("question_text", ""), r.get("seed", ""))
        for r in rows if r.get("question_text")
    )
    miss_rows_all = beam_misses + vote_misses + filter_misses
    miss_tmpls = collections.Counter(
        _extract_template(r.get("question_text", ""), r.get("seed", ""))
        for r in miss_rows_all if r.get("question_text")
    )

    template_rates: List[Tuple[float, int, int, str]] = []
    for tmpl, miss_cnt in miss_tmpls.items():
        total_cnt = all_tmpls[tmpl]
        if total_cnt >= 3:
            template_rates.append((miss_cnt / total_cnt, miss_cnt, total_cnt, tmpl))
    template_rates.sort(reverse=True)

    out.append(f"\n  {'Miss%':>6}  {'Misses':>6}  {'Total':>6}  Template")
    out.append(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*55}")
    for rate, miss_cnt, total_cnt, tmpl in template_rates[:top_n]:
        out.append(f"  {rate*100:>5.1f}%  {miss_cnt:>6}  {total_cnt:>6}  {tmpl}")

    # =======================================================================
    # 6. SEED ENTITY HOTSPOTS
    # =======================================================================
    out.append(_section("SEED ENTITY HOTSPOTS"))
    out.append("\n  Seed entities appearing in 3+ misses (all miss categories):")

    seed_misses = collections.Counter(r.get("seed", "") for r in miss_rows_all)
    hotspots    = [(s, c) for s, c in seed_misses.most_common() if c >= 3]

    if hotspots:
        for seed, cnt in hotspots[:top_n]:
            out.append(f"    {cnt:>3}x  {seed!r}")
        out.append(f"\n  Total hotspot seeds: {len(hotspots)}")
        out.append(
            "\n  These entities have many outgoing KB edges, creating traversal noise.\n"
            "  SUGGESTED FIX:\n"
            "    â†’ IDF-style seed-entity penalty: reduce scores for candidates\n"
            "      reachable from very high-degree seed entities.\n"
            "    â†’ Anchor bonus: if seedâ†’hop1 cosine score is high, boost that branch."
        )
    else:
        out.append("    No seed entities with 3+ misses.")

    # =======================================================================
    # 7. PRIORITY SUMMARY
    # =======================================================================
    out.append(_section("PRIORITY ACTION SUMMARY"))
    out.append("\n  Ranked by estimated H@1 impact:\n")

    priority_items: List[Tuple[float, str]] = []
    if beam_misses:
        pct_bm = len(beam_misses) / n_total * 100
        priority_items.append((pct_bm,
            f"beam_coverage ({len(beam_misses)} cases, {pct_bm:.1f}%):\n"
            "       expansion_kâ†‘, beam widening for guaranteed branches, or score floor"))
    if vote_misses:
        pct_vm = len(vote_misses) / n_total * 100
        priority_items.append((pct_vm,
            f"vote_convergence ({len(vote_misses)} cases, {pct_vm:.1f}%):\n"
            "       type penalties for unpenalized genre/lang entities + r2_boost re-tuning"))
    if filter_misses:
        pct_fm = len(filter_misses) / n_total * 100
        priority_items.append((pct_fm,
            f"filter_miss ({len(filter_misses)} cases, {pct_fm:.1f}%):\n"
            "       min_filter_sizeâ†‘ or TRB detection review"))

    priority_items.sort(reverse=True)
    for rank, (_, desc) in enumerate(priority_items, 1):
        out.append(f"  {rank}. {desc}\n")

    if low_hanging:
        out.append(f"  Quick wins: {len(low_hanging)} rank-2 misses â€” correct answer is #2,\n"
                   f"    a small score nudge could flip them. Run Optuna with tight bounds first.")

    unreachable_pct = len(beam_misses) / n_total * 100
    reachable_miss_pct = (len(vote_misses) + len(filter_misses)) / n_total * 100
    out.append(
        f"\n  â”€â”€ Upper bounds â”€â”€\n"
        f"  If beam_coverage fixed:    H@1 ceiling = {_pct(h1 + len(beam_misses), n_total)} (currently {_pct(h1, n_total)})\n"
        f"  If vote+filter fixed:      H@1 ceiling = {_pct(h1 + len(vote_misses) + len(filter_misses), n_total)}\n"
        f"  If all misses fixed:       H@1 ceiling = 100.0%\n"
        f"\n  Interpretation: scoring improvements are capped at {reachable_miss_pct:.1f}pp gain;\n"
        f"  traversal improvements (beam) unlock an additional {unreachable_pct:.1f}pp."
    )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MetaQA --diagnose-jsonl output and suggest fixes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("jsonl", help="Path to the --diagnose-jsonl output file")
    parser.add_argument("--top",  type=int, default=15, metavar="N",
                        help="Show top-N entries in ranked lists (default: 15)")
    parser.add_argument("--out",  type=str, default=None, metavar="PATH",
                        help="Write report to file instead of stdout")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[error] File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    rows: List[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] Line {line_num}: JSON parse error â€” {e}", file=sys.stderr)

    if not rows:
        print("[error] No valid JSONL records found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(rows):,} records from {jsonl_path.name}", file=sys.stderr)

    report = analyze(rows, top_n=args.top)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"  Report written to {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()

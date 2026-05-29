"""
SDRB Analysis — Schema-Derived Relation Boost correlation study.

For each tuner trial that has per-relation boost values (wb/db/ry/sa),
tests whether those values are predictable from KB fan-out statistics.

Hypothesis: boost(r) = γ × fan_out(r)^β

If R² ≥ 0.80 the formula is valid and we can eliminate 4 free parameters.

Usage:
    python -m benchmarks.sdrb_analysis
    python -m benchmarks.sdrb_analysis --min-h1 0.62 --kb benchmarks/data/metaqa/kb.txt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# KB fan-out computation
# ---------------------------------------------------------------------------

def compute_fan_out(kb_path: str) -> Dict[str, float]:
    """Return {relation: avg_tails_per_head} from a pipe-delimited KB file."""
    rel_heads: Dict[str, set] = defaultdict(set)
    rel_count: Dict[str, int] = defaultdict(int)
    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                h, r, _ = parts
                rel_heads[r].add(h)
                rel_count[r] += 1
    return {r: rel_count[r] / max(len(rel_heads[r]), 1) for r in rel_count}


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------

_BOOST_KEYS = ("wb_r2_boost", "db_r2_boost", "ry_r2_boost", "sa_r2_boost")
_BOOST_RELS = ("written_by", "directed_by", "release_year", "starred_actors")


def load_trials(jsonl_dir: str, min_h1: float = 0.0) -> List[dict]:
    """Load all trial records from *.jsonl files, optionally filtered by H@1."""
    trials = []
    for path in sorted(Path(jsonl_dir).glob("tuner_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "trial":
                    continue
                if not all(k in rec for k in _BOOST_KEYS):
                    continue
                if rec.get("h1", 0.0) >= min_h1:
                    trials.append(rec)
    return trials


# ---------------------------------------------------------------------------
# Power-law fit: boost = γ × fan_out^β  (OLS on log-log)
# ---------------------------------------------------------------------------

def fit_power_law(
    fan_outs: np.ndarray, boosts: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit log(boost) = log(γ) + β×log(fan_out) by OLS.
    Returns (gamma, beta, r_squared).
    """
    log_x = np.log(fan_outs)
    log_y = np.log(np.maximum(boosts, 1e-9))

    # OLS: [1, log_x] × [log_gamma, beta]
    X = np.column_stack([np.ones_like(log_x), log_x])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_y, rcond=None)
    log_gamma, beta = coeffs

    y_pred = log_gamma + beta * log_x
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return math.exp(log_gamma), beta, r2


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(kb_path: str, jsonl_dir: str, min_h1: float, top_n: int) -> None:
    print(f"\n{'='*68}")
    print("  SDRB Correlation Analysis")
    print(f"{'='*68}\n")

    # 1. KB statistics
    fan_out = compute_fan_out(kb_path)
    print("KB fan-out (avg tails per head):")
    for r in sorted(fan_out, key=lambda x: -fan_out[x]):
        print(f"  {r:30s}  {fan_out[r]:.4f}")

    fo_vec = np.array([fan_out.get(r, 1.0) for r in _BOOST_RELS])
    print()

    # 2. Load trials
    trials = load_trials(jsonl_dir, min_h1=min_h1)
    if not trials:
        print(f"No trials found with H@1 ≥ {min_h1:.0%} in {jsonl_dir}")
        return
    trials.sort(key=lambda t: -t["h1"])
    total = len(trials)
    print(f"Trials with per-relation boosts and H@1 >= {min_h1:.0%}: {total}")

    # Deduplicate by (wb,db,ry,sa) tuple rounded to 2dp — same config re-run
    seen = set()
    unique = []
    for t in trials:
        key = tuple(round(t[k], 2) for k in _BOOST_KEYS)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    print(f"Unique boost configurations: {len(unique)}\n")

    # 3. Per-trial fit
    gammas, betas, r2s = [], [], []
    for t in unique:
        boost_vec = np.array([t[k] for k in _BOOST_KEYS])
        if boost_vec.min() < 0.01:
            continue  # skip if any boost near-zero (log undefined)
        g, b, r2 = fit_power_law(fo_vec, boost_vec)
        gammas.append(g)
        betas.append(b)
        r2s.append(r2)

    if not r2s:
        print("Not enough non-zero boost configs for fitting.")
        return

    r2_arr = np.array(r2s)
    gamma_arr = np.array(gammas)
    beta_arr = np.array(betas)

    print(f"Per-trial power-law fit  (boost = γ × fan_out^β)")
    print(f"  Trials fitted  : {len(r2s)}")
    print(f"  R²  median     : {np.median(r2_arr):.4f}")
    print(f"  R²  mean       : {r2_arr.mean():.4f}")
    print(f"  R2  >= 0.80    : {(r2_arr >= 0.80).sum()} / {len(r2s)}  ({(r2_arr >= 0.80).mean():.0%})")
    print(f"  γ   median     : {np.median(gamma_arr):.4f}")
    print(f"  γ   mean±std   : {gamma_arr.mean():.4f} ± {gamma_arr.std():.4f}")
    print(f"  β   median     : {np.median(beta_arr):.4f}")
    print(f"  β   mean±std   : {beta_arr.mean():.4f} ± {beta_arr.std():.4f}")
    print()

    # 4. Top-N analysis (best performing trials only)
    top = unique[:top_n]
    top_boosts = np.array([[t[k] for k in _BOOST_KEYS] for t in top
                            if min(t[k] for k in _BOOST_KEYS) >= 0.01])
    if len(top_boosts) > 0:
        top_gamma, top_beta, top_r2 = fit_power_law(
            np.tile(fo_vec, (len(top_boosts), 1)).reshape(-1),
            top_boosts.reshape(-1),
        )
        print(f"Pooled fit on top-{len(top_boosts)} trials (H@1 ≥ {top[0]['h1']:.1%}):")
        print(f"  γ = {top_gamma:.4f}   β = {top_beta:.4f}   R² = {top_r2:.4f}")
        print()

        # Predicted vs actual for best trial
        best = top[0]
        best_boosts = np.array([best[k] for k in _BOOST_KEYS])
        predicted = top_gamma * fo_vec ** top_beta
        print(f"Best trial  H@1={best['h1']:.1%}  predicted vs actual:")
        print(f"  {'Relation':30s}  {'fan_out':>8}  {'predicted':>10}  {'actual':>8}  {'residual':>9}")
        print(f"  {'-'*67}")
        for rel, fo, pred, act in zip(_BOOST_RELS, fo_vec, predicted, best_boosts):
            print(f"  {rel:30s}  {fo:8.4f}  {pred:10.4f}  {act:8.4f}  {act-pred:+9.4f}")
        print()

    # 5. Boost ratio analysis — do ratios match fan_out ratios?
    print("Ratio analysis (are boost ratios ∝ fan_out ratios?)")
    print(f"  fan_out ratios relative to directed_by (fan_out={fo_vec[1]:.4f}):")
    fo_ratios = fo_vec / fo_vec[1]
    for rel, fro in zip(_BOOST_RELS, fo_ratios):
        print(f"    {rel:30s}  {fro:.4f}")
    print()

    all_boosts = np.array([[t[k] for k in _BOOST_KEYS] for t in unique])
    boost_ratios = all_boosts / all_boosts[:, 1:2]  # relative to directed_by
    print(f"  boost ratios relative to directed_by  (median across {len(unique)} configs):")
    for rel, med_ratio, fo_ratio in zip(_BOOST_RELS,
                                         np.median(boost_ratios, axis=0),
                                         fo_ratios):
        print(f"    {rel:30s}  boost_ratio={med_ratio:.4f}  fo_ratio={fo_ratio:.4f}  "
              f"ratio_match={'✓' if abs(med_ratio - fo_ratio) / fo_ratio < 0.4 else '✗'}")
    print()

    # 6. Linear fit (β forced to 1): boost = γ × fan_out
    all_fo = np.tile(fo_vec, len(unique))
    all_bv = all_boosts.reshape(-1)
    gamma_linear = np.dot(all_fo, all_bv) / np.dot(all_fo, all_fo)
    predicted_linear = gamma_linear * all_fo
    ss_res = np.sum((all_bv - predicted_linear) ** 2)
    ss_tot = np.sum((all_bv - all_bv.mean()) ** 2)
    r2_linear = 1.0 - ss_res / ss_tot

    print(f"Linear constraint fit (β=1, boost = γ × fan_out):")
    print(f"  γ = {gamma_linear:.4f}   R² = {r2_linear:.4f}")
    print()

    # 7. Verdict
    print("─" * 68)
    r2_verdict = np.median(r2_arr)
    print(f"VERDICT:")
    if r2_verdict >= 0.80:
        print(f"  R² median {r2_verdict:.4f} ≥ 0.80 — formula is VALID.")
        print(f"  Replacing wb/db/ry/sa with γ×fan_out^β reduces 4 params to 2.")
        if abs(np.median(beta_arr) - 1.0) < 0.2:
            print(f"  β ≈ 1.0 — LINEAR relationship. Can reduce to 1 param (γ only).")
            print(f"  Candidate universal constant: γ ≈ {np.median(gamma_arr):.3f}")
    elif r2_verdict >= 0.60:
        print(f"  R² median {r2_verdict:.4f} — moderate fit. fan_out is a partial predictor.")
        print(f"  Add semantic isolation from SRD embeddings as second predictor.")
    else:
        print(f"  R² median {r2_verdict:.4f} — weak fit. fan_out alone is insufficient.")
        print(f"  Consider: triple_count, unique_tails, or embedding-space isolation.")
    print("─" * 68)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SDRB correlation analysis")
    parser.add_argument("--kb",       default="benchmarks/data/metaqa/kb.txt",
                        help="Path to pipe-delimited KB file")
    parser.add_argument("--dir",      default="benchmarks",
                        help="Directory containing tuner_*.jsonl files")
    parser.add_argument("--min-h1",   type=float, default=0.60,
                        help="Minimum H@1 to include a trial (default: 0.60)")
    parser.add_argument("--top-n",    type=int,   default=10,
                        help="Number of top trials for pooled fit (default: 10)")
    args = parser.parse_args()

    analyse(
        kb_path=args.kb,
        jsonl_dir=args.dir,
        min_h1=args.min_h1,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()

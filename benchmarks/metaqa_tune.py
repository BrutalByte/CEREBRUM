"""
CEREBRUM MetaQA Hyperparameter Tuner — Phase 183.

Uses Optuna TPE to search scoring parameters that target the ranking-miss problem.
Each trial runs metaqa_eval on a small subsample; results are logged to MLflow.

Usage
-----
    # Quick search (30 trials × 500 questions ≈ 35 min):
    python -m benchmarks.metaqa_tune

    # Larger search (50 trials × 1000 questions ≈ 90 min):
    python -m benchmarks.metaqa_tune --n-trials 50 --sample 1000

    # Custom search space:
    python -m benchmarks.metaqa_tune --search vote_weight pss_weight r2_boost

Search space (defaults)
-----------------------
    pss_weight  : [0.00, 0.50]   Phase 179 path-specificity score blend
    vote_weight : [0.60, 0.90]   Convergence voting dominance
    r2_boost    : [0.20, 0.80]   R2 path-consistency reward

Fixed parameters (always applied, not tuned)
--------------------------------------------
    --embeddings sentence
    --beam-width 20
    --use-prior
    --fhrb-factor 3.0
    --hop 3
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Fixed base configuration (same as canonical Phase 182 run)
# ---------------------------------------------------------------------------
_BASE_ARGS = [
    sys.executable, "-u", "-m", "benchmarks.metaqa_eval",
    "--hop", "3",
    "--embeddings", "sentence",
    "--beam-width", "20",
    "--use-prior",
    "--fhrb-factor", "3.0",
    "--workers", str(os.cpu_count()),
]

_WORKDIR = str(Path(__file__).parent.parent)

# Parameters that are FIXED in Phase 182 canonical and not searched by default
_FIXED = {
    "r2_boost": 0.40,
    "pss_weight": 0.0,
    "vote_weight": 0.85,
    "idf_weight": 0.0,
}

# Search space bounds
_SPACE = {
    "pss_weight":  (0.00, 0.50),
    "vote_weight": (0.60, 0.90),
    "r2_boost":    (0.20, 0.80),
    "idf_weight":  (0.00, 0.80),
}


def _run_config(params: dict, sample: int, seed: int) -> dict | None:
    """Run metaqa_eval with given params, return metrics dict or None on error."""
    cmd = _BASE_ARGS + [
        "--sample",      str(sample),
        "--seed",        str(seed),
        "--pss-weight",  str(params.get("pss_weight",  _FIXED["pss_weight"])),
        "--vote-weight", str(params.get("vote_weight", _FIXED["vote_weight"])),
        "--r2-boost",    str(params.get("r2_boost",    _FIXED["r2_boost"])),
        "--idf-weight",  str(params.get("idf_weight",  _FIXED["idf_weight"])),
    ]
    t0 = time.time()
    # Write to temp file + CREATE_NEW_PROCESS_GROUP to avoid WinError 5:
    # metaqa_eval spawns 32 multiprocessing workers; nested handle duplication
    # fails when the parent subprocess inherits the tuner's handle table.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        tmp_path = tf.name
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        with open(tmp_path, 'w') as fout:
            result = subprocess.run(
                cmd, stdout=fout, stderr=subprocess.STDOUT,
                text=True, cwd=_WORKDIR,
                creationflags=creation_flags,
            )
        elapsed = time.time() - t0
        with open(tmp_path) as f:
            out = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.returncode != 0:
        print(f"  [trial error] rc={result.returncode}", flush=True)
        print(out[-500:], flush=True)
        return None
    m1  = re.search(r"Hits@1\s*:\s*([0-9.]+)", out)
    m10 = re.search(r"Hits@10\s*:\s*([0-9.]+)", out)
    mrr = re.search(r"\bMRR\s*:\s*([0-9.]+)", out)
    if not m1:
        return None

    return {
        "hits1":    float(m1.group(1)),
        "hits10":   float(m10.group(1)) if m10 else 0.0,
        "mrr":      float(mrr.group(1)) if mrr else 0.0,
        "elapsed":  elapsed,
    }


def build_objective(active_params: list[str], sample: int, mlflow_run=None):
    trial_num = [0]

    def objective(trial: optuna.Trial) -> float:
        trial_num[0] += 1
        params: dict = {}
        for name in active_params:
            lo, hi = _SPACE[name]
            params[name] = trial.suggest_float(name, lo, hi)

        label = "  ".join(f"{k}={v:.3f}" for k, v in params.items())
        print(f"\n[trial {trial_num[0]:3d}] {label}", flush=True)

        metrics = _run_config(params, sample=sample, seed=42)
        if metrics is None:
            raise optuna.TrialPruned()

        h1 = metrics["hits1"]
        print(f"           → H@1={h1*100:.2f}%  H@10={metrics['hits10']*100:.2f}%  "
              f"MRR={metrics['mrr']:.4f}  ({metrics['elapsed']:.0f}s)", flush=True)

        if mlflow_run is not None:
            try:
                import mlflow
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("h1",    h1)
                    mlflow.log_metric("h10",   metrics["hits10"])
                    mlflow.log_metric("mrr",   metrics["mrr"])
            except Exception:
                pass

        return h1

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for MetaQA")
    parser.add_argument("--n-trials",  type=int,   default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--sample",    type=int,   default=500,
                        help="Questions per trial (default: 500; ±2.2pp 95%% CI)")
    parser.add_argument("--search",    nargs="+",
                        default=["pss_weight", "vote_weight", "r2_boost"],
                        choices=list(_SPACE.keys()),
                        help="Parameters to search (default: pss_weight vote_weight r2_boost)")
    parser.add_argument("--mlflow",    action="store_true",
                        help="Log trials to MLflow under experiment 'cerebrum-metaqa-tune'")
    parser.add_argument("--study-name", type=str, default="metaqa_phase183",
                        help="Optuna study name (default: metaqa_phase183)")
    parser.add_argument("--validate",  type=int,   default=2000,
                        help="Questions for final validation run of best params (default: 2000). "
                             "Set 0 to skip.")
    args = parser.parse_args()

    print("=" * 60)
    print("CEREBRUM MetaQA Hyperparameter Search — Phase 183")
    print("=" * 60)
    print(f"  Trials      : {args.n_trials}")
    print(f"  Sample/trial: {args.sample} questions")
    print(f"  Searching   : {args.search}")
    print(f"  Fixed       : r2_boost={_FIXED['r2_boost']}  fhrb=3.0  bw=20  prior=yes")
    print(f"  Workers     : {os.cpu_count()}")
    print()

    # MLflow parent run (optional)
    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("cerebrum-metaqa-tune")
            mlflow_run = mlflow.start_run(run_name=f"optuna_{args.study_name}")
            mlflow.log_params({"n_trials": args.n_trials, "sample": args.sample,
                               "search": str(args.search)})
        except ImportError:
            print("  [warning] mlflow not installed, skipping tracking")

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
    )

    # Seed with Phase 182 baseline (all params at current defaults)
    baseline_params = {p: _FIXED[p] for p in args.search}
    print(f"[trial   0] BASELINE {baseline_params}", flush=True)
    baseline = _run_config(baseline_params, sample=args.sample, seed=42)
    if baseline:
        print(f"           → H@1={baseline['hits1']*100:.2f}%  "
              f"H@10={baseline['hits10']*100:.2f}%  MRR={baseline['mrr']:.4f}", flush=True)
        study.enqueue_trial(baseline_params)

    objective = build_objective(args.search, sample=args.sample, mlflow_run=mlflow_run)
    study.optimize(objective, n_trials=args.n_trials)

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)
    best = study.best_trial
    print(f"\n  Best H@1 (sample n={args.sample}): {best.value*100:.2f}%")
    if baseline:
        delta = (best.value - baseline["hits1"]) * 100
        print(f"  vs baseline: {delta:+.2f}pp")
    print(f"\n  Best parameters:")
    for k, v in best.params.items():
        print(f"    {k:20s}: {v:.4f}")

    # Show top-5 trials
    trials_sorted = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
    print(f"\n  Top 5 trials:")
    for t in trials_sorted[:5]:
        param_str = "  ".join(f"{k}={v:.3f}" for k, v in t.params.items())
        print(f"    trial {t.number:3d}  H@1={t.value*100:.2f}%  {param_str}")

    # ---------------------------------------------------------------------------
    # Validation run at best params on larger sample
    # ---------------------------------------------------------------------------
    if args.validate > 0 and best.value > (baseline["hits1"] if baseline else 0):
        print(f"\n  Running validation at n={args.validate} with best params...", flush=True)
        val = _run_config(best.params, sample=args.validate, seed=42)
        if val:
            print(f"\n  VALIDATION RESULT:")
            print(f"    H@1  : {val['hits1']*100:.2f}%")
            print(f"    H@10 : {val['hits10']*100:.2f}%")
            print(f"    MRR  : {val['mrr']:.4f}")
            if baseline:
                print(f"    Δ H@1 vs baseline: {(val['hits1']-baseline['hits1'])*100:+.2f}pp")
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metrics({
                        "best_val_h1":  val["hits1"],
                        "best_val_h10": val["hits10"],
                        "best_val_mrr": val["mrr"],
                    })
                    mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
                except Exception:
                    pass

    print(f"\n  To run canonical benchmark with best params:")
    cmd_parts = ["python -u -m benchmarks.metaqa_eval",
                 "--hop 3 --embeddings sentence --beam-width 20",
                 "--use-prior --fhrb-factor 3.0 --workers 8"]
    for k, v in best.params.items():
        cmd_parts.append(f"--{k.replace('_', '-')} {v:.4f}")
    print(f"    {' '.join(cmd_parts)}")

    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()

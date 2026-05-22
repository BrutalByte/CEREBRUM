"""
CEREBRUM Live Hyperparameter Tuner
====================================
Searches all scoring parameters via Optuna TPE and displays a live Rich terminal
dashboard updated after every trial.

Parameters searched (11 total):
  Core:           trb-factor, r2-boost, vote-weight, beam-width, idf-weight, branch-bonus
  Per-relation:   wb-r2-boost, db-r2-boost, ry-r2-boost, sa-r2-boost
  First-hop:      fhrb-factor

Usage
-----
    python -u benchmarks/cerebrum_tuner.py --n-trials 200 --sample 500
    python -u benchmarks/cerebrum_tuner.py --n-trials 200 --sample 500 --validate 14274
    cerebrum tune --n-trials 200 --sample 500 --validate 14274

Install dependencies
---------------------
    pip install "cerebrum-kg[tuning]"
    # or: pip install optuna rich
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich import box as rich_box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

_EVAL_SCRIPT = Path(__file__).parent / "metaqa_eval.py"

# ── search space ──────────────────────────────────────────────────────────────
# r2-boost ceiling expanded to 10.0 (was hitting 5.0 in Phase 196 run).
# trb-factor expanded to [2.0, 8.0] around the Phase 196 optimum of 4.5.
# Per-relation r2 overrides allow independent tuning of the 4 highest-failure
# relation types. fhrb-factor is the first-hop relation boost (default 0.0).
PARAM_SPACE: dict = {
    # Core scoring
    "trb_factor":   (2.0,  8.0),
    "r2_boost":     (0.0,  10.0),
    "vote_weight":  (0.85, 0.99),
    "beam_width":   [8, 10, 12, 15],
    "idf_weight":   (0.0,  0.3),
    "branch_bonus": (0.0,  1.5),
    # First-hop relation boost
    "fhrb_factor":  (0.0,  3.0),
    # Per-relation r2-boost overrides (written_by 57%, directed_by 60%,
    # release_year 63%, starred_actors 31% failure rates from Phase 195 diag)
    "wb_r2_boost":  (0.0,  10.0),
    "db_r2_boost":  (0.0,  10.0),
    "ry_r2_boost":  (0.0,  10.0),
    "sa_r2_boost":  (0.0,  10.0),
}


# ── trial record ──────────────────────────────────────────────────────────────
@dataclass
class TrialRecord:
    trial_id:     int
    trb_factor:   float
    r2_boost:     float
    vote_weight:  float
    beam_width:   int
    idf_weight:   float
    branch_bonus: float
    fhrb_factor:  float
    wb_r2_boost:  float
    db_r2_boost:  float
    ry_r2_boost:  float
    sa_r2_boost:  float
    h1:           float
    h10:          float
    mrr:          float
    elapsed_s:    float
    is_best:      bool = False


# ── live dashboard ────────────────────────────────────────────────────────────
class LiveDashboard:
    """Rich live terminal dashboard. Updated after every completed trial."""

    def __init__(self, n_trials: int, sample: int, study_name: str) -> None:
        self.n_trials   = n_trials
        self.sample     = sample
        self.study_name = study_name
        self.records:   list[TrialRecord]     = []
        self.best:      Optional[TrialRecord] = None
        self._start     = time.time()

    def push(self, rec: TrialRecord) -> None:
        if self.best is None or rec.h1 > self.best.h1:
            if self.best:
                self.best.is_best = False
            rec.is_best = True
            self.best = rec
        self.records.append(rec)

    # ── renderers ─────────────────────────────────────────────────────────
    def _header_panel(self) -> "Panel":
        elapsed = time.time() - self._start
        done    = len(self.records)
        eta_s   = (elapsed / done * (self.n_trials - done)) if done else 0.0
        eta_str = f"{eta_s/60:.0f}m" if eta_s > 90 else f"{eta_s:.0f}s"

        if self.best:
            b = self.best
            body = (
                f"[bold green]H@1 = {b.h1*100:.2f}%[/bold green]   "
                f"H@10 = {b.h10*100:.2f}%   MRR = {b.mrr:.4f}   "
                f"[dim](trial #{b.trial_id})[/dim]\n"
                f"  trb=[cyan]{b.trb_factor:.3f}[/cyan]  "
                f"r2=[cyan]{b.r2_boost:.3f}[/cyan]  "
                f"vote=[cyan]{b.vote_weight:.4f}[/cyan]  "
                f"beam=[cyan]{b.beam_width}[/cyan]  "
                f"idf=[cyan]{b.idf_weight:.3f}[/cyan]  "
                f"bbns=[cyan]{b.branch_bonus:.3f}[/cyan]  "
                f"fhrb=[cyan]{b.fhrb_factor:.3f}[/cyan]\n"
                f"  wb=[cyan]{b.wb_r2_boost:.3f}[/cyan]  "
                f"db=[cyan]{b.db_r2_boost:.3f}[/cyan]  "
                f"ry=[cyan]{b.ry_r2_boost:.3f}[/cyan]  "
                f"sa=[cyan]{b.sa_r2_boost:.3f}[/cyan]"
            )
        else:
            body = "[dim]Waiting for first trial...[/dim]"

        title = (
            f"[bold]CEREBRUM Tuner[/bold] — {self.study_name}   "
            f"Trial [bold]{done}[/bold]/{self.n_trials}   "
            f"Elapsed {elapsed:.0f}s   ETA ~{eta_str}   "
            f"Sample {self.sample:,}q"
        )
        border = "green" if self.best else "blue"
        return Panel(body, title=title, border_style=border)

    def _history_table(self) -> "Table":
        t = Table(
            box=rich_box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
            show_edge=False,
        )
        # Core columns
        t.add_column("#",     justify="right", style="dim", no_wrap=True, width=4)
        t.add_column("trb",   justify="right",              no_wrap=True, width=5)
        t.add_column("r2",    justify="right",              no_wrap=True, width=5)
        t.add_column("vote",  justify="right",              no_wrap=True, width=6)
        t.add_column("bm",    justify="right",              no_wrap=True, width=4)
        t.add_column("idf",   justify="right",              no_wrap=True, width=5)
        t.add_column("bbns",  justify="right",              no_wrap=True, width=5)
        t.add_column("fhrb",  justify="right",              no_wrap=True, width=5)
        # Per-relation r2 overrides
        t.add_column("wb",    justify="right",              no_wrap=True, width=5)
        t.add_column("db",    justify="right",              no_wrap=True, width=5)
        t.add_column("ry",    justify="right",              no_wrap=True, width=5)
        t.add_column("sa",    justify="right",              no_wrap=True, width=5)
        # Metrics
        t.add_column("H@1",   justify="right",              no_wrap=True, width=7)
        t.add_column("H@10",  justify="right",              no_wrap=True, width=7)
        t.add_column("MRR",   justify="right",              no_wrap=True, width=6)
        t.add_column("Δbest", justify="right",              no_wrap=True, width=7)
        t.add_column("sec",   justify="right",              no_wrap=True, width=5)

        for r in list(reversed(self.records))[:30]:
            delta_pp = (r.h1 - self.best.h1) * 100 if self.best else 0.0
            if r.is_best:
                delta_str = "[bold green]★best[/bold green]"
            elif delta_pp < -2:
                delta_str = f"[dim red]{delta_pp:+.1f}[/dim red]"
            elif delta_pp < 0:
                delta_str = f"[red]{delta_pp:+.1f}[/red]"
            else:
                delta_str = f"[green]{delta_pp:+.1f}[/green]"

            row_style = "bold green" if r.is_best else ""
            t.add_row(
                str(r.trial_id),
                f"{r.trb_factor:.2f}",
                f"{r.r2_boost:.2f}",
                f"{r.vote_weight:.3f}",
                str(r.beam_width),
                f"{r.idf_weight:.2f}",
                f"{r.branch_bonus:.2f}",
                f"{r.fhrb_factor:.2f}",
                f"{r.wb_r2_boost:.2f}",
                f"{r.db_r2_boost:.2f}",
                f"{r.ry_r2_boost:.2f}",
                f"{r.sa_r2_boost:.2f}",
                f"{r.h1*100:.1f}%",
                f"{r.h10*100:.1f}%",
                f"{r.mrr:.4f}",
                delta_str,
                f"{r.elapsed_s:.0f}",
                style=row_style,
            )
        return t

    def __rich__(self):
        from rich.console import Group
        return Group(self._header_panel(), self._history_table())


# ── JSONL trial logger ────────────────────────────────────────────────────────
def _write_log(path: Path, record: dict) -> None:
    """Append one JSON line to the trial log. Creates file on first write."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trial_record_to_dict(
    rec: "TrialRecord",
    sample: int,
    run_id: str,
    all_records: "list[TrialRecord]",
) -> dict:
    """Flatten a TrialRecord into a log-friendly dict with cross-trial context."""
    # Rank among all completed trials so far (1 = best H@1)
    sorted_h1  = sorted((r.h1 for r in all_records), reverse=True)
    rank       = sorted_h1.index(rec.h1) + 1
    best_so_far = max(r.h1 for r in all_records)
    # Per-param distance from previous trial (shows search direction)
    prev = all_records[-2] if len(all_records) >= 2 else None
    param_deltas: dict = {}
    if prev:
        for attr in ("trb_factor", "r2_boost", "vote_weight", "beam_width",
                     "idf_weight", "branch_bonus", "fhrb_factor",
                     "wb_r2_boost", "db_r2_boost", "ry_r2_boost", "sa_r2_boost"):
            param_deltas[f"d_{attr}"] = round(
                getattr(rec, attr) - getattr(prev, attr), 4
            )
    return {
        "type":              "trial",
        "run_id":            run_id,
        "trial_id":          rec.trial_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample":            sample,
        # Parameters
        "trb_factor":        rec.trb_factor,
        "r2_boost":          rec.r2_boost,
        "vote_weight":       rec.vote_weight,
        "beam_width":        rec.beam_width,
        "idf_weight":        rec.idf_weight,
        "branch_bonus":      rec.branch_bonus,
        "fhrb_factor":       rec.fhrb_factor,
        "wb_r2_boost":       rec.wb_r2_boost,
        "db_r2_boost":       rec.db_r2_boost,
        "ry_r2_boost":       rec.ry_r2_boost,
        "sa_r2_boost":       rec.sa_r2_boost,
        # Metrics
        "h1":                rec.h1,
        "h10":               rec.h10,
        "mrr":               rec.mrr,
        "elapsed_s":         round(rec.elapsed_s, 1),
        # Cross-trial context
        "is_best":           rec.is_best,
        "rank_so_far":       rank,
        "trials_completed":  len(all_records),
        "best_h1_so_far":    round(best_so_far, 6),
        "delta_from_best":   round(rec.h1 - best_so_far, 6),
        **param_deltas,
    }


# ── subprocess eval ───────────────────────────────────────────────────────────
def _run_eval(
    sample:       int,
    trb_factor:   float,
    r2_boost:     float,
    vote_weight:  float,
    beam_width:   int,
    idf_weight:   float,
    branch_bonus: float,
    fhrb_factor:  float,
    wb_r2_boost:  float,
    db_r2_boost:  float,
    ry_r2_boost:  float,
    sa_r2_boost:  float,
    timeout:      int = 600,
) -> tuple[float, float, float, float]:
    """
    Call metaqa_eval.py as a blocking subprocess and return (h1, h10, mrr, elapsed_s).
    Using subprocess.run() (not Popen) avoids WinError 232 pipe issues on Windows.
    """
    cmd = [
        sys.executable, "-u", str(_EVAL_SCRIPT),
        "--hop",           "3",
        "--sample",        str(sample),
        "--beam-width",    str(beam_width),
        "--embeddings",    "sentence",
        "--vote-weight",   str(vote_weight),
        "--trb-factor",    str(trb_factor),
        "--r2-boost",      str(r2_boost),
        "--idf-weight",    str(idf_weight),
        "--branch-bonus",  str(branch_bonus),
        "--fhrb-factor",   str(fhrb_factor),
        "--wb-r2-boost",   str(wb_r2_boost),
        "--db-r2-boost",   str(db_r2_boost),
        "--ry-r2-boost",   str(ry_r2_boost),
        "--sa-r2-boost",   str(sa_r2_boost),
        "--workers",       "1",
    ]
    t0   = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0

    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "3-hop" and len(parts) >= 5:
            try:
                return float(parts[2]), float(parts[3]), float(parts[4]), elapsed
            except ValueError:
                pass

    raise RuntimeError(
        f"Could not parse eval output (exit {proc.returncode}).\n"
        f"stdout: {proc.stdout[-800:]}\n"
        f"stderr: {proc.stderr[-400:]}"
    )


def _run_eval_logged(
    log_file: Path,
    run_id:   str,
    trial_id: int,
    **kwargs,
) -> tuple[float, float, float, float]:
    """
    Wrapper around _run_eval that captures and logs every stdout line from the
    eval subprocess as it completes, keyed by run_id + trial_id.
    """
    # Re-run eval but capture stdout line by line via Popen so we can log it.
    sample = kwargs["sample"]
    cmd = [
        sys.executable, "-u", str(_EVAL_SCRIPT),
        "--hop",           "3",
        "--sample",        str(sample),
        "--beam-width",    str(kwargs["beam_width"]),
        "--embeddings",    "sentence",
        "--vote-weight",   str(kwargs["vote_weight"]),
        "--trb-factor",    str(kwargs["trb_factor"]),
        "--r2-boost",      str(kwargs["r2_boost"]),
        "--idf-weight",    str(kwargs["idf_weight"]),
        "--branch-bonus",  str(kwargs["branch_bonus"]),
        "--fhrb-factor",   str(kwargs["fhrb_factor"]),
        "--wb-r2-boost",   str(kwargs["wb_r2_boost"]),
        "--db-r2-boost",   str(kwargs["db_r2_boost"]),
        "--ry-r2-boost",   str(kwargs["ry_r2_boost"]),
        "--sa-r2-boost",   str(kwargs["sa_r2_boost"]),
        "--workers",       "1",
    ]
    t0      = time.time()
    timeout = kwargs.get("timeout", 600)
    lines: list[str] = []
    h1 = h10 = mrr = 0.0
    parsed = False

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            parts = line.split()
            if parts and parts[0] == "3-hop" and len(parts) >= 5:
                try:
                    h1, h10, mrr = float(parts[2]), float(parts[3]), float(parts[4])
                    parsed = True
                except ValueError:
                    pass
            if time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError(f"Trial {trial_id} timed out after {timeout}s")
        proc.wait()
    finally:
        proc.stdout.close()

    elapsed = time.time() - t0

    # Log all eval output lines
    _write_log(log_file, {
        "type":      "trial_stdout",
        "run_id":    run_id,
        "trial_id":  trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lines":     lines,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
    })

    if not parsed:
        raise RuntimeError(
            f"Could not parse eval output for trial {trial_id} "
            f"(exit {proc.returncode}).\n" + "\n".join(lines[-20:])
        )
    return h1, h10, mrr, elapsed


# ── tuning loop ───────────────────────────────────────────────────────────────
def run_tuner(
    n_trials:   int           = 200,
    sample:     int           = 500,
    study_name: str           = "cerebrum-tuner",
    validate:   int           = 0,
    seed:       int           = 42,
    timeout_s:  int           = 600,
    log_file:   Optional[Path] = None,
) -> Optional[TrialRecord]:
    """
    Run Optuna TPE search and stream results to a Rich live dashboard.

    Parameters
    ----------
    n_trials   : total number of Optuna trials (200+ recommended for 11 params)
    sample     : questions per trial (500 ≈ 30s; 2000 ≈ 60s; 14274 = full run)
    study_name : label shown in the dashboard header
    validate   : if > 0, run best config on this many questions after search
    seed       : TPE sampler seed for reproducibility
    timeout_s  : per-trial subprocess timeout in seconds
    log_file   : JSONL file to append one record per trial (auto-generated if None)
    """
    if not _HAS_OPTUNA:
        print(
            "optuna is not installed.\n"
            "Install with:  pip install 'cerebrum-kg[tuning]'  or  pip install optuna rich",
            file=sys.stderr,
        )
        return None

    # ── log file setup ────────────────────────────────────────────────────
    run_id  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if log_file is None:
        log_file = Path(__file__).parent / f"tuner_{run_id}.jsonl"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Write run header as first line
    _write_log(log_file, {
        "type": "run_start", "run_id": run_id, "study_name": study_name,
        "n_trials": n_trials, "sample": sample, "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "param_space": {k: list(v) if isinstance(v, list) else list(v)
                        for k, v in PARAM_SPACE.items()},
    })
    print(f"  Trial log: {log_file}")

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=15)
    study   = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    console   = Console() if _HAS_RICH else None
    dashboard = LiveDashboard(n_trials, sample, study_name)

    def objective(trial: "optuna.Trial") -> float:
        trb_factor   = trial.suggest_float("trb_factor",   *PARAM_SPACE["trb_factor"])
        r2_boost     = trial.suggest_float("r2_boost",     *PARAM_SPACE["r2_boost"])
        vote_weight  = trial.suggest_float("vote_weight",  *PARAM_SPACE["vote_weight"])
        beam_width   = trial.suggest_categorical("beam_width", PARAM_SPACE["beam_width"])
        idf_weight   = trial.suggest_float("idf_weight",   *PARAM_SPACE["idf_weight"])
        branch_bonus = trial.suggest_float("branch_bonus", *PARAM_SPACE["branch_bonus"])
        fhrb_factor  = trial.suggest_float("fhrb_factor",  *PARAM_SPACE["fhrb_factor"])
        wb_r2_boost  = trial.suggest_float("wb_r2_boost",  *PARAM_SPACE["wb_r2_boost"])
        db_r2_boost  = trial.suggest_float("db_r2_boost",  *PARAM_SPACE["db_r2_boost"])
        ry_r2_boost  = trial.suggest_float("ry_r2_boost",  *PARAM_SPACE["ry_r2_boost"])
        sa_r2_boost  = trial.suggest_float("sa_r2_boost",  *PARAM_SPACE["sa_r2_boost"])

        h1, h10, mrr, elapsed = _run_eval_logged(
            log_file=log_file,
            run_id=run_id,
            trial_id=trial.number,
            sample=sample,
            trb_factor=trb_factor,
            r2_boost=r2_boost,
            vote_weight=vote_weight,
            beam_width=beam_width,
            idf_weight=idf_weight,
            branch_bonus=branch_bonus,
            fhrb_factor=fhrb_factor,
            wb_r2_boost=wb_r2_boost,
            db_r2_boost=db_r2_boost,
            ry_r2_boost=ry_r2_boost,
            sa_r2_boost=sa_r2_boost,
            timeout=timeout_s,
        )
        rec = TrialRecord(
            trial_id=trial.number,
            trb_factor=trb_factor,
            r2_boost=r2_boost,
            vote_weight=vote_weight,
            beam_width=beam_width,
            idf_weight=idf_weight,
            branch_bonus=branch_bonus,
            fhrb_factor=fhrb_factor,
            wb_r2_boost=wb_r2_boost,
            db_r2_boost=db_r2_boost,
            ry_r2_boost=ry_r2_boost,
            sa_r2_boost=sa_r2_boost,
            h1=h1, h10=h10, mrr=mrr,
            elapsed_s=elapsed,
        )
        dashboard.push(rec)
        _write_log(log_file, _trial_record_to_dict(rec, sample, run_id, dashboard.records))
        return h1

    # ── run with Rich live display ─────────────────────────────────────────
    if _HAS_RICH and console is not None:
        with Live(dashboard, console=console, refresh_per_second=2, screen=False):
            for _ in range(n_trials):
                study.optimize(objective, n_trials=1, catch=(Exception,))
    else:
        # Plain-text fallback when Rich is not installed
        header = (
            f"{'#':>4}  {'trb':>5}  {'r2':>5}  {'vote':>6}  {'bm':>3}  "
            f"{'idf':>5}  {'bbns':>5}  {'fhrb':>5}  "
            f"{'wb':>5}  {'db':>5}  {'ry':>5}  {'sa':>5}  "
            f"{'H@1':>7}  {'H@10':>7}  {'MRR':>6}  B"
        )
        print(header)
        print("-" * len(header))
        for _ in range(n_trials):
            study.optimize(objective, n_trials=1, catch=(Exception,))
            if dashboard.records:
                r = dashboard.records[-1]
                flag = "★" if r.is_best else " "
                print(
                    f"{r.trial_id:>4}  {r.trb_factor:>5.2f}  {r.r2_boost:>5.2f}  "
                    f"{r.vote_weight:>6.3f}  {r.beam_width:>3}  "
                    f"{r.idf_weight:>5.2f}  {r.branch_bonus:>5.2f}  {r.fhrb_factor:>5.2f}  "
                    f"{r.wb_r2_boost:>5.2f}  {r.db_r2_boost:>5.2f}  "
                    f"{r.ry_r2_boost:>5.2f}  {r.sa_r2_boost:>5.2f}  "
                    f"{r.h1*100:>6.2f}%  {r.h10*100:>6.2f}%  {r.mrr:>6.4f}  {flag}"
                )

    best = dashboard.best
    if best is None:
        (console.print if console else print)("[red]No successful trials.[/red]")
        return None

    # ── final summary ─────────────────────────────────────────────────────
    _print = console.print if console else print
    if console:
        console.rule("[bold green]Tuning Complete[/bold green]")
    _print(
        f"\nBest result:  H@1={best.h1*100:.2f}%  "
        f"H@10={best.h10*100:.2f}%  MRR={best.mrr:.4f}"
    )
    _print(
        f"  trb-factor={best.trb_factor:.3f}  r2-boost={best.r2_boost:.3f}  "
        f"vote-weight={best.vote_weight:.4f}  beam-width={best.beam_width}  "
        f"idf-weight={best.idf_weight:.3f}  branch-bonus={best.branch_bonus:.3f}  "
        f"fhrb-factor={best.fhrb_factor:.3f}"
    )
    _print(
        f"  wb-r2-boost={best.wb_r2_boost:.3f}  db-r2-boost={best.db_r2_boost:.3f}  "
        f"ry-r2-boost={best.ry_r2_boost:.3f}  sa-r2-boost={best.sa_r2_boost:.3f}"
    )
    canonical = (
        f"python -u benchmarks/metaqa_eval.py --hop 3 "
        f"--beam-width {best.beam_width} --embeddings sentence "
        f"--trb-factor {best.trb_factor:.3f} --r2-boost {best.r2_boost:.3f} "
        f"--vote-weight {best.vote_weight:.4f} --idf-weight {best.idf_weight:.3f} "
        f"--branch-bonus {best.branch_bonus:.3f} --fhrb-factor {best.fhrb_factor:.3f} "
        f"--wb-r2-boost {best.wb_r2_boost:.3f} --db-r2-boost {best.db_r2_boost:.3f} "
        f"--ry-r2-boost {best.ry_r2_boost:.3f} --sa-r2-boost {best.sa_r2_boost:.3f}"
    )
    _print(f"\nCanonical benchmark command:\n  {canonical}\n")
    _print(f"  Trial log: {log_file}")

    # Write best-config summary record to log
    best_dict = _trial_record_to_dict(best, sample, run_id, dashboard.records)
    best_dict["type"] = "best_config"
    best_dict["canonical_cmd"] = canonical
    _write_log(log_file, best_dict)

    # Write Optuna parameter importance scores
    try:
        importances = optuna.importance.get_param_importances(study)
        _write_log(log_file, {
            "type":       "param_importances",
            "run_id":     run_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method":     "fanova",
            "importances": dict(importances),
        })
        _print("\nParameter importances (fANOVA):")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(imp * 40)
            _print(f"  {param:<16} {imp:.4f}  {bar}")
    except Exception:
        pass

    # ── optional validation run ───────────────────────────────────────────
    if validate > 0:
        _print(f"Validating best config on {validate:,} questions...")
        try:
            vh1, vh10, vmrr, vel = _run_eval(
                sample=validate,
                trb_factor=best.trb_factor,
                r2_boost=best.r2_boost,
                vote_weight=best.vote_weight,
                beam_width=best.beam_width,
                idf_weight=best.idf_weight,
                branch_bonus=best.branch_bonus,
                fhrb_factor=best.fhrb_factor,
                wb_r2_boost=best.wb_r2_boost,
                db_r2_boost=best.db_r2_boost,
                ry_r2_boost=best.ry_r2_boost,
                sa_r2_boost=best.sa_r2_boost,
                timeout=3600,
            )
            _print(
                f"  Validation:  H@1={vh1*100:.2f}%  "
                f"H@10={vh10*100:.2f}%  MRR={vmrr:.4f}  ({vel:.0f}s)"
            )
            _write_log(log_file, {
                "type": "validation", "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sample": validate, "h1": vh1, "h10": vh10, "mrr": vmrr,
                "elapsed_s": round(vel, 1),
            })
        except Exception as exc:
            _print(f"  Validation failed: {exc}")

    return best


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cerebrum_tuner",
        description="CEREBRUM live hyperparameter tuner (Optuna + Rich dashboard)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=200, dest="n_trials",
        help="Optuna trials to run (default 200; ~100min at 500q/trial)",
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Questions per trial (default 500 ≈ 30s; use 2000 for stability)",
    )
    parser.add_argument(
        "--validate", type=int, default=0,
        help="After tuning, run best config on N questions (0=skip, 14274=full dataset)",
    )
    parser.add_argument(
        "--study-name", type=str, default="cerebrum-tuner", dest="study_name",
        help="Optuna study name shown in dashboard header",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-file", type=Path, default=None, dest="log_file",
        help="JSONL file to log every trial (default: benchmarks/tuner_<timestamp>.jsonl). "
             "Each line is a self-contained JSON record with params, metrics, and "
             "cross-trial context (rank, delta-from-best, per-param deltas vs previous trial).",
    )
    args = parser.parse_args()

    run_tuner(
        n_trials=args.n_trials,
        sample=args.sample,
        study_name=args.study_name,
        validate=args.validate,
        seed=args.seed,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()

"""
CEREBRUM Live Hyperparameter Tuner
====================================
Two-phase search: Phase 1 uses Sobol quasi-random sampling (QMCSampler) for
maximum uniform coverage of the 9D parameter space; Phase 2 uses CMA-ES
(CmaEsSampler) initialized from the best Phase 1 config and warmed with all
Phase 1 trials, enabling the optimizer to model parameter correlations that
TPE treats as independent dimensions.

Parameters searched (9 total):
  Core:           trb-factor, r2-boost, vote-weight, beam-width, idf-weight, branch-bonus
  First-hop:      fhrb-factor
  SDRB (Phase 203): gamma, beta  (boost(r) = gamma * fan_out(r)^beta)

Usage
-----
    # Two-phase (default): 60 exploration + 140 fine-tuning
    python -u benchmarks/cerebrum_tuner.py --sample 500 --validate 14274

    # Custom split
    python -u benchmarks/cerebrum_tuner.py --phase1-trials 80 --phase2-trials 160 --sample 500

    # Resume after restart — skip Phase 1, jump straight to Phase 2
    python -u benchmarks/cerebrum_tuner.py --resume benchmarks/tuner_<run_id>.jsonl --phase2-trials 140 --sample 2000 --workers 16

    # Single-phase (legacy TPE, backward-compatible)
    python -u benchmarks/cerebrum_tuner.py --n-trials 200 --sample 500

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
from dataclasses import dataclass
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

_EVAL_SCRIPT          = Path(__file__).parent / "metaqa_eval.py"
_HETIONET_EVAL_SCRIPT = Path(__file__).parent / "hetionet_param_eval.py"
_DEFAULT_KB  = Path(__file__).parent / "data" / "metaqa" / "kb.txt"

# ── search space (Phase 1 — wide exploration) ─────────────────────────────────
# Phase 203: two-parameter SDRB — gamma ceiling raised to 16.0 (hit 7.93/8.0 in
# Phase 202); beta added [0.5, 3.0] to control fan_out shaping.
# branch_bonus raised to [0.0, 1.0] — was #1 fANOVA driver at 46% in Phase 202.
# trb_factor ceiling raised to 35.0 — was #2 driver at 29%.
PARAM_SPACE_WIDE: dict = {
    "trb_factor":   (18.0, 35.0),
    "r2_boost":     (2.0,  10.0),
    "vote_weight":  (0.75, 0.95),
    "beam_width":   [8, 10, 12],
    "idf_weight":   (0.0,  0.15),
    "branch_bonus": (0.0,  1.0),
    "fhrb_factor":  (3.0,  8.0),
    "gamma":        (1.5,  16.0),  # Phase 203: expanded ceiling (hit 7.93/8.0 in P202)
    "beta":         (0.5,  3.0),   # Phase 203: power-law exponent for fan_out shaping
}

# Phase 206: Hetionet (typed_heterogeneous) search space.
# Centered around ParameterInitializer typed_heterogeneous predictions:
#   gamma~2.30, beta=1.0, trb~19.6, fhrb~2.64, r2~4.27, idf~0.043, branch=0.17
PARAM_SPACE_HETIONET: dict = {
    "trb_factor":   (2.0,  25.0),
    "r2_boost":     (1.0,  10.0),
    "vote_weight":  (0.60, 0.95),
    "beam_width":   [8, 10, 12],
    "idf_weight":   (0.0,  0.15),
    "branch_bonus": (0.0,  0.80),
    "fhrb_factor":  (1.0,  5.0),
    "gamma":        (0.5,  8.0),
    "beta":         (0.5,  2.0),
}

# Float param names (excludes categorical beam_width)
_FLOAT_PARAMS = tuple(k for k, v in PARAM_SPACE_WIDE.items() if not isinstance(v, list))


# ── param-space helpers ───────────────────────────────────────────────────────
def _suggest_params(trial: "optuna.Trial", space: dict) -> dict:
    """Suggest all params from the given space dict."""
    out: dict = {}
    for name, bounds in space.items():
        if isinstance(bounds, list):
            out[name] = trial.suggest_categorical(name, bounds)
        else:
            out[name] = trial.suggest_float(name, *bounds)
    return out


def _compute_refined_space(
    records: "list[TrialRecord]",
    wide_space: dict,
    top_k: int = 10,
    margin: float = 0.25,
) -> dict:
    """
    Derive Phase 2 bounds from the top-K trials of Phase 1.
    Each float param gets [min_topk - margin*span, max_topk + margin*span],
    clamped to the wide-space limits. Categorical params are unchanged.
    """
    top = sorted(records, key=lambda r: r.h1, reverse=True)[:min(top_k, len(records))]
    refined: dict = {}
    for name, bounds in wide_space.items():
        if isinstance(bounds, list):
            refined[name] = bounds
            continue
        vals = [getattr(r, name) for r in top]
        lo, hi = min(vals), max(vals)
        full_span = bounds[1] - bounds[0]
        span = max(hi - lo, full_span * 0.05)   # floor at 5% of full range
        new_lo = max(bounds[0], lo - span * margin)
        new_hi = min(bounds[1], hi + span * margin)
        refined[name] = (round(new_lo, 4), round(new_hi, 4))
    return refined


# ── resume helpers ───────────────────────────────────────────────────────────
def _load_resume(path: Path) -> tuple[list["TrialRecord"], Optional[dict]]:
    """
    Read a previous tuner JSONL and return (phase1_records, refined_space).
    refined_space comes from the saved phase2_bounds entry if present;
    otherwise it is None and the caller should recompute it.
    """
    raw_records: list[dict] = []
    refined_space: Optional[dict] = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "trial" and obj.get("phase") == 1:
                raw_records.append(obj)
            elif t == "phase2_bounds":
                raw = obj["refined_space"]
                refined_space = {}
                for k, v in raw.items():
                    if isinstance(PARAM_SPACE_WIDE.get(k), list):
                        refined_space[k] = v          # categorical — keep as list
                    else:
                        refined_space[k] = (v[0], v[1])  # float range tuple

    records: list[TrialRecord] = []
    best_h1 = -1.0
    for obj in raw_records:
        rec = TrialRecord(
            trial_id=obj["trial_id"],
            phase=1,
            trb_factor=obj["trb_factor"],
            r2_boost=obj["r2_boost"],
            vote_weight=obj["vote_weight"],
            beam_width=obj["beam_width"],
            idf_weight=obj["idf_weight"],
            branch_bonus=obj["branch_bonus"],
            fhrb_factor=obj["fhrb_factor"],
            gamma=obj["gamma"],
            beta=obj["beta"],
            h1=obj["h1"],
            h10=obj["h10"],
            mrr=obj["mrr"],
            elapsed_s=obj["elapsed_s"],
        )
        if rec.h1 > best_h1:
            best_h1 = rec.h1
            rec.is_best = True
            if records:
                records[-1].is_best = False  # clear prior best
            # mark previous best as not best
            for r in records:
                r.is_best = False
            rec.is_best = True
        records.append(rec)
    return records, refined_space


def _make_source_trials(records: "list[TrialRecord]", space: dict) -> list:
    """Build Optuna FrozenTrial objects from TrialRecord list for CMA-ES warm-start."""
    from optuna.distributions import FloatDistribution, CategoricalDistribution

    distributions: dict = {}
    for name, bounds in space.items():
        if isinstance(bounds, list):
            distributions[name] = CategoricalDistribution(bounds)
        else:
            distributions[name] = FloatDistribution(bounds[0], bounds[1])

    frozen: list = []
    for rec in records:
        params = {
            "trb_factor":   rec.trb_factor,
            "r2_boost":     rec.r2_boost,
            "vote_weight":  rec.vote_weight,
            "beam_width":   rec.beam_width,
            "idf_weight":   rec.idf_weight,
            "branch_bonus": rec.branch_bonus,
            "fhrb_factor":  rec.fhrb_factor,
            "gamma":        rec.gamma,
            "beta":         rec.beta,
        }
        frozen.append(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=rec.h1,
            )
        )
    return frozen


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
    gamma:        float   # Phase 203 SDRB: scale factor
    beta:         float   # Phase 203 SDRB: fan_out exponent; boost(r) = gamma * fan_out(r)^beta
    h1:           float
    h10:          float
    mrr:          float
    elapsed_s:    float
    is_best:      bool = False
    phase:        int  = 1


# ── live dashboard ────────────────────────────────────────────────────────────
class LiveDashboard:
    """Rich live terminal dashboard. Updated after every completed trial."""

    def __init__(self, n_trials: int, sample: int, study_name: str) -> None:
        self.n_trials    = n_trials
        self.sample      = sample
        self.study_name  = study_name
        self.phase_label = "Phase 1/2 — Exploring"
        self.records:    list[TrialRecord]     = []
        self.best:       Optional[TrialRecord] = None
        self._start      = time.time()

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
                f"fhrb=[cyan]{b.fhrb_factor:.3f}[/cyan]  "
                f"gamma=[cyan]{b.gamma:.3f}[/cyan]  "
                f"beta=[cyan]{b.beta:.3f}[/cyan]"
            )
        else:
            body = "[dim]Waiting for first trial...[/dim]"

        title = (
            f"[bold]CEREBRUM Tuner[/bold] — {self.study_name}   "
            f"[yellow]{self.phase_label}[/yellow]   "
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
        t.add_column("Ph",    justify="right", style="dim", no_wrap=True, width=3)
        t.add_column("#",     justify="right", style="dim", no_wrap=True, width=4)
        t.add_column("trb",   justify="right",              no_wrap=True, width=5)
        t.add_column("r2",    justify="right",              no_wrap=True, width=5)
        t.add_column("vote",  justify="right",              no_wrap=True, width=6)
        t.add_column("bm",    justify="right",              no_wrap=True, width=4)
        t.add_column("idf",   justify="right",              no_wrap=True, width=5)
        t.add_column("bbns",  justify="right",              no_wrap=True, width=5)
        t.add_column("fhrb",  justify="right",              no_wrap=True, width=5)
        t.add_column("gamma", justify="right",              no_wrap=True, width=6)
        t.add_column("beta",  justify="right",              no_wrap=True, width=5)
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

            ph_style  = "dim"  if r.phase == 1 else "cyan"
            row_style = "bold green" if r.is_best else ""
            t.add_row(
                f"[{ph_style}]P{r.phase}[/{ph_style}]",
                str(r.trial_id),
                f"{r.trb_factor:.2f}",
                f"{r.r2_boost:.2f}",
                f"{r.vote_weight:.3f}",
                str(r.beam_width),
                f"{r.idf_weight:.2f}",
                f"{r.branch_bonus:.2f}",
                f"{r.fhrb_factor:.2f}",
                f"{r.gamma:.3f}",
                f"{r.beta:.3f}",
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
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trial_record_to_dict(
    rec: "TrialRecord",
    sample: int,
    run_id: str,
    all_records: "list[TrialRecord]",
) -> dict:
    sorted_h1    = sorted((r.h1 for r in all_records), reverse=True)
    rank         = sorted_h1.index(rec.h1) + 1
    best_so_far  = max(r.h1 for r in all_records)
    prev         = all_records[-2] if len(all_records) >= 2 else None
    param_deltas: dict = {}
    if prev:
        for attr in _FLOAT_PARAMS:
            param_deltas[f"d_{attr}"] = round(getattr(rec, attr) - getattr(prev, attr), 4)
    return {
        "type":             "trial",
        "run_id":           run_id,
        "trial_id":         rec.trial_id,
        "phase":            rec.phase,
        "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample":           sample,
        "trb_factor":       rec.trb_factor,
        "r2_boost":         rec.r2_boost,
        "vote_weight":      rec.vote_weight,
        "beam_width":       rec.beam_width,
        "idf_weight":       rec.idf_weight,
        "branch_bonus":     rec.branch_bonus,
        "fhrb_factor":      rec.fhrb_factor,
        "gamma":            rec.gamma,
        "beta":             rec.beta,
        "h1":               rec.h1,
        "h10":              rec.h10,
        "mrr":              rec.mrr,
        "elapsed_s":        round(rec.elapsed_s, 1),
        "is_best":          rec.is_best,
        "rank_so_far":      rank,
        "trials_completed": len(all_records),
        "best_h1_so_far":   round(best_so_far, 6),
        "delta_from_best":  round(rec.h1 - best_so_far, 6),
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
    gamma:        float,
    beta:         float,
    timeout:      int = 600,
    workers:      int = 1,
) -> tuple[float, float, float, float]:
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
        "--gamma",         str(gamma),
        "--beta",          str(beta),
        "--workers",       str(workers),
    ]
    t0      = time.time()
    proc    = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    dataset = kwargs.get("dataset", "metaqa")
    if dataset == "hetionet":
        n_questions = kwargs.get("sample", 50)
        cmd = [
            sys.executable, "-u", str(_HETIONET_EVAL_SCRIPT),
            "--n-questions",  str(n_questions),
            "--beam-width",   str(kwargs["beam_width"]),
            "--vote-weight",  str(kwargs["vote_weight"]),
            "--trb-factor",   str(kwargs["trb_factor"]),
            "--r2-boost",     str(kwargs["r2_boost"]),
            "--idf-weight",   str(kwargs["idf_weight"]),
            "--branch-bonus", str(kwargs["branch_bonus"]),
            "--fhrb-factor",  str(kwargs["fhrb_factor"]),
            "--gamma",        str(kwargs["gamma"]),
            "--beta",         str(kwargs["beta"]),
            "--embeddings",   "sentence",
            "--max-loops",    "1",   # keep 1 during tuning for speed
            "--workers",      str(kwargs.get("workers", 1)),
            "--min-eval-hop", "2",   # skip near-ceiling 1-hop in tuning
            "--max-neighbors","50",  # tuning needs relative signal, not absolute accuracy
        ]
    else:
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
            "--gamma",         str(kwargs["gamma"]),
            "--beta",          str(kwargs["beta"]),
            "--workers",       str(kwargs.get("workers", 1)),
        ]
    t0      = time.time()
    timeout = kwargs.get("timeout", 600)
    lines:  list[str] = []
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


# ── ParameterInitializer warm-start ──────────────────────────────────────────
def _compute_param_init_x0(kb_file: Path) -> Optional[dict]:
    """
    Compute ParameterInitializer defaults from the KB and return as a param dict
    suitable for p1_study.enqueue_trial().  Returns None on any error (graceful
    fallback — tuner continues without the warm-start hint).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.graph_profiler import GraphProfiler
        from core.relation_boost_deriver import RelationBoostDeriver
        from core.parameter_initializer import ParameterInitializer
        import networkx as nx

        deriver = RelationBoostDeriver()
        deriver.build_from_file(str(kb_file))

        G = nx.Graph()
        with open(kb_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    h, _, t = parts
                    G.add_edge(h, t)

        class _FakeAdapter:
            def to_networkx(self): return G
            def is_directed(self): return False

        profile = GraphProfiler.profile(_FakeAdapter(), {})
        init    = ParameterInitializer.compute(profile, deriver)
        d       = init.as_dict()
        # Clamp to PARAM_SPACE_WIDE bounds so enqueue_trial doesn't error
        for name, bounds in PARAM_SPACE_WIDE.items():
            if isinstance(bounds, list):
                if d.get(name) not in bounds:
                    d[name] = bounds[0]
            else:
                lo, hi = bounds
                d[name] = max(lo, min(hi, float(d.get(name, lo))))
        return d
    except Exception as exc:
        print(f"  [ParameterInitializer] warm-start skipped: {exc}", file=sys.stderr)
        return None


# ── tuning loop ───────────────────────────────────────────────────────────────
def run_tuner(
    phase1_trials: int            = 60,
    phase2_trials: int            = 140,
    n_trials:      int            = 0,
    sample:        int            = 500,
    study_name:    str            = "cerebrum-tuner",
    validate:      int            = 0,
    seed:          int            = 42,
    timeout_s:     int            = 600,
    log_file:      Optional[Path] = None,
    top_k:         int            = 10,
    margin:        float          = 0.25,
    workers:       int            = 1,
    resume_file:   Optional[Path] = None,
    param_init:    bool           = False,
    kb_file:       Optional[Path] = None,
    dataset:       str            = "metaqa",
) -> Optional[TrialRecord]:
    """
    Two-phase hyperparameter search.

    Phase 1 — Exploration  : Sobol QMCSampler over PARAM_SPACE_WIDE. Low-discrepancy
                             sequences guarantee uniform 9D coverage without
                             accidental clustering (unlike pure random sampling).

    Phase 2 — Fine-tuning  : CmaEsSampler over bounds auto-derived from the top-K
                             Phase 1 trials (±margin). Initialized at the best Phase 1
                             config (x0) and warmed with all Phase 1 trials so CMA-ES
                             starts with a calibrated covariance estimate. Models
                             parameter correlations that TPE treats as independent.

    Pass n_trials > 0 to run single-phase Sobol-only (backward-compatible).
    """
    if not _HAS_OPTUNA:
        print(
            "optuna is not installed.\n"
            "Install with:  pip install 'cerebrum-kg[tuning]'  or  pip install optuna rich",
            file=sys.stderr,
        )
        return None

    # Single-phase fallback (--n-trials)
    if n_trials > 0:
        phase1_trials = n_trials
        phase2_trials = 0

    # Dataset-specific parameter space
    _param_space = PARAM_SPACE_HETIONET if dataset == "hetionet" else PARAM_SPACE_WIDE

    # ── resume: skip Phase 1, load from previous JSONL ───────────────────
    _resume_records:      list[TrialRecord] = []
    _resume_refined_space: Optional[dict]   = None
    if resume_file is not None:
        resume_file = Path(resume_file)
        if not resume_file.exists():
            print(f"ERROR: --resume file not found: {resume_file}", file=sys.stderr)
            return None
        _resume_records, _resume_refined_space = _load_resume(resume_file)
        if not _resume_records:
            print(f"ERROR: no Phase 1 trial records found in {resume_file}", file=sys.stderr)
            return None
        phase1_trials = len(_resume_records)
        print(f"  Resuming  : {resume_file.name}  ({phase1_trials} Phase 1 trials loaded)")

    total_trials = phase1_trials + phase2_trials

    # ── log file setup ────────────────────────────────────────────────────
    run_id   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if log_file is None:
        log_file = Path(__file__).parent / f"tuner_{run_id}.jsonl"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _write_log(log_file, {
        "type": "run_start", "run_id": run_id, "study_name": study_name,
        "phase1_trials": phase1_trials, "phase2_trials": phase2_trials,
        "sample": sample, "seed": seed,
        "sampler_p1": "sobol_qmc",
        "sampler_p2": "cmaes",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "param_space_wide": {
            k: list(v) if isinstance(v, list) else list(v)
            for k, v in _param_space.items()
        },
    })
    print(f"  Trial log : {log_file}")
    if resume_file:
        print(f"  Phase 1   : {phase1_trials} trials (loaded from resume — skipping)")
    else:
        print(f"  Phase 1   : {phase1_trials} trials (Sobol QMC, wide bounds)")
    if phase2_trials:
        print(f"  Phase 2   : {phase2_trials} trials (CMA-ES, refined bounds from top-{top_k})")

    console   = Console() if _HAS_RICH else None
    dashboard = LiveDashboard(total_trials, sample, study_name)

    # ── shared objective factory ──────────────────────────────────────────
    def make_objective(space: dict, phase: int):
        offset = len(dashboard.records)

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial, space)
            tid    = offset + trial.number
            h1, h10, mrr, elapsed = _run_eval_logged(
                log_file=log_file, run_id=run_id, trial_id=tid,
                sample=sample, timeout=timeout_s, workers=workers,
                dataset=dataset, **params,
            )
            rec = TrialRecord(
                trial_id=tid, phase=phase,
                h1=h1, h10=h10, mrr=mrr, elapsed_s=elapsed,
                **params,
            )
            dashboard.push(rec)
            _write_log(log_file, _trial_record_to_dict(rec, sample, run_id, dashboard.records))
            return h1

        return objective

    # ── run both phases under a single Live context ───────────────────────
    def _run_phase(study, objective_fn, n):
        for _ in range(n):
            study.optimize(objective_fn, n_trials=1, catch=(Exception,))

    def _run_phase_plain(study, objective_fn, n):
        for _ in range(n):
            study.optimize(objective_fn, n_trials=1, catch=(Exception,))
            if dashboard.records:
                r    = dashboard.records[-1]
                flag = "★" if r.is_best else " "
                print(
                    f"P{r.phase} {r.trial_id:>4}  {r.trb_factor:>5.2f}  {r.r2_boost:>5.2f}  "
                    f"{r.vote_weight:>6.3f}  {r.beam_width:>3}  "
                    f"{r.idf_weight:>5.2f}  {r.branch_bonus:>5.2f}  {r.fhrb_factor:>5.2f}  "
                    f"{r.gamma:>6.3f}  {r.beta:>5.3f}  "
                    f"{r.h1*100:>6.2f}%  {r.h10*100:>6.2f}%  {r.mrr:>6.4f}  {flag}"
                )

    # ── Phase 1 — Sobol QMC (or reload from resume) ──────────────────────
    p1_sampler = optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed, scramble=True)
    p1_study   = optuna.create_study(
        study_name=f"{study_name}-p1", direction="maximize", sampler=p1_sampler,
    )

    if param_init and not _resume_records:
        _kb = kb_file or _DEFAULT_KB
        x0_hint = _compute_param_init_x0(_kb)
        if x0_hint is not None:
            p1_study.enqueue_trial(x0_hint)
            print(f"  Param-init: warm-start trial enqueued from {_kb.name}")

    if _resume_records:
        # Pre-populate dashboard with loaded Phase 1 data; skip running Phase 1
        for rec in _resume_records:
            dashboard.push(rec)
        dashboard.phase_label = "Phase 2/2 — Fine-tuning (CMA-ES) [resumed]"
    else:
        p1_obj = make_objective(_param_space, phase=1)

    # ── helper: build Phase 2 study ───────────────────────────────────────
    def _build_p2_study(p1_records: list, p1_source_trials) -> tuple:
        """Return (refined_space, p2_study) from Phase 1 data."""
        if _resume_records and _resume_refined_space:
            rs = _resume_refined_space
        else:
            rs = _compute_refined_space(p1_records, _param_space, top_k=top_k, margin=margin)
            _write_log(log_file, {
                "type": "phase2_bounds", "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "top_k": top_k, "margin": margin,
                "refined_space": {
                    k: list(v) if isinstance(v, list) else list(v)
                    for k, v in rs.items()
                },
            })

        if p1_source_trials:
            # source_trials initializes x0/sigma0 from data — cannot specify both
            sampler = optuna.samplers.CmaEsSampler(
                seed=seed + 1,
                source_trials=p1_source_trials,
            )
        else:
            best_p1 = sorted(p1_records, key=lambda r: r.h1, reverse=True)[0]
            x0 = {
                name: max(bounds[0], min(bounds[1], float(getattr(best_p1, name))))
                for name, bounds in rs.items()
                if not isinstance(bounds, list)
            }
            sampler = optuna.samplers.CmaEsSampler(
                x0=x0,
                sigma0=0.3,
                seed=seed + 1,
            )
        study = optuna.create_study(
            study_name=f"{study_name}-p2", direction="maximize", sampler=sampler,
        )
        top_seeds = sorted(p1_records, key=lambda r: r.h1, reverse=True)[:top_k]
        for seed_rec in top_seeds:
            clamped: dict = {}
            for name, bounds in rs.items():
                val = getattr(seed_rec, name)
                clamped[name] = val if isinstance(bounds, list) else max(bounds[0], min(bounds[1], val))
            study.enqueue_trial(clamped)
        return rs, study

    if _HAS_RICH and console:
        with Live(dashboard, console=console, refresh_per_second=2, screen=False):
            if not _resume_records:
                _run_phase(p1_study, p1_obj, phase1_trials)

            if phase2_trials > 0:
                dashboard.phase_label = "Phase 2/2 — Fine-tuning (CMA-ES)"
                p1_source = (
                    _make_source_trials(_resume_records, _param_space)
                    if _resume_records else p1_study.trials
                )
                refined_space, p2_study = _build_p2_study(dashboard.records, p1_source)
                p2_obj = make_objective(refined_space, phase=2)
                _run_phase(p2_study, p2_obj, phase2_trials)
                final_study = p2_study
            else:
                final_study = p1_study
    else:
        # Plain-text fallback
        if not _resume_records:
            header = (
                f"{'Ph':>3}  {'#':>4}  {'trb':>5}  {'r2':>5}  {'vote':>6}  {'bm':>3}  "
                f"{'idf':>5}  {'bbns':>5}  {'fhrb':>5}  {'gamma':>6}  {'beta':>5}  "
                f"{'H@1':>7}  {'H@10':>7}  {'MRR':>6}  B"
            )
            print(header)
            print("-" * len(header))
            _run_phase_plain(p1_study, p1_obj, phase1_trials)

        if phase2_trials > 0:
            print(f"\n--- Phase 2: fine-tuning (CMA-ES, refined bounds) ---")
            p1_source = (
                _make_source_trials(_resume_records, _param_space)
                if _resume_records else p1_study.trials
            )
            refined_space, p2_study = _build_p2_study(dashboard.records, p1_source)
            p2_obj = make_objective(refined_space, phase=2)
            _run_phase_plain(p2_study, p2_obj, phase2_trials)
            final_study = p2_study
        else:
            final_study = p1_study

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
    _print(f"  gamma={best.gamma:.4f}  beta={best.beta:.4f}")
    _param_flags = (
        f"--beam-width {best.beam_width} "
        f"--trb-factor {best.trb_factor:.3f} --r2-boost {best.r2_boost:.3f} "
        f"--vote-weight {best.vote_weight:.4f} --idf-weight {best.idf_weight:.3f} "
        f"--branch-bonus {best.branch_bonus:.3f} --fhrb-factor {best.fhrb_factor:.3f} "
        f"--gamma {best.gamma:.4f} --beta {best.beta:.4f}"
    )
    if dataset == "hetionet":
        canonical = (
            f"python -u benchmarks/hetionet_param_eval.py "
            f"--n-questions 200 --min-eval-hop 1 --max-neighbors 200 --workers 8 "
            f"--embeddings sentence {_param_flags}"
        )
    else:
        canonical = (
            f"python -u benchmarks/metaqa_eval.py --hop 3 --embeddings sentence "
            f"{_param_flags}"
        )
    _print(f"\nCanonical benchmark command:\n  {canonical}\n")
    _print(f"  Trial log: {log_file}")

    best_dict = _trial_record_to_dict(best, sample, run_id, dashboard.records)
    best_dict["type"]         = "best_config"
    best_dict["canonical_cmd"] = canonical
    _write_log(log_file, best_dict)

    # fANOVA on whichever phase produced the best result (or final study)
    try:
        importances = optuna.importance.get_param_importances(final_study)
        _write_log(log_file, {
            "type":        "param_importances",
            "run_id":      run_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method":      "fanova",
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
        _print(f"Validating best config on {validate:,} questions (--workers {workers})...")
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
                gamma=best.gamma,
                beta=best.beta,
                timeout=3600,
                workers=workers,
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
        description="CEREBRUM live hyperparameter tuner — two-phase exploration + fine-tuning",
    )
    parser.add_argument(
        "--phase1-trials", type=int, default=60, dest="phase1_trials",
        help="Exploration trials (RandomSampler, wide bounds). Default: 60",
    )
    parser.add_argument(
        "--phase2-trials", type=int, default=140, dest="phase2_trials",
        help="Fine-tuning trials (TPE, refined bounds). Default: 140",
    )
    parser.add_argument(
        "--n-trials", type=int, default=0, dest="n_trials",
        help="Single-phase mode: run N TPE trials (overrides --phase1/2-trials). "
             "Legacy / backward-compatible.",
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Questions per trial (default 500 ~30s; use 2000 for stability)",
    )
    parser.add_argument(
        "--validate", type=int, default=0,
        help="After tuning, run best config on N questions (0=skip, 14274=full dataset)",
    )
    parser.add_argument(
        "--study-name", type=str, default="cerebrum-tuner", dest="study_name",
        help="Label shown in dashboard header",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k", type=int, default=10, dest="top_k",
        help="Top-K Phase-1 trials used to derive Phase-2 bounds (default 10)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.25,
        help="Fractional margin added around top-K range for Phase-2 bounds (default 0.25)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes per eval trial (default 1; try 4-8 on multi-core CPUs). "
             "Higher values speed up each trial but use more RAM.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, dest="log_file",
        help="JSONL file to log every trial (default: benchmarks/tuner_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, dest="resume_file",
        metavar="JSONL",
        help="Resume from a previous run: load Phase 1 trials from this JSONL and skip "
             "straight to Phase 2.  --phase1-trials is ignored when this is set.",
    )
    parser.add_argument(
        "--param-init", action="store_true", dest="param_init",
        help="(Phase 205) Enqueue a ParameterInitializer warm-start trial as the first "
             "Phase 1 evaluation.  Biases Sobol toward the analytically-derived optimum.",
    )
    parser.add_argument(
        "--kb-file", type=Path, default=None, dest="kb_file",
        help="KB triples file for --param-init (default: benchmarks/data/metaqa/kb.txt).",
    )
    parser.add_argument(
        "--dataset", choices=["metaqa", "hetionet"], default="metaqa", dest="dataset",
        help="(Phase 206) KB to tune against. 'hetionet' uses hetionet_param_eval.py with "
             "PARAM_SPACE_HETIONET. Default: metaqa.",
    )
    args = parser.parse_args()

    run_tuner(
        phase1_trials=args.phase1_trials,
        phase2_trials=args.phase2_trials,
        n_trials=args.n_trials,
        sample=args.sample,
        study_name=args.study_name,
        validate=args.validate,
        seed=args.seed,
        top_k=args.top_k,
        margin=args.margin,
        log_file=args.log_file,
        workers=args.workers,
        resume_file=args.resume_file,
        param_init=args.param_init,
        kb_file=args.kb_file,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()

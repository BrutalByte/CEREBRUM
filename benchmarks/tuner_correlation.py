"""
CEREBRUM Tuner Correlation Analysis
=====================================
Loads all tuner JSONL logs, filters to the top-N% trials by H@1, and computes:

  1. Pearson correlation matrix across all 11 hyperparameters
  2. Per-param correlation with H@1 (predictive power ranking)
  3. Strongly coupled pairs (|r| >= threshold)
  4. Heatmap PNG saved alongside this script

Usage
-----
    python benchmarks/tuner_correlation.py
    python benchmarks/tuner_correlation.py --top-pct 20 --threshold 0.5
    python benchmarks/tuner_correlation.py --log benchmarks/tuner_20260526T234630.jsonl
    python benchmarks/tuner_correlation.py --all-trials          # use every trial, not just top-%
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_PARAMS = [
    "trb_factor", "r2_boost", "vote_weight", "beam_width", "idf_weight",
    "branch_bonus", "fhrb_factor", "wb_r2_boost", "db_r2_boost",
    "ry_r2_boost", "sa_r2_boost",
]
_SHORT = {
    "trb_factor":   "trb",
    "r2_boost":     "r2",
    "vote_weight":  "vote",
    "beam_width":   "beam",
    "idf_weight":   "idf",
    "branch_bonus": "bbns",
    "fhrb_factor":  "fhrb",
    "wb_r2_boost":  "wb",
    "db_r2_boost":  "db",
    "ry_r2_boost":  "ry",
    "sa_r2_boost":  "sa",
}

_HERE = Path(__file__).parent


# ── data loading ──────────────────────────────────────────────────────────────
def _load_logs(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "trial":
                    continue
                if not all(p in rec for p in _PARAMS):
                    continue
                rows.append({
                    "log":    path.stem,
                    "run_id": rec.get("run_id", ""),
                    "phase":  rec.get("phase", 1),
                    **{p: rec[p] for p in _PARAMS},
                    "h1":     rec["h1"],
                    "h10":    rec.get("h10", float("nan")),
                    "mrr":    rec.get("mrr", float("nan")),
                })
    return pd.DataFrame(rows)


# ── analysis ──────────────────────────────────────────────────────────────────
def _correlation_with_h1(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for p in _PARAMS:
        r, pval = stats.pearsonr(df[p], df["h1"])
        results.append({"param": p, "r_with_h1": round(r, 4), "p_value": round(pval, 4)})
    return pd.DataFrame(results).sort_values("r_with_h1", ascending=False)


def _pairwise_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df[_PARAMS].corr(method="pearson")


def _coupled_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple]:
    pairs = []
    for i, p1 in enumerate(_PARAMS):
        for p2 in _PARAMS[i + 1:]:
            r = corr.loc[p1, p2]
            if abs(r) >= threshold:
                pairs.append((p1, p2, round(r, 4)))
    return sorted(pairs, key=lambda x: -abs(x[2]))


# ── display ───────────────────────────────────────────────────────────────────
def _print_h1_correlations(h1_corr: pd.DataFrame, console) -> None:
    _p = console.print if console else print
    if console:
        t = Table(
            title="[bold]Param vs H@1 Correlation[/bold]  (Pearson r)",
            box=rich_box.SIMPLE_HEAD, show_header=True,
            header_style="bold white", show_edge=False,
        )
        t.add_column("Param",   width=16)
        t.add_column("r",       justify="right", width=7)
        t.add_column("p-val",   justify="right", width=8)
        t.add_column("Signal",  width=30)
        for _, row in h1_corr.iterrows():
            r   = row["r_with_h1"]
            bar_len = int(abs(r) * 25)
            bar = ("+" if r >= 0 else "-") * bar_len
            color = "green" if r > 0 else "red"
            sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
            t.add_row(
                row["param"],
                f"{r:+.4f}",
                f"{row['p_value']:.4f}{sig}",
                f"[{color}]{bar}[/{color}]",
            )
        console.print(t)
    else:
        print(f"\n{'Param':<16}  {'r':>7}  {'p-val':>8}  Signal")
        print("-" * 55)
        for _, row in h1_corr.iterrows():
            r   = row["r_with_h1"]
            bar = ("+" if r >= 0 else "-") * int(abs(r) * 25)
            print(f"{row['param']:<16}  {r:+7.4f}  {row['p_value']:8.4f}  {bar}")


def _print_coupled_pairs(pairs: list[tuple], threshold: float, console) -> None:
    _p = console.print if console else print
    _p(f"\n[bold]Strongly coupled pairs[/bold]  (|r| >= {threshold})" if console
       else f"\nStrongly coupled pairs  (|r| >= {threshold})")
    if not pairs:
        _p("  [dim]None found.[/dim]" if console else "  None found.")
        return
    if console:
        t = Table(box=rich_box.SIMPLE_HEAD, show_header=True,
                  header_style="bold white", show_edge=False)
        t.add_column("Param A",   width=16)
        t.add_column("Param B",   width=16)
        t.add_column("r",         justify="right", width=7)
        t.add_column("Direction", width=12)
        for p1, p2, r in pairs:
            direction = "[green]co-move[/green]" if r > 0 else "[red]inverse[/red]"
            t.add_row(p1, p2, f"{r:+.4f}", direction)
        console.print(t)
    else:
        print(f"  {'Param A':<16}  {'Param B':<16}  {'r':>7}  Direction")
        print("  " + "-" * 55)
        for p1, p2, r in pairs:
            direction = "co-move" if r > 0 else "inverse"
            print(f"  {p1:<16}  {p2:<16}  {r:+7.4f}  {direction}")


def _print_corr_matrix(corr: pd.DataFrame, console) -> None:
    shorts = [_SHORT[p] for p in _PARAMS]
    if console:
        t = Table(
            title="[bold]Pairwise Correlation Matrix[/bold]",
            box=rich_box.SIMPLE_HEAD, show_header=True,
            header_style="bold white", show_edge=False,
        )
        t.add_column("", width=6)
        for s in shorts:
            t.add_column(s, justify="right", width=7)
        for p, s_row in zip(_PARAMS, shorts):
            cells = []
            for p2 in _PARAMS:
                r = corr.loc[p, p2]
                if p == p2:
                    cells.append("[dim] +1.0[/dim]")
                elif abs(r) >= 0.6:
                    color = "green" if r > 0 else "red"
                    cells.append(f"[bold {color}]{r:+.3f}[/bold {color}]")
                elif abs(r) >= 0.4:
                    color = "yellow"
                    cells.append(f"[{color}]{r:+.3f}[/{color}]")
                else:
                    cells.append(f"[dim]{r:+.3f}[/dim]")
            t.add_row(s_row, *cells)
        console.print(t)
    else:
        header = f"{'':6}" + "".join(f"{s:>6}" for s in shorts)
        print(f"\n{header}")
        for p, s_row in zip(_PARAMS, shorts):
            row_str = f"{s_row:6}"
            for p2 in _PARAMS:
                row_str += f"{corr.loc[p, p2]:+6.2f}"
            print(row_str)


# ── heatmap ───────────────────────────────────────────────────────────────────
def _save_heatmap(corr: pd.DataFrame, out_path: Path, n_trials: int, top_pct: int) -> None:
    shorts = [_SHORT[p] for p in _PARAMS]
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rw_g", ["#d62728", "#ffffff", "#2ca02c"]
    )
    data = corr.values
    im   = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shorts)))
    ax.set_yticks(range(len(shorts)))
    ax.set_xticklabels(shorts, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(shorts, fontsize=11)
    for i in range(len(_PARAMS)):
        for j in range(len(_PARAMS)):
            r = data[i, j]
            color = "white" if abs(r) > 0.55 else "black"
            ax.text(j, i, f"{r:+.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold" if abs(r) >= 0.6 else "normal")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    ax.set_title(
        f"CEREBRUM Hyperparameter Correlation Matrix\n"
        f"n={n_trials} trials (top {top_pct}% by H@1)",
        fontsize=13, pad=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tuner_correlation",
        description="Pairwise correlation analysis of CEREBRUM tuner trial logs",
    )
    parser.add_argument(
        "--log", type=Path, action="append", dest="logs",
        help="Specific JSONL log(s) to load. Default: all benchmarks/tuner_*.jsonl",
    )
    parser.add_argument(
        "--top-pct", type=int, default=30, dest="top_pct",
        help="Use only the top N%% of trials by H@1 (default 30)",
    )
    parser.add_argument(
        "--all-trials", action="store_true", dest="all_trials",
        help="Use every trial, not just top-%%",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Correlation threshold for 'coupled pairs' report (default 0.6)",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true", dest="no_heatmap",
        help="Skip saving the heatmap PNG",
    )
    args = parser.parse_args()

    # Resolve log files
    if args.logs:
        paths = [p for p in args.logs if p.exists()]
    else:
        paths = sorted(_HERE.glob("tuner_*.jsonl"))
    if not paths:
        print("No JSONL logs found.")
        return

    import sys, io
    console = Console(file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)) if _HAS_RICH else None
    _p      = console.print if console else print

    _p(f"\n[bold]CEREBRUM Tuner Correlation Analysis[/bold]" if console
       else "\nCEREBRUM Tuner Correlation Analysis")
    _p(f"Loading {len(paths)} log file(s)...")

    df = _load_logs(paths)
    if df.empty:
        _p("[red]No trial records found.[/red]" if console else "No trial records found.")
        return

    _p(f"  Total trials loaded : {len(df):,}")

    # Filter to top-%
    if not args.all_trials and args.top_pct < 100:
        cutoff = df["h1"].quantile(1 - args.top_pct / 100)
        df_filt = df[df["h1"] >= cutoff].copy()
        _p(f"  Top {args.top_pct}% cutoff    : H@1 >= {cutoff*100:.2f}%  ({len(df_filt):,} trials)")
    else:
        df_filt = df.copy()
        _p(f"  Using all {len(df_filt):,} trials")

    if len(df_filt) < 10:
        _p("[yellow]Warning: fewer than 10 trials after filtering — correlations may be unreliable.[/yellow]"
           if console else "Warning: fewer than 10 trials after filtering.")

    # Compute
    corr    = _pairwise_corr(df_filt)
    h1_corr = _correlation_with_h1(df_filt)
    pairs   = _coupled_pairs(corr, args.threshold)

    _print_h1_correlations(h1_corr, console)
    _print_coupled_pairs(pairs, args.threshold, console)
    _print_corr_matrix(corr, console)

    # Heatmap
    if _HAS_MPL and not args.no_heatmap:
        top_pct_label = 100 if args.all_trials else args.top_pct
        out_path = _HERE / "tuner_correlation_heatmap.png"
        _save_heatmap(corr, out_path, len(df_filt), top_pct_label)
        _p(f"\n  Heatmap saved: {out_path}")

    # Per-log breakdown
    _p(f"\n[bold]Logs included:[/bold]" if console else "\nLogs included:")
    for log_name, grp in df.groupby("log"):
        best = grp["h1"].max()
        _p(f"  {log_name}  ({len(grp)} trials, best H@1={best*100:.2f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LFR benchmark for CEREBRUM community detection (TSC) vs baselines.

Generates synthetic Lancichinetti–Fortunato–Radicchi graphs and measures
NMI and ARI for TSC, Leiden, Louvain, and Infomap (if available).

Results replace the $^\star$ placeholder values in:
  research/papers/02-community-detection/tsc-paper.tex  (LFR table, lines 217-219)

LFR parameters (matching paper table):
  n=1000, average_degree=10, mu in {0.1, 0.3, 0.5}
  tau1=2 (degree power law), tau2=1.5 (community size power law)
  min_community=20, max_community=50

Usage:
    python scripts/run_lfr_benchmark.py
    python scripts/run_lfr_benchmark.py --n 1000 --mu 0.1 0.3 0.5 --repeats 5
    python scripts/run_lfr_benchmark.py --update-paper
    python scripts/run_lfr_benchmark.py --quick   # n=500 for a fast sanity check
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    _SKLEARN = True
except ImportError:
    print("ERROR: scikit-learn is required. pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TSC_TEX_PATHS = [
    Path(__file__).resolve().parent.parent / "research" / "papers" / "02-community-detection" / "tsc-paper.tex",
    Path(__file__).resolve().parent.parent / "research" / "papers" / "02-community-detection" / "arxiv_submission" / "tsc-paper.tex",
]

# ---------------------------------------------------------------------------
# LFR graph generation with retry
# ---------------------------------------------------------------------------

def make_lfr(
    n: int,
    mu: float,
    average_degree: int = 10,
    tau1: float = 2.0,
    tau2: float = 1.5,
    min_community: int = 20,
    max_community: int = 50,
    seed: int = 42,
    max_attempts: int = 20,
) -> Optional[nx.Graph]:
    """Generate an LFR benchmark graph, retrying with different seeds on failure."""
    for attempt in range(max_attempts):
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n              = n,
                tau1           = tau1,
                tau2           = tau2,
                mu             = mu,
                average_degree = average_degree,
                min_community  = min_community,
                max_community  = max_community,
                seed           = seed + attempt,
            )
            return G
        except nx.ExceededMaxIterations:
            continue
        except Exception as e:
            print(f"    LFR error (attempt {attempt+1}): {e}")
            continue
    print(f"  WARN: LFR n={n} mu={mu} failed after {max_attempts} attempts — skipping")
    return None


def extract_ground_truth(G: nx.Graph) -> List[int]:
    """
    Extract per-node community labels from LFR benchmark node attributes.
    LFR stores community membership as frozenset; for non-overlapping
    communities each node is in exactly one community.
    """
    nodes = list(G.nodes())
    labels = []
    for node in nodes:
        comm = G.nodes[node].get("community", frozenset())
        # Non-overlapping: frozenset has exactly one element
        labels.append(next(iter(comm)) if comm else -1)
    return labels


def partition_to_labels(G: nx.Graph, partition: List[frozenset]) -> List[int]:
    """Convert a list-of-frozensets partition to a per-node label array."""
    node_list = list(G.nodes())
    node_to_label = {}
    for cid, community in enumerate(partition):
        for node in community:
            node_to_label[node] = cid
    return [node_to_label.get(n, -1) for n in node_list]


# ---------------------------------------------------------------------------
# Community detection algorithms
# ---------------------------------------------------------------------------

def run_tsc(G: nx.Graph) -> List[frozenset]:
    from core.community_engine import tsc_communities
    return tsc_communities(G)


def run_leiden(G: nx.Graph) -> List[frozenset]:
    from core.community_engine import leiden_communities
    return leiden_communities(G)


def run_louvain(G: nx.Graph) -> List[frozenset]:
    return list(nx.algorithms.community.louvain_communities(G, seed=42))


def run_infomap(G: nx.Graph) -> Optional[List[frozenset]]:
    try:
        import cdlib
        result = cdlib.algorithms.infomap(G)
        return [frozenset(c) for c in result.communities]
    except ImportError:
        return None
    except Exception as e:
        print(f"    Infomap error: {e}")
        return None


ALGORITHMS = {
    "Louvain": run_louvain,
    "Leiden":  run_leiden,
    "Infomap": run_infomap,
    "TSC":     run_tsc,
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_one(
    G: nx.Graph,
    algorithm: str,
    fn,
) -> Optional[Dict[str, float]]:
    """Run one algorithm on one graph and return NMI + ARI."""
    ground_truth = extract_ground_truth(G)

    t0 = time.time()
    try:
        partition = fn(G)
    except Exception as e:
        print(f"    {algorithm}: FAILED ({e})")
        return None

    if partition is None:
        return None  # algorithm not available

    predicted = partition_to_labels(G, partition)
    elapsed = time.time() - t0

    nmi = normalized_mutual_info_score(ground_truth, predicted, average_method="arithmetic")
    ari = adjusted_rand_score(ground_truth, predicted)

    return {"nmi": nmi, "ari": ari, "elapsed_s": elapsed, "n_communities": len(partition)}


def run_benchmark(
    n: int,
    mu_values: List[float],
    average_degree: int,
    repeats: int,
) -> Dict:
    """
    Returns nested dict: results[algorithm][mu] = {"nmi": float, "ari": float}
    """
    results: Dict[str, Dict[float, Dict]] = {alg: {} for alg in ALGORITHMS}

    for mu in mu_values:
        print(f"\n  mu={mu}:")
        nmi_lists: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
        ari_lists: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}

        for rep in range(repeats):
            G = make_lfr(n=n, mu=mu, average_degree=average_degree, seed=42 + rep * 7)
            if G is None:
                continue

            for alg, fn in ALGORITHMS.items():
                r = benchmark_one(G, alg, fn)
                if r is None:
                    continue
                nmi_lists[alg].append(r["nmi"])
                ari_lists[alg].append(r["ari"])
                print(f"    {alg:<10} rep {rep+1}: NMI={r['nmi']:.3f}  ARI={r['ari']:.3f}  "
                      f"({r['n_communities']} communities, {r['elapsed_s']:.1f}s)")

        for alg in ALGORITHMS:
            if nmi_lists[alg]:
                results[alg][mu] = {
                    "nmi": float(np.mean(nmi_lists[alg])),
                    "ari": float(np.mean(ari_lists[alg])),
                    "nmi_std": float(np.std(nmi_lists[alg])),
                    "ari_std": float(np.std(ari_lists[alg])),
                    "n_reps": len(nmi_lists[alg]),
                }
            else:
                results[alg][mu] = None

    return results


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results: Dict, mu_values: List[float]) -> None:
    print(f"\n{'='*70}")
    print(f"LFR Benchmark Results (averaged over repeats)")
    print(f"{'='*70}")
    header = f"  {'Algorithm':<12}" + "".join(
        f"  mu={mu:.1f} NMI  ARI" for mu in mu_values
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for alg in ALGORITHMS:
        row = f"  {alg:<12}"
        for mu in mu_values:
            r = results[alg].get(mu)
            if r is None:
                row += "  (unavailable)"
            else:
                row += f"  {r['nmi']:.3f}  {r['ari']:.3f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Paper update
# ---------------------------------------------------------------------------

def update_paper(results: Dict, mu_values: List[float]) -> None:
    """Replace TSC $^\star$ values in tsc-paper.tex."""
    tsc = results.get("TSC", {})

    # Build replacement values
    def nmi(mu: float) -> str:
        r = tsc.get(mu)
        return f"{r['nmi']:.2f}" if r else "???"

    def ari(mu: float) -> str:
        r = tsc.get(mu)
        return f"{r['ari']:.2f}" if r else "???"

    # The TSC table rows look like:
    # \textbf{TSC}       & \textbf{0.99}$^\star$ & \textbf{0.98}$^\star$
    #                    & \textbf{0.91}$^\star$ & \textbf{0.89}$^\star$
    #                    & \textbf{0.72}$^\star$ \\

    mu01 = mu_values[0] if len(mu_values) > 0 else 0.1
    mu03 = mu_values[1] if len(mu_values) > 1 else 0.3
    mu05 = mu_values[2] if len(mu_values) > 2 else 0.5

    new_tsc_row = (
        f"\\textbf{{TSC}}       & \\textbf{{{nmi(mu01)}}} & \\textbf{{{ari(mu01)}}}\n"
        f"                   & \\textbf{{{nmi(mu03)}}} & \\textbf{{{ari(mu03)}}}\n"
        f"                   & \\textbf{{{nmi(mu05)}}} \\\\"
    )

    old_pattern = re.compile(
        r"\\textbf\{TSC\}\s+&\s+\\textbf\{[^}]+\}\$\^\*\$\s+&\s+\\textbf\{[^}]+\}\$\^\*\$\s*\n"
        r"\s+&\s+\\textbf\{[^}]+\}\$\^\*\$\s+&\s+\\textbf\{[^}]+\}\$\^\*\$\s*\n"
        r"\s+&\s+\\textbf\{[^}]+\}\$\^\*\$\s+\\\\",
        re.MULTILINE,
    )

    for tex_path in TSC_TEX_PATHS:
        if not tex_path.exists():
            print(f"  SKIP {tex_path} (not found)")
            continue
        text = tex_path.read_text(encoding="utf-8")
        new_text, n = old_pattern.subn(new_tsc_row, text)
        if n == 0:
            print(f"  WARN: TSC row pattern not matched in {tex_path.name} — manual update needed")
            print("  Replacement row:")
            print(new_tsc_row)
        else:
            # Remove action note
            new_text = re.sub(
                r"\\noindent\\emph\{Action: run.*?\}",
                f"\\\\noindent\\\\emph{{LFR results: n={results.get('_n', 1000)}, "
                r"avg\\_degree=10, averaged over repeats.}}",
                new_text,
            )
            tex_path.write_text(new_text, encoding="utf-8")
            print(f"  UPDATED {tex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of nodes (default: 1000)")
    parser.add_argument("--mu", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="Mixing parameters (default: 0.1 0.3 0.5)")
    parser.add_argument("--average-degree", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Repetitions per (algorithm, mu) — averaged (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: n=500, 2 repeats, mu=0.1 0.3 only")
    parser.add_argument("--update-paper", action="store_true",
                        help="Write TSC values into tsc-paper.tex files")
    args = parser.parse_args()

    if args.quick:
        args.n = 500
        args.repeats = 2
        args.mu = [0.1, 0.3]

    print(f"=== CEREBRUM LFR Benchmark ===\n")
    print(f"  n={args.n}  average_degree={args.average_degree}")
    print(f"  mu={args.mu}  repeats={args.repeats}")

    results = run_benchmark(
        n              = args.n,
        mu_values      = args.mu,
        average_degree = args.average_degree,
        repeats        = args.repeats,
    )
    results["_n"] = args.n

    print_table(results, args.mu)

    if args.update_paper:
        if args.quick:
            print("WARN: --quick mode uses reduced n/repeats — values may not match paper params.")
        print("Updating paper files...")
        update_paper(results, args.mu)
        print("Done. Re-run publication_preflight.py to verify.")
    else:
        print("(Re-run with --update-paper to write TSC values into paper files)")


if __name__ == "__main__":
    main()

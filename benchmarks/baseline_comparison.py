"""
Ablation study: DSCF vs LPA vs BFS-only for MetaQA (Phase 4).

This module implements the ablation plan from Section 9.2 of PARALLAX.md.
Three variants are evaluated on the same MetaQA test questions:

  Variant A — Parallax (DSCF + CSA)          [the full system]
  Variant B — Parallax (LPA  + CSA)           [swap community detection only]
  Variant C — BFS baseline (uniform weights)  [no attention, no communities]

All three variants use the same:
  - KB graph (undirected MetaQA)
  - Beam width and max_hop settings
  - Random embeddings (so semantic signal is identical / zero)
  - Same test questions from the same hop level

The purpose is to isolate the contribution of:
  - DSCF vs LPA as the attention-head source (A vs B)
  - CSA attention vs no attention at all (A vs C)

Usage
-----
  python -m benchmarks.baseline_comparison --hop 1
  python -m benchmarks.baseline_comparison --hop 3 --sample 500
  python -m benchmarks.baseline_comparison          # all hops
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf, lpa_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

from benchmarks.metaqa_eval import (
    load_kb, load_qa, load_or_compute_communities,
    hits_at_k, reciprocal_rank, evaluate_hop,
)

DATA_DIR  = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"


# ---------------------------------------------------------------------------
# BFS CSA engine — uniform attention weights
# ---------------------------------------------------------------------------

class UniformCSAEngine(CSAEngine):
    """
    CSA engine that returns a constant weight for every edge.

    This makes BeamTraversal equivalent to BFS — no preference is given
    to any edge over any other. Used as the 'no-attention' ablation baseline.

    Weight = sigmoid(0) = 0.5 for all edges (all coefficients zeroed out).
    """

    def compute_weight(self, u, v, hop, edge_type="", edge_type_weights=None,
                       normalized_distance=0.0) -> float:
        return 0.5   # sigmoid(0) — perfectly neutral


# ---------------------------------------------------------------------------
# Build traversal for each variant
# ---------------------------------------------------------------------------

def build_dscf_traversal(
    adapter, G, embeddings, beam_width, max_hop, dscf_seed=42, use_cache=True
) -> BeamTraversal:
    """Variant A: full DSCF + CSA."""
    cmap = load_or_compute_communities(G, use_cache=use_cache, dscf_seed=dscf_seed)
    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    csa  = CSAEngine(communities=cmap, embeddings=embeddings)
    csa.set_community_graph(dist, adj)
    return BeamTraversal(adapter=adapter, csa_engine=csa, embeddings=embeddings,
                         communities=cmap, beam_width=beam_width, max_hop=max_hop), cmap


def build_lpa_traversal(
    adapter, G, embeddings, beam_width, max_hop
) -> BeamTraversal:
    """Variant B: LPA communities + CSA."""
    print("  Computing LPA communities...")
    t0    = time.time()
    parts = lpa_communities(G)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}
    print(f"  LPA: {len(parts)} communities in {time.time()-t0:.1f}s")
    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    csa  = CSAEngine(communities=cmap, embeddings=embeddings)
    csa.set_community_graph(dist, adj)
    return BeamTraversal(adapter=adapter, csa_engine=csa, embeddings=embeddings,
                         communities=cmap, beam_width=beam_width, max_hop=max_hop), cmap


def build_bfs_traversal(
    adapter, G, embeddings, beam_width, max_hop
) -> BeamTraversal:
    """Variant C: BFS — no community structure, uniform weights."""
    # Community map assigns all nodes to community 0 (irrelevant since weights are uniform)
    cmap = {node: 0 for node in G.nodes()}
    csa  = UniformCSAEngine(communities=cmap, embeddings=embeddings)
    return BeamTraversal(adapter=adapter, csa_engine=csa, embeddings=embeddings,
                         communities=cmap, beam_width=beam_width, max_hop=max_hop), cmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallax ablation study on MetaQA")
    parser.add_argument("--hop",        type=int, default=None,
                        help="Hop level to evaluate (1, 2, or 3). Default: all.")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Random sample of N questions per hop.")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--top-k",      type=int, default=10)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Recompute DSCF even if cache exists.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    hops = [args.hop] if args.hop else [1, 2, 3]

    print("\n=== Parallax — MetaQA Ablation Study ===\n")
    print("Variants:")
    print("  A  Parallax  DSCF + CSA   (full system)")
    print("  B  Parallax  LPA  + CSA   (LPA communities only)")
    print("  C  BFS       uniform      (no attention baseline)")
    print()

    print("Loading knowledge graph...")
    adapter = load_kb(undirected=True)
    G       = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} entities, {G.number_of_edges():,} edges")

    print("Building embeddings...")
    random.seed(args.seed)
    engine     = RandomEngine(dim=64)
    labels     = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    all_results = []

    for hop in hops:
        print(f"\n=== {hop}-hop ===")
        qa_pairs = load_qa(hop, sample=args.sample, seed=args.seed)
        n_label  = f"{len(qa_pairs):,}" + (" (sample)" if args.sample else "")
        print(f"  {n_label} test questions")

        row = {"hop": hop, "n": len(qa_pairs)}

        # Variant A — DSCF + CSA
        print("\n  [A] DSCF + CSA...")
        t_dscf, cmap_dscf = build_dscf_traversal(
            adapter, G, embeddings, args.beam_width, hop,
            dscf_seed=args.seed, use_cache=not args.no_cache
        )
        m_a = evaluate_hop(hop, t_dscf, qa_pairs, top_k=args.top_k)
        print(f"      Hits@1={m_a['hits_1']:.4f}  Hits@10={m_a['hits_10']:.4f}  MRR={m_a['mrr']:.4f}")
        row.update({"dscf_h1": m_a["hits_1"], "dscf_h10": m_a["hits_10"], "dscf_mrr": m_a["mrr"]})

        # Variant B — LPA + CSA
        print("\n  [B] LPA + CSA...")
        t_lpa, _ = build_lpa_traversal(adapter, G, embeddings, args.beam_width, hop)
        m_b = evaluate_hop(hop, t_lpa, qa_pairs, top_k=args.top_k)
        print(f"      Hits@1={m_b['hits_1']:.4f}  Hits@10={m_b['hits_10']:.4f}  MRR={m_b['mrr']:.4f}")
        row.update({"lpa_h1": m_b["hits_1"], "lpa_h10": m_b["hits_10"], "lpa_mrr": m_b["mrr"]})

        # Variant C — BFS
        print("\n  [C] BFS (uniform weights)...")
        t_bfs, _ = build_bfs_traversal(adapter, G, embeddings, args.beam_width, hop)
        m_c = evaluate_hop(hop, t_bfs, qa_pairs, top_k=args.top_k)
        print(f"      Hits@1={m_c['hits_1']:.4f}  Hits@10={m_c['hits_10']:.4f}  MRR={m_c['mrr']:.4f}")
        row.update({"bfs_h1": m_c["hits_1"], "bfs_h10": m_c["hits_10"], "bfs_mrr": m_c["mrr"]})

        all_results.append(row)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Ablation Summary ===\n")
    print(f"  {'':6} {'Hits@1':>24}   {'Hits@10':>24}   {'MRR':>24}")
    print(f"  {'Hop':<6} {'DSCF':>8} {'LPA':>8} {'BFS':>8}   "
          f"{'DSCF':>8} {'LPA':>8} {'BFS':>8}   "
          f"{'DSCF':>8} {'LPA':>8} {'BFS':>8}")
    print("  " + "-" * 80)
    for row in all_results:
        print(f"  {row['hop']}-hop  "
              f"{row['dscf_h1']:>8.4f} {row['lpa_h1']:>8.4f} {row['bfs_h1']:>8.4f}   "
              f"{row['dscf_h10']:>8.4f} {row['lpa_h10']:>8.4f} {row['bfs_h10']:>8.4f}   "
              f"{row['dscf_mrr']:>8.4f} {row['lpa_mrr']:>8.4f} {row['bfs_mrr']:>8.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CACHE_DIR / "ablation_results.csv"
    if all_results:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Ablation results saved to {out_file}")
    print()


if __name__ == "__main__":
    main()




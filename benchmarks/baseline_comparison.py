"""
Ablation study: DSCF vs LPA vs BFS-only for MetaQA (Phase 4).

This module implements the ablation plan from Section 9.2 of PARALLAX.md.
Three variants are evaluated on the same MetaQA test questions:

  Variant A — CEREBRUM (DSCF + CSA)          [the full system]
  Variant B — CEREBRUM (LPA  + CSA)           [swap community detection only]
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
from typing import Optional, Dict, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


from core.community_engine import lpa_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import (
    build_community_distance_matrix, adjacent_community_pairs,
    coarsen_communities, build_community_graph,
)

from reasoning.traversal import BeamTraversal
from core.resource_governor import ResourceGovernor

from core.cerebrum import CerebrumGraph

from benchmarks.metaqa_eval import (
    load_qa,
)

_COARSEN_TARGET = 500  # max communities before CSA signal degrades

# Benchmarks measure algorithmic quality, not governor behavior.
# Use a near-unlimited governor (99% threshold) so memory pressure on the
# host machine doesn't silently throttle CEREBRUM to zero answers.
_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)

DATA_DIR  = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"
KB_FILE   = DATA_DIR / "kb.txt"


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

    def compute_weight(
        self,
        u: str,
        v: str,
        hop: int,
        **kwargs: Any
    ) -> float:
        return 0.5   # sigmoid(0) — perfectly neutral


# ---------------------------------------------------------------------------
# Per-hop evaluation (local version for ablation study)
# ---------------------------------------------------------------------------

def evaluate_variant(
    hop:              int,
    traversal:        BeamTraversal,
    qa_pairs:         List[Tuple],
    top_k:            int            = 10,
) -> Dict:
    from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
    from reasoning.answer_extractor import extract
    
    h1 = h10 = 0
    mrr_sum  = 0.0
    skipped  = found = 0

    eval_min_hop = 2 if hop == 2 else 1

    t0 = time.time()
    for i, qa in enumerate(qa_pairs):
        seed, correct_answers = qa

        if (i + 1) % 100 == 0 or (i + 1) == len(qa_pairs):
            print(
                f"    {i+1:,}/{len(qa_pairs):,} questions "
                f"({time.time()-t0:.1f}s elapsed)",
                end="\r",
            )

        paths = traversal.traverse([seed])
        answers_obj = extract(paths, top_k=top_k, min_hop=eval_min_hop)
        pred = [a.entity_id for a in answers_obj]

        if not pred:
            skipped += 1
            continue

        found   += 1
        h1      += hits_at_k(pred, correct_answers, k=1)
        h10     += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    elapsed = time.time() - t0
    print()
    n = len(qa_pairs)

    return {
        "hits_1":     h1  / n if n > 0 else 0,
        "hits_10":    h10 / n if n > 0 else 0,
        "mrr":        mrr_sum / n if n > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CEREBRUM ablation study on MetaQA")
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

    print("\n=== CEREBRUM — MetaQA Ablation Study ===\n")
    print("Variants:")
    print("  A  CEREBRUM  DSCF + CSA   (full system)")
    print("  B  CEREBRUM  LPA  + CSA   (LPA communities only)")
    print("  C  BFS       uniform      (no attention baseline)")
    print()

    if not KB_FILE.exists():
        print(f"ERROR: kb.txt not found at {KB_FILE}")
        sys.exit(1)

    print("Loading knowledge graph...")
    # Load via CerebrumGraph for consistency
    graph = CerebrumGraph.from_kb(
        KB_FILE,
        sep       = "|",
        directed  = False,
        embeddings= "random",
        beam_width= args.beam_width,
        max_hop   = 3,
    )
    adapter = graph.adapter
    G = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} entities, {G.number_of_edges():,} edges")

    all_results = []

    for hop in hops:
        print(f"\n=== {hop}-hop ===")
        qa_pairs = load_qa(hop, sample=args.sample, seed=args.seed)
        n_label  = f"{len(qa_pairs):,}" + (" (sample)" if args.sample else "")
        print(f"  {n_label} test questions")

        row = {"hop": hop, "n": len(qa_pairs)}

        # Variant A — DSCF + CSA
        print("\n  [A] DSCF + CSA...")
        graph.build(
            cache_dir           = CACHE_DIR,
            min_community_size  = 20,
            force_rebuild       = args.no_cache,
            seed                = args.seed,
        )
        # Manually inject benchmark governor
        trav_a = BeamTraversal(
            adapter=graph.adapter,
            csa_engine=graph._csa,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR
        )
        
        m_a = evaluate_variant(hop, trav_a, qa_pairs, top_k=args.top_k)
        print(f"      Hits@1={m_a['hits_1']:.4f}  Hits@10={m_a['hits_10']:.4f}  MRR={m_a['mrr']:.4f}")
        row.update({"dscf_h1": m_a["hits_1"], "dscf_h10": m_a["hits_10"], "dscf_mrr": m_a["mrr"]})

        # Variant B — LPA + CSA
        print("\n  [B] LPA + CSA...")
        print("      Computing LPA communities...")
        parts = lpa_communities(G)
        cmap_lpa = {node: cid for cid, members in enumerate(parts) for node in members}
        if len(set(cmap_lpa.values())) > _COARSEN_TARGET:
            cmap_lpa = coarsen_communities(G, cmap_lpa, target_max=_COARSEN_TARGET)
        
        dist = build_community_distance_matrix(G, cmap_lpa)
        adj  = adjacent_community_pairs(G, cmap_lpa)
        cg   = build_community_graph(G, cmap_lpa)
        
        adapter_lpa = graph.adapter # sharing same adapter instance but switching maps
        adapter_lpa.community_map = cmap_lpa
        
        csa_lpa = CSAEngine(adapter=adapter_lpa)
        csa_lpa.set_community_graph(dist, adj, community_graph=cg)
        
        trav_b = BeamTraversal(
            adapter=adapter_lpa,
            csa_engine=csa_lpa,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR
        )
        
        m_b = evaluate_variant(hop, trav_b, qa_pairs, top_k=args.top_k)
        print(f"      Hits@1={m_b['hits_1']:.4f}  Hits@10={m_b['hits_10']:.4f}  MRR={m_b['mrr']:.4f}")
        row.update({"lpa_h1": m_b["hits_1"], "lpa_h10": m_b["hits_10"], "lpa_mrr": m_b["mrr"]})

        # Variant C — BFS
        print("\n  [C] BFS (uniform weights)...")
        # Reuse adapter but use UniformCSAEngine
        csa_bfs = UniformCSAEngine(adapter=graph.adapter)
        trav_c = BeamTraversal(
            adapter=graph.adapter,
            csa_engine=csa_bfs,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR
        )
        
        m_c = evaluate_variant(hop, trav_c, qa_pairs, top_k=args.top_k)
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
    print("  " + "-" * 90)
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
        import csv as _csv
        with open(out_file, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Ablation results saved to {out_file}")
    print()

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




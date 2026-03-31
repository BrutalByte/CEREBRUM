"""
Synthetic clustered graph benchmark for CEREBRUM (Phase 4).

Directly tests the core CSA hypothesis with controlled ground truth:
  "CSA's community membership scoring boosts intra-community paths."

The graph is a planted-partition model with high modularity, so DSCF
should recover communities nearly perfectly. Questions are generated
to require reasoning WITHIN a community — the regime where CSA is
theoretically superior to BFS.

Three variants are compared:
  A  DSCF + CSA   (full system)
  B  LPA  + CSA   (community detection swap)
  C  BFS          (uniform weights, no community guidance)

Expected result: A > C and B > C on intra-community questions.
This directly contradicts EF-004's MetaQA finding, confirming that
CSA is effective on the RIGHT class of benchmark.

Usage
-----
  python -m benchmarks.synthetic_eval
  python -m benchmarks.synthetic_eval --hop 1
  python -m benchmarks.synthetic_eval --n-communities 30 --community-size 60
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf, lpa_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from core.resource_governor import ResourceGovernor

from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
from benchmarks.baseline_comparison import (
    UniformCSAEngine,
)

_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)

DATA_DIR  = Path(__file__).parent / "data" / "synthetic"
CACHE_DIR = DATA_DIR / "cache"


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_clustered_graph(
    n_communities: int = 20,
    community_size: int = 50,
    k_intra: int = 5,
    m_inter: int = 3,
    seed: int = 42,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Generate a planted-partition graph with known community structure.

    Returns (G, ground_truth_cmap) where ground_truth_cmap is
    {node_id -> community_index}.

    Parameters
    ----------
    n_communities  : number of communities
    community_size : nodes per community
    k_intra        : intra-community edges per node (approx; deduped)
    m_inter        : inter-community edges per community (sparse bridges)
    seed           : RNG seed (uses a local Random, not global state)
    """
    rng = random.Random(seed)
    G = nx.Graph()
    ground_truth: Dict[str, int] = {}

    # Node IDs: "c{community}_n{index}"
    nodes_by_community: List[List[str]] = []
    for i in range(n_communities):
        community_nodes = [f"c{i}_n{j}" for j in range(community_size)]
        nodes_by_community.append(community_nodes)
        for node in community_nodes:
            G.add_node(node)
            ground_truth[node] = i

    # Intra-community edges (dense)
    intra_edges: set = set()
    for i, community_nodes in enumerate(nodes_by_community):
        for node in community_nodes:
            candidates = [n for n in community_nodes if n != node]
            chosen = rng.sample(candidates, min(k_intra, len(candidates)))
            for nb in chosen:
                edge = frozenset([node, nb])
                if edge not in intra_edges:
                    intra_edges.add(edge)
                    G.add_edge(node, nb, relation="intra_rel")

    # Inter-community edges (sparse bridges)
    inter_pairs: set = set()
    for i in range(n_communities):
        added = 0
        attempts = 0
        while added < m_inter and attempts < m_inter * 10:
            attempts += 1
            j = rng.randint(0, n_communities - 1)
            if j == i or frozenset([i, j]) in inter_pairs:
                continue
            inter_pairs.add(frozenset([i, j]))
            u = rng.choice(nodes_by_community[i])
            v = rng.choice(nodes_by_community[j])
            G.add_edge(u, v, relation="inter_rel")
            added += 1

    return G, ground_truth


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

def generate_qa_pairs(
    G: nx.Graph,
    ground_truth: Dict[str, int],
    hop: int,
    n_questions: int = 500,
    seed: int = 42,
) -> List[Tuple[str, List[str]]]:
    """
    Generate QA pairs where both seed and answer are in the same community.

    The traversal must follow k intra-community hops to reach the answer.
    All paths stay strictly within one community (no inter edges used).

    Returns list of (seed_entity, [answer_entity]).
    """
    rng = random.Random(seed)

    # Build intra-community adjacency: node -> [same-community neighbors]
    intra_adj: Dict[str, List[str]] = {n: [] for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        if data.get("relation") == "intra_rel":
            intra_adj[u].append(v)
            intra_adj[v].append(u)

    def find_intra_paths(start: str, depth: int) -> List[str]:
        """BFS within community to depth hops; return reachable nodes."""
        if depth == 0:
            return [start]
        visited = {start}
        frontier = [start]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                for nb in intra_adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            frontier = next_frontier
        return list(frontier)  # nodes exactly `depth` hops away (approx)

    candidates = list(G.nodes())
    rng.shuffle(candidates)

    pairs: List[Tuple[str, List[str]]] = []
    seen: set = set()

    for seed_node in candidates:
        if len(pairs) >= n_questions:
            break

        # Find nodes exactly `hop` intra-hops away
        reachable = find_intra_paths(seed_node, hop)
        # Remove the seed itself and already-seen (seed, answer) pairs
        reachable = [n for n in reachable if n != seed_node]
        if not reachable:
            continue

        answer = rng.choice(reachable)
        key = (seed_node, answer)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((seed_node, [answer]))

    return pairs


# ---------------------------------------------------------------------------
# Community detection (with disk cache per graph config)
# ---------------------------------------------------------------------------

def load_or_compute_communities(
    G: nx.Graph,
    label: str,
    use_cache: bool = True,
    n_trials: int = 5,
    dscf_seed: int = 42,
) -> Dict[str, int]:
    """Run DSCF and cache the result to CACHE_DIR/communities_{label}.pkl."""
    import pickle
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"communities_{label}.pkl"

    if use_cache and cache_file.exists():
        print(f"  Loading cached communities from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  Running DSCF (n_trials={n_trials})...")
    t0    = time.time()
    parts = best_of_n_dscf(G, n_trials=n_trials, seed=dscf_seed)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}
    print(f"  DSCF: {len(parts)} communities in {time.time()-t0:.1f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(cmap, f)
    return cmap


# ---------------------------------------------------------------------------
# DSCF recovery metric (Adjusted Rand Index approximation)
# ---------------------------------------------------------------------------

def compute_dscf_recovery(
    ground_truth: Dict[str, int],
    detected: Dict[str, int],
) -> float:
    """
    Compute Adjusted Rand Index between ground truth and detected partition.
    Falls back to simple pair-accuracy if sklearn is unavailable.
    Returns float in [-1, 1] where 1.0 = perfect recovery.
    """
    nodes = list(ground_truth.keys())
    labels_true = [ground_truth[n] for n in nodes]
    labels_pred = [detected.get(n, -1) for n in nodes]

    try:
        from sklearn.metrics import adjusted_rand_score
        return adjusted_rand_score(labels_true, labels_pred)
    except ImportError:
        pass

    # Fallback: fraction of same-community pairs that are co-assigned
    same_gt = same_det = both = total = 0
    for i in range(min(len(nodes), 2000)):  # sample for speed
        for j in range(i + 1, min(len(nodes), 2000)):
            gt_same  = labels_true[i] == labels_true[j]
            det_same = labels_pred[i] == labels_pred[j]
            if gt_same:
                same_gt += 1
            if det_same:
                same_det += 1
            if gt_same and det_same:
                both += 1
            total += 1
    if same_gt + same_det - both == 0:
        return 0.0
    return both / (same_gt + same_det - both)   # Jaccard on pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(
    variant_name: str,
    traversal: BeamTraversal,
    qa_pairs: List[Tuple[str, List[str]]],
    hop: int,
    top_k: int = 10,
) -> Dict:
    """Evaluate one variant on one hop level. Returns metrics dict."""
    h1 = h10 = 0
    mrr_sum = 0.0
    skipped = found = 0
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 100 == 0 or (i + 1) == len(qa_pairs):
            print(f"    {i+1:,}/{len(qa_pairs):,} questions "
                  f"({time.time()-t0:.1f}s elapsed)", end="\r")

        paths       = traversal.traverse([seed])
        answers_obj = extract(paths, top_k=top_k, min_hop=1)
        pred        = [a.entity_id for a in answers_obj]

        if not pred:
            skipped += 1
            continue

        found    += 1
        h1       += hits_at_k(pred, correct_answers, k=1)
        h10      += hits_at_k(pred, correct_answers, k=10)
        mrr_sum  += reciprocal_rank(pred, correct_answers)

    elapsed = time.time() - t0
    print()  # newline after progress
    n = len(qa_pairs)

    return {
        "variant":    variant_name,
        "hop":        hop,
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic clustered graph benchmark for CEREBRUM"
    )
    parser.add_argument("--n-communities",  type=int, default=20)
    parser.add_argument("--community-size", type=int, default=50)
    parser.add_argument("--k-intra",        type=int, default=5,
                        help="Intra-community edges per node")
    parser.add_argument("--m-inter",        type=int, default=3,
                        help="Inter-community bridge edges per community")
    parser.add_argument("--hop",            type=int, default=None,
                        help="Hop level to evaluate (1, 2, 3). Default: all.")
    parser.add_argument("--n-questions",    type=int, default=500)
    parser.add_argument("--beam-width",     type=int, default=10)
    parser.add_argument("--top-k",          type=int, default=10)
    parser.add_argument("--use-cache",      action="store_true", default=True)
    parser.add_argument("--no-cache",       action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    hops = [args.hop] if args.hop else [1, 2, 3]

    print("\n=== CEREBRUM — Synthetic Clustered Graph Benchmark ===\n")
    print("Hypothesis: DSCF+CSA outperforms BFS on intra-community questions.")
    print()

    # ------------------------------------------------------------------
    # Generate graph
    # ------------------------------------------------------------------
    print("Generating clustered graph...")
    G, ground_truth = generate_clustered_graph(
        n_communities=args.n_communities,
        community_size=args.community_size,
        k_intra=args.k_intra,
        m_inter=args.m_inter,
        seed=args.seed,
    )
    n_intra = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "intra_rel")
    n_inter = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "inter_rel")
    print(f"  {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges "
          f"({n_intra:,} intra, {n_inter:,} inter)")
    print(f"  {args.n_communities} communities × {args.community_size} nodes")
    print()

    adapter = NetworkXAdapter(G)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    print("Building embeddings (RandomEngine)...")
    random.seed(args.seed)
    engine     = RandomEngine(dim=64)
    labels     = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)

    # ------------------------------------------------------------------
    # Community detection + recovery metrics
    # ------------------------------------------------------------------
    print("Computing community structure...")
    cmap_dscf = load_or_compute_communities(
        G, label=f"dscf_{args.n_communities}x{args.community_size}",
        use_cache=args.use_cache, dscf_seed=args.seed
    )
    ari_dscf = compute_dscf_recovery(ground_truth, cmap_dscf)
    print(f"  DSCF recovery (ARI vs ground truth): {ari_dscf:.4f}")

    lpa_parts = lpa_communities(G)
    cmap_lpa  = {node: cid for cid, members in enumerate(lpa_parts) for node in members}
    ari_lpa   = compute_dscf_recovery(ground_truth, cmap_lpa)
    print(f"  LPA  recovery (ARI vs ground truth): {ari_lpa:.4f}")
    print()

    # ------------------------------------------------------------------
    # Evaluate each hop
    # ------------------------------------------------------------------
    all_results = []

    for hop in hops:
        print(f"=== {hop}-hop ===")
        qa_pairs = generate_qa_pairs(
            G, ground_truth, hop=hop,
            n_questions=args.n_questions, seed=args.seed
        )
        print(f"  {len(qa_pairs):,} intra-community QA pairs generated")

        # Variant A — DSCF + CSA
        print(f"\n  [A] DSCF + CSA  (ARI={ari_dscf:.3f})...")
        dist_dscf = build_community_distance_matrix(G, cmap_dscf)
        adj_dscf  = adjacent_community_pairs(G, cmap_dscf)
        
        adapter.community_map = cmap_dscf
        adapter.embeddings = embeddings
        
        csa_dscf  = CSAEngine(adapter=adapter)
        csa_dscf.set_community_graph(dist_dscf, adj_dscf)
        t_dscf = BeamTraversal(
            adapter=adapter,
            csa_engine=csa_dscf,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR,
        )
        m_a = evaluate_variant("DSCF+CSA", t_dscf, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_a['hits_1']:.4f}  Hits@10={m_a['hits_10']:.4f}  MRR={m_a['mrr']:.4f}")

        # Variant B — LPA + CSA
        print(f"\n  [B] LPA + CSA  (ARI={ari_lpa:.3f})...")
        dist_lpa = build_community_distance_matrix(G, cmap_lpa)
        adj_lpa  = adjacent_community_pairs(G, cmap_lpa)
        
        adapter.community_map = cmap_lpa
        adapter.embeddings = embeddings
        
        csa_lpa  = CSAEngine(adapter=adapter)
        csa_lpa.set_community_graph(dist_lpa, adj_lpa)
        t_lpa = BeamTraversal(
            adapter=adapter,
            csa_engine=csa_lpa,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR,
        )
        m_b = evaluate_variant("LPA+CSA", t_lpa, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_b['hits_1']:.4f}  Hits@10={m_b['hits_10']:.4f}  MRR={m_b['mrr']:.4f}")

        # Variant C — BFS
        print("\n  [C] BFS (uniform weights)...")
        cmap_bfs = {node: 0 for node in G.nodes()}
        
        adapter.community_map = cmap_bfs
        adapter.embeddings = embeddings
        
        csa_bfs  = UniformCSAEngine(adapter=adapter)
        t_bfs = BeamTraversal(
            adapter=adapter,
            csa_engine=csa_bfs,
            beam_width=args.beam_width,
            max_hop=hop,
            governor=_BENCH_GOVERNOR,
        )
        m_c = evaluate_variant("BFS", t_bfs, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_c['hits_1']:.4f}  Hits@10={m_c['hits_10']:.4f}  MRR={m_c['mrr']:.4f}")

        all_results.append({"hop": hop, "dscf": m_a, "lpa": m_b, "bfs": m_c})
        print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=== Summary: CSA Advantage on Intra-Community Questions ===\n")
    print(f"  {'':6} {'Hits@1':>24}   {'Hits@10':>24}   {'MRR':>24}   {'Hits@1 Delta(DSCF-BFS)':>22}")
    print(f"  {'Hop':<6} {'DSCF':>8} {'LPA':>8} {'BFS':>8}   "
          f"{'DSCF':>8} {'LPA':>8} {'BFS':>8}   "
          f"{'DSCF':>8} {'LPA':>8} {'BFS':>8}   {'Delta':>22}")
    print("  " + "-" * 100)
    for r in all_results:
        a, b, c = r["dscf"], r["lpa"], r["bfs"]
        delta = a["hits_1"] - c["hits_1"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {r['hop']}-hop  "
              f"{a['hits_1']:>8.4f} {b['hits_1']:>8.4f} {c['hits_1']:>8.4f}   "
              f"{a['hits_10']:>8.4f} {b['hits_10']:>8.4f} {c['hits_10']:>8.4f}   "
              f"{a['mrr']:>8.4f} {b['mrr']:>8.4f} {c['mrr']:>8.4f}   "
              f"{sign}{delta:>18.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CACHE_DIR / "synthetic_results.csv"
    rows = []
    for r in all_results:
        for variant_key, m in [("dscf", r["dscf"]), ("lpa", r["lpa"]), ("bfs", r["bfs"])]:
            rows.append({
                "hop": r["hop"], "variant": m["variant"],
                "n_total": m["n_total"], "n_answered": m["n_answered"],
                "hits_1": m["hits_1"], "hits_10": m["hits_10"], "mrr": m["mrr"],
                "elapsed_s": m["elapsed_s"],
                "n_communities": args.n_communities,
                "community_size": args.community_size,
                "ari_dscf": round(ari_dscf, 4),
                "ari_lpa": round(ari_lpa, 4),
            })
    if rows:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results saved to {out_file}")
    print()


if __name__ == "__main__":
    main()




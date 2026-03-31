"""
External graph algorithm comparison benchmark for CEREBRUM.

Compares CEREBRUM (DSCF+CSA beam traversal) against standard graph algorithms
that are the conventional baseline for knowledge-graph multi-hop reasoning:

  A  CEREBRUM       DSCF + CSA beam search          (this project)
  B  PPR            Personalized PageRank            (Brin & Page 1998)
  C  SP-BFS         Shortest-path BFS + PageRank rank (Dijkstra-equivalent)
  D  Degree-BFS     BFS expansion, ranked by degree   (heuristic KG baseline)
  E  Uniform-BFS    BFS, no ranking                   (random-order baseline)

All external algorithms use only NetworkX ? no embeddings, no community
detection, no training. They represent the "free lunch" a practitioner gets
from standard graph libraries before reaching for a specialized KG engine.

The comparison highlights when CEREBRUM's structural attention provides a
meaningful lift and when simpler methods are competitive.

Evaluation
----------
  Hits@1   correct answer in position 1
  Hits@10  correct answer in top-10 results
  MRR      Mean Reciprocal Rank

Two graph sources
-----------------
  --mode synthetic   Planted-partition graph (ground-truth communities, no
                     external data needed). Questions require intra-community
                     multi-hop reasoning ? the regime where CSA shines.
  --mode metaqa      Real-world movie KB (43K entities, 134K triples).
                     Requires data in benchmarks/data/metaqa/.

Usage
-----
  python -m benchmarks.graph_algo_comparison --mode synthetic
  python -m benchmarks.graph_algo_comparison --mode metaqa --sample 500
  python -m benchmarks.graph_algo_comparison --mode both --sample 200
  python -m benchmarks.graph_algo_comparison --mode synthetic --hop 2
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from core.resource_governor import ResourceGovernor
from reasoning.answer_extractor import extract

from benchmarks.metaqa_eval import (
    load_kb, load_qa, load_or_compute_communities,
    hits_at_k, reciprocal_rank,
)
from benchmarks.synthetic_eval import (
    generate_clustered_graph, generate_qa_pairs,
    load_or_compute_communities as synth_communities,
)

_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)

CACHE_DIR_META  = Path(__file__).parent / "data" / "metaqa" / "cache"
CACHE_DIR_SYNTH = Path(__file__).parent / "data" / "synthetic" / "cache"


# ---------------------------------------------------------------------------
# External algorithm implementations
# ---------------------------------------------------------------------------

def ppr_answers(
    G: nx.Graph,
    seed: str,
    top_k: int = 10,
    alpha: float = 0.85,
    max_iter: int = 100,
    _cache: dict = {},  # module-level cache across calls
) -> List[str]:
    """
    Personalized PageRank from seed entity.

    Runs full PPR on G each call. For large graphs, callers should pre-compute
    global PageRank once and use the degree-biased version instead.

    Returns the top-K non-seed nodes by PPR score.
    """
    if seed not in G:
        return []
    personalization = {n: (1.0 if n == seed else 0.0) for n in G.nodes()}
    try:
        scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                             max_iter=max_iter, nstart={seed: 1.0})
    except nx.PowerIterationFailedConvergence:
        scores = {n: G.degree(n) for n in G.nodes()}

    scores.pop(seed, None)
    ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
    return ranked[:top_k]


def sp_bfs_answers(
    G: nx.Graph,
    seed: str,
    hop: int,
    global_pr: Dict[str, float],
    top_k: int = 10,
) -> List[str]:
    """
    Shortest-Path BFS: collect all nodes at distance <= hop from seed,
    rank them by global PageRank score (a standard authority proxy).

    This is the canonical "reachability + ranking" baseline used in early
    KG reasoning papers before neural methods.
    """
    if seed not in G:
        return []
    try:
        lengths = nx.single_source_shortest_path_length(G, seed, cutoff=hop)
    except Exception:
        return []

    candidates = [n for n, d in lengths.items() if n != seed and d <= hop]
    candidates.sort(key=lambda n: global_pr.get(n, 0.0), reverse=True)
    return candidates[:top_k]


def degree_bfs_answers(
    G: nx.Graph,
    seed: str,
    hop: int,
    top_k: int = 10,
) -> List[str]:
    """
    Degree-Biased BFS: collect all nodes at distance <= hop from seed,
    rank by node degree (hubs assumed more likely to be answers).

    Simple but effective on hub-and-spoke KGs where popular entities
    appear frequently as answers.
    """
    if seed not in G:
        return []
    try:
        lengths = nx.single_source_shortest_path_length(G, seed, cutoff=hop)
    except Exception:
        return []

    candidates = [n for n, d in lengths.items() if n != seed and d <= hop]
    candidates.sort(key=lambda n: G.degree(n), reverse=True)
    return candidates[:top_k]


def uniform_bfs_answers(
    G: nx.Graph,
    seed: str,
    hop: int,
    top_k: int = 10,
    seed_rng: int = 0,
) -> List[str]:
    """
    Uniform BFS: collect all nodes at distance <= hop, return in arbitrary
    (but deterministic) order. No ranking heuristic applied.

    This is the weakest baseline ? equivalent to random selection among
    reachable nodes.
    """
    if seed not in G:
        return []
    try:
        lengths = nx.single_source_shortest_path_length(G, seed, cutoff=hop)
    except Exception:
        return []

    candidates = [n for n, d in lengths.items() if n != seed and d <= hop]
    # Deterministic shuffle so results are reproducible
    r = random.Random(seed_rng)
    r.shuffle(candidates)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# CEREBRUM traversal builder
# ---------------------------------------------------------------------------

def build_cerebrum(
    adapter: NetworkXAdapter,
    G: nx.Graph,
    embeddings: dict,
    beam_width: int,
    max_hop: int,
    cmap: Dict[str, int],
    pagerank: Optional[Dict[str, float]] = None,
    zeta: float = 0.1,
) -> BeamTraversal:
    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    adapter.community_map = cmap
    adapter.embeddings = embeddings
    csa = CSAEngine(adapter=adapter, pagerank=pagerank, zeta=zeta)
    csa.set_community_graph(dist, adj)
    return BeamTraversal(adapter=adapter, csa_engine=csa,
                         beam_width=beam_width, max_hop=max_hop,
                         governor=_BENCH_GOVERNOR)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_external(
    name: str,
    fn,               # callable(seed) -> List[str]
    qa_pairs: List[Tuple[str, List[str]]],
    top_k: int = 10,
) -> Dict:
    """Evaluate a function-based algorithm against QA pairs."""
    h1 = h10 = found = skipped = 0
    mrr_sum = 0.0
    t0 = time.time()
    n  = len(qa_pairs)

    for i, qa in enumerate(qa_pairs):
        seed, correct = qa[0], qa[1]
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    {i+1:,}/{n:,}  ({time.time()-t0:.1f}s)", end="\r")

        pred = fn(seed)
        if not pred:
            skipped += 1
            continue
        found   += 1
        h1      += hits_at_k(pred, correct, k=1)
        h10     += hits_at_k(pred, correct, k=top_k)
        mrr_sum += reciprocal_rank(pred, correct)

    print()
    elapsed = time.time() - t0
    return {
        "variant":    name,
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  round(elapsed, 2),
    }


def evaluate_cerebrum(
    traversal: BeamTraversal,
    qa_pairs: List[Tuple],
    top_k: int = 10,
    embed_fn=None,
) -> Dict:
    """
    Evaluate CEREBRUM beam traversal.

    Parameters
    ----------
    embed_fn : optional callable(question_text: str) -> np.ndarray
        When provided, encodes question text into a query embedding that is
        passed to the answer extractor for semantic re-ranking (Fix 1).
        QA pairs must be (seed, answers, question_text) triples for this to activate.
    """
    h1 = h10 = found = skipped = 0
    mrr_sum = 0.0
    t0 = time.time()
    n  = len(qa_pairs)

    for i, qa in enumerate(qa_pairs):
        seed, correct = qa[0], qa[1]
        q_text = qa[2] if len(qa) > 2 else None

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    {i+1:,}/{n:,}  ({time.time()-t0:.1f}s)", end="\r")

        paths = traversal.traverse([seed])

        # Build query embedding from question text if encoder is available
        q_emb = None
        if embed_fn is not None and q_text:
            try:
                q_emb = embed_fn(q_text)
            except Exception:
                pass

        answers = extract(paths, top_k=top_k, min_hop=1, query_embedding=q_emb)
        pred    = [a.entity_id for a in answers]

        if not pred:
            skipped += 1
            continue
        found   += 1
        h1      += hits_at_k(pred, correct, k=1)
        h10     += hits_at_k(pred, correct, k=top_k)
        mrr_sum += reciprocal_rank(pred, correct)

    print()
    elapsed = time.time() - t0
    return {
        "variant":    "CEREBRUM (DSCF+CSA)",
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict], title: str, hop: int) -> None:
    width = 100
    print(f"\n{'-'*width}")
    print(f"  {title}  |  {hop}-hop")
    print(f"{'-'*width}")
    print(f"  {'Algorithm':<30}  {'Hits@1':>8}  {'Hits@10':>8}  {'MRR':>8}  "
          f"{'Answered':>9}  {'Time(s)':>8}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*8}")

    baseline_h1  = results[-1]["hits_1"]   if results else 0
    baseline_mrr = results[-1]["mrr"]      if results else 0

    for r in results:
        delta_h1  = r["hits_1"]  - baseline_h1
        r["mrr"]     - baseline_mrr
        marker = ""
        if r is results[0]:
            marker = " <-- CEREBRUM"
        delta_str = f"  [{delta_h1:+.4f} vs BFS]" if r is not results[-1] else ""
        print(f"  {r['variant']:<30}  {r['hits_1']:>8.4f}  {r['hits_10']:>8.4f}  "
              f"{r['mrr']:>8.4f}  {r['n_answered']:>9,}  {r['elapsed_s']:>8.2f}"
              f"{delta_str}{marker}")
    print(f"{'-'*width}\n")


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

def run_synthetic(args) -> List[Dict]:
    print("\n" + "="*70)
    print("SYNTHETIC CLUSTERED GRAPH ? Intra-Community Reasoning")
    print("="*70)
    print("Hypothesis: DSCF+CSA outperforms graph heuristics when the answer")
    print("lies within the seed entity's community.\n")

    print("Generating graph...")
    G, ground_truth = generate_clustered_graph(
        n_communities=args.n_communities,
        community_size=args.community_size,
        k_intra=args.k_intra,
        m_inter=args.m_inter,
        seed=args.seed,
    )
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  {n_nodes:,} nodes | {n_edges:,} edges | "
          f"{args.n_communities} communities ? {args.community_size} nodes")

    adapter = NetworkXAdapter(G)
    engine  = RandomEngine(dim=64)
    embeddings = engine.encode_entities({n: n for n in G.nodes()})

    print("Computing DSCF community structure...")
    label = f"dscf_{args.n_communities}x{args.community_size}"
    cmap  = synth_communities(G, label=label, use_cache=not args.no_cache, dscf_seed=args.seed)

    print("Pre-computing global PageRank (for SP-BFS baseline)...")
    t0 = time.time()
    global_pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    print(f"  Done in {time.time()-t0:.1f}s")

    hops = [args.hop] if args.hop else [1, 2, 3]
    all_rows: List[Dict] = []

    for hop in hops:
        print(f"\n--- {hop}-hop ---")
        qa_pairs = generate_qa_pairs(
            G, ground_truth, hop=hop,
            n_questions=args.n_questions, seed=args.seed,
        )
        print(f"  {len(qa_pairs):,} QA pairs")

        results = []

        # A ? CEREBRUM
        print("\n  [A] CEREBRUM DSCF+CSA...")
        trav = build_cerebrum(adapter, G, embeddings, args.beam_width, hop, cmap,
                              pagerank=global_pr, zeta=0.1)
        m_a  = evaluate_cerebrum(trav, qa_pairs, top_k=args.top_k, embed_fn=None)
        results.append(m_a)

        # B ? PPR
        print("  [B] Personalized PageRank (PPR)...")
        m_b = evaluate_external(
            "Personalized PageRank (PPR)",
            lambda seed, h=hop: ppr_answers(G, seed, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_b)

        # C ? SP-BFS + PageRank ranking
        print("  [C] Shortest-Path BFS + PageRank rank...")
        m_c = evaluate_external(
            "SP-BFS + PageRank rank",
            lambda seed, h=hop: sp_bfs_answers(G, seed, h, global_pr, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_c)

        # D ? Degree BFS
        print("  [D] Degree-Biased BFS...")
        m_d = evaluate_external(
            "Degree-Biased BFS",
            lambda seed, h=hop: degree_bfs_answers(G, seed, h, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_d)

        # E ? Uniform BFS (no ranking)
        print("  [E] Uniform BFS (no ranking)...")
        m_e = evaluate_external(
            "Uniform BFS (no ranking)",
            lambda seed, h=hop: uniform_bfs_answers(G, seed, h, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_e)

        print_results_table(results, "Synthetic Clustered Graph", hop)

        for r in results:
            all_rows.append({"mode": "synthetic", "hop": hop, **r})

    return all_rows


# ---------------------------------------------------------------------------
# MetaQA benchmark
# ---------------------------------------------------------------------------

def run_metaqa(args) -> List[Dict]:
    print("\n" + "="*70)
    print("MetaQA ? Real-World Movie Knowledge Graph")
    print("="*70)
    print("  43,234 entities | 134,741 triples | 9 relation types")
    print(f"  Sample: {args.sample or 'full test set'} questions per hop\n")

    print("Loading MetaQA KB...")
    adapter = load_kb(undirected=True)
    G       = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} entities, {G.number_of_edges():,} edges")

    # Try to load cached MiniLM embeddings (384-dim) so path embeddings and
    # query embeddings share the same space. Fall back to RandomEngine if absent.
    _emb_cache = CACHE_DIR_META / "embeddings_minilm.pkl"
    if _emb_cache.exists():
        import pickle
        with open(_emb_cache, "rb") as _f:
            embeddings = pickle.load(_f)
        print(f"  Loaded MiniLM embeddings for {len(embeddings):,} entities (384-dim)")
    else:
        engine     = RandomEngine(dim=64)
        embeddings = engine.encode_entities({n: n for n in G.nodes()})
        print("  Using random embeddings (64-dim)")

    print("Loading/computing DSCF communities (MetaQA)...")
    cmap = load_or_compute_communities(G, use_cache=not args.no_cache, dscf_seed=args.seed)
    n_comm = len(set(cmap.values()))
    print(f"  {n_comm} communities detected")

    print("Pre-computing global PageRank (for SP-BFS baseline + CEREBRUM prior, ~30s)...")
    t0 = time.time()
    global_pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Set up sentence-transformer for query embedding (Fix 1)
    embed_fn = None
    try:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_fn(t):
            return _st.encode(t, show_progress_bar=False)
        print("  Sentence-transformer query encoding: enabled")
    except ImportError:
        print("  Sentence-transformer not available; running without query embedding")

    hops = [args.hop] if args.hop else [1, 2, 3]
    all_rows: List[Dict] = []

    for hop in hops:
        print(f"\n--- MetaQA {hop}-hop ---")
        qa_pairs = load_qa(hop, sample=args.sample, seed=args.seed,
                           include_question=(embed_fn is not None))
        print(f"  {len(qa_pairs):,} test questions")

        results = []

        # A ? CEREBRUM
        print("\n  [A] CEREBRUM DSCF+CSA...")
        trav = build_cerebrum(adapter, G, embeddings, args.beam_width, hop, cmap,
                              pagerank=global_pr, zeta=0.1)
        m_a  = evaluate_cerebrum(trav, qa_pairs, top_k=args.top_k, embed_fn=embed_fn)
        results.append(m_a)

        # B ? PPR  (expensive on 43K nodes; use nx.pagerank personalization)
        print("  [B] Personalized PageRank (PPR)...")
        m_b = evaluate_external(
            "Personalized PageRank (PPR)",
            lambda seed, h=hop: ppr_answers(G, seed, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_b)

        # C ? SP-BFS + PageRank ranking
        print("  [C] SP-BFS + PageRank rank...")
        m_c = evaluate_external(
            "SP-BFS + PageRank rank",
            lambda seed, h=hop: sp_bfs_answers(G, seed, h, global_pr, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_c)

        # D ? Degree BFS
        print("  [D] Degree-Biased BFS...")
        m_d = evaluate_external(
            "Degree-Biased BFS",
            lambda seed, h=hop: degree_bfs_answers(G, seed, h, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_d)

        # E ? Uniform BFS
        print("  [E] Uniform BFS (no ranking)...")
        m_e = evaluate_external(
            "Uniform BFS (no ranking)",
            lambda seed, h=hop: uniform_bfs_answers(G, seed, h, top_k=args.top_k),
            qa_pairs, top_k=args.top_k,
        )
        results.append(m_e)

        print_results_table(results, "MetaQA Movie KG", hop)

        for r in results:
            all_rows.append({"mode": "metaqa", "hop": hop, **r})

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CEREBRUM vs standard graph algorithms benchmark"
    )
    parser.add_argument("--mode",           choices=["synthetic", "metaqa", "both"],
                        default="synthetic",
                        help="Which benchmark to run (default: synthetic)")
    parser.add_argument("--hop",            type=int, default=None,
                        help="Single hop level (1/2/3). Default: all.")
    parser.add_argument("--sample",         type=int, default=None,
                        help="Sample N questions per hop (MetaQA). Default: full test set.")
    parser.add_argument("--n-questions",    type=int, default=500,
                        help="Questions per hop for synthetic benchmark.")
    parser.add_argument("--beam-width",     type=int, default=10)
    parser.add_argument("--top-k",          type=int, default=10)
    parser.add_argument("--n-communities",  type=int, default=20)
    parser.add_argument("--community-size", type=int, default=50)
    parser.add_argument("--k-intra",        type=int, default=5)
    parser.add_argument("--m-inter",        type=int, default=3)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--no-cache",       action="store_true")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  CEREBRUM vs Standard Graph Algorithms -Knowledge Graph QA")
    print("="*70)
    print()
    print("Algorithms under comparison:")
    print("  A  CEREBRUM (DSCF+CSA)          ? Community-Structured Attention beam search")
    print("  B  Personalized PageRank (PPR)   ? Random walk from seed (Brin & Page 1998)")
    print("  C  SP-BFS + PageRank rank        ? All reachable nodes, ranked by authority")
    print("  D  Degree-Biased BFS             ? Reachable nodes, ranked by hub degree")
    print("  E  Uniform BFS (no ranking)      ? Reachable nodes, no ranking heuristic")

    all_rows: List[Dict] = []

    if args.mode in ("synthetic", "both"):
        all_rows += run_synthetic(args)

    if args.mode in ("metaqa", "both"):
        all_rows += run_metaqa(args)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if all_rows:
        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "graph_algo_comparison.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Results saved to {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()

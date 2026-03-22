"""
MetaQA benchmark evaluation for Parallax (Phase 4).

Dataset
-------
MetaQA: Movie question-answering over a knowledge graph.
  KB     : 134,741 triples | 43,234 entities | 9 relation types
  1-hop  :  9,947 test questions
  2-hop  : 14,872 test questions
  3-hop  : 14,274 test questions
  Source : https://github.com/yuyuz/MetaQA (Zhang et al., ICLR 2018)
  License: MIT

Metrics
-------
  Hits@1   fraction of questions where correct answer is the top-1 result
  Hits@10  fraction of questions where correct answer is in top-10 results
  MRR      Mean Reciprocal Rank: mean(1 / rank_of_first_correct_answer)
           where rank = 0 (i.e. not found) contributes 0

Question format (vanilla splits)
---------------------------------
  <question with [seed entity] in brackets>\t<answer1>|<answer2>|...

KB format
---------
  subject|relation|object   (pipe-delimited, UTF-8)

Usage
-----
  # All three hop levels, full test set:
  python -m benchmarks.metaqa_eval

  # Single hop level:
  python -m benchmarks.metaqa_eval --hop 3

  # Quick development run (500 questions per hop):
  python -m benchmarks.metaqa_eval --sample 500

  # Wider beam for better recall at cost of speed:
  python -m benchmarks.metaqa_eval --beam-width 20

  # Use cached communities (skip DSCF recomputation):
  python -m benchmarks.metaqa_eval --use-cache

Notes
-----
  - KB is loaded as an undirected graph so traversal works in both directions.
    This is standard practice for MetaQA (see EmbedKGQA, NSM, KGT5 papers).
  - Default embedding engine is RandomEngine. This means the alpha
    (semantic similarity) term in CSA is noise; attention is driven by
    community structure (beta term) alone. For sentence-transformer results,
    install sentence-transformers and pass --embeddings sentence.
  - Community detection (DSCF) on 43K nodes takes ~60-120s on first run.
    Results are pickled to benchmarks/data/metaqa/cache/ for reuse.
"""

import argparse
import csv
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Confirm repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf, merge_small_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"
KB_FILE   = DATA_DIR / "kb.txt"

QA_FILES = {
    1: DATA_DIR / "1-hop" / "vanilla" / "qa_test.txt",
    2: DATA_DIR / "2-hop" / "vanilla" / "qa_test.txt",
    3: DATA_DIR / "3-hop" / "vanilla" / "qa_test.txt",
}

# ---------------------------------------------------------------------------
# KB loading
# ---------------------------------------------------------------------------

def load_kb(undirected: bool = True) -> NetworkXAdapter:
    """
    Load MetaQA kb.txt into a NetworkXAdapter.

    The KB is loaded as undirected by default so that traversal can follow
    edges in both directions (standard MetaQA practice).
    """
    G = nx.Graph() if undirected else nx.DiGraph()
    with open(KB_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            G.add_edge(subj.strip(), obj.strip(), relation=rel.strip())
    return NetworkXAdapter(G)


# ---------------------------------------------------------------------------
# QA file parsing
# ---------------------------------------------------------------------------

def load_qa(
    hop: int,
    sample: Optional[int] = None,
    seed: int = 42,
    include_question: bool = False,
) -> List[Tuple]:
    """
    Load QA pairs for a given hop level.

    Returns list of (seed_entity, [answer_entity, ...]) tuples by default.
    When include_question=True, returns (seed_entity, [answer_entity, ...], question_text)
    triples, where question_text has the entity brackets removed for clean encoding.
    """
    path = QA_FILES[hop]
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0]
            answers  = [a.strip() for a in parts[1].split("|") if a.strip()]

            m = re.search(r"\[(.+?)\]", question)
            if not m:
                continue
            seed_entity = m.group(1).strip()
            if seed_entity and answers:
                if include_question:
                    # Strip brackets so "what movies did [Tom Hanks] direct"
                    # becomes "what movies did Tom Hanks direct" for encoding
                    clean_q = re.sub(r"\[(.+?)\]", r"\1", question)
                    pairs.append((seed_entity, answers, clean_q))
                else:
                    pairs.append((seed_entity, answers))

    if sample is not None and sample < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample)

    return pairs


# ---------------------------------------------------------------------------
# Community detection with disk cache
# ---------------------------------------------------------------------------

def load_or_compute_communities(
    G: nx.Graph,
    use_cache: bool = True,
    n_trials: int = 3,
    dscf_seed: int = 42,
) -> Dict[str, int]:
    """
    Run DSCF on the graph and return {node -> community_id}.
    Caches the result to CACHE_DIR/communities.pkl.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "communities.pkl"

    if use_cache and cache_file.exists():
        print(f"  Loading cached communities from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  Running DSCF on {G.number_of_nodes():,} nodes "
          f"({G.number_of_edges():,} edges) — this may take 1-2 minutes...")
    t0    = time.time()
    parts = best_of_n_dscf(G, n_trials=n_trials, seed=dscf_seed)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}
    elapsed = time.time() - t0
    print(f"  DSCF complete: {len(parts)} communities in {elapsed:.1f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(cmap, f)
    print(f"  Communities cached to {cache_file}")

    return cmap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hits_at_k(answers: List[str], correct: List[str], k: int) -> int:
    """1 if any correct answer is in top-k answers, else 0."""
    top_k_set = set(answers[:k])
    return int(any(c in top_k_set for c in correct))


def reciprocal_rank(answers: List[str], correct: List[str]) -> float:
    """1/rank of first correct answer, or 0 if not found."""
    correct_set = set(correct)
    for rank, ans in enumerate(answers, 1):
        if ans in correct_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Single-hop evaluation
# ---------------------------------------------------------------------------

def evaluate_hop(
    hop: int,
    traversal: BeamTraversal,
    qa_pairs: List[Tuple[str, List[str]]],
    top_k: int = 10,
) -> Dict:
    """
    Evaluate one hop level. Returns a metrics dict.
    """
    h1 = h10 = mrr_sum = 0
    skipped = found = 0

    t0 = time.time()
    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 500 == 0 or (i + 1) == len(qa_pairs):
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(qa_pairs):,} questions "
                  f"({elapsed:.1f}s elapsed)", end="\r")

        paths   = traversal.traverse([seed])
        answers_obj = extract(paths, top_k=top_k, min_hop=1)
        pred    = [a.entity_id for a in answers_obj]

        if not pred:
            skipped += 1
            continue

        found  += 1
        h1     += hits_at_k(pred, correct_answers, k=1)
        h10    += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    elapsed = time.time() - t0
    n = len(qa_pairs)
    print()  # newline after progress

    return {
        "hop":        hop,
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1 / n,
        "hits_10":    h10 / n,
        "mrr":        mrr_sum / n,
        "elapsed_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MetaQA benchmark for Parallax")
    parser.add_argument("--hop",        type=int,   default=None,
                        help="Evaluate only this hop level (1, 2, or 3). Default: all.")
    parser.add_argument("--sample",     type=int,   default=None,
                        help="Evaluate on a random sample of N questions per hop.")
    parser.add_argument("--beam-width", type=int,   default=10,
                        help="Beam width for traversal (default: 10).")
    parser.add_argument("--top-k",      type=int,   default=10,
                        help="Extract top-K answers per query (default: 10).")
    parser.add_argument("--use-cache",  action="store_true", default=True,
                        help="Use cached DSCF communities if available (default: True).")
    parser.add_argument("--no-cache",   action="store_true",
                        help="Recompute DSCF even if cache exists.")
    parser.add_argument("--embeddings", choices=["random", "sentence"], default="random",
                        help="Embedding engine (default: random).")
    parser.add_argument("--min-community-size", type=int, default=0,
                        help="Merge communities smaller than this into best neighbor. "
                             "0 = disabled. Recommended: 20 for MetaQA.")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    hops = [args.hop] if args.hop else [1, 2, 3]

    # ------------------------------------------------------------------
    # Setup (done once, shared across all hop evaluations)
    # ------------------------------------------------------------------
    print("\n=== Parallax — MetaQA Benchmark ===\n")

    if not KB_FILE.exists():
        print(f"ERROR: kb.txt not found at {KB_FILE}")
        print("Place the MetaQA kb.txt file in benchmarks/data/metaqa/")
        sys.exit(1)

    for hop in hops:
        if not QA_FILES[hop].exists():
            print(f"ERROR: QA file for {hop}-hop not found: {QA_FILES[hop]}")
            sys.exit(1)

    print("Loading knowledge graph...")
    t0      = time.time()
    adapter = load_kb(undirected=True)
    G       = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} entities, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    print("Computing/loading community structure...")
    random.seed(args.seed)
    cmap = load_or_compute_communities(G, use_cache=args.use_cache, dscf_seed=args.seed)
    n_communities = len(set(cmap.values()))
    print(f"  {n_communities} communities (raw DSCF)")

    if args.min_community_size > 0:
        print(f"  Merging communities smaller than {args.min_community_size} members...")
        t0 = time.time()
        cmap = merge_small_communities(cmap, G, min_size=args.min_community_size)
        n_communities = len(set(cmap.values()))
        print(f"  {n_communities} communities after merge ({time.time()-t0:.1f}s)")

    print("Building entity embeddings...")
    t0 = time.time()
    if args.embeddings == "sentence":
        try:
            from core.embedding_engine import SentenceEngine
            engine = SentenceEngine()
            print(f"  Using SentenceEngine ({engine.dim}-dim)")
        except ImportError:
            print("  sentence-transformers not installed — falling back to RandomEngine")
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)
        print(f"  Using RandomEngine (64-dim, community structure only)")

    labels     = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)
    print(f"  {len(embeddings):,} entity vectors ({time.time()-t0:.1f}s)")

    print("Building CSA engine...")
    distances = build_community_distance_matrix(G, cmap)
    adj       = adjacent_community_pairs(G, cmap)
    
    # Attach communities and embeddings to adapter for lookups
    adapter.community_map = cmap
    adapter.embeddings = embeddings
    
    csa = CSAEngine(adapter=adapter)
    csa.set_community_graph(distances, adj)

    # ------------------------------------------------------------------
    # Evaluate each hop level
    # ------------------------------------------------------------------
    results = []

    for hop in hops:
        print(f"\n--- {hop}-hop evaluation ---")
        qa_pairs = load_qa(hop, sample=args.sample, seed=args.seed)
        n_label  = f"{len(qa_pairs):,}" + (f" (sample)" if args.sample else "")
        print(f"  {n_label} test questions")

        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=args.beam_width,
            max_hop=hop,
        )

        print(f"  Running traversal (beam_width={args.beam_width}, max_hop={hop})...")
        metrics = evaluate_hop(hop, traversal, qa_pairs, top_k=args.top_k)
        results.append(metrics)

        print(f"  Hits@1  : {metrics['hits_1']:.4f}  ({metrics['hits_1']*100:.1f}%)")
        print(f"  Hits@10 : {metrics['hits_10']:.4f}  ({metrics['hits_10']*100:.1f}%)")
        print(f"  MRR     : {metrics['mrr']:.4f}")
        print(f"  Answered: {metrics['n_answered']:,}/{metrics['n_total']:,}  "
              f"(skipped: {metrics['n_skipped']:,})")
        print(f"  Time    : {metrics['elapsed_s']:.1f}s")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Results Summary ===\n")
    print(f"  Model     : Parallax CSA + DSCF")
    print(f"  Embeddings: {args.embeddings}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Top-K     : {args.top_k}")
    if args.sample:
        print(f"  Sample    : {args.sample} per hop")
    print()
    print(f"  {'Hop':<6} {'N':>7} {'Hits@1':>8} {'Hits@10':>9} {'MRR':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")
    for m in results:
        print(f"  {m['hop']}-hop  {m['n_total']:>7,} "
              f"{m['hits_1']:>8.4f} {m['hits_10']:>9.4f} {m['mrr']:>8.4f}")

    # ------------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results_file = CACHE_DIR / "metaqa_results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hop", "n_total", "n_answered", "n_skipped",
            "hits_1", "hits_10", "mrr", "beam_width", "embeddings", "elapsed_s"
        ])
        writer.writeheader()
        for m in results:
            writer.writerow({**m, "beam_width": args.beam_width, "embeddings": args.embeddings})

    print(f"\n  Results saved to {results_file}")
    print()


if __name__ == "__main__":
    main()




"""
WebQSP benchmark evaluation for Parallax (Phase 4).

WebQSP: 4,737 questions, 1-2 hop reasoning over Freebase.
KB: FB15k-237 (PyKEEN subset).

Metrics:
  - Hits@1: answer entity in top-1 result
  - Hits@10: answer entity in top-10
  - MRR: mean reciprocal rank
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf, lpa_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine, UniformCSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent / "data" / "webqsp"
KB_FILE   = DATA_DIR / "freebase_subset.txt"
JSON_FILE = DATA_DIR / "WebQSP.test.json"
CACHE_DIR = DATA_DIR / "cache"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_fb_subset() -> NetworkXAdapter:
    """Load the FB15k-237 subset into a NetworkXAdapter."""
    G = nx.Graph()  # Undirected for reasoning
    if not KB_FILE.exists():
        print(f"ERROR: KB file not found at {KB_FILE}. Run setup_webqsp.py first.")
        sys.exit(1)
        
    print(f"Loading KB from {KB_FILE}...")
    with open(KB_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                s, r, o = parts
                G.add_edge(s, o, relation=r)
    return NetworkXAdapter(G)

def parse_webqsp_json(graph: nx.Graph) -> List[Tuple[str, List[str], str]]:
    """
    Parse WebQSP.test.json into (seed_entity, answers, question_text) tuples.
    Only returns questions where the seed entity and at least one answer
    exist in our graph subset.
    """
    if not JSON_FILE.exists():
        print(f"ERROR: JSON file not found at {JSON_FILE}.")
        sys.exit(1)
        
    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)
        
    qa_pairs = []
    skipped_missing_seed = 0
    skipped_no_reachable_answers = 0
    
    for q in data["Questions"]:
        raw_text = q["RawQuestion"]
        for p in q["Parses"]:
            if not p["Sparql"]:
                continue
            
            # Extract topic entity MIDs (e.g. ns:m.03_r3)
            seeds = re.findall(r"ns:(m\.[a-z0-9_]+)", p["Sparql"])
            if not seeds:
                continue
            
            # Map MID to /m/ format used in FB15k-237
            seed = "/" + seeds[0].replace(".", "/")
            
            if seed not in graph:
                skipped_missing_seed += 1
                continue
                
            # Extract answers
            answers = []
            if p["Answers"]:
                for ans in p["Answers"]:
                    ans_mid = "/" + ans["AnswerArgument"].replace(".", "/")
                    if ans_mid in graph:
                        answers.append(ans_mid)
            
            if not answers:
                skipped_no_reachable_answers += 1
                continue
                
            qa_pairs.append((seed, answers, raw_text))
            break # Use first valid parse
            
    print(f"Parsed {len(qa_pairs)} QA pairs from {len(data['Questions'])} questions.")
    print(f"  Skipped (seed not in subset): {skipped_missing_seed}")
    print(f"  Skipped (no answers in subset): {skipped_no_reachable_answers}")
    return qa_pairs

# ---------------------------------------------------------------------------
# Metrics (reused from metaqa_eval)
# ---------------------------------------------------------------------------

def hits_at_k(answers: List[str], correct: List[str], k: int) -> int:
    top_k_set = set(answers[:k])
    return int(any(c in top_k_set for c in correct))

def reciprocal_rank(answers: List[str], correct: List[str]) -> float:
    correct_set = set(correct)
    for rank, ans in enumerate(answers, 1):
        if ans in correct_set:
            return 1.0 / rank
    return 0.0

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(
    variant_name: str,
    traversal: BeamTraversal,
    qa_pairs: List[Tuple[str, List[str], str]],
    top_k: int = 10,
) -> Dict:
    h1 = h10 = mrr_sum = 0
    found = 0
    t0 = time.time()
    
    for i, (seed, correct_answers, _) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
            print(f"    {i+1}/{len(qa_pairs)} questions ({time.time()-t0:.1f}s)", end="\r")
            
        paths = traversal.traverse([seed])
        ans_obj = extract(paths, top_k=top_k, min_hop=1)
        pred = [a.entity_id for a in ans_obj]
        
        if pred:
            found += 1
            h1 += hits_at_k(pred, correct_answers, k=1)
            h10 += hits_at_k(pred, correct_answers, k=10)
            mrr_sum += reciprocal_rank(pred, correct_answers)
            
    print()
    n = len(qa_pairs)
    return {
        "variant": variant_name,
        "n": n,
        "hits_1": h1 / n if n else 0,
        "hits_10": h10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "time": time.time() - t0
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WebQSP benchmark for Parallax")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--max-hop", type=int, default=2)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--bridge-bonus", type=float, default=0.4)
    args = parser.parse_args()

    adapter = load_fb_subset()
    G = adapter.to_networkx()
    
    qa_pairs = parse_webqsp_json(G)
    if args.sample:
        import random
        random.seed(42)
        qa_pairs = random.sample(qa_pairs, min(args.sample, len(qa_pairs)))
        
    print(f"Evaluating {len(qa_pairs)} questions...")
    
    # Embeddings
    engine = RandomEngine(dim=64)
    embeddings = engine.encode_entities({n: n for n in G.nodes()})
    
    # Community detection
    print("Computing community structure (DSCF)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"communities_dscf_{G.number_of_nodes()}.json"
    
    if args.use_cache and cache_file.exists():
        print(f"  Loading from {cache_file}")
        with open(cache_file) as f:
            cmap_dscf = json.load(f)
    else:
        parts = best_of_n_dscf(G, n_trials=3, seed=42)
        cmap_dscf = {node: cid for cid, mem in enumerate(parts) for node in mem}
        with open(cache_file, "w") as f:
            json.dump(cmap_dscf, f)
            
    print("Computing LPA...")
    lpa_parts = lpa_communities(G)
    cmap_lpa = {node: cid for cid, mem in enumerate(lpa_parts) for node in mem}
    
    # Build Edge Weights (Bridge Bonus)
    edge_weights = {}
    for _, _, r in G.edges(data="relation"):
        if r:
            edge_weights[r] = args.bridge_bonus
            
    def run_eval(name, cmap, variant):
        print(f"\nVariant: {name}")
        if variant == "bfs":
            csa = UniformCSAEngine(communities=cmap, embeddings=embeddings)
        else:
            dist = build_community_distance_matrix(G, cmap)
            adj = adjacent_community_pairs(G, cmap)
            csa = CSAEngine(communities=cmap, embeddings=embeddings)
            csa.set_community_graph(dist, adj)
            
        traversal = BeamTraversal(
            adapter=adapter, csa_engine=csa, embeddings=embeddings,
            communities=cmap, beam_width=args.beam_width, max_hop=args.max_hop,
            edge_type_weights=edge_weights if variant != "bfs" else None
        )
        return evaluate_variant(name, traversal, qa_pairs)

    results = []
    results.append(run_eval("DSCF+CSA", cmap_dscf, "dscf"))
    results.append(run_eval("LPA+CSA", cmap_lpa, "lpa"))
    results.append(run_eval("BFS", {n: 0 for n in G.nodes()}, "bfs"))
    
    print("\n=== WebQSP Results ===")
    print(f"{'Variant':<12} {'Hits@1':>8} {'Hits@10':>9} {'MRR':>8}")
    for r in results:
        print(f"{r['variant']:<12} {r['hits_1']:>8.4f} {r['hits_10']:>9.4f} {r['mrr']:>8.4f}")

if __name__ == "__main__":
    main()

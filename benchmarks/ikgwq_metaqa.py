"""
CEREBRUM IKGWQ-MetaQA Benchmark — Phase 44
============================================

Adapts the IKGWQ (Incomplete Knowledge Graph) protocol to the MetaQA dataset.
Evaluates CEREBRUM's graceful degradation on 3-hop reasoning tasks
under controlled edge removal (Level 0 to Level 4).

Features:
- Standard IKGWQ edge removal protocol.
- REM Engine synthesis evaluation (IKGWQ-S).
- 10-parameter logit scoring (Phase 43).
"""
import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf
from core.embedding_engine import SentenceEngine
from core.attention_engine import CSAEngine
from core.rem_engine import REMEngine
from core.structural_encoder import (
    build_community_distance_matrix,
    adjacent_community_pairs,
    build_community_graph,
    coarsen_communities,
)
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

METAQA_DIR = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR  = Path(__file__).parent / "data" / "ikgwq_metaqa" / "cache"

INCOMPLETENESS_LEVELS = {
    0: 0.00,   # complete baseline
    1: 0.05,   # mild
    2: 0.15,   # moderate
    3: 0.30,   # severe
    4: 0.50,   # extreme
}

LEVEL_LABELS = {0: "Complete", 1: "Mild (5%)", 2: "Moderate (15%)",
                3: "Severe (30%)", 4: "Extreme (50%)"}

# ---------------------------------------------------------------------------
# MetaQA Specific Loading
# ---------------------------------------------------------------------------

def load_metaqa_graph() -> nx.Graph:
    kb_file = METAQA_DIR / "kb.txt"
    if not kb_file.exists():
        print(f"ERROR: MetaQA KB not found at {kb_file}")
        sys.exit(1)
    
    G = nx.Graph()
    with open(kb_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                s, r, o = parts
                G.add_edge(s, o, relation=r)
    return G

def extract_seed(text: str) -> Optional[str]:
    match = re.search(r"\[(.*?)\]", text)
    return match.group(1) if match else None

def load_metaqa_qa(hop: int = 3) -> List[Tuple[str, List[str], str]]:
    qa_file = METAQA_DIR / f"{hop}-hop" / "vanilla" / "qa_test.txt"
    if not qa_file.exists():
        print(f"ERROR: MetaQA QA file not found at {qa_file}")
        return []
    
    qa = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line: continue
            q_part, a_part = line.strip().split("\t")
            seed = extract_seed(q_part)
            if not seed: continue
            answers = a_part.split("|")
            qa.append((seed, answers, q_part))
    return qa

# ---------------------------------------------------------------------------
# Protocol Implementation
# ---------------------------------------------------------------------------

def apply_incompleteness(G_full: nx.Graph, qa_pairs: List[Tuple], removal_fraction: float, rng: random.Random) -> nx.Graph:
    if removal_fraction <= 0:
        return G_full.copy()

    G = G_full.copy()
    answer_nodes: Set[str] = set()
    seed_nodes:   Set[str] = set()
    for seed, answers, _ in qa_pairs:
        seed_nodes.add(seed)
        answer_nodes.update(answers)

    candidate_edges = []
    for u, v in G.edges():
        if (u in answer_nodes or v in answer_nodes) and not (u in seed_nodes and v in seed_nodes):
            candidate_edges.append((u, v))

    n_remove = int(len(candidate_edges) * removal_fraction)
    if n_remove > 0:
        to_remove = rng.sample(candidate_edges, min(n_remove, len(candidate_edges)))
        G.remove_edges_from(to_remove)

    return G

def evaluate_level(
    G_incomplete: nx.Graph,
    qa_pairs: List[Tuple],
    embeddings: Dict[str, np.ndarray],
    cmap_full: Dict[str, int],
    beam_width: int,
    use_rem: bool,
    level: int,
    removal_fraction: float,
) -> Dict:
    reachable_qa = [(s, a, q) for s, a, q in qa_pairs if s in G_incomplete]
    if not reachable_qa:
        return {"level": level, "hits_1": 0.0, "hits_10": 0.0, "mrr": 0.0, "ms_per_q": 0.0}

    adapter = NetworkXAdapter(G_incomplete)
    adapter.embeddings = embeddings
    
    # Community coarsening
    cmap_inc = {n: c for n, c in cmap_full.items() if n in G_incomplete}
    cmap_coarse = coarsen_communities(G_incomplete, cmap_inc, target_max=200)
    adapter.community_map = cmap_coarse

    distances = build_community_distance_matrix(G_incomplete, cmap_coarse)
    adj = adjacent_community_pairs(G_incomplete, cmap_coarse)
    cg = build_community_graph(G_incomplete, cmap_coarse)

    csa = CSAEngine(adapter=adapter)
    csa.set_community_graph(distances, adj, community_graph=cg)
    
    if use_rem and removal_fraction > 0:
        rem = REMEngine(adapter=adapter)
        rem.run(dry_run=False)

    traversal = BeamTraversal(adapter=adapter, csa_engine=csa, beam_width=beam_width, max_hop=3)
    
    h1 = h10 = mrr = 0.0
    t0 = time.time()
    for i, (seed, correct_answers, _) in enumerate(reachable_qa):
        paths = traversal.traverse([seed])
        answers = extract(paths, top_k=10)
        pred = [a.entity_id for a in answers]
        
        if pred:
            correct_set = set(correct_answers)
            h1 += int(pred[0] in correct_set)
            h10 += int(any(p in correct_set for p in pred[:10]))
            mrr += next((1.0/(r+1) for r, p in enumerate(pred) if p in correct_set), 0.0)

    n = len(reachable_qa)
    return {
        "level": level,
        "n": n,
        "hits_1": h1 / n,
        "hits_10": h10 / n,
        "mrr": mrr / n,
        "ms_per_q": (time.time() - t0) * 1000 / n
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--rem", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2, 4])
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    print("Loading MetaQA Graph...")
    G_full = load_metaqa_graph()
    print("Loading MetaQA 3-hop QA...")
    qa_all = load_metaqa_qa(3)
    
    if args.sample:
        qa_all = rng.sample(qa_all, args.sample)

    print(f"Building Embeddings for {G_full.number_of_nodes()} nodes...")
    se = SentenceEngine()
    embeddings = se.encode_entities({n: n for n in G_full.nodes()})

    print("Community Detection...")
    parts = best_of_n_dscf(G_full, n_trials=1)
    cmap_full = {n: cid for cid, members in enumerate(parts) for n in members}

    results = []
    for level in args.levels:
        frac = INCOMPLETENESS_LEVELS[level]
        print(f"Evaluating Level {level} ({frac*100}% removed)...")
        G_inc = apply_incompleteness(G_full, qa_all, frac, rng)
        res = evaluate_level(G_inc, qa_all, embeddings, cmap_full, 10, args.rem, level, frac)
        results.append(res)
        print(f"  H@1: {res['hits_1']:.4f} | H@10: {res['hits_10']:.4f} | MRR: {res['mrr']:.4f}")

    print("\n" + "="*30 + " RESULTS " + "="*30)
    for r in results:
        print(f"Level {r['level']}: H@10={r['hits_10']:.4f} ({r['ms_per_q']:.1f}ms/q)")

if __name__ == "__main__":
    main()

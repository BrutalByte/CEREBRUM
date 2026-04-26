
"""
Phase 141: Autonomous Tuning for H1SE.
Evaluates H1SE parameters against MetaQA to find the optimal Accuracy/Latency trade-off.
"""
import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np

from core.cerebrum import CerebrumGraph
from adapters.networkx_adapter import NetworkXAdapter
from benchmarks.full_system_eval import load_kb_thalamus, evaluate_hop, build_csa, get_communities, get_embeddings_sentence, coarsen_communities, CACHE_DIR
from benchmarks.metaqa_eval import load_qa

_log = logging.getLogger("tune_h1se")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--output", type=str, default="h1se_tuning_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Setup KB through Thalamus (standard ingestion)
    _log.info("Loading KB through Thalamus...")
    adapter_full, G_full, entity_norm = load_kb_thalamus()
    
    _log.info("Preparing communities and embeddings (using caches if available)...")
    cmap_raw = get_communities(G_full, CACHE_DIR / "communities_full_raw.pkl")
    cmap_full = coarsen_communities(G_full, cmap_raw, target_max=300)
    emb_full = get_embeddings_sentence(G_full, CACHE_DIR / "embeddings_sentence.pkl")
    
    csa_full = build_csa(adapter_full, G_full, cmap_full, emb_full)
    
    # 2. Load QA pairs and normalize
    qa_file = Path("benchmarks/data/metaqa") / f"qa_{args.hops}hop.txt"
    raw_qa = load_qa(args.hops, sample=args.sample)
    qa_norm = []
    for qa in raw_qa:
        seed = entity_norm(qa[0])
        ans = [entity_norm(a) for a in qa[1]]
        if seed in G_full:
            qa_norm.append((seed, ans))
    
    _log.info(f"Evaluating {len(qa_norm)} normalized questions.")

    # 3. Parameter Grid
    param_grid = [
        {"expansion_k": 5,  "use_adaptive": True},
        {"expansion_k": 10, "use_adaptive": True},
        {"expansion_k": 20, "use_adaptive": True},
        {"expansion_k": 10, "use_adaptive": False},
    ]

    results = []

    for params in param_grid:
        _log.info(f"Evaluating: {params}")
        
        from reasoning.expanded_traversal import HopExpandedTraversal
        het = HopExpandedTraversal(
            adapter=adapter_full,
            csa_engine=csa_full,
            beam_width=10,
            max_hop=args.hops,
            expansion_k=params["expansion_k"],
            use_adaptive_expansion=params["use_adaptive"],
            probabilistic=True,
            warm_start_strength=3,
        )
        
        t0 = time.perf_counter()
        m = evaluate_hop(args.hops, het, qa_norm, top_k=10, adapter=adapter_full)
        latency = (time.perf_counter() - t0) / len(qa_norm) * 1000
        
        res = {
            "params": params,
            "hits@10": m["hits_10"],
            "latency_ms": m["ms_per_q"],
            "efficiency": m["hits_10"] / np.log(max(2, m["ms_per_q"]))
        }
        results.append(res)
        _log.info(f"Result: Hits@10={res['hits@10']:.4f} Latency={res['latency_ms']:.1f}ms Efficiency={res['efficiency']:.4f}")

    # 4. Save and Report
    best = max(results, key=lambda x: x["efficiency"])
    _log.info(f"WINNER: {best['params']} with Efficiency={best['efficiency']:.4f}")
    
    with open(args.output, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

if __name__ == "__main__":
    main()

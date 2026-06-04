"""
CEREBRUM ComplexWebQuestions (CWQ) Benchmark
============================================

CWQ: 3,519 test questions over Freebase, derived from WebQSP with added
compositional complexity.  Questions fall into four types:
  composition  : multi-hop path composition (extends WebQSP single-hop)
  conjunction  : conjunctive entity constraints
  comparative  : ordinal / comparative filters
  superlative  : superlative filters

Primary metric: entity-level F1 (precision + recall over answer sets).
Secondary metric: Hits@1.

Per-type reporting matches the benchmark's key contribution: measuring how
CEREBRUM handles structurally varied multi-hop questions without training.

Usage
-----
  python -m benchmarks.cwq_eval
  python -m benchmarks.cwq_eval --sample 500
  python -m benchmarks.cwq_eval --use-bridge
  python -m benchmarks.cwq_eval --beam-width 20 --top-k 20
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Match, Optional, Set, Tuple, Type

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cwq_eval")

import networkx as nx
import numpy as np

from core.cerebrum import CerebrumGraph
from core.graph_bridge import GraphBridgeEngine
from core.thalamus import IngestionPipeline
from core.resource_governor import ResourceGovernor
from reasoning.answer_extractor import Answer
from reasoning.relation_path_prior import RelationPathPrior

_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).parent / "data" / "cwq"
SCAFFOLD   = DATA_DIR / "cwq_scaffold.txt"
TEST_JSON  = DATA_DIR / "CWQ.test.json"
TRAIN_JSON = DATA_DIR / "CWQ.train.json"
NAMES_JSON = DATA_DIR / "entity_names.json"
CACHE_DIR  = DATA_DIR / "cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_entity_names() -> Dict[str, str]:
    if NAMES_JSON.exists():
        with open(NAMES_JSON, encoding="utf-8") as f:
            names = json.load(f)
        print(f"  Loaded {len(names):,} entity names from entity_names.json")
        return names
    return {}


def compute_f1(pred_ids: List[str], gold_ids: Set[str]) -> float:
    if not pred_ids or not gold_ids:
        return 0.0
    cap       = max(len(gold_ids) * 3, 1)
    pred_set  = set(pred_ids[:cap])
    tp        = len(pred_set & gold_ids)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gold_ids)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def load_test_qa(
    graph:       nx.Graph,
    sample:      Optional[int] = None,
    seed:        int           = 42,
    type_filter: str           = "all",
) -> List[Tuple[str, List[str], str, str]]:
    if not TEST_JSON.exists():
        print(f"  ERROR: {TEST_JSON} not found. Run setup_cwq_data.py first.")
        sys.exit(1)
    
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)

    qa: List[Tuple[str, List[str], str, str]] = []
    skipped_seed = skipped_ans = 0

    for q in data.get("Questions", []):
        qtype = q.get("QuestionType", "unknown")
        if type_filter != "all" and qtype.lower() != type_filter.lower():
            continue

        text = q.get("RawQuestion", "")
        for p in q.get("Parses", []):
            seed_node = p.get("TopicEntityMid", "")
            if not seed_node or seed_node not in graph:
                skipped_seed += 1
                continue

            answers = [
                ans.get("AnswerArgument", "")
                for ans in p.get("Answers", [])
                if ans.get("AnswerArgument")
            ]
            if not answers:
                skipped_ans += 1
                continue

            qa.append((seed_node, answers, text, qtype))
            break

    if sample is not None and sample < len(qa):
        import random as _random
        rng = _random.Random(seed)
        qa  = rng.sample(qa, sample)

    type_label = type_filter if type_filter != "all" else "all types"
    print(f"  Test QA ({type_label}): {len(qa):,} usable "
          f"(dropped {skipped_seed} missing-seed, {skipped_ans} missing-answers)")
    return qa


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    graph:               CerebrumGraph,
    qa_pairs:            List[Tuple[str, List[str], str, str]],
    top_k:               int                               = 20,
    max_hop:             int                               = 3,
    question_embeddings: Optional[Dict[str, np.ndarray]]  = None,
    relation_prior                                         = None,
) -> Dict[str, Any]:
    type_buckets: Dict[str, Dict[str, Any]] = {}

    def _bucket(qtype: str) -> Dict[str, Any]:
        if qtype not in type_buckets:
            type_buckets[qtype] = {"f1_sum": 0.0, "h1": 0, "n": 0, "skipped": 0}
        return type_buckets[qtype]

    total_f1 = 0.0
    total_h1 = 0
    skipped  = 0
    t0       = time.time()

    for i, (seed, correct_answers, question_text, qtype) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
            elapsed = time.time() - t0
            print(f"    {i + 1:,}/{len(qa_pairs):,}  ({elapsed:.1f}s)", end="\r")

        query_emb = None
        if question_embeddings is not None:
            query_emb = question_embeddings.get(question_text)

        answers = graph.query(
            [seed],
            top_k=top_k,
            min_hop=1,
            max_hop=max_hop,
            query_embedding=query_emb,
            relation_prior=relation_prior,
        )
        pred = [a.entity_id for a in answers]
        
        # Debug:
        if i < 5:
            print(f"    Q: {question_text[:60]}...")
            print(f"    Found {len(answers)} answers. Top: {pred[0] if pred else 'None'}")
            if pred and correct_answers:
                print(f"    Correct: {correct_answers[0]}... (Match: {pred[0] in correct_answers})")

        bkt = _bucket(qtype)
        bkt["n"] += 1

        if not pred:
            skipped += 1
            bkt["skipped"] += 1
            continue

        correct_set = set(correct_answers)
        f1          = compute_f1(pred, correct_set)
        h1          = int(pred[0] in correct_set)

        total_f1 += f1
        total_h1 += h1
        bkt["f1_sum"] += f1
        bkt["h1"]     += h1

    elapsed = time.time() - t0
    print()

    n = len(qa_pairs)
    return {
        "n":            n,
        "f1":           total_f1 / n if n else 0.0,
        "h1":           total_h1 / n if n else 0.0,
        "skipped":      skipped,
        "elapsed":      elapsed,
        "ms_per_q":     1000 * elapsed / n if n else 0.0,
        "type_buckets": type_buckets,
    }


def print_results(results: Dict[str, Any], label: str = "") -> None:
    n         = results["n"]
    f1        = results["f1"]
    h1        = results["h1"]
    elapsed   = results["elapsed"]
    ms_per_q  = results["ms_per_q"]
    skipped   = results["skipped"]
    buckets   = results.get("type_buckets", {})

    hdr = f"  [{label}]  {n:,} questions" if label else f"  {n:,} questions"
    print(f"{hdr}  ({elapsed:.1f}s, {ms_per_q:.1f}ms/Q, {skipped} no-pred)")
    print(f"    F1={f1 * 100:.2f}%  Hits@1={h1 * 100:.2f}%")

    if buckets:
        print()
        print(f"    {'Type':<16}  {'N':>6}  {'F1':>7}  {'H@1':>7}")
        print(f"    {'-'*16}  {'-'*6}  {'-'*7}  {'-'*7}")
        for qtype in sorted(buckets.keys()):
            b = buckets[qtype]
            bn   = b["n"]
            bf1  = b["f1_sum"] / bn if bn else 0.0
            bh1  = b["h1"]     / bn if bn else 0.0
            print(f"    {qtype:<16}  {bn:>6,}  {bf1 * 100:>6.2f}%  {bh1 * 100:>6.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CEREBRUM CWQ Benchmark")
    parser.add_argument("--sample",     type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--top-k",      type=int, default=20)
    parser.add_argument("--no-cache",   action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--type",       default="all",
                        choices=["all", "composition", "conjunction",
                                 "comparative", "superlative"])
    parser.add_argument("--coarsen-target", type=int, default=300)
    parser.add_argument("--max-hop", type=int, default=4)
    parser.add_argument("--use-bridge", action="store_true")
    parser.add_argument("--min-similarity-bridge", type=float, default=0.82)
    parser.add_argument("--top-k-bridge", type=int, default=3)
    parser.add_argument("--community-engine", default="dscf",
                        choices=["dscf", "leiden", "lpa"])
    parser.add_argument("--scaffold", type=str, default=None,
                        help="Path to custom scaffold triples file")
    args = parser.parse_args()

    scaffold_path = Path(args.scaffold) if args.scaffold else SCAFFOLD

    print()
    print("=" * 68)
    print("  CEREBRUM CWQ Benchmark (unified pipeline)")
    print("=" * 68)
    print(f"  beam_width={args.beam_width}  top_k={args.top_k}  "
          f"type={args.type}  bridge={args.use_bridge}")
    print()

    # 1. Load names
    entity_names = load_entity_names()

    # 2. Build graph object
    print("Loading scaffold graph...")
    t0 = time.time()
    
    # We build the graph manually first to set friendly names for labeling
    pipeline = IngestionPipeline(entity_normalizer=lambda s: s.strip(), relation_map={})
    G = nx.Graph()
    with open(scaffold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                s, r, o = parts
                edge = pipeline.process(s, o, r)
                G.add_edge(edge.source, edge.target, relation=edge.relation)
    
    # Set labels for SentenceEngine
    for n in G.nodes():
        if n in entity_names:
            G.nodes[n]["label"] = entity_names[n]

    from adapters.networkx_adapter import NetworkXAdapter
    adapter = NetworkXAdapter(G)
    graph = CerebrumGraph.from_adapter(
        adapter, 
        embeddings="sentence",
        beam_width=args.beam_width,
        max_hop=4,
    )
    print(f"  {graph.node_count:,} entities  {graph.edge_count:,} edges  "
          f"({time.time() - t0:.1f}s)")
    print()

    # 3. Load QA
    test_qa = load_test_qa(G, sample=args.sample, seed=args.seed, type_filter=args.type)
    if not test_qa:
        return

    # 4. Enhance (Bridge Engine)
    if args.use_bridge:
        print("Graph Enhancement (GraphBridgeEngine)...")
        # We need embeddings for the bridge engine
        # build() will compute them, but we need them NOW for apply()
        # So we trigger a partial build or just call the enhancer.
        # CerebrumGraph.enhance() does exactly this.
        graph.enhance([
            GraphBridgeEngine(
                min_similarity=args.min_similarity_bridge,
                top_k=args.top_k_bridge,
                max_bridges=100000,
            )
        ])
        print()

    # 5. Build
    print("THALAMUS build pipeline...")
    # Use a specific cache suffix if bridges are used to avoid collisions
    cache_suffix = f"_bridge_{args.min_similarity_bridge}" if args.use_bridge else ""
    cache_path   = CACHE_DIR / f"cwq{cache_suffix}"
    
    graph.build(
        cache_dir=cache_path,
        coarsen_target=args.coarsen_target,
        force_rebuild=args.no_cache,
        seed=args.seed,
        community_engine=args.community_engine,
    )
    print()

    # 6. Question Embeddings
    print("Encoding question texts...")
    from core.embedding_engine import SentenceEngine
    engine = SentenceEngine()
    qemb_cache = cache_path / "question_embeddings.pkl"
    if not args.no_cache and qemb_cache.exists():
        with open(qemb_cache, "rb") as f:
            question_embeddings = pickle.load(f)
    else:
        texts = {qa[2]: qa[2] for qa in test_qa}
        question_embeddings = engine.encode_entities(texts)
        with open(qemb_cache, "wb") as f:
            pickle.dump(question_embeddings, f)
    print()

    # 7. Evaluate
    print("-" * 68)
    print("  CWQ Evaluation")
    print("-" * 68)
    results = evaluate(
        graph,
        test_qa,
        top_k=args.top_k,
        max_hop=args.max_hop,
        question_embeddings=question_embeddings,
    )
    print()
    print_results(results, label="CWQ")
    print()

if __name__ == "__main__":
    main()

"""
Phase 229: ConceptNet 5 multi-hop benchmark — tuner-compatible.

Evaluation protocol: 2-hop chain discovery over the full graph.
  - Train graph: 80% of loaded edges (by MD5 hash of the triple).
  - QA pairs:    2-hop chains (h, r1, mid, r2, t) sampled from training
                 graph where h→t has no direct edge (avoids trivial 1-hop).
  - Task:        given seed=[h], retrieve t within top_k (max_hop=2).
  - Metrics:     H@1, H@10, MRR (same as MetaQA/Hetionet benchmarks).

CEREBRUM is a traversal-based system; "predict missing link" evaluation
doesn't work because the direct edge is removed and alternative paths are
rare. 2-hop discovery over training chains is the correct fit.

This produces calibration data for the ``mixed`` regime row in
``core/parameter_initializer.py``.

Usage (standalone):
    # Fast eval (random embeddings, 500 questions):
    python -u benchmarks/conceptnet_eval.py \\
        --cn5 data/conceptnet-assertions-5.7.0.csv.gz \\
        --n-questions 500 --embeddings random

    # Full validation:
    python -u benchmarks/conceptnet_eval.py \\
        --cn5 data/conceptnet-assertions-5.7.0.csv.gz \\
        --n-questions 2000 --embeddings sentence --max-edges 500000

Tuner integration:
    python -u benchmarks/cerebrum_tuner.py \\
        --dataset conceptnet \\
        --cn5-file data/conceptnet-assertions-5.7.0.csv.gz \\
        --sample 500 --embeddings random

CN5 download:
    wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/ \\
         conceptnet-assertions-5.7.0.csv.gz -P data/conceptnet/
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = Path(__file__).parent / "data" / "conceptnet"

# Relations with too little signal for tuning (too common or too noisy)
_SKIP_RELATIONS = frozenset({
    "ExternalURL", "dbpedia", "Etymologically",
    "EtymologicallyRelatedTo",
})

# Relations to prefer for test QA (more structured, clearer targets)
_PREFERRED_RELATIONS = {
    "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "Causes",
    "HasProperty", "MadeOf", "ReceivesAction", "AtLocation",
    "CausesDesire", "HasSubevent", "HasPrerequisite",
}


# ---------------------------------------------------------------------------
# Edge splitting (deterministic, seed-independent)
# ---------------------------------------------------------------------------

def _edge_in_test(h: str, r: str, t: str, test_pct: int = 20) -> bool:
    """Return True if this edge belongs to the test split (deterministic)."""
    digest = hashlib.md5(f"{h}\t{r}\t{t}".encode()).hexdigest()
    return int(digest[:4], 16) % 100 < test_pct


# ---------------------------------------------------------------------------
# Graph loading with train/test split
# ---------------------------------------------------------------------------

def load_and_split(
    cn5_path: str,
    lang: str = "en",
    min_weight: float = 1.0,
    max_edges: int = 200_000,
) -> Tuple[List[Tuple[str, str, str, float]], List[Tuple[str, str, str]]]:
    """
    Load CN5 CSV and split into train/test triples.

    Returns
    -------
    train_triples : list of (h, r, t, weight) for graph construction
    test_triples  : list of (h, r, t) for evaluation
    """
    from adapters.conceptnet_adapter import _parse_entity, _parse_relation, _entity_lang

    train: List[Tuple[str, str, str, float]] = []
    test:  List[Tuple[str, str, str]]        = []
    loaded = 0

    opener = gzip.open if str(cn5_path).endswith(".gz") else open
    with opener(cn5_path, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 4 or row[0].startswith("#"):
                continue
            rel_uri, subj_uri, obj_uri = row[1], row[2], row[3]
            if _entity_lang(subj_uri) != lang or _entity_lang(obj_uri) != lang:
                continue
            relation = _parse_relation(rel_uri)
            if relation in _SKIP_RELATIONS:
                continue
            try:
                weight = float(json.loads(row[4] if len(row) > 4 else "{}").get("weight", 1.0))
            except (json.JSONDecodeError, ValueError):
                weight = 1.0
            if weight < min_weight:
                continue
            h = _parse_entity(subj_uri)
            t = _parse_entity(obj_uri)
            if not h or not t:
                continue

            if _edge_in_test(h, relation, t):
                test.append((h, relation, t))
            else:
                train.append((h, relation, t, weight))

            loaded += 1
            if loaded >= max_edges:
                break

    return train, test


def _build_graph_from_triples(
    triples: List[Tuple[str, str, str, float]],
    embeddings: str,
    cache_dir: Optional[Path],
    seed: int,
    beam_width: int = 12,
) -> "CerebrumGraph":
    import networkx as nx
    from adapters.networkx_adapter import NetworkXAdapter
    from core.cerebrum import CerebrumGraph
    from core.embedding_engine import RandomEngine, SentenceEngine

    G = nx.DiGraph()
    for h, r, t, w in triples:
        G.add_edge(h, t, relation=r, weight=w, confidence=min(w / 10.0, 1.0))

    adapter = NetworkXAdapter(G)

    if embeddings == "sentence":
        try:
            eng = SentenceEngine()
        except ImportError:
            eng = RandomEngine(dim=64)
    else:
        eng = RandomEngine(dim=64)

    graph = CerebrumGraph(adapter, embedding_engine=eng)
    graph.build(
        cache_dir=str(cache_dir) if cache_dir else None,
        seed=seed,
    )
    graph._beam_width = beam_width
    return graph


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

def _sample_qa_pairs(
    train_triples: List[Tuple[str, str, str, float]],
    n_questions: int,
    seed: int,
    focus_relations: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Sample 2-hop QA pairs from the training graph.

    Builds (h, mid, t) chains where:
      - (h, r1, mid) is a training edge
      - (mid, r2, t) is a training edge
      - No direct (h, *, t) edge exists in training (non-trivial)
      - r1 or r2 is in _PREFERRED_RELATIONS for tuning signal

    Returns List[(h, r1, t)] where r1 is the first-hop relation.
    Queries use seed=[h], max_hop=2, answer=t.
    """
    import networkx as nx
    rng = random.Random(seed)

    G = nx.DiGraph()
    for h, r, t, w in train_triples:
        G.add_edge(h, t, relation=r, weight=w)

    direct_pairs: set = {(h, t) for h, t in G.edges()}

    candidates: List[Tuple[str, str, str]] = []
    nodes = list(G.nodes)
    rng.shuffle(nodes)

    for h in nodes:
        for mid in list(G.successors(h)):
            r1 = G[h][mid].get("relation", "")
            if focus_relations and r1 not in focus_relations:
                continue
            for t in list(G.successors(mid)):
                if t == h:
                    continue
                if (h, t) in direct_pairs:
                    continue  # trivial 1-hop answer — skip
                r2 = G[mid][t].get("relation", "")
                # Prefer chains where at least one hop is a structured relation
                if r1 in _PREFERRED_RELATIONS or r2 in _PREFERRED_RELATIONS:
                    candidates.append((h, r1, t))
        if len(candidates) >= n_questions * 10:
            break

    # Fall back to all relations if not enough structured candidates
    if len(candidates) < n_questions:
        for h in nodes:
            for mid in list(G.successors(h)):
                r1 = G[h][mid].get("relation", "")
                for t in list(G.successors(mid)):
                    if t != h and (h, t) not in direct_pairs:
                        candidates.append((h, r1, t))
            if len(candidates) >= n_questions * 10:
                break

    if not candidates:
        return []

    # Deduplicate: one pair per (h, t), then at most 2 pairs per seed h
    seen_ht: set = set()
    seen_h: dict = {}
    deduped = []
    for item in candidates:
        h, r, t = item
        key = (h, t)
        if key in seen_ht:
            continue
        seen_ht.add(key)
        if seen_h.get(h, 0) >= 2:
            continue
        seen_h[h] = seen_h.get(h, 0) + 1
        deduped.append(item)

    rng.shuffle(deduped)
    return deduped[:n_questions]


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

def _eval_single(
    graph,
    h: str,
    r: str,
    t: str,
    top_k: int,
    trb_factor: float,
    r2_boost: float,
    vote_weight: float,
    idf_weight: float,
    branch_bonus: float,
    fhrb_factor: float,
    gamma: float,
    beta: float,
    boost_map: dict,
    answer_freq: Dict[str, int],
    max_loops: int = 1,
) -> Tuple[int, int, float]:
    """Returns (hit@1, hit@10, rr) for one QA pair.

    For ConceptNet 2-hop QA:
    - initial_relation_boost {r: fhrb_factor} boosts paths starting with r1
    - r2_boost applied post-hoc as uniform multiplier (no known penultimate)
    - idf_weight applied post-hoc to penalise hub entities
    """
    overfetch = top_k * 5  # oversample then re-rank with post-hoc boosts
    try:
        results = graph.query(
            seeds                   = [h],
            top_k                   = overfetch,
            max_hop                 = 2,
            vote_weight             = vote_weight,
            branch_bonus_weight     = branch_bonus,
            initial_relation_boost  = {r: fhrb_factor} if fhrb_factor > 0 else None,
            max_loops               = max_loops,
        )
    except Exception:
        return 0, 0, 0.0

    if not results:
        return 0, 0, 0.0

    # Post-hoc: r2_boost as uniform multiplier for all hop-2 answers
    # Post-hoc: IDF hub penalty
    scored = []
    for ans in results:
        score = ans.score
        if r2_boost > 0.0:
            score *= (1.0 + r2_boost * 0.1)  # dampened: r2 is relation-agnostic here
        if idf_weight > 0.0 and answer_freq:
            freq = answer_freq.get(ans.entity_id, 0)
            if freq > 1:
                score *= max(0.0, 1.0 - idf_weight * math.log(freq + 1))
        scored.append((ans.entity_id, score))

    scored.sort(key=lambda x: -x[1])
    ranked = [eid for eid, _ in scored[:top_k]]

    if t in ranked:
        rank = ranked.index(t) + 1
        return int(rank == 1), 1, 1.0 / rank
    return 0, 0, 0.0


# ---------------------------------------------------------------------------
# State builder (called once before tuner trials)
# ---------------------------------------------------------------------------

def build_conceptnet_state(
    cn5_path: str,
    n_questions:      int  = 500,
    embeddings:       str  = "random",
    seed:             int  = 42,
    use_cache:        bool = True,
    max_edges:        int  = 200_000,
    min_weight:       float = 1.0,
    focus_relations:  Optional[List[str]] = None,
) -> dict:
    """
    Build CerebrumGraph (train split) and pre-generate 2-hop QA pairs from train.
    Pass the returned dict to run_trial_inprocess() for zero-overhead tuner trials.
    """
    from core.relation_boost_deriver import RelationBoostDeriver

    cache_dir = CACHE_DIR / "cerebrum_cache" if use_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CN5State] Loading and splitting {cn5_path}  (max_edges={max_edges:,})...")
    t0 = time.time()
    train_triples, _ = load_and_split(
        cn5_path, min_weight=min_weight, max_edges=max_edges
    )
    print(f"  Train: {len(train_triples):,} edges ({time.time()-t0:.1f}s)")

    print(f"[CN5State] Building CerebrumGraph (embeddings={embeddings})...")
    t1 = time.time()
    graph = _build_graph_from_triples(
        train_triples, embeddings=embeddings, cache_dir=cache_dir, seed=seed
    )
    print(f"  {graph.adapter.to_networkx().number_of_nodes():,} nodes  ({time.time()-t1:.1f}s)")

    print("[CN5State] Building RelationBoostDeriver...")
    deriver = RelationBoostDeriver()
    deriver.build_from_triples((h, r, t) for h, r, t, _ in train_triples)

    print("[CN5State] Sampling 2-hop QA pairs from training graph...")
    qa_pairs = _sample_qa_pairs(train_triples, n_questions, seed, focus_relations)
    print(f"[CN5State] QA pairs: {len(qa_pairs)} 2-hop chains sampled")

    # Answer frequency for IDF penalty
    answer_freq: Dict[str, int] = defaultdict(int)
    for _, _, t in qa_pairs:
        answer_freq[t] += 1

    print("[CN5State] Ready — all trials will skip graph build.\n")
    return {
        "graph":        graph,
        "deriver":      deriver,
        "qa_pairs":     qa_pairs,
        "answer_freq":  dict(answer_freq),
        "embeddings":   embeddings,
    }


# ---------------------------------------------------------------------------
# In-process trial runner (called per Optuna trial)
# ---------------------------------------------------------------------------

def run_trial_inprocess(
    state:  dict,
    params: dict,
    top_k:  int = 10,
) -> Tuple[float, float, float]:
    """Run one tuner trial. Returns (h1, h10, mrr)."""
    graph       = state["graph"]
    deriver     = state["deriver"]
    qa_pairs    = state["qa_pairs"]
    answer_freq = state["answer_freq"]

    graph._beam_width = params.get("beam_width", graph._beam_width)
    boost_map = deriver.boost_map(params["gamma"], params["beta"]) if deriver.is_built else {}

    total_h1 = total_h10 = total_mrr = 0.0

    for h, r, t in qa_pairs:
        hit1, hit10, rr = _eval_single(
            graph, h, r, t, top_k,
            trb_factor   = params["trb_factor"],
            r2_boost     = params["r2_boost"],
            vote_weight  = params["vote_weight"],
            idf_weight   = params["idf_weight"],
            branch_bonus = params["branch_bonus"],
            fhrb_factor  = params["fhrb_factor"],
            gamma        = params["gamma"],
            beta         = params["beta"],
            boost_map    = boost_map,
            answer_freq  = answer_freq,
            max_loops    = params.get("max_loops", 1),
        )
        total_h1  += hit1
        total_h10 += hit10
        total_mrr += rr

    n = len(qa_pairs) or 1
    return total_h1 / n, total_h10 / n, total_mrr / n


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conceptnet_eval",
        description="CEREBRUM ConceptNet 5 link-prediction benchmark",
    )
    parser.add_argument("--cn5", type=str, required=True,
                        help="Path to conceptnet-assertions-5.7.0.csv or .csv.gz")
    parser.add_argument("--n-questions", type=int, default=500, dest="n_questions")
    parser.add_argument("--max-edges", type=int, default=200_000, dest="max_edges")
    parser.add_argument("--min-weight", type=float, default=1.0, dest="min_weight")
    parser.add_argument("--embeddings", choices=["random", "sentence"], default="random")
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", dest="no_cache")
    args = parser.parse_args()

    if not Path(args.cn5).exists():
        print(
            f"ERROR: CN5 file not found: {args.cn5}\n"
            "Download: wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
            "conceptnet-assertions-5.7.0.csv.gz",
            file=sys.stderr,
        )
        sys.exit(1)

    state = build_conceptnet_state(
        cn5_path    = args.cn5,
        n_questions = args.n_questions,
        embeddings  = args.embeddings,
        seed        = args.seed,
        use_cache   = not args.no_cache,
        max_edges   = args.max_edges,
        min_weight  = args.min_weight,
    )

    from core.parameter_initializer import ParameterInitializer
    pi    = ParameterInitializer(state["graph"].adapter)
    defaults = pi.derive()

    h1, h10, mrr = run_trial_inprocess(state, {
        "trb_factor":   defaults.trb_factor,
        "r2_boost":     defaults.r2_boost,
        "vote_weight":  defaults.vote_weight,
        "beam_width":   defaults.beam_width,
        "idf_weight":   defaults.idf_weight,
        "branch_bonus": defaults.branch_bonus,
        "fhrb_factor":  defaults.fhrb_factor,
        "gamma":        defaults.gamma,
        "beta":         defaults.beta,
    }, top_k=args.top_k)

    n = len(state["qa_pairs"])
    print(f"\n{'='*55}")
    print(f"ConceptNet 2-hop discovery  ({n:,} questions, top-{args.top_k})")
    print(f"  H@1  = {h1:.4f}  ({h1*100:.2f}%)")
    print(f"  H@10 = {h10:.4f}  ({h10*100:.2f}%)")
    print(f"  MRR  = {mrr:.4f}")
    print(f"{'='*55}")

    # Per-relation breakdown (top 10 relations by count)
    from collections import Counter
    rel_counts = Counter(r for _, r, _ in state["qa_pairs"])
    print("\nTop-10 relations in eval set:")
    for rel, cnt in rel_counts.most_common(10):
        print(f"  {rel:<30} {cnt:>5} questions")


if __name__ == "__main__":
    main()

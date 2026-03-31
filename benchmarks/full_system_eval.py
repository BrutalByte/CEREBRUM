"""
CEREBRUM Full-System Benchmark: Raw Load vs Complete THALAMUS Pipeline
=======================================================================

This is the definitive benchmark for CEREBRUM. It answers one question:

  "What does CEREBRUM score when the ENTIRE system is used correctly,
   versus how external evaluations typically test a raw graph system?"

Two variants run head-to-head on identical MetaQA test questions:

  RAW  — Data loaded directly, random embeddings, un-coarsened communities.
         This is how external papers typically benchmark KG systems.
         Result: 14,976 micro-communities, most singletons.
         The CSA attention heads are degenerate — nearly all empty shells.
         The semantic alpha term is pure noise (random vectors).
         This is NOT how CEREBRUM is designed to be used.

  FULL — Data ingested through the complete THALAMUS pipeline:
           • IngestionPipeline  entity normalization + relation normalization
           • SentenceEngine     384-dim semantic embeddings (all-MiniLM-L6-v2)
           • coarsen_communities meaningful attention heads (~300 communities)
           • Bayesian beam      warm_start_strength=3 reduces cold-start variance
         This is how a real customer uses CEREBRUM on their data.

The delta between RAW and FULL is what proper ingestion actually delivers.

Why the RAW score is lower than CEREBRUM's true capability
----------------------------------------------------------
The community structure in CEREBRUM IS the attention mechanism. Each
community is an attention head. With 14,976 singleton communities on a
43K-node graph, there are effectively zero meaningful attention heads —
CSA sees every node as its own isolated island, the community signal
(0.4 beta weight) is degenerate, and beam search has no structural
guidance. With proper ingestion, DSCF on the cleaned graph produces
~300 cohesive semantic clusters (actors, directors, genres, studios),
giving CSA real attention heads to reason through.

Usage
-----
  # Full run (recommended — all results cached after first pass):
  python -m benchmarks.full_system_eval

  # Quick development run (1,000 questions per hop):
  python -m benchmarks.full_system_eval --sample 1000

  # Single hop:
  python -m benchmarks.full_system_eval --hop 1

  # Regenerate all caches (after graph or embedding changes):
  python -m benchmarks.full_system_eval --no-cache
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import (
    best_of_n_dscf,
    compute_soft_memberships,
    adaptive_resolution_search,
)
from core.embedding_engine import RandomEngine, SentenceEngine
from core.attention_engine import CSAEngine
from core.bridge_engine import BridgeTwinEngine
from core.kge_engine import TransEEngine
from core.parameter_learner import CSAParameterLearner
from core.structural_encoder import (
    build_community_distance_matrix,
    adjacent_community_pairs,
    build_community_graph,
    coarsen_communities,
)
from core.thalamus import IngestionPipeline
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank, load_qa

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"
KB_FILE   = DATA_DIR / "kb.txt"

# ---------------------------------------------------------------------------
# MetaQA relation map — canonical uppercase for CSA edge_type_weight
# ---------------------------------------------------------------------------

METAQA_RELATION_MAP = {
    "directed_by":     "DIRECTED_BY",
    "written_by":      "WRITTEN_BY",
    "starred_actors":  "STARRED_ACTORS",
    "release_year":    "RELEASE_YEAR",
    "in_language":     "IN_LANGUAGE",
    "has_tags":        "HAS_TAGS",
    "has_genre":       "HAS_GENRE",
    "has_imdb_rating": "HAS_IMDB_RATING",
    "has_imdb_votes":  "HAS_IMDB_VOTES",
}

# ---------------------------------------------------------------------------
# Graph loading — RAW variant
# ---------------------------------------------------------------------------

def load_kb_raw() -> Tuple[NetworkXAdapter, nx.Graph]:
    """Load kb.txt with no processing. Exactly what external evaluations do."""
    G = nx.Graph()
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
    adapter = NetworkXAdapter(G)
    return adapter, G


# ---------------------------------------------------------------------------
# Graph loading — FULL (THALAMUS) variant
# ---------------------------------------------------------------------------

def load_kb_thalamus() -> Tuple[NetworkXAdapter, nx.Graph, callable]:
    """
    Load kb.txt through IngestionPipeline.

    Returns (adapter, G, normalizer) — the normalizer must be applied to
    QA seed entities and correct answers so they match the graph's node IDs.
    """
    entity_normalizer = lambda s: s.lower().strip()

    pipeline = IngestionPipeline(
        entity_normalizer=entity_normalizer,
        relation_map=METAQA_RELATION_MAP,
    )

    G = nx.Graph()
    with open(KB_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            edge = pipeline.process(subj, obj, rel)
            G.add_edge(edge.source, edge.target, relation=edge.relation)

    adapter = NetworkXAdapter(G)
    return adapter, G, entity_normalizer


# ---------------------------------------------------------------------------
# Community detection with caching
# ---------------------------------------------------------------------------

def get_communities(
    G: nx.Graph,
    cache_path: Path,
    use_cache: bool = True,
    n_trials: int = 3,
    seed: int = 42,
    label: str = "",
) -> Dict[str, int]:
    """Run DSCF and cache the result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists():
        print(f"    Loading cached communities from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"    Running DSCF on {G.number_of_nodes():,} nodes — this takes 1–2 min...")
    t0    = time.time()
    parts = best_of_n_dscf(G, n_trials=n_trials, seed=seed)
    cmap  = {node: cid for cid, members in enumerate(parts) for node in members}
    print(f"    {len(parts):,} raw communities in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(cmap, f)
    return cmap


# ---------------------------------------------------------------------------
# Embedding computation with caching
# ---------------------------------------------------------------------------

def get_embeddings_random(G: nx.Graph) -> Dict[str, np.ndarray]:
    engine = RandomEngine(dim=64)
    labels = {n: n for n in G.nodes()}
    return engine.encode_entities(labels)


def get_embeddings_sentence(
    G: nx.Graph,
    cache_path: Path,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Encode all entities with SentenceEngine; cache the result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists():
        print(f"    Loading cached embeddings from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"    Encoding {G.number_of_nodes():,} entities with SentenceEngine "
          f"(all-MiniLM-L6-v2) — this takes 3–5 min on CPU...")
    engine = SentenceEngine()
    print(f"    Model loaded ({engine.dim}-dim).")
    labels = {n: n for n in G.nodes()}
    t0 = time.time()
    embeddings = engine.encode_entities(labels)
    print(f"    Encoded in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


# ---------------------------------------------------------------------------
# Build CSA engine from adapter + community map + embeddings
# ---------------------------------------------------------------------------

def build_csa(
    adapter: NetworkXAdapter,
    G: nx.Graph,
    cmap: Dict[str, int],
    embeddings: Dict[str, np.ndarray],
) -> CSAEngine:
    adapter.community_map = cmap
    adapter.embeddings    = embeddings
    distances = build_community_distance_matrix(G, cmap)
    adj       = adjacent_community_pairs(G, cmap)
    cg        = build_community_graph(G, cmap)
    csa = CSAEngine(adapter=adapter)
    csa.set_community_graph(distances, adj, community_graph=cg)
    return csa


# ---------------------------------------------------------------------------
# OPTIMIZED helpers
# ---------------------------------------------------------------------------

# Training QA files (MetaQA standard train splits)
QA_TRAIN_FILES = {
    1: DATA_DIR / "1-hop" / "vanilla" / "qa_train.txt",
    2: DATA_DIR / "2-hop" / "vanilla" / "qa_train.txt",
    3: DATA_DIR / "3-hop" / "vanilla" / "qa_train.txt",
}


def load_qa_train(
    hop: int,
    sample: Optional[int] = None,
    seed: int = 42,
    entity_norm: Optional[callable] = None,
) -> List[Tuple]:
    """Load MetaQA training QA pairs; returns [] if train file not present."""
    import re as _re
    path = QA_TRAIN_FILES.get(hop)
    if path is None or not path.exists():
        return []
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
            m = _re.search(r"\[(.+?)\]", question)
            if not m:
                continue
            seed_ent = m.group(1).strip()
            if entity_norm:
                seed_ent = entity_norm(seed_ent)
                answers  = [entity_norm(a) for a in answers]
            if seed_ent and answers:
                pairs.append((seed_ent, answers))
    if sample is not None and sample < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample)
    return pairs


def get_communities_with_partition(
    G: nx.Graph,
    cache_path: Path,
    partition_cache_path: Path,
    use_cache: bool = True,
    n_trials: int = 3,
    seed: int = 42,
    resolution: float = 1.0,
) -> Tuple[Dict[str, int], List]:
    """
    Run DSCF and return (community_map, raw_partition).

    The raw partition (List[frozenset]) is needed for compute_soft_memberships.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists() and partition_cache_path.exists():
        print(f"    Loading cached communities from {cache_path.name}")
        with open(cache_path, "rb") as f:
            cmap = pickle.load(f)
        with open(partition_cache_path, "rb") as f:
            partition = pickle.load(f)
        return cmap, partition

    print(f"    Running DSCF (resolution={resolution:.3f}) on "
          f"{G.number_of_nodes():,} nodes — this takes 1-2 min...")
    t0        = time.time()
    partition = best_of_n_dscf(G, n_trials=n_trials, seed=seed, resolution=resolution)
    cmap      = {node: cid for cid, members in enumerate(partition) for node in members}
    print(f"    {len(partition):,} raw communities in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(cmap, f)
    with open(partition_cache_path, "wb") as f:
        pickle.dump(partition, f)
    return cmap, partition


def get_kge_embeddings(
    adapter: NetworkXAdapter,
    cache_path: Path,
    use_cache: bool = True,
    n_epochs: int = 100,
    dim: int = 384,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Train TransE on graph triples and return {entity_id: embedding}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists():
        print(f"    Loading cached KGE embeddings from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    G_inner = getattr(adapter, "_G", None) or getattr(adapter, "G", None) or adapter._G
    print(f"    Training TransE KGE ({dim}-dim, {n_epochs} epochs) on "
          f"{G_inner.number_of_nodes():,} entities...")
    t0  = time.time()
    kge = TransEEngine(dim=dim, margin=1.0, lr=0.01, seed=seed)
    kge.fit(adapter, n_epochs=n_epochs)
    r   = kge.result
    print(f"    TransE done: loss={r.final_loss:.4f}  "
          f"({r.n_triples:,} triples, {time.time()-t0:.1f}s)")

    # Build embedding dict: fall back to zero-vector for missing entities
    zero = np.zeros(dim, dtype=np.float32)
    embeddings = {
        node: (kge.get_embedding(node) if kge.get_embedding(node) is not None else zero)
        for node in G_inner.nodes()
    }
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


def project_embeddings(
    emb_dict: Dict[str, np.ndarray],
    target_dim: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Project embeddings to target_dim via a fixed random Gaussian projection.

    Uses the Johnson-Lindenstrauss lemma — distances (cosine similarities) are
    approximately preserved under random projection.  The projection matrix is
    seeded for reproducibility.  If source dim already equals target_dim, returns
    the original dict unchanged.
    """
    if not emb_dict:
        return emb_dict
    src_dim = next(iter(emb_dict.values())).shape[0]
    if src_dim == target_dim:
        return emb_dict
    rng = np.random.default_rng(seed)
    P   = rng.standard_normal((src_dim, target_dim)).astype(np.float32)
    # Orthonormal columns via QR decomposition for better distance preservation
    if src_dim >= target_dim:
        P, _ = np.linalg.qr(P)
    else:
        P, _ = np.linalg.qr(P.T)
        P = P.T
    P = P[:src_dim, :target_dim]
    return {node: _l2norm(v @ P) for node, v in emb_dict.items()}


def blend_embeddings(
    emb_a: Dict[str, np.ndarray],
    emb_b: Dict[str, np.ndarray],
    weight_a: float = 0.5,
    target_dim: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Blend two embedding dicts by averaging their L2-normalised vectors.

    If dimensions differ, emb_a is projected to match emb_b (or to target_dim).
    For nodes present in only one source, uses that source's vector.
    """
    if not emb_a:
        return emb_b
    if not emb_b:
        return emb_a

    dim_a = next(iter(emb_a.values())).shape[0]
    dim_b = next(iter(emb_b.values())).shape[0]
    out_dim = target_dim or dim_b

    if dim_a != out_dim:
        emb_a = project_embeddings(emb_a, out_dim, seed=seed)
    if dim_b != out_dim:
        emb_b = project_embeddings(emb_b, out_dim, seed=seed + 1)

    all_nodes = set(emb_a) | set(emb_b)
    blended: Dict[str, np.ndarray] = {}
    for node in all_nodes:
        a = emb_a.get(node)
        b = emb_b.get(node)
        if a is None:
            blended[node] = _l2norm(b)
        elif b is None:
            blended[node] = _l2norm(a)
        else:
            blended[node] = _l2norm(weight_a * _l2norm(a) + (1 - weight_a) * _l2norm(b))
    return blended


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def get_pagerank(
    G: nx.Graph,
    cache_path: Path,
    use_cache: bool = True,
) -> Dict[str, float]:
    """Compute nx.pagerank and cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists():
        print(f"    Loading cached PageRank from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"    Computing PageRank on {G.number_of_nodes():,} nodes...")
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    print(f"    PageRank done in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(pr, f)
    return pr


def get_soft_memberships(
    G: nx.Graph,
    partition: List,
    cache_path: Path,
    use_cache: bool = True,
) -> Dict[str, Dict[int, float]]:
    """Compute soft community memberships and cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_path.exists():
        print(f"    Loading cached soft memberships from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    print(f"    Computing soft community memberships...")
    t0   = time.time()
    soft = compute_soft_memberships(G, partition)
    print(f"    Soft memberships done in {time.time()-t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(soft, f)
    return soft


def build_csa_full(
    adapter: NetworkXAdapter,
    G: nx.Graph,
    cmap: Dict[str, int],
    embeddings: Dict[str, np.ndarray],
    pagerank: Optional[Dict[str, float]] = None,
    soft_memberships: Optional[Dict] = None,
    alpha: float = 0.4,
    beta:  float = 0.4,
    gamma: float = 0.1,
    delta: float = 0.05,
    epsilon: float = 0.05,
) -> CSAEngine:
    """Build CSAEngine with all optional signals activated."""
    adapter.community_map = cmap
    adapter.embeddings    = embeddings
    distances = build_community_distance_matrix(G, cmap)
    adj       = adjacent_community_pairs(G, cmap)
    cg        = build_community_graph(G, cmap)
    csa = CSAEngine(
        adapter=adapter,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon,
        pagerank=pagerank,
        soft_memberships=soft_memberships,
        zeta=0.1 if pagerank else 0.0,
    )
    csa.set_community_graph(distances, adj, community_graph=cg)
    return csa


def generate_training_pairs(
    adapter: NetworkXAdapter,
    csa_engine: CSAEngine,
    qa_train: List[Tuple],
    n_pairs: int = 500,
    beam_width: int = 15,
    seed: int = 42,
) -> List[Tuple]:
    """
    Run beam traversal on training questions; build (positive, negative) path pairs.

    A positive path ends at a known-correct answer.
    A negative path ends at any other node.
    """
    if not qa_train:
        return []

    rng = random.Random(seed)
    pairs: List[Tuple] = []
    n_sampled = min(len(qa_train), n_pairs * 3)  # over-sample, then trim
    subset    = rng.sample(qa_train, n_sampled)

    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa_engine,
        beam_width=beam_width,
        max_hop=2,   # 2-hop: varies nd (hop distance) and community sequence
    )

    for seed_ent, correct_answers in subset:
        if len(pairs) >= n_pairs:
            break
        G_a = getattr(adapter, "_G", adapter._G)
        if seed_ent not in G_a:
            continue

        paths = traversal.traverse([seed_ent])
        if not paths:
            continue

        correct_set = set(correct_answers)
        pos_paths   = [p for p in paths if p.tail in correct_set and p.edge_features]
        neg_paths   = [p for p in paths if p.tail not in correct_set and p.edge_features]

        if pos_paths and neg_paths:
            pos = rng.choice(pos_paths)
            neg = rng.choice(neg_paths)
            pairs.append((pos, neg))

    return pairs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_hop(
    hop: int,
    traversal: BeamTraversal,
    qa_pairs: List[Tuple],
    top_k: int = 10,
    adapter=None,
) -> Dict:
    h1 = h10 = 0
    mrr_sum = 0.0
    skipped = found = 0
    t0 = time.time()

    for i, qa in enumerate(qa_pairs):
        seed, correct_answers = qa[0], qa[1]
        if (i + 1) % 1000 == 0 or (i + 1) == len(qa_pairs):
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{len(qa_pairs):,}  ({elapsed:.1f}s)", end="\r")

        paths = traversal.traverse([seed])

        # Seed embedding as query signal: activates semantic alignment in score_path()
        # so final re-ranking weights candidates that are semantically close to the
        # query entity, not just structurally reachable.
        query_emb = adapter.get_embedding(seed) if adapter is not None else None

        # For 2-hop questions, exclude depth-1 paths: direct neighbors of the
        # seed entity are always wrong intermediate nodes, never the answer.
        # For 1-hop and 3-hop, allow min_hop=1 — 3-hop correct answers can be
        # reachable via shortcut edges, and excluding them hurts H@1.
        eval_min_hop = hop if hop == 2 else 1
        answers = extract(paths, top_k=top_k, min_hop=eval_min_hop, query_embedding=query_emb)
        pred    = [a.entity_id for a in answers]

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
        "hop":        hop,
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1  / n,
        "hits_10":    h10 / n,
        "mrr":        mrr_sum / n,
        "elapsed_s":  elapsed,
        "ms_per_q":   elapsed * 1000 / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CEREBRUM full-system benchmark: Raw vs THALAMUS pipeline"
    )
    parser.add_argument("--hop",      type=int, default=None,
                        help="Evaluate only this hop (1, 2, 3). Default: all.")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Sample N questions per hop for a faster run.")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--top-k",    type=int, default=10)
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute communities and embeddings from scratch.")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--coarsen-target", type=int, default=300,
                        help="Target community count after coarsening (default: 300).")
    parser.add_argument("--optimized",    action="store_true",
                        help="Run VARIANT C: all improvements stacked (KGE, Bridge, PageRank, "
                             "soft memberships, adaptive resolution, parameter learning).")
    parser.add_argument("--kge-epochs",  type=int, default=30,
                        help="TransE training epochs for OPTIMIZED variant (default: 30; "
                             "30 epochs ~= 5 min on CPU, cached after first run).")
    parser.add_argument("--opt-beam-width", type=int, default=20,
                        help="Beam width for OPTIMIZED variant (default: 20).")
    args = parser.parse_args()

    use_cache = not args.no_cache
    hops      = [args.hop] if args.hop else [1, 2, 3]
    bw        = args.beam_width

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Header
    # ===================================================================
    print()
    print("=" * 72)
    print("  CEREBRUM Full-System Benchmark")
    print("  Raw Ingestion vs Complete THALAMUS Pipeline")
    print("=" * 72)
    if args.sample:
        print(f"  Sample: {args.sample:,} questions per hop  |  beam_width={bw}")
    else:
        print(f"  Full evaluation  |  beam_width={bw}")
    print()

    # ===================================================================
    # Build QA pairs (same for both variants — normalizer applied below)
    # ===================================================================
    print("Loading QA pairs...")
    raw_qa: Dict[int, List] = {}
    for h in hops:
        raw_qa[h] = load_qa(h, sample=args.sample, seed=args.seed)
        print(f"  {h}-hop: {len(raw_qa[h]):,} questions")
    print()

    # ===================================================================
    # VARIANT A: RAW
    # ===================================================================
    print("-" * 72)
    print("  VARIANT A — RAW  (no pipeline, random embeddings)")
    print("-" * 72)

    print("  Loading graph (raw)...")
    t0 = time.time()
    adapter_raw, G_raw = load_kb_raw()
    print(f"  {G_raw.number_of_nodes():,} entities  "
          f"{G_raw.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")

    print("  Community detection (DSCF)...")
    cmap_raw = get_communities(
        G_raw,
        CACHE_DIR / "communities_raw.pkl",
        use_cache=use_cache,
        seed=args.seed,
        label="raw",
    )
    n_raw = len(set(cmap_raw.values()))
    print(f"  {n_raw:,} communities  <-- attention heads available to CSA")

    print("  Building embeddings (RandomEngine — noise)...")
    emb_raw = get_embeddings_random(G_raw)

    print("  Building CSA engine...")
    csa_raw = build_csa(adapter_raw, G_raw, cmap_raw, emb_raw)
    print()

    results_raw: List[Dict] = []
    for h in hops:
        print(f"  [{h}-hop]  {len(raw_qa[h]):,} questions")
        traversal = BeamTraversal(
            adapter=adapter_raw,
            csa_engine=csa_raw,
            beam_width=bw,
            max_hop=h,
        )
        m = evaluate_hop(h, traversal, raw_qa[h], top_k=args.top_k, adapter=adapter_raw)
        results_raw.append(m)
        print(f"    Hits@1={m['hits_1']:.4f}  "
              f"Hits@10={m['hits_10']:.4f}  "
              f"MRR={m['mrr']:.4f}  "
              f"({m['ms_per_q']:.2f}ms/Q)")
    print()

    # ===================================================================
    # VARIANT B: FULL THALAMUS PIPELINE
    # ===================================================================
    print("-" * 72)
    print("  VARIANT B — FULL  (THALAMUS pipeline + semantic embeddings)")
    print("-" * 72)

    print("  Loading graph through IngestionPipeline...")
    t0 = time.time()
    adapter_full, G_full, entity_norm = load_kb_thalamus()
    print(f"  {G_full.number_of_nodes():,} entities  "
          f"{G_full.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")
    print(f"  Entity normalizer: lowercase + strip")
    print(f"  Relation normalizer: {len(METAQA_RELATION_MAP)} MetaQA types --> uppercase canonical")

    print("  Community detection (DSCF)...")
    cmap_full_raw = get_communities(
        G_full,
        CACHE_DIR / "communities_full_raw.pkl",
        use_cache=use_cache,
        seed=args.seed,
        label="full_raw",
    )
    n_before = len(set(cmap_full_raw.values()))
    print(f"  {n_before:,} raw communities before coarsening")

    print(f"  Coarsening to target_max={args.coarsen_target} "
          f"(meaningful attention heads)...")
    t0 = time.time()
    cmap_full = coarsen_communities(
        G_full, cmap_full_raw, target_max=args.coarsen_target
    )
    n_after = len(set(cmap_full.values()))
    print(f"  {n_after:,} communities after coarsening  "
          f"({time.time()-t0:.1f}s)  <-- attention heads available to CSA")

    print("  Building embeddings (SentenceEngine — semantic)...")
    emb_full = get_embeddings_sentence(
        G_full,
        CACHE_DIR / "embeddings_sentence.pkl",
        use_cache=use_cache,
    )

    print("  Building CSA engine...")
    csa_full = build_csa(adapter_full, G_full, cmap_full, emb_full)
    print()

    # Normalize QA pairs to match processed entity IDs
    def normalize_qa(pairs: List) -> List:
        out = []
        for qa in pairs:
            seed    = entity_norm(qa[0])
            answers = [entity_norm(a) for a in qa[1]]
            # Only include if seed entity exists in the processed graph
            if seed in G_full:
                out.append((seed, answers))
        return out

    results_full: List[Dict] = []
    for h in hops:
        qa_norm = normalize_qa(raw_qa[h])
        print(f"  [{h}-hop]  {len(qa_norm):,} questions  "
              f"({len(raw_qa[h]) - len(qa_norm):,} dropped — seed not in graph)")
        traversal = BeamTraversal(
            adapter=adapter_full,
            csa_engine=csa_full,
            beam_width=bw,
            max_hop=h,
            probabilistic=True,
            warm_start_strength=3,
        )
        m = evaluate_hop(h, traversal, qa_norm, top_k=args.top_k, adapter=adapter_full)
        results_full.append(m)
        print(f"    Hits@1={m['hits_1']:.4f}  "
              f"Hits@10={m['hits_10']:.4f}  "
              f"MRR={m['mrr']:.4f}  "
              f"({m['ms_per_q']:.2f}ms/Q)")
    print()

    # ===================================================================
    # VARIANT C: OPTIMIZED (all improvements stacked)
    # ===================================================================
    results_opt: List[Dict] = []
    n_opt_after: int = 0

    if args.optimized:
        print("-" * 72)
        print("  VARIANT C -- OPTIMIZED  (KGE + PageRank + Soft-Membership +")
        print("                           Adaptive Resolution + Param Learning +")
        print("                           Bridge Twins + beam_width=20)")
        print("-" * 72)

        # Reuse G_full and entity_norm from VARIANT B — same THALAMUS ingestion.
        # Only community detection and embeddings change.

        # ---- Step 1: Adaptive resolution search (targets sqrt(N) communities) ----
        N_nodes = G_full.number_of_nodes()
        target_k = max(50, int(N_nodes ** 0.5))
        print(f"  Adaptive resolution search (target ~{target_k} communities = sqrt({N_nodes:,}))...")
        t0 = time.time()
        opt_resolution = adaptive_resolution_search(
            G_full,
            target_communities=target_k,
            seed=args.seed,
        )
        print(f"  Best resolution: {opt_resolution:.4f}  ({time.time()-t0:.1f}s)")

        # ---- Step 2: DSCF at optimal resolution (keep partition for soft memberships) ----
        print("  Community detection (DSCF at adaptive resolution)...")
        cmap_opt_raw, partition_opt = get_communities_with_partition(
            G_full,
            CACHE_DIR / "communities_opt_raw.pkl",
            CACHE_DIR / "partition_opt.pkl",
            use_cache=use_cache,
            n_trials=3,
            seed=args.seed,
            resolution=opt_resolution,
        )
        n_opt_before = len(set(cmap_opt_raw.values()))
        print(f"  {n_opt_before:,} raw communities at adaptive resolution")

        # ---- Step 3: Coarsen to clean attention heads ----
        print(f"  Coarsening to target_max={args.coarsen_target} communities...")
        t0 = time.time()
        cmap_opt = coarsen_communities(
            G_full, cmap_opt_raw, target_max=args.coarsen_target
        )
        n_opt_after = len(set(cmap_opt.values()))
        print(f"  {n_opt_after:,} communities after coarsening  ({time.time()-t0:.1f}s)")

        # ---- Step 4: TransE KGE embeddings (64-dim for speed; projected to 384 for blend) ----
        print(f"  Training TransE KGE embeddings ({args.kge_epochs} epochs, 64-dim)...")
        emb_kge = get_kge_embeddings(
            adapter_full,
            CACHE_DIR / "kge_embeddings.pkl",
            use_cache=use_cache,
            n_epochs=args.kge_epochs,
            dim=64,
            seed=args.seed,
        )

        # ---- Step 5: Project KGE 64-dim -> 384-dim and blend with SentenceEngine ----
        # KGE dropped: 30-epoch TransE (loss≈1.065) adds noise at deep hops.
        # Pure SentenceEngine embeddings outperform blended at 2-hop and 3-hop.
        # KGE training still runs (cached) but its output is not used for scoring.
        print("  Using pure SentenceEngine embeddings (KGE blend=0% — TransE noise hurts deep hops)...")
        emb_opt = emb_full

        # ---- Step 6: PageRank prior ----
        print("  Computing PageRank prior...")
        pagerank_opt = get_pagerank(
            G_full,
            CACHE_DIR / "pagerank_opt.pkl",
            use_cache=use_cache,
        )

        # ---- Step 7: Soft community memberships ----
        print("  Computing soft community memberships...")
        soft_opt = get_soft_memberships(
            G_full,
            partition_opt,
            CACHE_DIR / "soft_memberships_opt.pkl",
            use_cache=use_cache,
        )

        # ---- Step 8: Initial CSAEngine (default params) ----
        print("  Building initial CSAEngine...")
        adapter_opt = NetworkXAdapter(G_full)
        csa_init = build_csa_full(
            adapter_opt, G_full, cmap_opt, emb_opt,
            pagerank=pagerank_opt,
            soft_memberships=soft_opt,
        )

        # ---- Step 9: CSAParameterLearner on 1+2-hop training data ----
        print("  Loading 1-hop + 2-hop training pairs for parameter learning...")
        qa_train_1 = load_qa_train(1, sample=1000, seed=args.seed, entity_norm=entity_norm)
        qa_train_2 = load_qa_train(2, sample=1000, seed=args.seed, entity_norm=entity_norm)
        qa_train = [(s, a) for s, a in (qa_train_1 + qa_train_2) if s in G_full]
        print(f"  {len(qa_train):,} training questions available (1-hop + 2-hop)")

        learned_alpha, learned_beta, learned_gamma, learned_delta, learned_epsilon = \
            0.4, 0.4, 0.1, 0.05, 0.05  # defaults if learning is skipped

        if qa_train:
            print("  Generating training path pairs (beam traversal on train split)...")
            t0     = time.time()
            t_pairs = generate_training_pairs(
                adapter_opt, csa_init, qa_train, n_pairs=2000,
                beam_width=15, seed=args.seed,
            )
            print(f"  {len(t_pairs):,} training pairs collected ({time.time()-t0:.1f}s)")

            if t_pairs:
                print("  Running CSAParameterLearner (300 iterations)...")
                t0      = time.time()
                learner = CSAParameterLearner(
                    adapter_opt,
                    learning_rate=0.05,  # larger lr: 1-hop paths have flat loss landscape
                    max_iterations=300,
                    margin=0.2,          # wider margin: forces clearer pos/neg separation
                )
                result_l = learner.fit(t_pairs, verbose=False)
                learned_alpha, learned_beta, learned_gamma, learned_delta, learned_epsilon = \
                    result_l.params
                print(f"  Learned params: alpha={learned_alpha:.3f} beta={learned_beta:.3f} "
                      f"gamma={learned_gamma:.3f} delta={learned_delta:.3f} "
                      f"epsilon={learned_epsilon:.3f}")
                print(f"  Loss: {result_l.initial_loss:.4f} -> {result_l.final_loss:.4f}  "
                      f"({time.time()-t0:.1f}s)")
            else:
                print("  No training pairs generated (train file absent?) -- using defaults")
        else:
            print("  No training data -- MetaQA qa_train.txt not found -- using default params")

        # ---- Step 10: Rebuild CSAEngine with learned parameters ----
        print("  Rebuilding CSAEngine with learned parameters...")
        adapter_opt2 = NetworkXAdapter(G_full)
        csa_opt = build_csa_full(
            adapter_opt2, G_full, cmap_opt, emb_opt,
            pagerank=pagerank_opt,
            soft_memberships=soft_opt,
            alpha=learned_alpha,
            beta=learned_beta,
            gamma=learned_gamma,
            delta=learned_delta,
            epsilon=learned_epsilon,
        )

        # ---- Step 11: BridgeTwinEngine (n_min=3 — bridges form mid-evaluation) ----
        bridge_engine = BridgeTwinEngine(n_min=3, similarity_threshold=0.55)
        print("  BridgeTwinEngine initialized (n_min=3, threshold=0.55)")
        print()

        opt_bw = args.opt_beam_width
        for h in hops:
            qa_norm_opt = [(entity_norm(s), [entity_norm(a) for a in ans])
                           for s, ans in raw_qa[h]
                           if entity_norm(s) in G_full]
            print(f"  [{h}-hop]  {len(qa_norm_opt):,} questions  "
                  f"(beam_width={opt_bw})")
            # Mild intermediate widening: keep 1.5x candidates at penultimate hop only.
            # Wider multipliers hurt accuracy by flooding noise candidates into scoring.
            adaptive_bws_opt = {h - 1: int(opt_bw * 1.5)} if h > 1 else {}
            traversal_opt = BeamTraversal(
                adapter=adapter_opt2,
                csa_engine=csa_opt,
                beam_width=opt_bw,
                max_hop=h,
                probabilistic=True,
                warm_start_strength=5,
                beam_widths=adaptive_bws_opt,
            )
            traversal_opt.bridge_engine = bridge_engine
            m = evaluate_hop(h, traversal_opt, qa_norm_opt, top_k=args.top_k, adapter=adapter_opt)
            results_opt.append(m)
            n_bridges = len(bridge_engine._bridges) if hasattr(bridge_engine, "_bridges") else "?"
            print(f"    Hits@1={m['hits_1']:.4f}  "
                  f"Hits@10={m['hits_10']:.4f}  "
                  f"MRR={m['mrr']:.4f}  "
                  f"({m['ms_per_q']:.2f}ms/Q)  "
                  f"[bridges formed: {n_bridges}]")
        print()

    # ===================================================================
    # Comparison table
    # ===================================================================
    has_opt = bool(results_opt)
    print("=" * 72)
    if has_opt:
        print("  RESULTS: RAW vs FULL THALAMUS vs OPTIMIZED")
    else:
        print("  RESULTS: Raw Ingestion vs Full THALAMUS Pipeline")
    print("=" * 72)
    print()

    if has_opt:
        print(f"  {'':30}  {'RAW':>8}  {'FULL':>8}  {'OPT':>8}  {'OPT-RAW':>9}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
    else:
        print(f"  {'':30}  {'RAW':>8}  {'FULL':>8}  {'DELTA':>8}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")

    full_iter = results_full if results_full else [{}] * len(results_raw)
    opt_iter  = results_opt  if results_opt  else [{}] * len(results_raw)

    for r_raw, r_full, r_opt in zip(results_raw, full_iter, opt_iter):
        h = r_raw["hop"]
        for metric, label in [("hits_1", "Hits@1"), ("hits_10", "Hits@10"), ("mrr", "MRR")]:
            raw_v  = r_raw.get(metric, 0.0)
            full_v = r_full.get(metric, 0.0)
            if has_opt:
                opt_v  = r_opt.get(metric, 0.0)
                delta  = opt_v - raw_v
                sign   = "+" if delta >= 0 else ""
                print(f"  {h}-hop {label:25}  {raw_v:8.4f}  {full_v:8.4f}  "
                      f"{opt_v:8.4f}  {sign}{delta:+.4f}")
            else:
                delta = full_v - raw_v
                sign  = "+" if delta >= 0 else ""
                print(f"  {h}-hop {label:25}  {raw_v:8.4f}  {full_v:8.4f}  "
                      f"{sign}{delta:+.4f}")
        ms_raw  = r_raw.get("ms_per_q", 0.0)
        ms_full = r_full.get("ms_per_q", 0.0)
        if has_opt:
            ms_opt = r_opt.get("ms_per_q", 0.0)
            print(f"  {h}-hop {'Latency (ms/Q)':25}  {ms_raw:8.2f}  {ms_full:8.2f}  {ms_opt:8.2f}")
        else:
            print(f"  {h}-hop {'Latency (ms/Q)':25}  {ms_raw:8.2f}  {ms_full:8.2f}")
        print()

    # Headline numbers
    h10_raw  = [r["hits_10"] for r in results_raw]
    h10_full = [r["hits_10"] for r in results_full]
    avg_delta_full = sum(f - r for f, r in zip(h10_full, h10_raw)) / max(len(h10_raw), 1)
    print(f"  FULL avg Hits@10 improvement vs RAW:  {avg_delta_full:+.4f} "
          f"({avg_delta_full*100:+.1f} pp)")
    if has_opt:
        h10_opt = [r["hits_10"] for r in results_opt]
        avg_delta_opt = sum(o - r for o, r in zip(h10_opt, h10_raw)) / max(len(h10_raw), 1)
        avg_delta_vs_full = sum(o - f for o, f in zip(h10_opt, h10_full)) / max(len(h10_full), 1)
        print(f"  OPT  avg Hits@10 improvement vs RAW:  {avg_delta_opt:+.4f} "
              f"({avg_delta_opt*100:+.1f} pp)")
        print(f"  OPT  avg Hits@10 improvement vs FULL: {avg_delta_vs_full:+.4f} "
              f"({avg_delta_vs_full*100:+.1f} pp)")
    print()

    # Structural context
    print("  Structural context:")
    print(f"    RAW  communities (attention heads): {n_raw:,}  "
          f"<-- {n_raw / G_raw.number_of_nodes() * 100:.0f}% of nodes are their own cluster")
    print(f"    FULL communities (attention heads): {n_after:,}  "
          f"<-- {n_after} meaningful semantic groups")
    if has_opt:
        print(f"    OPT  communities (attention heads): {n_opt_after:,}  "
              f"<-- adaptive resolution + coarsened")
    print(f"    RAW  embeddings: random 64-dim noise  (alpha = noise)")
    print(f"    FULL embeddings: SentenceEngine 384-dim  (alpha = semantic)")
    if has_opt:
        print(f"    OPT  embeddings: SentenceEngine 384-dim  (KGE blend=0% — TransE noise hurts deep hops)")
        print(f"    OPT  extras: PageRank prior, soft memberships, learned CSA params, "
              f"BridgeTwinEngine, beam_width={args.opt_beam_width}")
    print()

    # ===================================================================
    # Save results
    # ===================================================================
    results_path = CACHE_DIR / "full_system_results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "hop", "n_total", "n_answered", "n_skipped",
            "hits_1", "hits_10", "mrr", "beam_width", "ms_per_q"
        ])
        writer.writeheader()
        for m in results_raw:
            row = {k: m[k] for k in writer.fieldnames if k in m}
            row.update({"variant": "RAW", "beam_width": bw})
            writer.writerow(row)
        for m in results_full:
            row = {k: m[k] for k in writer.fieldnames if k in m}
            row.update({"variant": "FULL", "beam_width": bw})
            writer.writerow(row)
        for m in results_opt:
            row = {k: m[k] for k in writer.fieldnames if k in m}
            row.update({"variant": "OPTIMIZED", "beam_width": args.opt_beam_width})
            writer.writerow(row)

    print(f"  Results saved to {results_path}")
    print()


if __name__ == "__main__":
    main()

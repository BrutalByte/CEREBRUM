"""
Hetionet biomedical knowledge graph benchmark for CEREBRUM (Phase 4).

Hetionet integrates multiple biomedical databases into a single graph:
  Nodes : 47,031 across 11 types (Gene, Disease, Compound, Anatomy,
          Pathway, Biological Process, Molecular Function,
          Cellular Component, Pharmacologic Class, Side Effect, Symptom)
  Edges : ~2.25M across 24 metaedge types
  Source: https://github.com/hetio/hetionet
  License: CC0 (public domain)

Unlike MetaQA (star/bipartite topology), Hetionet has rich community
structure: diseases cluster with diseases, genes with genes, compounds
with compounds. This makes it suitable for testing CSA's core value.

QA templates are path patterns extracted from the graph structure:
  1-hop: "What diseases does [compound] treat?"
  1-hop: "What genes are associated with [disease]?"
  2-hop: "What pathways involve genes associated with [disease]?"
  2-hop: "What diseases can [compound] treat via gene binding?"
  3-hop: "What compounds bind genes upregulated in [anatomy] for [disease]?"

Type alignment score: measures how well DSCF communities align with
Hetionet's 11 known node types. High alignment = community structure
matches biological taxonomy.

Usage
-----
  # Quick dev run (subset):
  python -m benchmarks.hetionet_eval --max-edges 100000 --n-questions 50

  # Full evaluation:
  python -m benchmarks.hetionet_eval --use-cache

  # Single template:
  python -m benchmarks.hetionet_eval --template compound_treats_disease
"""

import argparse
import bz2
import csv
import json
import pickle
import random
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf, lpa_communities, merge_small_communities
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
from benchmarks.baseline_comparison import UniformCSAEngine

DATA_DIR   = Path(__file__).parent / "data" / "hetionet"
JSON_FILE  = DATA_DIR / "hetionet-v1.0.json.bz2"
GRAPH_CACHE = DATA_DIR / "hetionet_graph.pkl"
CACHE_DIR  = DATA_DIR / "cache"

HETIONET_URL = (
    "https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2"
)

# QA templates: (name, hop, seed_type, answer_type, required_metaedge_chain)
# Metaedge labels verified against Hetionet TSV headers.
QA_TEMPLATES = {
    "compound_treats_disease": (
        1, "Compound", "Disease", ["Compound-treats-Disease"]
    ),
    "disease_associates_gene": (
        1, "Disease", "Gene", ["Disease-associates-Gene"]
    ),
    "gene_participates_pathway": (
        1, "Gene", "Pathway", ["Gene-participates-Pathway"]
    ),
    "disease_gene_pathway": (
        2, "Disease", "Pathway",
        ["Disease-associates-Gene", "Gene-participates-Pathway"]
    ),
    "compound_gene_disease": (
        2, "Compound", "Disease",
        ["Compound-binds-Gene", "Gene-associates-Disease"]
    ),
    "disease_gene_compound": (
        3, "Disease", "Compound",
        ["Disease-associates-Gene", "Gene-associates-Disease",
         "Disease-treated_by-Compound"]
    ),
}

# Bridge bonus for inter-type metaedges used in reasoning tasks.
# Offsets the cross-community penalty for typed communities.
HETIONET_BRIDGE_BONUS = 0.4


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_hetionet(force: bool = False) -> Path:
    """
    Download hetionet-v1.0.json.bz2 from the Hetionet GitHub repository.

    The JSON format is used instead of the TSV because the TSV is stored in
    Git LFS and cannot be downloaded via raw GitHub URLs.

    File size: ~16 MB compressed, downloads in ~5s on a fast connection.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if JSON_FILE.exists() and not force:
        return JSON_FILE

    print(f"  Downloading Hetionet JSON from GitHub (~16 MB)...")
    print(f"  URL: {HETIONET_URL}")
    print(f"  Destination: {JSON_FILE}")

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(120)
    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size / total_size * 100, 100)
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(HETIONET_URL, JSON_FILE, reporthook=_progress)
        print()
        print(f"  Download complete: {JSON_FILE.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"\n  ERROR: Download failed: {e}")
        raise FileNotFoundError(f"Hetionet data not found at {JSON_FILE}") from e
    finally:
        socket.setdefaulttimeout(old_timeout)

    return JSON_FILE


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def _node_id(kind: str, identifier) -> str:
    """Construct canonical node ID from kind and identifier."""
    return f"{kind}::{identifier}"


def load_hetionet(
    max_edges: Optional[int] = None,
    node_type_filter: Optional[List[str]] = None,
    undirected: bool = True,
    use_graph_cache: bool = True,
) -> Tuple[NetworkXAdapter, Dict[str, str]]:
    """
    Load Hetionet from hetionet-v1.0.json.bz2 into a NetworkXAdapter.

    Parses the Hetionet JSON format:
      nodes: [{kind, identifier, name, data}, ...]
      edges: [{source_id: [kind, id], target_id: [kind, id], kind, ...}, ...]

    Node IDs are constructed as "Kind::identifier" to match the TSV convention.
    A NetworkX pickle cache is stored at DATA_DIR/hetionet_graph.pkl for fast
    reloads (loading JSON takes ~28s; pickle loads in ~3s).

    Parameters
    ----------
    max_edges        : cap edges loaded (for development runs; disables cache)
    node_type_filter : only load edges where both endpoints are in this type list
    undirected       : load as undirected graph (standard for QA traversal)
    use_graph_cache  : use pickle cache when max_edges/filter not set
    """
    # Use graph-level pickle cache only for full unfiltered loads
    can_cache = (max_edges is None and node_type_filter is None and use_graph_cache)

    if can_cache and GRAPH_CACHE.exists():
        print(f"  Loading cached NetworkX graph from {GRAPH_CACHE.name}...")
        t0 = time.time()
        with open(GRAPH_CACHE, "rb") as f:
            G, node_type_map = pickle.load(f)
        print(f"  Loaded in {time.time()-t0:.1f}s")
        return NetworkXAdapter(G), node_type_map

    print(f"  Parsing {JSON_FILE.name} (this takes ~30s first time)...")
    t0 = time.time()
    with open(JSON_FILE, "rb") as f:
        raw = bz2.decompress(f.read())
    obj = json.loads(raw)
    del raw
    print(f"  JSON parsed in {time.time()-t0:.1f}s")

    G = nx.Graph() if undirected else nx.DiGraph()
    node_type_map: Dict[str, str] = {}
    filter_set = set(node_type_filter) if node_type_filter else None

    # Add all nodes (so isolated nodes are included in node_type_map)
    for node_data in obj["nodes"]:
        kind = node_data["kind"]
        if filter_set and kind not in filter_set:
            continue
        nid = _node_id(kind, node_data["identifier"])
        node_type_map[nid] = kind
        G.add_node(nid, type=kind, name=node_data.get("name", ""))

    # Add edges
    for i, edge_data in enumerate(obj["edges"]):
        if max_edges is not None and i >= max_edges:
            break
        s_kind, s_id = edge_data["source_id"]
        t_kind, t_id = edge_data["target_id"]
        if filter_set and (s_kind not in filter_set or t_kind not in filter_set):
            continue
        s_nid = _node_id(s_kind, s_id)
        t_nid = _node_id(t_kind, t_id)
        # Metaedge label: "SourceKind-kind-TargetKind"
        metaedge = f"{s_kind}-{edge_data['kind']}-{t_kind}"
        G.add_edge(s_nid, t_nid, relation=metaedge)

    print(f"  Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s total)")

    if can_cache:
        print(f"  Caching graph to {GRAPH_CACHE.name}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_CACHE, "wb") as f:
            pickle.dump((G, node_type_map), f)
        print(f"  Cached ({GRAPH_CACHE.stat().st_size / 1e6:.1f} MB)")

    return NetworkXAdapter(G), node_type_map


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

def generate_hetionet_qa(
    G: nx.Graph,
    template: str,
    n_questions: int = 200,
    seed: int = 42,
) -> List[Tuple[str, List[str]]]:
    """
    Generate QA pairs for a named template by traversing known graph paths.

    For each seed entity, all valid answer entities reachable via the
    template's metaedge chain are collected as ground-truth answers.
    This matches MetaQA's multi-answer convention.

    Returns list of (seed_entity, [answer_entities]).
    """
    hop, seed_type, answer_type, metaedge_chain = QA_TEMPLATES[template]
    rng = random.Random(seed)

    # Collect valid seeds (nodes of seed_type with at least one path)
    # Index edges by (source, metaedge) for fast lookup
    edge_index: Dict[Tuple[str, str], List[str]] = {}
    for u, v, data in G.edges(data=True):
        me = data.get("relation", "")
        # Index both directions (undirected graph)
        edge_index.setdefault((u, me), []).append(v)
        edge_index.setdefault((v, me), []).append(u)

    def follow_chain(start: str, chain: List[str]) -> List[str]:
        """BFS along metaedge chain, return endpoints."""
        current = [start]
        for metaedge in chain:
            next_nodes = []
            for node in current:
                next_nodes.extend(edge_index.get((node, metaedge), []))
            current = list(set(next_nodes))
            if not current:
                return []
        return current

    # Collect all seeds of the right type
    all_seeds = [n for n in G.nodes()
                 if G.nodes[n].get("type", n.split("::")[0] if "::" in n else "") == seed_type]

    if not all_seeds:
        print(f"  WARNING: No nodes of type '{seed_type}' found for template '{template}'")
        return []

    rng.shuffle(all_seeds)

    pairs: List[Tuple[str, List[str]]] = []
    for seed_node in all_seeds:
        if len(pairs) >= n_questions:
            break
        answers = follow_chain(seed_node, metaedge_chain)
        if answers:
            pairs.append((seed_node, answers))

    return pairs


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def load_or_compute_communities(
    G: nx.Graph,
    label: str,
    use_cache: bool = True,
    n_trials: int = 3,
    dscf_seed: int = 42,
    min_community_size: int = 0,
) -> Dict[str, int]:
    """Run DSCF and cache to CACHE_DIR/communities_{label}.pkl."""
    import pickle
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"communities_{label}.pkl"

    if use_cache and cache_file.exists():
        print(f"  Loading cached communities from {cache_file}")
        with open(cache_file, "rb") as f:
            cmap = pickle.load(f)
    else:
        print(f"  Running DSCF on {G.number_of_nodes():,} nodes "
              f"({G.number_of_edges():,} edges)...")
        print(f"  Estimated time: 2-15 minutes depending on graph size.")
        t0    = time.time()
        parts = best_of_n_dscf(G, n_trials=n_trials, seed=dscf_seed)
        cmap  = {node: cid for cid, members in enumerate(parts) for node in members}
        print(f"  DSCF: {len(parts)} communities in {time.time()-t0:.1f}s")

        with open(cache_file, "wb") as f:
            pickle.dump(cmap, f)
        print(f"  Communities cached to {cache_file}")

    n_raw = len(set(cmap.values()))
    print(f"  {n_raw} communities (raw DSCF)")

    if min_community_size > 0:
        print(f"  Merging communities smaller than {min_community_size} members...")
        t0   = time.time()
        cmap = merge_small_communities(cmap, G, min_size=min_community_size)
        n_merged = len(set(cmap.values()))
        print(f"  {n_merged} communities after merge ({time.time()-t0:.1f}s)")

    return cmap


# ---------------------------------------------------------------------------
# Type alignment score
# ---------------------------------------------------------------------------

def compute_type_alignment(
    node_type_map: Dict[str, str],
    detected_cmap: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute community purity with respect to Hetionet node types.

    For each community, find the plurality node type. Purity = fraction
    of nodes in that community that match the plurality type.
    Returns (mean_purity_weighted_by_size, {community_id -> purity}).

    High purity (> 0.8) means DSCF recovered biologically meaningful
    communities aligned with node types.
    """
    # Build community -> {type: count}
    com_type_counts: Dict[int, Dict[str, int]] = {}
    for node, cid in detected_cmap.items():
        ntype = node_type_map.get(node, "Unknown")
        com_type_counts.setdefault(cid, {})
        com_type_counts[cid][ntype] = com_type_counts[cid].get(ntype, 0) + 1

    total_nodes = 0
    weighted_purity = 0.0
    per_community: Dict[int, float] = {}

    for cid, type_counts in com_type_counts.items():
        size = sum(type_counts.values())
        plurality = max(type_counts.values())
        purity = plurality / size
        per_community[cid] = purity
        weighted_purity += purity * size
        total_nodes += size

    mean_purity = weighted_purity / total_nodes if total_nodes else 0.0
    return mean_purity, per_community


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
    """Evaluate one variant. Returns metrics dict."""
    h1 = h10 = mrr_sum = 0
    skipped = found = 0
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 50 == 0 or (i + 1) == len(qa_pairs):
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
    print()
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


def build_traversal(
    adapter, G, cmap, embeddings, beam_width, max_hop,
    variant="dscf", edge_type_weights=None
) -> BeamTraversal:
    """Build a BeamTraversal for a given community map and variant type."""
    # Attach communities and embeddings to adapter for lookups
    if cmap:
        adapter.community_map = cmap
    if embeddings:
        adapter.embeddings = embeddings

    if variant == "bfs":
        cmap_uniform = {node: 0 for node in G.nodes()}
        adapter.community_map = cmap_uniform
        csa = UniformCSAEngine(adapter=adapter)
        return BeamTraversal(adapter=adapter, csa_engine=csa,
                             beam_width=beam_width, max_hop=max_hop)

    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    csa  = CSAEngine(adapter=adapter)
    csa.set_community_graph(dist, adj)
    return BeamTraversal(adapter=adapter, csa_engine=csa,
                         beam_width=beam_width, max_hop=max_hop,
                         edge_type_weights=edge_type_weights)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hetionet biomedical KG benchmark for CEREBRUM"
    )
    parser.add_argument("--template",       type=str,  default=None,
                        help=f"QA template to evaluate. Options: "
                             f"{', '.join(QA_TEMPLATES.keys())}. Default: all.")
    parser.add_argument("--n-questions",    type=int,  default=200)
    parser.add_argument("--beam-width",     type=int,  default=10)
    parser.add_argument("--top-k",          type=int,  default=10)
    parser.add_argument("--max-edges",      type=int,  default=None,
                        help="Cap edges loaded (for dev runs, e.g. 100000)")
    parser.add_argument("--node-types",     type=str,  default=None,
                        help="Comma-separated node types to include "
                             "(e.g. Disease,Gene,Compound). Default: all.")
    parser.add_argument("--use-cache",      action="store_true", default=True)
    parser.add_argument("--no-cache",       action="store_true")
    parser.add_argument("--no-download",    action="store_true",
                        help="Fail if data is not already present.")
    parser.add_argument("--min-community-size", type=int, default=0,
                        help="Merge communities smaller than this. 0=disabled.")
    parser.add_argument("--embeddings",     choices=["random", "sentence"],
                        default="random")
    parser.add_argument("--seed",           type=int,  default=42)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    node_type_filter = (
        [t.strip() for t in args.node_types.split(",")]
        if args.node_types else None
    )

    templates = (
        [args.template] if args.template
        else list(QA_TEMPLATES.keys())
    )

    print("\n=== CEREBRUM — Hetionet Biomedical KG Benchmark ===\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not JSON_FILE.exists():
        if args.no_download:
            print(f"ERROR: {JSON_FILE} not found and --no-download specified.")
            print(f"Download from: {HETIONET_URL}")
            sys.exit(1)
        download_hetionet()

    print("Loading Hetionet graph...")
    t0 = time.time()
    adapter, node_type_map = load_hetionet(
        max_edges=args.max_edges,
        node_type_filter=node_type_filter,
        use_graph_cache=args.use_cache,
    )
    G = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({time.time()-t0:.1f}s)")

    # Print node type distribution
    type_counts: Dict[str, int] = {}
    for ntype in node_type_map.values():
        type_counts[ntype] = type_counts.get(ntype, 0) + 1
    print(f"  Node types:")
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ntype:<30} {count:>7,}")
    print()

    # Store type info on graph nodes for lookup during QA generation
    for node, ntype in node_type_map.items():
        if node in G.nodes:
            G.nodes[node]["type"] = ntype

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    print("Building embeddings...")
    random.seed(args.seed)
    if args.embeddings == "sentence":
        try:
            from core.embedding_engine import SentenceEngine
            engine = SentenceEngine()
            print(f"  Using SentenceEngine ({engine.dim}-dim)")
        except ImportError:
            print("  sentence-transformers not installed — using RandomEngine")
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)
        print(f"  Using RandomEngine (64-dim)")
    labels     = {n: n for n in G.nodes()}
    embeddings = engine.encode_entities(labels)
    print(f"  {len(embeddings):,} entity vectors")
    print()

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------
    print("Computing community structure (DSCF)...")
    cache_label = f"dscf_{G.number_of_nodes()}_{G.number_of_edges()}"
    cmap_dscf = load_or_compute_communities(
        G, label=cache_label,
        use_cache=args.use_cache, dscf_seed=args.seed,
        min_community_size=args.min_community_size,
    )
    purity_dscf, _ = compute_type_alignment(node_type_map, cmap_dscf)
    print(f"  Type alignment (community purity vs node type): {purity_dscf:.4f}")

    print("Computing LPA communities...")
    t0 = time.time()
    lpa_parts = lpa_communities(G)
    cmap_lpa  = {node: cid for cid, members in enumerate(lpa_parts) for node in members}
    print(f"  LPA: {len(lpa_parts)} communities in {time.time()-t0:.1f}s")
    purity_lpa, _ = compute_type_alignment(node_type_map, cmap_lpa)
    print(f"  Type alignment (LPA communities): {purity_lpa:.4f}")
    print()

    # Precompute bridge bonus weights for metaedges used in reasoning tasks
    edge_type_weights = {}
    for _, _, _, chain in QA_TEMPLATES.values():
        for metaedge in chain:
            edge_type_weights[metaedge] = HETIONET_BRIDGE_BONUS

    # ------------------------------------------------------------------
    # Evaluate each template
    # ------------------------------------------------------------------
    all_results = []

    for template_name in templates:
        if template_name not in QA_TEMPLATES:
            print(f"  WARNING: Unknown template '{template_name}', skipping.")
            continue

        hop, seed_type, answer_type, chain = QA_TEMPLATES[template_name]
        print(f"--- Template: {template_name} ({hop}-hop: {seed_type} -> {answer_type}) ---")

        qa_pairs = generate_hetionet_qa(
            G, template=template_name,
            n_questions=args.n_questions, seed=args.seed,
        )
        if not qa_pairs:
            print(f"  No QA pairs generated — skipping (metaedge may not be in "
                  f"loaded graph subset).")
            print()
            continue
        print(f"  {len(qa_pairs):,} QA pairs generated")

        # Variant A — DSCF + CSA
        print(f"\n  [A] DSCF + CSA (purity={purity_dscf:.3f}, bonus={HETIONET_BRIDGE_BONUS})...")
        t_dscf = build_traversal(adapter, G, cmap_dscf, embeddings,
                                 args.beam_width, hop, variant="dscf",
                                 edge_type_weights=edge_type_weights)
        m_a = evaluate_variant("DSCF+CSA", t_dscf, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_a['hits_1']:.4f}  Hits@10={m_a['hits_10']:.4f}  MRR={m_a['mrr']:.4f}")

        # Variant B — LPA + CSA
        print(f"\n  [B] LPA + CSA (purity={purity_lpa:.3f}, bonus={HETIONET_BRIDGE_BONUS})...")
        t_lpa = build_traversal(adapter, G, cmap_lpa, embeddings,
                                args.beam_width, hop, variant="lpa",
                                edge_type_weights=edge_type_weights)
        m_b = evaluate_variant("LPA+CSA", t_lpa, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_b['hits_1']:.4f}  Hits@10={m_b['hits_10']:.4f}  MRR={m_b['mrr']:.4f}")

        # Variant C — BFS
        print(f"\n  [C] BFS (uniform weights)...")
        t_bfs = build_traversal(adapter, G, None, embeddings,
                                args.beam_width, hop, variant="bfs")
        m_c = evaluate_variant("BFS", t_bfs, qa_pairs, hop, args.top_k)
        print(f"      Hits@1={m_c['hits_1']:.4f}  Hits@10={m_c['hits_10']:.4f}  MRR={m_c['mrr']:.4f}")

        delta = m_a["hits_1"] - m_c["hits_1"]
        sign  = "+" if delta >= 0 else ""
        print(f"\n  DSCF-BFS Hits@1 delta: {sign}{delta:.4f}")
        print()

        all_results.append({
            "template": template_name, "hop": hop,
            "dscf": m_a, "lpa": m_b, "bfs": m_c,
        })

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if all_results:
        print("=== Summary by Template ===\n")
        print(f"  {'Template':<35} {'Hop':<5} {'DSCF H@1':>10} {'LPA H@1':>9} "
              f"{'BFS H@1':>9} {'d DSCF-BFS':>12}")
        print("  " + "-" * 85)
        for r in all_results:
            a, b, c = r["dscf"], r["lpa"], r["bfs"]
            delta = a["hits_1"] - c["hits_1"]
            sign  = "+" if delta >= 0 else ""
            print(f"  {r['template']:<35} {r['hop']:<5} "
                  f"{a['hits_1']:>10.4f} {b['hits_1']:>9.4f} {c['hits_1']:>9.4f} "
                  f"{sign}{delta:>10.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CACHE_DIR / "hetionet_results.csv"
    rows = []
    for r in all_results:
        for m in [r["dscf"], r["lpa"], r["bfs"]]:
            rows.append({
                "template": r["template"], "hop": r["hop"],
                "variant": m["variant"],
                "n_total": m["n_total"], "n_answered": m["n_answered"],
                "hits_1": m["hits_1"], "hits_10": m["hits_10"], "mrr": m["mrr"],
                "elapsed_s": m["elapsed_s"],
                "purity_dscf": round(purity_dscf, 4),
                "purity_lpa": round(purity_lpa, 4),
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




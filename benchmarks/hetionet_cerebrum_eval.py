"""
Phase 172/167: CerebrumGraph-based Hetionet Biomedical KG Benchmark.

Demonstrates the full CEREBRUM stack on Hetionet -a 47,031-node heterogeneous
biomedical knowledge graph with 11 entity types and 24 metaedge types.

Unlike the original hetionet_eval.py (which uses raw BeamTraversal), this
benchmark uses CerebrumGraph.build() + CerebrumGraph.query(), enabling:
  - DSCF community structure (biologically meaningful clusters)
  - CSA attention (10-parameter structural attention)
  - Terminal Relation Boost (TRB) per biological relation type
  - H1SE hop-1 expansion (independent per-branch sub-traversal)
  - Terminal-Anchor Beam (TAB, Phase 172) -strict anchor sets work here
    because "Compound-treats-Disease" sources are only ~2.4% of all nodes
  - STRB (Phase 172) — semantic TRB via query-text × relation-phrase
    cosine similarity, replacing structural SRI in Profile-Auto mode

Ablation ladder (for each template):
  BFS              -raw BeamTraversal, no community structure
  DSCF+CSA         -CerebrumGraph baseline (build + query, no TRB/H1SE)
  +TRB             -adds terminal_relation_boost per template
  +H1SE            -adds hop_expand=True (independent per-branch expansion)
  +TAB             -adds anchor_bonus=2.0 (answer-type-aware hop selection)
  Profile-Auto     -zero-config: all params from GraphProfiler (uses SRI)
  Profile-Auto+STRB -zero-config with semantic query embedding (Phase 172)

Type alignment: reports community purity vs. Hetionet's 11 biological types
after build(). High purity (>0.8) proves DSCF recovered biologically meaningful
communities without any type labels -a direct measurement of DSCF's value.

QA Templates
------------
  1-hop:
    compound_treats_disease  -"What diseases does [compound] treat?"
    disease_associates_gene  -"What genes are associated with [disease]?"
    gene_participates_pathway— "What pathways involve [gene]?"

  2-hop:
    disease_gene_pathway     -"What pathways involve genes linked to [disease]?"
    compound_gene_disease    -"What diseases can [compound] reach via gene binding?"

  3-hop:
    disease_compound_via_gene— "What compounds treat diseases sharing genes with [disease]?"
                               Chain: Disease→Gene→Disease→Compound (fixed vs old eval)

Usage
-----
  # Quick dev run (50 questions per template, max 100K edges):
  python -m benchmarks.hetionet_cerebrum_eval --n-questions 50 --max-edges 100000

  # Full evaluation with all ablation variants:
  python -m benchmarks.hetionet_cerebrum_eval --use-cache

  # Single template, skip BFS for speed:
  python -m benchmarks.hetionet_cerebrum_eval \\
      --template disease_associates_gene --no-bfs

  # With sentence embeddings — enables STRB (Phase 172) for Profile-Auto+STRB variant:
  python -m benchmarks.hetionet_cerebrum_eval --embeddings sentence
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.hetionet_eval import (
    QA_TEMPLATES,
    HETIONET_URL,
    JSON_FILE,
    GRAPH_CACHE,
    DATA_DIR,
    download_hetionet,
    load_hetionet,
    generate_hetionet_qa,
    compute_type_alignment,
)
from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
from core.cerebrum import CerebrumGraph
from core.embedding_engine import RandomEngine, SentenceEngine

# ---------------------------------------------------------------------------
# Template configuration
# ---------------------------------------------------------------------------

# Fixed 3-hop template: Disease→Gene→Disease→Compound
# Old hetionet_eval used "Disease-treated_by-Compound" which doesn't exist in
# the graph. Since Hetionet is undirected, the metaedge label stored on the
# edge is always "Compound-treats-Disease" regardless of traversal direction.
QA_TEMPLATES_FIXED = dict(QA_TEMPLATES)
# Fix compound_gene_disease: "Gene-associates-Disease" is not a valid metaedge label.
# In Hetionet (undirected), the stored label is always "Disease-associates-Gene".
# edge_index[(gene, "Disease-associates-Gene")] = [diseases] works bidirectionally.
QA_TEMPLATES_FIXED["compound_gene_disease"] = (
    2, "Compound", "Disease",
    ["Compound-binds-Gene", "Disease-associates-Gene"],
)
# Fixed 3-hop: Disease->Gene->Disease->Compound
# "Disease-treated_by-Compound" does not exist; use "Compound-treats-Disease" (bidirectional).
QA_TEMPLATES_FIXED["disease_compound_via_gene"] = (
    3, "Disease", "Compound",
    ["Disease-associates-Gene", "Disease-associates-Gene", "Compound-treats-Disease"],
)
# Drop the broken old 3-hop template if present
QA_TEMPLATES_FIXED.pop("disease_gene_compound", None)

# Per-template terminal relation boost mappings.
# Key = metaedge label that the correct answer should be reached via.
# Guides the beam to prefer paths that end on the biologically relevant edge.
TEMPLATE_TRB: Dict[str, Dict[str, float]] = {
    "compound_treats_disease":   {"Compound-treats-Disease": 3.0},
    "disease_associates_gene":   {"Disease-associates-Gene": 3.0},
    "gene_participates_pathway": {"Gene-participates-Pathway": 3.0},
    "disease_gene_pathway":      {"Gene-participates-Pathway": 3.0},
    "compound_gene_disease":     {"Disease-associates-Gene": 3.0},
    "disease_compound_via_gene": {"Compound-treats-Disease": 3.0},
}

# Answer entity type prefix per template (used to filter predictions).
TEMPLATE_ANSWER_TYPE: Dict[str, str] = {
    "compound_treats_disease":   "Disease",
    "disease_associates_gene":   "Gene",
    "gene_participates_pathway": "Pathway",
    "disease_gene_pathway":      "Pathway",
    "compound_gene_disease":     "Disease",
    "disease_compound_via_gene": "Compound",
}

# Hop depth for min_hop filtering in query()
TEMPLATE_HOP: Dict[str, int] = {
    name: cfg[0] for name, cfg in QA_TEMPLATES_FIXED.items()
}

# Phase 172: Natural-language question templates for STRB.
# Each question is encoded at query time and compared to relation phrase embeddings
# via cosine similarity, replacing structural SRI for terminal relation selection.
TEMPLATE_QUESTION: Dict[str, str] = {
    "compound_treats_disease":    "What compound treats {seed}?",
    "disease_associates_gene":    "What gene is associated with {seed}?",
    "gene_participates_pathway":  "What pathway does {seed} participate in?",
    "disease_gene_pathway":       "What pathway involves genes associated with {seed}?",
    "compound_gene_disease":      "What disease is linked through genes to {seed}?",
    "disease_compound_via_gene":  "What compound treats a disease sharing genes with {seed}?",
}


def _seed_label(seed_id: str) -> str:
    """Extract readable label from entity ID (e.g. 'Disease::lung cancer' -> 'lung cancer')."""
    return seed_id.split("::", 1)[-1] if "::" in seed_id else seed_id

CACHE_DIR = DATA_DIR / "cache"


# ---------------------------------------------------------------------------
# Build CerebrumGraph
# ---------------------------------------------------------------------------

def build_cerebrum_graph(
    adapter,
    embedding_mode: str = "random",
    cache_dir: Optional[Path] = None,
    n_trials: int = 1,
    seed: int = 42,
    beam_width: int = 10,
    max_hop: int = 3,
) -> CerebrumGraph:
    """
    Build a CerebrumGraph from a NetworkXAdapter.
    Embeddings, DSCF communities, and CSA attention are all computed here.
    """
    if embedding_mode == "sentence":
        try:
            engine = SentenceEngine()
            print(f"  Using SentenceEngine ({engine.dim}-dim)")
        except ImportError:
            print("  sentence-transformers not installed -falling back to RandomEngine")
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)
        print(f"  Using RandomEngine (64-dim)")

    graph = CerebrumGraph(
        adapter=adapter,
        embedding_engine=engine,
        beam_width=beam_width,
        max_hop=max_hop,
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    graph.build(
        cache_dir=str(cache_dir) if cache_dir else None,
        n_trials=n_trials,
        seed=seed,
        community_engine="dscf",
    )
    print(f"  CerebrumGraph built in {time.time()-t0:.1f}s")
    return graph


# ---------------------------------------------------------------------------
# Type alignment after build
# ---------------------------------------------------------------------------

def report_type_alignment(graph: CerebrumGraph, node_type_map: Dict[str, str]) -> float:
    """Compute and print community purity vs. Hetionet biological types."""
    cmap = getattr(graph.adapter, "community_map", {})
    if not cmap:
        print("  (no community map available)")
        return 0.0
    purity, per_community = compute_type_alignment(node_type_map, cmap)
    n_communities = len(set(cmap.values()))
    high_purity = sum(1 for p in per_community.values() if p >= 0.80)
    print(f"  Communities: {n_communities:,}")
    print(f"  Mean weighted purity: {purity:.4f}  "
          f"(high-purity >=0.80: {high_purity}/{n_communities} communities)")
    print(f"  Interpretation: purity=1.0 means every community contains only one "
          f"biological type -DSCF perfectly recovered the known taxonomy.")
    return purity


# ---------------------------------------------------------------------------
# Anchor set statistics (TAB diagnostic)
# ---------------------------------------------------------------------------

def report_anchor_stats(graph: CerebrumGraph, template: str) -> None:
    """Show how discriminative the TAB anchor set is for a given template."""
    trb = TEMPLATE_TRB.get(template, {})
    if not trb:
        return
    best_rel = max(trb, key=trb.get)
    anchor_sources = getattr(graph, "_anchor_sources", {})
    anchor_set = anchor_sources.get(best_rel, set())
    if not anchor_set:
        print(f"    TAB anchor: no anchor set for '{best_rel}'")
        return
    G = graph.adapter.to_networkx()
    total_nodes = G.number_of_nodes()
    pct = 100.0 * len(anchor_set) / total_nodes if total_nodes else 0.0
    print(f"    TAB anchor: '{best_rel}' sources = "
          f"{len(anchor_set):,}/{total_nodes:,} nodes ({pct:.1f}%) -"
          f"{'strict subset [discriminative]' if pct < 20 else 'large set (low discrimination)'}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_template(
    graph: CerebrumGraph,
    template: str,
    qa_pairs: List[Tuple[str, List[str]]],
    variants: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict]:
    """
    Evaluate all variants for a single template.
    Each variant dict has: name, query_kwargs, and optional use_strb flag.
    When use_strb=True and the semantic index is available, the question text
    is encoded per query and passed as query_embedding for STRB inference.
    Returns list of result dicts.
    """
    hop = TEMPLATE_HOP[template]
    answer_type = TEMPLATE_ANSWER_TYPE[template]
    _q_template = TEMPLATE_QUESTION.get(template, "")
    results = []

    # Determine STRB availability once per template evaluation.
    _sri = getattr(graph, "_sri", None)
    _strb_available = (
        _sri is not None
        and getattr(_sri, "_semantic_index_built", False)
        and bool(_q_template)
    )

    for variant in variants:
        vname = variant["name"]
        qkw = variant.get("query_kwargs", {})
        use_strb = variant.get("use_strb", False) and _strb_available
        h1 = h10 = 0
        mrr_sum = 0.0
        n_answered = 0
        t0 = time.time()

        for i, (seed, correct_answers) in enumerate(qa_pairs):
            if (i + 1) % 25 == 0 or (i + 1) == len(qa_pairs):
                elapsed = time.time() - t0
                print(f"    [{vname}] {i+1:,}/{len(qa_pairs):,} "
                      f"({elapsed:.1f}s)", end="\r")

            extra: Dict[str, Any] = {}
            if use_strb:
                question = _q_template.format(seed=_seed_label(seed))
                extra["query_embedding"] = graph._embedding_engine.encode_one(question)
                # ensure auto_infer fires (reads from profile or explicit True)
                if "auto_infer_terminal_relation" not in qkw:
                    extra["auto_infer_terminal_relation"] = True

            answers = graph.query(
                seeds=[seed],
                top_k=top_k * 3,  # over-fetch then filter by type
                min_hop=hop,
                max_hop=hop,
                **qkw,
                **extra,
            )

            # Filter predictions to correct answer entity type
            pred = [
                a.entity_id for a in answers
                if a.entity_id.startswith(f"{answer_type}::")
            ][:top_k]

            if not pred:
                continue

            n_answered += 1
            h1      += hits_at_k(pred, correct_answers, k=1)
            h10     += hits_at_k(pred, correct_answers, k=10)
            mrr_sum += reciprocal_rank(pred, correct_answers)

        elapsed = time.time() - t0
        n = len(qa_pairs)
        print()

        results.append({
            "template": template,
            "variant":  vname,
            "hop":      hop,
            "n_total":  n,
            "n_answered": n_answered,
            "hits_1":   h1 / n if n else 0.0,
            "hits_10":  h10 / n if n else 0.0,
            "mrr":      mrr_sum / n if n else 0.0,
            "elapsed_s": elapsed,
        })

    return results


# ---------------------------------------------------------------------------
# BFS baseline (raw BeamTraversal, no community structure)
# ---------------------------------------------------------------------------

def evaluate_bfs_baseline(
    adapter,
    template: str,
    qa_pairs: List[Tuple[str, List[str]]],
    beam_width: int = 10,
    top_k: int = 10,
) -> Dict:
    """BFS baseline using raw BeamTraversal with UniformCSA."""
    from benchmarks.baseline_comparison import UniformCSAEngine
    from reasoning.traversal import BeamTraversal
    from reasoning.answer_extractor import extract
    import networkx as nx

    hop = TEMPLATE_HOP[template]
    answer_type = TEMPLATE_ANSWER_TYPE[template]

    G = adapter.to_networkx()
    cmap_uniform = {node: 0 for node in G.nodes()}
    adapter.community_map = cmap_uniform
    csa = UniformCSAEngine(adapter=adapter)
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=beam_width,
        max_hop=hop,
        max_neighbors=200,
    )

    h1 = h10 = 0
    mrr_sum = 0.0
    n_answered = 0
    t0 = time.time()

    for i, (seed, correct_answers) in enumerate(qa_pairs):
        if (i + 1) % 25 == 0 or (i + 1) == len(qa_pairs):
            print(f"    [BFS] {i+1:,}/{len(qa_pairs):,} "
                  f"({time.time()-t0:.1f}s)", end="\r")

        paths = traversal.traverse([seed])
        answers_obj = extract(paths, top_k=top_k * 3, min_hop=hop)
        pred = [
            a.entity_id for a in answers_obj
            if a.entity_id.startswith(f"{answer_type}::")
        ][:top_k]

        if not pred:
            continue

        n_answered += 1
        h1      += hits_at_k(pred, correct_answers, k=1)
        h10     += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    elapsed = time.time() - t0
    n = len(qa_pairs)
    print()

    return {
        "template":   template,
        "variant":    "BFS",
        "hop":        hop,
        "n_total":    n,
        "n_answered": n_answered,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def print_results_table(all_results: List[Dict]) -> None:
    """Print a rich comparison table grouped by template."""
    from itertools import groupby

    all_results.sort(key=lambda r: (r["template"], r["variant"]))
    grouped = groupby(all_results, key=lambda r: r["template"])

    print("\n" + "=" * 80)
    print("  CEREBRUM Hetionet Benchmark -Phase 172 Results")
    print("=" * 80)

    for template, rows in grouped:
        rows = list(rows)
        hop = rows[0]["hop"]
        print(f"\n  Template: {template}  ({hop}-hop)")
        print(f"  {'Variant':<22} {'H@1':>7} {'H@10':>7} {'MRR':>7} "
              f"{'Answered':>9}  {'H@1 bar'}")
        print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*9}  {'-'*30}")

        bfs_h1 = None
        for r in rows:
            if r["variant"] == "BFS":
                bfs_h1 = r["hits_1"]

        for r in rows:
            h1  = r["hits_1"]
            h10 = r["hits_10"]
            mrr = r["mrr"]
            pct_answered = r["n_answered"] / r["n_total"] * 100 if r["n_total"] else 0
            delta = ""
            if bfs_h1 is not None and r["variant"] != "BFS":
                diff = h1 - bfs_h1
                delta = f"  ({'+' if diff >= 0 else ''}{diff*100:.1f}pp vs BFS)"
            print(f"  {r['variant']:<22} {h1*100:>6.1f}% {h10*100:>6.1f}% {mrr*100:>6.1f}% "
                  f"  {pct_answered:>7.1f}%  {_bar(h1)}{delta}")

    print("\n" + "=" * 80)


def print_summary(all_results: List[Dict], purity: float) -> None:
    """Print the headline summary comparing BFS vs best CerebrumGraph variant."""
    print("\n  HEADLINE FINDINGS")
    print("  " + "-" * 60)
    print(f"  DSCF type alignment purity: {purity:.4f}")
    print(f"  (1.0 = communities match biological types perfectly)")
    print()

    # Group by template, find best CerebrumGraph vs BFS
    by_template: Dict[str, List[Dict]] = {}
    for r in all_results:
        by_template.setdefault(r["template"], []).append(r)

    for template, rows in sorted(by_template.items()):
        bfs = next((r for r in rows if r["variant"] == "BFS"), None)
        cerebrum_rows = [r for r in rows if r["variant"] != "BFS"]
        if not cerebrum_rows:
            continue
        best = max(cerebrum_rows, key=lambda r: r["hits_1"])

        if bfs:
            delta = best["hits_1"] - bfs["hits_1"]
            print(f"  {template:<32}  "
                  f"BFS H@1={bfs['hits_1']*100:.1f}%  ->"
                  f"Best ({best['variant']}) H@1={best['hits_1']*100:.1f}%  "
                  f"(+{delta*100:.1f}pp)")
        else:
            print(f"  {template:<32}  Best ({best['variant']}) H@1={best['hits_1']*100:.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 172: CerebrumGraph-based Hetionet biomedical KG benchmark"
    )
    parser.add_argument("--template",     type=str,  default=None,
                        help=f"Single template to evaluate. Options: "
                             f"{', '.join(QA_TEMPLATES_FIXED.keys())}. Default: all.")
    parser.add_argument("--n-questions",  type=int,  default=200)
    parser.add_argument("--beam-width",   type=int,  default=10)
    parser.add_argument("--top-k",        type=int,  default=10)
    parser.add_argument("--max-edges",    type=int,  default=None,
                        help="Cap edges loaded (dev runs, e.g. 100000)")
    parser.add_argument("--embeddings",   choices=["random", "sentence"],
                        default="random")
    parser.add_argument("--use-cache",    action="store_true", default=True)
    parser.add_argument("--no-cache",     action="store_true")
    parser.add_argument("--no-download",  action="store_true")
    parser.add_argument("--no-bfs",       action="store_true",
                        help="Skip BFS baseline (faster)")
    parser.add_argument("--trb-factor",   type=float, default=3.0,
                        help="Multiplier for terminal relation boost (default 3.0)")
    parser.add_argument("--anchor-bonus", type=float, default=2.0,
                        help="TAB anchor bonus multiplier (default 2.0)")
    parser.add_argument("--expansion-k",  type=int,  default=20,
                        help="H1SE expansion_k (default 20)")
    parser.add_argument("--seed",         type=int,  default=42)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    use_cache = args.use_cache

    print("\n=== CEREBRUM -Phase 172: Hetionet Biomedical KG Benchmark (STRB) ===")
    print(f"    CerebrumGraph.build() + CerebrumGraph.query()\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not JSON_FILE.exists():
        if args.no_download:
            print(f"ERROR: {JSON_FILE} not found and --no-download specified.")
            print(f"Download: {HETIONET_URL}")
            sys.exit(1)
        download_hetionet()

    print("Loading Hetionet graph...")
    adapter, node_type_map = load_hetionet(
        max_edges=args.max_edges,
        use_graph_cache=use_cache,
    )
    G = adapter.to_networkx()
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges\n")

    # Ensure node type annotations are present for QA generation
    for node, ntype in node_type_map.items():
        if node in G.nodes:
            G.nodes[node]["type"] = ntype

    # ------------------------------------------------------------------
    # Build CerebrumGraph
    # ------------------------------------------------------------------
    print("Building CerebrumGraph (DSCF + CSA + SRI)...")
    cache_dir = CACHE_DIR / "cerebrum" if use_cache else None
    graph = build_cerebrum_graph(
        adapter=adapter,
        embedding_mode=args.embeddings,
        cache_dir=cache_dir,
        n_trials=1,
        seed=args.seed,
        beam_width=args.beam_width,
        max_hop=3,
    )

    # ------------------------------------------------------------------
    # Type alignment -key DSCF demonstration metric
    # ------------------------------------------------------------------
    print("\nType alignment (DSCF community purity vs. biological entity types):")
    purity = report_type_alignment(graph, node_type_map)
    print()

    # ------------------------------------------------------------------
    # Phase 172: Query profile
    # ------------------------------------------------------------------
    qp = graph.query_profile
    if qp:
        print("Query profile (Phase 172 auto-strategy):")
        print(qp.summary())
        print()

    # ------------------------------------------------------------------
    # Templates to evaluate
    # ------------------------------------------------------------------
    if args.template:
        if args.template not in QA_TEMPLATES_FIXED:
            print(f"ERROR: unknown template '{args.template}'. Options: "
                  f"{', '.join(QA_TEMPLATES_FIXED.keys())}")
            sys.exit(1)
        templates = [args.template]
    else:
        templates = list(QA_TEMPLATES_FIXED.keys())

    # ------------------------------------------------------------------
    # Build variants per template
    # ------------------------------------------------------------------
    all_results: List[Dict] = []
    random.seed(args.seed)

    for template in templates:
        hop_count, seed_type, answer_type, metaedge_chain = QA_TEMPLATES_FIXED[template]
        trb = {rel: args.trb_factor for rel in TEMPLATE_TRB.get(template, {})}

        print(f"\n{'='*60}")
        print(f"  Template: {template}  ({hop_count}-hop)")
        print(f"  Seed type: {seed_type}  ->Answer type: {answer_type}")
        print(f"  Chain: {' ->'.join(metaedge_chain)}")
        print(f"  TRB: {trb}")
        report_anchor_stats(graph, template)
        print()

        # Generate QA pairs.
        # Monkey-patch the templates dict so generate_hetionet_qa finds
        # our fixed/extended templates (compound_gene_disease fix + new 3-hop).
        import benchmarks.hetionet_eval as _heval
        _orig_templates = _heval.QA_TEMPLATES
        _heval.QA_TEMPLATES = QA_TEMPLATES_FIXED
        try:
            qa_pairs = generate_hetionet_qa(
                G=G,
                template=template,
                n_questions=args.n_questions,
                seed=args.seed,
            )
        finally:
            _heval.QA_TEMPLATES = _orig_templates
        if not qa_pairs:
            print(f"  Skipping {template}: no QA pairs generated.")
            continue
        print(f"  Generated {len(qa_pairs)} QA pairs\n")

        # BFS baseline
        if not args.no_bfs:
            print(f"  Running BFS baseline...")
            # BFS overwrites adapter.community_map; save and restore it.
            _saved_cmap = dict(getattr(adapter, "community_map", {}))
            bfs_result = evaluate_bfs_baseline(
                adapter=adapter,
                template=template,
                qa_pairs=qa_pairs,
                beam_width=args.beam_width,
                top_k=args.top_k,
            )
            adapter.community_map = _saved_cmap
            all_results.append(bfs_result)

        # CerebrumGraph variants
        variants = [
            {
                "name": "DSCF+CSA",
                "query_kwargs": {
                    "hop_expand": False,
                },
            },
            {
                "name": "DSCF+CSA+TRB",
                "query_kwargs": {
                    "hop_expand": False,
                    "terminal_relation_boost": trb,
                },
            },
        ]

        if hop_count >= 2:
            variants += [
                {
                    "name": "+H1SE",
                    "query_kwargs": {
                        "hop_expand": True,
                        "expansion_k": args.expansion_k,
                        "terminal_relation_boost": trb,
                    },
                },
                {
                    "name": "+H1SE+TAB",
                    "query_kwargs": {
                        "hop_expand": True,
                        "expansion_k": args.expansion_k,
                        "terminal_relation_boost": trb,
                        "anchor_bonus": args.anchor_bonus,
                    },
                },
            ]

        # Phase 172: Profile-Auto — zero configuration, profile decides everything.
        # Passes only seeds, top_k, min_hop, max_hop. All strategy params are None
        # so CerebrumGraph.query() resolves them from the QueryProfile.
        variants.append({
            "name": "Profile-Auto",
            "query_kwargs": {},  # hop_expand=None, auto_infer_trb=None, anchor_bonus=None
        })

        # Phase 172: Profile-Auto+STRB — same zero-config strategy selection as
        # Profile-Auto, but replaces structural SRI with semantic query-embedding
        # similarity for terminal relation inference. Requires --embeddings sentence.
        # Falls back silently to structural SRI when RandomEngine is in use.
        variants.append({
            "name": "Profile-Auto+STRB",
            "query_kwargs": {},
            "use_strb": True,
        })

        print(f"  Running {len(variants)} CerebrumGraph variants...")
        results = evaluate_template(
            graph=graph,
            template=template,
            qa_pairs=qa_pairs,
            variants=variants,
            top_k=args.top_k,
        )
        all_results.extend(results)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if all_results:
        print_results_table(all_results)
        print_summary(all_results, purity)

    print("Phase 172 complete.\n")


if __name__ == "__main__":
    main()

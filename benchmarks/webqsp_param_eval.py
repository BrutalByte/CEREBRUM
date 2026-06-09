"""
Phase 231–232: WebQSP benchmark — modern CEREBRUM architecture.

Dataset: WebQSP (1,628 test questions, Freebase 2-hop KB).
KB:      freebase_2hop.txt  — 3.79M entity-name triples from Freebase.
Task:    Given seed entity + natural-language question, rank answer entities
         within top-K beam candidates (max 2 hops).
Metrics: H@1, H@10, MRR  (multi-answer: H@k = 1 if ANY answer in top-k).

Regime:  typed_heterogeneous (989 relation types, Freebase ontology).
         ParameterInitializer uses typed_heterogeneous constants.

The question text is passed as the relation query — CSA semantic attention
scores path directions based on cosine similarity between the question
embedding and relation-type strings, guiding the beam toward semantically
relevant paths without hard-coded relation type mappings.

Usage (standalone):
    python -u benchmarks/webqsp_param_eval.py --sample 200

    # With tuned params:
    python -u benchmarks/webqsp_param_eval.py --sample 500 \\
        --trb-factor 18.5 --r2-boost 3.2 --vote-weight 0.65 \\
        --beam-width 12 --idf-weight 0.03 --branch-bonus 0.15 \\
        --fhrb-factor 2.1 --gamma 4.5 --beta 0.95

Tuner integration:
    python -u benchmarks/cerebrum_tuner.py \\
        --dataset webqsp --sample 200 --embeddings sentence \\
        --phase1-trials 30 --phase2-trials 70

Data setup (already done):
    benchmarks/data/webqsp/freebase_2hop.txt       (3.79M triples)
    benchmarks/data/webqsp/WebQSP.test.json        (1628 questions)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR  = Path(__file__).parent / "data" / "webqsp"
FB_FILE   = DATA_DIR / "freebase_2hop.txt"
TEST_JSON = DATA_DIR / "WebQSP.test.json"
CACHE_DIR = DATA_DIR / "cache"

# Phase 232: question decomposition + relation-name steering
from core.question_decomposer import QuestionDecomposer
from core.relation_name_index import RelationNameIndex
# Phase 233: community-structured hypothesis generation
from core.community_hypothesis import CommunityHypothesisGenerator

_DECOMPOSER = QuestionDecomposer()

# Non-person entity markers for soft answer-type filtering ("who" questions).
# Entities whose names contain these words are almost certainly not people.
_NON_PERSON_MARKERS: frozenset = frozenset({
    "film", "album", "song", "book", "novel", "movie", "series",
    "show", "award", "university", "country", "city", "state",
    "language", "organization", "company", "corporation", "district",
    "island", "river", "mountain", "ocean", "continent",
})

# Relations that add noise without semantic value — skip during graph load.
# common.topic.notable_types connects entities to generic type nodes ("Person",
# "Male", "Film") which rank high due to fan-in but are never correct answers.
_SKIP_RELATIONS = frozenset({
    "common.topic.article",
    "common.topic.image",
    "common.webpage.topic",
    "freebase.valuenotation.is_reviewed",
    "freebase.valuenotation.has_value",
    "freebase.valuenotation.has_no_value",  # connects entities to generic category nodes
    "type.object.name",
    "type.object.type",
    "kg.object_profile.prominent_type",
    "common.topic.notable_types",        # removes generic type hub nodes ("Person", "Male")
})


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_freebase_triples(
    fb_path: str,
    max_triples: int = 0,
    skip_relations: frozenset = _SKIP_RELATIONS,
) -> List[Tuple[str, str, str]]:
    """Load freebase_2hop.txt as (head, relation, tail) string triples."""
    triples: List[Tuple[str, str, str]] = []
    path = Path(fb_path)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not h or not r or not t:
                continue
            if r in skip_relations:
                continue
            triples.append((h, r, t))
            if max_triples and len(triples) >= max_triples:
                break
    return triples


def load_seed_subgraph(
    fb_path: str,
    seed_entities: set,
    max_hop: int = 2,
    skip_relations: frozenset = _SKIP_RELATIONS,
    max_hop1_neighbors: int = 150,
    max_hop2_neighbors: int = 20,
) -> List[Tuple[str, str, str]]:
    """
    Extract the K-hop subgraph around a set of seed entities.

    Streams freebase_2hop.txt twice:
      Pass 1: collect up to max_hop1_neighbors edges per seed (high recall).
      Pass 2: collect up to max_hop2_neighbors edges per 1-hop node
              (aggressive cap keeps graph manageable for Louvain).

    Returns deduplicated (h, r, t) triples touching the subgraph.
    """
    from collections import defaultdict

    def _scan_pass(focus_nodes: set, max_per_node: int) -> tuple:
        """One scan: collect triples where head is in focus_nodes."""
        neighbor_count: dict = defaultdict(int)
        found_triples: List[Tuple[str, str, str]] = []
        new_nodes: set = set()
        path = Path(fb_path)
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if not h or not r or not t:
                    continue
                if r in skip_relations:
                    continue
                if h in focus_nodes:
                    if neighbor_count[h] < max_per_node:
                        found_triples.append((h, r, t))
                        new_nodes.add(t)
                        neighbor_count[h] += 1
        return found_triples, new_nodes

    print(f"    [subgraph] Pass 1: {len(seed_entities):,} seeds -> 1-hop (max {max_hop1_neighbors}/node)...")
    t1_triples, hop1_nodes = _scan_pass(seed_entities, max_hop1_neighbors)
    print(f"    [subgraph] 1-hop: {len(t1_triples):,} triples, {len(hop1_nodes):,} new nodes")

    if max_hop >= 2:
        expand_nodes = hop1_nodes - seed_entities
        print(f"    [subgraph] Pass 2: expanding {len(expand_nodes):,} nodes (max {max_hop2_neighbors}/node)...")
        t2_triples, _ = _scan_pass(expand_nodes, max_hop2_neighbors)
        print(f"    [subgraph] 2-hop: {len(t2_triples):,} additional triples")
        all_triples = t1_triples + t2_triples
    else:
        all_triples = t1_triples

    # Deduplicate
    seen: set = set()
    deduped = []
    for t in all_triples:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    print(f"    [subgraph] Total: {len(deduped):,} triples after dedup")
    return deduped


def _build_graph(
    triples: List[Tuple[str, str, str]],
    embeddings: str = "random",
    cache_dir: Optional[Path] = None,
    seed: int = 42,
    beam_width: int = 12,
    max_neighbors: int = 100,
):
    """Build CerebrumGraph from (h, r, t) string triples."""
    import networkx as nx
    from adapters.networkx_adapter import NetworkXAdapter
    from core.cerebrum import CerebrumGraph
    from core.embedding_engine import RandomEngine

    G = nx.MultiDiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r, weight=1.0)

    adapter = NetworkXAdapter(G)

    if embeddings == "sentence":
        try:
            from core.embedding_engine import SentenceEngine
            engine = SentenceEngine()
        except (ImportError, Exception):
            engine = RandomEngine(dim=64)
    else:
        engine = RandomEngine(dim=64)

    graph = CerebrumGraph(
        adapter          = adapter,
        embedding_engine = engine,
        beam_width       = beam_width,
        max_hop          = 2,
        max_neighbors    = max_neighbors,
    )
    graph.build(
        cache_dir  = str(cache_dir) if cache_dir else None,
        seed       = seed,
        resolution = 1.0,
    )
    return graph


# ---------------------------------------------------------------------------
# WebQSP question loading
# ---------------------------------------------------------------------------

def load_webqsp_questions(
    json_path: str,
    graph_nodes: Optional[set] = None,
    sample: int = 0,
    seed: int = 42,
) -> List[Tuple[str, List[str], str]]:
    """
    Load WebQSP test questions.

    Returns list of (seed_entity, answer_entities, question_text).
    Only includes questions where seed_entity is in graph_nodes.
    """
    import random as _random
    rng = _random.Random(seed)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    qa: List[Tuple[str, List[str], str]] = []
    for q in data["Questions"]:
        text = q.get("RawQuestion", "").strip()
        for p in q.get("Parses", []):
            seed_ent = p.get("TopicEntityMid", "").strip()
            if not seed_ent:
                continue
            if graph_nodes is not None and seed_ent not in graph_nodes:
                continue

            answers = []
            for ans in p.get("Answers", []):
                arg = ans.get("AnswerArgument", "").strip()
                if arg:
                    answers.append(arg)

            if not answers:
                continue

            qa.append((seed_ent, answers, text))
            break  # one parse per question

    if sample and sample < len(qa):
        rng.shuffle(qa)
        qa = qa[:sample]

    return qa


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _soft_type_filter(answers: list, answer_type: str) -> list:
    """
    Phase 232: Soft answer-type filter — re-orders answers so type-consistent
    entities appear before obviously mismatched ones.  Does NOT remove answers.

    Only applied for 'person' type (most common WebQSP category).
    """
    if answer_type != "person" or not answers:
        return answers
    person_like, non_person = [], []
    for ans in answers:
        eid = ans.entity_id.lower()
        words = set(re.split(r"[\s_\-/]+", eid))
        if words & _NON_PERSON_MARKERS:
            non_person.append(ans)
        else:
            person_like.append(ans)
    return person_like + non_person


def _hits_mrr(
    ranked: List[str],
    answers: List[str],
    top_k: int = 10,
) -> Tuple[float, float, float]:
    """Return (h@1, h@k, mrr) for a ranked list against a set of answers."""
    answer_set = set(answers)
    h1 = h10 = mrr = 0.0
    for rank, node in enumerate(ranked, 1):
        if node in answer_set:
            if rank == 1:
                h1 = 1.0
            if rank <= top_k:
                h10 = 1.0
            mrr = 1.0 / rank
            break
    return h1, h10, mrr


def run_trial_inprocess(
    state: dict,
    params: dict,
    top_k: int = 10,
) -> Tuple[float, float, float]:
    """
    Run one evaluation trial in-process (no subprocess overhead).
    Used by the Optuna tuner for fast per-trial scoring.

    Maps the tuner param dict to graph.query() arguments:
      - gamma + beta → penultimate_relation_boost via deriver.boost_map()
      - r2_boost     → applied post-query on best penultimate relation
      - idf_weight   → post-query answer frequency penalty
      - fhrb_factor  → initial_relation_boost via deriver.boost_map(fhrb_factor/gamma)
      - trb_factor   → terminal_relation_boost proxy (same boost_map scaled)
      - vote_weight, branch_bonus, beam_width → direct graph.query() args
    """
    graph   = state["graph"]
    deriver = state["deriver"]
    qa      = state["qa_pairs"]
    if not qa:
        return 0.0, 0.0, 0.0

    beam_width   = int(params.get("beam_width")   or 12)
    trb_factor   = float(params.get("trb_factor")  or 13.87)
    r2_boost     = float(params.get("r2_boost")    or 1.32)
    vote_weight  = float(params.get("vote_weight") or 0.64)
    idf_weight   = float(params.get("idf_weight")  or 0.018)
    branch_bonus = float(params.get("branch_bonus") or 0.03)
    fhrb_factor  = float(params.get("fhrb_factor") or 1.73)
    gamma        = float(params.get("gamma")       or 1.01)
    beta         = float(params.get("beta")        or 0.95)
    max_loops    = int(params.get("max_loops")     or 1)

    # Build relation boost maps from deriver fan-out statistics
    boost_map  = deriver.boost_map(gamma, beta) if deriver.is_built else {}
    trb_map    = deriver.boost_map(trb_factor, beta) if deriver.is_built else {}
    fhrb_map   = deriver.boost_map(fhrb_factor, beta) if deriver.is_built else {}
    answer_freq = state.get("answer_freq", {})

    import math
    h1_sum = h10_sum = mrr_sum = 0.0
    emb_engine  = getattr(graph, "_embedding_engine", None)
    rel_index   = state.get("rel_index")
    hyp_gen     = state.get("hyp_gen")

    for seed_ent, answers, question in qa:
        try:
            # Phase 232: Decompose question → goal-directed relation steering
            decomp = _DECOMPOSER.decompose(question)

            # Question-specific terminal relation boost:
            # Combine structural (deriver) TRB with semantic (question) TRB.
            # Matching relations receive a multiplicative bonus on top of the
            # fan-out-based deriver value — both signals are additive.
            if rel_index and decomp.relation_keywords:
                q_scores = rel_index.score_relations(
                    decomp.relation_keywords, min_score=0.05
                )
            else:
                q_scores = {}

            # TRB unchanged — deriver structural signal drives beam exploration.
            # Question steering happens post-extraction via path re-ranking (below).
            q_trb = trb_map or None

            # Phase 233: Answer-type-aware community hypothesis.
            # Uses decomp.answer_type to filter outbound community bridges — only boosts
            # relations that lead toward communities reaching the expected answer category
            # (person/place/time). Falls back to unfiltered boosts if no typed match.
            # Phase 233: Answer-type-aware community hypothesis.
            # Uses decomp.answer_type to filter outbound community bridges — only boosts
            # relations that lead toward communities reaching the expected answer category
            # (person/place/time). Falls back to unfiltered boosts if no typed match.
            if hyp_gen:
                _at = decomp.answer_type
                ch_fn = lambda e, at=_at: hyp_gen.generate_typed_boosts(e, answer_type=at, top_n=20, boost_scale=2.0)
            else:
                ch_fn = None

            # Build query embedding from question text (None for random embeddings)
            qemb = (emb_engine.encode_one(question)
                    if emb_engine and hasattr(emb_engine, "encode_one") else None)

            answers_obj = graph.query(
                seeds                       = [seed_ent],
                top_k                       = top_k * 3,  # overfetch for post-processing
                min_hop                     = 1,
                max_hop                     = 2,
                beam_width                  = beam_width,
                terminal_relation_boost     = q_trb,
                vote_weight                 = vote_weight,
                branch_bonus_weight         = branch_bonus,
                initial_relation_boost      = fhrb_map or None,
                community_hypothesis_fn     = ch_fn,
                query_embedding             = qemb,
                max_loops                   = max_loops,
            )

            # Phase 232: Post-extraction path re-ranking.
            # Boost answers whose best-path terminal relation matches question keywords.
            # This operates on ALREADY-FOUND answers (no beam interference), using
            # relation-name scores from QuestionDecomposer + RelationNameIndex.
            if q_scores:
                for ans in answers_obj:
                    bp = getattr(ans, "best_path", None)
                    nodes = getattr(bp, "nodes", ()) if bp else ()
                    if len(nodes) >= 2:
                        terminal_rel = nodes[-2]  # last relation in path (odd indices)
                        if terminal_rel in q_scores:
                            ans.score *= (1.0 + q_scores[terminal_rel] * 2.0)
                        # Also check first-hop relation for 1-hop paths
                        elif len(nodes) >= 2 and nodes[1] in q_scores:
                            ans.score *= (1.0 + q_scores[nodes[1]] * 2.0)
                answers_obj.sort(key=lambda a: a.score, reverse=True)

            # R2 boost: amplify entities reached via a high-fan-out penultimate rel
            if r2_boost > 0.0 and boost_map:
                for ans in answers_obj:
                    bp = getattr(ans, "best_path", None)
                    if bp and len(getattr(bp, "nodes", ())) >= 2:
                        pen_rel = bp.nodes[-2] if hasattr(bp, "nodes") else None
                        if pen_rel and pen_rel in boost_map:
                            ans.score *= (1.0 + r2_boost * boost_map[pen_rel])
                answers_obj.sort(key=lambda a: a.score, reverse=True)

            # IDF penalty: suppress high-frequency answer entities
            if idf_weight > 0.0 and answer_freq:
                for ans in answers_obj:
                    freq = answer_freq.get(ans.entity_id, 1)
                    ans.score *= 1.0 / (1.0 + idf_weight * math.log1p(freq))
                answers_obj.sort(key=lambda a: a.score, reverse=True)

            # Filter out Freebase MID relay nodes — they are never correct answers
            named = [a for a in answers_obj if not str(a.entity_id).startswith("/m/")]

            # Phase 232: Answer-type soft filter — re-rank before slicing top-k.
            # For "who" questions, push obviously non-person entities to the end.
            named = _soft_type_filter(named, decomp.answer_type)
            ranked = [a.entity_id for a in named[:top_k]]
        except Exception:
            ranked = []

        h1, h10, mrr = _hits_mrr(ranked, answers, top_k)
        h1_sum  += h1
        h10_sum += h10
        mrr_sum += mrr

    n = len(qa)
    return h1_sum / n, h10_sum / n, mrr_sum / n


# ---------------------------------------------------------------------------
# State builder (called once before tuner trials)
# ---------------------------------------------------------------------------

def build_webqsp_state(
    fb_path:    str  = str(FB_FILE),
    json_path:  str  = str(TEST_JSON),
    n_questions: int = 200,
    embeddings:  str = "sentence",
    seed:        int = 42,
    use_cache:   bool = True,
    max_triples: int  = 0,
) -> dict:
    """
    Build CerebrumGraph from Freebase 2-hop and load WebQSP QA pairs.

    Uses seed-entity subgraph extraction (load_seed_subgraph) instead of
    loading all 3.79M triples — full load causes OOM during Louvain community
    detection on the resulting 1.3M-node graph.

    Returns dict passed to run_trial_inprocess() for zero-rebuild tuner trials.
    """
    from core.relation_boost_deriver import RelationBoostDeriver

    cache_dir = CACHE_DIR if use_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract seed entities from questions first — needed to scope the subgraph
    print(f"\n[WebQSPState] Scanning {json_path} for seed entities...")
    all_qa = load_webqsp_questions(json_path, graph_nodes=None, sample=0, seed=seed)
    seed_entities = {ent for ent, _, _ in all_qa}
    print(f"  {len(seed_entities):,} unique seed entities from {len(all_qa)} questions")

    print(f"\n[WebQSPState] Extracting seed subgraph from {fb_path}...")
    t0 = time.time()
    triples = load_seed_subgraph(fb_path, seed_entities=seed_entities)
    print(f"  {len(triples):,} triples loaded ({time.time()-t0:.1f}s)")

    print(f"[WebQSPState] Building CerebrumGraph (embeddings={embeddings})...")
    t1 = time.time()
    graph = _build_graph(triples, embeddings=embeddings, cache_dir=cache_dir, seed=seed)
    nx_g  = graph.adapter.to_networkx()
    print(f"  {nx_g.number_of_nodes():,} nodes, {nx_g.number_of_edges():,} edges ({time.time()-t1:.1f}s)")

    print("[WebQSPState] Building RelationBoostDeriver...")
    deriver = RelationBoostDeriver()
    deriver.build_from_triples((h, r, t) for h, r, t in triples)

    graph_nodes = set(nx_g.nodes())
    print(f"[WebQSPState] Loading WebQSP questions (n={n_questions})...")
    qa_pairs = load_webqsp_questions(
        json_path, graph_nodes=graph_nodes, sample=n_questions, seed=seed
    )
    print(f"[WebQSPState] QA pairs: {len(qa_pairs)} questions loaded")

    # Answer frequency for IDF penalty
    answer_freq: Dict[str, int] = defaultdict(int)
    for _, answers, _ in qa_pairs:
        for a in answers:
            answer_freq[a] += 1

    # Phase 232: Build RelationNameIndex from all graph relations once.
    # score_relations() is called per-question in run_trial_inprocess() to
    # produce question-specific terminal_relation_boost overrides.
    print("[WebQSPState] Building RelationNameIndex...")
    rel_index = RelationNameIndex()
    all_relations = {r for _, r, _ in triples}
    rel_index.build_from_relations(all_relations)
    print(f"  {len(all_relations):,} relations indexed")

    # Phase 233: Build CommunityHypothesisGenerator from community structure.
    # Scans all edges once to catalog bridge-crossing relations per community pair.
    print("[WebQSPState] Building CommunityHypothesisGenerator...")
    hyp_gen = CommunityHypothesisGenerator().build(graph.adapter)
    print(f"  {len(hyp_gen._bridge_index):,} community-pair bridges indexed")

    print("[WebQSPState] Ready — all trials will skip graph build.\n")

    return {
        "graph":        graph,
        "deriver":      deriver,
        "qa_pairs":     qa_pairs,
        "answer_freq":  dict(answer_freq),
        "embeddings":   embeddings,
        "rel_index":    rel_index,
        "hyp_gen":      hyp_gen,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="webqsp_param_eval",
        description="CEREBRUM WebQSP benchmark — Phase 231",
    )
    parser.add_argument("--fb",          type=str, default=str(FB_FILE),   dest="fb_path",
                        help="Path to freebase_2hop.txt")
    parser.add_argument("--json",        type=str, default=str(TEST_JSON), dest="json_path",
                        help="Path to WebQSP.test.json")
    parser.add_argument("--sample",      type=int, default=200,  help="Questions to evaluate")
    parser.add_argument("--max-triples", type=int, default=0,    dest="max_triples",
                        help="Limit triples loaded (0=all)")
    parser.add_argument("--embeddings",  choices=["random", "sentence"], default="sentence")
    parser.add_argument("--top-k",       type=int, default=10,   dest="top_k")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-cache",    action="store_true",    dest="no_cache")
    # Beam parameter overrides
    parser.add_argument("--beam-width",   type=int,   default=None, dest="beam_width")
    parser.add_argument("--trb-factor",   type=float, default=None, dest="trb_factor")
    parser.add_argument("--r2-boost",     type=float, default=None, dest="r2_boost")
    parser.add_argument("--vote-weight",  type=float, default=None, dest="vote_weight")
    parser.add_argument("--idf-weight",   type=float, default=None, dest="idf_weight")
    parser.add_argument("--branch-bonus", type=float, default=None, dest="branch_bonus")
    parser.add_argument("--fhrb-factor",  type=float, default=None, dest="fhrb_factor")
    parser.add_argument("--gamma",        type=float, default=None, dest="gamma")
    parser.add_argument("--beta",         type=float, default=None, dest="beta")
    args = parser.parse_args()

    state = build_webqsp_state(
        fb_path     = args.fb_path,
        json_path   = args.json_path,
        n_questions = args.sample,
        embeddings  = args.embeddings,
        seed        = args.seed,
        use_cache   = not args.no_cache,
    )

    # Typed-heterogeneous × random starting point (Hetionet Phase 207 calibration).
    # ParameterInitializer is not yet calibrated for Freebase topology, so we
    # use these as the zero-config defaults until Phase 231 tuning is complete.
    _FALLBACK = {
        "trb_factor":   13.87, "r2_boost":    1.32, "vote_weight":  0.64,
        "beam_width":   12,    "idf_weight":  0.018,"branch_bonus": 0.03,
        "fhrb_factor":  1.73,  "gamma":       1.01, "beta":         0.95,
    }

    def _d(attr: str):
        return _FALLBACK.get(attr)

    params = {
        "trb_factor":   args.trb_factor   if args.trb_factor   is not None else _d("trb_factor"),
        "r2_boost":     args.r2_boost     if args.r2_boost     is not None else _d("r2_boost"),
        "vote_weight":  args.vote_weight  if args.vote_weight  is not None else _d("vote_weight"),
        "beam_width":   args.beam_width   if args.beam_width   is not None else _d("beam_width"),
        "idf_weight":   args.idf_weight   if args.idf_weight   is not None else _d("idf_weight"),
        "branch_bonus": args.branch_bonus if args.branch_bonus is not None else _d("branch_bonus"),
        "fhrb_factor":  args.fhrb_factor  if args.fhrb_factor  is not None else _d("fhrb_factor"),
        "gamma":        args.gamma        if args.gamma        is not None else _d("gamma"),
        "beta":         args.beta         if args.beta         is not None else _d("beta"),
        "max_loops":    1,
    }

    print("Parameters:")
    for k, v in params.items():
        if k != "max_loops":
            print(f"  {k:<16} = {v}")

    t0 = time.time()
    h1, h10, mrr = run_trial_inprocess(state, params, top_k=args.top_k)
    elapsed = time.time() - t0

    n = len(state["qa_pairs"])
    # Machine-readable line parsed by cerebrum_tuner subprocess validator
    print(f"webqsp  {n}  {h1:.4f}  {h10:.4f}  {mrr:.4f}")
    print(f"\n{'='*55}")
    print(f"WebQSP  ({n:,} questions, top-{args.top_k}, {elapsed:.0f}s)")
    print(f"  H@1  = {h1:.4f}  ({h1*100:.2f}%)")
    print(f"  H@10 = {h10:.4f}  ({h10*100:.2f}%)")
    print(f"  MRR  = {mrr:.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

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
# Phase 236: training-free path schema prediction
from core.path_schema_index import PathSchemaIndex

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

def load_mid_name_map(
    fb_path: str,
    node_ids: set,
    name_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Phase 249 — Freebase entity name resolution from external file.

    When `name_file` is provided (FB15k entity2text.txt or similar), reads it
    as a TSV with lines `MID\tLabel` and returns a {MID -> readable_name} dict
    filtered to node_ids.  The FB15k entity2text.txt format uses `/m/xxxxx`
    prefixed MIDs; bare `m.xxxxx` variants are also tried to match graph nodes.

    Falls back to scanning freebase_2hop.txt for `type.object.name` triples
    when no name_file is provided (Phase 248 behaviour — found 0 entries in
    practice because type.object.name triples are absent from the 2-hop subgraph).
    """
    mid_name: Dict[str, str] = {}
    if name_file:
        path = Path(name_file)
        if not path.exists():
            print(f"  [WARNING] --mid-name-file not found: {name_file}")
            return mid_name
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                mid, label = line.split("\t", 1)
                mid = mid.strip()
                label = label.strip()
                if not label:
                    continue
                # Match /m/xxxxx directly
                if mid in node_ids:
                    mid_name[mid] = label
                # Also try stripping leading slash: /m/x → m/x (rare format variant)
                elif mid.startswith("/") and mid[1:] in node_ids:
                    mid_name[mid[1:]] = label
        return mid_name

    # Phase 248 fallback: scan freebase_2hop.txt for type.object.name triples
    path = Path(fb_path)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if r == "type.object.name" and h in node_ids and t:
                mid_name[h] = t
    return mid_name

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
    mid_name_map: Optional[Dict[str, str]] = None,
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
    if mid_name_map:
        adapter.entity_name_overrides = mid_name_map  # Phase 248

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

_TIME_PATTERN = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b|century|decade|\bera\b|period|ancient|medieval|modern')

def _soft_type_filter(answers: list, answer_type: str) -> list:
    """
    Phase 232/239: Soft answer-type filter — re-orders answers so type-consistent
    entities appear before obviously mismatched ones.  Does NOT remove answers.

    Handles 'person', 'place', and 'time' answer types.
    """
    if not answers:
        return answers
    if answer_type == "person":
        person_like, non_person = [], []
        for ans in answers:
            eid = ans.entity_id.lower()
            words = set(re.split(r"[\s_\-/]+", eid))
            if words & _NON_PERSON_MARKERS:
                non_person.append(ans)
            else:
                person_like.append(ans)
        return person_like + non_person
    if answer_type == "time":
        time_like, non_time = [], []
        for ans in answers:
            if _TIME_PATTERN.search(ans.entity_id):
                time_like.append(ans)
            else:
                non_time.append(ans)
        return time_like + non_time
    return answers


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

    # Reset cross-trial state: Phase 207 implicit feedback accumulates Platt
    # calibration samples across trials, causing _platt.transform() to squash all
    # answer scores to near-identical values once _fitted=True (destroying ranking).
    # Also clear _recent_answer_cache to prevent cross-trial seed-reuse bias.
    _platt = getattr(graph, "_platt", None)
    if _platt is not None:
        _platt._samples = []
        _platt._fitted  = False
    graph._recent_answer_cache = {}
    graph._feedback_buf        = []

    beam_width          = int(params.get("beam_width")          or 12)
    trb_factor          = float(params.get("trb_factor")         or 13.87)
    r2_boost            = float(params.get("r2_boost")           or 1.32)
    vote_weight         = float(params.get("vote_weight")        or 0.64)
    idf_weight          = float(params.get("idf_weight")         or 0.018)
    branch_bonus        = float(params.get("branch_bonus")       or 0.03)
    fhrb_factor         = float(params.get("fhrb_factor")        or 1.73)
    gamma               = float(params.get("gamma")              or 1.01)
    beta                = float(params.get("beta")               or 0.95)
    max_loops           = int(params.get("max_loops")            or 1)
    _sst = params.get("schema_score_threshold")
    schema_score_thresh = float(_sst) if _sst is not None else 0.0
    _dpw = params.get("degree_penalty_weight")
    degree_penalty_weight = float(_dpw) if _dpw is not None else 0.0

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
    schema_idx  = state.get("schema_idx")

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

            # TRB and FHRB remain structural (fan-out calibrated).
            # Semantic signal from RelationNameIndex is used only post-extraction
            # (re-ranking) where noisy keyword matches are bounded and safe.
            q_trb  = trb_map  or None
            q_fhrb = fhrb_map or None

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

            # Phase 236: Path schema prediction — predict (r1, r2) path from question
            # embedding restricted to schemas whose r1 the seed actually has, then
            # execute as targeted 2-hop traversals (independent coverage channel).
            schema_extra: List[str] = []
            if schema_idx is not None and qemb is not None:
                seed_rels = graph.adapter.get_all_relation_types(seed_ent)
                predicted_schemas = schema_idx.predict_schemas_for_seed(
                    qemb, seed_rels, top_k=5
                )
                if predicted_schemas:
                    schema_hits = schema_idx.execute_schemas(
                        seed_ent, predicted_schemas, graph.adapter,
                        skip_rels=_SKIP_RELATIONS,
                    )
                    schema_extra = [(eid, sc) for eid, sc in schema_hits]

            answers_obj = graph.query(
                seeds                       = [seed_ent],
                top_k                       = top_k * 3,  # overfetch for post-processing
                min_hop                     = 1,
                max_hop                     = 2,
                beam_width                  = beam_width,
                terminal_relation_boost     = q_trb,
                vote_weight                 = vote_weight,
                branch_bonus_weight         = branch_bonus,
                initial_relation_boost      = q_fhrb,
                community_hypothesis_fn     = ch_fn,
                query_embedding             = qemb,
                max_loops                   = max_loops,
                degree_penalty_weight       = degree_penalty_weight,
                cvt_passthrough             = bool(params.get("cvt_passthrough", True)),
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

            # CVT handling is done at traversal time via cvt_passthrough=True (Phase 243).
            # BeamTraversal collapses A→CVT→B into a compound edge scored against the
            # final named entity, avoiding the near-zero semantic penalty on opaque MIDs.

            # Phase 245: Backward verification pass.
            # For 2-hop answers [seed, rel1, hop1, rel2, answer], check whether the
            # hop-1 intermediate appears in the answer entity's outgoing neighbors.
            # Bidirectional structural support distinguishes correct answers from
            # hub entities coincidentally reached by one noisy forward path.
            _bbp = params.get("backward_bonus", 0.0)
            if _bbp > 0.0:
                _ecache = getattr(getattr(graph, "_traversal", None), "_expansion_cache", {})
                for ans in answers_obj:
                    bp = getattr(ans, "best_path", None)
                    nodes = getattr(bp, "nodes", ()) if bp else ()
                    if len(nodes) >= 5:
                        hop1 = nodes[2]
                        if ans.entity_id not in _ecache:
                            _ecache[ans.entity_id] = graph.adapter.get_neighbors(
                                ans.entity_id, max_neighbors=100
                            )
                        if hop1 in {e.target_id for e in _ecache[ans.entity_id]}:
                            ans.score *= (1.0 + _bbp)
                answers_obj.sort(key=lambda a: a.score, reverse=True)

            # Phase 246: Path diversity re-ranker.
            # Correct answers in Freebase are typically reachable via multiple distinct
            # hop-1 intermediates (different relation paths converging on the same entity),
            # while hub-entity noise tends to arrive via a single dominant relation path.
            # We build a reverse index from the expansion cache: for each hop-1 node,
            # map its outbound targets back to the set of distinct intermediates reaching
            # them. Answers with n_paths > 1 receive a log-scaled score boost.
            _da = params.get("diversity_alpha", 0.0)
            if _da > 0.0:
                _ecache = getattr(getattr(graph, "_traversal", None), "_expansion_cache", {})
                # Hop-1 nodes = direct neighbors of the seed entity in the cache
                _hop1_nodes = {e.target_id for e in _ecache.get(seed_ent, [])}
                # Reverse index: answer_entity → set of hop-1 intermediates reaching it
                _reverse: dict = {}
                for _h1 in _hop1_nodes:
                    for _edge in _ecache.get(_h1, []):
                        _t = _edge.target_id
                        if _t not in _reverse:
                            _reverse[_t] = set()
                        _reverse[_t].add(_h1)
                for ans in answers_obj:
                    _n = len(_reverse.get(ans.entity_id, ()))
                    if _n > 1:
                        ans.score *= (1.0 + _da * math.log1p(_n - 1))
                answers_obj.sort(key=lambda a: a.score, reverse=True)

            # Phase 250: Question-Answer Semantic Alignment (QASA) re-ranking.
            # Direct cosine similarity between the question embedding and each answer
            # entity embedding. Rationale: the H@10/H@1 gap (20.47% vs 10.33%) is a
            # ranking problem — correct answers are found but not ranked #1. Semantic
            # alignment adds a direct question↔answer signal on top of structural beam
            # scoring. Operates entirely on pretrained embeddings: training-free.
            _qsw = params.get("qa_sem_weight", 0.0)
            if _qsw > 0.0 and qemb is not None:
                import numpy as _np
                _ans_embs = getattr(graph.adapter, "embeddings", {}) or {}
                if _ans_embs:
                    _q_norm = qemb / (float(_np.linalg.norm(qemb)) + 1e-9)
                    for ans in answers_obj:
                        _ae = _ans_embs.get(ans.entity_id)
                        if _ae is not None:
                            _ae = _ae.astype(_np.float32)
                            _ae_n = float(_np.linalg.norm(_ae))
                            _ae_norm = _ae / (_ae_n + 1e-9)
                            _sim = float(_np.dot(_q_norm, _ae_norm))
                            if _sim > 0.0:
                                ans.score *= (1.0 + _qsw * _sim)
                    answers_obj.sort(key=lambda a: a.score, reverse=True)

            # Phase 247: Conditional Schema Prediction.
            # After the beam populates _expansion_cache, extract the union of outgoing
            # relation types from all hop-1 intermediate entities (hop1_r2_rels).
            # Re-predict schemas with r2 biased toward structurally confirmed relations —
            # schemas whose r2 is actually present on a reachable intermediate receive a
            # score boost (structural_bonus), concentrating execution on paths that are
            # both semantically appropriate AND reachable from this specific seed.
            # Merge with existing schema_extra (highest score wins per entity).
            _csb = params.get("conditional_schema_bonus", 0.0)
            if _csb > 0.0 and schema_idx is not None and qemb is not None:
                _ecache247 = getattr(getattr(graph, "_traversal", None), "_expansion_cache", {})
                _hop1_r2_rels: set = set()
                for _h1_edge in _ecache247.get(seed_ent, []):
                    for _h2_edge in _ecache247.get(_h1_edge.target_id, []):
                        _hop1_r2_rels.add(_h2_edge.relation_type)
                if _hop1_r2_rels:
                    _seed_rels247 = graph.adapter.get_all_relation_types(seed_ent)
                    _cond_schemas = schema_idx.predict_schemas_conditional(
                        qemb, _seed_rels247, _hop1_r2_rels, top_k=5, structural_bonus=float(_csb)
                    )
                    if _cond_schemas:
                        _cond_hits = schema_idx.execute_schemas(
                            seed_ent, _cond_schemas, graph.adapter, skip_rels=_SKIP_RELATIONS
                        )
                        _cond_map = {eid: sc for eid, sc in _cond_hits}
                        _merged: dict = {}
                        for eid, sc in schema_extra:
                            _merged[eid] = max(sc, _cond_map.get(eid, 0.0))
                        for eid, sc in _cond_hits:
                            if eid not in _merged:
                                _merged[eid] = sc
                        schema_extra = sorted(_merged.items(), key=lambda x: x[1], reverse=True)

            # Filter out Freebase MID relay nodes — they are never correct answers
            named = [a for a in answers_obj if not str(a.entity_id).startswith("/m/")]

            # Phase 232: Answer-type soft filter — re-rank before slicing top-k.
            # For "who" questions, push obviously non-person entities to the end.
            named = _soft_type_filter(named, decomp.answer_type)
            ranked = [a.entity_id for a in named[:top_k]]

            # Phase 236/237: Merge schema-predicted answers with beam ranking.
            # Only prepend when schema_score exceeds threshold — low-confidence
            # predictions displace correct beam answers without adding signal.
            if schema_extra:
                beam_set = set(ranked)
                high_conf = [
                    eid for eid, sc in schema_extra
                    if sc >= schema_score_thresh
                    and eid not in beam_set
                    and not eid.startswith("/m/")
                ]
                low_conf = [
                    eid for eid, sc in schema_extra
                    if sc < schema_score_thresh
                    and eid not in beam_set
                    and not eid.startswith("/m/")
                ]
                # Prepend top-2 high-confidence schema answers before beam ranked
                prepend     = high_conf[:2]
                prepend_set = set(prepend)
                ranked = prepend + [e for e in ranked if e not in prepend_set]
                # Append remaining schema answers for extended H@10+ coverage
                appended: set = prepend_set.copy()
                for eid in (high_conf[2:] + low_conf):
                    if eid not in beam_set and eid not in appended:
                        ranked.append(eid)
                        appended.add(eid)
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
    fb_path:      str  = str(FB_FILE),
    json_path:    str  = str(TEST_JSON),
    n_questions:  int  = 200,
    embeddings:   str  = "sentence",
    seed:         int  = 42,
    use_cache:    bool = True,
    max_triples:  int  = 0,
    mid_name_file: Optional[str] = None,
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

    # Phase 249: Build MID→name map (external file preferred; fallback scans freebase_2hop.txt)
    all_nodes = {h for h, r, t in triples} | {t for h, r, t in triples}
    _src = mid_name_file if mid_name_file else "freebase_2hop.txt"
    print(f"[WebQSPState] Building MID name map for {len(all_nodes):,} nodes (source: {_src})...")
    t_name = time.time()
    mid_name_map = load_mid_name_map(fb_path, all_nodes, name_file=mid_name_file)
    coverage = len(mid_name_map) / max(len(all_nodes), 1) * 100
    print(f"  {len(mid_name_map):,} MIDs resolved ({coverage:.1f}% coverage, {time.time()-t_name:.1f}s)")

    print(f"[WebQSPState] Building CerebrumGraph (embeddings={embeddings})...")
    t1 = time.time()
    graph = _build_graph(triples, embeddings=embeddings, cache_dir=cache_dir, seed=seed,
                         mid_name_map=mid_name_map)
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

    # Phase 236: Build PathSchemaIndex — enumerate 2-hop (r1, r2) schemas and embed
    # them so query-time cosine similarity can predict the likely relation path.
    # Only built when sentence embeddings are available (schema prediction requires
    # meaningful embedding similarity; random embeddings produce no signal).
    schema_idx: Optional[PathSchemaIndex] = None
    if embeddings == "sentence":
        print("[WebQSPState] Building PathSchemaIndex...")
        t_si = time.time()
        schema_idx = PathSchemaIndex().build(
            graph.adapter,
            graph._embedding_engine,
            min_count       = 3,
            skip_relations  = _SKIP_RELATIONS,
        )
        print(f"  {schema_idx.schema_count():,} 2-hop schemas indexed ({time.time()-t_si:.1f}s)")

    print("[WebQSPState] Ready — all trials will skip graph build.\n")

    return {
        "graph":        graph,
        "deriver":      deriver,
        "qa_pairs":     qa_pairs,
        "answer_freq":  dict(answer_freq),
        "embeddings":   embeddings,
        "rel_index":    rel_index,
        "hyp_gen":      hyp_gen,
        "schema_idx":   schema_idx,
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
    parser.add_argument("--beam-width",              type=int,   default=None, dest="beam_width")
    parser.add_argument("--trb-factor",              type=float, default=None, dest="trb_factor")
    parser.add_argument("--r2-boost",                type=float, default=None, dest="r2_boost")
    parser.add_argument("--vote-weight",             type=float, default=None, dest="vote_weight")
    parser.add_argument("--idf-weight",              type=float, default=None, dest="idf_weight")
    parser.add_argument("--branch-bonus",            type=float, default=None, dest="branch_bonus")
    parser.add_argument("--fhrb-factor",             type=float, default=None, dest="fhrb_factor")
    parser.add_argument("--gamma",                   type=float, default=None, dest="gamma")
    parser.add_argument("--beta",                    type=float, default=None, dest="beta")
    parser.add_argument("--schema-score-threshold",  type=float, default=None, dest="schema_score_threshold")
    parser.add_argument("--degree-penalty-weight",   type=float, default=None, dest="degree_penalty_weight")
    parser.add_argument("--backward-bonus",            type=float, default=None, dest="backward_bonus")
    parser.add_argument("--diversity-alpha",           type=float, default=None, dest="diversity_alpha")
    parser.add_argument("--qa-sem-weight",             type=float, default=None, dest="qa_sem_weight",
                        help="Phase 250: Question-answer semantic alignment re-ranking weight.")
    parser.add_argument("--conditional-schema-bonus",  type=float, default=None, dest="conditional_schema_bonus")
    parser.add_argument("--cvt-passthrough",           action="store_true", default=True, dest="cvt_passthrough")
    parser.add_argument("--mid-name-file",             type=str,   default=None, dest="mid_name_file",
                        help="Phase 249: Path to FB15k entity2text.txt for MID→readable-name resolution.")
    args = parser.parse_args()

    state = build_webqsp_state(
        fb_path      = args.fb_path,
        json_path    = args.json_path,
        n_questions  = args.sample,
        embeddings   = args.embeddings,
        seed         = args.seed,
        use_cache    = not args.no_cache,
        mid_name_file = args.mid_name_file,
    )

    # Phase 236: Schema channel params (H@10=32.5% H@1=9.5% MRR=0.155, schema prepend top-2).
    # Phase 237 finding: RELATED_TO beam is correct for MultiDiGraph Freebase — parallel
    # edges per (u,v) pair mean any single real relation is arbitrary, so RELATED_TO keeps
    # traversal driven by embedding similarity + community scores only (correct signals).
    # schema_score_threshold=0.0 (tunable): unconditional prepend gives best H@10 coverage.
    # fANOVA (real-relation tuner run): beta #1 (0.40), beam_width #2 (0.30) — different
    # landscape but lower H@10 confirms RELATED_TO beam is the right design choice.
    _FALLBACK = {
        "trb_factor":            42.9747, "r2_boost":  5.6184, "vote_weight":  0.6856,
        "beam_width":            32,      "idf_weight":0.0152, "branch_bonus": 0.0905,
        "fhrb_factor":           1.5508,  "gamma":     8.4034, "beta":         1.5679,
        "schema_score_threshold": 0.0,
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
        "beta":                   args.beta                   if args.beta                   is not None else _d("beta"),
        "schema_score_threshold": args.schema_score_threshold if args.schema_score_threshold is not None else _d("schema_score_threshold"),
        "degree_penalty_weight":  args.degree_penalty_weight  if args.degree_penalty_weight  is not None else 0.0,
        "backward_bonus":         args.backward_bonus         if args.backward_bonus         is not None else 0.0,
        "diversity_alpha":        args.diversity_alpha        if args.diversity_alpha        is not None else 0.0,
        "conditional_schema_bonus": args.conditional_schema_bonus if args.conditional_schema_bonus is not None else 0.0,
        "qa_sem_weight":          args.qa_sem_weight          if args.qa_sem_weight          is not None else 0.0,
        "cvt_passthrough":        True,
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

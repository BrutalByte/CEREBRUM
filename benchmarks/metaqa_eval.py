"""
MetaQA benchmark evaluation for CEREBRUM.

Dataset
-------
MetaQA: Movie question-answering over a knowledge graph.
  KB     : 134,741 triples | 43,234 entities | 9 relation types
  1-hop  :  9,947 test questions
  2-hop  : 14,872 test questions
  3-hop  : 14,274 test questions
  Source : https://github.com/yuyuz/MetaQA (Zhang et al., ICLR 2018)
  License: MIT

Metrics
-------
  Hits@1   fraction of questions where correct answer is the top-1 result
  Hits@10  fraction of questions where correct answer is in top-10 results
  MRR      Mean Reciprocal Rank

Question format (vanilla splits)
---------------------------------
  <question with [seed entity] in brackets>\\t<answer1>|<answer2>|...

KB format
---------
  subject|relation|object   (pipe-delimited, UTF-8)

Usage
-----
  # All three hop levels, full test set:
  python -m benchmarks.metaqa_eval

  # Single hop level:
  python -m benchmarks.metaqa_eval --hop 3

  # Quick development run (500 questions per hop):
  python -m benchmarks.metaqa_eval --sample 500

  # Sentence-transformer embeddings (more accurate, requires sentence-transformers):
  python -m benchmarks.metaqa_eval --embeddings sentence

  # Wider beam for better recall:
  python -m benchmarks.metaqa_eval --beam-width 20

Notes
-----
  - KB is loaded as undirected (standard MetaQA practice — traversal works
    in both directions without needing InverseRule).
  - With --embeddings random the CSA alpha (semantic) term is noise; attention
    is driven by community structure (beta) alone.
  - With --embeddings sentence entity names ("Tom Hanks", "The Green Mile")
    produce meaningful cosine-similarity attention.
  - Community detection on 43K nodes takes ~60s on first run; cached after.
"""

import argparse
import csv
import math
import multiprocessing
import os

# Optional experiment tracking — imported lazily when --mlflow / --wandb used
_mlflow = None
_wandb  = None
import pickle
import csv
import re
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cerebrum import CerebrumGraph

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parent / "data" / "metaqa"
CACHE_DIR = DATA_DIR / "cache"
KB_FILE   = DATA_DIR / "kb.txt"

QA_FILES = {
    1: DATA_DIR / "1-hop" / "vanilla" / "qa_test.txt",
    2: DATA_DIR / "2-hop" / "vanilla" / "qa_test.txt",
    3: DATA_DIR / "3-hop" / "vanilla" / "qa_test.txt",
}

TRAIN_FILES = {
    1: DATA_DIR / "1-hop" / "vanilla" / "qa_train.txt",
    2: DATA_DIR / "2-hop" / "vanilla" / "qa_train.txt",
    3: DATA_DIR / "3-hop" / "vanilla" / "qa_train.txt",
}

# ---------------------------------------------------------------------------
# QA file parsing  (benchmark-specific, not part of CerebrumGraph)
# ---------------------------------------------------------------------------

def load_qa(
    hop:              int,
    sample:           Optional[int] = None,
    seed:             int           = 42,
    include_question: bool          = False,
) -> List[Tuple]:
    """
    Load QA pairs for a given hop level.

    Returns list of (seed_entity, [answer_entity, ...]) 2-tuples by default.
    When include_question=True, returns 3-tuples
    (seed_entity, [answer_entity, ...], question_text).
    """
    path = QA_FILES[hop]
    pairs: List[Any] = []
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

            m = re.search(r"\[(.+?)\]", question)
            if not m:
                continue
            seed_entity = m.group(1).strip()
            if seed_entity and answers:
                if include_question:
                    clean_q = re.sub(r"\[(.+?)\]", r"\1", question)
                    pairs.append((seed_entity, answers, clean_q))
                else:
                    pairs.append((seed_entity, answers))

    if sample is not None and sample < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample)
    return pairs


# ---------------------------------------------------------------------------
# Metrics  (benchmark-specific)
# ---------------------------------------------------------------------------

def hits_at_k(answers: List[str], correct: List[str], k: int) -> int:
    return int(any(c in set(answers[:k]) for c in correct))


def reciprocal_rank(answers: List[str], correct: List[str]) -> float:
    correct_set = set(correct)
    for rank, ans in enumerate(answers, 1):
        if ans in correct_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# RelationPathPrior  (optional, 2-hop and 3-hop only)
# ---------------------------------------------------------------------------

def build_or_load_prior(
    hop:           int,
    graph:         CerebrumGraph,
    beam_width:    int,
    use_cache:     bool,
    force_rebuild: bool,
    train_sample:  int = 20000,
):
    """
    Build a RelationPathPrior for the given hop level from the training split.

    The prior is learned from (paths, correct_answers) pairs on training data —
    it counts which relation-sequence patterns most often reach correct answers.
    This is a frequency heuristic over the *search*, not a modification to the
    *graph structure* or the *answer claims*.

    Only applied for 2-hop and 3-hop: the 1-hop prior has only 9 unique patterns
    (one per relation type) and does not improve results.

    Caches to CACHE_DIR/prior_{hop}hop.pkl.
    """
    from reasoning.relation_path_prior import RelationPathPrior
    from reasoning.traversal import BeamTraversal

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"prior_{hop}hop.pkl"

    if use_cache and not force_rebuild and cache_file.exists():
        print(f"    Loading cached {hop}-hop prior from {cache_file.name}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    path = TRAIN_FILES[hop]
    if not path.exists():
        print(f"    Training file not found: {path} — skipping prior")
        return None

    # Load training QA — read from train file, not test file.
    # Cap at train_sample to keep prior-building time under ~2 minutes.
    train_path = TRAIN_FILES[hop]
    all_train: List[Tuple] = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0]
            answers  = [a.strip() for a in parts[1].split("|") if a.strip()]
            m = re.search(r"\[(.+?)\]", question)
            if not m:
                continue
            seed_e = m.group(1).strip()
            if seed_e and answers:
                all_train.append((seed_e, answers))

    if train_sample < len(all_train):
        rng = random.Random(99)   # fixed seed — different from eval seed 42
        train_pairs = rng.sample(all_train, train_sample)
    else:
        train_pairs = all_train
    G = graph.adapter.to_networkx()

    n_train = len(train_pairs)
    print(f"    Building {hop}-hop prior from {n_train:,} training questions "
          f"(of {len(all_train):,} total)...")
    prior  = RelationPathPrior(smoothing=1.0, max_len=3, min_count=3)
    trav   = BeamTraversal(
        adapter       = graph.adapter,
        csa_engine    = graph._csa,
        beam_width    = beam_width,
        max_hop       = hop,
        max_neighbors = 100,
    )
    t0 = time.time()
    for seed_e, correct in train_pairs:
        if seed_e not in G:
            continue
        paths = trav.traverse([seed_e])
        prior.update(paths, set(correct))
    prior.freeze()
    elapsed = time.time() - t0
    print(f"    {len(prior._total):,} unique sequences in {elapsed:.1f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(prior, f)
    return prior


# ---------------------------------------------------------------------------
# Phase 146: Zero-shot relation detection from question text
# ---------------------------------------------------------------------------

# Ordered keyword rules: first match wins.
_RELATION_KEYWORDS = [
    ("directed_by",    ["direct", "director"]),
    ("written_by",     ["writ", "wrote", "writer", "written", "screenwriter", "screenplay"]),
    ("starred_actors", ["star", "actor", "actress", "appear", "acted"]),
    ("has_genre",      ["genre", "type of film", "types", "what type"]),
    ("has_tags",       ["tag", "keyword"]),
    ("in_language",    ["language", "spoken", "in which language"]),
    ("release_year",   ["year", "release", "when was", "in which year"]),
    ("has_imdb_rating",["rating", "imdb", "score"]),
    ("has_imdb_votes", ["votes"]),
]


def detect_target_relation(
    question: str,
    kb_relations: List[str],
    prefix_words: int = 4,
    suffix_words: int = 6,
    exclude_relation: Optional[str] = None,
) -> Optional[str]:
    """
    Map a question string to a MetaQA KB relation type via keyword matching.

    Phase 152 fix: Two-pass detection — prefix first, suffix as fallback only.
    Phase 153 fix: Three targeted pre-passes before the two-pass keyword scan:
      (1) "when ..." → release_year — "when did the films STARRED by X release"
          has "star" in the prefix which would fire starred_actors first.
      (2) Answer type at end of sentence ("...in which LANGUAGES/GENRES/YEARS")
          — scan ONLY the terminal 1-3 words before entity contamination hits.
      (3) "what are the primary RELATION" — prefix_words=6 only for "what are"
          starters, safe because "share actors" never appears there.

    Pass 1: prefix (first 4 or 6 words). Pass 2: suffix if no match.
    """
    words = question.lower().split()

    # Pre-pass 1: "when..." is always release_year (unambiguous temporal marker).
    if words and words[0] == "when" and "release_year" in kb_relations:
        return "release_year"

    # Pre-pass 2: "...in which TERM" suffix — the answer type is the LAST WORD
    # when the template ends with "in which X" (genres, languages, years, etc.).
    # Check only the last word to avoid entity-name contamination of the suffix.
    if len(words) >= 2 and words[-2] in ("which", "what"):
        last_word = words[-1]
        for relation, keywords in _RELATION_KEYWORDS:
            if relation not in kb_relations:
                continue
            if any(kw in last_word for kw in keywords):
                return relation

    # Pre-pass 3: "what are the primary TERM" — extend prefix only for "what
    # are/is" starters (safe because "share actors" never follows "what are").
    # Pre-pass 4: "who is listed as RELATION_TYPE of films ..." — the answer-type
    # keyword is at word[4] (index 4), just outside the default prefix_words=4 window.
    # Without this pass, the suffix catches "starred/actors" from the path description
    # and misidentifies directed_by/written_by questions as starred_actors.
    if len(words) >= 5 and words[0] == "who" and words[1] == "is" and words[2] == "listed" and words[3] == "as":
        for relation, keywords in _RELATION_KEYWORDS:
            if relation not in kb_relations:
                continue
            if any(kw in words[4] for kw in keywords):
                return relation

    effective_prefix = 6 if (len(words) >= 2 and words[0] == "what" and words[1] in ("are", "is")) else prefix_words
    prefix = " ".join(words[:effective_prefix])
    suffix = " ".join(words[-suffix_words:])

    for scan in (prefix, suffix):
        for relation, keywords in _RELATION_KEYWORDS:
            if relation not in kb_relations:
                continue
            if relation == exclude_relation:
                continue
            if any(kw in scan for kw in keywords):
                return relation
    return None


# ---------------------------------------------------------------------------
# Phase 182: Question-level multiprocessing — worker globals and functions
# ---------------------------------------------------------------------------

# Module-level globals populated by _worker_init in each spawned process.
_W_GRAPH               = None
_W_QUERY_ENGINE        = None
_W_RELATION_ANSWER_SET = None
_W_ANSWER_FREQ         = None
_W_PRIORS              = None
_W_R2_FOR_R3           = None
_W_FAN_OUT             = None
_W_KB_RELATIONS        = None


def _worker_init(graph_args: dict, shared: dict) -> None:
    """Initializer called once per spawned worker process."""
    global _W_GRAPH, _W_QUERY_ENGINE, _W_RELATION_ANSWER_SET, _W_ANSWER_FREQ
    global _W_PRIORS, _W_R2_FOR_R3, _W_FAN_OUT, _W_KB_RELATIONS

    _W_GRAPH = CerebrumGraph.from_kb(
        graph_args["kb_file"],
        sep="|", directed=False,
        embeddings=graph_args["embeddings"],
        beam_width=graph_args["beam_width"],
        max_hop=3, max_neighbors=100,
    )
    _W_GRAPH.build(
        cache_dir=graph_args["cache_dir"],
        min_community_size=graph_args["min_community_size"],
        force_rebuild=False,
        seed=graph_args["seed"],
        use_graphsage=graph_args["graphsage"],
        use_kge=graph_args["kge"],
        kge_blend=graph_args["kge_blend"],
        kge_epochs=graph_args["kge_epochs"],
        community_engine=graph_args["community_engine"],
    )
    _W_QUERY_ENGINE = (
        _W_GRAPH._embedding_engine if graph_args["embeddings"] == "sentence" else None
    )

    # Build KB relation vocabulary from the worker's own graph
    try:
        _W_KB_RELATIONS = list({
            e.relation_type
            for eid in list(_W_GRAPH.adapter._G.nodes())[:500]
            for e in _W_GRAPH.adapter.get_neighbors(eid)
        })
    except Exception:
        _W_KB_RELATIONS = []

    _W_RELATION_ANSWER_SET = shared["relation_answer_set"]
    _W_ANSWER_FREQ         = shared["answer_freq"]
    _W_PRIORS              = shared["priors"]
    _W_R2_FOR_R3           = shared["r2_for_r3"]
    _W_FAN_OUT             = shared["fan_out"]


def _worker_process_question(task: tuple) -> tuple:
    """
    Process one question. Returns (q_idx, pred_or_None, diag_or_None).
    pred_or_None is None when the question produces no answers (skipped).
    Reads shared state from module globals set by _worker_init.
    """
    try:
        (q_idx, seed, correct_answers, question_text,
         hop, top_k, beam_width, eval_min_hop,
         vote_weight_3hop, branch_bonus_weight, trb_factor_3hop, idf_weight,
         hop2_beam_width, r2_boost, r2_boost_map, min_filter_size,
         expansion_k, structural_trb, anchor_bonus, pss_weight, fhrb_factor,
         do_diag, do_diag_jsonl) = task

        graph               = _W_GRAPH
        embedding_engine    = _W_QUERY_ENGINE
        kb_relations        = _W_KB_RELATIONS or []
        relation_answer_set = _W_RELATION_ANSWER_SET or {}
        answer_freq_map     = _W_ANSWER_FREQ
        relation_prior      = (_W_PRIORS.get(hop) if _W_PRIORS else None)
        r2_map              = _W_R2_FOR_R3 or {}
        fan_out             = _W_FAN_OUT or {}

        _is_3hop   = (hop == 3)
        _raw_top_k = 100 if _is_3hop else top_k

        # Encode question text as query embedding
        query_emb = None
        if question_text and embedding_engine is not None:
            try:
                encode_query_fn = getattr(embedding_engine, "encode_query", None)
                if encode_query_fn is not None:
                    query_emb = encode_query_fn([question_text])[0]
                else:
                    q_vecs    = embedding_engine.encode_entities({"__q__": question_text})
                    query_emb = q_vecs.get("__q__")
            except Exception:
                pass

        # Terminal Relation Boost
        _trb: Dict[str, float] = {}
        if not structural_trb and question_text and kb_relations:
            detected = detect_target_relation(question_text, kb_relations)
            if detected:
                boost_factor = trb_factor_3hop if _is_3hop else 3.0
                _trb = {detected: boost_factor}

        # Penultimate Relation Boost
        _prb: Dict[str, float] = {}
        if _is_3hop and _trb and r2_map:
            _r2 = r2_map.get(next(iter(_trb)))
            if _r2:
                _prb = {_r2: next(iter(_trb.values())) ** 0.5}

        # First-Hop Relation Boost (Phase 180)
        _irb: Optional[Dict[str, float]] = None
        if _is_3hop and fhrb_factor > 0.0 and question_text and kb_relations:
            _r1_candidate = detect_target_relation(
                question_text, kb_relations,
                exclude_relation=next(iter(_trb)) if _trb else None,
            )
            if _r1_candidate:
                _irb = {_r1_candidate: fhrb_factor}

        # Per-hop beam widths
        _beam_widths: Optional[Dict[int, int]] = None
        if _is_3hop and hop2_beam_width is not None and hop2_beam_width > beam_width:
            _beam_widths = {1: hop2_beam_width}

        answers_obj = graph.query(
            seeds                        = [seed],
            top_k                        = _raw_top_k,
            min_hop                      = eval_min_hop,
            max_hop                      = hop,
            hop_expand                   = (hop >= 2),
            query_embedding              = query_emb,
            relation_prior               = relation_prior,
            terminal_relation_boost      = _trb,
            penultimate_relation_boost   = _prb,
            vote_weight                  = vote_weight_3hop if _is_3hop else 0.45,
            branch_bonus_weight          = branch_bonus_weight if _is_3hop else 0.0,
            beam_widths                  = _beam_widths,
            expansion_k                  = expansion_k if _is_3hop else None,
            auto_infer_terminal_relation = structural_trb and _is_3hop,
            anchor_bonus                 = anchor_bonus if _is_3hop else None,
            fan_out                      = fan_out if (_is_3hop and pss_weight > 0.0) else None,
            weight_specificity           = pss_weight if _is_3hop else 0.0,
            initial_relation_boost       = _irb,
        )

        # IDF hub-entity penalty
        if _is_3hop and _trb and idf_weight > 0.0 and answer_freq_map:
            _detected_r3 = next(iter(_trb))
            _freq_ctr    = answer_freq_map.get(_detected_r3)
            _n_unique    = len(relation_answer_set.get(_detected_r3, set()))
            if _freq_ctr and _n_unique > 100:
                for _ans in answers_obj:
                    _freq = _freq_ctr.get(_ans.entity_id, 1)
                    _ans.score *= 1.0 / (1.0 + idf_weight * math.log1p(_freq))
                answers_obj.sort(key=lambda a: a.score, reverse=True)

        # r2 path-consistency boost
        if _is_3hop and _trb and r2_map:
            _detected_r3 = next(iter(_trb))
            _eff_r2_boost = (
                r2_boost_map.get(_detected_r3, r2_boost) if r2_boost_map else r2_boost
            )
            _expected_r2 = r2_map.get(_detected_r3) if _eff_r2_boost > 0.0 else None
            if _expected_r2:
                _changed = False
                for _ans in answers_obj:
                    _bp = _ans.best_path
                    if _bp is not None and len(_bp.nodes) >= 2:
                        if _bp.nodes[1] == _expected_r2:
                            _ans.score *= (1.0 + _eff_r2_boost)
                            _changed = True
                if _changed:
                    answers_obj.sort(key=lambda a: a.score, reverse=True)

        # Phase 185/186: Cross-type penalty (worker path).
        _FORMAT_TAGS_W: frozenset = frozenset()  # unused post-186
        if _is_3hop and _trb:
            _det_r3_xtp = next(iter(_trb))
            _py_rels = {"written_by", "directed_by", "starred_actors", "release_year"}
            if _det_r3_xtp in _py_rels:
                _py_answers = frozenset(
                    e for r in _py_rels
                    for e in relation_answer_set.get(r, set())
                )
                _pure_g = frozenset(
                    relation_answer_set.get("has_genre", set())
                ) - _py_answers
                _pure_g_lower = frozenset(g.lower() for g in _pure_g)
                _lang_ents = frozenset(
                    relation_answer_set.get("in_language", set())
                ) - _py_answers
                _changed_xtp = False
                for _ans in answers_obj:
                    eid = _ans.entity_id
                    if eid in _pure_g or eid.lower() in _pure_g_lower:
                        _ans.score *= 0.10
                        _changed_xtp = True
                    elif _det_r3_xtp == "release_year" and eid in _lang_ents:
                        _ans.score *= 0.10
                        _changed_xtp = True
                if _changed_xtp:
                    answers_obj.sort(key=lambda a: a.score, reverse=True)

        pred = [a.entity_id for a in answers_obj]

        # Diagnostic row (Phase 184: enriched for --diagnose-jsonl)
        _diag: Optional[Dict] = None
        if (do_diag or do_diag_jsonl) and _is_3hop:
            _correct_set_w = set(correct_answers)
            _correct_in_beam = any(e in _correct_set_w for e in pred)
            _beam_rank = next((idx + 1 for idx, e in enumerate(pred) if e in _correct_set_w), 0)
            _diag = {
                "q_idx": q_idx,
                "seed": seed,
                "correct_answer": next(iter(correct_answers), ""),
                "in_beam_top100": _correct_in_beam,
                "beam_rank": _beam_rank,
                "detected_rel": next(iter(_trb), "") if _trb else "",
                "n_filtered": -1,
                "correct_in_filtered": False,
                "final_rank": 0,
                "filter_applied": False,
                "filter_fell_back": False,
            }
            if do_diag_jsonl:
                _ca_score = 0.0
                _t1_score = answers_obj[0].score if answers_obj else 0.0
                _t1_entity = answers_obj[0].entity_id if answers_obj else ""
                for _a in answers_obj:
                    if _a.entity_id in _correct_set_w:
                        _ca_score = _a.score
                        break
                _diag.update({
                    "question_text": question_text or "",
                    "all_correct_answers": "|".join(str(a) for a in correct_answers),
                    "n_candidates_before_filter": len(pred),
                    "predicted_top10": "|".join(pred[:10]),
                    "correct_answer_score": _ca_score,
                    "top1_answer": _t1_entity,
                    "top1_score": _t1_score,
                })

        if not pred:
            return (q_idx, None, _diag)

        # Answer-type filter
        if _is_3hop and _trb:
            detected_rel  = next(iter(_trb))
            valid_answers = relation_answer_set.get(detected_rel)
            if valid_answers:
                filtered = [p for p in pred if p in valid_answers]
                if _diag is not None:
                    _diag["n_filtered"] = len(filtered)
                    _diag["correct_in_filtered"] = any(e in correct_answers for e in filtered)
                    _diag["filter_applied"] = True
                    _diag["filter_fell_back"] = len(filtered) < min_filter_size
                pred = filtered[:top_k] if len(filtered) >= min_filter_size else pred[:top_k]
            else:
                pred = pred[:top_k]
        else:
            pred = pred[:top_k]

        if _diag is not None:
            _diag["final_rank"] = next(
                (i + 1 for i, e in enumerate(pred) if e in correct_answers), 0
            )

        return (q_idx, pred, _diag)

    except Exception as e:
        import traceback
        print(f"\n[worker] Error q_idx={task[0]}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return (task[0], None, None)


# ---------------------------------------------------------------------------
# Per-hop evaluation  (thin harness over CerebrumGraph.query)
# ---------------------------------------------------------------------------

def evaluate_hop(
    hop:              int,
    graph:            CerebrumGraph,
    qa_pairs:         List[Tuple],
    top_k:            int            = 10,
    beam_width:       int            = 10,
    embedding_engine=None,
    relation_prior=None,
    relation_answer_set: Optional[Dict] = None,
    branch_bonus_weight: float       = 0.0,
    r2_map:           Optional[Dict[str, str]] = None,
    vote_weight_3hop: float          = 0.85,
    trb_factor_3hop:  float          = 5.0,
    answer_freq_map:  Optional[Dict] = None,
    idf_weight:       float          = 0.0,
    hop2_beam_width:  Optional[int]  = None,
    r2_boost:         float          = 0.0,
    diagnose_path:      Optional[str]  = None,
    diagnose_jsonl_path: Optional[str] = None,
    min_filter_size:  int            = 1,
    expansion_k:      Optional[int]  = None,
    eval_min_hop_3hop: Optional[int] = None,
    r2_boost_map:     Optional[Dict[str, float]] = None,
    structural_trb:   bool                        = False,
    anchor_bonus:     Optional[float]             = None,
    fan_out:          Optional[Dict]              = None,
    pss_weight:       float                       = 0.0,
    fhrb_factor:      float                       = 0.0,
    pool=None,
) -> Dict:
    """
    Evaluate one hop level using the unified CerebrumGraph.query() interface.

    Parameters
    ----------
    hop              : 1, 2, or 3
    graph            : built CerebrumGraph instance
    qa_pairs         : list of (seed, answers) or (seed, answers, question_text)
    top_k            : number of answers to return per query
    embedding_engine : optional engine for encoding question text as query_embedding.
                       Pass when --embeddings sentence is active.
    relation_prior   : optional RelationPathPrior for re-ranking (2-hop, 3-hop).
    """
    h1 = h10 = 0
    mrr_sum  = 0.0
    skipped  = found = 0

    has_question = qa_pairs and len(qa_pairs[0]) == 3

    # For 2-hop: exclude depth-1 paths (direct neighbors are never the right answer).
    # For 1-hop: allow min_hop=1. For 3-hop: default min_hop=1 but allow override.
    if hop == 2:
        eval_min_hop = 2
    elif hop == 3 and eval_min_hop_3hop is not None:
        eval_min_hop = eval_min_hop_3hop
    else:
        eval_min_hop = 1

    # Phase 146: discover KB relation vocabulary once for TRB detection
    _kb_relations: List[str] = []
    try:
        _kb_relations = list({
            e.relation_type
            for eid in list(graph.adapter._G.nodes())[:500]
            for e in graph.adapter.get_neighbors(eid)
        })
    except Exception:
        pass

    # Phase 152: Answer-type constraint index — passed in from main() where it
    # is built directly from KB triples (objects only — not reversed edges).
    _relation_answer_set: Dict[str, set] = relation_answer_set or {}

    # Phase 185/186: Cross-type penalty.
    # _pure_genre: the 23 has_genre labels (Action, Drama, Comedy…) that never
    # appear as valid written_by/directed_by/starred_actors/release_year answers.
    # _language_entities: in_language answers — safe to penalize for release_year
    # (a language is never a year) but NOT for person relations where multi-hop
    # paths can legitimately end at a language entity.
    _person_year_relations: frozenset = frozenset(
        {"written_by", "directed_by", "starred_actors", "release_year"}
    )
    _person_year_answers: frozenset = frozenset(
        e for r in _person_year_relations
        for e in _relation_answer_set.get(r, set())
    )
    _pure_genre: frozenset = frozenset(
        _relation_answer_set.get("has_genre", set())
    ) - _person_year_answers
    # Case-insensitive genre set — the beam can return lowercase variants
    # ("animation" vs "Animation"). Safe because the 23 genre labels cannot
    # be person names (unlike the 4892-entry has_tags set).
    _pure_genre_lower: frozenset = frozenset(g.lower() for g in _pure_genre)
    _language_entities: frozenset = frozenset(
        _relation_answer_set.get("in_language", set())
    ) - _person_year_answers
    _FORMAT_TAG_BLOCKLIST: frozenset = frozenset()  # placeholder, unused post-186

    _diag_rows: List[Dict] = []

    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Phase 182: Parallel path                                             #
    # ------------------------------------------------------------------ #
    if pool is not None:
        _do_diag       = bool(diagnose_path)
        _do_diag_jsonl = bool(diagnose_jsonl_path)
        _eval_min = eval_min_hop
        _r2bmap   = ({k: v for k, v in r2_boost_map.items()} if r2_boost_map else None)
        tasks = [
            (
                i,
                (qa[0] if not has_question else qa[0]),
                (qa[1] if not has_question else qa[1]),
                (qa[2] if has_question else None),
                hop, top_k, beam_width, _eval_min,
                vote_weight_3hop, branch_bonus_weight, trb_factor_3hop, idf_weight,
                hop2_beam_width, r2_boost, _r2bmap, min_filter_size,
                expansion_k, structural_trb, anchor_bonus, pss_weight, fhrb_factor,
                _do_diag, _do_diag_jsonl,
            )
            for i, qa in enumerate(qa_pairs)
        ]
        completed = 0
        n_total   = len(qa_pairs)
        for q_idx, pred, diag in pool.imap_unordered(
            _worker_process_question, tasks, chunksize=64
        ):
            completed += 1
            if completed % 500 == 0 or completed == n_total:
                print(
                    f"    {completed:,}/{n_total:,} questions "
                    f"({time.time()-t0:.1f}s elapsed)",
                    end="\r",
                )
            if pred is None:
                skipped += 1
            else:
                correct = qa_pairs[q_idx][1]
                found    += 1
                h1       += hits_at_k(pred, correct, k=1)
                h10      += hits_at_k(pred, correct, k=10)
                mrr_sum  += reciprocal_rank(pred, correct)
            if diag is not None:
                _diag_rows.append(diag)

        elapsed = time.time() - t0
        print()
        n = len(qa_pairs)
        if diagnose_path and _diag_rows:
            _diag_rows.sort(key=lambda r: r["q_idx"])
            _fields = ["q_idx", "seed", "correct_answer", "in_beam_top100", "beam_rank",
                       "detected_rel", "n_filtered", "correct_in_filtered", "final_rank"]
            with open(diagnose_path, "w", newline="", encoding="utf-8") as _csvf:
                _w = csv.DictWriter(_csvf, fieldnames=_fields, extrasaction="ignore")
                _w.writeheader()
                _w.writerows(_diag_rows)
            print(f"  Diagnostic CSV written: {diagnose_path} ({len(_diag_rows):,} rows)")
        if diagnose_jsonl_path and _diag_rows:
            import json as _json
            _diag_rows.sort(key=lambda r: r["q_idx"])
            with open(diagnose_jsonl_path, "w", encoding="utf-8") as _jf:
                for _row in _diag_rows:
                    _jf.write(_json.dumps(_row, default=str) + "\n")
            print(f"  Diagnostic JSONL written: {diagnose_jsonl_path} ({len(_diag_rows):,} rows)")
        return {
            "hop": hop, "n_total": n, "n_answered": found, "n_skipped": skipped,
            "hits_1": h1 / n, "hits_10": h10 / n, "mrr": mrr_sum / n,
            "elapsed_s": elapsed,
        }
    # ------------------------------------------------------------------ #
    # Serial path (unchanged)                                              #
    # ------------------------------------------------------------------ #
    for i, qa in enumerate(qa_pairs):
        if has_question:
            seed, correct_answers, question_text = qa
        else:
            seed, correct_answers = qa
            question_text         = None

        if (i + 1) % 500 == 0 or (i + 1) == len(qa_pairs):
            print(
                f"    {i+1:,}/{len(qa_pairs):,} questions "
                f"({time.time()-t0:.1f}s elapsed)",
                end="\r",
            )

        # Encode question text as query_embedding when engine is available.
        # Use encode_query() when available (BGE instruction prefix for asymmetric
        # retrieval), falling back to encode_entities() for symmetric models.
        query_emb = None
        if question_text and embedding_engine is not None:
            try:
                encode_query_fn = getattr(embedding_engine, "encode_query", None)
                if encode_query_fn is not None:
                    query_emb = encode_query_fn([question_text])[0]
                else:
                    q_vecs    = embedding_engine.encode_entities({"__q__": question_text})
                    query_emb = q_vecs.get("__q__")
            except Exception:
                pass

        # Phase 146/147: Terminal Relation Boost — detect target relation from
        # question prefix (first 5 words).  Applied at all hops: prefix-only
        # scanning avoids false hits from intermediate-hop keywords in multi-hop
        # questions.  Penultimate cascade fires automatically inside traversal.
        # Phase 172: --structural-trb skips keyword detection entirely and lets
        # CerebrumGraph.query() infer TRB from graph topology (SRI).
        _trb: Dict[str, float] = {}
        if not structural_trb and question_text and _kb_relations:
            detected = detect_target_relation(question_text, _kb_relations)
            if detected:
                # Phase 151: TRB factor — 5.0 is sufficient with vote_weight=0.0;
                # higher values (25.0) were counter-productive per ablation.
                boost_factor = trb_factor_3hop if hop == 3 else 3.0
                _trb = {detected: boost_factor}

        # Phase 152: Answer-type constraint + vote_weight tuning for 3-hop.
        # With correct TRB detection, vote_weight=0.70 + answer-type filter +
        # wider top_k retrieval (100) gives the best H@1 for 3-hop queries.
        # The type filter removes wrong-type candidates (actors appearing as
        # answers to genre questions, etc.) before top-k truncation, allowing
        # high vote_weight to amplify convergence on the correct answer type.
        # degree_penalty NOT used — MetaQA correct answers are high-degree hubs.
        _is_3hop = (hop == 3)
        _raw_top_k = 100 if _is_3hop else top_k

        # Phase 156: Penultimate Relation Boost — boost expected r2 at hop N-1
        _prb: Dict[str, float] = {}
        if _is_3hop and _trb and r2_map:
            _detected_r3 = next(iter(_trb))
            _r2 = r2_map.get(_detected_r3)
            if _r2:
                _prb = {_r2: _trb[_detected_r3] ** 0.5}

        # Phase 180: First-Hop Relation Boost — detect r1 from question and
        # steer hop-1 beam along the intended first relation.
        # r1 is typically a DIFFERENT relation from r3 (terminal). We detect
        # both and use the non-r3 one as r1. Edges matching r1 are boosted;
        # all others receive a 0.1 penalty (same mechanism as TRB at last hop).
        _irb: Optional[Dict[str, float]] = None
        if _is_3hop and fhrb_factor > 0.0 and question_text and _kb_relations:
            _detected_r3_for_r1 = next(iter(_trb)) if _trb else None
            _r1_candidate = detect_target_relation(
                question_text, _kb_relations,
                exclude_relation=_detected_r3_for_r1,
            )
            if _r1_candidate:
                _irb = {_r1_candidate: fhrb_factor}

        # Phase 158: per-hop beam widths — {1: hop2_beam_width} widens the H1SE
        # sub-traversal's hop-1 (= original hop-2), improving coverage by allowing
        # more intermediate entities to survive into the final hop.
        _beam_widths: Optional[Dict[int, int]] = None
        if _is_3hop and hop2_beam_width is not None and hop2_beam_width > beam_width:
            _beam_widths = {1: hop2_beam_width}

        answers_obj = graph.query(
            seeds                         = [seed],
            top_k                         = _raw_top_k,
            min_hop                       = eval_min_hop,
            max_hop                       = hop,
            hop_expand                    = (hop >= 2),
            query_embedding               = query_emb,
            relation_prior                = relation_prior,
            terminal_relation_boost       = _trb,
            penultimate_relation_boost    = _prb,
            vote_weight                   = vote_weight_3hop if _is_3hop else 0.45,
            branch_bonus_weight           = branch_bonus_weight if _is_3hop else 0.0,
            beam_widths                   = _beam_widths,
            expansion_k                   = expansion_k if _is_3hop else None,
            auto_infer_terminal_relation  = structural_trb and _is_3hop,
            anchor_bonus                  = anchor_bonus if _is_3hop else None,
            fan_out                       = fan_out if _is_3hop else None,
            weight_specificity            = pss_weight if _is_3hop else 0.0,
            initial_relation_boost        = _irb,       # Phase 180: FHRB
        )
        # Phase 157: IDF hub-entity penalty — applied before answer-type filter.
        # Penalizes entities that appear very frequently as objects of the target
        # relation (e.g., Tom Hanks in 100 films). Only applied for high-cardinality
        # relations (>100 unique answer entities) where hub inflation is a problem.
        # Categorical relations (genres ~20, languages ~30) are skipped.
        if _is_3hop and _trb and idf_weight > 0.0 and answer_freq_map:
            _detected_r3_idf = next(iter(_trb))
            _freq_ctr = answer_freq_map.get(_detected_r3_idf)
            _n_unique = len(_relation_answer_set.get(_detected_r3_idf, set()))
            if _freq_ctr and _n_unique > 100:
                for _ans in answers_obj:
                    _freq = _freq_ctr.get(_ans.entity_id, 1)
                    _ans.score *= 1.0 / (1.0 + idf_weight * math.log1p(_freq))
                answers_obj.sort(key=lambda a: a.score, reverse=True)

        # Phase 158/160: r2 path-consistency boost — answers whose best path reached
        # hop-2 via the expected r2 (from the r3→r2 training map) are boosted.
        # Phase 172: r2_boost_map allows per-relation tuning (e.g., higher boost for
        # starred_actors where hub-entity dominance is worst).
        if _is_3hop and _trb and r2_map:
            _detected_r3_pc = next(iter(_trb))
            _eff_r2_boost = (r2_boost_map.get(_detected_r3_pc, r2_boost)
                             if r2_boost_map else r2_boost)
            _expected_r2_pc = r2_map.get(_detected_r3_pc) if _eff_r2_boost > 0.0 else None
            if _expected_r2_pc:
                _changed = False
                for _ans in answers_obj:
                    _bp = _ans.best_path
                    if _bp is not None and len(_bp.nodes) >= 2:
                        _path_r2 = _bp.nodes[1]  # odd index 1 = first edge relation
                        if _path_r2 == _expected_r2_pc:
                            _ans.score *= (1.0 + _eff_r2_boost)
                            _changed = True
                if _changed:
                    answers_obj.sort(key=lambda a: a.score, reverse=True)

        # Phase 185/186: Cross-type penalty.
        # Pure genre labels + explicit format-tag blocklist suppressed for all
        # person/year relations. Language entities suppressed for release_year only.
        if _is_3hop and _trb:
            _det_r3_xtp = next(iter(_trb))
            if _det_r3_xtp in _person_year_relations:
                _changed_xtp = False
                for _ans in answers_obj:
                    eid = _ans.entity_id
                    if eid in _pure_genre or eid.lower() in _pure_genre_lower:
                        _ans.score *= 0.10
                        _changed_xtp = True
                    elif _det_r3_xtp == "release_year" and eid in _language_entities:
                        _ans.score *= 0.10
                        _changed_xtp = True
                if _changed_xtp:
                    answers_obj.sort(key=lambda a: a.score, reverse=True)

        pred = [a.entity_id for a in answers_obj]

        # Phase 159/184: diagnostic — capture pre-filter beam state
        _diag: Optional[Dict] = None
        if (diagnose_path or diagnose_jsonl_path) and _is_3hop:
            _correct_set_s = set(correct_answers)
            _correct_in_beam = any(e in _correct_set_s for e in pred)
            _beam_rank = next((idx + 1 for idx, e in enumerate(pred) if e in _correct_set_s), 0)
            _diag = {
                "q_idx": i,
                "seed": seed,
                "correct_answer": next(iter(correct_answers), ""),
                "in_beam_top100": _correct_in_beam,
                "beam_rank": _beam_rank,
                "detected_rel": next(iter(_trb), "") if _trb else "",
                "n_filtered": -1,
                "correct_in_filtered": False,
                "final_rank": 0,
                "filter_applied": False,
                "filter_fell_back": False,
            }
            if diagnose_jsonl_path:
                _ca_score = 0.0
                _t1_score = answers_obj[0].score if answers_obj else 0.0
                _t1_entity = answers_obj[0].entity_id if answers_obj else ""
                for _a in answers_obj:
                    if _a.entity_id in _correct_set_s:
                        _ca_score = _a.score
                        break
                _diag.update({
                    "question_text": question_text or "",
                    "all_correct_answers": "|".join(str(a) for a in correct_answers),
                    "n_candidates_before_filter": len(pred),
                    "predicted_top10": "|".join(pred[:10]),
                    "correct_answer_score": _ca_score,
                    "top1_answer": _t1_entity,
                    "top1_score": _t1_score,
                })

        if not pred:
            skipped += 1
            if _diag is not None:
                _diag_rows.append(_diag)
            continue

        # Phase 152: apply answer-type filter for 3-hop when TRB detected a relation.
        # Phase 159: soft fallback — only apply hard filter when len(filtered) >= min_filter_size.
        # Prevents wrong TRB detection from locking out correct answers via thin filter results.
        if _is_3hop and _trb:
            detected_rel = next(iter(_trb))
            valid_answers = _relation_answer_set.get(detected_rel)
            if valid_answers:
                filtered = [p for p in pred if p in valid_answers]
                if _diag is not None:
                    _diag["n_filtered"] = len(filtered)
                    _diag["correct_in_filtered"] = any(e in correct_answers for e in filtered)
                    _diag["filter_applied"] = True
                    _diag["filter_fell_back"] = len(filtered) < min_filter_size
                if len(filtered) >= min_filter_size:
                    pred = filtered[:top_k]
                else:
                    pred = pred[:top_k]
            else:
                pred = pred[:top_k]
        else:
            pred = pred[:top_k]

        if _diag is not None:
            _diag["final_rank"] = next((i + 1 for i, e in enumerate(pred) if e in correct_answers), 0)
            _diag_rows.append(_diag)

        found   += 1
        h1      += hits_at_k(pred, correct_answers, k=1)
        h10     += hits_at_k(pred, correct_answers, k=10)
        mrr_sum += reciprocal_rank(pred, correct_answers)

    elapsed = time.time() - t0
    print()
    n = len(qa_pairs)

    if diagnose_path and _diag_rows:
        _fields = ["q_idx", "seed", "correct_answer", "in_beam_top100", "beam_rank",
                   "detected_rel", "n_filtered", "correct_in_filtered", "final_rank"]
        with open(diagnose_path, "w", newline="", encoding="utf-8") as _csvf:
            _w = csv.DictWriter(_csvf, fieldnames=_fields, extrasaction="ignore")
            _w.writeheader()
            _w.writerows(_diag_rows)
        print(f"  Diagnostic CSV written: {diagnose_path} ({len(_diag_rows):,} rows)")
    if diagnose_jsonl_path and _diag_rows:
        import json as _json
        with open(diagnose_jsonl_path, "w", encoding="utf-8") as _jf:
            for _row in _diag_rows:
                _jf.write(_json.dumps(_row, default=str) + "\n")
        print(f"  Diagnostic JSONL written: {diagnose_jsonl_path} ({len(_diag_rows):,} rows)")

    return {
        "hop":        hop,
        "n_total":    n,
        "n_answered": found,
        "n_skipped":  skipped,
        "hits_1":     h1  / n,
        "hits_10":    h10 / n,
        "mrr":        mrr_sum / n,
        "elapsed_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Phase 181: GPU startup cleanup
# ---------------------------------------------------------------------------

def _cleanup_stale_gpu_processes() -> None:
    """Kill idle metaqa_eval processes holding VRAM from prior crashed/OOM runs.

    Only kills processes with near-zero CPU activity — actively running
    benchmarks are left untouched.
    """
    import os
    import signal

    current_pid = os.getpid()

    try:
        import psutil
        current_start = psutil.Process(current_pid).create_time()
        candidates = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue
                name = (proc.info["name"] or "").lower()
                if "python" not in name:
                    continue
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "metaqa_eval" not in cmdline:
                    continue
                # Only target processes that started >60s before us — this
                # prevents killing our own pool workers or a concurrent run
                # that started at roughly the same time.
                if proc.create_time() > current_start - 60:
                    continue
                # Sample CPU over 1 second — skip actively working processes
                cpu = proc.cpu_percent(interval=1.0)
                if cpu > 2.0:
                    continue  # process is alive and working — leave it alone
                candidates.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if candidates:
            print(f"[startup] Cleaning up {len(candidates)} idle metaqa_eval process(es): {candidates}")
            for pid in candidates:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
    except ImportError:
        pass

    # Flush any cached CUDA allocations in the current process
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MetaQA benchmark for CEREBRUM")
    parser.add_argument("--hop",        type=int,   default=None,
                        help="Evaluate only this hop level (1, 2, or 3). Default: all.")
    parser.add_argument("--sample",     type=int,   default=None,
                        help="Evaluate on a random sample of N questions per hop.")
    parser.add_argument("--beam-width", type=int,   default=10)
    parser.add_argument("--top-k",      type=int,   default=10)
    parser.add_argument("--use-cache",  action="store_true", default=True)
    parser.add_argument("--no-cache",   action="store_true",
                        help="Recompute DSCF and embeddings even if cached.")
    parser.add_argument("--embeddings", choices=["random", "sentence"], default="random")
    parser.add_argument("--community-engine",
                        choices=["dscf", "tsc", "leiden", "lpa"],
                        default="dscf",
                        help="Community detection algorithm (default: dscf). "
                             "Use 'tsc' to measure TSC vs DSCF downstream quality.")
    parser.add_argument("--graphsage", action="store_true", default=False,
                        help="Apply GraphSAGE neighborhood smoothing to entity embeddings "
                             "after encoding. Enriches embeddings with 2-hop context. "
                             "Most useful with --embeddings sentence.")
    parser.add_argument("--kge", action="store_true", default=False,
                        help="Train TransE/RotatE on graph triples and blend with base "
                             "embeddings (Phase 135). Topology-aware enrichment.")
    parser.add_argument("--kge-blend", type=float, default=0.5,
                        help="KGE blend weight: final = (1-blend)*base + blend*kge (default 0.5).")
    parser.add_argument("--kge-epochs", type=int, default=100,
                        help="KGE training epochs (default 100; cached after first run).")
    parser.add_argument("--min-community-size", type=int, default=0,
                        help="Merge communities smaller than this. "
                             "Recommended: 20 for MetaQA.")
    parser.add_argument("--use-prior",  action="store_true", default=False,
                        help="Build and use RelationPathPrior from training data "
                             "for 2-hop and 3-hop re-ranking (cached after first run).")
    parser.add_argument("--branch-bonus", type=float, default=0.25,
                        help="Phase 154 DBC: branch-diversity bonus weight for 3-hop "
                             "reranking (default 0.25; set 0.0 to disable).")
    parser.add_argument("--vote-weight", type=float, default=0.85,
                        help="Vote convergence weight for 3-hop scoring "
                             "(default 0.85, tuned Phase 157).")
    parser.add_argument("--trb-factor",  type=float, default=5.0,
                        help="Terminal Relation Boost multiplier for 3-hop "
                             "(default 5.0).")
    parser.add_argument("--idf-weight",  type=float, default=0.0,
                        help="IDF penalty weight for hub-entity de-ranking. "
                             "Applied only to high-cardinality relations (>100 unique "
                             "answer entities). score *= 1/(1+w*log1p(freq)). "
                             "Default 0.0 (disabled).")
    parser.add_argument("--hop2-beam-width", type=int, default=None,
                        help="Per-hop beam width override at hop-2 for 3-hop traversal. "
                             "Default: same as --beam-width.")
    parser.add_argument("--r2-boost", type=float, default=3.0,
                        help="Phase 158/186: path-consistency r2 boost. Answers whose best "
                             "path has r2 == expected r2 (from training r3->r2 map) get "
                             "score *= (1 + r2-boost). Default 3.0 (tuned Phase 186).")
    parser.add_argument("--sa-r2-boost", type=float, default=None,
                        help="Phase 172: r2-boost override for starred_actors questions. "
                             "Default None uses --r2-boost for all relations. "
                             "starred_actors has worst H@1 (27%%) due to hub-entity "
                             "dominance; a higher boost (e.g. 0.70) may close the gap. "
                             "Other relations keep --r2-boost value.")
    parser.add_argument("--eval-min-hop", type=int, default=None,
                        help="Phase 172: minimum hop depth for 3-hop answer extraction. "
                             "Default None=1 (current: allows 1-hop and 2-hop paths). "
                             "Set to 2 to exclude 1-hop paths (seed's own direct relations "
                             "like directed_by/has_genre). Set to 3 for strict 3-hop only. "
                             "May reduce ranking miss from short-path contamination.")
    parser.add_argument("--expansion-k", type=int, default=None,
                        help="Phase 159: number of hop-1 entities given deep sub-traversals "
                             "in HopExpandedTraversal (default: graph default of 20). "
                             "Increasing to 50-100 covers more actors/directors at hop-1 "
                             "for 3-hop queries, reducing coverage miss at the cost of "
                             "higher compute per query.")
    parser.add_argument("--diagnose", type=str, default=None, metavar="PATH",
                        help="Phase 159: write per-question diagnostic CSV to PATH. "
                             "Columns: in_beam_top100, beam_rank, n_filtered, "
                             "correct_in_filtered, final_rank. 3-hop only.")
    parser.add_argument("--diagnose-jsonl", type=str, default=None, metavar="PATH",
                        help="Phase 184: write per-question rich diagnostic JSONL to PATH. "
                             "Superset of --diagnose: adds question_text, predicted_top10, "
                             "n_candidates_before_filter, correct_answer_score, top1_score, "
                             "filter_applied, filter_fell_back. 3-hop only. "
                             "Used by benchmarks/phase184_diagnose.py.")
    parser.add_argument("--min-filter-size", type=int, default=1,
                        help="Phase 159: minimum number of type-filtered answers required "
                             "to apply the hard answer-type filter. If fewer than this many "
                             "answers pass the filter, fall back to unfiltered top-k. "
                             "Default 1 (current behaviour). Try 3 or 5 to reduce filter "
                             "false negatives from wrong TRB detection.")
    parser.add_argument("--structural-trb", action="store_true",
                        help="Phase 172: use StructuralRelationInferrer (SRI) for terminal "
                             "relation boost instead of keyword detection. Fully agnostic — "
                             "no question text, no domain keywords. Skips the answer-type "
                             "hard filter (which requires KB relation names). Measures the "
                             "gap between graph-structural inference and domain-assisted TRB.")
    parser.add_argument("--anchor-bonus", type=float, default=None,
                        help="Phase 172: Terminal-Anchor bonus multiplier applied at the "
                             "penultimate hop (hop-2 for 3-hop) to entities that are sources "
                             "of the terminal relation. Default None (disabled).")
    parser.add_argument("--pss-weight", type=float, default=0.0,
                        help="Phase 179: Path Specificity Score weight [0,1]. "
                             "Penalises hub-like traversals by inverse relation fan-out. "
                             "Default 0.0 (disabled). Try 0.15–0.30 for 3-hop H@1.")
    parser.add_argument("--fhrb-factor", type=float, default=0.0,
                        help="Phase 180: First-Hop Relation Boost factor. When >0, "
                             "detects r1 from question keywords and boosts hop-1 edges "
                             "matching r1 by this factor; non-matching edges get 0.1 penalty. "
                             "Default 0.0 (disabled). Try 3.0–8.0 alongside --use-prior.")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--mlflow",  action="store_true", default=False,
                        help="Log metrics and params to MLflow (tracking URI: ./mlruns).")
    parser.add_argument("--mlflow-uri", type=str, default=None,
                        help="MLflow tracking URI (default: local ./mlruns).")
    parser.add_argument("--wandb",   action="store_true", default=False,
                        help="Log metrics and params to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="cerebrum-metaqa",
                        help="W&B project name (default: cerebrum-metaqa).")
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Worker processes for question-level parallelism. "
             "Default: os.cpu_count(). Set to 1 to disable multiprocessing.",
    )
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    hops = [args.hop] if args.hop else [1, 2, 3]

    # ------------------------------------------------------------------
    # Experiment tracking init (MLflow / W&B)
    # ------------------------------------------------------------------
    global _mlflow, _wandb
    _run_params = {
        "hop": args.hop, "beam_width": args.beam_width, "top_k": args.top_k,
        "embeddings": args.embeddings, "use_prior": args.use_prior,
        "vote_weight": args.vote_weight, "trb_factor": args.trb_factor,
        "r2_boost": args.r2_boost, "fhrb_factor": args.fhrb_factor,
        "pss_weight": args.pss_weight, "idf_weight": args.idf_weight,
        "workers": args.workers, "sample": args.sample, "seed": args.seed,
    }

    if args.mlflow:
        try:
            import mlflow as _mlf
            _mlflow = _mlf
            uri = args.mlflow_uri or "mlruns"
            _mlflow.set_tracking_uri(uri)
            _mlflow.set_experiment("cerebrum-metaqa")
            _mlflow.start_run(run_name=f"metaqa_hop{args.hop or 'all'}_bw{args.beam_width}")
            _mlflow.log_params(_run_params)
            print(f"  MLflow tracking: {uri}")
        except ImportError:
            print("  [warning] mlflow not installed — run: pip install mlflow")
            _mlflow = None

    if args.wandb:
        try:
            import wandb as _wb
            _wandb = _wb
            _wandb.init(
                project=args.wandb_project,
                name=f"metaqa_hop{args.hop or 'all'}_bw{args.beam_width}",
                config=_run_params,
            )
            print(f"  W&B tracking: project={args.wandb_project}")
        except ImportError:
            print("  [warning] wandb not installed — run: pip install wandb")
            _wandb = None

    # ------------------------------------------------------------------
    # Phase 181: GPU startup cleanup — kill stale Python processes that
    # hold VRAM from prior crashed/OOM runs before allocating our own.
    # ------------------------------------------------------------------
    _cleanup_stale_gpu_processes()

    # ------------------------------------------------------------------
    # Validate data files exist
    # ------------------------------------------------------------------
    print("\n=== CEREBRUM MetaQA Benchmark ===\n")

    if not KB_FILE.exists():
        print(f"ERROR: kb.txt not found at {KB_FILE}")
        print("Place the MetaQA kb.txt file in benchmarks/data/metaqa/")
        sys.exit(1)

    for hop in hops:
        if not QA_FILES[hop].exists():
            print(f"ERROR: QA file for {hop}-hop not found: {QA_FILES[hop]}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build CerebrumGraph  (THALAMUS pipeline — done once for all hops)
    # ------------------------------------------------------------------
    print("Loading knowledge graph...")
    t0 = time.time()

    graph = CerebrumGraph.from_kb(
        KB_FILE,
        sep       = "|",
        directed  = False,           # MetaQA standard: undirected
        embeddings= args.embeddings,
        beam_width= args.beam_width,
        max_hop   = 3,               # always 3; min_hop filters at extract time
        max_neighbors = 100,
    )
    print(f"  {graph.node_count:,} entities, {graph.edge_count:,} edges "
          f"({time.time()-t0:.1f}s)")

    print("Computing/loading community structure...")
    graph.build(
        cache_dir           = CACHE_DIR,
        min_community_size  = args.min_community_size,
        force_rebuild       = not args.use_cache,
        seed                = args.seed,
        use_graphsage       = args.graphsage,
        use_kge             = args.kge,
        kge_blend           = args.kge_blend,
        kge_epochs          = args.kge_epochs,
        community_engine    = args.community_engine,
    )
    print(f"  {graph.community_count} communities")

    # ------------------------------------------------------------------
    # Optional: sentence engine for question-text query embeddings
    # ------------------------------------------------------------------
    query_engine = None
    if args.embeddings == "sentence":
        query_engine = graph._embedding_engine

    # ------------------------------------------------------------------
    # Optional: RelationPathPrior for 2-hop and 3-hop re-ranking
    # ------------------------------------------------------------------
    priors: Dict[int, Any] = {}
    if args.use_prior:
        print("\nBuilding RelationPathPrior (2-hop and 3-hop)...")
        for hop in hops:
            if hop == 1:
                continue   # 1-hop has only 9 patterns; prior does not help
            prior = build_or_load_prior(
                hop           = hop,
                graph         = graph,
                beam_width    = args.beam_width,
                use_cache     = args.use_cache,
                force_rebuild = not args.use_cache,
            )
            if prior is not None:
                priors[hop] = prior

    # ------------------------------------------------------------------
    # Phase 152: Build answer-type constraint index + KB forward index.
    # Both are built from KB triples once and shared across all hop evals.
    # ------------------------------------------------------------------
    from collections import defaultdict as _dd, Counter as _Counter
    _relation_answer_set: Dict[str, set] = _dd(set)
    _answer_freq: Dict[str, Any] = _dd(_Counter)   # Phase 157: IDF freq per relation
    _kb_index: Dict[str, Dict[str, List[str]]] = {}   # entity -> {rel: [targets]}
    try:
        with open(KB_FILE, encoding="utf-8", errors="replace") as _kbf:
            for _line in _kbf:
                _p = _line.strip().split("|")
                if len(_p) >= 3:
                    _s, _r, _o = _p[0].strip(), _p[1].strip(), _p[2].strip()
                    _relation_answer_set[_r].add(_o)
                    _answer_freq[_r][_o] += 1      # Phase 157: count appearances
                    _kb_index.setdefault(_s, {}).setdefault(_r, []).append(_o)
    except Exception:
        _relation_answer_set = {}
        _answer_freq = {}

    # ------------------------------------------------------------------
    # Phase 156: Build r3 → most-common-r2 map from 3-hop training data.
    # Walks all correct (seed, answer) paths in the KB to count r2 frequencies
    # per terminal relation r3.
    # ------------------------------------------------------------------
    _r2_for_r3: Dict[str, str] = {}
    if 3 in hops and _kb_index and TRAIN_FILES[3].exists():
        _r2_counts: Dict[str, Any] = _dd(_Counter)
        try:
            with open(TRAIN_FILES[3], encoding="utf-8", errors="replace") as _tf:
                for _line in _tf:
                    _parts = _line.strip().split("\t", 1)
                    if len(_parts) < 2:
                        continue
                    _qtext = _parts[0]
                    _m = re.search(r"\[(.+?)\]", _qtext)
                    if not _m:
                        continue
                    _seed3 = _m.group(1).strip()
                    _targets3 = {a.strip() for a in _parts[1].split("|") if a.strip()}
                    for _r1, _h1s in _kb_index.get(_seed3, {}).items():
                        for _h1 in _h1s:
                            for _r2, _h2s in _kb_index.get(_h1, {}).items():
                                for _h2 in _h2s:
                                    for _r3, _h3s in _kb_index.get(_h2, {}).items():
                                        if any(t in _targets3 for t in _h3s):
                                            _r2_counts[_r3][_r2] += 1
        except Exception:
            pass
        _r2_for_r3 = {
            r3: ctr.most_common(1)[0][0]
            for r3, ctr in _r2_counts.items() if ctr
        }
        if _r2_for_r3:
            print(f"  r3->r2 template map: {_r2_for_r3}")

    # ------------------------------------------------------------------
    # Phase 179: Build relation fan-out index for Path Specificity Score.
    # fan_out[entity][relation] = number of distinct targets in the KB.
    # Derived entirely from graph structure — no training labels used.
    # ------------------------------------------------------------------
    _fan_out: Dict[str, Dict[str, int]] = {}
    if args.pss_weight > 0.0 and _kb_index:
        for _ent, _rels in _kb_index.items():
            _fan_out[_ent] = {_r: len(_tgts) for _r, _tgts in _rels.items()}
        print(f"  PSS fan-out index: {len(_fan_out):,} entities "
              f"(weight={args.pss_weight})")

    # ------------------------------------------------------------------
    # Phase 182: Resolve worker count and build multiprocessing pool
    # ------------------------------------------------------------------
    n_workers = max(1, args.workers if args.workers is not None else os.cpu_count())
    _pool = None
    if n_workers > 1:
        print(f"\nSpawning {n_workers} worker processes...")
        _ctx  = multiprocessing.get_context("spawn")
        _pool = _ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(
                {
                    "kb_file":             str(KB_FILE),
                    "cache_dir":           str(CACHE_DIR),
                    "embeddings":          args.embeddings,
                    "beam_width":          args.beam_width,
                    "min_community_size":  args.min_community_size,
                    "seed":                args.seed,
                    "graphsage":           args.graphsage,
                    "kge":                 args.kge,
                    "kge_blend":           args.kge_blend,
                    "kge_epochs":          args.kge_epochs,
                    "community_engine":    args.community_engine,
                },
                {
                    "relation_answer_set": dict(_relation_answer_set),
                    "answer_freq":         dict(_answer_freq),
                    "priors":              priors,
                    "r2_for_r3":           _r2_for_r3,
                    "fan_out":             _fan_out or {},
                },
            ),
        )
        print(f"  Workers ready.")

    # ------------------------------------------------------------------
    # Evaluate each hop level
    # ------------------------------------------------------------------
    results = []

    try:
        for hop in hops:
            print(f"\n--- {hop}-hop evaluation ---")
            qa_pairs = load_qa(
                hop,
                sample           = args.sample,
                seed             = args.seed,
                include_question = True,
            )
            n_label = f"{len(qa_pairs):,}" + (" (sample)" if args.sample else "")
            print(f"  {n_label} test questions")
            prior_label = " + RelationPrior" if hop in priors else ""
            print(f"  Running traversal (beam_width={args.beam_width}, max_hop={hop}{prior_label})...")

            metrics = evaluate_hop(hop, graph, qa_pairs,
                                   top_k=args.top_k, beam_width=args.beam_width,
                                   embedding_engine=query_engine,
                                   relation_prior=priors.get(hop),
                                   relation_answer_set=_relation_answer_set,
                                   branch_bonus_weight=args.branch_bonus,
                                   r2_map=_r2_for_r3,
                                   vote_weight_3hop=args.vote_weight,
                                   trb_factor_3hop=args.trb_factor,
                                   answer_freq_map=_answer_freq,
                                   idf_weight=args.idf_weight,
                                   hop2_beam_width=args.hop2_beam_width,
                                   r2_boost=args.r2_boost,
                                   diagnose_path=args.diagnose if hop == 3 else None,
                                   diagnose_jsonl_path=(args.diagnose_jsonl if hop == 3 else None),
                                   min_filter_size=args.min_filter_size,
                                   expansion_k=args.expansion_k,
                                   eval_min_hop_3hop=args.eval_min_hop,
                                   r2_boost_map=({"starred_actors": args.sa_r2_boost}
                                                 if args.sa_r2_boost is not None else None),
                                   structural_trb=args.structural_trb,
                                   anchor_bonus=args.anchor_bonus,
                                   fan_out=_fan_out if args.pss_weight > 0.0 else None,
                                   pss_weight=args.pss_weight,
                                   fhrb_factor=args.fhrb_factor,
                                   pool=_pool)
            results.append(metrics)

            print(f"  Hits@1  : {metrics['hits_1']:.4f}  ({metrics['hits_1']*100:.1f}%)")
            print(f"  Hits@10 : {metrics['hits_10']:.4f}  ({metrics['hits_10']*100:.1f}%)")
            print(f"  MRR     : {metrics['mrr']:.4f}")
            print(f"  Answered: {metrics['n_answered']:,}/{metrics['n_total']:,}  "
                  f"(skipped: {metrics['n_skipped']:,})")
            print(f"  Time    : {metrics['elapsed_s']:.1f}s")

            # Experiment tracking
            _h = hop
            if _mlflow:
                _mlflow.log_metrics({
                    f"hop{_h}_hits1":    metrics["hits_1"],
                    f"hop{_h}_hits10":   metrics["hits_10"],
                    f"hop{_h}_mrr":      metrics["mrr"],
                    f"hop{_h}_elapsed":  metrics["elapsed_s"],
                    f"hop{_h}_answered": metrics["n_answered"],
                })
            if _wandb:
                _wandb.log({
                    f"hop{_h}/hits1":    metrics["hits_1"],
                    f"hop{_h}/hits10":   metrics["hits_10"],
                    f"hop{_h}/mrr":      metrics["mrr"],
                    f"hop{_h}/elapsed":  metrics["elapsed_s"],
                    f"hop{_h}/answered": metrics["n_answered"],
                })
    finally:
        if _pool is not None:
            _pool.terminate()
            _pool.join()
        if _mlflow:
            _mlflow.end_run()
        if _wandb:
            _wandb.finish()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Results Summary ===\n")
    print(f"  Model     : CEREBRUM CSA + DSCF")
    print(f"  Embeddings: {args.embeddings}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Prior     : {'yes (2-hop, 3-hop)' if args.use_prior else 'no'}")
    _h2bw_str = str(args.hop2_beam_width) if args.hop2_beam_width else f"{args.beam_width} (flat)"
    print(f"  VoteWt(3h): {args.vote_weight}  TRB-factor(3h): {args.trb_factor}  hop2-bw: {_h2bw_str}")
    if args.sample:
        print(f"  Sample    : {args.sample} per hop")
    print()
    print(f"  {'Hop':<6} {'N':>7} {'Hits@1':>8} {'Hits@10':>9} {'MRR':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")
    for m in results:
        print(f"  {m['hop']}-hop  {m['n_total']:>7,} "
              f"{m['hits_1']:>8.4f} {m['hits_10']:>9.4f} {m['mrr']:>8.4f}")

    # ------------------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results_file = CACHE_DIR / "metaqa_results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "hop", "n_total", "n_answered", "n_skipped",
            "hits_1", "hits_10", "mrr", "beam_width", "embeddings", "elapsed_s",
        ])
        writer.writeheader()
        for m in results:
            writer.writerow({**m, "beam_width": args.beam_width,
                             "embeddings": args.embeddings})

    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()

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
import pickle
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
    effective_prefix = 6 if (len(words) >= 2 and words[0] == "what" and words[1] in ("are", "is")) else prefix_words
    prefix = " ".join(words[:effective_prefix])
    suffix = " ".join(words[-suffix_words:])

    for scan in (prefix, suffix):
        for relation, keywords in _RELATION_KEYWORDS:
            if relation not in kb_relations:
                continue
            if any(kw in scan for kw in keywords):
                return relation
    return None


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
    # For 1-hop and 3-hop: allow min_hop=1 (shortcuts are valid).
    eval_min_hop = 2 if hop == 2 else 1

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

    t0 = time.time()
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

        # Encode question text as query_embedding when engine is available
        query_emb = None
        if question_text and embedding_engine is not None:
            try:
                q_vecs    = embedding_engine.encode_entities({"__q__": question_text})
                query_emb = q_vecs.get("__q__")
            except Exception:
                pass

        # Phase 146/147: Terminal Relation Boost — detect target relation from
        # question prefix (first 5 words).  Applied at all hops: prefix-only
        # scanning avoids false hits from intermediate-hop keywords in multi-hop
        # questions.  Penultimate cascade fires automatically inside traversal.
        _trb: Dict[str, float] = {}
        if question_text and _kb_relations:
            detected = detect_target_relation(question_text, _kb_relations)
            if detected:
                # Phase 151: TRB factor — 5.0 is sufficient with vote_weight=0.0;
                # higher values (25.0) were counter-productive per ablation.
                boost_factor = 5.0 if hop == 3 else 3.0
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
        answers_obj = graph.query(
            seeds                   = [seed],
            top_k                   = _raw_top_k,
            min_hop                 = eval_min_hop,
            max_hop                 = hop,
            hop_expand              = (hop >= 2),
            query_embedding         = query_emb,
            relation_prior          = relation_prior,
            terminal_relation_boost = _trb,
            vote_weight             = 0.70 if _is_3hop else 0.45,
        )
        pred = [a.entity_id for a in answers_obj]

        if not pred:
            skipped += 1
            continue

        # Phase 152: apply answer-type filter for 3-hop when TRB detected a relation
        if _is_3hop and _trb:
            detected_rel = next(iter(_trb))
            valid_answers = _relation_answer_set.get(detected_rel)
            if valid_answers:
                filtered = [p for p in pred if p in valid_answers]
                if filtered:
                    pred = filtered[:top_k]
                else:
                    pred = pred[:top_k]
            else:
                pred = pred[:top_k]
        else:
            pred = pred[:top_k]

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
    }


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
    parser.add_argument("--min-community-size", type=int, default=0,
                        help="Merge communities smaller than this. "
                             "Recommended: 20 for MetaQA.")
    parser.add_argument("--use-prior",  action="store_true", default=False,
                        help="Build and use RelationPathPrior from training data "
                             "for 2-hop and 3-hop re-ranking (cached after first run).")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    hops = [args.hop] if args.hop else [1, 2, 3]

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
    # Phase 152: Build answer-type constraint index from KB triples.
    # Only KB objects (not subjects) are valid answer candidates for each relation.
    # Built once and passed to all evaluate_hop() calls.
    # ------------------------------------------------------------------
    from collections import defaultdict as _dd
    _relation_answer_set: Dict[str, set] = _dd(set)
    try:
        with open(KB_FILE, encoding="utf-8", errors="replace") as _kbf:
            for _line in _kbf:
                _p = _line.strip().split("|")
                if len(_p) >= 3:
                    _relation_answer_set[_p[1].strip()].add(_p[2].strip())
    except Exception:
        _relation_answer_set = {}

    # ------------------------------------------------------------------
    # Evaluate each hop level
    # ------------------------------------------------------------------
    results = []

    for hop in hops:
        print(f"\n--- {hop}-hop evaluation ---")
        qa_pairs = load_qa(
            hop,
            sample           = args.sample,
            seed             = args.seed,
            include_question = True,  # Phase 146: always load for TRB detection
        )
        n_label = f"{len(qa_pairs):,}" + (" (sample)" if args.sample else "")
        print(f"  {n_label} test questions")
        prior_label = " + RelationPrior" if hop in priors else ""
        print(f"  Running traversal (beam_width={args.beam_width}, max_hop={hop}{prior_label})...")

        metrics = evaluate_hop(hop, graph, qa_pairs,
                               top_k=args.top_k, beam_width=args.beam_width,
                               embedding_engine=query_engine,
                               relation_prior=priors.get(hop),
                               relation_answer_set=_relation_answer_set)
        results.append(metrics)

        print(f"  Hits@1  : {metrics['hits_1']:.4f}  ({metrics['hits_1']*100:.1f}%)")
        print(f"  Hits@10 : {metrics['hits_10']:.4f}  ({metrics['hits_10']*100:.1f}%)")
        print(f"  MRR     : {metrics['mrr']:.4f}")
        print(f"  Answered: {metrics['n_answered']:,}/{metrics['n_total']:,}  "
              f"(skipped: {metrics['n_skipped']:,})")
        print(f"  Time    : {metrics['elapsed_s']:.1f}s")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== Results Summary ===\n")
    print(f"  Model     : CEREBRUM CSA + DSCF")
    print(f"  Embeddings: {args.embeddings}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Prior     : {'yes (2-hop, 3-hop)' if args.use_prior else 'no'}")
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

"""
ComplexWebQuestions (CWQ) Data Setup — CEREBRUM
================================================

Downloads and prepares the CWQ benchmark from HuggingFace.

Dataset: rmanluo/RoG-cwq  (same RoG schema as rmanluo/RoG-webqsp)
  - train split : 27,639 questions — build scaffold graph + RelationPathPrior
  - test  split :  3,531 questions — evaluation QA pairs

CWQ has four question types within test:
  composition  : compositional multi-hop questions (derived from WebQSP)
  conjunction  : conjunctive constraints ("X that is also Y")
  comparative  : ordinal/comparative constraints ("X that has the most Y")
  superlative  : superlative constraints ("X with the highest Y")

CWQ shares the same Freebase graph as WebQSP.  This script supplements the
existing WebQSP Freebase graph with any additional triples surfaced by CWQ
train questions.

RoG-CWQ schema (identical to RoG-WebQSP):
  id                   : str
  question             : str
  answer               : list[str]   — answer text strings
  q_entity             : list[str]   — seed entity IDs (Freebase MIDs)
  a_entity             : list[str]   — answer entity IDs (Freebase MIDs)
  graph                : list[list[str]]  — KG triples [[s, r, o], ...]
  compositionality_type: str         — composition | conjunction | comparative | superlative

This script:
  1. Loads the dataset via HuggingFace datasets library
  2. Aggregates all unique triples from the graph column -> cwq_scaffold.txt
     (includes WebQSP Freebase graph for maximum coverage)
  3. Converts QA pairs to CEREBRUM JSON format
  4. Prints coverage stats

Output
------
  benchmarks/data/cwq/cwq_scaffold.txt      — TSV triples: head<TAB>relation<TAB>tail
  benchmarks/data/cwq/CWQ.test.json         — test QA pairs (CEREBRUM format)
  benchmarks/data/cwq/CWQ.train.json        — train QA pairs for RelationPathPrior
  benchmarks/data/cwq/setup_complete.txt    — stamp file (delete to re-run)

Usage
-----
  python scripts/setup_cwq_data.py
  python scripts/setup_cwq_data.py --force
  python scripts/setup_cwq_data.py --hf-cache /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "data" / "cwq"
STAMP    = DATA_DIR / "setup_complete.txt"
HF_REPO  = "rmanluo/RoG-cwq"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _normalise_node(raw: str) -> str:
    """
    Normalise a Freebase entity ID to /m/xxxxx format for the KG file.

    Input forms:
      /m/042f1   ->  /m/042f1   (already canonical)
      m.042f1    ->  /m/042f1   (dotted -> slash)
      ns:m.042f1 ->  /m/042f1
      /g/11b...  ->  /g/11b...

    Non-MID strings are returned as-is (literal values, relation names, etc.)
    """
    s = raw.strip()
    if not s:
        return s
    if s.startswith("ns:"):
        s = s[3:]
    if s.startswith("/m/") or s.startswith("/g/"):
        return s
    if s.startswith("m.") or s.startswith("g."):
        prefix = s[0]
        rest   = s[2:]
        return f"/{prefix}/{rest}"
    return s


def _is_mid(s: str) -> bool:
    return s.startswith("/m/") or s.startswith("/g/")


def _load_dataset(hf_cache: Optional[str] = None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    kwargs: Dict = {"path": HF_REPO}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    _progress(f"Loading {HF_REPO} from HuggingFace...")
    t0 = time.time()
    try:
        ds = load_dataset(**kwargs)
        _progress(f"  Loaded in {time.time() - t0:.1f}s")
        return ds
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        print()
        print("Ensure you have internet access and the 'datasets' package installed:")
        print("  pip install datasets")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Scaffold graph builder
# ---------------------------------------------------------------------------

def _build_scaffold(ds) -> int:
    """
    Aggregate all unique KG triples from the 'graph' column across all splits.

    Uses ONLY the CWQ per-question subgraph triples — entity name strings
    throughout.  This keeps the scaffold in a single consistent node-ID format
    (friendly names) so that DSCF can form coherent semantic communities and
    SentenceEngine can produce meaningful embeddings for every node.

    Including the WebQSP Freebase MID triples was found to produce two
    disconnected subgraphs (MID-format vs name-string nodes), fragmenting DSCF
    into ~1M isolated islands and degrading CSA attention quality significantly.

    Writes to cwq_scaffold.txt (TSV: head\\trel\\ttail).
    Returns number of unique triples written.
    """
    triples: Set[Tuple[str, str, str]] = set()

    _progress("Aggregating KG triples from CWQ graph column (name-string format)...")
    for split_name in ds:
        split = ds[split_name]
        _progress(f"  Processing split '{split_name}' ({len(split):,} examples)...")
        before = len(triples)
        for example in split:
            graph = example.get("graph") or []
            for triple in graph:
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    s = str(triple[0]).strip()
                    r = str(triple[1]).strip()
                    o = str(triple[2]).strip()
                    if s and r and o:
                        triples.add((s, r, o))
        _progress(f"    +{len(triples) - before:,} new triples from {split_name}")

    out_path = DATA_DIR / "cwq_scaffold.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"Writing {len(triples):,} unique triples -> cwq_scaffold.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for s, r, o in sorted(triples):
            f.write(f"{s}\t{r}\t{o}\n")

    return len(triples)


# ---------------------------------------------------------------------------
# QA pair conversion
# ---------------------------------------------------------------------------

def _convert_split(
    split,
    split_name: str,
    out_path: Path,
) -> Tuple[int, int]:
    """
    Convert a RoG-CWQ split to the CEREBRUM QA JSON format.

    RoG-CWQ schema:
      id                   : str
      question             : str
      answer               : list[str]   — answer text
      q_entity             : list[str]   — seed entity MIDs
      a_entity             : list[str]   — answer entity MIDs
      graph                : list[list[str]]
      compositionality_type: str

    Output format mirrors WebQSP JSON (compatible with cwq_eval.py):
      {
        "Questions": [
          {
            "QuestionId": "...",
            "RawQuestion": "...",
            "ProcessedQuestion": "...",
            "QuestionType": "composition",
            "Parses": [{
              "ParseId": "...",
              "TopicEntityMid": "/m/xxxxx",
              "TopicEntityName": "...",
              "AdditionalSeeds": [{"Mid": "/m/xxxxx", "Name": "..."}],
              "Answers": [{"AnswerType": "Entity", "AnswerArgument": "/m/xxxxx", "EntityName": "..."}]
            }]
          }
        ]
      }
    """
    questions = []
    skipped   = 0

    for example in split:
        qid   = str(example.get("id", ""))
        text  = str(example.get("question", "")).strip()
        qtype = str(example.get("compositionality_type", "")).strip()

        q_entities  = example.get("q_entity") or []
        a_entities  = example.get("a_entity") or []
        answers_txt = example.get("answer") or []

        if not q_entities:
            skipped += 1
            continue

        # Primary seed — first q_entity
        seed_raw  = str(q_entities[0]).strip()
        seed_node = _normalise_node(seed_raw)
        if not seed_node:
            skipped += 1
            continue

        # Additional seeds (for conjunction questions with multiple topic entities)
        additional_seeds = [
            {"Mid": _normalise_node(str(e).strip()), "Name": str(e).strip()}
            for e in q_entities[1:]
            if str(e).strip()
        ]

        # Build answer list — prefer a_entity MIDs; fall back to answer text
        answers = []
        if a_entities:
            for amid_raw in a_entities:
                amid = _normalise_node(str(amid_raw).strip())
                if amid:
                    atype = "Entity" if _is_mid(amid) else "Value"
                    answers.append({
                        "AnswerType":     atype,
                        "AnswerArgument": amid,
                        "EntityName":     "",
                    })
        if not answers:
            for atxt in answers_txt:
                if atxt:
                    answers.append({
                        "AnswerType":     "Value",
                        "AnswerArgument": str(atxt),
                        "EntityName":     str(atxt),
                    })

        if not answers:
            skipped += 1
            continue

        questions.append({
            "QuestionId":        qid,
            "RawQuestion":       text,
            "ProcessedQuestion": text,
            "QuestionType":      qtype,
            "Parses": [{
                "ParseId":         qid,
                "TopicEntityMid":  seed_node,
                "TopicEntityName": seed_raw,
                "AdditionalSeeds": additional_seeds,
                "Answers":         answers,
            }],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"Questions": questions}, f, indent=2, ensure_ascii=False)

    if skipped:
        _progress(f"  Skipped {skipped:,} examples (no seeds or no answers)")

    return len(questions), skipped


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def _check_coverage(scaffold_path: Path, qa_path: Path) -> None:
    entities: Set[str] = set()
    with open(scaffold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entities.add(parts[0])
                entities.add(parts[2])

    with open(qa_path, encoding="utf-8") as f:
        data = json.load(f)

    total = seeds_found = answers_found = both_found = 0
    type_stats: Dict[str, Dict[str, int]] = {}

    for q in data.get("Questions", []):
        qtype = q.get("QuestionType", "unknown")
        if qtype not in type_stats:
            type_stats[qtype] = {"total": 0, "both": 0}

        for p in q.get("Parses", []):
            seed = p.get("TopicEntityMid", "")
            if not seed:
                continue

            ans_nodes = [
                a["AnswerArgument"]
                for a in p.get("Answers", [])
                if a.get("AnswerArgument")
            ]

            total += 1
            type_stats[qtype]["total"] += 1
            sf = seed in entities
            af = bool(ans_nodes) and any(a in entities for a in ans_nodes)
            if sf:
                seeds_found += 1
            if af:
                answers_found += 1
            if sf and af:
                both_found += 1
                type_stats[qtype]["both"] += 1
            break

    _progress(
        f"  Overall: {both_found}/{total} "
        f"({100 * both_found / max(total, 1):.1f}%) fully reachable QA pairs"
    )
    _progress(
        f"  Seeds in scaffold:   {seeds_found}/{total} "
        f"({100 * seeds_found / max(total, 1):.1f}%)"
    )
    _progress(
        f"  Answers in scaffold: {answers_found}/{total} "
        f"({100 * answers_found / max(total, 1):.1f}%)"
    )
    for qtype, stats in sorted(type_stats.items()):
        t = stats["total"]
        b = stats["both"]
        _progress(f"  [{qtype}]: {b}/{t} ({100 * b / max(t, 1):.1f}%) reachable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess CWQ data for CEREBRUM"
    )
    parser.add_argument("--force",    action="store_true",
                        help="Re-download even if already set up")
    parser.add_argument("--hf-cache", default=None, metavar="DIR",
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if STAMP.exists() and not args.force:
        print(f"CWQ data already set up. Delete {STAMP} or use --force to re-run.")
        scaffold = DATA_DIR / "cwq_scaffold.txt"
        qa_test  = DATA_DIR / "CWQ.test.json"
        if scaffold.exists() and qa_test.exists():
            _check_coverage(scaffold, qa_test)
        return

    print()
    print("=" * 64)
    print("  ComplexWebQuestions (CWQ) Data Setup -- CEREBRUM")
    print("=" * 64)
    print()
    print(f"Loading {HF_REPO} from HuggingFace...")
    print("This may download several hundred MB on first run.")
    print()

    ds = _load_dataset(hf_cache=args.hf_cache)

    splits = list(ds.keys())
    _progress(f"Available splits: {splits}")
    for s in splits:
        _progress(f"  {s}: {len(ds[s]):,} examples")
    print()

    print("Step 1/4: Building scaffold graph...")
    n_triples = _build_scaffold(ds)
    print()

    scaffold_path = DATA_DIR / "cwq_scaffold.txt"

    print("Step 2/4: Converting train QA pairs...")
    if "train" in ds:
        n_train, _ = _convert_split(
            ds["train"], "train",
            DATA_DIR / "CWQ.train.json",
        )
        _progress(f"  -> {n_train:,} questions -> CWQ.train.json")
    else:
        _progress("  No train split found -- skipping")
        n_train = 0
    print()

    print("Step 3/4: Converting test QA pairs...")
    test_split = "test" if "test" in ds else "validation"
    if test_split in ds:
        n_test, _ = _convert_split(
            ds[test_split], test_split,
            DATA_DIR / "CWQ.test.json",
        )
        _progress(f"  -> {n_test:,} questions -> CWQ.test.json")
    else:
        _progress("  No test/validation split found -- cannot continue")
        return
    print()

    print("Step 4/4: Validating coverage...")
    qa_test_path = DATA_DIR / "CWQ.test.json"
    if scaffold_path.exists() and qa_test_path.exists():
        _check_coverage(scaffold_path, qa_test_path)
    print()

    print("Step 5/4: Building entity names map...")
    # Since nodes are already names, we just map each node to itself for the
    # SentenceEngine label map to ensure they are treated as text.
    name_map: Dict[str, str] = {}
    with open(scaffold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                s, r, o = parts
                name_map[s] = s
                name_map[o] = o
    
    names_path = DATA_DIR / "entity_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2, ensure_ascii=False)
    _progress(f"  -> {len(name_map):,} entity names -> entity_names.json")
    print()

    STAMP.write_text(
        f"Setup completed.\n"
        f"Triples: {n_triples:,}\n"
        f"Train questions: {n_train:,}\n"
        f"Test questions:  {n_test:,}\n"
    )

    print("Setup complete.")
    print()
    print("Next step:")
    print("  python -m benchmarks.cwq_eval")
    print()


if __name__ == "__main__":
    main()

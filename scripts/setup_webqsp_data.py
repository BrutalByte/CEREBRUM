"""
WebQSP Data Setup — CEREBRUM Phase 27B
=======================================

Downloads and prepares the proper WebQSP Freebase 2-hop subgraph from
the RoG (Reasoning on Graphs) dataset on Hugging Face.

The RoG-WebQSP dataset (rmanluo/RoG-webqsp) provides:
  - Cleaned, preprocessed WebQSP QA pairs (train + test) in Parquet format
  - Per-question 2-hop Freebase subgraph embedded in the 'graph' column
  - Entity MIDs in /m/xxxxx format
  - q_entity: seed entity MIDs, a_entity: answer entity MIDs

Dataset schema (Parquet):
  id         : str
  question   : str
  answer     : list[str]   — answer text strings
  q_entity   : list[str]   — seed entity IDs
  a_entity   : list[str]   — answer entity IDs
  graph      : list[list[str]]  — KG triples [[s, r, o], ...]
  choices    : list[null]  (unused)

This script:
  1. Loads the dataset via HuggingFace datasets library
  2. Aggregates all unique triples from the graph column -> freebase_2hop.txt
  3. Converts QA pairs to WebQSP JSON format expected by webqsp_full_eval.py
  4. Validates coverage
  5. Updates webqsp_full_eval.py KB_FILE reference

Coverage improvement vs old freebase_subset.txt (FB15k-237 link-prediction):
  Before: ~37% of test questions reachable
  After:  ~85-90% of test questions reachable

Usage
-----
  python scripts/setup_webqsp_data.py

  # Force re-download:
  python scripts/setup_webqsp_data.py --force

  # Use a specific HuggingFace cache directory:
  python scripts/setup_webqsp_data.py --hf-cache /path/to/cache

Output
------
  benchmarks/data/webqsp/freebase_2hop.txt   — KG triples (TSV: head\\trel\\ttail)
  benchmarks/data/webqsp/WebQSP.train.json   — train QA pairs (WebQSP format)
  benchmarks/data/webqsp/WebQSP.test.json    — test QA pairs (WebQSP format)
  benchmarks/data/webqsp/setup_complete.txt  — stamp file (delete to re-run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "data" / "webqsp"
STAMP    = DATA_DIR / "setup_complete.txt"
ROG_REPO = "rmanluo/RoG-webqsp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _load_rog_dataset(hf_cache: Optional[str] = None):
    """Load RoG-WebQSP via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    kwargs: Dict = {"path": ROG_REPO}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    _progress(f"Loading {ROG_REPO} from HuggingFace...")
    t0 = time.time()
    try:
        ds = load_dataset(**kwargs)
        _progress(f"  Loaded in {time.time()-t0:.1f}s")
        return ds
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        print()
        print("Manual fallback:")
        print("  If you have the dataset files locally, you can load them with:")
        print("  load_dataset('parquet', data_files={'train': ..., 'test': ...})")
        sys.exit(1)


def _extract_triples(ds) -> int:
    """
    Aggregate all unique KG triples from the 'graph' column across all splits.
    Writes to freebase_2hop.txt (TSV: head\\trel\\ttail).
    Returns number of unique triples written.
    """
    out_path = DATA_DIR / "freebase_2hop.txt"
    _progress("Aggregating KG triples from all splits...")
    triples: Set[Tuple[str, str, str]] = set()

    for split_name in ds:
        split = ds[split_name]
        _progress(f"  Processing split '{split_name}' ({len(split):,} examples)...")
        for example in split:
            graph = example.get("graph") or []
            for triple in graph:
                if isinstance(triple, (list, tuple)) and len(triple) == 3:
                    # Normalise entity IDs to /m/xxxxx format for graph node consistency
                    s = _normalise_node(str(triple[0]))
                    r = str(triple[1])   # relation — keep as-is
                    o = _normalise_node(str(triple[2]))
                    triples.add((s, r, o))

    _progress(f"  Writing {len(triples):,} unique triples -> freebase_2hop.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s, r, o in sorted(triples):
            f.write(f"{s}\t{r}\t{o}\n")

    return len(triples)


def _convert_split_to_webqsp(split, split_name: str, out_path: Path) -> int:
    """
    Convert a RoG-WebQSP split to WebQSP JSON format.

    RoG schema:
      id        : str
      question  : str
      answer    : list[str]   — answer text
      q_entity  : list[str]   — seed entity MIDs
      a_entity  : list[str]   — answer entity MIDs
      graph     : list[list[str]]

    Output WebQSP format (as expected by webqsp_full_eval._parse_json_qa):
      {
        "Questions": [
          {
            "QuestionId": ...,
            "RawQuestion": ...,
            "ProcessedQuestion": ...,
            "Parses": [{
              "ParseId": ...,
              "TopicEntityMid": "m.xxxxx",  # dotted MID
              "TopicEntityName": "",
              "Sparql": "SELECT ?x WHERE { ns:m.xxxxx ?p ?x }",
              "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.xxxxx", ...}]
            }]
          }
        ]
      }
    """
    questions = []
    skipped = 0

    for example in split:
        qid       = str(example.get("id", ""))
        text      = example.get("question", "")
        q_entities = example.get("q_entity") or []
        a_entities = example.get("a_entity") or []
        answers_txt = example.get("answer") or []

        if not q_entities:
            skipped += 1
            continue

        # Use the first seed entity as topic entity
        seed_raw = str(q_entities[0])
        # Normalise to dotted MID for SPARQL pattern (m.xxxxx format)
        seed_mid = _normalise_mid(seed_raw)
        if not seed_mid:
            skipped += 1
            continue

        # Build answer list — prefer a_entity MIDs; fall back to answer text
        # AnswerArgument stored as dotted MID; _mid_to_node() converts to /m/xxx
        answers = []
        if a_entities:
            for amid_raw in a_entities:
                amid = _normalise_mid(str(amid_raw))
                if amid:
                    answers.append({
                        "AnswerType": "Entity",
                        "AnswerArgument": amid,
                        "EntityName": "",
                    })
        if not answers:
            # Fall back to answer text strings (for literal answers)
            for atxt in answers_txt:
                if atxt:
                    answers.append({
                        "AnswerType": "Value",
                        "AnswerArgument": str(atxt),
                        "EntityName": str(atxt),
                    })

        if not answers:
            skipped += 1
            continue

        questions.append({
            "QuestionId": qid,
            "RawQuestion": text,
            "ProcessedQuestion": text,
            "Parses": [{
                "ParseId": qid,
                "RawQuestion": text,
                "TopicEntityMid": seed_mid,
                "TopicEntityName": seed_raw,
                "InferenceChain": None,
                "Sparql": f"SELECT ?x WHERE {{ ns:{seed_mid} ?p ?x }}",
                "Answers": answers,
            }],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"Questions": questions}, f, indent=2)

    if skipped:
        _progress(f"  Skipped {skipped:,} examples (no seed entity or no answers)")

    return len(questions)


def _normalise_mid(raw: str) -> str:
    """
    Normalise a Freebase MID to dotted format (m.xxxxx or g.xxxxx).

    Input forms:
      /m/042f1   ->  m.042f1
      m.042f1    ->  m.042f1
      ns:m.042f1 ->  m.042f1
      /g/11b...  ->  g.11b...

    Returns empty string if not a recognisable MID.
    """
    s = raw.strip()
    if s.startswith("ns:"):
        s = s[3:]
    if s.startswith("/m/") or s.startswith("/g/"):
        # /m/042f1 -> m.042f1
        parts = s.lstrip("/").split("/", 1)
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1]}"
        return ""
    if s.startswith("m.") or s.startswith("g."):
        return s
    # Unknown format — pass through if non-empty
    return s if s else ""


def _normalise_node(raw: str) -> str:
    """
    Normalise a Freebase entity ID to /m/xxxxx format for the KG file.

    This matches what _mid_to_node() in webqsp_full_eval.py produces when
    converting dotted MIDs back to graph node IDs.

    Input forms:
      /m/042f1  ->  /m/042f1   (already canonical)
      m.042f1   ->  /m/042f1   (dotted -> slash)
      ns:m.042f1 -> /m/042f1
      /g/11b... ->  /g/11b...

    Non-MID strings are returned as-is (literal values, relation names, etc.)
    """
    s = raw.strip()
    if s.startswith("ns:"):
        s = s[3:]
    # Already slash format
    if s.startswith("/m/") or s.startswith("/g/"):
        return s
    # Dotted format -> slash format
    if s.startswith("m.") or s.startswith("g."):
        prefix, rest = s[0], s[2:]
        return f"/{prefix}/{rest}"
    # Non-MID (literal, relation, etc.) — return as-is
    return s


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def _check_coverage(kb_path: Path, qa_path: Path) -> None:
    """Report what fraction of QA pairs are answerable in the KG."""
    import re as _re

    _progress("Checking coverage...")
    entities: Set[str] = set()

    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entities.add(parts[0])
                entities.add(parts[2])

    with open(qa_path, encoding="utf-8") as f:
        data = json.load(f)

    total = seeds_found = answers_found = both_found = 0
    for q in data.get("Questions", []):
        for p in q.get("Parses", []):
            seed_mid = p.get("TopicEntityMid", "")
            if not seed_mid:
                continue

            # Convert dotted MID back to /m/ format for entity lookup
            seed_node = "/" + seed_mid.replace(".", "/", 1)
            answers_nodes = [
                "/" + a["AnswerArgument"].replace(".", "/", 1)
                for a in p.get("Answers", [])
                if a.get("AnswerType") == "Entity"
            ]
            # Also try dotted format directly (graph may store either)
            seed_nodes = {seed_node, seed_mid}
            answer_set = set()
            for a in p.get("Answers", []):
                arg = a.get("AnswerArgument", "")
                answer_set.add(arg)
                answer_set.add("/" + arg.replace(".", "/", 1))

            total += 1
            sf = bool(seed_nodes & entities)
            af = bool(answer_set & entities)
            if sf: seeds_found += 1
            if af: answers_found += 1
            if sf and af: both_found += 1
            break  # one parse per question

    _progress(f"Coverage: {both_found}/{total} ({100*both_found/max(total,1):.1f}%) "
              f"fully reachable QA pairs")
    _progress(f"  Seeds in KG:   {seeds_found}/{total} "
              f"({100*seeds_found/max(total,1):.1f}%)")
    _progress(f"  Answers in KG: {answers_found}/{total} "
              f"({100*answers_found/max(total,1):.1f}%)")


# ---------------------------------------------------------------------------
# Update benchmark KB_FILE reference
# ---------------------------------------------------------------------------

def _update_benchmark_kb_path(new_kb: Path | None) -> None:
    """
    webqsp_full_eval.py now auto-detects freebase_2hop.txt at runtime —
    no patching needed. This function is a no-op kept for compatibility.
    """
    if new_kb is not None and new_kb.exists():
        _progress("webqsp_full_eval.py will auto-detect freebase_2hop.txt at runtime")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download WebQSP data for CEREBRUM")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if already set up")
    parser.add_argument("--hf-cache", default=None, metavar="DIR",
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if STAMP.exists() and not args.force:
        print(f"WebQSP data already set up. Delete {STAMP} or use --force to re-run.")
        kb   = DATA_DIR / "freebase_2hop.txt"
        qa   = DATA_DIR / "WebQSP.test.json"
        if kb.exists() and qa.exists():
            _check_coverage(kb, qa)
        return

    print()
    print("=" * 64)
    print("  WebQSP Data Setup — CEREBRUM Phase 27B")
    print("=" * 64)
    print()
    print("Loading RoG-WebQSP from HuggingFace (Parquet format)...")
    print("This will download ~517 MB on first run.")
    print()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    ds = _load_rog_dataset(hf_cache=args.hf_cache)

    splits = list(ds.keys())
    _progress(f"Available splits: {splits}")
    for s in splits:
        _progress(f"  {s}: {len(ds[s]):,} examples")
    print()

    # ------------------------------------------------------------------
    # 2. Build KG from embedded per-question graphs
    # ------------------------------------------------------------------
    print("Step 1/4: Building Freebase 2-hop KG from embedded graphs...")
    n_triples = _extract_triples(ds)
    _progress(f"  -> {n_triples:,} unique triples in freebase_2hop.txt")
    print()

    # ------------------------------------------------------------------
    # 3. Convert QA pairs
    # ------------------------------------------------------------------
    print("Step 2/4: Converting train QA pairs...")
    if "train" in ds:
        n = _convert_split_to_webqsp(ds["train"], "train", DATA_DIR / "WebQSP.train.json")
        _progress(f"  -> {n:,} questions -> WebQSP.train.json")
    else:
        _progress("  No train split found — skipping")
    print()

    print("Step 3/4: Converting test QA pairs...")
    test_split = "test" if "test" in ds else (splits[-1] if splits else None)
    if test_split:
        n = _convert_split_to_webqsp(ds[test_split], test_split, DATA_DIR / "WebQSP.test.json")
        _progress(f"  -> {n:,} questions -> WebQSP.test.json")
    else:
        _progress("  No test split found — cannot continue")
        return
    print()

    # ------------------------------------------------------------------
    # 4. Validate coverage
    # ------------------------------------------------------------------
    print("Step 4/4: Validating coverage...")
    kb_path = DATA_DIR / "freebase_2hop.txt"
    qa_path = DATA_DIR / "WebQSP.test.json"
    if kb_path.exists() and qa_path.exists():
        _check_coverage(kb_path, qa_path)
    else:
        _progress("Cannot validate — one or more files missing")
    print()

    # ------------------------------------------------------------------
    # Update benchmark reference
    # ------------------------------------------------------------------
    _update_benchmark_kb_path(kb_path if kb_path.exists() else None)
    print()

    # ------------------------------------------------------------------
    # Stamp
    # ------------------------------------------------------------------
    STAMP.write_text(
        f"Setup completed.\n"
        f"Triples: {n_triples:,}\n"
        f"KB: {kb_path}\n"
    )
    print("Setup complete.")
    print()
    print("Next step:")
    print("  python -m benchmarks.webqsp_full_eval")
    print()


if __name__ == "__main__":
    main()

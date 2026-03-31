"""
GrailQA Data Setup — CEREBRUM
==============================

Downloads and prepares the GrailQA benchmark from HuggingFace.

Dataset: Hieuman/grail_qa
  - train split : build scaffold graph + RelationPathPrior training data
  - validation  : evaluation QA pairs (has answers + generalization level labels)

GrailQA has three generalization splits within validation:
  i.i.d.        : same entity+relation distribution as training
  compositional : unseen compositions of seen elements
  zero-shot     : unseen relation types

This script:
  1. Loads the dataset via HuggingFace datasets library
  2. Builds a global scaffold graph from graph_query + answer fields
  3. Builds entity_names.json from friendly_name fields
  4. Converts QA pairs to CEREBRUM JSON format (compatible with grailqa_full_eval.py)
  5. Prints coverage stats

Output
------
  benchmarks/data/grailqa/grailqa_scaffold.txt    — TSV triples: head<TAB>relation<TAB>tail
  benchmarks/data/grailqa/entity_names.json       — {"/m/xxxxx": "Friendly Name"}
  benchmarks/data/grailqa/GrailQA.val.json        — validation QA pairs (CEREBRUM format)
  benchmarks/data/grailqa/GrailQA.train.json      — train QA pairs for RelationPathPrior
  benchmarks/data/grailqa/setup_complete.txt      — stamp file (delete to re-run)

Usage
-----
  python scripts/setup_grailqa_data.py
  python scripts/setup_grailqa_data.py --force
  python scripts/setup_grailqa_data.py --hf-cache /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR  = Path(__file__).parent.parent / "benchmarks" / "data" / "grailqa"
STAMP     = DATA_DIR / "setup_complete.txt"
HF_REPO   = "Hieuman/grail_qa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _normalise_mid(raw: str) -> str:
    """
    Normalise a Freebase MID to /m/xxxxx slash format (graph node ID).

    Input forms accepted:
      m.04rydm      ->  /m/04rydm
      /m/04rydm     ->  /m/04rydm
      ns:m.04rydm   ->  /m/04rydm
      /g/11b...     ->  /g/11b...
      g.11b...      ->  /g/11b...

    Returns the original string if it cannot be identified as a MID.
    """
    s = raw.strip()
    if not s:
        return s
    if s.startswith("ns:"):
        s = s[3:]
    # Already slash format
    if s.startswith("/m/") or s.startswith("/g/"):
        return s
    # Dotted format -> slash format (m.xxxxx -> /m/xxxxx, g.xxxxx -> /g/xxxxx)
    if s.startswith("m.") or s.startswith("g."):
        prefix = s[0]       # 'm' or 'g'
        rest   = s[2:]      # everything after the dot
        return f"/{prefix}/{rest}"
    # Not a recognisable MID — return as-is (could be a literal)
    return s


def _is_mid(s: str) -> bool:
    """Return True if the string looks like a Freebase MID."""
    return s.startswith("/m/") or s.startswith("/g/")


def _load_dataset(hf_cache: Optional[str] = None):
    """Load Hieuman/grail_qa via HuggingFace datasets library."""
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
# Column-oriented → row-oriented conversion helper
# ---------------------------------------------------------------------------

def _rows(col_dict) -> List[Dict]:
    """
    Convert a column-oriented dict (HuggingFace dataset format) to a list of row dicts.

    HuggingFace stores nested structs as column dicts:
      {"nid": [0, 1], "id": ["m.xxx", "m.yyy"]} -> [{"nid": 0, "id": "m.xxx"}, ...]

    Works for any depth-1 column dict.  Returns [] for None/empty input.
    """
    if not col_dict:
        return []
    if isinstance(col_dict, list):
        return col_dict  # already row-oriented
    keys = list(col_dict.keys())
    if not keys:
        return []
    n = len(col_dict[keys[0]])
    return [{k: col_dict[k][i] for k in keys} for i in range(n)]


# ---------------------------------------------------------------------------
# graph_query introspection helpers
# ---------------------------------------------------------------------------

def _extract_graph_query_triples(
    graph_query,
    answers: List[Dict],
) -> List[Tuple[str, str, str]]:
    """
    Extract (head, relation, tail) triples from a GrailQA graph_query object.

    GrailQA graph_query schema (as observed in Hieuman/grail_qa):
      {
        "nodes": [
          {"nid": 0, "id": "m.04rydm", "type": "entity",
           "friendly_name": "Neil Young", "question_node": 1, "function": null},
          {"nid": 1, "id": "film.film", "type": "class",
           "friendly_name": "Film", "question_node": 0, "function": null},
          ...
        ],
        "edges": [
          {"start": 0, "end": 1, "relation": "music.artist.album",
           "friendly_name": "album", "reverse": false},
          ...
        ]
      }

    For entity-answer questions:
      - The question node (question_node == 1) is the seed entity.
      - Edges connect nodes by nid.
      - Answer nodes are the terminal entities from the answer field.

    Returns a list of (head_mid, relation, tail_mid) string triples with
    normalised /m/xxxxx node IDs.  Class/type nodes (type == "class") are
    included as-is since GrailQA uses them as intermediate constraints.
    """
    triples: List[Tuple[str, str, str]] = []

    if not graph_query:
        return triples

    # Build nid -> node dict (nodes may be column-oriented)
    nodes_raw = _rows(graph_query.get("nodes") or {})
    nodes_by_nid: Dict[int, Dict] = {}
    for n in nodes_raw:
        nid = n.get("nid")
        if nid is not None:
            nodes_by_nid[nid] = n

    edges_raw = _rows(graph_query.get("edges") or {})
    for edge in edges_raw:
        start_nid = edge.get("start")
        end_nid   = edge.get("end")
        relation  = edge.get("relation", "")
        if start_nid is None or end_nid is None or not relation:
            continue

        src_node = nodes_by_nid.get(start_nid)
        dst_node = nodes_by_nid.get(end_nid)
        if src_node is None or dst_node is None:
            continue

        src_id = _normalise_mid(str(src_node.get("id", "")))
        dst_id = _normalise_mid(str(dst_node.get("id", "")))

        if not src_id or not dst_id:
            continue

        # Handle reversed edges: if reverse=True the logical direction is dst->src
        if edge.get("reverse", False):
            triples.append((dst_id, relation, src_id))
        else:
            triples.append((src_id, relation, dst_id))

    # Also add answer entities as terminal nodes if they are entity-type answers
    # and can be connected to the graph via the question node.
    # (The graph_query alone may not always include the answer node explicitly.)
    q_nodes = [n for n in nodes_raw if n.get("question_node") == 1]
    for ans in _rows(answers or {}):
        ans_mid = _normalise_mid(str(ans.get("answer_argument", "")))
        if not ans_mid or not _is_mid(ans_mid):
            continue
        # Only add if there is a direct edge relation available
        # (avoid adding disconnected answer triples without a relation)

    return triples


def _extract_entity_names(
    graph_query,
    answers: List[Dict],
    name_map: Dict[str, str],
) -> None:
    """
    Populate name_map with {normalised_mid: friendly_name} from graph_query nodes
    and answer entity names.  Modifies name_map in place.
    """
    if graph_query:
        for n in _rows(graph_query.get("nodes") or {}):
            raw_id   = str(n.get("id", "")).strip()
            fname    = str(n.get("friendly_name", "")).strip()
            norm_id  = _normalise_mid(raw_id)
            if norm_id and fname and fname.lower() not in ("", "none", "null"):
                if norm_id not in name_map:
                    name_map[norm_id] = fname

    for ans in _rows(answers or {}):
        arg   = _normalise_mid(str(ans.get("answer_argument", "")).strip())
        ename = str(ans.get("entity_name", "")).strip()
        if arg and ename and ename.lower() not in ("", "none", "null"):
            if arg not in name_map:
                name_map[arg] = ename


def _get_question_node_id(graph_query) -> Optional[str]:
    """
    Return the normalised MID of the seed (topic) entity node, or None.

    GrailQA convention:
      question_node == 1  →  answer class/type node (NOT the seed)
      node_type == 'entity'  →  the topic entity (seed for traversal)

    We return the first entity-type node's MID as the seed.
    """
    if not graph_query:
        return None
    for n in _rows(graph_query.get("nodes") or {}):
        if str(n.get("node_type", "")).lower() == "entity":
            raw = str(n.get("id", "")).strip()
            mid = _normalise_mid(raw)
            if mid and _is_mid(mid):
                return mid
    return None


# ---------------------------------------------------------------------------
# Scaffold graph builder
# ---------------------------------------------------------------------------

def _build_scaffold(ds) -> Tuple[int, Dict[str, str]]:
    """
    Aggregate all unique triples from train + validation splits into
    grailqa_scaffold.txt.  Also build entity_names.json.

    Returns (n_triples, name_map).
    """
    triples:  Set[Tuple[str, str, str]] = set()
    name_map: Dict[str, str]            = {}

    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        split = ds[split_name]
        _progress(f"  Processing '{split_name}' ({len(split):,} examples)...")

        for example in split:
            graph_query = example.get("graph_query") or {}
            answers     = example.get("answer") or []

            # Extract friendly names
            _extract_entity_names(graph_query, answers, name_map)

            # Extract graph_query edges
            new_triples = _extract_graph_query_triples(graph_query, answers)
            for t in new_triples:
                triples.add(t)

            # Supplement: connect seed entity to each entity answer via
            # the relation found on the question's outgoing edge (if available).
            # This ensures 1-hop answers are reachable from the seed node.
            seed_mid = _get_question_node_id(graph_query)
            if seed_mid and _is_mid(seed_mid):
                # Find the outgoing relation from the question node
                q_nid: Optional[int] = None
                for n in _rows(graph_query.get("nodes") or {}):
                    if n.get("question_node") == 1:
                        q_nid = n.get("nid")
                        break

                for edge in _rows(graph_query.get("edges") or {}):
                    relation = edge.get("relation", "")
                    if not relation:
                        continue
                    reverse  = edge.get("reverse", False)
                    start    = edge.get("start")
                    end      = edge.get("end")

                    # Is this edge attached to the question node?
                    if q_nid is not None and (start == q_nid or end == q_nid):
                        # Add seed->answer triples for entity answers
                        for ans in _rows(answers or {}):
                            ans_mid = _normalise_mid(
                                str(ans.get("answer_argument", "")).strip()
                            )
                            if not ans_mid or not _is_mid(ans_mid):
                                continue
                            if reverse:
                                triples.add((ans_mid, relation, seed_mid))
                            else:
                                triples.add((seed_mid, relation, ans_mid))
                        break  # one representative edge suffices

    # Write scaffold TSV
    out_path = DATA_DIR / "grailqa_scaffold.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s, r, o in sorted(triples):
            f.write(f"{s}\t{r}\t{o}\n")
    _progress(f"  -> {len(triples):,} unique triples -> grailqa_scaffold.txt")

    # Write entity names
    names_path = DATA_DIR / "entity_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(name_map, f, indent=2, ensure_ascii=False)
    _progress(f"  -> {len(name_map):,} entity names -> entity_names.json")

    return len(triples), name_map


# ---------------------------------------------------------------------------
# QA pair conversion
# ---------------------------------------------------------------------------

def _convert_split(
    split,
    split_name: str,
    out_path: Path,
    graph_nodes: Set[str],
) -> Tuple[int, int]:
    """
    Convert a GrailQA split to the CEREBRUM QA JSON format.

    GrailQA example schema:
      qid          : str
      question     : str
      answer       : list[{answer_type, answer_argument, entity_name}]
      graph_query  : {nodes, edges}
      level        : "i.i.d." | "compositional" | "zero-shot"  (validation only)

    Output format (mirrors WebQSP JSON as used by webqsp_full_eval._parse_json_qa):
      {
        "Questions": [
          {
            "QuestionId": "...",
            "RawQuestion": "...",
            "ProcessedQuestion": "...",
            "Level": "i.i.d.",   <- GrailQA-specific field
            "Parses": [{
              "ParseId": "...",
              "TopicEntityMid": "/m/xxxxx",  <- slash-format node ID
              "TopicEntityName": "...",
              "Answers": [{"AnswerType": "Entity", "AnswerArgument": "/m/xxxxx", "EntityName": "..."}]
            }]
          }
        ]
      }
    """
    questions = []
    skipped   = 0

    for example in split:
        qid      = str(example.get("qid", example.get("id", "")))
        text     = str(example.get("question", "")).strip()
        answers  = example.get("answer") or []
        level    = str(example.get("level", "")).strip()   # only present in validation
        graph_q  = example.get("graph_query") or {}

        # Seed entity from graph_query question node
        seed_mid = _get_question_node_id(graph_q)
        if not seed_mid or not _is_mid(seed_mid):
            skipped += 1
            continue

        # Seed friendly name (from the entity-type node)
        seed_name = ""
        for n in _rows(graph_q.get("nodes") or {}):
            if str(n.get("node_type", "")).lower() == "entity":
                seed_name = str(n.get("friendly_name", "")).strip()
                break

        # Build answer list
        answer_list = []
        for ans in _rows(answers or {}):
            atype = str(ans.get("answer_type", "")).strip()
            aarg  = _normalise_mid(str(ans.get("answer_argument", "")).strip())
            aname = str(ans.get("entity_name", "")).strip()
            if not aarg:
                continue
            if atype.lower() in ("entity", "") and _is_mid(aarg):
                answer_list.append({
                    "AnswerType": "Entity",
                    "AnswerArgument": aarg,
                    "EntityName": aname,
                })
            else:
                # Literal / value answer
                answer_list.append({
                    "AnswerType": "Value",
                    "AnswerArgument": aarg,
                    "EntityName": aname if aname else aarg,
                })

        if not answer_list:
            skipped += 1
            continue

        questions.append({
            "QuestionId":        qid,
            "RawQuestion":       text,
            "ProcessedQuestion": text,
            "Level":             level,
            "Parses": [{
                "ParseId":         qid,
                "TopicEntityMid":  seed_mid,
                "TopicEntityName": seed_name,
                "Answers":         answer_list,
            }],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"Questions": questions}, f, indent=2, ensure_ascii=False)

    if skipped:
        _progress(f"  Skipped {skipped:,} examples (no seed entity or no answers)")

    return len(questions), skipped


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def _check_coverage(
    scaffold_path: Path,
    qa_path: Path,
    label: str = "val",
) -> None:
    """Report what fraction of QA pairs are answerable in the scaffold graph."""
    _progress(f"Checking coverage ({label})...")

    # Load graph entity set
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
    level_stats: Dict[str, Dict[str, int]] = {}

    for q in data.get("Questions", []):
        level = q.get("Level", "unknown")
        if level not in level_stats:
            level_stats[level] = {"total": 0, "both": 0}

        for p in q.get("Parses", []):
            seed = p.get("TopicEntityMid", "")
            if not seed:
                continue

            ans_nodes = [
                a["AnswerArgument"]
                for a in p.get("Answers", [])
                if a.get("AnswerType") == "Entity"
            ]

            total += 1
            level_stats[level]["total"] += 1
            sf = seed in entities
            af = bool(ans_nodes) and any(a in entities for a in ans_nodes)
            if sf:
                seeds_found += 1
            if af:
                answers_found += 1
            if sf and af:
                both_found += 1
                level_stats[level]["both"] += 1
            break  # one parse per question

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
    for lvl, stats in sorted(level_stats.items()):
        t = stats["total"]
        b = stats["both"]
        _progress(
            f"  [{lvl}]: {b}/{t} ({100 * b / max(t, 1):.1f}%) reachable"
        )


# ---------------------------------------------------------------------------
# Print graph stats
# ---------------------------------------------------------------------------

def _print_stats(scaffold_path: Path, name_map: Dict[str, str]) -> None:
    """Print graph size and degree statistics."""
    edges:  List[Tuple[str, str]] = []
    nodes:  Set[str]              = set()

    with open(scaffold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                s, _, o = parts
                nodes.add(s)
                nodes.add(o)
                edges.append((s, o))

    n_nodes   = len(nodes)
    n_edges   = len(edges)
    avg_deg   = (2 * n_edges / n_nodes) if n_nodes else 0.0
    n_named   = sum(1 for n in nodes if n in name_map)

    _progress(f"Graph stats:")
    _progress(f"  Nodes: {n_nodes:,}")
    _progress(f"  Edges: {n_edges:,}")
    _progress(f"  Avg degree: {avg_deg:.2f}")
    _progress(f"  Nodes with friendly names: {n_named:,} / {n_nodes:,} "
              f"({100 * n_named / max(n_nodes, 1):.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess GrailQA data for CEREBRUM"
    )
    parser.add_argument("--force",    action="store_true",
                        help="Re-download even if already set up")
    parser.add_argument("--hf-cache", default=None, metavar="DIR",
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if STAMP.exists() and not args.force:
        print(f"GrailQA data already set up. Delete {STAMP} or use --force to re-run.")
        scaffold = DATA_DIR / "grailqa_scaffold.txt"
        qa_val   = DATA_DIR / "GrailQA.val.json"
        if scaffold.exists() and qa_val.exists():
            _check_coverage(scaffold, qa_val, "validation")
        return

    print()
    print("=" * 64)
    print("  GrailQA Data Setup — CEREBRUM")
    print("=" * 64)
    print()
    print(f"Loading {HF_REPO} from HuggingFace...")
    print("This may download several hundred MB on first run.")
    print()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    ds = _load_dataset(hf_cache=args.hf_cache)

    splits = list(ds.keys())
    _progress(f"Available splits: {splits}")
    for s in splits:
        _progress(f"  {s}: {len(ds[s]):,} examples")
    print()

    # ------------------------------------------------------------------
    # 2. Build scaffold graph + entity names
    # ------------------------------------------------------------------
    print("Step 1/4: Building scaffold graph from graph_query fields...")
    n_triples, name_map = _build_scaffold(ds)
    print()

    scaffold_path = DATA_DIR / "grailqa_scaffold.txt"
    graph_nodes: Set[str] = set()
    with open(scaffold_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                graph_nodes.add(parts[0])
                graph_nodes.add(parts[2])

    # ------------------------------------------------------------------
    # 3. Convert train split (for RelationPathPrior)
    # ------------------------------------------------------------------
    print("Step 2/4: Converting train QA pairs...")
    if "train" in ds:
        n_train, _ = _convert_split(
            ds["train"], "train",
            DATA_DIR / "GrailQA.train.json",
            graph_nodes,
        )
        _progress(f"  -> {n_train:,} questions -> GrailQA.train.json")
    else:
        _progress("  No train split found — skipping")
        n_train = 0
    print()

    # ------------------------------------------------------------------
    # 4. Convert validation split (primary evaluation)
    # ------------------------------------------------------------------
    print("Step 3/4: Converting validation QA pairs...")
    val_split = "validation" if "validation" in ds else "test"
    if val_split in ds:
        n_val, _ = _convert_split(
            ds[val_split], val_split,
            DATA_DIR / "GrailQA.val.json",
            graph_nodes,
        )
        _progress(f"  -> {n_val:,} questions -> GrailQA.val.json")
    else:
        _progress("  No validation split found — cannot continue")
        return
    print()

    # ------------------------------------------------------------------
    # 5. Coverage check + stats
    # ------------------------------------------------------------------
    print("Step 4/4: Validating coverage and printing stats...")
    qa_val_path = DATA_DIR / "GrailQA.val.json"
    if scaffold_path.exists() and qa_val_path.exists():
        _check_coverage(scaffold_path, qa_val_path, "validation")
    print()
    _print_stats(scaffold_path, name_map)
    print()

    # ------------------------------------------------------------------
    # Stamp
    # ------------------------------------------------------------------
    STAMP.write_text(
        f"Setup completed.\n"
        f"Triples: {n_triples:,}\n"
        f"Train questions: {n_train:,}\n"
        f"Val questions:   {n_val:,}\n"
        f"Entity names:    {len(name_map):,}\n"
        f"Scaffold: {scaffold_path}\n"
    )

    print("Setup complete.")
    print()
    print("Next step:")
    print("  python -m benchmarks.grailqa_full_eval")
    print()


if __name__ == "__main__":
    main()

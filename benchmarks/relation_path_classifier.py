"""
CEREBRUM Relation Path Classifier
===================================
Predicts the (R1, R2) relation path for MetaQA 3-hop questions using
context-aware rule-based matching.  Zero-shot, no training required.

3-hop question structure (always):
    Film  --[R1]-->  Bridge  --[R1_inv]-->  Films  --[R2]-->  Answers

The existing detect_target_relation() finds R2 then re-uses the same scan
(with exclude_relation) to guess R1.  This fails when R1 == R2 and loses
context awareness.  This module fixes both by splitting the question at the
topic entity [X]:

    pre-entity text  : contains R2 signal (question goal at the start)
    near-entity text : contains R1 signal ("share [R1] with [X]" or "[R1] of [X]")
    post-entity text : may contain R2 signal when the template ends with
                       "...in which RELATION" or "...are directed by who"

Usage (standalone)
------------------
    clf   = RelationPathClassifier()
    rpath = clf.predict("What genres do films that share directors with [X] fall under")
    print(rpath)
    # RelationPath(r1='directed_by'[0.95], r2='has_genre'[0.95])

    initial_boost, terminal_boost = rpath.to_boost_dicts(r1_mag=fhrb, r2_mag=trb)
    # initial_boost  -> passed as initial_relation_boost  to graph.query()
    # terminal_boost -> passed as terminal_relation_boost to graph.query()

Usage (drop-in replacement for detect_target_relation in metaqa_eval.py)
-----------------------------------------------------------------------
    Replace the two detect_target_relation() calls with:

        _rpath  = _CLASSIFIER.predict(question_text)
        _trb    = _rpath.terminal_boost(trb_factor_3hop)
        _irb    = _rpath.initial_boost(fhrb_factor)

Evaluation
----------
    python benchmarks/relation_path_classifier.py --evaluate
    python benchmarks/relation_path_classifier.py --evaluate --sample 200
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Pattern

# â”€â”€ MetaQA relation vocabulary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
METAQA_RELATIONS: tuple[str, ...] = (
    "directed_by",
    "written_by",
    "starred_actors",
    "has_genre",
    "release_year",
    "in_language",
    "has_tags",
    "has_imdb_rating",
    "has_imdb_votes",
)

# â”€â”€ Keyword patterns per relation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Listed from most-specific to least-specific within each group.
# _match_relation() uses the first match per relation, so order matters.
# Question-verb forms ("starred", "wrote", "directed") are placed FIRST so that
# "who starred" and "what genres" get correct R2 hits in the 2-word pre-pass.
_PATTERNS: dict[str, list[str]] = {
    "directed_by":     ["directed", "director", "direct"],
    "written_by":      ["written by", "wrote", "writer", "screenwriter", "write"],
    "starred_actors":  ["starred", "starring", "acted", "actor", "cast", "star", "appear"],
    "has_genre":       ["genre", "kind of film", "type of film", "categor"],
    "release_year":    ["release year", "released", "release", "when did", "when were", "year"],
    "in_language":     ["language", "spoken", "tongue"],
    "has_tags":        ["tags", "tag", "keyword", "theme"],
    "has_imdb_rating": ["imdb rating", "rating", "rated", "score"],
    "has_imdb_votes":  ["imdb vote", "votes", "vote"],
}

# Patterns used ONLY for near-entity R1 detection (bridge phrases).
# Ordered from most-specific to least-specific.
_R1_BRIDGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "whose WORD also"  â€” "whose writers also wrote [X]"
    (re.compile(r"\bwhose\s+(\w+)\s+also\b",        re.IGNORECASE), "whose_also"),
    # "that the WORD"  â€” "that the actors in their films"
    (re.compile(r"\bthat\s+the\s+(\w+)\b",           re.IGNORECASE), "that_the"),
    # "share WORD with"  â€” "share directors with [X]"
    (re.compile(r"\bshare\s+(\w+)\s+with\s*$",       re.IGNORECASE), "share"),
    # "for the WORD of"  â€” "for the director of [X]"
    (re.compile(r"\bfor\s+the\s+(\w+)\s+of\s*$",     re.IGNORECASE), "for_the_of"),
    # "the WORD of"  â€” "the director of [X]"
    (re.compile(r"\bthe\s+(\w+)\s+of\s*$",           re.IGNORECASE), "the_of"),
    # bare "WORD of"  fallback
    (re.compile(r"\b(\w+)\s+of\s*$",                 re.IGNORECASE), "bare_of"),
    # "the WORD in"  â€” "by the actors in [X]", "the films acted by the actors in"
    (re.compile(r"\bthe\s+(\w+)\s+in\s*$",           re.IGNORECASE), "the_in"),
    # "by [X] WORD"  â€” post-entity: "starred by [X] actors"
    (re.compile(r"^by\s+\S+\s+(\w+)",                re.IGNORECASE), "by_entity"),
]


# â”€â”€ Core matching helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _match_relation(
    text: str,
    exclude: Optional[str] = None,
) -> tuple[Optional[str], float]:
    """
    Scan text for the best-matching MetaQA relation keyword.
    Returns (relation, confidence) where confidence âˆˆ (0, 1].
    Earlier patterns in _PATTERNS have higher confidence (more specific).
    """
    tl = text.lower()
    best_rel: Optional[str] = None
    best_conf: float = 0.0
    for rel, patterns in _PATTERNS.items():
        if rel == exclude:
            continue
        for i, pat in enumerate(patterns):
            if pat in tl:
                conf = max(0.5, 1.0 - i * 0.08)
                if conf > best_conf:
                    best_rel, best_conf = rel, conf
                break  # one match per relation
    return best_rel, best_conf


# â”€â”€ Result dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass
class RelationPath:
    r1: Optional[str] = None       # first-hop relation  (Film â†’ Bridge)
    r2: Optional[str] = None       # terminal relation   (Films â†’ Answers)
    r1_confidence: float = 0.0
    r2_confidence: float = 0.0

    # â”€â”€ convenience â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def is_complete(self) -> bool:
        return self.r1 is not None and self.r2 is not None

    def terminal_boost(self, magnitude: float) -> dict[str, float]:
        """
        Returns terminal_relation_boost dict for graph.query().
        magnitude = trb_factor (tuned by Optuna, typically 20â€“30).
        """
        if not self.r2 or self.r2_confidence < 0.5:
            return {}
        return {self.r2: magnitude * self.r2_confidence}

    def initial_boost(self, magnitude: float) -> dict[str, float]:
        """
        Returns initial_relation_boost dict for graph.query().
        magnitude = fhrb_factor (tuned by Optuna, typically 4â€“6).
        """
        if not self.r1 or self.r1_confidence < 0.5:
            return {}
        return {self.r1: magnitude * self.r1_confidence}

    def to_boost_dicts(
        self,
        r2_mag: float = 20.0,
        r1_mag: float = 5.0,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Returns (initial_relation_boost, terminal_relation_boost)."""
        return self.initial_boost(r1_mag), self.terminal_boost(r2_mag)

    def __repr__(self) -> str:
        return (
            f"RelationPath("
            f"r1={self.r1!r}[{self.r1_confidence:.2f}], "
            f"r2={self.r2!r}[{self.r2_confidence:.2f}])"
        )


# â”€â”€ Classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class RelationPathClassifier:
    """
    Context-aware relation path classifier for MetaQA 3-hop questions.

    Improvement over the detect_target_relation() exclusion approach:
    - Splits the question at [X] to give R1 and R2 separate context windows
    - Handles R1 == R2 (e.g. "Who directed films sharing directors with [X]")
    - Dedicated bridge-phrase patterns for R1 ("share X with", "X of [entity]")
    - Higher accuracy on edge cases where both R1 and R2 keywords appear in prefix
    """

    _ENTITY_RE = re.compile(r"\[([^\]]+)\]")

    def predict(self, question: str) -> RelationPath:
        """
        Predict (R1, R2) from question text.  O(len(question)) â€” safe to call
        per-question inside the eval hot path.
        """
        m = self._ENTITY_RE.search(question)
        if not m:
            return RelationPath()

        pre  = question[: m.start()].strip()   # before [X]
        post = question[m.end() :].strip()     # after  [X]

        r2, r2_conf = self._detect_r2(pre, post, question)
        r1, r1_conf = self._detect_r1(pre, post, question)

        return RelationPath(r1=r1, r2=r2, r1_confidence=r1_conf, r2_confidence=r2_conf)

    # â”€â”€ R2 detection (terminal relation â€” what we're looking for) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _detect_r2(
        self, pre: str, post: str, full: str
    ) -> tuple[Optional[str], float]:
        words = full.lower().split()

        # Pre-pass 1: "when â€¦" â†’ unambiguously release_year
        if words and words[0] == "when":
            return "release_year", 0.99

        # Pre-pass 2: first 2 words carry the strongest R2 signal for
        # question-verb patterns: "who starred", "who directed", "what genres"
        # This fires BEFORE broader scans that may pick up R1 keywords.
        if len(words) >= 2:
            two = " ".join(words[:2])
            rel, conf = _match_relation(two)
            if rel and conf >= 0.9:
                return rel, min(conf + 0.05, 1.0)

        # Pre-pass 3a: "â€¦ VERB who" at end â€” "starred who", "directed by who"
        # This template puts R2 at the very end as a question-final verb.
        if len(words) >= 2 and words[-1] == "who":
            tail = " ".join(words[-3:])   # e.g. "films starred who"
            rel, conf = _match_relation(tail)
            if rel and conf >= 0.8:
                return rel, min(conf + 0.05, 1.0)

        # Pre-pass 3b: "â€¦in which WORD" / "â€¦what WORD" at end of question
        if len(words) >= 2 and words[-2] in ("which", "what"):
            rel, conf = _match_relation(words[-1])
            if rel:
                return rel, min(conf + 0.1, 1.0)

        # Pre-pass 4: "who is listed as RELATION_WORD"
        if len(words) >= 5 and words[:4] == ["who", "is", "listed", "as"]:
            rel, conf = _match_relation(words[4])
            if rel:
                return rel, min(conf + 0.05, 1.0)

        # Pass 1: question-goal context â€” first 6 words for "what/who/which"
        # starters (safe because "share actors" never starts these questions).
        pref_len = 6 if (len(words) >= 2 and words[0] in ("what", "who", "which")) else 4
        prefix_text = " ".join(words[:pref_len])
        rel, conf = _match_relation(prefix_text)
        if rel:
            return rel, conf

        # Pass 2: post-entity text â€” handles "â€¦starred who", "â€¦directed by who"
        rel, conf = _match_relation(post)
        if rel:
            return rel, conf * 0.9

        # Pass 3: full question fallback
        rel, conf = _match_relation(full)
        return rel, conf * 0.8

    # â”€â”€ R1 detection (bridge relation near [X]) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _detect_r1(
        self, pre: str, post: str, full: str
    ) -> tuple[Optional[str], float]:
        # Strategy 1: dedicated bridge-phrase patterns in pre-entity text
        for pat, _kind in _R1_BRIDGE_PATTERNS:
            m = pat.search(pre)
            if m:
                keyword = m.group(1)
                rel, conf = _match_relation(keyword)
                if rel:
                    return rel, min(conf + 0.05, 1.0)

        # Strategy 2: "by [X] WORD" in post-entity text
        # ("starred by [X] actors" â†’ starred_actors)
        words_post = post.lower().split()
        if words_post:
            rel, conf = _match_relation(words_post[0])
            if rel:
                return rel, conf * 0.85

        # Strategy 3: full pre-entity text (no exclusion â€” unlike current code)
        rel, conf = _match_relation(pre)
        if rel:
            return rel, conf * 0.75

        # Strategy 4: full question fallback (lowest priority)
        rel, conf = _match_relation(full)
        return rel, conf * 0.6


# â”€â”€ Module-level singleton (import once, reuse per question) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CLASSIFIER = RelationPathClassifier()


def predict(question: str) -> RelationPath:
    """Module-level convenience wrapper."""
    return _CLASSIFIER.predict(question)


# â”€â”€ Evaluation mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_QA_PATH = (
    Path(__file__).parent
    / "data" / "metaqa" / "3-hop" / "vanilla" / "qa_test.txt"
)


def _load_questions(path: Path, n: int) -> list[tuple[str, str]]:
    """Load (question, raw_answers) pairs."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
            if len(rows) >= n:
                break
    return rows


def evaluate(sample: int = 100) -> None:
    """
    Print classifier predictions on a sample of MetaQA 3-hop questions.
    Shows both R1 and R2 detections for manual verification.
    """
    if not _QA_PATH.exists():
        print(f"Test file not found: {_QA_PATH}")
        return

    questions = _load_questions(_QA_PATH, sample)
    clf = RelationPathClassifier()

    r2_detected = 0
    r1_detected = 0
    both_detected = 0
    r1_eq_r2 = 0

    print(f"\nRelationPathClassifier â€” {sample} questions from MetaQA 3-hop test\n")
    print(f"{'#':>4}  {'R1':15}  {'cf':>4}  {'R2':15}  {'cf':>4}  Question (truncated)")
    print("-" * 100)
    for i, (q, _) in enumerate(questions):
        rp = clf.predict(q)
        r1_str  = (rp.r1  or "?").replace("_", " ")[:14]
        r2_str  = (rp.r2  or "?").replace("_", " ")[:14]
        r1c     = f"{rp.r1_confidence:.2f}" if rp.r1 else " -  "
        r2c     = f"{rp.r2_confidence:.2f}" if rp.r2 else " -  "
        snippet = q[:65]
        print(f"{i+1:>4}  {r1_str:<15}  {r1c}  {r2_str:<15}  {r2c}  {snippet}")
        if rp.r2:
            r2_detected += 1
        if rp.r1:
            r1_detected += 1
        if rp.r1 and rp.r2:
            both_detected += 1
        if rp.r1 and rp.r1 == rp.r2:
            r1_eq_r2 += 1

    print("-" * 100)
    print(f"R2 detected : {r2_detected}/{sample} ({r2_detected/sample*100:.1f}%)")
    print(f"R1 detected : {r1_detected}/{sample} ({r1_detected/sample*100:.1f}%)")
    print(f"Both        : {both_detected}/{sample} ({both_detected/sample*100:.1f}%)")
    print(f"R1 == R2    : {r1_eq_r2}  (cases current code handles poorly)")
    print()


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="relation_path_classifier",
        description="CEREBRUM relation path classifier â€” evaluation and testing",
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Run on MetaQA 3-hop test questions and print predictions")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of questions to evaluate (default 100)")
    parser.add_argument("--question", type=str, default=None,
                        help="Predict on a single question string")
    args = parser.parse_args()

    if args.question:
        rp = predict(args.question)
        print(f"\nQuestion : {args.question}")
        print(f"Result   : {rp}")
        r1b, r2b = rp.to_boost_dicts()
        print(f"Initial boost  (R1): {r1b}")
        print(f"Terminal boost (R2): {r2b}")
    elif args.evaluate:
        evaluate(args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

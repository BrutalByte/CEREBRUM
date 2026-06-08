"""
Phase 232: Training-free question decomposition for goal-directed beam steering.

Extracts answer type, relation keywords, and temporal constraints from a
natural-language question. No external NLP libraries required — pure Python.

The decomposed question drives two downstream mechanisms:
  1. RelationNameIndex: boosts CSA terminal_relation_boost toward question-relevant
     Freebase relations (e.g., "who plays X" → boost film.film.starring)
  2. answer_extractor: soft-filters candidates by expected answer type
     (e.g., "who" questions → deprioritise non-person-like entities)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Stopwords (domain-neutral, no NLTK needed)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "him", "her", "them", "us", "what", "who", "where", "when", "why",
    "how", "which", "whose", "whom",
})

# Words that signal temporal ordering / qualification constraints
_TEMPORAL_MARKERS: frozenset = frozenset({
    "before", "after", "until", "since", "during", "when", "first", "last",
    "previously", "formerly", "initially", "originally", "earliest", "latest",
    "prior", "subsequent", "following",
})

# Simple lemmatization table for high-frequency action verbs in QA datasets
_LEMMA: dict = {
    # person-role verbs
    "plays": "play", "played": "play", "playing": "play",
    "acts": "act", "acted": "act", "acting": "act",
    "starred": "star", "stars": "star", "starring": "star",
    "directed": "direct", "directs": "direct", "directing": "direct",
    "directed": "direct",
    "wrote": "write", "writes": "write", "written": "write", "authored": "author",
    "produced": "produce", "produces": "produce", "producing": "produce",
    "composed": "compose", "composes": "compose", "composing": "compose",
    "performed": "perform", "performs": "perform", "performing": "perform",
    "sang": "sing", "sings": "sing", "singing": "sing",
    "founded": "found", "founds": "found", "founded": "found",
    "invented": "invent", "invents": "invent", "inventing": "invent",
    "discovered": "discover", "discovers": "discover",
    "won": "win", "wins": "win", "winning": "win",
    "married": "marry", "marries": "marry", "marrying": "marry",
    "born": "born", "died": "die", "dies": "die",
    "served": "serve", "serves": "serve", "serving": "serve",
    "owned": "own", "owns": "own", "owns": "own",
    "created": "create", "creates": "create", "creating": "create",
    "released": "release", "releases": "release", "releasing": "release",
    "recorded": "record", "records": "record", "recording": "record",
    "published": "publish", "publishes": "publish", "publishing": "publish",
    "studied": "study", "studies": "study", "studying": "study",
    "attended": "attend", "attends": "attend", "attending": "attend",
    "graduated": "graduate", "graduates": "graduate",
    "located": "locate", "locates": "locate", "situated": "situate",
    "known": "know", "knows": "know", "called": "call",
    "named": "name", "names": "name",
    "used": "use", "uses": "use", "using": "use",
    "speaks": "speak", "spoke": "speak", "spoken": "speak",
    "based": "base", "bases": "base",
    "set": "set", "sets": "set",
    "took": "take", "takes": "take", "taken": "take",
    "made": "make", "makes": "make", "making": "make",
    "built": "build", "builds": "build", "building": "build",
    "came": "come", "comes": "come", "coming": "come",
    "went": "go", "goes": "go", "going": "go",
    "worked": "work", "works": "work", "working": "work",
    "lived": "live", "lives": "live", "living": "live",
    "raised": "raise", "raises": "raise", "raising": "raise",
    "played": "play",  # duplicate — keep for coverage
}

# WH-word → answer type
_WH_TYPE: dict = {
    "who":   "person",
    "whom":  "person",
    "whose": "person",
    "where": "place",
    "when":  "time",
    "how many": "quantity",
    "how much": "quantity",
    "how long": "duration",
    "how old":  "age",
    "what":  "thing",
    "which": "thing",
    "why":   "reason",
    "how":   "method",
}


@dataclass
class DecomposedQuestion:
    """Structured representation of a parsed natural-language question."""

    wh_word: str = ""
    """The interrogative word detected: 'who', 'what', 'where', 'when', etc."""

    answer_type: str = "thing"
    """Expected answer type: 'person' | 'place' | 'time' | 'quantity' | 'thing' | 'method'"""

    relation_keywords: List[str] = field(default_factory=list)
    """Content words after stopword removal and lemmatization — used for relation scoring."""

    has_temporal_constraint: bool = False
    """True when the question contains a temporal ordering constraint ('before X', 'first', etc.)"""

    is_comparative: bool = False
    """True for superlative/comparative questions ('largest', 'most', 'first ever', etc.)"""


class QuestionDecomposer:
    """
    Training-free question decomposer.

    Parses a natural-language question into answer type, relation keywords,
    and structural constraints — without any external NLP dependencies.

    Usage:
        qd = QuestionDecomposer()
        result = qd.decompose("who plays ken barlow in coronation street")
        # DecomposedQuestion(wh_word='who', answer_type='person',
        #     relation_keywords=['play', 'ken', 'barlow', 'coronation', 'street'])
    """

    _COMPARATIVE_MARKERS: frozenset = frozenset({
        "most", "least", "largest", "smallest", "highest", "lowest",
        "first", "last", "oldest", "newest", "youngest", "biggest",
        "best", "worst", "longest", "shortest", "earliest", "latest",
    })

    def decompose(self, question: str) -> DecomposedQuestion:
        """Parse a natural-language question into a DecomposedQuestion."""
        q = question.strip().lower()
        # Normalise punctuation
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        tokens = q.split()

        wh_word, answer_type = self._detect_wh(q, tokens)
        keywords = self._extract_keywords(tokens)
        has_temporal = bool(_TEMPORAL_MARKERS & set(tokens))
        is_comparative = bool(self._COMPARATIVE_MARKERS & set(tokens))

        return DecomposedQuestion(
            wh_word=wh_word,
            answer_type=answer_type,
            relation_keywords=keywords,
            has_temporal_constraint=has_temporal,
            is_comparative=is_comparative,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_wh(self, q: str, tokens: list) -> tuple[str, str]:
        """Return (wh_word, answer_type) by scanning for interrogative patterns."""
        # Multi-word patterns first (order matters: "how many" before "how")
        for phrase in ("how many", "how much", "how long", "how old"):
            if phrase in q:
                return phrase, _WH_TYPE[phrase]

        # Single-token WH words
        for tok in tokens[:3]:  # WH-word is almost always in the first 3 tokens
            if tok in _WH_TYPE:
                return tok, _WH_TYPE[tok]

        # Implicit WH — questions starting with "name", "list", "give" are thing-typed
        if tokens and tokens[0] in ("name", "list", "give", "tell", "find"):
            return "", "thing"

        return "", "thing"

    def _extract_keywords(self, tokens: list) -> list:
        """Remove stopwords and WH-words; lemmatize remaining tokens."""
        result = []
        for tok in tokens:
            if tok in _STOPWORDS:
                continue
            if tok in _WH_TYPE:
                continue
            # Lemmatize if a known inflection
            lemma = _LEMMA.get(tok, tok)
            # Drop single-character tokens (punctuation artefacts)
            if len(lemma) < 2:
                continue
            result.append(lemma)
        return result

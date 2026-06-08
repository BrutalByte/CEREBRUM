"""
Phase 232: Training-free relation-name scoring for goal-directed beam steering.

Tokenizes Freebase (or any dotted/underscore-separated) relation names and
scores them against a set of question keywords using token overlap + verb bonus.

The resulting scores feed directly into graph.query(terminal_relation_boost=...)
to steer the beam toward relations semantically relevant to the question — without
any learned embeddings or training data.

Example:
    idx = RelationNameIndex()
    idx.build_from_relations(["film.film.starring", "tv.regular_cast.actor",
                               "music.artist.genre"])
    idx.score_relations(["play", "actor"])
    # {"film.film.starring": 0.67, "tv.regular_cast.actor": 0.80, ...}
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set


# Tokens that appear in virtually every Freebase relation name and add no
# discriminative signal.  Dropping them prevents "common" from matching
# every common.* relation for unrelated questions.
_RELATION_STOPWORDS: frozenset = frozenset({
    "common", "topic", "type", "object", "freebase", "kg", "base",
    "user", "key", "id", "guid", "mid", "name", "description",
})

# Action verbs that are strong signals when they appear in a relation token.
# When a question keyword matches one of these in a relation name, the score
# receives a bonus multiplier to reflect that verb alignment is highly
# discriminative (e.g., "play" in question + "starring" in relation).
_VERB_SYNONYMS: Dict[str, Set[str]] = {
    "play":    {"starring", "cast", "actor", "role", "perform"},
    "act":     {"starring", "cast", "actor", "role"},
    "star":    {"starring", "cast", "actor"},
    "direct":  {"director", "directed"},
    "write":   {"author", "writer", "written", "composer", "lyricist"},
    "author":  {"author", "writer", "written"},
    "produce": {"producer", "produced"},
    "compose": {"composer", "composition"},
    "sing":    {"singer", "vocalist", "artist"},
    "found":   {"founder", "founders", "founded"},
    "invent":  {"inventor", "invented"},
    "discover": {"discoverer"},
    "win":     {"award", "winner", "won"},
    "marry":   {"spouse", "married"},
    "born":    {"birth", "birthplace", "birthdate"},
    "die":     {"death", "died", "deceased"},
    "serve":   {"office", "position", "served"},
    "own":     {"owner", "ownership"},
    "create":  {"creator", "created"},
    "release": {"released", "release_date"},
    "record":  {"recording", "recorded"},
    "publish": {"publisher", "published"},
    "attend":  {"education", "school", "university", "alumni"},
    "graduate": {"education", "alumni", "degree"},
    "speak":   {"language", "spoken"},
    "locate":  {"location", "place", "city", "country", "region"},
    "base":    {"location", "headquarters", "place"},
    "work":    {"employer", "employment", "job", "occupation", "profession"},
    "live":    {"residence", "location", "home"},
}


class RelationNameIndex:
    """
    Inverted index over tokenized relation names for question-driven scoring.

    Build once per graph (from its relation set), then call score_relations()
    per question to get a boost map for terminal_relation_boost.
    """

    def __init__(self) -> None:
        self._rel_tokens: Dict[str, List[str]] = {}  # relation → tokens
        self._token_index: Dict[str, List[str]] = defaultdict(list)  # token → relations

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_from_relations(self, relations: Iterable[str]) -> None:
        """
        Index all relation names.

        Freebase names like "film.film.starring" are tokenized by splitting on
        dots and underscores; namespace prefix tokens (first dotted component)
        are retained for context but weighted down at scoring time.
        """
        self._rel_tokens.clear()
        self._token_index.clear()

        for rel in relations:
            tokens = self._tokenize(rel)
            if not tokens:
                continue
            self._rel_tokens[rel] = tokens
            for tok in set(tokens):
                self._token_index[tok].append(rel)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_relations(
        self,
        question_tokens: List[str],
        min_score: float = 0.0,
    ) -> Dict[str, float]:
        """
        Score all indexed relations against question_tokens.

        Scoring formula:
          base  = |shared_tokens| / |rel_tokens|        (precision over relation)
          bonus = 1 + 0.5 per verb-synonym match        (action-verb alignment)
          final = base * bonus, clamped to [0, 1]

        Only relations with final > min_score are returned.
        Relations never in the token index score 0.
        """
        if not question_tokens or not self._rel_tokens:
            return {}

        qtok_set = set(question_tokens)
        candidate_rels: Set[str] = set()
        for tok in qtok_set:
            if tok in self._token_index:
                candidate_rels.update(self._token_index[tok])
            # Also check verb synonyms
            if tok in _VERB_SYNONYMS:
                for syn in _VERB_SYNONYMS[tok]:
                    if syn in self._token_index:
                        candidate_rels.update(self._token_index[syn])

        scores: Dict[str, float] = {}
        for rel in candidate_rels:
            rtoks = self._rel_tokens[rel]
            rtok_set = set(rtoks)

            shared = qtok_set & rtok_set
            if not shared:
                # Check verb synonyms
                syn_hits = 0
                for qtok in qtok_set:
                    if qtok in _VERB_SYNONYMS:
                        if rtok_set & _VERB_SYNONYMS[qtok]:
                            syn_hits += 1
                if not syn_hits:
                    continue
                base = 0.0
            else:
                # Precision: how many of the relation's tokens are in the question?
                base = len(shared) / len(rtoks)

            # Verb synonym bonus
            verb_bonus = 0.0
            for qtok in qtok_set:
                if qtok in _VERB_SYNONYMS:
                    if rtok_set & _VERB_SYNONYMS[qtok]:
                        verb_bonus += 0.5

            score = min(1.0, base + verb_bonus * (base if base > 0 else 0.3))
            if score > min_score:
                scores[rel] = score

        return scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(relation_name: str) -> List[str]:
        """
        Tokenize a relation name into discriminative lowercase tokens.

        "film.film.starring"       → ["film", "starring"]
        "tv.regular_cast.actor"    → ["tv", "regular", "cast", "actor"]
        "common.topic.notable_for" → ["notable", "for"]  (common/topic dropped)
        """
        # Split on dots and underscores
        raw = re.split(r"[._]", relation_name.lower())
        tokens = []
        for tok in raw:
            tok = tok.strip()
            if not tok or len(tok) < 2:
                continue
            if tok in _RELATION_STOPWORDS:
                continue
            tokens.append(tok)
        return tokens

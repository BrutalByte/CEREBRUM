from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from core.graph_adapter import GraphAdapter

# Freebase relation-suffix → broad answer category.
# The last dot-segment of a Freebase relation name encodes the property type.
# These sets let us classify communities by what entity types they connect TO
# without requiring explicit /type/object/type triples.
_PERSON_SUFFIXES: FrozenSet[str] = frozenset({
    "actor", "player", "founder", "director", "creator", "participant",
    "performer", "artist", "athlete", "politician", "author", "composer",
    "character", "cast_member", "award_winner", "award_nominee",
    "gender", "nationality", "profession", "spouse_s", "parents",
    "sibling_s", "children", "person", "celebrity", "musician",
})
_PLACE_SUFFIXES: FrozenSet[str] = frozenset({
    "location", "place", "country", "city", "state", "region",
    "continent", "neighborhood", "containedby", "place_of_birth",
    "place_of_death", "place_lived", "headquarters", "capital",
    "administrative_area", "location_of_ceremony",
})
_TIME_SUFFIXES: FrozenSet[str] = frozenset({
    "date", "year", "start_date", "end_date", "from", "to",
    "first_aired", "last_aired", "date_of_birth", "date_of_death",
    "release_date", "inception", "dissolution", "ceremony_date",
})

_SUFFIX_TO_TYPE: Dict[str, str] = {}
for _s in _PERSON_SUFFIXES:
    _SUFFIX_TO_TYPE[_s] = "person"
for _s in _PLACE_SUFFIXES:
    _SUFFIX_TO_TYPE[_s] = "place"
for _s in _TIME_SUFFIXES:
    _SUFFIX_TO_TYPE[_s] = "time"


def _rel_suffix(rel: str) -> str:
    """Return the last dot-segment of a relation name."""
    return rel.rsplit(".", 1)[-1] if "." in rel else rel


class CommunityHypothesisGenerator:
    """
    Generates beam relation hypotheses from community structure.

    Catalogs which relation types serve as inter-community bridges by scanning
    the full edge set once at build time.  At query time, given a seed entity's
    community and optionally the expected answer type, returns relation boosts
    that steer the beam toward paths leading to the right community type.

    Two modes:
      - Untyped: boost all community-crossing relations (generate_hop_boosts)
      - Typed  : boost only relations bridging toward answer-type communities
                 (generate_typed_boosts) — combines community topology with
                 question decomposer answer_type for better precision

    Novel property: purely topology-derived, requires no relation-name text,
    language-agnostic, and transfers to any KG with community assignments.
    """

    def __init__(self) -> None:
        self._community_map: Dict[str, int] = {}
        # (src_cid, dst_cid) -> Counter of relation types
        self._bridge_index: Dict[Tuple[int, int], Counter] = {}
        # src_cid -> Counter of all outbound bridge-crossing relation types
        self._outbound_index: Dict[int, Counter] = {}
        # cid -> frozenset of answer-type tags this community connects TO
        # e.g., {99: frozenset({"person", "place"})} means community 99 has
        # outbound bridges that lead toward person and place entities
        self._community_reach_types: Dict[int, FrozenSet[str]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, adapter: "GraphAdapter") -> "CommunityHypothesisGenerator":
        """Scan adapter edges once; build bridge, outbound, and type-reach indexes."""
        self._community_map = getattr(adapter, "community_map", {})
        if not self._community_map:
            return self

        G = adapter.to_networkx()
        bridge_index: Dict[Tuple[int, int], Counter] = {}
        outbound: Dict[int, Counter] = {}

        for u, v, data in G.edges(data=True):
            cu = self._community_map.get(u, -1)
            cv = self._community_map.get(v, -1)
            if cu < 0 or cv < 0 or cu == cv:
                continue

            rel: str = (
                data.get("relation_type")
                or data.get("relation")
                or data.get("label")
                or ""
            )
            if not rel:
                continue

            key = (cu, cv)
            if key not in bridge_index:
                bridge_index[key] = Counter()
            bridge_index[key][rel] += 1

            if cu not in outbound:
                outbound[cu] = Counter()
            outbound[cu][rel] += 1

        self._bridge_index = bridge_index
        self._outbound_index = outbound
        self._community_reach_types = self._build_reach_types()
        return self

    def _build_reach_types(self) -> Dict[int, FrozenSet[str]]:
        """
        For each community, determine which answer-type categories it directly
        connects to via its outbound bridge relation suffixes.

        Uses the LAST dot-segment of each outbound bridge relation as a type
        proxy (e.g., "people.person.actor" → suffix "actor" → "person"), which
        works for Freebase-style dotted names without needing /type/object/type.
        A community with outbound bridges `people.person.actor` and
        `location.location.containedby` would have reach_types = {"person", "place"}.
        """
        reach: Dict[int, Set[str]] = {}
        for cid, counter in self._outbound_index.items():
            types: Set[str] = set()
            for rel in counter:
                t = _SUFFIX_TO_TYPE.get(_rel_suffix(rel))
                if t:
                    types.add(t)
            if types:
                reach[cid] = frozenset(types)
        return {cid: frozenset(types) for cid, types in reach.items()}

    # ------------------------------------------------------------------
    # Query-time
    # ------------------------------------------------------------------

    def generate_hop_boosts(
        self,
        seed_entity: str,
        top_n: int = 20,
        boost_scale: float = 2.0,
    ) -> Dict[str, float]:
        """
        Return beam boost dict based on seed entity's community outbound bridges.

        Returns top_n bridge relations with boosts in [1.0, 1.0+boost_scale].
        Non-listed relations unaffected (default=1.0 for penultimate_relation_boost).
        Returns {} when entity is unknown or community has no outbound bridges.
        """
        src_cid = self._community_map.get(seed_entity, -1)
        if src_cid < 0:
            return {}

        outbound = self._outbound_index.get(src_cid)
        if not outbound:
            return {}

        top_rels = outbound.most_common(top_n)
        if not top_rels:
            return {}

        max_count = top_rels[0][1]
        return {rel: 1.0 + boost_scale * (count / max_count) for rel, count in top_rels}

    def generate_typed_boosts(
        self,
        seed_entity: str,
        answer_type: str = "",
        top_n: int = 20,
        boost_scale: float = 2.0,
    ) -> Dict[str, float]:
        """
        Answer-type-aware community hypothesis.

        Filters the seed entity's community outbound bridge relations to only
        those whose relation suffix matches the expected answer type. For example,
        for "who" questions (answer_type="person"), only keeps relations with
        person-type suffixes (actor, player, director, founder, …). This avoids
        boosting award/film relations when the question asks for a person.

        Falls back to generate_hop_boosts() when answer_type is unrecognized or
        when typed filtering yields no candidates (all bridge rels have non-matching
        suffixes — a common case for 2-hop paths where hop-1 goes through CVTs).

        This is the primary hypothesis function for typed KGQA — it combines the
        community topology (frequency of relation types in graph crossings) with
        the question type signal from QuestionDecomposer.
        """
        if not answer_type:
            return self.generate_hop_boosts(seed_entity, top_n, boost_scale)

        src_cid = self._community_map.get(seed_entity, -1)
        if src_cid < 0:
            return {}

        outbound = self._outbound_index.get(src_cid)
        if not outbound:
            return {}

        # Keep only bridge relations whose suffix matches the expected answer type.
        # The relation suffix (last dot-segment) is a Freebase-specific type proxy:
        # "people.person.actor" → "actor" → person; "location.location.containedby" → place.
        typed_counter: Counter = Counter({
            rel: count for rel, count in outbound.items()
            if _SUFFIX_TO_TYPE.get(_rel_suffix(rel)) == answer_type
        })

        if not typed_counter:
            # No typed match — common for 2-hop paths via CVTs; use unfiltered boosts
            return self.generate_hop_boosts(seed_entity, top_n, boost_scale)

        top_rels = typed_counter.most_common(top_n)
        max_count = top_rels[0][1]
        return {rel: 1.0 + boost_scale * (count / max_count) for rel, count in top_rels}

    def top_bridge_relations(
        self,
        src_cid: int,
        dst_cid: int,
        top_n: int = 20,
    ) -> List[Tuple[str, int]]:
        """Return (relation, count) pairs for the src_cid→dst_cid bridge."""
        return self._bridge_index.get((src_cid, dst_cid), Counter()).most_common(top_n)

    def community_of(self, entity_id: str) -> int:
        """Return community id for entity, or -1 if unknown."""
        return self._community_map.get(entity_id, -1)

    def adjacent_community_count(self, src_cid: int) -> int:
        """Number of distinct communities directly reachable from src_cid."""
        return sum(1 for (c1, _) in self._bridge_index if c1 == src_cid)

    def community_reach_types(self, src_cid: int) -> FrozenSet[str]:
        """Answer types reachable from this community via outbound bridges."""
        return self._community_reach_types.get(src_cid, frozenset())

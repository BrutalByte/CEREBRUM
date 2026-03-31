"""
ConversationManager — CEREBRUM multi-turn dialogue layer (Phase 20).

Wraps the QueryParser → BeamTraversal → PathVerbalizer pipeline with
session memory so that:

  - Pronouns ("he", "it", "there") resolve to previously mentioned entities
  - Follow-up questions ("what else?", "and?") continue from the current focus
  - Topic shifts ("now tell me about leibniz") update the focus correctly
  - Answers already given are not repeated within a session
  - Ambiguous entity matches surface clarification options
  - Knowledge gaps are reported explicitly rather than returning silence
  - Graph summaries ("tell me everything about X") are supported

Architecture
------------
ConversationSession   — immutable-style state dataclass; updated after each turn
ConversationTurn      — one complete Q→A exchange with metadata
ConversationManager   — coordinates all resolution + traversal + verbalization

Usage
-----
    from core.conversation import ConversationManager, new_session

    manager = ConversationManager(adapter, engine, csa, traversal)
    session = manager.new_session()

    turn = manager.process("What did newton influence?", session)
    print(turn.answer_text)
    # "Q: What did newton influence?\\n1. newton influenced faraday ..."

    turn = manager.process("What did he discover?", session)
    # "he" resolves to "newton" — continues seamlessly

    turn = manager.process("What else?", session)
    # Follow-up: excludes already-returned answers, advances to next results
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Pronoun resolution tables
# ---------------------------------------------------------------------------

# pronoun stem → entity question-type it refers to
_PRONOUN_TYPE: Dict[str, str] = {
    "he":    "person",
    "him":   "person",
    "his":   "person",
    "she":   "person",
    "her":   "person",
    "hers":  "person",
    "they":  "any",
    "them":  "any",
    "their": "any",
    "it":    "thing",
    "this":  "thing",
    "that":  "thing",
    "there": "place",
}

# Phrases that indicate a follow-up question (no new topic)
_FOLLOWUP_PREFIXES = (
    "and ", "also ", "what else", "who else", "what other",
    "tell me more", "anything else", "more about", "what about that",
    "go on", "continue", "and what", "and who", "and where",
    "and how", "and why", "expand on", "elaborate",
)

# Explicit topic-shift markers
_SHIFT_MARKERS = (
    "now tell me about", "let's talk about", "switch to",
    "what about", "how about", "tell me about",
)

# Entity type → pronouns it absorbs
_TYPE_PRONOUNS: Dict[str, List[str]] = {
    "person": ["he", "him", "his", "she", "her", "hers", "they", "them", "their"],
    "place":  ["there", "it", "this", "that"],
    "thing":  ["it", "this", "that"],
    "time":   ["it", "this", "that"],
    "reason": ["it", "this", "that"],
    "method": ["it", "this", "that"],
    "any":    ["they", "them", "their"],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """One complete question-answer exchange."""

    turn_number: int
    raw_question: str
    resolved_question: str
    """Question after pronoun substitution — what was actually queried."""

    seed_entity: Optional[str]
    """Graph entity the traversal started from."""

    seed_entity_label: str
    answer_text: str
    """Verbalized response ready for display."""

    new_entities: List[str]
    """Entity IDs surfaced for the first time this turn."""

    is_followup: bool
    """True if this continued from the previous focus without a topic shift."""

    focus_shift: bool
    """True if the topic changed to a new entity."""

    clarification_needed: bool
    """True if the seed entity was ambiguous and multiple candidates scored closely."""

    clarification_options: List[Tuple[str, str]]
    """[(entity_id, label), ...] when clarification is needed."""

    knowledge_gap: bool
    """True if no graph paths were found at all."""

    knowledge_gap_hint: str
    """Explanation of what information is missing when knowledge_gap=True."""

    hop_hint: int
    """Traversal depth that was used."""


@dataclass
class ConversationSession:
    """
    Mutable state for one conversation session.

    Updated in-place by ConversationManager after each turn.
    """

    session_id: str
    focus_entity: Optional[str]
    """The entity currently 'in focus' — target for pronoun resolution."""

    focus_entity_label: str
    focus_entity_type: str
    """question_type of the focus entity: 'person', 'place', 'thing', etc."""

    entity_history: List[str]
    """All entity IDs mentioned so far, in order of first appearance."""

    answer_history: Set[str]
    """Entity IDs already returned as answers — excluded from future turns."""

    pronoun_map: Dict[str, str]
    """Live pronoun → entity_id map: {"he": "newton", "it": "calculus"}."""

    entity_label_map: Dict[str, str]
    """entity_id → label for all entities seen this session."""

    turns: List[ConversationTurn]
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def reset(self) -> None:
        """Clear context while preserving session identity."""
        self.focus_entity       = None
        self.focus_entity_label = ""
        self.focus_entity_type  = "thing"
        self.entity_history.clear()
        self.answer_history.clear()
        self.pronoun_map.clear()
        self.entity_label_map.clear()
        self.turns.clear()
        self.last_active = time.time()


def new_session(session_id: Optional[str] = None) -> ConversationSession:
    """Create a fresh ConversationSession."""
    return ConversationSession(
        session_id=session_id or str(uuid.uuid4()),
        focus_entity=None,
        focus_entity_label="",
        focus_entity_type="thing",
        entity_history=[],
        answer_history=set(),
        pronoun_map={},
        entity_label_map={},
        turns=[],
    )


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------

class ConversationManager:
    """
    Multi-turn dialogue coordinator for CEREBRUM.

    Parameters
    ----------
    adapter : GraphAdapter
    embedding_engine : EmbeddingEngine
    csa_engine : CSAEngine
    beam_traversal : BeamTraversal
    top_k : int
        Max answers per turn.
    clarification_gap : float
        If the top-2 entity candidates are within this score gap, ask for
        clarification rather than guessing.
    """

    def __init__(
        self,
        adapter,
        embedding_engine,
        csa_engine,
        beam_traversal,
        top_k: int = 5,
        clarification_gap: float = 0.08,
    ) -> None:
        from core.query_parser import QueryParser
        from core.verbalizer import PathVerbalizer

        self._adapter      = adapter
        self._engine       = embedding_engine
        self._csa          = csa_engine
        self._traversal    = beam_traversal
        self._top_k        = top_k
        self._clarify_gap  = clarification_gap
        self._parser       = QueryParser(adapter, embedding_engine)
        self._verbalizer   = PathVerbalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Create a fresh session managed by this ConversationManager."""
        return new_session(session_id)

    def process(
        self,
        question: str,
        session: ConversationSession,
    ) -> ConversationTurn:
        """
        Process one conversation turn.

        Resolves pronouns, detects follow-ups, runs graph traversal,
        verbalizes answers, and updates session state.

        Parameters
        ----------
        question : str
            Raw user input.
        session : ConversationSession
            Current session state — mutated in place.

        Returns
        -------
        ConversationTurn
        """
        session.last_active = time.time()
        raw = question.strip()
        q_lower = raw.lower()

        # ── 1. Resolve pronouns ──────────────────────────────────────
        resolved = self._resolve_pronouns(raw, session)

        # ── 2. Parse resolved question ───────────────────────────────
        parsed = self._parser.parse(resolved)

        # ── 3. Detect follow-up vs topic shift ───────────────────────
        is_followup  = self._is_followup(q_lower, session)
        focus_shift  = False

        # Determine effective seed entity
        seed_id: Optional[str] = None
        seed_label: str = ""
        seed_score: float = 0.0

        if is_followup and session.focus_entity:
            seed_id    = session.focus_entity
            seed_label = session.focus_entity_label
            seed_score = 1.0
        elif parsed.seed_entity_id is not None:
            seed_id    = parsed.seed_entity_id
            seed_label = parsed.seed_entity_label or str(seed_id)
            seed_score = parsed.seed_entity_score
            if session.focus_entity and seed_id != session.focus_entity:
                focus_shift = True
        else:
            # No entity found — try to use focus as fallback
            seed_id    = session.focus_entity
            seed_label = session.focus_entity_label
            seed_score = 0.0

        # ── 4. Clarification check ───────────────────────────────────
        clarification_needed  = False
        clarification_options: List[Tuple[str, str]] = []

        if (
            not is_followup
            and parsed.candidates
            and len(parsed.candidates) >= 2
            and seed_score < 0.95
        ):
            top_score = parsed.candidates[0][2]
            second_score = parsed.candidates[1][2]
            if top_score - second_score < self._clarify_gap:
                clarification_needed = True
                clarification_options = [(c[0], c[1]) for c in parsed.candidates[:3]]

        # ── 5. Handle no seed ───────────────────────────────────────
        if seed_id is None:
            turn = ConversationTurn(
                turn_number=session.turn_count + 1,
                raw_question=raw,
                resolved_question=resolved,
                seed_entity=None,
                seed_entity_label="",
                answer_text=self._no_entity_response(raw, session),
                new_entities=[],
                is_followup=is_followup,
                focus_shift=False,
                clarification_needed=clarification_needed,
                clarification_options=clarification_options,
                knowledge_gap=True,
                knowledge_gap_hint="No matching entity found in the graph.",
                hop_hint=parsed.hop_hint,
            )
            session.turns.append(turn)
            return turn

        # ── 6. Traverse ──────────────────────────────────────────────
        from reasoning.answer_extractor import extract

        paths   = self._traversal.traverse([seed_id])
        answers = extract(paths, top_k=self._top_k * 3)  # fetch extra to allow dedup

        # ── 7. Deduplicate against answer history ────────────────────
        if is_followup and not focus_shift:
            fresh = [a for a in answers if a.entity_id not in session.answer_history]
        else:
            fresh = answers  # topic shift: start fresh

        # Exclude seed itself from answers
        fresh = [a for a in fresh if a.entity_id != seed_id]
        fresh = fresh[:self._top_k]

        # ── 8. Knowledge gap detection ───────────────────────────────
        knowledge_gap      = len(fresh) == 0
        knowledge_gap_hint = ""
        if knowledge_gap:
            knowledge_gap_hint = self._gap_hint(seed_id, seed_label, session, is_followup)

        # ── 9. Verbalize ─────────────────────────────────────────────
        if clarification_needed:
            answer_text = self._clarification_prompt(raw, clarification_options)
        elif knowledge_gap:
            answer_text = f"I don't have any {'more ' if is_followup else ''}information" \
                          f" about {seed_label!r} that I haven't already shared.\n\n" \
                          f"{knowledge_gap_hint}"
        else:
            answer_text = self._verbalizer.verbalize_answers(
                fresh,
                adapter=self._adapter,
                question=resolved,
                top_k=self._top_k,
            )
            if is_followup and session.focus_entity:
                answer_text = f"[Continuing from {seed_label!r}]\n\n" + answer_text

        # ── 10. Update session state ─────────────────────────────────
        new_entity_ids = [a.entity_id for a in fresh]

        # Track entity history
        if seed_id not in session.entity_history:
            session.entity_history.append(seed_id)
        for eid in new_entity_ids:
            if eid not in session.entity_history:
                session.entity_history.append(eid)

        # Update answer history
        session.answer_history.update(new_entity_ids)

        # Update labels
        session.entity_label_map[seed_id] = seed_label
        for a in fresh:
            lbl = a.entity_id
            try:
                ent = self._adapter.get_entity(a.entity_id)
                if ent and ent.label:
                    lbl = ent.label
            except Exception:
                pass
            session.entity_label_map[a.entity_id] = lbl

        # Update focus
        if not is_followup or focus_shift:
            session.focus_entity       = seed_id
            session.focus_entity_label = seed_label
            session.focus_entity_type  = parsed.question_type if not is_followup else session.focus_entity_type
            self._update_pronoun_map(session, seed_id, seed_label, parsed.question_type)

        turn = ConversationTurn(
            turn_number=session.turn_count + 1,
            raw_question=raw,
            resolved_question=resolved,
            seed_entity=seed_id,
            seed_entity_label=seed_label,
            answer_text=answer_text,
            new_entities=new_entity_ids,
            is_followup=is_followup,
            focus_shift=focus_shift,
            clarification_needed=clarification_needed,
            clarification_options=clarification_options,
            knowledge_gap=knowledge_gap,
            knowledge_gap_hint=knowledge_gap_hint,
            hop_hint=parsed.hop_hint,
        )
        session.turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # Pronoun resolution
    # ------------------------------------------------------------------

    def _resolve_pronouns(self, question: str, session: ConversationSession) -> str:
        """
        Substitute pronouns in *question* with their resolved entity labels.

        Only substitutes when the pronoun map has a confident mapping.
        Leaves unresolvable pronouns in place (QueryParser will handle them).
        """
        if not session.pronoun_map:
            return question

        words  = question.split()
        result = []
        for word in words:
            clean = word.lower().rstrip("?,.'\"!;:")
            if clean in session.pronoun_map:
                eid   = session.pronoun_map[clean]
                label = session.entity_label_map.get(eid, eid)
                # Preserve trailing punctuation
                suffix = word[len(clean):]
                result.append(label + suffix)
            else:
                result.append(word)
        return " ".join(result)

    def _update_pronoun_map(
        self,
        session: ConversationSession,
        entity_id: str,
        label: str,
        question_type: str,
    ) -> None:
        """Map appropriate pronouns to this entity based on its type."""
        pronouns = _TYPE_PRONOUNS.get(question_type, _TYPE_PRONOUNS["thing"])
        for p in pronouns:
            session.pronoun_map[p] = entity_id

    # ------------------------------------------------------------------
    # Follow-up and topic shift detection
    # ------------------------------------------------------------------

    def _is_followup(self, q_lower: str, session: ConversationSession) -> bool:
        """
        Return True if the question is a follow-up continuing the current focus.

        Heuristics (in priority order):
        1. Starts with a follow-up prefix ("and ", "what else", etc.)
        2. Is very short (≤ 4 words) and contains a pronoun we can resolve
        3. Contains "what about" + entity already in entity_history
        """
        if session.focus_entity is None:
            return False

        # Normalize: remove all punctuation so "also, what" matches "also "
        normalized = re.sub(r"[^\w\s]", " ", q_lower).strip()

        # Explicit follow-up prefixes
        for prefix in _FOLLOWUP_PREFIXES:
            if normalized.startswith(prefix) or q_lower.startswith(prefix):
                return True

        # Short question with a resolvable pronoun
        words = q_lower.split()
        if len(words) <= 5:
            for w in words:
                clean = w.rstrip("?,.'\"!;:")
                if clean in session.pronoun_map:
                    return True

        return False

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _no_entity_response(self, question: str, session: ConversationSession) -> str:
        lines = [f"I couldn't identify a graph entity in: {question!r}"]
        if session.focus_entity:
            lines.append(
                f"\nIf you meant to continue asking about "
                f"{session.focus_entity_label!r}, try rephrasing with the entity name."
            )
        if session.entity_history:
            labels = [session.entity_label_map.get(e, e)
                      for e in session.entity_history[-5:]]
            lines.append(f"\nEntities I know about in this session: {', '.join(labels)}")
        return "\n".join(lines)

    def _gap_hint(
        self,
        entity_id: str,
        label: str,
        session: ConversationSession,
        is_followup: bool,
    ) -> str:
        """Generate a helpful knowledge-gap explanation."""
        G = self._adapter.to_networkx()
        if entity_id not in G:
            return f"Entity {label!r} exists in the graph but has no connections."

        degree = G.degree(entity_id)
        if degree == 0:
            return f"{label!r} is an isolated node with no edges in the graph."

        if is_followup and session.answer_history:
            seen_count = len(session.answer_history)
            return (
                f"I've already shared {seen_count} result(s) about {label!r}. "
                f"The graph has {degree} connection(s) total from this entity."
            )

        return (
            f"{label!r} has {degree} connection(s) in the graph, but none matched "
            f"the current traversal parameters. Try a broader query or increase hops."
        )

    def _clarification_prompt(
        self,
        question: str,
        options: List[Tuple[str, str]],
    ) -> str:
        lines = [
            "I found multiple possible matches for your question. "
            "Which did you mean?",
            "",
        ]
        for i, (eid, label) in enumerate(options, 1):
            lines.append(f"  {i}. {label}  (id: {eid})")
        lines.append("")
        lines.append("Please rephrase your question using the full name.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Session-STDP: potentiate edges from high-engagement sessions
    # ------------------------------------------------------------------

    def potentiate_session(
        self,
        session: ConversationSession,
        delta: float = 0.05,
        min_turns: int = 3,
        min_followup_rate: float = 0.4,
    ) -> int:
        """
        Apply STDP-style weight increments to graph edges implied by the
        session's answer trail.

        Engagement signal is computed from:
          - ``turn_count >= min_turns``
          - Follow-up rate (turns where is_followup=True) >= min_followup_rate

        When both thresholds are met, edges between consecutive entities in
        ``session.entity_history`` are potentiated by ``delta * engagement``.

        No LLM dependency.  The graph's own edge weights are the memory.

        Parameters
        ----------
        delta : float
            Maximum weight increment per potentiated edge.
        min_turns : int
            Minimum session length to trigger potentiation.
        min_followup_rate : float
            Minimum fraction of follow-up turns to trigger potentiation.

        Returns
        -------
        int
            Number of edges potentiated.
        """
        if session.turn_count < min_turns:
            return 0

        n_followups = sum(1 for t in session.turns if t.is_followup)
        followup_rate = n_followups / session.turn_count
        if followup_rate < min_followup_rate:
            return 0

        engagement = followup_rate  # scalar in [min_followup_rate, 1.0]
        increment  = delta * engagement

        G = self._adapter.to_networkx()
        potentiated = 0

        # Pairs derived from seed → answer in each turn
        for turn in session.turns:
            if turn.seed_entity is None:
                continue
            for eid in turn.new_entities:
                for u, v in [(turn.seed_entity, eid), (eid, turn.seed_entity)]:
                    if G.has_edge(u, v):
                        data = G[u][v]
                        old_w = data.get("weight", 1.0)
                        data["weight"] = round(old_w + increment, 6)
                        potentiated += 1
                        break  # only potentiate one direction per pair

        return potentiated

    # ------------------------------------------------------------------
    # Session summary (for /history and debugging)
    # ------------------------------------------------------------------

    def session_summary(self, session: ConversationSession) -> str:
        """Return a human-readable session summary."""
        lines = [
            f"Session {session.session_id[:8]}...",
            f"  Turns        : {session.turn_count}",
            f"  Focus entity : {session.focus_entity_label or 'none'}",
            f"  Entities seen: {len(session.entity_history)}",
            f"  Answers given: {len(session.answer_history)}",
        ]
        if session.entity_history:
            labels = [session.entity_label_map.get(e, e) for e in session.entity_history]
            lines.append(f"  Entity trail : {' → '.join(labels)}")
        return "\n".join(lines)

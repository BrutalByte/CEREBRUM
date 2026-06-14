"""
DialogueGraph — Phase 275.

Materializes dialogue acts (utterances, intents, responses) as CEREBRUM
nodes and semantic transitions as edges inside AuraMemory. Communication
becomes graph traversal — no transformer required for the memory layer.

Each conversation turn produces:
  - A DialogueAct node: (act_id, text, intent, speaker, ts)
  - Semantic transition edges to related entities in AuraMemory or the
    main knowledge graph:

Transition types (edges):
  entails         — this utterance logically follows from another
  contradicts     — this utterance conflicts with a prior fact
  elaborates      — adds detail to a referenced topic
  requests        — initiates a retrieval or action request
  confirms        — affirms a prior fact or proposal
  references      — mentions an entity in the KG
  responds_to     — direct reply to a previous DialogueAct

The dialogue subgraph lives in AuraMemory so it inherits:
  - persistence across restarts
  - FederatedGraphRegistry federation with the engineering KB
  - BFS recall ("what did we discuss about X?")
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict

logger = logging.getLogger(__name__)

_TRANSITION_TYPES = {
    "entails", "contradicts", "elaborates", "requests",
    "confirms", "references", "responds_to",
}


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class DialogueAct:
    """A single utterance in a dialogue session."""

    act_id:    str
    text:      str
    intent:    str    # "question" | "statement" | "command" | "clarification"
    speaker:   str    # e.g. "Bryan" | "AURA" | "CEREBRUM"
    session_id: str
    ts:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    parent_act_id: Optional[str] = None   # act this responds to
    referenced_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueTurn:
    """One full exchange: utterance → optional response."""

    turn_id:    str
    session_id: str
    utterance:  DialogueAct
    response:   Optional[DialogueAct] = None


# ── DialogueGraph ─────────────────────────────────────────────────────────────

class DialogueGraph:
    """
    Records dialogue acts into AuraMemory as a subgraph.

    Transformer-free communication memory: each turn is a node,
    semantic links between turns and KG entities are edges.

    Usage
    -----
        from core.aura_memory import AuraMemory
        from core.dialogue_graph import DialogueGraph, DialogueAct

        memory = AuraMemory()
        dg = DialogueGraph(memory)

        session = dg.open_session(speaker="Bryan")
        act = dg.record_utterance(session, "What phase are we on?", intent="question")
        resp = dg.record_response(act, "We are on Phase 275.", speaker="AURA")
        dg.link_entity(act, "phase_275", transition="references")
    """

    def __init__(self, aura_memory: Any) -> None:
        self._memory = aura_memory
        self._lock   = threading.Lock()
        self._sessions: Dict[str, List[DialogueAct]] = {}

    # ── Session management ────────────────────────────────────────────────────

    def open_session(self, speaker: str = "user") -> str:
        """Start a new dialogue session. Returns session_id."""
        session_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sessions[session_id] = []
        logger.debug("DialogueGraph: opened session %s for %s.", session_id, speaker)
        return session_id

    def close_session(self, session_id: str) -> int:
        """Close a session. Returns number of acts recorded."""
        with self._lock:
            acts = self._sessions.pop(session_id, [])
        return len(acts)

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_utterance(
        self,
        session_id:           str,
        text:                 str,
        intent:               str = "statement",
        speaker:              str = "user",
        referenced_entities:  Optional[List[str]] = None,
        parent_act_id:        Optional[str] = None,
    ) -> DialogueAct:
        """
        Record a spoken/written utterance. Materializes:
          - A node for the DialogueAct in AuraMemory
          - responds_to edge to parent_act if given
          - references edges to any referenced_entities
        """
        act = DialogueAct(
            act_id               = str(uuid.uuid4()),
            text                 = text,
            intent               = intent,
            speaker              = speaker,
            session_id           = session_id,
            parent_act_id        = parent_act_id,
            referenced_entities  = referenced_entities or [],
        )
        self._materialize_act(act)
        with self._lock:
            self._sessions.setdefault(session_id, []).append(act)
        return act

    def record_response(
        self,
        utterance: DialogueAct,
        text:      str,
        speaker:   str = "AURA",
        referenced_entities: Optional[List[str]] = None,
    ) -> DialogueAct:
        """
        Record AURA's (or CEREBRUM's) response to an utterance.
        Automatically links responds_to the triggering utterance.
        """
        return self.record_utterance(
            session_id          = utterance.session_id,
            text                = text,
            intent              = "response",
            speaker             = speaker,
            referenced_entities = referenced_entities,
            parent_act_id       = utterance.act_id,
        )

    def link_entity(
        self,
        act:        DialogueAct,
        entity_id:  str,
        transition: str = "references",
    ) -> None:
        """
        Add a semantic transition edge from a DialogueAct to a KG entity.
        Enables queries like: "what dialogue acts referenced project_x?"
        """
        if transition not in _TRANSITION_TYPES:
            logger.warning("DialogueGraph: unknown transition type '%s'.", transition)
        self._memory.ingest_triple(
            subject    = act.act_id,
            relation   = transition,
            obj        = entity_id,
            confidence = 1.0,
            source     = "dialogue_graph",
        )

    # ── Traversal ─────────────────────────────────────────────────────────────

    def get_session_history(self, session_id: str) -> List[DialogueAct]:
        """Return all acts for a session in chronological order."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def recall_about(self, entity_id: str, max_hops: int = 2) -> List[str]:
        """
        Return dialogue act IDs that referenced `entity_id` (graph traversal).
        Answers: "what did we discuss about X?"
        Traverses in-edges to find who points at this entity.
        """
        G = self._memory.adapter._G
        act_ids: List[str] = []
        try:
            if entity_id not in G:
                # Case-insensitive fallback
                lower = entity_id.lower()
                matches = [n for n in G.nodes if n.lower() == lower]
                if not matches:
                    return []
                entity_id = matches[0]
            for pred, _, data in G.in_edges(entity_id, data=True):
                rel = data.get("relation", "")
                if rel in ("references", "elaborates", "confirms", "requests", "entails"):
                    act_ids.append(pred)
        except Exception:
            pass
        return act_ids

    # ── Internal ──────────────────────────────────────────────────────────────

    def _materialize_act(self, act: DialogueAct) -> None:
        """Write the DialogueAct node and its edges to AuraMemory."""
        # Node identity via the act_id
        self._memory.ingest_triple(
            subject    = act.act_id,
            relation   = "dialogue_act_of",
            obj        = act.session_id,
            confidence = 1.0,
            source     = "dialogue_graph",
        )
        # Speaker link
        self._memory.ingest_triple(
            subject    = act.speaker,
            relation   = "last_discussed",
            obj        = _slugify(act.text[:60]),
            confidence = 1.0,
            source     = "dialogue_graph",
        )
        # responds_to chain
        if act.parent_act_id:
            self._memory.ingest_triple(
                subject    = act.act_id,
                relation   = "responds_to",
                obj        = act.parent_act_id,
                confidence = 1.0,
                source     = "dialogue_graph",
            )
        # Referenced entity edges
        for entity in act.referenced_entities:
            self._memory.ingest_triple(
                subject    = act.act_id,
                relation   = "references",
                obj        = entity,
                confidence = 1.0,
                source     = "dialogue_graph",
            )


# ── Utility ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return text.lower().strip().replace(" ", "_").replace("-", "_")[:120]

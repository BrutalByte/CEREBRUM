"""
AuraMemory — Phase 275.

Personal episodic knowledge graph for the AURA personal AI.
Runs as a second CerebrumGraph alongside the engineering KB via
FederatedGraphRegistry. Stores personal facts, preferences, project
states, and conversation acts as (subject, episodic_relation, object)
triples.

All knowledge is graph-traversal accessible — no LLM required for
recall. AURA is the real-world interface; CEREBRUM is the backend.

Schema
------
Episodic relations:
  prefers           (Bryan, prefers, "dark mode")
  knows_about       (Bryan, knows_about, "phase_260")
  is_working_on     (Bryan, is_working_on, "cerebrum_phase_275")
  last_discussed    (Bryan, last_discussed, "aura_memory")
  reminded_me       (Bryan, reminded_me, "submit_arxiv_paper")
  context_of        (arxiv_paper, context_of, "cerebrum_project")
  located_in        (Bryan, located_in, "home_office")
  dislikes          (Bryan, dislikes, "morning_meetings")
  knows_person      (Bryan, knows_person, "colleague_name")

Edge attributes (in addition to GraphAdapter standard):
  ts          ISO-8601 timestamp of when the fact was recorded
  confidence  how certain we are (default 1.0 for stated facts)
  source      "user_stated" | "inferred" | "aura_observed"
  expires_at  optional ISO-8601; if set, fact is considered stale after this
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter

logger = logging.getLogger(__name__)

_EPISODIC_RELATIONS = {
    "prefers", "knows_about", "is_working_on", "last_discussed",
    "reminded_me", "context_of", "located_in", "dislikes", "knows_person",
    "owns", "completed", "plans_to", "interested_in",
}

_PERSIST_PATH = Path(__file__).parent.parent / "data" / "aura_memory.jsonl"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EpisodicFact:
    subject:    str
    relation:   str
    obj:        str
    confidence: float = 1.0
    source:     str   = "user_stated"
    expires_at: Optional[str] = None
    fact_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    ts:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "fact_id":    self.fact_id,
            "subject":    self.subject,
            "relation":   self.relation,
            "obj":        self.obj,
            "confidence": self.confidence,
            "source":     self.source,
            "expires_at": self.expires_at,
            "ts":         self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicFact":
        return cls(
            subject    = d["subject"],
            relation   = d["relation"],
            obj        = d["obj"],
            confidence = d.get("confidence", 1.0),
            source     = d.get("source", "user_stated"),
            expires_at = d.get("expires_at"),
            fact_id    = d.get("fact_id", str(uuid.uuid4())),
            ts         = d.get("ts", ""),
        )


@dataclass
class RecallResult:
    query:     str
    facts:     List[EpisodicFact]
    traversal_depth: int
    duration_ms: float


# ── AuraMemory ────────────────────────────────────────────────────────────────

class AuraMemory:
    """
    In-memory episodic knowledge graph for AURA's personal context.

    Backed by a NetworkXAdapter so it can be registered with
    FederatedGraphRegistry under domain "aura_memory".

    Thread-safe. Facts are persisted to data/aura_memory.jsonl so
    knowledge survives server restarts.
    """

    def __init__(
        self,
        persist_path: Path = _PERSIST_PATH,
        federated_registry: Optional[Any] = None,
    ) -> None:
        self._G        = nx.DiGraph()
        self._adapter  = NetworkXAdapter(self._G)
        self._lock     = threading.RLock()
        self._persist  = persist_path
        self._registry = federated_registry

        self._persist.parent.mkdir(parents=True, exist_ok=True)
        self._load()

        if self._registry is not None:
            self._register()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, fact: EpisodicFact) -> str:
        """
        Add a personal fact to the memory graph.
        Overwrites existing fact if (subject, relation, obj) already exists.
        Returns fact_id.
        """
        with self._lock:
            self._G.add_node(fact.subject, label=fact.subject, type="entity")
            self._G.add_node(fact.obj,     label=fact.obj,     type="entity")
            self._G.add_edge(
                fact.subject, fact.obj,
                relation    = fact.relation,
                relation_type = fact.relation.upper(),
                weight      = fact.confidence,
                confidence  = fact.confidence,
                provenance  = f"aura:{fact.source}",
                ts          = fact.ts,
                fact_id     = fact.fact_id,
                expires_at  = fact.expires_at or "",
            )
            self._append_persist(fact)
        logger.debug("AuraMemory: ingested (%s, %s, %s).", fact.subject, fact.relation, fact.obj)
        return fact.fact_id

    def ingest_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "user_stated",
        expires_at: Optional[str] = None,
    ) -> str:
        """Convenience wrapper — builds EpisodicFact and calls ingest()."""
        fact = EpisodicFact(
            subject    = subject,
            relation   = relation,
            obj        = obj,
            confidence = confidence,
            source     = source,
            expires_at = expires_at,
        )
        return self.ingest(fact)

    def recall(
        self,
        query: str,
        max_hops: int = 2,
        relation_filter: Optional[List[str]] = None,
    ) -> RecallResult:
        """
        Graph-BFS recall starting from `query` node.
        Returns all reachable facts within max_hops.
        No LLM required — pure graph traversal.
        """
        t0 = time.time()
        query_lower = query.lower().strip()

        with self._lock:
            # Exact match first, then case-insensitive substring
            if query in self._G:
                query_slug = query
            else:
                candidates = [n for n in self._G.nodes
                              if query_lower in n.lower() or n.lower() in query_lower]
                if not candidates:
                    return RecallResult(
                        query=query, facts=[], traversal_depth=0,
                        duration_ms=(time.time() - t0) * 1000,
                    )
                query_slug = candidates[0]

            visited: set = set()
            frontier = {query_slug}
            facts: List[EpisodicFact] = []
            depth = 0

            while frontier and depth <= max_hops:
                next_frontier: set = set()
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    for _, neighbor, data in self._G.out_edges(node, data=True):
                        rel = data.get("relation", "")
                        if relation_filter and rel not in relation_filter:
                            continue
                        if _is_expired(data.get("expires_at", "")):
                            continue
                        facts.append(EpisodicFact(
                            subject    = node,
                            relation   = rel,
                            obj        = neighbor,
                            confidence = data.get("confidence", 1.0),
                            source     = data.get("provenance", ""),
                            expires_at = data.get("expires_at") or None,
                            fact_id    = data.get("fact_id", ""),
                            ts         = data.get("ts", ""),
                        ))
                        next_frontier.add(neighbor)
                frontier = next_frontier - visited
                depth += 1

        return RecallResult(
            query        = query,
            facts        = facts,
            traversal_depth = depth,
            duration_ms  = (time.time() - t0) * 1000,
        )

    def forget(self, fact_id: str) -> bool:
        """Remove a specific fact by ID. Returns True if found and removed."""
        with self._lock:
            for u, v, k, data in list(self._G.edges(data=True, keys=True)):
                if data.get("fact_id") == fact_id:
                    self._G.remove_edge(u, v)
                    return True
        return False

    def update_last_discussed(self, topic: str, subject: str = "Bryan") -> str:
        """Convenience: record that `subject` last discussed `topic` right now."""
        return self.ingest_triple(
            subject  = subject,
            relation = "last_discussed",
            obj      = _slugify(topic),
            source   = "aura_observed",
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "nodes": self._G.number_of_nodes(),
                "edges": self._G.number_of_edges(),
                "relations": list({d.get("relation", "") for _, _, d in self._G.edges(data=True)}),
            }

    @property
    def adapter(self) -> NetworkXAdapter:
        return self._adapter

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._persist.exists():
            return
        count = 0
        with self._persist.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = EpisodicFact.from_dict(json.loads(line))
                    if not _is_expired(fact.expires_at or ""):
                        self._G.add_node(fact.subject, label=fact.subject, type="entity")
                        self._G.add_node(fact.obj,     label=fact.obj,     type="entity")
                        self._G.add_edge(
                            fact.subject, fact.obj,
                            relation      = fact.relation,
                            relation_type = fact.relation.upper(),
                            weight        = fact.confidence,
                            confidence    = fact.confidence,
                            provenance    = f"aura:{fact.source}",
                            ts            = fact.ts,
                            fact_id       = fact.fact_id,
                            expires_at    = fact.expires_at or "",
                        )
                        count += 1
                except Exception:
                    pass
        logger.info("AuraMemory: loaded %d facts from %s.", count, self._persist)

    def _append_persist(self, fact: EpisodicFact) -> None:
        try:
            with self._persist.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fact.to_dict()) + "\n")
        except Exception:
            logger.debug("AuraMemory: persist write failed.", exc_info=True)

    def _register(self) -> None:
        try:
            self._registry.graphs["aura_memory"] = self._adapter
            logger.info("AuraMemory: registered with FederatedGraphRegistry as 'aura_memory'.")
        except Exception:
            logger.debug("AuraMemory: FederatedGraphRegistry registration failed.", exc_info=True)


# ── Utility ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return text.lower().strip().replace(" ", "_").replace("-", "_")[:120]


def _is_expired(expires_at: str) -> bool:
    if not expires_at:
        return False
    try:
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return exp < datetime.now(timezone.utc)
    except Exception:
        return False

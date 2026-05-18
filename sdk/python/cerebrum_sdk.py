"""
CEREBRUM Python SDK — clean entry point for developers.

    from cerebrum_sdk import Cerebrum

    c = Cerebrum.from_csv("kb.csv")
    result = c.ask("Who directed Inception?")
    print(result.answer, result.trace_path)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class TraceStep:
    entity: str
    relation: str


@dataclass
class Result:
    answer: str
    confidence: float
    trace_path: List[TraceStep]
    top_k: List[dict]
    elapsed_ms: float
    raw: Any = field(repr=False, default=None)

    def __str__(self):
        path = " -> ".join(
            f"{s.entity}[{s.relation}]" for s in self.trace_path
        ) + (f" -> {self.answer}" if self.trace_path else self.answer)
        return f"{self.answer} (conf={self.confidence:.3f}, path: {path})"


def _extract_trace(answer) -> List[TraceStep]:
    """Build trace steps from a TraversalPath whose nodes alternate entity/relation."""
    if answer is None or answer.best_path is None:
        return []
    nodes = answer.best_path.nodes  # [e0, r0, e1, r1, e2, ...]
    steps = []
    i = 0
    while i + 1 < len(nodes):
        steps.append(TraceStep(entity=nodes[i], relation=nodes[i + 1]))
        i += 2
    return steps


class Cerebrum:
    """
    High-level interface to a CEREBRUM knowledge graph.

    Parameters
    ----------
    adapter     : loaded graph adapter (use factory methods below)
    embeddings  : "random" | "sentence"
    beam_width  : beam search width (default 10)
    max_hop     : maximum reasoning hops (default 3)
    top_k       : number of candidates to return (default 5)
    """

    def __init__(self, adapter, *, embeddings="random", beam_width=10, max_hop=3, top_k=5):
        self._adapter = adapter
        self._embeddings = embeddings
        self._beam_width = beam_width
        self._max_hop = max_hop
        self._top_k = top_k
        self._graph = None
        self._built = False

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        source_col: str = "source",
        target_col: str = "target",
        relation_col: str = "relation",
        embeddings: str = "random",
        beam_width: int = 10,
        max_hop: int = 3,
        top_k: int = 5,
    ) -> "Cerebrum":
        """Load from a CSV edge-list."""
        from adapters.csv_adapter import load_csv_adapter
        adapter = load_csv_adapter(
            str(path),
            source_col=source_col,
            target_col=target_col,
            relation_col=relation_col,
        )
        obj = cls(adapter, embeddings=embeddings, beam_width=beam_width, max_hop=max_hop, top_k=top_k)
        obj._build()
        return obj

    @classmethod
    def from_triples(
        cls,
        triples: list,
        *,
        directed: bool = True,
        embeddings: str = "random",
        beam_width: int = 10,
        max_hop: int = 3,
        top_k: int = 5,
    ) -> "Cerebrum":
        """Load from a list of (subject, relation, object) tuples."""
        from adapters.networkx_adapter import NetworkXAdapter
        adapter = NetworkXAdapter.from_triples(triples, directed=directed)
        obj = cls(adapter, embeddings=embeddings, beam_width=beam_width, max_hop=max_hop, top_k=top_k)
        obj._build()
        return obj

    @classmethod
    def from_kb(
        cls,
        path: str,
        *,
        sep: str = "|",
        embeddings: str = "random",
        beam_width: int = 10,
        max_hop: int = 3,
        top_k: int = 5,
    ) -> "Cerebrum":
        """Load from a pipe-separated KB triples file (MetaQA/Freebase format)."""
        from core.cerebrum import CerebrumGraph
        graph = CerebrumGraph.from_kb(path, sep=sep, embeddings=embeddings)
        graph.build(seed=42)
        obj = cls.__new__(cls)
        obj._adapter = graph.adapter
        obj._embeddings = embeddings
        obj._beam_width = beam_width
        obj._max_hop = max_hop
        obj._top_k = top_k
        obj._graph = graph
        obj._built = True
        return obj

    # ── Core API ─────────────────────────────────────────────────────────────

    def ask(self, question: str, *, beam_width: int = None, max_hop: int = None, top_k: int = None) -> Result:
        """
        Answer a natural-language question by traversing the knowledge graph.

        Returns a Result with .answer, .confidence, .trace_path, .top_k, .elapsed_ms.
        """
        from core.query_parser import QueryParser
        from core.embedding_engine import RandomEngine
        from reasoning.traversal import BeamTraversal
        from reasoning.answer_extractor import extract
        from core.attention_engine import CSAEngine
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs

        if not self._built:
            self._build()

        bw  = beam_width or self._beam_width
        hop = max_hop or self._max_hop
        k   = top_k or self._top_k

        t0 = time.perf_counter()

        # Community graph for CSA
        import networkx as nx
        G = self._adapter.to_networkx()
        if not hasattr(self._adapter, "community_map") or not self._adapter.community_map:
            from core.community_engine import best_of_n_dscf
            parts = best_of_n_dscf(G, n_trials=3, seed=42)
            cm = {}
            for cid, members in enumerate(parts):
                for node in members:
                    cm[node] = cid
            self._adapter.community_map = cm
        if not hasattr(self._adapter, "embeddings") or self._adapter.embeddings is None:
            engine = RandomEngine(dim=64)
            labels = {n: (self._adapter.get_entity(n).label if self._adapter.get_entity(n) else n)
                      for n in G.nodes()}
            self._adapter.embeddings = engine.encode_entities(labels)

        dist = build_community_distance_matrix(G, self._adapter.community_map)
        adj  = adjacent_community_pairs(G, self._adapter.community_map)
        csa  = CSAEngine(adapter=self._adapter)
        csa.set_community_graph(dist, adj)

        # Seed entity
        engine = RandomEngine(dim=64)
        qparser = QueryParser(self._adapter, engine)
        parsed = qparser.parse(question)

        if parsed.seed_entity_id is None:
            return Result(
                answer="",
                confidence=0.0,
                trace_path=[],
                top_k=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        traversal = BeamTraversal(adapter=self._adapter, csa_engine=csa, beam_width=bw, max_hop=hop)
        paths = traversal.traverse([parsed.seed_entity_id])
        answers = extract(paths, top_k=k)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not answers:
            return Result(answer="", confidence=0.0, trace_path=[], top_k=[], elapsed_ms=elapsed_ms)

        best = answers[0]
        trace = _extract_trace(best)

        top_k_list = [
            {"entity": a.entity_id, "confidence": round(float(a.score), 4)}
            for a in answers
        ]

        return Result(
            answer=best.entity_id,
            confidence=round(float(best.score), 4),
            trace_path=trace,
            top_k=top_k_list,
            elapsed_ms=round(elapsed_ms, 1),
            raw=answers,
        )

    def query(self, entity: str, *, beam_width: int = None, max_hop: int = None, top_k: int = None) -> Result:
        """Like ask() but takes an entity ID or label directly (no NL parsing)."""
        from reasoning.traversal import BeamTraversal
        from reasoning.answer_extractor import extract
        from core.attention_engine import CSAEngine
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
        from core.embedding_engine import RandomEngine

        if not self._built:
            self._build()

        bw  = beam_width or self._beam_width
        hop = max_hop or self._max_hop
        k   = top_k or self._top_k

        t0 = time.perf_counter()
        G = self._adapter.to_networkx()
        if not hasattr(self._adapter, "community_map") or not self._adapter.community_map:
            from core.community_engine import best_of_n_dscf
            parts = best_of_n_dscf(G, n_trials=3, seed=42)
            cm = {}
            for cid, members in enumerate(parts):
                for node in members:
                    cm[node] = cid
            self._adapter.community_map = cm
        if not hasattr(self._adapter, "embeddings") or self._adapter.embeddings is None:
            engine = RandomEngine(dim=64)
            labels = {n: (self._adapter.get_entity(n).label if self._adapter.get_entity(n) else n)
                      for n in G.nodes()}
            self._adapter.embeddings = engine.encode_entities(labels)

        dist = build_community_distance_matrix(G, self._adapter.community_map)
        adj  = adjacent_community_pairs(G, self._adapter.community_map)
        csa  = CSAEngine(adapter=self._adapter)
        csa.set_community_graph(dist, adj)

        seeds = [entity] if entity in G else [
            e.id for e in self._adapter.find_entities(entity, top_k=3) if e
        ]
        if not seeds:
            return Result(answer="", confidence=0.0, trace_path=[], top_k=[], elapsed_ms=0.0)

        traversal = BeamTraversal(adapter=self._adapter, csa_engine=csa, beam_width=bw, max_hop=hop)
        paths = traversal.traverse(seeds)
        answers = extract(paths, top_k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not answers:
            return Result(answer="", confidence=0.0, trace_path=[], top_k=[], elapsed_ms=elapsed_ms)

        best = answers[0]

        return Result(
            answer=best.entity_id,
            confidence=round(float(best.score), 4),
            trace_path=_extract_trace(best),
            top_k=[{"entity": a.entity_id, "confidence": round(float(a.score), 4)} for a in answers],
            elapsed_ms=round(elapsed_ms, 1),
            raw=answers,
        )

    @property
    def stats(self) -> dict:
        """Return KB statistics: entity count, relation count, community count."""
        G = self._adapter.to_networkx()
        rels = {d.get("relation_type") or d.get("relation", "") for _, _, d in G.edges(data=True)}
        cm = getattr(self._adapter, "community_map", {})
        return {
            "entities": G.number_of_nodes(),
            "relations": G.number_of_edges(),
            "relation_types": len(rels),
            "communities": len(set(cm.values())) if cm else 0,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build(self):
        from core.cerebrum import CerebrumGraph
        self._graph = CerebrumGraph.from_adapter(self._adapter, embeddings=self._embeddings)
        self._graph.build(seed=42)
        self._adapter = self._graph.adapter
        self._built = True

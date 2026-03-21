"""
In-memory graph adapter backed by a NetworkX graph.

This is the default adapter — no external dependencies beyond networkx.
Use it for:
  - Loading from CSV (via csv_adapter.py)
  - Testing and prototyping
  - In-memory graph construction from Python dicts or triples
"""
import difflib
from typing import List, Optional, Dict

import networkx as nx

from core.graph_adapter import GraphAdapter, Entity, Edge


class NetworkXAdapter(GraphAdapter):
    """
    GraphAdapter implementation wrapping a networkx Graph or DiGraph.

    Node data convention:
      G.nodes[node_id] may carry {label, type, ...} attributes.
      Edge data may carry {relation, weight, ...} attributes.
      If absent, node ID is used as label and "entity" as type.
    """

    def __init__(
        self,
        G: nx.Graph,
        entity_types: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        G            : networkx Graph or DiGraph
        entity_types : optional {node_id -> type_string} override
        """
        self._G            = G
        self._entity_types = entity_types or {}

    # ------------------------------------------------------------------
    # Required GraphAdapter methods
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        if entity_id not in self._G:
            return None
        data = self._G.nodes[entity_id]
        return Entity(
            id=entity_id,
            label=data.get("label", entity_id),
            type=self._entity_types.get(entity_id, data.get("type", "entity")),
            properties=dict(data),
        )

    def get_neighbors(
        self,
        entity_id: str,
        edge_types: Optional[List[str]] = None,
        max_neighbors: int = 50,
    ) -> List[Edge]:
        if entity_id not in self._G:
            return []

        edges = []
        neighbors = (
            self._G.successors(entity_id)
            if self._G.is_directed()
            else self._G.neighbors(entity_id)
        )

        for neighbor in neighbors:
            edge_data = self._G.get_edge_data(entity_id, neighbor) or {}
            rel_type  = edge_data.get("relation", "RELATED_TO")

            if edge_types and rel_type not in edge_types:
                continue

            edges.append(
                Edge(
                    source_id=entity_id,
                    target_id=neighbor,
                    relation_type=rel_type,
                    weight=float(edge_data.get("weight", 1.0)),
                    properties=dict(edge_data),
                )
            )

            if len(edges) >= max_neighbors:
                break

        return edges

    def find_entities(self, query: str, top_k: int = 10) -> List[Entity]:
        """
        Fuzzy match query against node IDs and labels.
        Uses an internal n-gram index for O(1)-ish performance on large graphs.
        """
        if not query:
            return []

        # 1. Exact match check (fastest)
        if query in self._G:
            exact = self.get_entity(query)
            # Still do a fuzzy search for alternatives, but put exact first
            fuzzy = self._fuzzy_search(query, top_k=top_k)
            seen = {query}
            result = [exact]
            for e in fuzzy:
                if e.id not in seen:
                    result.append(e)
                    seen.add(e.id)
            return result[:top_k]

        return self._fuzzy_search(query, top_k=top_k)

    def find_entities_masked(self, query: str, top_k: int = 10) -> List[Dict]:
        """Return IDs and scores but hide labels."""
        if not hasattr(self, "_ngram_index"):
            self._build_ngram_index()

        q_ngrams = self._get_ngrams(query.lower())
        scores: Dict[str, float] = {}

        for ng in q_ngrams:
            for node_id in self._ngram_index.get(ng, []):
                scores[node_id] = scores.get(node_id, 0.0) + 1.0

        if not scores:
            return []

        # Normalize by length
        sorted_ids = sorted(
            scores.keys(),
            key=lambda nid: scores[nid] / (len(q_ngrams) + self._node_ngram_len.get(nid, 1)),
            reverse=True
        )

        results = []
        for nid in sorted_ids[:top_k]:
            ent = self.get_entity(nid)
            if ent:
                score = scores[nid] / (len(q_ngrams) + self._node_ngram_len.get(nid, 1))
                results.append({"id": nid, "type": ent.type, "score": float(score)})
        return results

    def _fuzzy_search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Internal n-gram based fuzzy search."""
        if not hasattr(self, "_ngram_index"):
            self._build_ngram_index()

        q_ngrams = self._get_ngrams(query.lower())
        scores: Dict[str, float] = {}

        for ng in q_ngrams:
            for node_id in self._ngram_index.get(ng, []):
                scores[node_id] = scores.get(node_id, 0.0) + 1.0

        if not scores:
            return []

        # Normalize by length (Jaccard-like)
        sorted_ids = sorted(
            scores.keys(),
            key=lambda nid: scores[nid] / (len(q_ngrams) + self._node_ngram_len.get(nid, 1)),
            reverse=True
        )

        return [self.get_entity(nid) for nid in sorted_ids[:top_k] if self.get_entity(nid)]

    def _get_ngrams(self, text: str, n: int = 3) -> List[str]:
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _build_ngram_index(self):
        """Build a simple inverted index of n-grams for all node IDs and labels."""
        self._ngram_index: Dict[str, List[str]] = {}
        self._node_ngram_len: Dict[str, int] = {}

        for node_id in self._G.nodes():
            e = self.get_entity(node_id)
            text = (e.label + " " + node_id).lower() if e else node_id.lower()
            ngrams = set(self._get_ngrams(text))
            self._node_ngram_len[node_id] = len(ngrams)
            for ng in ngrams:
                self._ngram_index.setdefault(ng, []).append(node_id)

    def to_networkx(self) -> nx.Graph:
        return self._G

    def get_community(self, entity_id: str) -> int:
        """Requires communities to be attached to the adapter instance."""
        if hasattr(self, "community_map"):
            return self.community_map.get(entity_id, -1)
        return -1

    def get_embedding(self, entity_id: str) -> Optional["np.ndarray"]:
        """Requires embeddings to be attached to the adapter instance."""
        if hasattr(self, "embeddings"):
            return self.embeddings.get(entity_id)
        return None

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def get_all_entities(self) -> List[Entity]:
        return [self.get_entity(n) for n in self._G.nodes()]

    def get_edge_types(self) -> List[str]:
        types = set()
        for _, _, data in self._G.edges(data=True):
            types.add(data.get("relation", "RELATED_TO"))
        return sorted(types)

    def node_count(self) -> int:
        return self._G.number_of_nodes()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_triples(
        cls,
        triples: List[tuple],
        directed: bool = True,
    ) -> "NetworkXAdapter":
        """
        Build an adapter from a list of (subject, predicate, object) triples.

        Example:
            adapter = NetworkXAdapter.from_triples([
                ("newton", "INFLUENCED", "einstein"),
                ("einstein", "COLLABORATED", "bohr"),
            ])
        """
        G = nx.DiGraph() if directed else nx.Graph()
        for s, p, o in triples:
            G.add_edge(str(s), str(o), relation=str(p))
        return cls(G)




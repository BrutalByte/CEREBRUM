"""
In-memory graph adapter backed by a NetworkX graph.

This is the default adapter — no external dependencies beyond networkx.
Use it for:
  - Loading from CSV (via csv_adapter.py)
  - Testing and prototyping
  - In-memory graph construction from Python dicts or triples
"""
from typing import List, Optional, Dict, TYPE_CHECKING

import networkx as nx
import numpy as np

from core.graph_adapter import GraphAdapter, Entity, Edge

if TYPE_CHECKING:
    from core.thalamus import IngestionPipeline


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
        self.embeddings: Dict[str, np.ndarray] = {}
        self.community_map: Dict[str, int] = {}

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
        max_neighbors: int = 100,
        context_embedding: Optional["np.ndarray"] = None,
    ) -> List[Edge]:
        if entity_id not in self._G:
            return []

        neighbors = (
            self._G.successors(entity_id)  # type: ignore
            if self._G.is_directed()
            else self._G.neighbors(entity_id)
        )

        edges = []
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
                    confidence=float(edge_data.get("confidence", 1.0)),
                    provenance=str(edge_data.get("provenance", "")),
                    valid_from=edge_data.get("valid_from"),
                    valid_to=edge_data.get("valid_to"),
                )
            )

        # Truncate to max_neighbors using insertion order.
        # NOTE: cosine-similarity pre-sorting was evaluated and removed — it
        # biases toward same-type neighbors (path embedding ≈ source entity)
        # and suppresses correct cross-type hops (actor→movie, movie→genre).
        # The CSA attention formula in BeamTraversal already handles relevance
        # scoring; pre-filtering at the adapter level double-counts the wrong
        # signal.  A larger cap (100 vs the old 50) provides better coverage
        # without insertion-order bias affecting common-degree nodes.
        return edges[:max_neighbors]

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
            if exact:
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

        return [e for nid in sorted_ids[:top_k] if (e := self.get_entity(nid)) is not None]

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

    def build_communities(self, resolution: float = 1.0) -> None:
        """
        Run DSCF community detection and attach results to this adapter.

        Populates ``self.community_map`` (node -> int) and
        ``self._partition`` (List[frozenset]) so that get_community()
        returns meaningful IDs and compute_soft_memberships() has a
        partition to work with.

        Safe to call multiple times (re-runs detection each time).
        """
        from core.community_engine import dscf_communities
        G_und = self._G.to_undirected() if self._G.is_directed() else self._G
        parts = dscf_communities(G_und, resolution=resolution)
        self._partition = parts
        cm: Dict[str, int] = {}
        for cid, members in enumerate(parts):
            for node in members:
                cm[node] = cid
        self.community_map = cm

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

    def get_reasoning_branches(
        self,
        seed_id: str,
        context_embedding: Optional[np.ndarray] = None,
        max_hop: int = 2,
        beam_width: int = 5,
        max_budget: int = 500,
    ) -> List[Dict]:
        """Local reasoning branch expansion using BeamTraversal."""
        # Use local imports to avoid circularity
        from core.attention_engine import CSAEngine
        from reasoning.traversal import BeamTraversal
        
        # 1. Setup local reasoning context
        # Note: This uses default weights and requires community_map/embeddings 
        # to be attached to the adapter (Standard CEREBRUM server behavior).
        csa = CSAEngine(adapter=self)
        # If we have community graph metadata, attach it
        if hasattr(self, "csa_metadata") and self.csa_metadata:
            csa.set_community_graph(
                self.csa_metadata.get("distances", {}),
                self.csa_metadata.get("adjacent_pairs", set())
            )
        
        # 2. Run local beam search
        traversal = BeamTraversal(
            adapter=self,
            csa_engine=csa,
            beam_width=beam_width,
            max_hop=max_hop,
            max_budget=max_budget
        )
        
        paths = traversal.traverse([seed_id], query_embedding=context_embedding)
        
        # 3. Serialize
        return [p.to_dict() for p in paths if len(p.nodes) > 1]

    def find_similar(
        self, 
        embedding: "np.ndarray", 
        top_k: int = 10
    ) -> List[Entity]:
        """Brute-force cosine similarity over all entities with embeddings."""
        if not hasattr(self, "embeddings") or not self.embeddings:
            return []

        # Ensure embedding is normalized for cosine sim
        norm_q = np.linalg.norm(embedding)
        if norm_q == 0:
            return []
        
        q_norm = embedding / norm_q
        
        scores = []
        for eid, emb in self.embeddings.items():
            norm_e = np.linalg.norm(emb)
            if norm_e == 0:
                continue
            sim = np.dot(q_norm, emb / norm_e)
            scores.append((eid, sim))
            
        # Sort by similarity
        sorted_eids = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [e for eid, _ in sorted_eids[:top_k] if (e := self.get_entity(eid)) is not None]

    # ------------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------------

    def get_all_entities(self) -> List[Entity]:
        return [e for n in self._G.nodes() if (e := self.get_entity(n)) is not None]

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
        pipeline: Optional["IngestionPipeline"] = None,
    ) -> "NetworkXAdapter":
        """
        Build an adapter from a list of (subject, predicate, object) triples.

        Parameters
        ----------
        triples  : list of (source, relation, target) tuples. A fourth element
                   may be a dict of metadata passed to the pipeline.
        directed : use DiGraph if True (default), Graph if False
        pipeline : optional IngestionPipeline for normalization and enrichment

        Example:
            adapter = NetworkXAdapter.from_triples([
                ("newton", "INFLUENCED", "einstein"),
                ("einstein", "COLLABORATED", "bohr"),
            ])
        """
        G = nx.DiGraph() if directed else nx.Graph()
        for t in triples:
            s, p, o = str(t[0]), str(t[1]), str(t[2])
            meta = t[3] if len(t) > 3 and isinstance(t[3], dict) else {}

            if pipeline is not None:
                edge = pipeline.process(s, o, p, meta)
                G.add_edge(
                    edge.source,
                    edge.target,
                    relation=edge.relation,
                    confidence=edge.confidence,
                    provenance=edge.provenance,
                    weight=edge.weight,
                    **edge.properties,
                )
            else:
                G.add_edge(s, o, relation=p)
        return cls(G)




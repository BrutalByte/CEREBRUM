"""
In-memory graph adapter backed by a NetworkX graph.

This is the default adapter â€” no external dependencies beyond networkx.
Use it for:
  - Loading from CSV (via csv_adapter.py)
  - Testing and prototyping
  - In-memory graph construction from Python dicts or triples
"""
from typing import Counter, Dict, List, Optional, TYPE_CHECKING

import networkx as nx
import numpy as np

from core.graph_adapter import GraphAdapter, Entity, Edge, MetaEdge

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
        self._meta_graph: Optional[nx.DiGraph] = None  # Phase 217: built lazily

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

        is_multi = self._G.is_multigraph()
        edges = []
        for neighbor in neighbors:
            raw = self._G.get_edge_data(entity_id, neighbor) or {}
            # MultiDiGraph: raw = {key: {attr: val, ...}, ...}
            # DiGraph:      raw = {attr: val, ...}
            inner_edges = list(raw.values()) if is_multi else [raw]

            if edge_types:
                # Filtered mode: emit one Edge per matching (neighbor, relation) pair.
                # This lets callers requesting a specific relation type see all
                # parallel edges, e.g. schema execute hop-1/hop-2 traversals.
                for edge_data in inner_edges:
                    rel_type = edge_data.get("relation", "RELATED_TO")
                    if rel_type not in edge_types:
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
            else:
                # Unfiltered mode: emit ONE Edge per unique neighbor.
                # For MultiDiGraph (e.g. Freebase WebQSP), multiple parallel edges
                # exist per (u,v) pair with different relation types.  Picking one
                # arbitrarily would add noise to beam scoring; returning RELATED_TO
                # as a structural placeholder keeps traversal driven by embedding
                # similarity and community scores — the correct signals for a graph
                # with high parallel-edge density.  The schema channel uses the
                # edge_types-filtered path (above) which correctly enumerates all
                # parallel edges; unfiltered callers (beam traversal) see RELATED_TO.
                edge_data = raw if not is_multi else {}
                rel_type = edge_data.get("relation", "RELATED_TO")
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

        # Truncate to max_neighbors. When the cap fires we shuffle first so that
        # minority-type edges (e.g. Disease edges among a compound's 300+ Side
        # Effect and Gene edges) are not systematically excluded by insertion order.
        # CSA in BeamTraversal handles relevance scoring; pre-sorting by embedding
        # similarity would double-count the same-type bias and was removed.
        if len(edges) > max_neighbors:
            import random as _random
            _random.shuffle(edges)
        return edges[:max_neighbors]

    def get_all_relation_types(self, entity_id: str) -> set:
        """Return the set of all outgoing relation types from entity_id.

        For MultiDiGraph, iterates every parallel edge key to capture all
        relation types — not just the first per (u, v) pair as get_neighbors does.
        O(out_degree) with no fan-out cap; suitable for seed_rels collection.
        """
        if entity_id not in self._G:
            return set()
        is_multi = self._G.is_multigraph()
        rels: set = set()
        for _, _, data in self._G.out_edges(entity_id, data=True):
            if is_multi:
                # data is the inner attr dict (NetworkX iterates per-key for MultiGraph)
                rels.add(data.get("relation", "RELATED_TO"))
            else:
                rels.add(data.get("relation", "RELATED_TO"))
        return rels

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
        return self.embeddings.get(entity_id) if hasattr(self, 'embeddings') else None

    def get_degree(self, entity_id: str) -> int:
        return self._G.degree(entity_id) if entity_id in self._G else 0
        return None

    def add_edge(
        self,
        u: str,
        v: str,
        relation: str,
        confidence: float = 1.0,
        provenance: str = "",
        synthetic: bool = False,
    ) -> None:
        """Add an edge to the graph (for Phase 65 materialization)."""
        if not self._G.has_node(u):
            self._G.add_node(u, label=u)
        if not self._G.has_node(v):
            self._G.add_node(v, label=v)
            
        self._G.add_edge(
            u, v,
            relation=relation,
            confidence=confidence,
            provenance=provenance,
            synthetic=synthetic
        )

    def remove_edge(self, u: str, v: str, relation: str) -> None:
        """
        Remove an edge from the graph (for Phase 76 provenance rollback).

        On a MultiGraph, removes the first edge keyed by *relation*.
        On a simple Graph, removes the edge if it exists and has the
        matching relation attribute.

        Raises ValueError if no matching edge is found.
        """
        if not self._G.has_edge(u, v):
            raise ValueError(f"No edge between {u!r} and {v!r}.")

        if self._G.is_multigraph():
            # MultiGraph: edges keyed by integer key; find the one matching relation
            edge_data = self._G.get_edge_data(u, v) or {}
            for key, attrs in edge_data.items():
                if attrs.get("relation") == relation:
                    self._G.remove_edge(u, v, key=key)
                    return
            raise ValueError(
                f"No edge {u!r} -[{relation}]-> {v!r} found (MultiGraph)."
            )
        else:
            ed = self._G.get_edge_data(u, v) or {}
            if ed.get("relation") == relation:
                self._G.remove_edge(u, v)
                return
            raise ValueError(
                f"No edge {u!r} -[{relation}]-> {v!r} found."
            )

    def update_edge_weight(
        self, u: str, v: str, relation: str,
        delta: float = 0.0, max_weight: float = 2.0, min_weight: float = 0.0
    ) -> int:
        """Add delta to edge weight, clamped to [min_weight, max_weight].

        Supports both LTP (positive delta, Phase 96) and LTD (negative delta,
        Phase 97). Returns 1 if the matching edge was found and updated, 0 otherwise.
        Matches by relation attribute. Works for both simple Graph and MultiGraph.
        """
        if not self._G.has_edge(u, v):
            return 0
        if self._G.is_multigraph():
            for _key, attrs in (self._G.get_edge_data(u, v) or {}).items():
                if attrs.get("relation") == relation:
                    attrs["weight"] = min(max_weight, max(min_weight, attrs.get("weight", 1.0) + delta))
                    return 1
        else:
            ed = self._G.get_edge_data(u, v) or {}
            if ed.get("relation") == relation:
                ed["weight"] = min(max_weight, max(min_weight, ed.get("weight", 1.0) + delta))
                return 1
        return 0

    def update_edge_valence(
        self, u: str, v: str, relation: str,
        delta: float = 0.0, min_val: float = -1.0, max_val: float = 1.0
    ) -> int:
        """Add delta to edge valence attribute, clamped to [min_val, max_val].

        Valence is separate from weight and lives in edge attribute "valence".
        Returns 1 if matching edge found and updated, 0 otherwise.
        """
        if not self._G.has_edge(u, v):
            return 0
        if self._G.is_multigraph():
            for _key, attrs in (self._G.get_edge_data(u, v) or {}).items():
                if attrs.get("relation") == relation:
                    attrs["valence"] = min(max_val, max(min_val, attrs.get("valence", 0.0) + delta))
                    return 1
        else:
            ed = self._G.get_edge_data(u, v) or {}
            if ed.get("relation") == relation:
                ed["valence"] = min(max_val, max(min_val, ed.get("valence", 0.0) + delta))
                return 1
        return 0

    def get_edge_valence(self, u: str, v: str, relation: str) -> float:
        """Return the valence attribute of the matching edge, or 0.0 if absent."""
        if not self._G.has_edge(u, v):
            return 0.0
        if self._G.is_multigraph():
            for attrs in (self._G.get_edge_data(u, v) or {}).values():
                if attrs.get("relation") == relation:
                    return float(attrs.get("valence", 0.0))
        else:
            ed = self._G.get_edge_data(u, v) or {}
            if ed.get("relation") == relation:
                return float(ed.get("valence", 0.0))
        return 0.0

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

    def get_all_edges(self, limit: int = 500) -> List[Edge]:
        edges = []
        for src, tgt, data in self._G.edges(data=True):
            if len(edges) >= limit:
                break
            edges.append(Edge(
                source_id=src,
                target_id=tgt,
                relation_type=data.get("relation", "RELATED_TO"),
                weight=float(data.get("weight", 1.0)),
                properties=dict(data),
                confidence=float(data.get("confidence", 1.0)),
                provenance=str(data.get("provenance", "")),
                valid_from=data.get("valid_from"),
                valid_to=data.get("valid_to"),
            ))
        return edges

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
                # Direct addition from triples (no normalization)
                # Ensure core metadata keys make it to edge attributes
                G.add_edge(
                    s, o, 
                    relation=p, 
                    weight=meta.get("weight", 1.0),
                    confidence=meta.get("confidence", 1.0),
                    provenance=meta.get("provenance", "triples"),
                    valid_from=meta.get("valid_from"),
                    valid_to=meta.get("valid_to"),
                    **{k: v for k, v in meta.items() if k not in ["weight", "confidence", "provenance", "valid_from", "valid_to"]}
                )
        return cls(G)

    # ------------------------------------------------------------------
    # Phase 217: Meta-Relation Graph
    # ------------------------------------------------------------------

    def build_meta_graph(self) -> nx.DiGraph:
        """
        Build and cache a meta-relation graph from co-occurrence statistics.

        For every triple Aâ†’[r1]â†’Bâ†’[r2]â†’C in the entity graph, add a
        meta-edge r1â†’r2 with weight = co-occurrence count.  Weights are
        then TF-IDF normalised so ubiquitous meta-paths are downweighted.

        The meta-graph is stored in self._meta_graph and returned.
        """
        from collections import defaultdict, Counter

        # Count co-occurrences of (r1, r2) where r2 follows r1 via a shared node
        co_occur: Dict = defaultdict(int)
        r1_total: Counter = Counter()
        r2_total: Counter = Counter()

        for mid in self._G.nodes():
            in_rels = [
                d.get("relation", d.get("relation_type", ""))
                for _, _, d in self._G.in_edges(mid, data=True)
            ]
            out_rels = [
                d.get("relation", d.get("relation_type", ""))
                for _, _, d in self._G.out_edges(mid, data=True)
            ]
            for r1 in in_rels:
                if not r1:
                    continue
                r1_total[r1] += 1
                for r2 in out_rels:
                    if not r2:
                        continue
                    co_occur[(r1, r2)] += 1
                    r2_total[r2] += 1

        # TF-IDF normalisation: downweight r1â†’r2 pairs where r2 is ubiquitous
        n_nodes = max(1, self._G.number_of_nodes())
        meta = nx.DiGraph()
        for (r1, r2), count in co_occur.items():
            idf = 1.0 + (n_nodes / (r2_total[r2] + 1))
            weight = count * idf / max(1, r1_total[r1])
            if meta.has_edge(r1, r2):
                meta[r1][r2]["weight"] = max(meta[r1][r2]["weight"], weight)
            else:
                meta.add_edge(r1, r2, weight=weight, meta_relation="precedes")

        self._meta_graph = meta
        return meta

    def get_meta_neighbors(self, relation_type: str) -> List[MetaEdge]:
        """Phase 217: Return meta-edges for this relation type."""
        if self._meta_graph is None:
            self.build_meta_graph()
        result = []
        for _, tgt, data in self._meta_graph.out_edges(relation_type, data=True):
            result.append(MetaEdge(
                source_relation=relation_type,
                target_relation=tgt,
                meta_relation=data.get("meta_relation", "precedes"),
                weight=data.get("weight", 1.0),
            ))
        return result

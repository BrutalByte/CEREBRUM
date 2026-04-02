"""
CerebrumGraph — unified THALAMUS → CORTEX pipeline.

This is the main entry point for building and querying a knowledge graph
with CEREBRUM.  It replaces the manual wiring that previously lived inside
individual benchmark scripts.

Pipeline
--------
THALAMUS (ingestion):
  1. Load graph via adapter (from_kb / from_csv / from_triples / from_adapter)
  2. Optional: apply provable completion rules (InverseRule, CompositionRule)
  3. Encode entity embeddings (RandomEngine or SentenceEngine)
  4. Detect communities (DSCF) and optionally coarsen
  5. Build CSA attention engine with community distance matrix

CORTEX (reasoning):
  6. BeamTraversal over the CSA-weighted graph
  7. AnswerExtractor with path scoring and convergence voting

Usage
-----
    from core.cerebrum import CerebrumGraph
    from core.graph_completion import InverseRule, CompositionRule

    # Build from a pipe-separated triples file (MetaQA KB format)
    graph = CerebrumGraph.from_kb(
        "benchmarks/data/metaqa/kb.txt",
        sep="|",
        embeddings="sentence",
    )

    # Optional: augment with provable inference rules before building
    graph.complete([
        InverseRule("starred_actors"),          # symmetric
        InverseRule("directed_by", "director_of"),
    ])

    # Run THALAMUS pipeline (embeddings + communities + CSA)
    graph.build(cache_dir="cache/metaqa", min_community_size=20)

    # Query
    answers = graph.query(["Tom Hanks"], top_k=10, max_hop=2, min_hop=1)
    for a in answers:
        print(a.entity_id, a.score, a.path_confidence)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np

from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from core.embedding_engine import EmbeddingEngine, RandomEngine
from core.graph_adapter import GraphAdapter
from reasoning.answer_extractor import Answer, extract
from reasoning.traversal import BeamTraversal

logger = logging.getLogger("cerebrum.graph")


# ---------------------------------------------------------------------------
# CerebrumGraph
# ---------------------------------------------------------------------------

class CerebrumGraph:
    """
    Encapsulates the full THALAMUS → CORTEX pipeline for a single KG.

    Parameters
    ----------
    adapter          : NetworkXAdapter (or any GraphAdapter) wrapping the graph
    embedding_engine : EmbeddingEngine instance.  Defaults to RandomEngine(64).
    beam_width       : default beam width for traversal (default 10)
    max_hop          : maximum hop depth for traversal (default 3)
    max_neighbors    : per-node neighbor expansion cap (default 100)
    probabilistic    : enable Bayesian Beta-distribution path model (default False)
    warm_start_strength : warm-start strength for probabilistic mode (default 0)
    """

    def __init__(
        self,
        adapter:              GraphAdapter,
        embedding_engine:     Optional[EmbeddingEngine] = None,
        beam_width:           int   = 10,
        max_hop:              int   = 3,
        max_neighbors:        int   = 100,
        probabilistic:        bool  = False,
        warm_start_strength:  float = 0.0,
    ):
        self.adapter             = adapter
        self._embedding_engine   = embedding_engine or RandomEngine(dim=64)
        self._beam_width         = beam_width
        self._max_hop            = max_hop
        self._max_neighbors      = max_neighbors
        self._probabilistic      = probabilistic
        self._warm_start_strength = warm_start_strength

        self._csa:       Optional[CSAEngine]    = None
        self._traversal: Optional[BeamTraversal] = None
        self._built      = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_kb(
        cls,
        path:             Union[str, Path],
        sep:              str  = "|",
        directed:         bool = True,
        embeddings:       str  = "random",
        embedding_dim:    int  = 64,
        pipeline=None,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Load from a flat triples file (one triple per line).

        Format:  head<sep>relation<sep>tail

        Parameters
        ----------
        path          : path to the KB file
        sep           : field separator (default "|" for MetaQA)
        directed      : use DiGraph if True (default), Graph if False
        embeddings    : "random" or "sentence"
        embedding_dim : dimension for random embeddings (ignored for sentence)
        pipeline      : optional IngestionPipeline for normalisation
        """
        G = nx.DiGraph() if directed else nx.Graph()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Basic header detection
                if line.lower().startswith("source") or line.lower().startswith("head"):
                    continue

                parts = line.split(sep)
                if len(parts) < 3:
                    # Try comma/tab fallback if split fails
                    if sep == "|" and "," in line:
                        parts = line.split(",")
                    elif sep == "|" and "\t" in line:
                        parts = line.split("\t")
                    else:
                        continue
                
                if len(parts) < 3:
                    continue

                s = parts[0].strip()
                r = parts[1].strip()
                o = parts[2].strip()
                if not s or not r or not o:
                    continue

                if pipeline is not None:
                    edge = pipeline.process(s, o, r)
                    G.add_edge(
                        edge.source, edge.target,
                        relation=edge.relation,
                        confidence=float(edge.confidence),
                        provenance=str(edge.provenance),
                        weight=float(edge.weight),
                    )
                else:
                    G.add_edge(s, o, relation=r, weight=1.0, confidence=1.0,
                               provenance="kb_file")

        adapter = NetworkXAdapter(G)
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_csv(
        cls,
        path:          Union[str, Path],
        embeddings:    str = "random",
        embedding_dim: int = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Load from a CSV triples file (CEREBRUM CSV adapter format).
        """
        from adapters.csv_adapter import load_csv_adapter
        adapter = load_csv_adapter(str(path))
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_triples(
        cls,
        triples:       List[tuple],
        directed:      bool = True,
        pipeline=None,
        embeddings:    str  = "random",
        embedding_dim: int  = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """
        Build from a list of (subject, predicate, object[, meta_dict]) tuples.
        """
        adapter = NetworkXAdapter.from_triples(triples, directed=directed,
                                               pipeline=pipeline)
        eng     = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @classmethod
    def from_adapter(
        cls,
        adapter:       GraphAdapter,
        embeddings:    str = "random",
        embedding_dim: int = 64,
        **kwargs,
    ) -> "CerebrumGraph":
        """Wrap an existing adapter (e.g. Neo4jAdapter, FederatedAdapter)."""
        eng = cls._make_engine(embeddings, embedding_dim)
        return cls(adapter, embedding_engine=eng, **kwargs)

    @staticmethod
    def _make_engine(embeddings: str, dim: int) -> EmbeddingEngine:
        if embeddings == "sentence":
            from core.embedding_engine import SentenceEngine
            return SentenceEngine()
        elif embeddings == "random":
            return RandomEngine(dim=dim)
        else:
            raise ValueError(
                f"Unknown embeddings type {embeddings!r}. "
                "Use 'random' or 'sentence'."
            )

    # ------------------------------------------------------------------
    # Provable graph completion (pre-build step)
    # ------------------------------------------------------------------

    def complete(self, rules: List[Any]) -> "CerebrumGraph":
        """
        Apply provable inference rules to augment the graph before building.

        Every rule adds synthetic edges with full provenance metadata.
        No statistical predictions are made — only logical deductions.

        Parameters
        ----------
        rules : list of CompletionRule instances
                (InverseRule, CompositionRule, ...)

        Returns self for chaining.
        """
        total = 0
        for rule in rules:
            n = rule.apply(self.adapter)
            logger.info("%s: +%d edges", rule.describe(), n)
            total += n
        if total:
            logger.info("graph_completion: %d synthetic edges added total", total)
        # Invalidate built state — must rebuild after completing
        self._built = False
        return self

    # ------------------------------------------------------------------
    # Proactive enhancement (second layer)
    # ------------------------------------------------------------------

    def enhance(self, enhancers: List[Any]) -> "CerebrumGraph":
        """
        Apply proactive enhancement engines to the graph before building.

        Enhancements (like GraphBridgeEngine) use heuristics or embeddings to
        synthesize new links, unlike complete() which uses pure logic.

        Parameters
        ----------
        enhancers : list of enhancement engine instances
                    (GraphBridgeEngine, ...)

        Returns self for chaining.
        """
        total = 0
        for enhancer in enhancers:
            # We pass self._embedding_engine because many enhancers
            # (like GraphBridgeEngine) require it for similarity compute.
            n = enhancer.apply(self.adapter, self._embedding_engine)
            logger.info("%s: +%d edges", enhancer.describe(), n)
            total += n
        if total:
            logger.info("graph_enhancement: %d synthetic edges added total", total)
        # Invalidate built state
        self._built = False
        return self

    # ------------------------------------------------------------------
    # THALAMUS build pipeline
    # ------------------------------------------------------------------

    def build(
        self,
        cache_dir:            Optional[Union[str, Path]] = None,
        min_community_size:   int   = 0,
        coarsen_target:       Optional[int] = None,
        n_trials:             int   = 1,
        seed:                 int   = 42,
        force_rebuild:        bool  = False,
        community_engine:     str   = "dscf",
    ) -> "CerebrumGraph":
        """
        Run the THALAMUS pipeline: embeddings → DSCF communities → CSA.

        Parameters
        ----------
        cache_dir           : directory for caching embeddings and communities.
                              Pass None to skip caching.
        min_community_size  : merge communities smaller than this (0 = disabled).
        coarsen_target      : alternatively, coarsen to at most this many
                              communities (e.g. 300 for GrailQA).
        n_trials            : DSCF runs to take the best of (default 1).
        seed                : random seed for reproducibility.
        force_rebuild       : ignore cache and recompute everything.
        community_engine    : 'dscf', 'leiden', or 'lpa' (default 'dscf').
                              'leiden' is significantly faster for large graphs (>1M nodes).

        Returns self for chaining.
        """
        from core.community_engine import dscf_communities, leiden_communities, lpa_communities
        from core.structural_encoder import (
            build_community_distance_matrix,
            build_community_graph,
            adjacent_community_pairs,
            coarsen_communities,
            compute_structural_features,
            encode_structural_features,
        )
        import time

        cache = Path(cache_dir) if cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

        G     = self.adapter.to_networkx()
        nodes = list(G.nodes())

        if not nodes:
            raise ValueError("Graph has no nodes — nothing to build.")

        # ----------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------
        emb_cache = cache / "embeddings.pkl" if cache else None

        if not force_rebuild and emb_cache and emb_cache.exists():
            logger.info("Loading cached embeddings from %s", emb_cache)
            with open(emb_cache, "rb") as f:
                self.adapter.embeddings = pickle.load(f)
        else:
            # Build label map: node_id → human-readable label
            entity_labels: Dict[str, str] = {}
            for n in nodes:
                data  = G.nodes[n]
                label = data.get("label") or data.get("friendly_name") or str(n)
                entity_labels[str(n)] = label

            logger.info(
                "Encoding %d entities with %s",
                len(entity_labels),
                type(self._embedding_engine).__name__,
            )
            self.adapter.embeddings = self._embedding_engine.encode_entities(
                entity_labels
            )

            if emb_cache:
                with open(emb_cache, "wb") as f:
                    pickle.dump(self.adapter.embeddings, f)

        # ----------------------------------------------------------
        # 2. Structural & Temporal Enrichment (Phase 33)
        # ----------------------------------------------------------
        # Compute raw graph features (PageRank, Betweenness, Recency)
        logger.info("Computing structural features (Phase 33)...")
        struct_features = compute_structural_features(G, current_time=time.time())
        
        # If we didn't load from cache, apply structural enrichment to embeddings
        if force_rebuild or not (cache / "embeddings.pkl" if cache else None) or not (cache / "embeddings.pkl").exists():
            # Encode features into a vector of the same dimension as existing embeddings
            # We use a residual connection: h = LayerNorm(h + structural_encoding)
            if self.adapter.embeddings:
                sample_emb = next(iter(self.adapter.embeddings.values()))
                emb_dim = len(sample_emb)
                
                logger.info("Encoding structural features (dim=%d)...", emb_dim)
                encoded_struct = encode_structural_features(struct_features, dim=emb_dim, seed=seed)
                
                # Residual addition + normalization
                for node, s_vec in encoded_struct.items():
                    if node in self.adapter.embeddings:
                        h = self.adapter.embeddings[node] + s_vec
                        norm = np.linalg.norm(h)
                        if norm > 0:
                            self.adapter.embeddings[node] = h / norm

                if emb_cache:
                    with open(emb_cache, "wb") as f:
                        pickle.dump(self.adapter.embeddings, f)

        # ----------------------------------------------------------
        # 3. Community detection
        # ----------------------------------------------------------
        comm_cache = cache / "communities.pkl" if cache else None
        G_und      = G.to_undirected() if G.is_directed() else G

        if not force_rebuild and comm_cache and comm_cache.exists():
            logger.info("Loading cached communities from %s", comm_cache)
            with open(comm_cache, "rb") as f:
                parts = pickle.load(f)
        else:
            logger.info(
                "Running %s on %d nodes, %d edges...",
                community_engine, G.number_of_nodes(), G.number_of_edges(),
            )
            if community_engine == "dscf":
                if n_trials > 1:
                    from core.community_engine import best_of_n_dscf
                    parts = best_of_n_dscf(G_und, n_trials=n_trials, seed=seed)
                else:
                    parts = dscf_communities(G_und)
            elif community_engine == "leiden":
                parts = leiden_communities(G_und)
            elif community_engine == "lpa":
                parts = lpa_communities(G_und)
            else:
                raise ValueError(f"Unknown community_engine {community_engine!r}")

            if comm_cache:
                with open(comm_cache, "wb") as f:
                    pickle.dump(parts, f)

        self.adapter._partition = parts
        cm: Dict = {n: cid for cid, members in enumerate(parts) for n in members}
        n_raw = len(set(cm.values()))
        logger.info("%d raw DSCF communities", n_raw)

        # ----------------------------------------------------------
        # 3. Optional coarsening
        # ----------------------------------------------------------
        if min_community_size > 0:
            from core.structural_encoder import coarsen_communities as _size_coarsen
            # merge_small_communities is in metaqa_eval; use structural_encoder's version
            cm = _size_coarsen(G_und, cm, min_size=min_community_size)
            logger.info(
                "%d communities after min_size=%d coarsening",
                len(set(cm.values())), min_community_size,
            )
        elif coarsen_target is not None:
            cm = coarsen_communities(G_und, cm, target_max=coarsen_target)
            logger.info(
                "%d communities after target_max=%d coarsening",
                len(set(cm.values())), coarsen_target,
            )

        self.adapter.community_map = cm

        # ----------------------------------------------------------
        # 4. CSA engine
        # ----------------------------------------------------------
        logger.info("Building CSA engine...")
        distances = build_community_distance_matrix(G_und, cm)
        adj       = adjacent_community_pairs(G_und, cm)
        cg        = build_community_graph(G_und, cm)

        self._csa = CSAEngine(self.adapter)
        self._csa.set_community_graph(distances, adj, community_graph=cg)

        # Phase 33: Pass PageRank from structural features to CSA
        pr_map = {node: data.get("pagerank", 0.0) for node, data in struct_features.items()}
        self._csa.set_pagerank(pr_map)

        # ----------------------------------------------------------
        # 5. BeamTraversal
        # ----------------------------------------------------------
        self._traversal = BeamTraversal(
            adapter             = self.adapter,
            csa_engine          = self._csa,
            beam_width          = self._beam_width,
            max_hop             = self._max_hop,
            max_neighbors       = self._max_neighbors,
            probabilistic       = self._probabilistic,
            warm_start_strength = self._warm_start_strength,
        )

        self._built = True
        n_comm = len(set(cm.values()))
        logger.info(
            "Build complete: %d nodes, %d edges, %d communities",
            G.number_of_nodes(), G.number_of_edges(), n_comm,
        )
        return self

    # ------------------------------------------------------------------
    # CORTEX query
    # ------------------------------------------------------------------

    def query(
        self,
        seeds:             List[str],
        top_k:             int            = 10,
        min_hop:           int            = 1,
        max_hop:           Optional[int]  = None,
        beam_width:        Optional[int]  = None,
        query_embedding:   Optional[np.ndarray] = None,
        relation_prior=None,
        vote_weight:       float          = 0.30,
        memory_threshold_pct: float       = 95.0,
    ) -> List[Answer]:
        """
        Traverse the graph from ``seeds`` and return ranked answers.

        Parameters
        ----------
        seeds           : list of entity IDs to start from
        top_k           : number of answers to return
        min_hop         : minimum hop depth (1 = include 1-hop; 2 = exclude seeds)
        max_hop         : maximum traversal depth. Defaults to the value set at
                          build time. Override per-query to constrain depth (e.g.,
                          max_hop=1 for 1-hop eval avoids flooding with deep paths).
        beam_width      : beam width for this query (default build-time value).
        query_embedding : optional query vector for semantic alignment
        relation_prior  : optional RelationPathPrior or GraphRelationPrior
        vote_weight     : convergence voting weight (default 0.30)
        memory_threshold_pct : safety threshold for resource usage (default 95.0)

        Returns
        -------
        List[Answer] sorted by score descending.
        """
        if not self._built:
            raise RuntimeError(
                "Graph has not been built. Call graph.build() first."
            )

        # Handle per-query overrides by creating a temporary traversal if needed
        bw = beam_width or self._beam_width
        mh = max_hop or self._max_hop
        
        needs_custom = (mh != self._max_hop or bw != self._beam_width or memory_threshold_pct != 95.0)

        if needs_custom:
            from core.resource_governor import ResourceGovernor
            traversal = BeamTraversal(
                adapter             = self.adapter,
                csa_engine          = self._csa,
                beam_width          = bw,
                max_hop             = mh,
                max_neighbors       = self._max_neighbors,
                probabilistic       = self._probabilistic,
                warm_start_strength = self._warm_start_strength,
                governor            = ResourceGovernor(memory_threshold_pct=memory_threshold_pct)
            )
        else:
            traversal = self._traversal

        paths = traversal.traverse(
            seeds,
            query_embedding=query_embedding,
        )

        return extract(
            paths,
            top_k        = top_k,
            min_hop      = min_hop,
            query_embedding = query_embedding,
            relation_prior  = relation_prior,
            vote_weight     = vote_weight,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self.adapter.node_count()

    @property
    def edge_count(self) -> int:
        return self.adapter.to_networkx().number_of_edges()

    @property
    def community_count(self) -> int:
        cm = getattr(self.adapter, "community_map", {})
        return len(set(cm.values()))

    @property
    def communities(self) -> List[frozenset[str]]:
        """Return the partition of nodes into communities."""
        # adapter._partition stores the list of frozensets
        return getattr(self.adapter, "_partition", [])

    @property
    def is_built(self) -> bool:
        return self._built

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"CerebrumGraph({self.node_count} nodes, "
            f"{self.edge_count} edges, "
            f"emb={type(self._embedding_engine).__name__}, "
            f"{status})"
        )

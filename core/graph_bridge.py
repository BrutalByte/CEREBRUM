"""
GraphBridgeEngine — proactive cross-component link synthesis for CEREBRUM.

This module implements the second layer of the Enhancement Layer:

  THALAMUS → [complete()] → [enhance()] → build() → CORTEX

Problem
-------
CWQ and similar KG benchmarks contain per-question 2-hop subgraphs that were
extracted independently.  When merged into a single scaffold, entities that
appear in *different* questions but refer to the same or semantically related
concepts end up in disconnected graph components.  DSCF therefore produces
hundreds of thousands of isolated communities, and BeamTraversal cannot cross
component boundaries — resulting in near-zero multi-hop recall.

Solution
--------
GraphBridgeEngine is a *proactive* bridge synthesizer that runs BEFORE
BeamTraversal.  It:

  1. Detects connected components in the graph.
  2. Identifies "frontier" nodes — low-degree nodes on the periphery of small
     components that are likely missing inter-component links.
  3. Uses pre-trained SentenceEngine embeddings (fixed, not trained on task
     data) to find pairs of frontier nodes in *different* components whose
     cosine similarity exceeds ``min_similarity``.
  4. Materializes a bridge edge with a glass-box provenance string that records
     the similarity score, the source/target frontier nodes, and the component
     IDs being connected.

Every synthetic edge is fully traceable:
    provenance = "rule:bridge_embed|sim:{s:.4f}|from:{F}|to:{N}|components:{c1}->{c2}"

This is NOT link prediction in the ML sense — no model is trained on the task.
SentenceEngine is used purely as a fixed semantic projector, identical to how
it encodes entities during the standard build() pipeline.

Crystal- vs Glass-box classification
-------------------------------------
  CompositionRule / GeneralizedCompositionRule = Crystal-box
    (purely logical, no numerical thresholds, zero uncertainty)
  GraphBridgeEngine = Glass-box
    (transparent algorithm + confidence scores; uses a pre-trained, frozen
    embedding model as a fixed distance metric — no task-specific training)

Parameters
----------
min_similarity    : cosine similarity threshold for creating a bridge edge
                    (default 0.80 — empirically tuned for Freebase name strings)
top_k             : maximum bridges per frontier node (default 5)
max_degree        : maximum degree for a node to qualify as a frontier node
                    (default 2 — keeps only genuinely peripheral nodes)
max_bridges       : hard cap on total bridge edges added (default 500_000)
max_frontier_nodes: maximum frontier nodes to embed (default 10_000).
                    Nodes are sampled from components with 1–50 nodes.
batch_size        : embedding batch size (default 512)

Usage
-----
    from core.graph_bridge import GraphBridgeEngine
    bridge_engine = GraphBridgeEngine(min_similarity=0.82, top_k=3)
    n_added = bridge_engine.apply(adapter, embedding_engine)
"""
from __future__ import annotations

import logging
import random
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from core.security import FederatedAuth
from core.node_registry import NodeRegistry
from adapters.networkx_adapter import NetworkXAdapter
from core.embedding_engine import EmbeddingEngine

logger = logging.getLogger("cerebrum.graph_bridge")

class GraphBridgeEngine:
    """
    Proactive cross-component bridge synthesizer with authenticated signatures.
    """
    def __init__(
        self,
        min_similarity: float = 0.80,
        top_k: int = 5,
        max_degree: int = 2,
        max_bridges: int = 500_000,
        max_frontier_nodes: int = 10_000,
        batch_size: int = 512,
        seed: int = 42,
        node_id: str = "local-node",
        registry: Optional[NodeRegistry] = None,
    ):
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.max_degree = max_degree
        self.max_bridges = max_bridges
        self.max_frontier_nodes = max_frontier_nodes
        self.batch_size = batch_size
        self.seed = seed
        self.node_id = node_id
        self.registry = registry

    def describe(self) -> str:
        return f"GraphBridgeEngine(min_sim={self.min_similarity}, top_k={self.top_k})"

    def _sign_signal(self, payload: Dict[str, Any]) -> bytes:
        """Sign a signal payload."""
        data = json.dumps(payload, sort_keys=True).encode()
        return FederatedAuth.sign_payload(data)

    def _add_authenticated_edge(self, G, src, tgt, attrs, sim_val, src_comp, tgt_comp):
        prov = (
            f"rule:bridge_embed"
            f"|sim:{sim_val:.4f}"
            f"|from:{src}"
            f"|to:{tgt}"
            f"|components:{src_comp}->{tgt_comp}"
            f"|node:{self.node_id}"
        )
        
        # Update attributes with provenance
        attrs["provenance"] = prov
        
        # Create a copy of attributes to sign the payload
        payload_attrs = attrs.copy()
        payload = {"src": src, "tgt": tgt, "attrs": payload_attrs, "prov": prov}
        signature = self._sign_signal(payload).hex()
        
        # Update attributes with signature and signer
        attrs.update({
            "signature": signature,
            "signer": self.node_id
        })
        
        G.add_edge(src, tgt, **attrs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply(
        self,
        adapter:          NetworkXAdapter,
        embedding_engine: EmbeddingEngine,
    ) -> int:
        """
        Apply the bridge engine to ``adapter`` using ``embedding_engine``
        as a fixed semantic distance metric.

        Returns the number of bridge edges added.
        """
        import networkx as nx

        G         = adapter._G
        G_und     = G.to_undirected() if G.is_directed() else G
        rng       = random.Random(self.seed)

        # ------------------------------------------------------------------
        # 1. Identify connected components and their sizes
        # ------------------------------------------------------------------
        logger.info("GraphBridgeEngine: computing connected components...")
        components = list(nx.connected_components(G_und))
        n_comp     = len(components)
        logger.info("  %d connected components", n_comp)

        if n_comp <= 1:
            logger.info("  Graph is fully connected — no bridging needed.")
            return 0

        # Build node → component_id map
        node_to_comp: Dict[str, int] = {}
        for cid, members in enumerate(components):
            for n in members:
                node_to_comp[n] = cid

        # ------------------------------------------------------------------
        # 2. Select frontier nodes
        #    Criteria: degree <= max_degree, from components of size 1..50
        # ------------------------------------------------------------------
        logger.info(
            "GraphBridgeEngine: selecting frontier nodes "
            "(max_degree=%d, component size 1-50)...",
            self.max_degree,
        )
        frontier_nodes: List[str] = []
        for cid, members in enumerate(components):
            comp_size = len(members)
            if comp_size > 50:
                # Large components have enough internal connectivity — only
                # add a small sample of their peripheral nodes
                candidates = [
                    n for n in members
                    if G_und.degree(n) <= self.max_degree
                ]
                if candidates:
                    sample_n = min(5, len(candidates))
                    frontier_nodes.extend(rng.sample(candidates, sample_n))
            else:
                # Small components: include all low-degree nodes
                for n in members:
                    if G_und.degree(n) <= self.max_degree:
                        frontier_nodes.append(n)

        logger.info("  %d frontier nodes identified", len(frontier_nodes))

        if not frontier_nodes:
            logger.info("  No frontier nodes found — skipping bridging.")
            return 0

        # Sample down to max_frontier_nodes
        if len(frontier_nodes) > self.max_frontier_nodes:
            logger.info(
                "  Sampling %d frontier nodes from %d candidates",
                self.max_frontier_nodes, len(frontier_nodes),
            )
            frontier_nodes = rng.sample(frontier_nodes, self.max_frontier_nodes)

        # ------------------------------------------------------------------
        # 3. Encode frontier nodes with the fixed embedding engine
        # ------------------------------------------------------------------
        logger.info(
            "GraphBridgeEngine: encoding %d frontier nodes...",
            len(frontier_nodes),
        )
        entity_labels: Dict[str, str] = {}
        for n in frontier_nodes:
            data  = G.nodes[n] if n in G.nodes else {}
            label = data.get("label") or data.get("friendly_name") or str(n)
            entity_labels[n] = label

        embeddings_map = embedding_engine.encode_entities(entity_labels)

        # Build ordered array
        valid_nodes = [n for n in frontier_nodes if n in embeddings_map]
        if not valid_nodes:
            logger.warning("GraphBridgeEngine: no embeddings produced — aborting.")
            return 0

        emb_array = np.array(
            [embeddings_map[n] for n in valid_nodes],
            dtype=np.float32,
        )

        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb_norm = emb_array / norms

        logger.info(
            "GraphBridgeEngine: embedding matrix %s — computing similarities...",
            emb_norm.shape,
        )

        # ------------------------------------------------------------------
        # 4. Batch cosine similarity + bridge selection
        # ------------------------------------------------------------------
        comp_ids = np.array([node_to_comp[n] for n in valid_nodes], dtype=np.int32)
        n_nodes  = len(valid_nodes)

        to_add:   List[Tuple[str, str, Dict]] = []
        added_set: set = set()

        # Process in batches (query side)
        for batch_start in range(0, n_nodes, self.batch_size):
            batch_end   = min(batch_start + self.batch_size, n_nodes)
            batch_emb   = emb_norm[batch_start:batch_end]      # (B, D)
            batch_comps = comp_ids[batch_start:batch_end]       # (B,)
            batch_nodes = valid_nodes[batch_start:batch_end]

            # Cosine similarities: (B, N)
            sim_matrix = batch_emb @ emb_norm.T                 # (B, N)

            for i, (src_node, src_comp) in enumerate(
                zip(batch_nodes, batch_comps)
            ):
                sims     = sim_matrix[i]                        # (N,)
                # Zero out same-component and self matches
                mask     = comp_ids == src_comp
                sims_filt = sims.copy()
                sims_filt[mask] = 0.0

                # Get top-k indices above threshold
                current_top_k = min(self.top_k, len(sims_filt))
                if current_top_k == 0:
                    continue
                top_idx = np.argpartition(sims_filt, -current_top_k)[-current_top_k:]
                top_idx = top_idx[sims_filt[top_idx] >= self.min_similarity]
                if len(top_idx) == 0:
                    continue

                # Sort descending by similarity
                top_idx = top_idx[np.argsort(sims_filt[top_idx])[::-1]]

                for j in top_idx:
                    tgt_node = valid_nodes[j]
                    tgt_comp = int(comp_ids[j])
                    sim_val  = float(sims_filt[j])
                    pair     = (
                        (src_node, tgt_node)
                        if src_node < tgt_node
                        else (tgt_node, src_node)
                    )
                    if pair in added_set:
                        continue
                    if G.has_edge(src_node, tgt_node):
                        continue
                    added_set.add(pair)
                    prov = (
                        f"rule:bridge_embed"
                        f"|sim:{sim_val:.4f}"
                        f"|from:{src_node}"
                        f"|to:{tgt_node}"
                        f"|components:{src_comp}->{tgt_comp}"
                    )
                    # Add bidirectional edges for DiGraph so traversal can cross
                    to_add.append((src_node, tgt_node, {
                        "relation":   "bridge:similar",
                        "weight":     sim_val,
                        "confidence": sim_val,
                        "provenance": prov,
                        "synthetic":  True,
                    }))
                    if len(to_add) < self.max_bridges and G.is_directed():
                        to_add.append((tgt_node, src_node, {
                            "relation":   "bridge:similar",
                            "weight":     sim_val,
                            "confidence": sim_val,
                            "provenance": prov,
                            "synthetic":  True,
                        }))

                    if len(to_add) >= self.max_bridges:
                        break

                if len(to_add) >= self.max_bridges:
                    logger.warning(
                        "GraphBridgeEngine: hit max_bridges cap %d", self.max_bridges
                    )
                    break

            if len(to_add) >= self.max_bridges:
                break

        # ------------------------------------------------------------------
        # 5. Apply edges to graph
        # ------------------------------------------------------------------
        for src, tgt, attrs in to_add:
            sim_val = attrs["confidence"]
            src_comp = attrs["provenance"].split("components:")[1].split("->")[0]
            tgt_comp = attrs["provenance"].split("components:")[1].split("->")[1].split("|")[0]
            self._add_authenticated_edge(G, src, tgt, attrs, sim_val, src_comp, tgt_comp)

        new = len(to_add)
        if new:
            logger.info("GraphBridgeEngine: added %d bridge edges", new)
            if hasattr(adapter, "_ngram_index"):
                del adapter._ngram_index

        return new

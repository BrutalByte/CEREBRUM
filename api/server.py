"""
Parallax FastAPI REST server.

Endpoints:
  GET  /health       — service health and readiness
  POST /query        — multi-hop KG reasoning query
  GET  /communities  — current community assignments

Startup: loads graph, computes communities and embeddings once, serves queries.

Usage:
    uvicorn api.server:app --port 8200 --reload

Or programmatically:
    from api.server import create_app
    app = create_app(adapter, embedding_engine, community_map)
"""
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import (
    QueryRequest, QueryResponse, CommunitiesResponse,
    HealthResponse, PathResult, PathNode, CommunityInfo,
    EntityResponse, EdgeResponse, SearchResponse,
    CommunityResponse, EmbeddingResponse
)

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_state = {
    "adapter":          None,
    "community_map":    None,   # {node_id -> community_id}
    "embeddings":       None,   # {node_id -> np.ndarray}
    "csa_metadata":     None,   # {"distances": dict, "adjacent_pairs": set}
    "default_edge_type_weights": None,
}


def _is_ready() -> bool:
    return all(_state[k] is not None for k in ("adapter", "community_map", "embeddings", "csa_metadata"))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    adapter=None,
    embedding_engine=None,
    community_map: Optional[dict] = None,
    hierarchical_dscf_enabled: bool = False,
    target_communities: int = 500,
    default_edge_type_weights: Optional[Dict[str, float]] = None,
    cache_path: Optional[str] = None,
) -> FastAPI:
    """
    Create a configured FastAPI app.

    Pass adapter + embedding_engine to load at creation time, or
    call /reload (not yet implemented) to hot-reload after startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if adapter is not None and embedding_engine is not None:
            _load(
                adapter,
                embedding_engine,
                community_map,
                hierarchical_dscf_enabled,
                target_communities,
                default_edge_type_weights,
                cache_path,
            )
        yield

    app = FastAPI(
        title="Parallax KG Reasoning API",
        description="Community-Structured Graph Attention for Knowledge Graph Reasoning",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        cm  = _state["community_map"] or {}
        emb = _state["embeddings"] or {}
        return HealthResponse(
            status="ok" if _is_ready() else "loading",
            adapter_loaded=_state["adapter"] is not None,
            communities_loaded=_state["community_map"] is not None,
            embeddings_loaded=_state["embeddings"] is not None,
            node_count=len(emb),
            community_count=len(set(cm.values())) if cm else 0,
        )

    @app.post("/query", response_model=QueryResponse, tags=["reasoning"])
    async def query(req: QueryRequest):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready — call load() first")

        from core.attention_engine import CSAEngine
        from reasoning.traversal import BeamTraversal
        from reasoning.answer_extractor import extract
        from llm_bridge.context_formatter import to_structured

        adapter      = _state["adapter"]
        community_map = _state["community_map"]
        embeddings   = _state["embeddings"]
        csa_meta     = _state["csa_metadata"]
        default_edge_type_weights = _state["default_edge_type_weights"]

        # Resolve seeds
        if req.seeds:
            seeds = req.seeds
        else:
            entities = adapter.find_entities(req.query, top_k=5)
            seeds    = [e.id for e in entities if e]

        if not seeds:
            raise HTTPException(status_code=404, detail=f"No entities found for query: {req.query!r}")

        # Build CSA engine using precomputed metadata
        edge_type_weights_to_use = req.edge_type_weights or default_edge_type_weights

        csa = CSAEngine(
            adapter=adapter,
            edge_type_weights=edge_type_weights_to_use
        )
        csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])

        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=req.beam_width,
            max_hop=req.max_hop,
        )
        paths   = traversal.traverse(seeds)
        answers = extract(paths, top_k=req.top_k)

        # Format response
        structured = to_structured(answers, query=req.query, adapter=adapter)
        path_results = []
        for p in structured["paths"]:
            path_results.append(PathResult(
                rank=p["rank"],
                answer_entity=p["answer_entity"],
                score=p["score"],
                score_breakdown=p["score_breakdown"],
                path=[PathNode(**n) for n in p["path"]],
            ))

        return QueryResponse(
            query=req.query,
            seeds_used=seeds,
            paths=path_results,
            total_paths_explored=len(paths),
        )

    @app.get("/communities", response_model=CommunitiesResponse, tags=["graph"])
    async def communities():
        if _state["community_map"] is None:
            raise HTTPException(status_code=503, detail="Communities not loaded")

        cm = _state["community_map"]
        community_members: dict = {}
        for node, cid in cm.items():
            community_members.setdefault(cid, []).append(node)

        community_infos = [
            CommunityInfo(
                community_id=cid,
                size=len(members),
                sample_members=members[:10],
            )
            for cid, members in sorted(community_members.items())
        ]

        return CommunitiesResponse(
            node_to_community=cm,
            community_count=len(community_members),
            node_count=len(cm),
            communities=community_infos,
        )

    # ---------------------------------------------------------------------------
    # Low-level Graph Access (for Federated/Remote Adapters)
    # ---------------------------------------------------------------------------

    @app.get("/entities/{entity_id}", response_model=EntityResponse, tags=["graph"])
    async def get_entity(entity_id: str):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")
        
        adapter = _state["adapter"]
        ent = adapter.get_entity(entity_id)
        if not ent:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")
        
        return EntityResponse(
            id=ent.id, 
            label=ent.label, 
            type=ent.type, 
            properties=ent.properties
        )

    @app.get("/entities/{entity_id}/neighbors", response_model=List[EdgeResponse], tags=["graph"])
    async def get_neighbors(entity_id: str, edge_types: Optional[str] = None, max_neighbors: int = 50):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        types_list = edge_types.split(",") if edge_types else None
        
        edges = adapter.get_neighbors(entity_id, edge_types=types_list, max_neighbors=max_neighbors)
        return [
            EdgeResponse(
                source_id=e.source_id, 
                target_id=e.target_id, 
                relation_type=e.relation_type, 
                weight=e.weight, 
                properties=e.properties
            ) for e in edges
        ]

    @app.get("/entities/{entity_id}/community", response_model=CommunityResponse, tags=["graph"])
    async def get_entity_community(entity_id: str):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        cid = adapter.get_community(entity_id)
        return CommunityResponse(entity_id=entity_id, community_id=cid)

    @app.get("/entities/{entity_id}/embedding", response_model=EmbeddingResponse, tags=["graph"])
    async def get_entity_embedding(entity_id: str):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        emb = adapter.get_embedding(entity_id)
        if emb is None:
            raise HTTPException(status_code=404, detail=f"Embedding for '{entity_id}' not found")
        
        return EmbeddingResponse(entity_id=entity_id, embedding=emb.tolist())

    @app.get("/search", response_model=SearchResponse, tags=["graph"])
    async def search(q: str, top_k: int = 10):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        ents = adapter.find_entities(q, top_k=top_k)
        
        # Filter None results if adapter returns sparse list
        valid_ents = [e for e in ents if e]
        
        return SearchResponse(
            query=q,
            results=[
                EntityResponse(
                    id=e.id, 
                    label=e.label, 
                    type=e.type, 
                    properties=e.properties
                ) for e in valid_ents
            ]
        )

    return app


def _load(
    adapter,
    embedding_engine,
    community_map=None,
    hierarchical_dscf_enabled: bool = False,
    target_communities: int = 500,
    default_edge_type_weights: Optional[Dict[str, float]] = None,
    cache_path: Optional[str] = None,
):
    """Load graph state into the global _state dict."""
    import random
    from core.community_engine import best_of_n_dscf, hierarchical_dscf
    from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
    from core.persistence import is_state_cached, load_state, save_state

    # 0. Check cache
    if cache_path and is_state_cached(cache_path):
        print(f"  [API] Loading state from cache: {cache_path}")
        try:
            state = load_state(cache_path)
            _state.update({
                "adapter": state["adapter"],
                "community_map": state["community_map"],
                "embeddings": state["embeddings"],
                "csa_metadata": state["csa_metadata"],
                "default_edge_type_weights": state.get("default_edge_type_weights"),
            })
            return
        except Exception as e:
            print(f"  [API] Cache load failed: {e}. Falling back to computation.")

    _state["adapter"] = adapter
    _state["default_edge_type_weights"] = default_edge_type_weights

    G = adapter.to_networkx()

    # 1. Community detection
    if community_map is None:
        if hierarchical_dscf_enabled:
            parts = hierarchical_dscf(G, target_communities=target_communities)
        else:
            parts = best_of_n_dscf(G, n_trials=3, resolution=adapter.adaptive_resolution())

        community_map = {}
        for cid, members in enumerate(parts):
            for node in members:
                community_map[node] = cid
    _state["community_map"] = community_map
    # Attach to adapter for get_community() lookups
    adapter.community_map = community_map

    # 2. Precompute CSA metadata (distances and adjacent pairs)
    # This avoids O(E) or O(C^2) calculations on every query.
    _state["csa_metadata"] = {
        "distances": build_community_distance_matrix(G, community_map),
        "adjacent_pairs": adjacent_community_pairs(G, community_map),
    }

    # 3. Entity embeddings + Structural Encoding (STEP 2)
    from core.structural_encoder import compute_structural_features, encode_structural_features

    entity_labels = {}
    for node in G.nodes():
        e = adapter.get_entity(node)
        entity_labels[node] = e.label if e else node
    
    # 3.1 Get base embeddings (e.g. from SentenceTransformers)
    base_embeddings = embedding_engine.encode_entities(entity_labels)
    emb_dim = len(next(iter(base_embeddings.values()))) if base_embeddings else 384

    # 3.2 Compute structural features (PageRank, Betweenness, Degree)
    struct_feats = compute_structural_features(G)
    struct_embs  = encode_structural_features(struct_feats, dim=emb_dim)

    # 3.3 Fuse: h0_i = h_semantic + h_structural
    # (Simple addition; LayerNorm is handled inside BeamTraversal)
    fused_embeddings = {}
    for node in G.nodes():
        sem = base_embeddings.get(node, np.zeros(emb_dim, dtype=np.float32))
        strc = struct_embs.get(node, np.zeros(emb_dim, dtype=np.float32))
        fused_embeddings[node] = sem + strc
    
    _state["embeddings"] = fused_embeddings
    # Attach to adapter for get_embedding() lookups
    adapter.embeddings = fused_embeddings

    # 4. Save to cache
    if cache_path:
        save_state(
            cache_path,
            adapter=_state["adapter"],
            community_map=_state["community_map"],
            embeddings=_state["embeddings"],
            csa_metadata=_state["csa_metadata"],
            default_edge_type_weights=_state["default_edge_type_weights"],
        )





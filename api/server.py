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
from typing import Optional

from fastapi import FastAPI, HTTPException

from api.schemas import (
    QueryRequest, QueryResponse, CommunitiesResponse,
    HealthResponse, PathResult, PathNode, CommunityInfo,
)

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_state = {
    "adapter":          None,
    "community_map":    None,   # {node_id -> community_id}
    "embeddings":       None,   # {node_id -> np.ndarray}
    "csa_engine":       None,
    "traversal":        None,
}


def _is_ready() -> bool:
    return all(_state[k] is not None for k in ("adapter", "community_map", "embeddings"))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    adapter=None,
    embedding_engine=None,
    community_map: Optional[dict] = None,
) -> FastAPI:
    """
    Create a configured FastAPI app.

    Pass adapter + embedding_engine to load at creation time, or
    call /reload (not yet implemented) to hot-reload after startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if adapter is not None and embedding_engine is not None:
            _load(adapter, embedding_engine, community_map)
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
        from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
        from reasoning.traversal import BeamTraversal
        from reasoning.answer_extractor import extract
        from llm_bridge.context_formatter import to_structured

        adapter      = _state["adapter"]
        community_map = _state["community_map"]
        embeddings   = _state["embeddings"]

        # Resolve seeds
        if req.seeds:
            seeds = req.seeds
        else:
            entities = adapter.find_entities(req.query, top_k=5)
            seeds    = [e.id for e in entities if e]

        if not seeds:
            raise HTTPException(status_code=404, detail=f"No entities found for query: {req.query!r}")

        # Build CSA engine
        G  = adapter.to_networkx()
        dist = build_community_distance_matrix(G, community_map)
        adj  = adjacent_community_pairs(G, community_map)

        csa = CSAEngine(communities=community_map, embeddings=embeddings)
        csa.set_community_graph(dist, adj)

        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            embeddings=embeddings,
            communities=community_map,
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

    return app


def _load(adapter, embedding_engine, community_map=None):
    """Load graph state into the global _state dict."""
    import random
    from core.community_engine import best_of_n_dscf

    _state["adapter"] = adapter

    G = adapter.to_networkx()

    # Community detection
    if community_map is None:
        parts         = best_of_n_dscf(G, n_trials=3, resolution=adapter.adaptive_resolution())
        community_map = {}
        for cid, members in enumerate(parts):
            for node in members:
                community_map[node] = cid
    _state["community_map"] = community_map

    # Entity embeddings
    entity_labels = {}
    for node in G.nodes():
        e = adapter.get_entity(node)
        entity_labels[node] = e.label if e else node
    _state["embeddings"] = embedding_engine.encode_entities(entity_labels)


# ---------------------------------------------------------------------------
# Default app instance (for uvicorn)
# ---------------------------------------------------------------------------

app = FastAPI(title="Parallax KG Reasoning API", version="0.1.0")


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "note": "Load data via create_app() for full functionality"}

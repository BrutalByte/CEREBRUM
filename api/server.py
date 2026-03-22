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
import json
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED
from core.security import FederatedAuth

from api.schemas import (
    QueryRequest, QueryResponse, CommunitiesResponse,
    HealthResponse, PathResult, PathNode, CommunityInfo,
    EntityResponse, EdgeResponse, SearchResponse,
    CommunityResponse, EmbeddingResponse,
    MaskedEntityResponse, MaskedSearchResponse,
    CommunitySignatureSchema, HologramResponse,
    HandshakeResponse, ReasoningCallbackRequest, ReasoningCallbackResponse,
    StreamIngestRequest, StreamIngestResponse, StreamStatusResponse,
    BridgeRecordSchema, BridgesResponse,
)

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

async def get_authenticated_node(
    api_key: Optional[str] = Security(api_key_header),
    token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Dict:
    """
    Unified security: accepts either static X-API-Key or JWT Bearer token.
    Returns a 'node' context (identity and scopes).
    """
    # 1. Try API Key (Backwards compatibility / Local dev)
    if api_key:
        expected_key = os.getenv("PARALLAX_API_KEY", "dev-secret")
        if api_key == expected_key:
            return {"node_id": "local-admin", "scopes": ["query", "search", "graph"]}
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key")

    # 2. Try JWT Bearer Token (Federated / Production)
    if token:
        try:
            payload = FederatedAuth.validate_token(token.credentials)
            return {"node_id": payload["sub"], "scopes": payload.get("scopes", [])}
        except Exception:
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or expired JWT")

    # 3. No credentials provided
    raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Authentication required")

def check_scope(required_scope: str):
    """Dependency factory for checking specific scopes on the authenticated node."""
    async def scope_checker(node: Dict = Depends(get_authenticated_node)):
        if required_scope not in node["scopes"] and "local-admin" != node["node_id"]:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, 
                detail=f"Scope {required_scope!r} required"
            )
        return node
    return scope_checker

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_state = {
    "adapter":          None,
    "community_map":    None,   # {node_id -> community_id}
    "embeddings":       None,   # {node_id -> np.ndarray}
    "csa_metadata":     None,   # {"distances": dict, "adjacent_pairs": set}
    "default_edge_type_weights": None,
    "hologram":         None,   # List[CommunitySignature]
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
    async def query(req: QueryRequest, node: Dict = Depends(check_scope("query"))):
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
            max_budget=req.max_budget,
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

    @app.post("/query/stream", tags=["reasoning"])
    async def query_stream(req: QueryRequest, node: Dict = Depends(check_scope("query"))):
        """
        Streaming version of /query. Yields paths hop-by-hop as JSON lines.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        from core.attention_engine import CSAEngine
        from reasoning.traversal import AsyncBeamTraversal
        from llm_bridge.context_formatter import to_structured

        adapter      = _state["adapter"]
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

        async def generate():
            from reasoning.answer_extractor import extract
            csa = CSAEngine(adapter=adapter, edge_type_weights=req.edge_type_weights or default_edge_type_weights)
            csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])

            traversal = AsyncBeamTraversal(
                adapter=adapter,
                csa_engine=csa,
                beam_width=req.beam_width,
                max_hop=req.max_hop,
                max_budget=req.max_budget
            )

            hop_count = 0
            async for hop_paths in traversal.traverse_stream(seeds):
                # Format this hop's paths
                # convert paths to Answers for to_structured
                answers = extract(hop_paths, top_k=req.top_k, min_hop=0)
                structured = to_structured(answers, query=req.query, adapter=adapter)
                
                # Add metadata
                chunk = {
                    "hop": hop_count,
                    "paths": structured["paths"],
                    "status": "complete" if hop_count == req.max_hop else "reasoning"
                }
                yield json.dumps(chunk) + "\n"
                hop_count += 1

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.get("/communities", response_model=CommunitiesResponse, tags=["graph"])
    async def communities(node: Dict = Depends(get_authenticated_node)):
        if _state["community_map"] is None:
            raise HTTPException(status_code=503, detail="Communities not loaded")

        cm = _state["community_map"]
        community_members: dict = {}
        for node_id, cid in cm.items():
            community_members.setdefault(cid, []).append(node_id)

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
    async def get_entity(entity_id: str, node: Dict = Depends(get_authenticated_node)):
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
    async def get_neighbors(entity_id: str, edge_types: Optional[str] = None, max_neighbors: int = 50, node: Dict = Depends(get_authenticated_node)):
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
    async def get_entity_community(entity_id: str, node: Dict = Depends(get_authenticated_node)):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        cid = adapter.get_community(entity_id)
        return CommunityResponse(entity_id=entity_id, community_id=cid)

    @app.get("/entities/{entity_id}/embedding", response_model=EmbeddingResponse, tags=["graph"])
    async def get_entity_embedding(entity_id: str, node: Dict = Depends(get_authenticated_node)):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        emb = adapter.get_embedding(entity_id)
        if emb is None:
            raise HTTPException(status_code=404, detail=f"Embedding for '{entity_id}' not found")
        
        return EmbeddingResponse(entity_id=entity_id, embedding=emb.tolist())

    @app.get("/search", response_model=SearchResponse, tags=["graph"])
    async def search(q: str, top_k: int = 10, node: Dict = Depends(check_scope("search"))):
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

    @app.get("/search/masked", response_model=MaskedSearchResponse, tags=["graph"])
    async def search_masked(q: str, top_k: int = 10, node: Dict = Depends(check_scope("search"))):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Security hardening: Cap top_k to prevent enumeration attacks or DoS
        if top_k > 100:
            top_k = 100

        adapter = _state["adapter"]
        results = adapter.find_entities_masked(q, top_k=top_k)
        
        return MaskedSearchResponse(
            query=q,
            results=[MaskedEntityResponse(**r) for r in results]
        )

    @app.get("/hologram", response_model=HologramResponse, tags=["federated"])
    async def get_hologram(node: Dict = Depends(get_authenticated_node)):
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")
        
        sigs = _state["hologram"]
        if sigs is None:
            raise HTTPException(status_code=503, detail="Hologram not computed")
            
        return HologramResponse(
            adapter_name="local", # Future: make configurable
            signatures=[CommunitySignatureSchema(**s.to_dict()) for s in sigs]
        )

    @app.get("/handshake", response_model=HandshakeResponse, tags=["federated"])
    async def handshake():
        if not _is_ready():
            return HandshakeResponse(
                version="0.2.0",
                capabilities=[],
                entity_types=[],
                relation_types=[],
                node_count=0,
                community_count=0
            )
        
        adapter = _state["adapter"]
        cm = _state["community_map"]
        
        # Infer types from adapter if supported, or sample
        # For NetworkXAdapter we can look at node/edge attributes
        entity_types = set()
        relation_types = set()
        
        # Sample some nodes/edges to find types (limited for performance)
        G = adapter.to_networkx()
        for _, data in list(G.nodes(data=True))[:100]:
            entity_types.add(data.get("type", "entity"))
        for _, _, data in list(G.edges(data=True))[:100]:
            relation_types.add(data.get("relation", "link"))

        return HandshakeResponse(
            version="0.2.0",
            capabilities=["query", "search", "masked_search", "hologram", "traversal", "reasoning_callback"],
            entity_types=list(entity_types),
            relation_types=list(relation_types),
            node_count=adapter.node_count(),
            community_count=len(set(cm.values()))
        )

    @app.post("/reason", response_model=ReasoningCallbackResponse, tags=["federated"])
    async def reason_callback(req: ReasoningCallbackRequest, node: Dict = Depends(check_scope("query"))):
        """
        Verify if a path exists between source and target in this graph.
        Used for advanced federated path verification.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        from core.attention_engine import CSAEngine
        from reasoning.traversal import BeamTraversal
        from reasoning.answer_extractor import extract
        from llm_bridge.context_formatter import to_structured

        adapter      = _state["adapter"]
        community_map = _state["community_map"]
        csa_meta     = _state["csa_metadata"]

        csa = CSAEngine(adapter=adapter)
        csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])

        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=5, # Narrower for verification
            max_hop=req.max_hop,
            max_budget=req.max_budget,
        )
        
        paths = traversal.traverse([req.source_id])
        
        # Filter paths that end at target_id
        target_paths = [p for p in paths if p.tail == req.target_id]
        
        if not target_paths:
            return ReasoningCallbackResponse(found=False, paths=[])

        # Format matches
        structured = to_structured(target_paths, query=f"link {req.source_id} to {req.target_id}", adapter=adapter)
        path_results = []
        for p in structured["paths"]:
            path_results.append(PathResult(
                rank=p["rank"],
                answer_entity=p["answer_entity"],
                score=p["score"],
                score_breakdown=p["score_breakdown"],
                path=[PathNode(**n) for n in p["path"]],
            ))

        return ReasoningCallbackResponse(found=True, paths=path_results)

    # ---------------------------------------------------------------------------
    # Phase 11 — Streaming Endpoints
    # ---------------------------------------------------------------------------

    @app.post("/stream/ingest", response_model=StreamIngestResponse, tags=["streaming"])
    async def stream_ingest(req: StreamIngestRequest, node: Dict = Depends(check_scope("query"))):
        """
        Push a batch of StreamEvents into the live graph.

        The adapter must be a StreamAdapter (loaded via create_stream_app()).
        Regular static adapters will return 501.
        """
        from adapters.stream_adapter import StreamAdapter
        from core.stream_engine import StreamEvent as SE

        adapter = _state["adapter"]
        if not isinstance(adapter, StreamAdapter):
            raise HTTPException(
                status_code=501,
                detail="Stream ingestion requires a StreamAdapter. "
                       "Start the server with create_stream_app().",
            )

        import time as _time
        events = [
            SE(
                source=ev.source,
                relation=ev.relation,
                target=ev.target,
                timestamp=ev.timestamp or _time.time(),
                metadata=ev.metadata,
                ttl=ev.ttl,
            )
            for ev in req.events
        ]
        adapter.ingest_batch(events)

        # Sync updated community map back to server state
        _state["community_map"] = adapter.community_map

        stats = adapter.live_stats()
        return StreamIngestResponse(
            ingested=len(events),
            nodes=stats["nodes"],
            edges=stats["edges"],
            communities=stats["communities"],
        )

    @app.get("/stream/status", response_model=StreamStatusResponse, tags=["streaming"])
    async def stream_status(node: Dict = Depends(get_authenticated_node)):
        """
        Return live statistics for the streaming graph.
        Works with both StreamAdapter (live stats) and static adapters (snapshot).
        """
        from adapters.stream_adapter import StreamAdapter

        adapter = _state["adapter"]
        if isinstance(adapter, StreamAdapter):
            s = adapter.live_stats()
            return StreamStatusResponse(**s)

        # Fallback for static adapters
        cm = _state["community_map"] or {}
        emb = _state["embeddings"] or {}
        try:
            G = adapter.to_networkx()
            n_edges = G.number_of_edges()
        except Exception:
            n_edges = 0
        return StreamStatusResponse(
            running=_is_ready(),
            nodes=len(emb),
            edges=n_edges,
            communities=len(set(cm.values())) if cm else 0,
            buffer_size=0,
            sources=0,
            events_per_second=0.0,
            total_ingested=0,
            total_evicted=0,
            total_community_updates=0,
        )

    @app.get("/stream/events", tags=["streaming"])
    async def stream_events(node: Dict = Depends(get_authenticated_node)):
        """
        Server-Sent Events (SSE) stream of live graph mutations.

        Each event is a JSON object:
            {"action": "add"|"remove", "source": "...", "relation": "...", "target": "...",
             "timestamp": 1234567890.0, "metadata": {...}}

        Clients subscribe and receive real-time graph updates.
        Only available when the adapter is a StreamAdapter.
        """
        from adapters.stream_adapter import StreamAdapter
        import asyncio

        adapter = _state["adapter"]
        if not isinstance(adapter, StreamAdapter):
            raise HTTPException(
                status_code=501,
                detail="SSE stream requires a StreamAdapter.",
            )

        event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        loop = asyncio.get_event_loop()

        def on_mutation(action: str, event):
            try:
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    {"action": action, "source": event.source,
                     "relation": event.relation, "target": event.target,
                     "timestamp": event.timestamp, "metadata": event.metadata},
                )
            except Exception:
                pass  # Queue full — drop event rather than block

        adapter.add_mutation_listener(on_mutation)

        async def generate():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                        yield f"data: {json.dumps(item)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                adapter.remove_mutation_listener(on_mutation)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/bridges", response_model=BridgesResponse, tags=["reasoning"])
    async def list_bridges(node: Dict = Depends(get_authenticated_node)):
        """
        List all active bridge twin nodes.

        Bridge twins are structural relay nodes created when a cross-community
        traversal occurs repeatedly (>= n_min times) and the crossing node
        semantically fits the destination community. Each bridge completes a
        bidirectional circuit between two communities, eliminating the
        exponential distance penalty for frequently-used crossings.

        Returns an empty list if the BridgeTwinEngine has not been attached
        to the current BeamTraversal instance.
        """
        from core.bridge_engine import BridgeTwinEngine

        bridge_engine: Optional[BridgeTwinEngine] = _state.get("bridge_engine")
        if bridge_engine is None:
            return BridgesResponse(total=0, bridges=[])

        records = bridge_engine.active_bridges()
        return BridgesResponse(
            total=len(records),
            bridges=[
                BridgeRecordSchema(
                    original_id=r.original_id,
                    twin_id=r.twin_id,
                    source_community=r.source_community,
                    destination_community=r.destination_community,
                    traversal_count=r.traversal_count,
                    age_days=r.age_days(),
                    idle_days=r.idle_days(),
                    similarity_at_creation=r.similarity_at_creation,
                )
                for r in records
            ],
        )

    return app


def create_stream_app(
    stream_adapter=None,
    time_window_seconds: float = 60.0,
    max_edges: int = 10_000,
    **kwargs,
) -> "FastAPI":
    """
    Convenience factory for a streaming-enabled Parallax API.

    Creates (or wraps) a StreamAdapter and calls create_app().
    The adapter is registered in _state so /stream/ingest and
    /stream/events work out of the box.

    Parameters
    ----------
    stream_adapter        : existing StreamAdapter to use; created fresh if None
    time_window_seconds   : sliding window (ignored if stream_adapter provided)
    max_edges             : max live edges (ignored if stream_adapter provided)
    **kwargs              : forwarded to create_app()
    """
    from adapters.stream_adapter import StreamAdapter
    from core.embedding_engine import RandomEngine

    if stream_adapter is None:
        stream_adapter = StreamAdapter(
            time_window_seconds=time_window_seconds,
            max_edges=max_edges,
        )

    embedding_engine = kwargs.pop("embedding_engine", RandomEngine(dim=64))
    return create_app(adapter=stream_adapter, embedding_engine=embedding_engine, **kwargs)


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

    # 4. Compute Holographic Index (Phase 8)
    from core.holographic_index import build_signatures
    _state["hologram"] = build_signatures(adapter, community_map, fused_embeddings)

    # 5. Save to cache
    if cache_path:
        save_state(
            cache_path,
            adapter=_state["adapter"],
            community_map=_state["community_map"],
            embeddings=_state["embeddings"],
            csa_metadata=_state["csa_metadata"],
            default_edge_type_weights=_state["default_edge_type_weights"],
            hologram=_state["hologram"],
        )





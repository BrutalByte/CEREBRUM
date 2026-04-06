"""
CEREBRUM FastAPI REST server.

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
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any

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
    SimilarSearchRequest, SimilarSearchResponse, FeedbackRequest,
    CommunityResponse, EmbeddingResponse,
    MaskedEntityResponse, MaskedSearchResponse,
    CommunitySignatureSchema, HologramResponse,
    HandshakeResponse, TraversalBranchRequest, TraversalBranchResponse, ReasoningCallbackRequest, ReasoningCallbackResponse,
    StreamIngestRequest, StreamIngestResponse, StreamStatusResponse,
    BridgeRecordSchema, BridgesResponse,
    REMRunRequest, REMReportSchema, REMStatusResponse, REMRollbackResponse,
    InferenceReportSchema, InferenceStatusResponse, InferenceRollbackResponse,
    InferenceProposalSchema,
    ChatRequest, ChatResponse, ChatResetResponse, ConversationTurnSchema,
    IngestTextRequest, IngestReportSchema, IngestTripleSchema,
    InsightEventSchema, InsightStatusResponse, InsightScanResponse,
    InsightValidateResponse, InsightValidateAllResponse,
    MetaInsightEventSchema, MetaInsightStatusResponse,
    InsightGraphNodeSchema, InsightGraphEdgeSchema, InsightGraphResponse,
    ParamsResponse,
    ParamsImportRequest,
    RetrainRequest,
    RetrainResponse,
    HypothesisProposalSchema, HypothesizeRequest, HypothesizeResponse,
    HypothesisMaterializeRequest, HypothesisMaterializeResponse,
    HypothesisStatusResponse, TraversalPathSchema,
    ResearchCandidateSchema, ResearchFindingSchema, ResearchStatusResponse,
    ResearchScanResponse, ResearchApproveResponse, ResearchRejectResponse,
    LiteratureHitSchema, ValidationReportSchema,
    ValidateProposalsRequest, ValidateProposalsResponse,
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

_state: Dict[str, Any] = {
    "adapter":          None,
    "community_map":    None,   # {node_id -> community_id}
    "embeddings":       None,   # {node_id -> np.ndarray}
    "csa_metadata":     None,   # {"distances": dict, "adjacent_pairs": set}
    "default_edge_type_weights": None,
    "hologram":         None,   # List[CommunitySignature]
    "meta_learner":     None,   # MetaParameterLearner (Milestone 4)
    "feedback_buffer":  [],     # list of {"path": TraversalPath, "reward": float}
    "rem_engine":       None,   # REMEngine (lazy-initialized on first /rem call)
    "infer_engine":     None,   # TransitiveInferenceEngine (lazy-initialized on first /infer call)
    "hypothesis_engine": None,  # HypothesisEngine (lazy-initialized on first /hypothesize call)
    "research_agent":   None,   # ResearchAgent (lazy-initialized on first /research call)
    "external_validator": None, # ExternalValidator (lazy-initialized on first /research/validate call)
    "chat_manager":     None,   # ConversationManager (lazy-initialized on first /chat call)
    "chat_sessions":    {},     # session_id -> ConversationSession
    "text_ingestor":    None,   # TextIngestor (lazy-initialized on first /ingest call)
    "insight_engine":   None,   # InsightEngine (lazy-initialized on first /insight call)
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
    use_meta_learning: bool = True,
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
                use_meta_learning=use_meta_learning,
            )
        yield

    app = FastAPI(
        title="CEREBRUM KG Reasoning API",
        description="Community-Structured Graph Attention for Knowledge Graph Reasoning",
        version="1.2.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(node: Dict = Depends(get_authenticated_node)):
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
        _state["community_map"]
        _state["embeddings"]
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
            edge_type_weights=edge_type_weights_to_use,
            community_params=csa_meta.get("community_params"),
        )
        csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])
        
        # Milestone 4: Attach meta-learner for adaptive tuning
        if _state["meta_learner"]:
            csa.set_meta_learner(_state["meta_learner"])

        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=req.beam_width,
            max_hop=req.max_hop,
            max_budget=req.max_budget,
        )
        paths   = traversal.traverse(seeds)
        
        # Pass seed embedding as query signal for semantic re-ranking
        q_emb = adapter.get_embedding(seeds[0]) if seeds else None
        answers = extract(paths, top_k=req.top_k, query_embedding=q_emb)

        # Format response
        structured = to_structured(answers, query=req.query, adapter=adapter)
        path_results = []
        for p, answer in zip(structured["paths"], answers):
            best_path = answer.best_path
            path_results.append(PathResult(
                rank=p["rank"],
                answer_entity=p["answer_entity"],
                score=p["score"],
                score_breakdown=p["score_breakdown"],
                path=[PathNode(**n) for n in p["path"]],
                edge_features=[list(f) for f in best_path.edge_features],
                community_sequence=list(best_path.community_sequence),
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
            csa = CSAEngine(
                adapter=adapter, 
                edge_type_weights=req.edge_type_weights or default_edge_type_weights,
                community_params=csa_meta.get("community_params"),
            )
            csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])
            
            if _state["meta_learner"]:
                csa.set_meta_learner(_state["meta_learner"])

            traversal = AsyncBeamTraversal(
                adapter=adapter,
                csa_engine=csa,
                beam_width=req.beam_width,
                max_hop=req.max_hop,
                max_budget=req.max_budget
            )

            hop_count = 0
            q_emb = adapter.get_embedding(seeds[0]) if seeds else None
            async for hop_paths in traversal.traverse_stream(seeds):
                # Format this hop's paths
                # convert paths to Answers for to_structured
                answers = extract(hop_paths, top_k=req.top_k, min_hop=0, query_embedding=q_emb)
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

    @app.post("/feedback", tags=["reasoning"])
    async def feedback(req: FeedbackRequest, node: Dict = Depends(check_scope("query"))):
        """
        Record user feedback for a specific reasoning path.
        Triggers online parameter adaptation via MetaParameterLearner.
        """
        if not _state["meta_learner"]:
            raise HTTPException(status_code=501, detail="Meta-learning is not enabled")

        from reasoning.traversal import TraversalPath
        
        # Note: We only need edge_features and community_sequence for the update
        dummy_path = TraversalPath(
            nodes=req.path_nodes,
            edge_features=[tuple(f) for f in req.edge_features],
            community_sequence=req.community_sequence
        )
        
        _state["meta_learner"].update_from_feedback(dummy_path, req.reward)

        # Buffer for batch retraining (POST /retrain)
        _state["feedback_buffer"].append({"path": dummy_path, "reward": req.reward})

        return {
            "status": "success",
            "message": "Feedback recorded, model updated",
            "buffer_size": len(_state["feedback_buffer"]),
        }

    @app.get("/params", response_model=ParamsResponse, tags=["reasoning"])
    async def get_params(node: Dict = Depends(check_scope("query"))):
        """
        Return the current learned CSA parameter state.

        Shows the global 10-parameter vector and any per-community overrides
        accumulated via POST /feedback.  Use this to inspect or checkpoint
        the online-learned parameter state.
        """
        from core.parameter_learner import _PARAM_NAMES, _DEFAULT_INIT

        ml = _state["meta_learner"]
        if ml is None:
            global_params = list(_DEFAULT_INIT)
            overrides: Dict[str, list] = {}
        else:
            global_params = [float(v) for v in ml.global_prior]
            overrides = {
                str(cid): [float(v) for v in vec]
                for cid, vec in ml.community_overrides.items()
            }

        return ParamsResponse(
            param_names=list(_PARAM_NAMES),
            global_params=global_params,
            community_count=len(overrides),
            community_overrides=overrides,
        )

    @app.post("/params", response_model=ParamsResponse, tags=["reasoning"])
    async def set_params(req: ParamsImportRequest, node: Dict = Depends(check_scope("query"))):
        """
        Restore a previously exported parameter state.

        Accepts the payload produced by GET /params (or a subset thereof) and
        replaces the running MetaParameterLearner's global prior and community
        overrides.  Use this to:

        - Reload a checkpoint after a server restart.
        - Deploy a batch-trained parameter set (from CSAParameterLearner.fit()).
        - Reset learned parameters to defaults (pass empty community_overrides).
        """
        from core.parameter_learner import MetaParameterLearner, _PARAM_NAMES, _DEFAULT_INIT
        import numpy as np

        ml = _state["meta_learner"]
        if ml is None:
            # Meta-learning not enabled; create a learner to hold the imported state.
            ml = MetaParameterLearner()
            _state["meta_learner"] = ml

        # Validate global_prior length
        if len(req.global_prior) != len(_DEFAULT_INIT):
            raise HTTPException(
                status_code=422,
                detail=f"global_prior must have {len(_DEFAULT_INIT)} elements, got {len(req.global_prior)}",
            )

        ml.global_prior = np.array(req.global_prior, dtype=np.float32)
        ml._n = len(ml.global_prior)
        ml.community_overrides = {
            int(cid): np.array(vec, dtype=np.float32)
            for cid, vec in req.community_overrides.items()
        }
        if req.learning_rate is not None:
            ml.learning_rate = req.learning_rate
        if req.momentum is not None:
            ml.momentum = req.momentum
        # Reset velocity — stale momentum from previous run is misleading.
        ml._velocity = {}

        overrides = {
            str(cid): [float(v) for v in vec]
            for cid, vec in ml.community_overrides.items()
        }
        return ParamsResponse(
            param_names=list(_PARAM_NAMES),
            global_params=[float(v) for v in ml.global_prior],
            community_count=len(overrides),
            community_overrides=overrides,
        )

    @app.post("/retrain", response_model=RetrainResponse, tags=["reasoning"])
    async def retrain(
        req: RetrainRequest = RetrainRequest(),
        node: Dict = Depends(check_scope("query")),
    ):
        """
        Run batch parameter retraining on accumulated feedback pairs.

        Collects all positive-reward and negative-reward paths buffered by
        POST /feedback, cross-pairs them, and runs CSAParameterLearner.fit()
        using the current global prior as the starting point.  The learned
        10-parameter vector replaces MetaParameterLearner.global_prior so that
        all future queries (including communities with no specific override) use
        the improved values.

        Requires at least one positive-reward AND one negative-reward item in
        the buffer; returns 422 if either is absent.
        """
        import random as _random
        from core.parameter_learner import CSAParameterLearner, _PARAM_NAMES

        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        ml = _state["meta_learner"]
        if ml is None:
            raise HTTPException(status_code=501, detail="Meta-learning is not enabled")

        buf = _state["feedback_buffer"]
        positives = [item["path"] for item in buf if item["reward"] > 0]
        negatives = [item["path"] for item in buf if item["reward"] < 0]

        if not positives or not negatives:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Need ≥1 positive and ≥1 negative feedback item. "
                    f"Buffer has {len(positives)} positive, {len(negatives)} negative."
                ),
            )

        # Build cross-product pairs, capped at max_pairs
        all_pairs = [(p, n) for p in positives for n in negatives]
        if len(all_pairs) > req.max_pairs:
            all_pairs = _random.sample(all_pairs, req.max_pairs)

        learner = CSAParameterLearner(
            adapter=_state["adapter"],
            init_params=tuple(float(v) for v in ml.global_prior),
            learning_rate=req.learning_rate,
            max_iterations=req.max_iterations,
        )
        result = learner.fit(all_pairs)

        # Update global prior with learned values; reset stale momentum
        import numpy as np
        ml.global_prior = np.array(result.params, dtype=np.float32)
        ml._velocity = {}

        if req.clear_buffer:
            _state["feedback_buffer"] = []

        learned_params = {
            name: float(val)
            for name, val in zip(_PARAM_NAMES, result.params)
        }
        return RetrainResponse(
            status="success",
            pairs_used=len(all_pairs),
            iterations=result.n_iterations,
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            converged=result.converged,
            learned_params=learned_params,
            buffer_remaining=len(_state["feedback_buffer"]),
        )

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

    @app.post("/search/similar", response_model=SimilarSearchResponse, tags=["graph"])
    async def search_similar(req: SimilarSearchRequest, node: Dict = Depends(check_scope("search"))):
        """
        Perform semantic vector search across the graph.
        Returns entities most similar to the provided embedding vector.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        adapter = _state["adapter"]
        emb_query = np.array(req.embedding, dtype=np.float32)
        
        ents = adapter.find_similar(emb_query, top_k=req.top_k)
        
        return SimilarSearchResponse(
            query_vector=req.embedding,
            results=[
                EntityResponse(
                    id=e.id, 
                    label=e.label, 
                    type=e.type, 
                    properties=e.properties
                ) for e in ents if e
            ]
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
                version="1.2.0",
                capabilities=[],
                entity_types=[],
                relation_types=[],
                node_count=0,
                community_count=0
            )
        
        adapter = _state["adapter"]
        cm = _state["community_map"]
        
        # Infer types from adapter if supported, or sample
        entity_types = set()
        relation_types = set()
        
        G = adapter.to_networkx()
        for _, data in list(G.nodes(data=True))[:100]:
            entity_types.add(data.get("type", "entity"))
        for _, _, data in list(G.edges(data=True))[:100]:
            relation_types.add(data.get("relation", "link"))

        return HandshakeResponse(
            version="1.2.0",
            capabilities=["query", "search", "masked_search", "hologram", "traversal", "reasoning_callback", "feedback", "search_similar"],
            entity_types=list(entity_types),
            relation_types=list(relation_types),
            node_count=adapter.node_count(),
            community_count=len(set(cm.values()))
        )

    @app.post("/traverse", response_model=TraversalBranchResponse, tags=["federated"])
    async def traverse_callback(req: TraversalBranchRequest, node: Dict = Depends(check_scope("query"))):
        """
        Delegate multi-hop reasoning branches to this node.
        Returns serialized TraversalPaths starting from seed_id.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Service not ready")

        from core.attention_engine import CSAEngine
        from reasoning.traversal import BeamTraversal

        adapter      = _state["adapter"]
        csa_meta     = _state["csa_metadata"]
        default_weights = _state["default_edge_type_weights"]

        # 1. Setup local reasoning context
        csa = CSAEngine(
            adapter=adapter,
            edge_type_weights=default_weights,
            community_params=csa_meta.get("community_params")
        )
        csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])
        
        if _state["meta_learner"]:
            csa.set_meta_learner(_state["meta_learner"])

        # 2. Run local beam search
        traversal = BeamTraversal(
            adapter=adapter,
            csa_engine=csa,
            beam_width=req.beam_width,
            max_hop=req.max_hop,
            max_budget=req.max_budget,
        )
        
        # Use context_embedding if provided to seed the traversal
        context_emb = np.array(req.context_embedding, dtype=np.float32) if req.context_embedding else None
        
        # We start a fresh traversal from seed_id
        # Future: allow passing a partial TraversalPath to resume
        paths = traversal.traverse([req.seed_id], query_embedding=context_emb)
        
        # 3. Serialize results
        branches = [p.to_dict() for p in paths if len(p.nodes) > 1] # Only return actual paths
        
        return TraversalBranchResponse(seed_id=req.seed_id, branches=branches)

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
        from llm_bridge.context_formatter import to_structured

        adapter      = _state["adapter"]
        _state["community_map"]
        csa_meta     = _state["csa_metadata"]

        csa = CSAEngine(
            adapter=adapter,
            community_params=csa_meta.get("community_params")
        )
        csa.set_community_graph(csa_meta["distances"], csa_meta["adjacent_pairs"])
        
        if _state["meta_learner"]:
            csa.set_meta_learner(_state["meta_learner"])

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

    # ---------------------------------------------------------------------------
    # Phase 15 — Insight Engine Endpoints
    # ---------------------------------------------------------------------------

    def _get_insight_engine():
        """Lazy-initialize InsightEngine on first call."""
        from core.insight_engine import InsightEngine
        if _state["insight_engine"] is None:
            if not _is_ready():
                raise HTTPException(status_code=503, detail="Service not ready — call load() first")
            _state["insight_engine"] = InsightEngine(_state["adapter"])
        return _state["insight_engine"]

    def _insight_event_schema(ev) -> InsightEventSchema:
        return InsightEventSchema(
            bridging_node=ev.bridging_node,
            source=ev.source,
            target=ev.target,
            insight_score=ev.insight_score,
            explanatory_power=ev.explanatory_power,
            community_leap=ev.community_leap,
            edge_created=ev.edge_created,
            timestamp=ev.timestamp,
            id=getattr(ev, "id", ""),
            validation_status=getattr(ev, "validation_status", "pending"),
            corroboration_count=getattr(ev, "corroboration_count", 0),
        )

    @app.get("/insight/status", response_model=InsightStatusResponse, tags=["insight"])
    async def insight_status(node: Dict = Depends(get_authenticated_node)):
        """
        Return InsightEngine status: ring buffer occupancy, total events fired,
        pause state, and the 10 most recent InsightEvents.
        """
        ie = _get_insight_engine()
        return InsightStatusResponse(
            total_events=ie.total_events,
            ring_buffer_size=ie.buffer_size,
            ring_buffer_capacity=ie.buffer_capacity,
            paused=ie.is_paused,
            recent_events=[_insight_event_schema(ev) for ev in ie.recent_events(10)],
        )

    @app.get("/insight/events", response_model=InsightStatusResponse, tags=["insight"])
    async def insight_events(n: int = 20, node: Dict = Depends(get_authenticated_node)):
        """Return the last n InsightEvents (default 20)."""
        ie = _get_insight_engine()
        return InsightStatusResponse(
            total_events=ie.total_events,
            ring_buffer_size=ie.buffer_size,
            ring_buffer_capacity=ie.buffer_capacity,
            paused=ie.is_paused,
            recent_events=[_insight_event_schema(ev) for ev in ie.recent_events(n)],
        )

    @app.post("/insight/scan", response_model=InsightScanResponse, tags=["insight"])
    async def insight_scan(node: Dict = Depends(check_scope("query"))):
        """
        Trigger an on-demand cold-path community boundary scan.

        Finds high-similarity unconnected node pairs at community edges and
        materializes INSIGHT_LINK edges where warranted. Safe to call at any
        time — no mutations if no insights meet the salience threshold.
        """
        ie = _get_insight_engine()
        events = ie.scan_boundaries()
        return InsightScanResponse(
            events_found=len(events),
            events=[_insight_event_schema(ev) for ev in events],
        )

    # ---------------------------------------------------------------------------
    # Phase 16 — InsightValidator endpoints
    # ---------------------------------------------------------------------------

    def _get_insight_validator():
        """Lazy-initialize InsightValidator on first call."""
        from core.insight_validator import InsightValidator
        if "insight_validator" not in _state:
            _state["insight_validator"] = None
        if _state.get("insight_validator") is None:
            if not _is_ready():
                raise HTTPException(status_code=503, detail="Service not ready — call load() first")
            _state["insight_validator"] = InsightValidator(_state["adapter"])
        return _state["insight_validator"]

    def _get_meta_insight_engine():
        """Lazy-initialize MetaInsightEngine on first call."""
        from core.meta_insight_engine import MetaInsightEngine
        if "meta_insight_engine" not in _state:
            _state["meta_insight_engine"] = None
        if _state.get("meta_insight_engine") is None:
            _state["meta_insight_engine"] = MetaInsightEngine()
        return _state["meta_insight_engine"]

    @app.post("/insight/validate/all", response_model=InsightValidateAllResponse, tags=["insight"])
    async def insight_validate_all(node: Dict = Depends(check_scope("query"))):
        """
        Validate all pending InsightEvents via bilateral reverse traversal and
        multi-seed corroboration. Updates validation_status and corroboration_count
        on each event. Promotes INSIGHT_LINK edge confidence for confirmed insights.
        """
        ie = _get_insight_engine()
        validator = _get_insight_validator()
        events = ie.recent_events(500)
        pending = [ev for ev in events if getattr(ev, "validation_status", "pending") == "pending"]
        validator.validate_all(pending)
        return InsightValidateAllResponse(
            validated=len(pending),
            results=[
                InsightValidateResponse(
                    event_id=ev.id,
                    validation_status=ev.validation_status,
                    corroboration_count=ev.corroboration_count,
                    insight_score=ev.insight_score,
                )
                for ev in pending
            ],
        )

    @app.post("/insight/validate/{event_id}", response_model=InsightValidateResponse, tags=["insight"])
    async def insight_validate_one(event_id: str, node: Dict = Depends(check_scope("query"))):
        """
        Validate a single InsightEvent by its ID. Returns 404 if not found.
        """
        ie = _get_insight_engine()
        validator = _get_insight_validator()
        all_events = ie.recent_events(500)
        target = next((ev for ev in all_events if getattr(ev, "id", "") == event_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail=f"InsightEvent '{event_id}' not found")
        validator.validate(target)
        return InsightValidateResponse(
            event_id=target.id,
            validation_status=target.validation_status,
            corroboration_count=target.corroboration_count,
            insight_score=target.insight_score,
        )

    # ---------------------------------------------------------------------------
    # Phase 16 — MetaInsightEngine endpoints
    # ---------------------------------------------------------------------------

    def _meta_insight_event_schema(ev) -> MetaInsightEventSchema:
        return MetaInsightEventSchema(
            id=ev.id,
            insight_a_id=ev.insight_a_id,
            insight_b_id=ev.insight_b_id,
            connection_type=ev.connection_type,
            meta_score=ev.meta_score,
            depth=ev.depth,
            timestamp=ev.timestamp,
            chain_ids=ev.chain_ids,
        )

    @app.get("/meta-insight/status", response_model=MetaInsightStatusResponse, tags=["meta-insight"])
    async def meta_insight_status(node: Dict = Depends(get_authenticated_node)):
        """
        Return MetaInsightEngine status: total meta-events, InsightGraph size,
        and the 10 most recent MetaInsightEvents.
        """
        meta = _get_meta_insight_engine()
        nodes, edges = meta.insight_graph_size
        return MetaInsightStatusResponse(
            total_meta_events=meta.total_meta_events,
            insight_graph_nodes=nodes,
            insight_graph_edges=edges,
            recent_meta_events=[_meta_insight_event_schema(ev) for ev in meta.recent_meta_events(10)],
        )

    @app.get("/meta-insight/events", response_model=MetaInsightStatusResponse, tags=["meta-insight"])
    async def meta_insight_events(n: int = 20, node: Dict = Depends(get_authenticated_node)):
        """Return the last n MetaInsightEvents (default 20)."""
        meta = _get_meta_insight_engine()
        nodes, edges = meta.insight_graph_size
        return MetaInsightStatusResponse(
            total_meta_events=meta.total_meta_events,
            insight_graph_nodes=nodes,
            insight_graph_edges=edges,
            recent_meta_events=[_meta_insight_event_schema(ev) for ev in meta.recent_meta_events(n)],
        )

    @app.get("/meta-insight/graph", response_model=InsightGraphResponse, tags=["meta-insight"])
    async def meta_insight_graph(node: Dict = Depends(get_authenticated_node)):
        """
        Export the full InsightGraph as nodes and edges.

        The InsightGraph is the second-order reasoning graph where nodes are
        InsightEvent IDs and edges are structural relationships between insights
        (chain, shared_entity, community_overlap, temporal_cluster).
        """
        meta = _get_meta_insight_engine()
        data = meta.export_insight_graph()
        return InsightGraphResponse(
            nodes=[
                InsightGraphNodeSchema(
                    id=n["id"],
                    source_entity=n["source_entity"],
                    target_entity=n["target_entity"],
                    insight_score=n["insight_score"],
                    community_leap=n["community_leap"],
                    timestamp=n["timestamp"],
                )
                for n in data["nodes"]
            ],
            edges=[
                InsightGraphEdgeSchema(
                    from_id=e["from"],
                    to_id=e["to"],
                    connection_type=e["connection_type"],
                    score=e["score"],
                )
                for e in data["edges"]
            ],
        )

    # ---------------------------------------------------------------------------
    # Phase 14 — REM Cycle Endpoints
    # ---------------------------------------------------------------------------

    def _get_rem_engine():
        """Lazy-initialize REMEngine on first call."""
        from core.rem_engine import REMEngine
        if _state["rem_engine"] is None:
            if not _is_ready():
                raise HTTPException(status_code=503, detail="Service not ready — call load() first")
            _state["rem_engine"] = REMEngine(_state["adapter"])
        return _state["rem_engine"]

    @app.post("/rem/run", response_model=REMReportSchema, tags=["rem"])
    async def rem_run(req: REMRunRequest, node: Dict = Depends(check_scope("query"))):
        """
        Execute one REM cycle (prune → consolidate → synthesize).

        Set dry_run=true to preview changes without mutating the graph.
        """
        rem = _get_rem_engine()
        report = rem.run(dry_run=req.dry_run)
        return REMReportSchema(
            pruned_edges=report.pruned_edges,
            synthesized_edges=report.synthesized_edges,
            communities_updated=report.communities_updated,
            duration_seconds=report.duration_seconds,
            pruned_edge_list=[list(t) for t in report.pruned_edge_list],
            synthesized_edge_list=[list(t) for t in report.synthesized_edge_list],
            dry_run=report.dry_run,
            timestamp=report.timestamp,
        )

    @app.post("/rem/rollback", response_model=REMRollbackResponse, tags=["rem"])
    async def rem_rollback(node: Dict = Depends(check_scope("query"))):
        """
        Undo the most recent non-dry-run REM cycle.

        Restores pruned edges with original attributes and removes synthesized
        edges. Only one level of undo is supported (the last real run).
        Returns 409 if no snapshot exists.
        """
        rem = _get_rem_engine()
        if not rem.can_rollback:
            raise HTTPException(status_code=409, detail="No REM cycle to roll back — run a real cycle first.")
        ops = rem.rollback()
        return REMRollbackResponse(
            operations=ops,
            message=f"Rolled back last REM cycle: {ops} edge operation(s) reversed.",
        )

    @app.get("/rem/status", response_model=REMStatusResponse, tags=["rem"])
    async def rem_status(node: Dict = Depends(get_authenticated_node)):
        """Return the last REMReport and whether a rollback is currently available."""
        rem = _get_rem_engine()
        last = rem.last_report
        schema = None
        if last is not None:
            schema = REMReportSchema(
                pruned_edges=last.pruned_edges,
                synthesized_edges=last.synthesized_edges,
                communities_updated=last.communities_updated,
                duration_seconds=last.duration_seconds,
                pruned_edge_list=[list(t) for t in last.pruned_edge_list],
                synthesized_edge_list=[list(t) for t in last.synthesized_edge_list],
                dry_run=last.dry_run,
                timestamp=last.timestamp,
            )
        return REMStatusResponse(last_report=schema, can_rollback=rem.can_rollback)

    # ------------------------------------------------------------------
    # Inference endpoints  (/infer/*)
    # ------------------------------------------------------------------

    def _get_infer_engine():
        from core.inference_engine import TransitiveInferenceEngine
        if _state["infer_engine"] is None:
            _state["infer_engine"] = TransitiveInferenceEngine(_state["adapter"])
        return _state["infer_engine"]

    def _infer_report_to_schema(report) -> InferenceReportSchema:
        return InferenceReportSchema(
            proposal_count=report.proposal_count,
            materialized=report.materialized,
            rules_applied=report.rules_applied,
            skipped_existing=report.skipped_existing,
            duration_seconds=report.duration_seconds,
            dry_run=report.dry_run,
            timestamp=report.timestamp,
            proposals=[
                InferenceProposalSchema(
                    source=p.source,
                    via=p.via,
                    target=p.target,
                    derived_relation=p.derived_relation,
                    confidence=p.confidence,
                    domain=p.rule.domain,
                    note=p.rule.note,
                    derivation=p.derivation_str,
                )
                for p in report.proposals
            ],
        )

    @app.post("/infer/run", response_model=InferenceReportSchema, tags=["inference"])
    async def run_inference(
        dry_run: bool = False,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Run one transitive inference cycle.

        Discovers new knowledge by composing existing graph relations
        according to domain-specific rules.  Returns a full report of
        proposed (and optionally materialized) edges.

        Set ``dry_run=true`` to inspect discoveries without modifying the graph.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        infer = _get_infer_engine()
        try:
            report = infer.run(dry_run=dry_run)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return _infer_report_to_schema(report)

    @app.post("/infer/rollback", response_model=InferenceRollbackResponse, tags=["inference"])
    async def infer_rollback(node: Dict = Depends(get_authenticated_node)):
        """
        Undo the most recent non-dry inference run.

        Removes all edges that were added during that cycle.
        Raises 409 if no prior run exists.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        infer = _get_infer_engine()
        try:
            removed = infer.rollback()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return InferenceRollbackResponse(
            removed=removed,
            message=f"Rolled back {removed} inferred edges.",
        )

    @app.get("/infer/status", response_model=InferenceStatusResponse, tags=["inference"])
    async def infer_status(node: Dict = Depends(get_authenticated_node)):
        """Return the last InferenceReport and whether rollback is available."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        infer = _get_infer_engine()
        last = infer.last_report
        schema = _infer_report_to_schema(last) if last is not None else None
        return InferenceStatusResponse(
            last_report=schema,
            can_rollback=infer.can_rollback,
            active_rule_count=infer.rule_count(),
        )

    # ------------------------------------------------------------------
    # Hypothesis endpoints  (/hypothesize/*)
    # ------------------------------------------------------------------

    def _get_hypothesis_engine():
        from core.hypothesis_engine import HypothesisEngine
        if _state["hypothesis_engine"] is None:
            _state["hypothesis_engine"] = HypothesisEngine(
                adapter=_state["adapter"],
                csa_metadata=_state["csa_metadata"],
            )
        return _state["hypothesis_engine"]

    def _proposal_to_schema(prop) -> HypothesisProposalSchema:
        return HypothesisProposalSchema(
            hypothesis_id=prop.hypothesis_id,
            source=prop.source,
            target=prop.target,
            derived_relation=prop.derived_relation,
            confidence=prop.confidence,
            path_count=prop.path_count,
            independence_scores=prop.independence_scores,
            contradiction_score=prop.contradiction_score,
            derivation_text=prop.derivation_text,
            intersection_nodes=prop.intersection_nodes,
            supporting_paths=[
                TraversalPathSchema(
                    nodes=p.nodes,
                    score=p.score,
                    attention_weights=p.attention_weights,
                    community_sequence=p.community_sequence,
                    edge_confidences=p.edge_confidences,
                    edge_provenances=p.edge_provenances,
                    edge_features=[list(f) for f in p.edge_features],
                    beta_alpha=p.beta_alpha,
                    beta_beta=p.beta_beta,
                )
                for p in prop.supporting_paths
            ],
        )

    @app.post("/hypothesize", response_model=HypothesizeResponse, tags=["hypothesis"])
    async def hypothesize(
        req: HypothesizeRequest,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Find multi-hop paths between source_id and target_id, compose
        relation chains, and return hypothesis proposals backed by
        Noisy-OR combined evidence (equifinality).

        Set ``auto_materialize=true`` to immediately write accepted
        proposals to the graph.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        if req.source_id == req.target_id:
            raise HTTPException(status_code=422, detail="source_id and target_id must be different")

        adapter = _state["adapter"]
        if adapter.get_entity(req.source_id) is None:
            raise HTTPException(status_code=404, detail=f"Source entity not found: {req.source_id!r}")
        if adapter.get_entity(req.target_id) is None:
            raise HTTPException(status_code=404, detail=f"Target entity not found: {req.target_id!r}")

        engine = _get_hypothesis_engine()
        t0 = __import__("time").time()

        try:
            proposals = engine.generate(
                source_id=req.source_id,
                target_id=req.target_id,
                max_paths=req.max_paths,
                max_hop=req.max_hop,
                beam_width=req.beam_width,
                max_budget=req.max_budget,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        if req.auto_materialize and proposals:
            filtered = [p for p in proposals if p.confidence >= req.min_confidence]
            engine.materialize(filtered)

        return HypothesizeResponse(
            source_id=req.source_id,
            target_id=req.target_id,
            proposals=[_proposal_to_schema(p) for p in proposals],
            paths_explored=len(proposals),
            duration_seconds=__import__("time").time() - t0,
        )

    @app.post(
        "/hypothesize/materialize",
        response_model=HypothesisMaterializeResponse,
        tags=["hypothesis"],
    )
    async def hypothesize_materialize(
        req: HypothesisMaterializeRequest,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Materialize proposals from the most recent /hypothesize run.

        Pass ``hypothesis_ids`` to select specific proposals; omit (or pass
        an empty list) to materialize all proposals above ``min_confidence``.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        engine = _get_hypothesis_engine()
        last = engine.last_proposals
        if not last:
            raise HTTPException(status_code=409, detail="No proposals available. Call POST /hypothesize first.")

        if req.hypothesis_ids:
            id_set = set(req.hypothesis_ids)
            to_mat = [p for p in last if p.hypothesis_id in id_set and p.confidence >= req.min_confidence]
        else:
            to_mat = [p for p in last if p.confidence >= req.min_confidence]

        try:
            added = engine.materialize(to_mat)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return HypothesisMaterializeResponse(materialized=len(to_mat), edges_added=added)

    @app.post(
        "/hypothesize/rollback",
        response_model=HypothesisMaterializeResponse,
        tags=["hypothesis"],
    )
    async def hypothesize_rollback(node: Dict = Depends(get_authenticated_node)):
        """
        Remove all edges added by the most recent materialize call.
        Raises 409 if nothing to roll back.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        engine = _get_hypothesis_engine()
        try:
            removed = engine.rollback()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return HypothesisMaterializeResponse(materialized=0, edges_added=-removed)

    @app.get(
        "/hypothesize/status",
        response_model=HypothesisStatusResponse,
        tags=["hypothesis"],
    )
    async def hypothesize_status(node: Dict = Depends(get_authenticated_node)):
        """Return the last hypothesis run info and rollback availability."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        engine = _get_hypothesis_engine()
        last = engine.last_proposals or []
        return HypothesisStatusResponse(
            last_source=engine.last_source,
            last_target=engine.last_target,
            proposal_count=len(last),
            can_rollback=engine.can_rollback,
            materialized_count=engine.materialized_count,
        )

    # ------------------------------------------------------------------
    # Research endpoints  (/research/*)
    # ------------------------------------------------------------------

    def _get_research_agent():
        from core.research_agent import ResearchAgent
        if _state["research_agent"] is None:
            if not _is_ready():
                raise HTTPException(status_code=503, detail="Graph not loaded")
            _state["research_agent"] = ResearchAgent(
                adapter=_state["adapter"],
                hypothesis_engine=_get_hypothesis_engine(),
                insight_engine=_state.get("insight_engine"),
            )
        return _state["research_agent"]

    def _get_external_validator():
        from core.external_validator import ExternalValidator
        if _state["external_validator"] is None:
            _state["external_validator"] = ExternalValidator()
        return _state["external_validator"]

    def _finding_to_schema(f) -> ResearchFindingSchema:
        cand = f.candidate
        return ResearchFindingSchema(
            finding_id=f.finding_id,
            candidate=ResearchCandidateSchema(
                source_id=cand.source_id,
                target_id=cand.target_id,
                discovery_potential=cand.discovery_potential,
                gap_score=cand.gap_score,
                community_distance=cand.community_distance,
                seeded_by=cand.seeded_by,
                local_density=cand.local_density,
                created_at=cand.created_at,
            ),
            proposals=[_proposal_to_schema(p) for p in f.proposals],
            best_confidence=f.best_confidence,
            literature_status=f.literature_status,
            found_at=f.found_at,
        )

    def _validation_report_to_schema(r) -> ValidationReportSchema:
        return ValidationReportSchema(
            hypothesis_id=r.hypothesis_id,
            source_id=r.source_id,
            target_id=r.target_id,
            derived_relation=r.derived_relation,
            literature_status=r.literature_status,
            novelty_score=r.novelty_score,
            hit_count=r.hit_count,
            hits=[
                LiteratureHitSchema(
                    adapter=h.adapter,
                    external_id=h.external_id,
                    title=h.title,
                    year=h.year,
                    relevance_score=h.relevance_score,
                )
                for h in r.hits
            ],
            adapters_queried=r.adapters_queried,
            checked_at=r.checked_at,
            error=r.error,
        )

    @app.get("/research/status", response_model=ResearchStatusResponse, tags=["research"])
    async def research_status(node: Dict = Depends(get_authenticated_node)):
        """Return ResearchAgent running state and scan statistics."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        s = agent.status
        return ResearchStatusResponse(**s)

    @app.post("/research/start", response_model=ResearchStatusResponse, tags=["research"])
    async def research_start(node: Dict = Depends(get_authenticated_node)):
        """Start the background scanning daemon (idempotent)."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        agent.start()
        return ResearchStatusResponse(**agent.status)

    @app.post("/research/stop", response_model=ResearchStatusResponse, tags=["research"])
    async def research_stop(node: Dict = Depends(get_authenticated_node)):
        """Stop the background scanning daemon."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        agent.stop()
        return ResearchStatusResponse(**agent.status)

    @app.post("/research/scan", response_model=ResearchScanResponse, tags=["research"])
    async def research_scan(node: Dict = Depends(get_authenticated_node)):
        """Trigger an immediate one-shot scan and return any new findings."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        t0 = __import__("time").time()
        findings = agent.scan_once()
        return ResearchScanResponse(
            findings=[_finding_to_schema(f) for f in findings],
            candidates_evaluated=len(findings),
            duration_seconds=__import__("time").time() - t0,
        )

    @app.get("/research/proposals", response_model=ResearchScanResponse, tags=["research"])
    async def research_proposals(node: Dict = Depends(get_authenticated_node)):
        """List all pending findings in the ring buffer."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        findings = agent.findings
        return ResearchScanResponse(
            findings=[_finding_to_schema(f) for f in findings],
            candidates_evaluated=len(findings),
            duration_seconds=0.0,
        )

    @app.post(
        "/research/approve/{finding_id}",
        response_model=ResearchApproveResponse,
        tags=["research"],
    )
    async def research_approve(
        finding_id: str,
        node: Dict = Depends(get_authenticated_node),
    ):
        """Materialize all proposals in a finding to the graph."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        try:
            edges_added = agent.approve(finding_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return ResearchApproveResponse(finding_id=finding_id, edges_added=edges_added)

    @app.post(
        "/research/reject/{finding_id}",
        response_model=ResearchRejectResponse,
        tags=["research"],
    )
    async def research_reject(
        finding_id: str,
        node: Dict = Depends(get_authenticated_node),
    ):
        """Discard a finding from the ring buffer."""
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        agent = _get_research_agent()
        try:
            agent.reject(finding_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return ResearchRejectResponse(finding_id=finding_id, removed=True)

    @app.post(
        "/research/validate",
        response_model=ValidateProposalsResponse,
        tags=["research"],
    )
    async def research_validate(
        req: ValidateProposalsRequest,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Validate hypothesis proposals against external literature databases
        (PubMed, ClinicalTrials.gov, arXiv, OpenAlex).

        Pass ``hypothesis_ids`` to target specific proposals; omit to validate
        all proposals from the most recent /hypothesize run.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        validator = _get_external_validator()
        hyp_engine = _get_hypothesis_engine()
        last = hyp_engine.last_proposals or []

        if req.hypothesis_ids:
            id_set = set(req.hypothesis_ids)
            to_validate = [p for p in last if p.hypothesis_id in id_set]
        else:
            to_validate = list(last)

        if not to_validate:
            return ValidateProposalsResponse(validated=0, reports=[], duration_seconds=0.0)

        t0 = __import__("time").time()
        try:
            reports = validator.validate_batch(to_validate)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return ValidateProposalsResponse(
            validated=len(reports),
            reports=[_validation_report_to_schema(r) for r in reports],
            duration_seconds=__import__("time").time() - t0,
        )

    @app.get(
        "/research/validate/{hypothesis_id}",
        response_model=ValidationReportSchema,
        tags=["research"],
    )
    async def research_validate_one(
        hypothesis_id: str,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Get the cached validation report for a specific proposal.
        Returns 404 if the proposal has not been validated yet.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        hyp_engine = _get_hypothesis_engine()
        last = hyp_engine.last_proposals or []
        proposal = next((p for p in last if p.hypothesis_id == hypothesis_id), None)
        if proposal is None:
            raise HTTPException(status_code=404, detail=f"Proposal {hypothesis_id!r} not found")

        validator = _get_external_validator()
        # Check cache only — do not trigger a new network call
        cache_key = (proposal.source, proposal.derived_relation, proposal.target)
        with validator._lock:
            cached = validator._cache.get(cache_key)
        if cached is None:
            raise HTTPException(
                status_code=404,
                detail="No validation report found. Call POST /research/validate first.",
            )
        _, report = cached
        return _validation_report_to_schema(report)

    # ------------------------------------------------------------------
    # Chat endpoints  (/chat/*)
    # ------------------------------------------------------------------

    _SESSION_TTL = 1800.0  # 30 minutes

    def _get_chat_manager():
        from core.conversation import ConversationManager
        from reasoning.traversal import BeamTraversal
        if _state["chat_manager"] is None:
            meta = _state["csa_metadata"]
            from core.attention_engine import CSAEngine
            csa = CSAEngine(
                adapter=_state["adapter"],
                community_params=meta.get("community_params")
            )
            csa.set_community_graph(meta["distances"], meta["adjacent_pairs"])
            
            if _state["meta_learner"]:
                csa.set_meta_learner(_state["meta_learner"])

            trav = BeamTraversal(
                adapter=_state["adapter"],
                csa_engine=csa,
                beam_width=10,
                max_hop=3,
            )
            from core.embedding_engine import RandomEngine
            eng = getattr(_state["adapter"], "_embedding_engine", None) \
                  or RandomEngine(dim=64)
            _state["chat_manager"] = ConversationManager(
                adapter=_state["adapter"],
                embedding_engine=eng,
                csa_engine=csa,
                beam_traversal=trav,
            )
        return _state["chat_manager"]

    def _purge_stale_sessions():
        now = time.time()
        stale = [sid for sid, sess in _state["chat_sessions"].items()
                 if now - sess.last_active > _SESSION_TTL]
        for sid in stale:
            del _state["chat_sessions"][sid]

    def _turn_to_schema(turn) -> ConversationTurnSchema:
        return ConversationTurnSchema(
            turn_number=turn.turn_number,
            raw_question=turn.raw_question,
            resolved_question=turn.resolved_question,
            seed_entity=turn.seed_entity,
            seed_entity_label=turn.seed_entity_label,
            answer_text=turn.answer_text,
            new_entities=turn.new_entities,
            is_followup=turn.is_followup,
            focus_shift=turn.focus_shift,
            clarification_needed=turn.clarification_needed,
            clarification_options=[[c[0], c[1]] for c in turn.clarification_options],
            knowledge_gap=turn.knowledge_gap,
            knowledge_gap_hint=turn.knowledge_gap_hint,
            hop_hint=turn.hop_hint,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["conversation"])
    async def chat(
        request: ChatRequest,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Send one conversational turn to CEREBRUM.

        Omit ``session_id`` to start a new session.  Include it in subsequent
        requests to continue the conversation.  Sessions expire after 30 minutes
        of inactivity.

        The response includes the full verbalized answer plus session metadata
        (focus entity, entity trail, turn count) so clients can render context.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")

        _purge_stale_sessions()
        manager = _get_chat_manager()

        # Get or create session
        sid = request.session_id
        if sid and sid in _state["chat_sessions"]:
            session = _state["chat_sessions"][sid]
        else:
            session = manager.new_session(sid)
            _state["chat_sessions"][session.session_id] = session
            sid = session.session_id

        try:
            turn = manager.process(request.question, session)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        entity_trail = [
            session.entity_label_map.get(e, e)
            for e in session.entity_history[-10:]
        ]

        return ChatResponse(
            session_id=sid,
            turn=_turn_to_schema(turn),
            focus_entity=session.focus_entity,
            focus_entity_label=session.focus_entity_label,
            entity_trail=entity_trail,
            turn_count=session.turn_count,
        )

    @app.post("/chat/reset", response_model=ChatResetResponse, tags=["conversation"])
    async def chat_reset(
        session_id: str,
        node: Dict = Depends(get_authenticated_node),
    ):
        """Reset (clear context of) an active session without deleting it."""
        if session_id not in _state["chat_sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        _state["chat_sessions"][session_id].reset()
        return ChatResetResponse(
            session_id=session_id,
            message="Session context cleared.",
        )

    # ------------------------------------------------------------------
    # Text ingest endpoints  (/ingest/*)
    # ------------------------------------------------------------------

    def _get_text_ingestor(min_confidence: float, create_new: bool):
        from core.text_ingestor import TextIngestor
        # Always create fresh ingestor so per-request options are respected
        return TextIngestor(
            _state["adapter"],
            min_confidence=min_confidence,
            create_new_entities=create_new,
        )

    def _report_to_schema(report) -> IngestReportSchema:
        return IngestReportSchema(
            text_length=report.text_length,
            sentences_processed=report.sentences_processed,
            entities_found=report.entities_found,
            entities_linked=report.entities_linked,
            entities_new=report.entities_new,
            triples_extracted=report.triples_extracted,
            triples_accepted=report.triples_accepted,
            triples_skipped_duplicate=report.triples_skipped_duplicate,
            triples_skipped_low_confidence=report.triples_skipped_low_confidence,
            edges_added=report.edges_added,
            nodes_added=report.nodes_added,
            added_triples=[
                IngestTripleSchema(source=s, relation=r, target=d, confidence=c)
                for s, r, d, c in report.added_triples
            ],
            duration_seconds=report.duration_seconds,
            provenance=report.provenance,
            dry_run=report.dry_run,
            timestamp=report.timestamp,
        )

    @app.post("/ingest/text", response_model=IngestReportSchema, tags=["ingest"])
    async def ingest_text(
        request: IngestTextRequest,
        node: Dict = Depends(get_authenticated_node),
    ):
        """
        Extract triples from plain text and add them to the graph.

        CEREBRUM scans the text for entity mentions (known graph entities and
        capitalized noun phrases) and infers relations from verb phrases between
        them.  Every edge added carries ``provenance="text_ingest"`` and the
        source sentence for full traceability.

        Set ``dry_run=true`` to inspect what would be extracted without
        modifying the graph.
        """
        if not _is_ready():
            raise HTTPException(status_code=503, detail="Graph not loaded")
        ingestor = _get_text_ingestor(request.min_confidence, request.create_new_entities)
        try:
            report = ingestor.ingest_text(request.text, dry_run=request.dry_run)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return _report_to_schema(report)

    return app


def create_stream_app(
    stream_adapter=None,
    time_window_seconds: float = 60.0,
    max_edges: int = 10_000,
    **kwargs,
) -> "FastAPI":
    """
    Convenience factory for a streaming-enabled CEREBRUM API.

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
    use_meta_learning: bool = True,
):
    """Load graph state into the global _state dict."""
    from core.community_engine import best_of_n_dscf, hierarchical_dscf
    from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
    from core.persistence import is_state_cached, load_state, save_state
    from core.parameter_learner import MetaParameterLearner

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
            if use_meta_learning:
                _state["meta_learner"] = MetaParameterLearner()
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
        ent = adapter.get_entity(node)
        entity_labels[node] = ent.label if ent else node
    
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
    
    # 5. Milestone 4: Adaptive Parameter Learning
    if use_meta_learning:
        _state["meta_learner"] = MetaParameterLearner()

    # 6. Save to cache
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


def _unload():
    """Clear global state."""
    for k in _state:
        if isinstance(_state[k], dict):
            _state[k] = {}
        else:
            _state[k] = None

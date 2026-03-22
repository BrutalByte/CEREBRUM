"""Pydantic schemas for the Parallax REST API."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query or entity ID")
    seeds: List[str] = Field(default=[], description="Optional explicit seed entity IDs")
    top_k: int = Field(default=5, ge=1, le=50)
    max_hop: int = Field(default=3, ge=1, le=6)
    beam_width: int = Field(default=10, ge=1, le=50) # Capped at 50 for security
    max_budget: int = Field(default=1000, ge=10, le=5000, description="Max neighbor expansions allowed")
    edge_types: Optional[List[str]] = Field(default=None, description="Filter traversal to these relation types")
    edge_type_weights: Optional[Dict[str, float]] = Field(default=None, description="Bridge Bonus: {relation_type -> weight}")


class PathNode(BaseModel):
    type: str       # "entity" or "relation"
    id: Optional[str] = None
    label: str
    community: Optional[int] = None


class PathResult(BaseModel):
    rank: int
    answer_entity: str
    score: float
    score_breakdown: Dict[str, float]
    path: List[PathNode]


class QueryResponse(BaseModel):
    query: str
    seeds_used: List[str]
    paths: List[PathResult]
    total_paths_explored: int


class CommunityInfo(BaseModel):
    community_id: int
    size: int
    sample_members: List[str]


class CommunitiesResponse(BaseModel):
    node_to_community: Dict[str, int]
    community_count: int
    node_count: int
    communities: List[CommunityInfo]


class HealthResponse(BaseModel):
    status: str
    adapter_loaded: bool
    communities_loaded: bool
    embeddings_loaded: bool
    node_count: int = 0
    community_count: int = 0


class EntityResponse(BaseModel):
    id: str
    label: str
    type: str = "entity"
    properties: Dict[str, Any] = {}


class EdgeResponse(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: List[EntityResponse]
    query: str


class CommunityResponse(BaseModel):
    entity_id: str
    community_id: int


class EmbeddingResponse(BaseModel):
    entity_id: str
    embedding: List[float]


class MaskedEntityResponse(BaseModel):
    id: str
    type: str = "entity"
    score: float
    label: str = "[REDACTED]"
    properties: Dict[str, Any] = Field(default_factory=dict, description="Redacted in masked mode")


class MaskedSearchResponse(BaseModel):
    query: str
    results: List[MaskedEntityResponse]


class CommunitySignatureSchema(BaseModel):
    community_id: int
    centroid: List[float]
    bloom_hex: str
    bloom_cap: int
    bloom_err: float
    size: int


class HologramResponse(BaseModel):
    adapter_name: str
    signatures: List[CommunitySignatureSchema]


class HandshakeResponse(BaseModel):
    version: str
    capabilities: List[str]
    entity_types: List[str]
    relation_types: List[str]
    node_count: int
    community_count: int


class ReasoningCallbackRequest(BaseModel):
    source_id: str
    target_id: str
    max_hop: int = 2
    max_budget: int = 500


class ReasoningCallbackResponse(BaseModel):
    found: bool
    paths: List[PathResult]


# ---------------------------------------------------------------------------
# Phase 11 — Streaming schemas
# ---------------------------------------------------------------------------

class StreamEventRequest(BaseModel):
    """A single streaming event pushed to /stream/ingest."""
    source: str = Field(..., description="Source entity ID")
    relation: str = Field(default="RELATED_TO", description="Edge relation type")
    target: str = Field(..., description="Target entity ID")
    timestamp: Optional[float] = Field(default=None, description="Unix timestamp; defaults to server time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary event metadata")
    ttl: float = Field(default=0.0, description="Time-to-live in seconds; 0 = window default; -1 = permanent")


class StreamIngestRequest(BaseModel):
    """Batch of events for POST /stream/ingest."""
    events: List[StreamEventRequest] = Field(..., description="One or more StreamEvents to ingest")


class StreamIngestResponse(BaseModel):
    ingested: int
    nodes: int
    edges: int
    communities: int


class StreamStatusResponse(BaseModel):
    running: bool
    nodes: int
    edges: int
    communities: int
    buffer_size: int
    sources: int
    events_per_second: float
    total_ingested: int
    total_evicted: int
    total_community_updates: int




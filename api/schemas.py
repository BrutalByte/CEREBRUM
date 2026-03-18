"""Pydantic schemas for the Parallax REST API."""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query or entity ID")
    seeds: List[str] = Field(default=[], description="Optional explicit seed entity IDs")
    top_k: int = Field(default=5, ge=1, le=50)
    max_hop: int = Field(default=3, ge=1, le=6)
    beam_width: int = Field(default=10, ge=1, le=100)
    edge_types: Optional[List[str]] = Field(default=None, description="Filter traversal to these relation types")


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

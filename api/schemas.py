"""Pydantic schemas for the CEREBRUM REST API."""
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
    edge_features: List[List[float]] = Field(
        default_factory=list,
        description=(
            "Per-hop 10-element feature vectors "
            "(sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding). "
            "Pass directly to POST /feedback to enable online parameter learning."
        ),
    )
    community_sequence: List[int] = Field(
        default_factory=list,
        description="Community ID for each entity node along this path (required for /feedback).",
    )


class QueryResponse(BaseModel):
    query: str
    seeds_used: List[str]
    paths: List[PathResult]
    total_paths_explored: int
    partial: bool = False
    error: Optional[str] = None


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


class SimilarSearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="Query vector for semantic search")
    top_k: int = Field(default=10, ge=1, le=100)


class SimilarSearchResponse(BaseModel):
    query_vector: List[float]
    results: List[EntityResponse]


class FeedbackRequest(BaseModel):
    path_nodes: List[str] = Field(..., description="The sequence of nodes in the path")
    edge_features: List[List[float]] = Field(
        ...,
        description=(
            "Per-hop 10-element feature vectors "
            "(sim, cs, etw, nd, hd, pr_v, td, nr_v, sd, grounding). "
            "Use the edge_features field from POST /query PathResult directly."
        ),
    )
    community_sequence: List[int] = Field(..., description="Community ID for each entity node in the path")
    reward: float = Field(..., description="Feedback value: 1.0 for helpful, -1.0 for noise")


class ParamsResponse(BaseModel):
    param_names: List[str] = Field(description="Names of the 10 CSA parameters (alpha … theta)")
    global_params: List[float] = Field(description="Current global parameter vector")
    community_count: int = Field(description="Number of communities with learned overrides")
    community_overrides: Dict[str, List[float]] = Field(
        description="Per-community parameter overrides {community_id -> [alpha…theta]}",
    )


class ParamsImportRequest(BaseModel):
    """Body for POST /params — restore a previously exported parameter state."""

    global_prior: List[float] = Field(
        ...,
        description="10-element global parameter vector (alpha … theta).",
    )
    community_overrides: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Per-community overrides to restore {community_id -> [alpha…theta]}.",
    )
    learning_rate: Optional[float] = Field(
        default=None,
        description="Override the learning rate (leave null to keep current).",
    )
    momentum: Optional[float] = Field(
        default=None,
        description="Override momentum (leave null to keep current).",
    )


class RetrainRequest(BaseModel):
    """Optional body for POST /retrain."""

    max_pairs: int = Field(
        default=500,
        ge=1,
        description="Maximum number of (positive, negative) training pairs to sample.",
    )
    max_iterations: int = Field(
        default=200,
        ge=1,
        description="Iteration cap for CSAParameterLearner.fit().",
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        description="Gradient descent step size for this retrain run.",
    )
    clear_buffer: bool = Field(
        default=True,
        description="Clear the feedback buffer after retraining (default True).",
    )


class RetrainResponse(BaseModel):
    """Result of POST /retrain."""

    status: str
    pairs_used: int = Field(description="Number of (positive, negative) pairs used.")
    iterations: int
    initial_loss: float
    final_loss: float
    converged: bool
    learned_params: Dict[str, float] = Field(
        description="New global parameter values {param_name -> value}.",
    )
    buffer_remaining: int = Field(
        description="Feedback items remaining in buffer after this call.",
    )


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


class TraversalPathSchema(BaseModel):
    """Serialized TraversalPath for federated reasoning."""
    nodes: List[str]
    score: float
    embedding: Optional[List[float]] = None
    attention_weights: List[float] = []
    community_sequence: List[int] = []
    edge_confidences: List[float] = []
    edge_provenances: List[str] = []
    edge_features: List[List[float]] = []
    beta_alpha: float = 1.0
    beta_beta: float = 1.0


class TraversalBranchRequest(BaseModel):
    seed_id: str
    context_embedding: Optional[List[float]] = None
    max_hop: int = Field(default=2, ge=1, le=3)
    beam_width: int = Field(default=5, ge=1, le=20)
    max_budget: int = Field(default=500, ge=10, le=2000)


class TraversalBranchResponse(BaseModel):
    seed_id: str
    branches: List[TraversalPathSchema]


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


class BridgeRecordSchema(BaseModel):
    """One active bridge twin relationship."""
    original_id: str
    twin_id: str
    source_community: int
    destination_community: int
    traversal_count: int
    age_days: float
    idle_days: float
    similarity_at_creation: float


class BridgesResponse(BaseModel):
    """Response for GET /bridges."""
    total: int
    bridges: List[BridgeRecordSchema]


# ---------------------------------------------------------------------------
# Phase 14 — REM Cycle schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 15 — Insight Engine schemas
# ---------------------------------------------------------------------------

class InsightEventSchema(BaseModel):
    bridging_node: str
    source: str
    target: str
    insight_score: float
    explanatory_power: float
    community_leap: int
    edge_created: bool
    timestamp: float
    # Phase 16 — validation fields
    id: str = ""
    validation_status: str = "pending"
    corroboration_count: int = 0


class InsightStatusResponse(BaseModel):
    total_events: int
    ring_buffer_size: int
    ring_buffer_capacity: int
    paused: bool
    recent_events: List[InsightEventSchema]


class InsightScanResponse(BaseModel):
    events_found: int
    events: List[InsightEventSchema]


# ---------------------------------------------------------------------------
# Phase 16 — InsightValidator schemas
# ---------------------------------------------------------------------------

class InsightValidateResponse(BaseModel):
    event_id: str
    validation_status: str
    corroboration_count: int
    insight_score: float


class InsightValidateAllResponse(BaseModel):
    validated: int
    results: List[InsightValidateResponse]


# ---------------------------------------------------------------------------
# Phase 16 — MetaInsightEngine schemas
# ---------------------------------------------------------------------------

class MetaInsightEventSchema(BaseModel):
    id: str
    insight_a_id: str
    insight_b_id: str
    connection_type: str
    meta_score: float
    depth: int
    timestamp: float
    chain_ids: List[str]


class MetaInsightStatusResponse(BaseModel):
    total_meta_events: int
    insight_graph_nodes: int
    insight_graph_edges: int
    recent_meta_events: List[MetaInsightEventSchema]


class InsightGraphNodeSchema(BaseModel):
    id: str
    source_entity: str
    target_entity: str
    insight_score: float
    community_leap: int
    timestamp: float


class InsightGraphEdgeSchema(BaseModel):
    from_id: str
    to_id: str
    connection_type: str
    score: float


class InsightGraphResponse(BaseModel):
    nodes: List[InsightGraphNodeSchema]
    edges: List[InsightGraphEdgeSchema]


# ---------------------------------------------------------------------------
# Phase 14 — REM Cycle schemas
# ---------------------------------------------------------------------------

class REMRunRequest(BaseModel):
    dry_run: bool = Field(
        default=False,
        description="If true, return the report without mutating the graph.",
    )


class REMReportSchema(BaseModel):
    pruned_edges: int
    synthesized_edges: int
    communities_updated: bool
    duration_seconds: float
    pruned_edge_list: List[List[str]]    # [[source, target, relation], ...]
    synthesized_edge_list: List[List[str]]
    dry_run: bool
    timestamp: float


class REMStatusResponse(BaseModel):
    last_report: Optional[REMReportSchema]
    can_rollback: bool


class REMRollbackResponse(BaseModel):
    operations: int
    message: str


# ---------------------------------------------------------------------------
# Inference engine schemas
# ---------------------------------------------------------------------------

class InferenceProposalSchema(BaseModel):
    source: str
    via: str
    target: str
    derived_relation: str
    confidence: float
    domain: str
    note: str
    derivation: str


class InferenceReportSchema(BaseModel):
    proposal_count: int
    materialized: int
    rules_applied: Dict[str, int]
    skipped_existing: int
    duration_seconds: float
    dry_run: bool
    timestamp: float
    proposals: List[InferenceProposalSchema]


class InferenceStatusResponse(BaseModel):
    last_report: Optional[InferenceReportSchema]
    can_rollback: bool
    active_rule_count: int


class InferenceRollbackResponse(BaseModel):
    removed: int
    message: str


# ---------------------------------------------------------------------------
# Conversation schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    """Omit to start a new session; provide to continue an existing one."""
    question: str


class ConversationTurnSchema(BaseModel):
    turn_number: int
    raw_question: str
    resolved_question: str
    seed_entity: Optional[str]
    seed_entity_label: str
    answer_text: str
    new_entities: List[str]
    is_followup: bool
    focus_shift: bool
    clarification_needed: bool
    clarification_options: List[List[str]]   # [[id, label], ...]
    knowledge_gap: bool
    knowledge_gap_hint: str
    hop_hint: int


class ChatResponse(BaseModel):
    session_id: str
    turn: ConversationTurnSchema
    focus_entity: Optional[str]
    focus_entity_label: str
    entity_trail: List[str]    # human-readable labels of entities mentioned so far
    turn_count: int


class ChatResetResponse(BaseModel):
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# Text ingest schemas
# ---------------------------------------------------------------------------

class IngestTextRequest(BaseModel):
    text: str
    dry_run: bool = False
    min_confidence: float = 0.30
    create_new_entities: bool = True


class IngestTripleSchema(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float


class IngestReportSchema(BaseModel):
    text_length: int
    sentences_processed: int
    entities_found: int
    entities_linked: int
    entities_new: int
    triples_extracted: int
    triples_accepted: int
    triples_skipped_duplicate: int
    triples_skipped_low_confidence: int
    edges_added: int
    nodes_added: int
    added_triples: List[IngestTripleSchema]
    duration_seconds: float
    provenance: str
    dry_run: bool
    timestamp: float


# ---------------------------------------------------------------------------
# Phase 50 — HypothesisEngine schemas
# ---------------------------------------------------------------------------

class HypothesisProposalSchema(BaseModel):
    """A single proposed edge from multi-path abductive reasoning."""
    hypothesis_id: str
    source: str
    target: str
    derived_relation: str
    confidence: float
    path_count: int
    independence_scores: List[float]
    contradiction_score: float
    derivation_text: str
    supporting_paths: List[TraversalPathSchema]
    intersection_nodes: List[str] = Field(
        default_factory=list,
        description="Intermediate nodes appearing in ≥2 independent paths (equifinality hubs).",
    )


class HypothesizeRequest(BaseModel):
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    max_paths: int = Field(default=10, ge=1, le=50, description="Max target-reaching paths to evaluate")
    max_hop: int = Field(default=3, ge=1, le=5)
    beam_width: int = Field(default=10, ge=1, le=30)
    max_budget: int = Field(default=500, ge=10, le=2000)
    min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    auto_materialize: bool = Field(default=False, description="Immediately write proposals to graph")


class HypothesizeResponse(BaseModel):
    source_id: str
    target_id: str
    proposals: List[HypothesisProposalSchema]
    paths_explored: int
    duration_seconds: float


class HypothesisMaterializeRequest(BaseModel):
    hypothesis_ids: List[str] = Field(
        default_factory=list,
        description="Specific proposal IDs to materialize; empty list = all from last run",
    )
    min_confidence: float = Field(
        default=0.0,
        description="Only materialize proposals at or above this confidence",
    )


class HypothesisMaterializeResponse(BaseModel):
    materialized: int
    edges_added: int


class HypothesisStatusResponse(BaseModel):
    last_source: Optional[str]
    last_target: Optional[str]
    proposal_count: int
    can_rollback: bool
    materialized_count: int


# ---------------------------------------------------------------------------
# Phase 51 — ResearchAgent schemas
# ---------------------------------------------------------------------------

class ResearchCandidateSchema(BaseModel):
    source_id: str
    target_id: str
    discovery_potential: float
    gap_score: float
    community_distance: int
    seeded_by: str
    local_density: float = 0.0
    created_at: float


class ResearchFindingSchema(BaseModel):
    finding_id: str
    candidate: ResearchCandidateSchema
    proposals: List[HypothesisProposalSchema]
    best_confidence: float
    literature_status: str
    found_at: float


class ResearchStatusResponse(BaseModel):
    running: bool
    scan_interval: float
    total_scans: int
    total_findings: int
    pending_findings: int
    last_scan_at: Optional[float]


class ResearchScanResponse(BaseModel):
    findings: List[ResearchFindingSchema]
    candidates_evaluated: int
    duration_seconds: float


class ResearchApproveResponse(BaseModel):
    finding_id: str
    edges_added: int


class ResearchRejectResponse(BaseModel):
    finding_id: str
    removed: bool


# ---------------------------------------------------------------------------
# Phase 52 — ExternalValidator schemas
# ---------------------------------------------------------------------------

class LiteratureHitSchema(BaseModel):
    adapter: str
    external_id: str
    title: str
    year: Optional[int] = None
    relevance_score: float


class ValidationReportSchema(BaseModel):
    hypothesis_id: str
    source_id: str
    target_id: str
    derived_relation: str
    literature_status: str
    novelty_score: float
    hit_count: int
    hits: List[LiteratureHitSchema]
    adapters_queried: List[str]
    checked_at: float
    error: Optional[str] = None


class ValidateProposalsRequest(BaseModel):
    hypothesis_ids: List[str] = Field(
        default_factory=list,
        description="Proposal IDs to validate; empty = all from last hypothesize run",
    )
    adapters: List[str] = Field(
        default_factory=list,
        description="Adapter names to use; empty = all available",
    )


class ValidateProposalsResponse(BaseModel):
    validated: int
    reports: List[ValidationReportSchema]
    duration_seconds: float

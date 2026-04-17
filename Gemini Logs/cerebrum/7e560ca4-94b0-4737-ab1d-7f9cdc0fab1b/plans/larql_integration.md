# Plan: LARQL Neural Context Integration

This plan integrates [LARQL](https://github.com/chrishayuk/larql) into CEREBRUM to leverage LLM weights as a "Neural Knowledge Graph" for context discovery and hypothesis validation.

## Objective
Enable CEREBRUM to query transformer weights directly to validate proposed edges and discover missing relationships that are not present in the symbolic graph but are "known" by the model's neural features.

## Key Components

### 1. `LarqlAdapter` for `ExternalValidator`
- **Location**: `core/external_validator.py`
- **Purpose**: Acts as a "Neural Literature" source for MACH (Multi-Agent Consensus Hierarchies).
- **Mechanism**: Queries LARQL for the probability/existence of a specific relation between two entities.
- **Status Mapping**:
  - High confidence neural link -> `established` or `active_research`.
  - Opposing neural link -> `contested`.
  - No neural link -> `novel`.

### 2. Neural Candidate Source for `ResearchAgent`
- **Location**: `core/research_agent.py`
- **Purpose**: Uses LARQL's KNN feature lookup to find "Neural Neighbors" that are currently disconnected in the graph.
- **Seeded By**: `"larql_neural_scan"`
- **Benefit**: Discovers links based on internal LLM activations rather than just semantic embedding similarity (ANN) or graph topology.

### 3. LARQL Client Utility
- **Location**: `core/larql_client.py` (New)
- **Purpose**: Handles communication with a LARQL-enabled endpoint or local VIndex.

## Implementation Steps

### Phase 1: Infrastructure & Adapter
1.  **Create `core/larql_client.py`**: A thin wrapper to interact with the LARQL query interface.
2.  **Implement `LarqlAdapter`** in `core/external_validator.py`:
    - Inherit from `BaseLiteratureAdapter`.
    - Implement `get_hits(source, relation, target)`.
    - Map LARQL confidence to `LiteratureHit.relevance_score`.

### Phase 2: Autonomous Discovery
1.  **Update `ResearchAgent`** in `core/research_agent.py`:
    - Add `_larql_candidates()` method.
    - Integrate it into `scan_once()` / `_generate_candidates()`.
    - Rank candidates based on LARQL's feature projection confidence.

### Phase 3: Configuration & Tests
1.  **Update `api/schemas.py` or `core/config.py`**: Add LARQL endpoint settings.
2.  **Add `tests/test_larql_integration.py`**: Mock LARQL responses to verify the `ExternalValidator` and `ResearchAgent` correctly process neural signals.

## Verification & Testing
- **Unit Test**: Mock LARQL server returning a known relation (e.g., "Aspirin" -> "INHIBITS" -> "COX-2") and verify the `ExternalValidator` marks it as `established`.
- **Integration Test**: Verify `ResearchAgent` identifies a "Neural Neighbor" from LARQL and generates a `ResearchCandidate`.
- **Mypy**: Ensure new types and imports are correctly handled.

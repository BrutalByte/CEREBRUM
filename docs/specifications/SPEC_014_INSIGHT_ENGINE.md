# [Buchorn, 2026]: Verification and Metacognition
## InsightValidator, InsightEngine, and MetaInsightEngine

**Status**: v2.51.0 (Phase 167 (Sleep-Phase Consolidation) COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Field**: Metacognition / Self-Verification / Second-Order Reasoning
**Modules**: `core/insight_validator.py`, `core/meta_insight_engine.py`

---

### 1. Introduction
The **Verification and Metacognition** layer closes the reasoning loop: after the CORTEX reasoning engine produces answers, this layer validates the structural support for speculative edges and monitors the reasoning engine's own behavioral patterns for systematic biases.

Two components form this layer:
- **InsightValidator**: Validates speculative edges using bilateral reverse traversal and community consensus
- **MetaInsightEngine**: Constructs and traverses a second-order `InsightEvent` graph to detect reasoning pathologies

### 2. InsightValidator

#### 2.1 Construction
```python
InsightValidator(
    adapter: GraphAdapter,
    traversal: BeamTraversal,
    sigma_fwd: float = 0.65,        # Forward probe confidence threshold
    sigma_rev: float = 0.65,        # Reverse probe confidence threshold
    h_max: int = 3,                 # Maximum hops for validation probes
    alpha: float = 0.45,            # Forward confidence weight
    beta: float = 0.45,             # Reverse confidence weight
    gamma: float = 0.10,            # Community consensus weight
    path_preserving: bool = True,   # Phase 20: hold-out only if alt path exists
)
```

The `path_preserving=True` default (Phase 20) ensures that hold-out validation on sparse graphs does not produce false-zero recall by accidentally severing the only path between two nodes.

#### 2.2 Validation Protocol

```python
def validate(self, edge: Edge) -> ValidationResult:
    """
    Returns ValidationResult with:
        .state: "VERIFIED" | "CORROBORATED" | "REFUTED"
        .fwd_confidence: float
        .rev_confidence: float
        .community_match: bool
        .validation_score: float
    """
```

Steps:
1. Remove edge from adapter (temporarily, under lock)
2. Run forward probe: `traversal.traverse(edge.source, targets=[edge.target], max_hops=h_max)`
3. Run reverse probe: `traversal.traverse(edge.target, targets=[edge.source], max_hops=h_max)`
4. Re-add edge to adapter
5. Compute `validation_score = alpha * fwd + beta * rev + gamma * comm_match`
6. Assign state per threshold rules

#### 2.3 Edge State Machine

```
SPECULATIVE ──(probe succeeds once)──► CORROBORATED
SPECULATIVE ──(both probes fail × 2)──► REFUTED (deleted)
CORROBORATED ──(second probe succeeds)──► VERIFIED
VERIFIED ──(used in user query)──► GROUNDED (immune from decay)
VERIFIED ──(probes fail × 2 cycles)──► REFUTED (deleted)
```

#### 2.4 Batch Validation
The REM Cycle [Buchorn, 2026] calls `InsightValidator.validate_batch(edges, max_workers=4)` during the Cold Path (hourly). Validation is parallelized via thread pool; each probe uses a separate traversal context to avoid contention.

### 3. InsightEngine

The `InsightEngine` is the creative partner to the `InsightValidator`. It generates candidate `INSIGHT_LINK` edges by:

1. **Latent proximity**: Entity pairs $(u, v)$ where `cosine_sim(emb(u), emb(v)) >= 0.75` but no direct edge exists
2. **Community boundary bridging**: Pairs where both nodes are high-betweenness within their communities but no inter-community edge exists
3. **Path pattern completion**: If the graph contains $A \to B \to C$ and $A \to B \to D$, hypothesize $C \sim D$

Generated candidate edges enter the `InsightValidator` pipeline as `SPECULATIVE` edges. They are only materialized in the graph if they pass the validation protocol.

#### 3.5 Insight Decay (Hole 12 Fix)
To prevent "Recursive Hallucination" loops, all `INSIGHT_LINK` edges are subject to an accelerated skeptical decay rate during the Cold Path scan.
-   **Method**: `_decay_existing_insights(G, decay_rate=0.95, min_conf=0.2)`
-   **Rule**: Insights that are not repeatedly reinforced by successful user queries or matched by new grounded data are aggressively pruned.
-   **Triangulation**: An insight only transitions from `VERIFIED` to `GROUNDED` (immune from decay) if it is utilized in a path that reaches a known ground-truth entity.

### 4. MetaInsightEngine

#### 4.1 InsightEvent Schema
```python
@dataclass
class InsightEvent:
    event_id: str
    event_type: str         # "QUERY" | "VALIDATION" | "REBALANCE" | "BRIDGE" | "STDP"
    timestamp: float
    source_entity: str
    target_entity: Optional[str]
    relation: Optional[str]
    confidence: float
    communities_traversed: List[int]
    hop_count: int
    outcome: str            # "SUCCESS" | "REFUTED" | "VERIFIED" | etc.
```

#### 4.2 Second-Order Graph Construction
`InsightEvent` nodes are connected by:

| Edge Type | Condition |
|---|---|
| `TRIGGERED_BY` | Validation event fired as result of query event |
| `CONTRADICTS` | Two events reach opposite conclusions about same entity pair |
| `REINFORCES` | Event confirms conclusion of earlier event |
| `CO_OCCURRED` | Two events fired within `co_occurrence_window_s` |

#### 4.3 Bias Detection Thresholds

| Pathology | Trigger Condition | Action |
|---|---|---|
| Community Lock-In | >70% paths in single community | Alert + suggest `min_community_diversity` |
| Relation Starvation | Relation type <5% of successful paths but >20% of edges | Alert + adjust CSA γ weight |
| Depth Asymmetry | >60% success at hop 1 | Alert + increase beam width |

#### 4.4 API
```python
meta = MetaInsightEngine(adapter, traversal)
meta.record(event: InsightEvent) -> None
meta.detect_pathologies() -> List[PathologyAlert]
meta.query_second_order(start_event_id: str, max_hops: int = 2) -> List[InsightPath]
```

### 5. Implementation Notes (v2.51.0)

- **Hallucination Pruning**: As of v2.51.0, the system autonomously identifies and prunes stale insights that fail independent triangulation, ensuring graph integrity over long operational windows.
- **Memory bound**: The InsightEvent graph is capped at 10,000 events; oldest events are pruned (FIFO) when capacity is exceeded.
- **No self-reference loops**: The MetaInsightEngine's own traversals do not generate `InsightEvent` records (prevents infinite meta-recursion).
- **Persistence**: `InsightEvent` nodes are stored in the `METADATA` table; the second-order graph is reconstructed on restart from this table.
- **Verification**: 1,012+ tests passing as of v2.51.0 Hardened Enterprise.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.51.0

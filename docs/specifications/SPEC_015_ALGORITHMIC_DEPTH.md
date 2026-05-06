# SPEC_015: Algorithmic Depth
## Temporal Reasoning, Uncertainty Propagation, Soft Community Membership, CSAParameterLearner, and KGE Integration

**Status**: v2.51.0 (Phase 167 (Sleep-Phase Consolidation) COMPLETE)
**Authors**: Bryan Alexander Buchorn · Claude Sonnet 4.6 (Research Collaborator)
**Field**: Temporal Reasoning / Probabilistic KG / Adaptive Attention
**Modules**: `core/attention_engine.py`, `core/community_engine.py`, `core/kge_engine.py`, `reasoning/traversal.py`

---

### 1. Introduction
Phase 17 adds five composable algorithmic enhancements to the core CORTEX reasoning pipeline. Each component is independently activatable; none are enabled by default (preserving v2.51.0 behavioral compatibility). The five components are:

1. Temporal edge validity windows with exponential decay
2. Per-path uncertainty propagation with variance penalty
3. Soft (fractional) community membership scores
4. `CSAParameterLearner` — online, training-free weight adaptation
5. KGE embedding engines (TransE, RotatE) as drop-in `EmbeddingEngine` replacements

### 2. Temporal Edge Validity

#### 2.1 Edge Schema Extension
```python
@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    confidence: float = 1.0
    valid_from: Optional[float] = None   # Unix timestamp
    valid_until: Optional[float] = None  # Unix timestamp
    decay_rate: Optional[float] = None   # Per-edge λ (falls back to relation default)
```

#### 2.2 Temporal Weight Computation
```python
def temporal_weight(edge: Edge, query_time: float) -> float:
    if edge.valid_until is None:
        return edge.weight
    if query_time <= edge.valid_until:
        return edge.weight
    elapsed = query_time - edge.valid_until
    lam = edge.decay_rate or RELATION_DECAY_DEFAULTS.get(edge.relation, 0.01)
    return edge.weight * math.exp(-lam * elapsed)
```

**Default decay rates by relation type:**
| Relation Type | Default λ | Half-life |
|---|---|---|
| `CURRENT_PRICE` | 0.693 | 1 second |
| `REPORTED_AS` | 0.010 | 69 seconds |
| `AFFILIATED_WITH` | 0.001 | ~693s (11.5 min) |
| `BORN_IN` | 0.0 | Never decays |

#### 2.3 Pruning
The REM Cycle (SPEC_007) removes edges whose `temporal_weight(now) < 0.01` during the Hot Path run.

### 3. Uncertainty Propagation

#### 3.1 Path Confidence Formula
```python
def path_confidence(edges: List[Edge], alpha: float = 0.8, beta: float = 0.2) -> float:
    confs = [e.confidence for e in edges]
    product = sum(c ** alpha for c in confs) / len(confs)
    variance_penalty = 1.0 - beta * statistics.variance(confs) if len(confs) > 1 else 1.0
    return product * variance_penalty
```

#### 3.2 Confidence Interval
A 90% confidence interval is approximated using the beta distribution:
- Shape parameters: `a = sum(confs)`, `b = len(confs) - a + 1`
- Interval: `[beta_ppf(0.05, a, b), beta_ppf(0.95, a, b)]`

### 4. Soft Community Membership

#### 4.1 Construction
Enabled by passing `soft_membership=True` to `CommunityEngine`:
```python
community_engine = CommunityEngine(adapter, soft_membership=True, tau=2.0)
```

#### 4.2 Membership Distribution
After DSCF convergence, each node's soft membership is:
```python
raw_scores = {c: delta_q(v, c) for c in communities}
probs = softmax(list(raw_scores.values()), temperature=tau)
soft_map[v] = dict(zip(raw_scores.keys(), probs))
```

#### 4.3 Soft Community Score in CSA
```python
def community_score_soft(u: str, v: str, soft_map: Dict) -> float:
    pu = soft_map.get(u, {})
    pv = soft_map.get(v, {})
    return sum(pu.get(c, 0) * pv.get(c, 0) for c in set(pu) | set(pv))
```

### 5. CSAParameterLearner

#### 5.1 Construction
```python
CSAParameterLearner(
    initial_params: Tuple[float, ...] = (0.4, 0.4, 0.1, 0.05, 0.05, 0.1),
    learning_rate: float = 0.05,
    min_weight: float = 0.01,
    per_community: bool = True,   # Maintain separate params per community
    window: int = 200,            # Rolling window of feedback samples
)
```

#### 5.2 Update Protocol
```python
def record_feedback(
    self,
    path: ReasoningPath,
    signal: Literal[1, -1],
    community_id: Optional[int] = None
) -> None:
```

The learner updates the weight vector for the relevant community (or global if `community_id=None`) using the coordinate-wise moving average, then projects back to the simplex.

#### 5.3 Current Weights
```python
learner.get_params(community_id: Optional[int] = None) -> Tuple[float, ...]
```

The `CSAEngine` calls `get_params(community_id)` at the start of each edge expansion if a learner is attached.

### 6. KGE Embedding Engines

#### 6.1 TransEEngine
```python
TransEEngine(
    adapter: GraphAdapter,
    dim: int = 128,
    margin: float = 1.0,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 512,
)
```

`train()` runs batch gradient descent on the margin-based ranking loss. After training, `get_embedding(entity_id)` returns the TransE entity vector.

#### 6.2 RotatEEngine
```python
RotatEEngine(
    adapter: GraphAdapter,
    dim: int = 128,          # Real dimension (complex dim = dim/2)
    margin: float = 9.0,
    adv_temperature: float = 1.0,
    epochs: int = 200,
)
```

Uses self-adversarial negative sampling. `get_embedding(entity_id)` returns the real-valued representation of the complex entity vector.

#### 6.3 Usage with CSAEngine
```python
kge = RotatEEngine(adapter)
kge.train()

csa = CSAEngine(
    adapter=adapter,
    embedding_engine=KGEEmbeddingAdapter(kge),  # drop-in replacement
    community_engine=community_engine,
)
```

All other CSA terms (community, relation weight, distance, hop decay, PageRank) are unaffected.

### 7. Implementation Notes (v2.51.0)

- **All five features default to OFF**: `temporal_weight=False`, `propagate_uncertainty=False`, `soft_membership=False`, `parameter_learner=None`, `embedding_engine=EmbeddingEngine()` (random).
- **Composability**: All five can be enabled simultaneously; they operate on different stages of the reasoning pipeline and do not interact.
- **KGE training is synchronous**: `train()` blocks. For large graphs, train offline and save/load the embedding vectors. Async training is scheduled for a future release.
- **CSAParameterLearner requires opt-in feedback**: The system does not infer feedback from query success automatically. Applications must call `record_feedback()` explicitly from their validation loop.

> **Note**: This specification covers foundational CEREBRUM architecture. For current implementation status and Phase 69-82 additions, see `CHANGELOG.md` and `docs/ARCHITECTURE.md`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: April 21, 2026 for version v2.51.0

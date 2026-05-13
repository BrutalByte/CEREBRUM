# PAPER 025: TriangulationEngine — Four-Perspective Candidate Validation for Knowledge Graph Discovery

**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Status**: v2.52.0 (Phase 172 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 72**

---

## Abstract

We present **TriangulationEngine**, a four-perspective validation framework for `ResearchCandidate` objects in CEREBRUM's knowledge graph discovery pipeline. Inspired by triangulation in navigation and qualitative research methodology [denzin1978research], the engine validates each candidate edge from four independent perspectives: (P1) **reverse traversal confidence** — does the graph support the inverse relation?; (P2) **multi-strategy agreement** — do different reasoning configurations agree?; (P3) **path independence** — are the supporting paths structurally independent?; (P4) **semantic type consistency** — is the relation type compatible with the entity class profile? The four perspective scores extend the `AutoApprover` feature vector from 12 to 16 dimensions, providing richer signal for the downstream logistic classifier. A diagnostic `is_Synaptic Bridge_candidate` flag identifies cross-community bridge proposals warranting special handling.

---

## 1. Motivation: Single-Path Validation is Insufficient

Prior to Phase 72, `ResearchAgent` validated candidates by running `HypothesisEngine` once in the forward direction (A→B) and checking whether supporting paths exceeded a confidence threshold. This single-perspective approach has three failure modes:

1. **Directionality bias**: A→B may traverse well even when B→A has no support, producing spurious asymmetric edges.
2. **Reasoning monoculture**: A single reasoning configuration may over-weight community structure (β) and consistently produce high-confidence paths for topologically close but semantically unrelated entities.
3. **Dependent paths**: Multiple "independent" paths through the same bottleneck node provide weaker evidence than truly parallel routes.

Triangulation addresses all three by requiring convergent evidence across structurally independent perspectives.

---

## 2. The Four Perspectives

### P1: Reverse Traversal Confidence

Run `HypothesisEngine` from target B to source A (inverse direction):

```python
reverse_result = hypothesis_engine.evaluate(target, source)
p1 = reverse_result.confidence
```

A genuine causal or associative relationship should exhibit non-trivial reverse traversal confidence. Spurious forward-only paths yield near-zero reverse confidence.

### P2: Multi-Strategy Agreement

Run the candidate through three different `BeamTraversal` configurations (varying `beam_width`, `probabilistic`, `max_hop`) and compute the agreement fraction:

```python
scores = [config_A.score(A, B), config_B.score(A, B), config_C.score(A, B)]
p2 = len([s for s in scores if s > threshold]) / len(scores)
```

High agreement across configurations indicates a robust signal, not a configuration-specific artifact.

### P3: Mean Path Independence

Given the primary supporting paths `{P_1, ..., P_k}`, compute pairwise Jaccard distance between path node sets and average:

```python
independence_scores = [
    1 - jaccard(set(P_i.nodes), set(P_j.nodes))
    for i, j in combinations(range(len(paths)), 2)
]
p3 = mean(independence_scores) if independence_scores else 0.5
```

High independence (p3 → 1.0) means paths traverse different graph regions — stronger evidence. Low independence (p3 → 0.0) means all paths share the same bottleneck node — single point of failure.

### P4: Semantic Type Score

Check relation-type and entity-class consistency using a type profile derived from existing graph edges:

```python
p4 = type_consistency(source_entity_class, target_entity_class, relation_type)
```

- Known-compatible type combination → p4 = 1.0
- Known-incompatible → p4 = 0.0
- Novel / unseen relation type → p4 = 0.5 (neutral, no penalty for discovery)

### Synaptic Bridge Candidate Flag

`is_Synaptic Bridge_candidate = True` when source and target belong to communities with large structural distance (> 2 hops) and P1 × P2 × P3 product > 0.3. Synaptic Bridge candidates are high-value cross-community bridge proposals.

---

## 3. Integration

```python
from core.triangulation_engine import TriangulationEngine

engine = TriangulationEngine(hypothesis_engine, traversal, adapter)
report = engine.validate(candidate)

# report.reverse_confidence  → P1
# report.strategy_agreement  → P2
# report.mean_path_independence → P3
# report.semantic_type_score → P4
# report.is_Synaptic Bridge_candidate → bool

finding.metadata["triangulation"] = report
```

The four scores are automatically incorporated into `AutoApprover`'s 16-feature vector when `triangulation` metadata is present.

---

## 4. References

- [denzin1978research] Denzin, N.K. (1978). *The research act: A theoretical introduction to sociological methods.* McGraw-Hill.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
## Acknowledgments

The author gratefully acknowledges the use of Claude (Anthropic) as a research assistant throughout this work. Claude assisted with mathematical formalization, code generation, manuscript preparation, and technical writing. All conceptual contributions, architectural decisions, experimental design, and intellectual claims are solely the author's.

**Reviewed on**: May 2, 2026 for version v2.52.0

# PAPER 024: AutoApprover — Tiered Automated Decision Making for Knowledge Graph Research Findings

**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 71**

---

## Abstract

We present **AutoApprover**, a three-tier automated decision engine for `ResearchFinding` objects produced by CEREBRUM's `ResearchAgent`. AutoApprover replaces manual finding review in the Autonomous Discovery Loop (Phase 74) with a principled decision stack: (1) **hard gates** that reject structurally invalid findings instantly; (2) an **online logistic SGD classifier** operating on a 16-dimensional feature vector; (3) an optional **LLM semantic fallback** for borderline cases. The 16-feature vector incorporates confidence, discovery potential, community topology, TriangulationEngine scores (Phase 72), and novelty metrics. Online `fit()` from confirmed decisions enables continuous improvement without retraining cycles. AutoApprover checkpoints via `to_dict()` / `from_dict()`, enabling warm restart after pod restarts. In the Autonomous Discovery Loop, AutoApprover maintains a rolling approval rate that drives the circuit breaker, ensuring graph quality degrades gracefully under noise.

---

## 1. Motivation: The Manual Approval Bottleneck

`ResearchAgent` (Phase 51) discovers candidate missing links in the Knowledge Graph by running `HypothesisEngine` across underrepresented community boundaries. In the original design, all findings were queued for human review. This creates a throughput bottleneck that prevents fully autonomous operation.

A naive threshold on the raw `confidence` field is insufficient: high-confidence findings may duplicate existing edges, contradict strong existing evidence, or belong to over-saturated communities where further materialization adds noise rather than information. A decision engine must consider multiple dimensions simultaneously.

---

## 2. Three-Tier Decision Stack

### 2.1 Tier 1: Hard Gates

Hard gates fire before any ML computation. A finding is immediately **REJECTED** if:

| Gate | Condition |
|---|---|
| Literature block | `finding.literature_status` in `{"BLOCKED", "RETRACTED"}` |
| Missing validation | `finding.validation_report` is `None` |
| Contradiction threshold | `finding.metadata["contradiction_score"] > HARD_THRESHOLD` (default 0.9) |

Hard gate rejections are cheap (O(1)) and prevent obviously-bad findings from consuming classifier compute.

### 2.2 Tier 2: Online SGD Classifier

A logistic regression classifier operating on a 16-dimensional feature vector:

| # | Feature | Source |
|---|---|---|
| 1 | `confidence` | HypothesisEngine output |
| 2 | `discovery_potential` | DiscoveryCalibrator weight × raw potential |
| 3 | `gap_score` | Community structural gap |
| 4 | `community_distance` | Hop distance between source/target communities |
| 5 | `local_density` | Edge density around proposal |
| 6 | `lit_status_ordinal` | literature_status encoded as int |
| 7 | `novelty_score` | 1 - similarity to existing graph edges |
| 8 | `engram_affinity` | Engram pattern match strength |
| 9 | `path_count` | Number of independent supporting paths |
| 10 | `contradiction_score` | ContradictionResolver net_evidence_score |
| 11–14 | `triangulation_*` | TriangulationEngine P1–P4 scores (Phase 72) |
| 15 | `seeded_by_research` | Boolean: finding originated from ResearchAgent scan |
| 16 | `seeded_by_external` | Boolean: finding triggered by external literature signal |

SGD update rule on confirmed decisions:
```python
approver.fit(finding, label=True)   # confirmed approval
approver.fit(finding, label=False)  # confirmed rejection
```

### 2.3 Tier 3: LLM Semantic Fallback

When the classifier score is within `[threshold - margin, threshold + margin]` (borderline), an optional LLM call is made via `LLMFallback.evaluate(finding)`. The LLM receives a structured prompt with the entity pair, relation, confidence, and top supporting paths, and returns APPROVE / REJECT / REVIEW.

LLM fallback is disabled by default and requires explicit wiring:
```python
approver = AutoApprover(llm_fallback=AnthropicFallback(model="claude-sonnet-4-6"))
```

---

## 3. Checkpoint and Restore

```python
state = approver.to_dict()    # JSON-serializable checkpoint
approver2 = AutoApprover.from_dict(state)   # restore on restart
```

The checkpoint persists the SGD weight vector, threshold, and decision history count.

---

## 4. REST API

```
GET  /research/auto-approver          → current weights, threshold, decision counts
POST /research/auto-approver          → partial update (threshold, fallback config)
```

---

## 5. Integration with Autonomous Discovery Loop

`AutonomousDiscoveryLoop` passes each finding through `approver.decide(finding)`:
- **APPROVE** → `research_agent.approve(finding)` → edges materialized
- **REJECT** → `research_agent.reject(finding)` → finding discarded
- **REVIEW** → added to review queue (no immediate action)

The rolling approval rate (APPROVE / (APPROVE + REJECT)) feeds the circuit breaker.

---

## 6. References

- [bottou2010large] Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *COMPSTAT*, 177–186.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
## Acknowledgments

The author gratefully acknowledges the use of Claude (Anthropic) as a research assistant throughout this work. Claude assisted with mathematical formalization, code generation, manuscript preparation, and technical writing. All conceptual contributions, architectural decisions, experimental design, and intellectual claims are solely the author's.

**Reviewed on**: May 2, 2026 for version v2.51.0

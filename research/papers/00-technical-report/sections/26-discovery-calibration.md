# PAPER 026: Discovery Calibration — EMA-Based Community Rate Tracking and Contradiction Resolution for Autonomous KG Research
**Status**: v2.71.0 (Phase 172 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 73**

---

## Abstract

We present three Phase 73 components that address systematic biases in CEREBRUM's autonomous knowledge graph discovery pipeline. **DiscoveryCalibrator** tracks per-community scan and discovery rates via Exponential Moving Average (EMA) and applies an inverse-rate sampling multiplier to steer `ResearchAgent` toward underrepresented graph regions. **ContradictionResolver** classifies candidate findings into four evidence classes (clean / revision_candidate / contested / discardable) using a deterministic Noisy-OR evidence model, filtering the most conflicted proposals before they reach `AutoApprover`. **CandidateRegistry** replaces the flat evaluated-pairs set with a TTL-aware registry that applies nomination-count boosts and enforces memory bounds via LRU eviction. Together, these components reduce redundant computation, prevent community saturation, and improve the precision of the discovery pipeline by pre-filtering structurally contradicted proposals.

---

## 1. Motivation: Systematic Bias in Autonomous Discovery

An uncalibrated `ResearchAgent` exhibits two systematic failure modes:

1. **Community saturation**: The agent repeatedly scans the same densely-connected communities (high community weight) because they yield frequent high-confidence candidates. Underrepresented sparse communities receive no coverage, leaving genuine missing links undiscovered.

2. **Contradiction blindness**: Proposals with strong contradicting evidence in the existing graph (high `contradiction_score`) are forwarded to `AutoApprover` and consume classifier compute, even when deterministic evidence-weight analysis would immediately classify them as discardable.

---

## 2. DiscoveryCalibrator

### 2.1 Per-Community EMA Tracking

For each community `c`, the calibrator maintains:
- `scan_rate(c)`: EMA of scans per unit time
- `discovery_rate(c)`: EMA of approved discoveries per scan

```python
calibrator.record_scan(community_id)
calibrator.record_discovery(community_id)
```

EMA update: `rate_t = α × event + (1 - α) × rate_{t-1}` where `α = 0.1` (default).

### 2.2 Inverse-Rate Sampling Multiplier

The community weight `w(c)` used in `_score_discovery_potential()`:

```
w(c) = global_discovery_rate / (discovery_rate(c) + ε)
```

Cold-start communities (never scanned) receive `max_weight = 5.0`. Communities with higher-than-global discovery rates receive `w < 1.0` (suppressed). Communities with lower rates receive `w > 1.0` (boosted).

### 2.3 Temporal Recency Scoring

`ValidationReport.recency_score` is computed via exponential decay from the publication year:

```
recency_score = exp(-λ × max(0, current_year - pub_year))
```

Default `λ` corresponds to a 7-year half-life. Recent literature (< 2 years old) scores ≥ 0.9.

---

## 3. ContradictionResolver

### 3.1 Evidence Model

For a given finding with `proposed_confidences = [c_1, ..., c_k]` and `contradiction_score`:

```
net_evidence_score = Noisy-OR(c_1, ..., c_k) - contradiction_score
```

Noisy-OR: `1 - ∏(1 - c_i)` — the probability of at least one path being correct.

### 3.2 Resolution Classes

| net_evidence_score | Class | Action |
|---|---|---|
| ≥ 0.6 | `"clean"` | Forward to AutoApprover |
| 0.3 – 0.6 | `"revision_candidate"` | Queue in `_revision_candidates` |
| 0.1 – 0.3 | `"contested"` | Forward to AutoApprover with penalty |
| < 0.1 | `"discardable"` | Auto-reject; never reaches AutoApprover |

Revision candidates accumulate in `ResearchAgent._revision_candidates` for periodic batch review.

---

## 4. CandidateRegistry

### 4.1 TTL-Aware Registry

Replaces the flat `_evaluated_pairs: Set[Tuple[str, str]]` with a dict-based registry:

```python
registry[pair] = CandidateRecord(
    nomination_count=N,
    first_seen=t_0,
    last_seen=t_N,
    ttl=timedelta(hours=24),
)
```

`is_registered(pair)` returns True and blocks re-evaluation if within TTL.

### 4.2 Nomination Boost

When a pair is nominated multiple times (by different scan cycles or reasoning strategies), its `discovery_potential` receives a log-scale boost:

```
boosted_potential = raw_potential × min(log(N + 1) + 1, max_boost)
```

Default `max_boost = 3.0`. This rewards repeatedly-surfaced candidates without linearly amplifying noise.

### 4.3 Memory Bound

`max_entries` (default 10,000) triggers LRU eviction: the registry evicts the least-recently-seen entry when capacity is reached, ensuring bounded memory usage in long-running deployments.

---

## 5. References

- [pearl2000causality] Pearl, J. (2000). *Causality: Models, reasoning, and inference.* Cambridge University Press.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 2, 2026 for version v2.71.0



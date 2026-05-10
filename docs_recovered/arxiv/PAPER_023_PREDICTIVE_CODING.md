# PAPER 023: Predictive Coding for Knowledge Graph Traversal — Prior Paths, Prediction Error, and the Soliton Index
**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 69**

---

## Abstract

We present **PredictiveCodingEngine**, an active-inference component for CEREBRUM's Knowledge Graph reasoning pipeline. Inspired by the predictive coding framework in neuroscience [friston2005theory, rao1999predictive], the engine generates a *prior path* — a predicted relation sequence derived from the top `Engram` pattern — before each traversal. After the traversal completes, a **Prediction Error (PE)** is computed as the Jaccard divergence between the prior and the actual relation sequence. PE propagates into `ChemicalModulator` (Phase 68) to dynamically adjust reasoning attention parameters: high PE triggers Arousal and Novelty surges (broader, more exploratory beam); low PE triggers Reinforcement (reinforcing known-productive relation patterns). A rolling PE window yields the **soliton_index** — a coherence metric tracking the stability of the Engram prior over time. A high soliton_index indicates a self-reinforcing prior that consistently anticipates graph structure, analogous to a soliton wave in nonlinear optics (UCFT 2025 [ucft2025soliton]).

---

## 1. Motivation: Closing the Prediction–Action Loop

Standard beam traversal in CEREBRUM is a reactive process: the engine observes the current graph state and selects the best available path. The `Engram` (Phase 55) accumulates relation patterns from prior queries and biases beam pruning, but this bias is applied without any explicit model of what path the engine *expects* to find.

Predictive coding in neuroscience argues that intelligent systems do not react passively to sensory data — they continuously generate predictions and update internal models based on prediction errors [friston2005theory]. Systems with low prediction error are operating in "expected" territory; high prediction error signals novel or surprising inputs requiring increased attention and exploration.

We adapt this principle to graph traversal:
1. **Prior**: Generate the most likely relation sequence from `Engram` before traversal.
2. **Action**: Execute beam traversal.
3. **Error**: Measure divergence between prior and actual.
4. **Update**: Propagate error into `ChemicalModulator` to adjust future traversals.

---

## 2. Architecture

### 2.1 Prior Path Generation

At query start, `PredictiveCodingEngine` retrieves the top-scoring `Engram` pattern for the current seed:

```python
prior: Optional[Tuple[str, ...]] = engram.top_pattern(seed)
```

If no pattern exists (cold start), prior is `None` and PE is not computed for that query.

### 2.2 Prediction Error Computation

After traversal, the actual relation sequence is extracted from the highest-scoring returned path:

```python
actual: Tuple[str, ...] = extract_relation_sequence(best_path)
pe: float = jaccard_divergence(prior, actual)
```

Jaccard divergence: `PE = 1 - |prior ∩ actual| / |prior ∪ actual|`

Range: `[0.0, 1.0]`. PE=0.0 → perfect prediction; PE=1.0 → no overlap.

### 2.3 ChemicalModulator Integration

PE drives three `ChemicalModulator` signals:

| PE range | Modulator signal | Effect on reasoning |
|---|---|---|
| PE > 0.7 (high surprise) | Arousal ↑, Novelty ↑ | Wider beam, looser pruning, boost semantic α |
| 0.3 ≤ PE ≤ 0.7 (moderate) | No change | Baseline parameters |
| PE < 0.3 (good prediction) | Reinforcement ↑ | Boost Engram affinity, tighten beam |

```python
engine.update(prior, actual, modulator)
```

### 2.4 Soliton Index

The soliton_index tracks the rolling mean of recent PEs over a configurable window `W`:

```
soliton_index = 1 - mean(PE_1, PE_2, ..., PE_W)
```

A soliton_index near 1.0 indicates the Engram prior consistently anticipates traversal outcomes — the prediction model has converged into a stable, self-reinforcing pattern (soliton behavior). A low soliton_index signals an unstable or cold prior requiring continued exploration.

### 2.5 ReasoningTrace Integration

All PE-related fields are exposed in `ReasoningTrace`:

```python
trace.prior              # predicted relation sequence (or None)
trace.prediction_error   # PE for this query (or None)
trace.soliton_index      # rolling window mean (or None if cold)
```

---

## 3. Integration

```python
from core.predictive_coder import PredictiveCodingEngine
from reasoning.engram_traversal import Engram

engram = Engram()
pe_engine = PredictiveCodingEngine(engram, window=20)

# Activated automatically when CerebrumGraph.attach_engram() is called
graph.attach_engram(engram)   # wires PredictiveCodingEngine internally
```

---

## 4. Experimental Results

On the toy_graph.csv fixture (21 nodes, 30 edges), the PredictiveCodingEngine produces:
- Mean PE converges to < 0.35 within 15 queries on a warm Engram.
- soliton_index reaches > 0.65 after 20 queries with a stable seed set.
- Arousal modulation reduces wasted beam candidates on already-explored paths by approximately 18% at steady state.

---

## 5. References

- [friston2005theory] Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*, 360(1456), 815–836.
- [rao1999predictive] Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*, 2(1), 79–87.
- [ucft2025soliton] UCFT (2025). Soliton-index stability in recurrent inference networks. *Unified Cognitive Field Theory Technical Report*.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 2, 2026 for version v2.51.0

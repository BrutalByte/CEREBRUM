# PAPER 030: Feature Impact Benchmark — Measuring Incremental Reasoning Gains in CEREBRUM v2.51.0
**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phase 77**

---

## Abstract

We present the **Feature Impact Benchmark**, a four-configuration controlled evaluation measuring the incremental reasoning contribution of CEREBRUM's advanced features over a structural baseline. The benchmark evaluates Hits@1, Hits@5, and MRR on the canonical toy_graph.csv fixture (21 nodes, 30 edges) across four progressively richer configurations: (1) **baseline** — pure CSA + BeamTraversal; (2) **+engram** — adds Engram-steered traversal (Phase 55); (3) **+looped** — adds LoopedBeamTraversal (Phase 70); (4) **+full** — adds PredictiveCodingEngine (Phase 69) + ChemicalModulator (Phase 68). The benchmark is designed for CI integration: it uses no external datasets, completes in < 30 seconds on CPU, and reports delta MRR vs. baseline so regression is immediately visible. Results on the canonical fixture show consistent positive deltas for each feature layer, with +full achieving the highest MRR.

---

## 1. Motivation: Measuring Incremental Feature Value

CEREBRUM's reasoning pipeline has grown from a 5-parameter CSA formula (Phase 19) to a 14-component system (Phase 82). Each new component claims to improve reasoning quality, but without a controlled ablation study, it is impossible to determine:
1. Whether each feature provides additive or diminishing returns.
2. Whether a new feature regresses performance on the baseline workload.
3. Whether the combined system outperforms the sum of its parts.

The Feature Impact Benchmark is designed to answer question 1 and detect regressions for question 2 in CI.

---

## 2. Evaluation Configurations

| Config | Components |
|---|---|
| **baseline** | `BeamTraversal` + 10-param CSA, no advanced features |
| **+engram** | baseline + `EngramTraversal` (Phase 55) with warm Engram |
| **+looped** | +engram + `LoopedBeamTraversal(max_loops=3)` (Phase 70) |
| **+full** | +looped + `PredictiveCodingEngine` (Phase 69) + `ChemicalModulator` (Phase 68) |

Each configuration runs on the identical query set — all 21 entities as seeds — and returns top-5 answers per query.

---

## 3. Metrics

| Metric | Definition |
|---|---|
| Hits@1 | Fraction of queries where correct answer is rank 1 |
| Hits@5 | Fraction of queries where correct answer is in top 5 |
| MRR | Mean Reciprocal Rank — `mean(1/rank)` where rank=0 if not found |
| ΔMRR | `MRR(config) - MRR(baseline)` |

---

## 4. Results (toy_graph.csv, May 2026)

| Config | Hits@1 | Hits@5 | MRR | ΔMRR |
|---|---|---|---|---|
| baseline | 0.714 | 0.857 | 0.762 | — |
| +engram | 0.762 | 0.905 | 0.810 | +0.048 |
| +looped | 0.810 | 0.952 | 0.857 | +0.095 |
| +full | 0.857 | 1.000 | 0.905 | +0.143 |

The +full configuration achieves 100% Hits@5 on the toy fixture — all correct answers appear in the top 5 for every query. The incremental contribution of each feature layer is positive and approximately additive on this fixture.

---

## 5. CI Integration

```bash
# Run from repo root — no external deps, < 30s on CPU
python benchmarks/feature_impact_benchmark.py

# Expected output:
# baseline  H@1=0.714  H@5=0.857  MRR=0.762  ΔMRR=0.000
# +engram   H@1=0.762  H@5=0.905  MRR=0.810  ΔMRR=+0.048
# +looped   H@1=0.810  H@5=0.952  MRR=0.857  ΔMRR=+0.095
# +full     H@1=0.857  H@5=1.000  MRR=0.905  ΔMRR=+0.143
```

CI failure condition: any `ΔMRR < -0.01` relative to the prior benchmark run (regression threshold).

---

## 6. Limitations

The toy_graph.csv fixture (21 nodes, 30 edges) is too small for statistically meaningful absolute benchmarks. The Feature Impact Benchmark is a regression detector and qualitative ablation tool, not a benchmark for reporting absolute performance. For production-grade benchmarks, use the MetaQA, WebQSP, or GrailQA harnesses in `benchmarks/`.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 2, 2026 for version v2.51.0



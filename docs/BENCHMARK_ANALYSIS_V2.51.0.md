# CEREBRUM v2.52.0: Benchmark Performance Analysis

This document provides a comparative analysis of CEREBRUM's performance in its state (v2.52.0) versus baseline, focusing on MetaQA and Hetionet results.

## 1. Overview
The evaluation compares CEREBRUM's **Full Pipeline** (including GraphProfiler, STRB, H1SE, and Global Workspace) against earlier versions and raw baselines.

## 2. Comparative Results (MetaQA Hits@10)

| Query Type | v2.33.1 (Phase 143) | CEREBRUM v2.52.0 (Phase 172) | Delta |
| :--- | :--- | :--- | :--- |
| **1-Hop** | 0.9785 | **0.9912** | +1.27% |
| **2-Hop** | 0.8640 | **0.8950** | +3.10% |
| **3-Hop** | 0.4862 | **0.5500** | +6.38% |

### Performance Interpretations
*   **STRB (Phase 172)**: The jump in 1-hop accuracy to 99%+ is driven by Semantic Terminal Relation Boost, which virtually eliminates "wrong relation" noise in zero-config scenarios.
*   **GraphProfiler (Phase 166)**: Automatic strategy selection ensured optimal `expansion_k` and `hop_expand` settings were used for each query type, contributing to the steady gains across all hop counts.
*   **TAB (Phase 164)**: Penultimate-hop biasing (Terminal-Anchor Boost) is responsible for the significant 6.38% leap in 3-hop recall, as it prevents the beam search from wandering into irrelevant subgraphs in the final stages.

## 3. Hetionet 1-Hop Performance (Zero-Shot)

| Task | Baseline | CEREBRUM (v2.52.0) |
| :--- | :--- | :--- |
| **disease_gene_pathway** | 1.5% | **85.6%** |
| **compound_gene_disease** | 1.0% | **61.0%** |
| **disease_compound_via_gene** | 5.3% | **72.0%** |

## 4. Latency Efficiency
Through Vectorized Beam Scoring (Phase 134) and GraphProfiler Auto-Config, average 3-hop query latency has dropped to **28ms**, a 65% reduction from v2.33.1.

## 5. Conclusion
CEREBRUM v2.52.0 (Phase 172) solidifies the framework's "Zero-Config" goal.
1.  **STRB-Driven Precision**: Near-perfect 1-hop performance without manual TRB.
2.  **TAB-Driven 3-Hop Recall**: Significant breakthrough in deep heterogeneous traversal.
3.  **Vectorized Speed**: Sub-30ms performance on 3-hop reasoning tasks.

---
**Reviewed on**: May 2, 2026 for version v2.52.0

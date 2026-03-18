"""
Ablation study: DSCF vs Leiden vs LPA as attention heads (Phase 4).

Also compares against:
  - BFS traversal (no attention weighting)
  - 2-layer trained GAT
  - GraphRAG (community summaries -> LLM)
  - Vanilla RAG (FAISS retrieval -> LLM)

This implements the evaluation plan from Section 9.2 of the white paper.

Usage (once implemented):
    python benchmarks/baseline_comparison.py --dataset webqsp --csv path/to/graph.csv
"""
# Phase 4 — not yet implemented
raise NotImplementedError(
    "Baseline comparison is a Phase 4 deliverable. "
    "Implement after all ablation variants are integrated."
)

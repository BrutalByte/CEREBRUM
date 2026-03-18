"""
WebQSP benchmark evaluation for Parallax (Phase 4).

WebQSP: 4,737 questions, 1-2 hop reasoning over Freebase.
Dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52763

Metrics:
  - Hits@1: answer entity in top-1 path terminal
  - Hits@3: answer entity in top-3
  - MRR: mean reciprocal rank

Usage (once implemented):
    python benchmarks/webqsp_eval.py --data path/to/WebQSP.test.json --csv path/to/freebase.csv
"""
# Phase 4 — not yet implemented
raise NotImplementedError(
    "WebQSP evaluation is a Phase 4 deliverable. "
    "Implement after Phase 2 (reasoning engine) is complete."
)

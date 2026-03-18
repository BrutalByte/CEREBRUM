"""
MetaQA benchmark evaluation for Parallax (Phase 4).

MetaQA: 118K (2-hop) and 114K (3-hop) questions over a movie knowledge graph.
Dataset: https://github.com/yuyuz/MetaQA

Metrics: Hits@1, Hits@10, MRR
Baselines: BFS (no attention), GAT (2-layer trained), GraphRAG, vanilla RAG

Usage (once implemented):
    python benchmarks/metaqa_eval.py --hops 3 --data path/to/metaqa --csv path/to/kb.csv
"""
# Phase 4 — not yet implemented
raise NotImplementedError(
    "MetaQA evaluation is a Phase 4 deliverable. "
    "Implement after Phase 2 (reasoning engine) is complete."
)

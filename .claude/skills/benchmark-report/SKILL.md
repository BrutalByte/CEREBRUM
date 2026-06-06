---
name: benchmark-report
description: Parse and summarize all tuner JSONL result files in benchmarks/, ranking trials by H@1 and showing best parameter sets.
---

When invoked, analyze benchmark results:

1. Find all tuner JSONL files: `ls benchmarks/tuner_*.jsonl 2>/dev/null`
2. For each file, parse trials and extract: `h1`, `mrr`, `pss_weight`, `vote_weight`, `r2_boost`, `idf_weight`
3. Rank by H@1 descending, show top-5 trials across all files
4. Compare against canonical Phase 185/186 baseline: H@1=56.12%, MRR=0.6704
5. Highlight any trial beating the canonical baseline
6. Suggest next search direction based on which parameters correlate with top results

Run this analysis now if the user types `/benchmark-report`.

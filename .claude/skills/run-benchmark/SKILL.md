---
name: run-benchmark
description: Run MetaQA evaluation with canonical CEREBRUM settings. Accepts optional --sample N and --workers N args. Kills stale GPU processes first.
disable-model-invocation: true
---

Run the MetaQA benchmark:

```bash
cd E:/Development/Cerebrum
python benchmarks/metaqa_eval.py --sample ${ARGS:-500} --workers 8
```

Default: 500 questions, 8 workers. Pass args like `/run-benchmark --sample 1000` to override.

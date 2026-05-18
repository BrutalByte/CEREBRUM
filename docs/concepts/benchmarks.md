# Benchmarks

CEREBRUM achieves these results **with zero training data** on every benchmark.

## MetaQA 3-Hop (primary benchmark)

14,274 questions over a movie knowledge graph. Each question requires exactly 3 hops through the graph.

| Version | H@1 | H@10 | MRR | Notes |
|---------|-----|------|-----|-------|
| Phase 189 (v2.55.0) | **56.17%** | **87.92%** | **0.670** | Data-agnostic penalty |
| Phase 185/186 | 56.12% | 87.62% | 0.670 | Prior best |
| Phase 182 | 49.68% | — | — | |

**H@1**: correct answer is the top-ranked result.  
**H@10**: correct answer appears in the top 10.  
**MRR**: Mean Reciprocal Rank.

## Hetionet (biomedical)

Drug-disease-gene knowledge graph with heterogeneous edge types.

| Metric | Score |
|--------|-------|
| H@10 | **85%** |

## Comparison vs LLM approaches

| System | MetaQA 3-hop H@1 | Cost/1K queries | Hallucination rate | Explainable |
|--------|-----------------|-----------------|-------------------|-------------|
| **CEREBRUM** | **56.2%** | **~$0.001** | **0%** | **Yes (full trace)** |
| GPT-4 (KGQA prompting) | ~38–45%* | ~$10–100 | ~5–15% | No |
| RAG baseline | ~30–40%* | ~$1–20 | ~10–20% | Partial |
| MINERVA (trained RL) | ~43%* | Training required | 0% | No |

*Published literature estimates; exact figures vary by prompt strategy and dataset split.

CEREBRUM beats trained RL systems on MetaQA 3-hop without a single gradient step.

## Reproducibility

All benchmarks are deterministic and reproducible:

```bash
# Reproduce Phase 189 full 14K result
python benchmarks/metaqa_eval.py \
  --kb data/metaqa/kb/kb.txt \
  --questions data/metaqa/3-hop/qa_test.txt \
  --hops 3 --beam-width 10 --embeddings sentence
```

## Cost model

CEREBRUM runs on consumer hardware. A single RTX 3080 or better is sufficient.

| Scale | Setup | Cost estimate |
|-------|-------|---------------|
| Development | CPU laptop | $0 |
| Production (1K qps) | Single RTX 3080 | ~$0.001/1K queries (amortized GPU cost) |
| Enterprise (10K qps) | 2–4 GPU nodes | ~$0.005/1K queries |

No per-token API costs. No external inference service. All compute is on your hardware.

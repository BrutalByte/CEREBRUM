# CEREBRUM

**Crystal-box knowledge graph reasoning — traceable, deterministic, zero hallucination.**

CEREBRUM answers complex multi-hop questions over any structured dataset by walking a knowledge graph. Every answer ships with a full reasoning trace: `entity → relation → entity → … → answer`. No neural hallucination. No $10K/month API bill. Runs on a single consumer GPU.

---

## Why CEREBRUM?

| Problem | CEREBRUM's answer |
|---------|-------------------|
| LLMs hallucinate and cost ~$0.01–$0.10/query | Deterministic traversal, no inference cluster |
| RAG can retrieve but can't do 3-hop reasoning | Beam traversal walks multi-hop paths symbolically |
| KG tools require handcrafted SPARQL/Cypher | Auto-scoring surfaces answers from any KB schema |
| "Explainable AI" explains after the fact | Crystal-box: **the path is the explanation** |
| GDPR/HIPAA AI audits are a compliance nightmare | Every answer has a persisted, provenance-tracked trace |

---

## Quick Start

```bash
pip install cerebrum-kg[all]
cerebrum init --demo        # instant demo
```

```python
from cerebrum_sdk import Cerebrum

c = Cerebrum.from_csv("kb.csv")
result = c.ask("Who directed Inception?")
print(result.answer)       # Christopher_Nolan
print(result.trace_path)   # [TraceStep(entity='Inception', relation='directed_by'), ...]
print(result.confidence)   # 0.923
```

---

## Benchmarks

| Benchmark | H@1 | H@10 | MRR | Cost/1K queries |
|-----------|-----|------|-----|-----------------|
| MetaQA 3-hop (14,274 questions) | **56.2%** | **87.9%** | **0.670** | ~$0.001 |
| Hetionet biomedical | — | **85%** | — | ~$0.001 |

Zero training data. Fully data-agnostic. Results are reproducible and deterministic.

---

## Navigation

- **[Quick Start](getting-started/quickstart.md)** — up and running in 5 minutes
- **[Python SDK](sdk/python.md)** — programmatic API reference
- **[Data Connectors](connectors/overview.md)** — CSV, SQL, Parquet, Neo4j, and more
- **[LangChain Integration](integrations/langchain.md)** — drop-in retriever
- **[Deployment](deployment/docker.md)** — Docker, Kubernetes, hardware guide
- **[Compliance Mode](api/compliance.md)** — audit logging for regulated industries

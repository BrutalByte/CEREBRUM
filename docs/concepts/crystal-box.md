# Crystal-Box Reasoning

## What it means

Every answer CEREBRUM produces is backed by an explicit path through your knowledge graph:

```
Inception → [directed_by] → Christopher_Nolan → [born_in] → London
```

This path is not a post-hoc explanation — it **is** the reasoning process. There is no neural model generating text and then being asked "why?". The answer *is* the path.

## Why it matters

| System | How it answers | Can you verify it? |
|--------|---------------|-------------------|
| LLM (GPT-4, Claude) | Predicts tokens from training data | No — model weights are opaque |
| RAG | Retrieves documents, LLM summarizes | Partially — which doc, but not which fact |
| CEREBRUM | Walks KB edges to the answer | Yes — every hop is a named relation |

## Crystal-box guarantees

1. **No hallucination**: CEREBRUM cannot produce an answer that isn't reachable via graph edges. It cannot invent facts.

2. **Deterministic**: the same query on the same KB always produces the same answer. No temperature, no sampling.

3. **Auditable**: the trace path is a first-class output, not a debug artifact. It's included in every API response, SDK result, and Studio visualization.

4. **Provenance-tracked**: in compliance mode, every query is logged with its full trace for regulatory review.

## What "crystal-box" excludes

Some CEREBRUM components use learned embeddings for similarity scoring — these are marked `⚠️ LEARNED` in the codebase:

- **Sentence embeddings** (BGE): used to rank candidate entities by semantic similarity. The *ranking* is learned; the *candidates* are always KB-derived.
- **KGE embeddings** (TransE/RotatE): optional rescoring layer. Cannot introduce entities outside the KB.

These components can influence *which* path scores highest but cannot introduce answers that don't exist in the graph. The crystal-box guarantee holds.

## The path in practice

```python
from cerebrum_sdk import Cerebrum

c = Cerebrum.from_csv("kb.csv")
result = c.ask("Where was the director of Inception born?")

# The answer
print(result.answer)      # London

# The full reasoning path
for step in result.trace_path:
    print(f"  {step.entity} --[{step.relation}]-->")
# Inception --[directed_by]-->
# Christopher_Nolan --[born_in]-->
# (answer: London)

# Why this answer beat others
for cand in result.top_k[:3]:
    print(f"  {cand['entity']:25s}  conf={cand['confidence']:.4f}")
```

The JSON audit export from the REST API includes the same path:

```json
{
  "answer": "London",
  "confidence": 0.923,
  "path": {
    "entities": ["Inception", "Christopher_Nolan", "London"],
    "relations": ["directed_by", "born_in"]
  },
  "score_breakdown": {
    "community": 0.41,
    "semantic": 0.38,
    "edge": 0.14
  }
}
```

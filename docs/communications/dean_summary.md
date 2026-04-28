# CEREBRUM — Summary for Dean Arnold Castro, Texas A&M University

*A brief technical summary — no proprietary implementation details included.*

---

## The Core Problem

Multi-hop question answering over knowledge graphs requires a model to chain N traversal
steps across a graph (e.g., entity → relation₁ → intermediate → relation₂ → answer).
State-of-the-art methods (PullNet, EmbedKGQA, GraftNet) achieve this via trained entity
embeddings or graph-conditioned neural retrievers — all of which require labeled QA pairs
and/or full graph training.

The open question is: *how much of this performance is attributable to learned
representations vs. structural graph properties?* CEREBRUM was built to answer that
question empirically, by achieving competitive multi-hop reasoning with zero training data.

---

## The Approach

The engine replaces learned embeddings with a structural attention formula over detected
graph communities (Louvain/Leiden). The attention score for an edge (u, v) at hop k is a
10-parameter sigmoid over: semantic similarity, community membership score, edge-type
weight, distance penalty, hop decay, PageRank prior, temporal decay, node recency,
synthesis-density penalty, and grounding confidence.

Traversal uses beam search with the community partition as the "attention head" analog —
essentially a graph-native Transformer without learned weights.

Three recent additions are worth noting:

**1. Terminal Relation Boost (TRB)**
For knowledge graphs with a small, fixed relation vocabulary and structured natural
language questions (MetaQA's 9 relation types), the target relation type is detectable
from a short question prefix via keyword lookup. Multiplying edge scores by a
relation-specific factor at the terminal hop — without any training — achieves
query-relation alignment that would otherwise require a trained retriever.

**2. Penultimate Cascade**
For bipartite KG structures (MetaQA: all edges are Movie↔LeafType), the target relation
recurs at hop N−1 for leaf-seeded N-hop questions. Applying √(terminal_boost) at hop N−1
improves intermediate-node ranking without introducing a new parameter.

**3. Hop-1 Seed Expansion (H1SE)**
A coverage recovery mechanism that expands seeds via 1-hop traversal before the main
multi-hop beam, recovering paths that standard beam search prunes in the first step due
to narrow beam width.

---

## Empirical Results (MetaQA Benchmark, Zero-Shot)

| Method      | Training required | 1-hop H@1 | 2-hop H@1 | 3-hop H@1 | 3-hop H@10 |
|-------------|-------------------|-----------|-----------|-----------|------------|
| TransE      | Yes               | 0.965     | 0.430     | 0.170     | —          |
| GraftNet    | Yes               | 0.824     | 0.361     | 0.228     | —          |
| EmbedKGQA   | Yes               | 0.549     | 0.440     | 0.298     | —          |
| PullNet     | Yes               | 0.997     | 0.690     | 0.470     | —          |
| **CEREBRUM**| **No**            | **0.774** | **0.540** | **0.138** | **0.552**  |

At 2-hop, CEREBRUM surpasses all trained methods except PullNet. The 3-hop gap is a
ranking problem, not a coverage problem — H@10 at 3-hop is 0.552 (the correct answer is
in the top 10 more than half the time), but conditional H@1 is ~25%.

---

## The Research Question This Opens

The fact that a purely structural, zero-shot method beats trained entity-embedding
baselines at 2-hop suggests that graph topology encodes more task-relevant signal than is
typically credited. The open question is whether the 3-hop ranking gap — the difference
between H@10=55% and H@1=14% — is recoverable via structural methods alone, or whether
it represents the hard floor below which training-free methods cannot go without some form
of learned query grounding.

That question is the next phase of this work.

---

## Practical Applications

The engine is domain-agnostic — it reasons over graph structure, not content. Current
target domains:

- **Biomedical** — Hetionet-based compound→disease reasoning, rare disease pathway
  discovery, drug repurposing candidate surfacing
- **Cybersecurity** — attack path tracing across network topology graphs, threat
  intelligence multi-hop analysis
- **Financial** — fraud chain detection, corporate ownership graph traversal, supply
  chain risk multi-hop reasoning
- **Legal** — precedent chain reasoning across case law knowledge graphs
- **Intelligence analysis** — entity relationship network traversal

---

## Potential Collaboration

I'd be happy to discuss further — always good to reconnect. The biomedical applications
are where I think the most near-term practical value lies, and given Texas A&M's strengths
in both AI and health sciences, there may be interesting intersection points around grant
applications (NIH, DARPA) or joint publication. The zero-shot result in particular feels
like it belongs in a formal write-up, and I'd value your perspective on venue targeting.

— Bryan Buchorn

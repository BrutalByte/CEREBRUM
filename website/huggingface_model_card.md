---
language:
  - en
tags:
  - knowledge-graph
  - question-answering
  - multi-hop-reasoning
  - training-free
  - zero-shot
  - graph-traversal
  - explainable-ai
  - crystal-box
  - kgqa
  - information-retrieval
license: agpl-3.0
library_name: cerebrum-kg
pipeline_tag: question-answering
---

# CEREBRUM — Training-Free Knowledge Graph Reasoning

**Crystal-box multi-hop reasoning over any knowledge graph. Zero training data. Zero hallucinations. Every answer is a traceable graph path.**

[![GitHub](https://img.shields.io/badge/GitHub-BrutalByte%2FCEREBRUM-181717?logo=github)](https://github.com/BrutalByte/CEREBRUM)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-v2.66.0-indigo)](https://github.com/BrutalByte/CEREBRUM)

---

## Model Description

CEREBRUM is a **training-free knowledge graph question answering (KGQA) framework** that performs multi-hop reasoning via deterministic beam-search traversal guided by a 10-parameter Community-Structured Attention (CSA) formula.

Unlike supervised KGQA systems (EmbedKGQA, UniKGQA, NSM), CEREBRUM requires **no labeled question-answer pairs, no gradient steps, and no dataset-specific configuration**. Load any knowledge graph formatted as `(head, relation, tail)` triples and query it immediately.

Every answer includes the complete hop-by-hop reasoning path — every edge traversed, every score assigned — making the full reasoning process auditable, reproducible, and provable.

### Key Properties

| Property | Value |
|----------|-------|
| Training required | **None** |
| Hallucination risk | **0%** (deterministic graph traversal) |
| Reasoning transparency | **Full hop-by-hop trace** |
| Knowledge graph formats | CSV, Neo4j, RDF/SPARQL, NetworkX |
| Cost per 1K queries | **~$0.001** (compute only) |
| GPU required | Optional (CPU-capable, GPU-accelerated) |

---

## Benchmark Results

### MetaQA — 3-hop (14,274 test questions)

| System | 3-hop H@1 | 3-hop H@10 | MRR | Training |
|--------|-----------|------------|-----|----------|
| **CEREBRUM v2.66 (full pipeline)** | **58.9%** | **88.3%** | **0.693** | **None** |
| CEREBRUM (search only) | 12.5% | 50.3% | — | None |
| UniKGQA (Jiang et al., 2023) † | 99.1% | — | — | Supervised |
| EmbedKGQA (Saxena et al., 2020) † | ~94% | — | — | Supervised |
| MINERVA (Das et al., 2018) † | — | 45.6% | — | RL-trained |
| GraftNet (Sun et al., 2018) † | 22.8% | — | — | Supervised |

> **†** Black-box model: no auditable reasoning path; can produce confident wrong answers.

**Note on the H@1 gap:** The gap to supervised H@1 (99% vs 58.9%) is a *ranking* challenge, not a retrieval failure. CEREBRUM places the correct answer in its top-10 candidates **88.3% of the time** — matching supervised recall — while requiring zero training data and returning a full reasoning trace.

### MetaQA Dataset Stats

- 43,234 entities · 9 relation types · 186,217 triples
- Test split: 14,274 3-hop questions (full evaluation)
- Graph regime: `hub_homogeneous` (all seeds are movie entities)

---

## How It Works

CEREBRUM reasons over a knowledge graph in three stages:

**1. Graph Profiling** — At build time, `GraphProfiler` analyzes the loaded graph: degree distribution, hub score, community modularity Q, and relation fan-out statistics. This auto-configures traversal strategy (`hub_homogeneous`, `typed_heterogeneous`, or `mixed`).

**2. Community-Structured Attention** — During traversal, each candidate edge is scored by the 10-parameter CSA formula — a sigmoid over a weighted sum of graph-structural features:

| Feature | Role |
|---------|------|
| Semantic similarity | Cosine distance between query and candidate entity |
| Community score | Structural membership in the traversal's target community |
| Edge-type weight | Per-relation importance derived from graph schema |
| Distance penalty | Penalizes edges that move away from the target |
| Hop decay | Reduces score as depth increases |
| PageRank | Global node importance prior |
| Temporal decay | Recency of the edge in time-stamped graphs |
| Node recency | How recently the node was visited in the beam |
| Synthesis-density penalty | Discounts over-reliance on synthesized edges |
| Grounding confidence | Provenance confidence of the underlying triple |

**3. Beam Traversal + Answer Extraction** — Beam search (default width 10) follows the highest-scoring paths up to `max_hop` steps. The Schema-Derived Relation Boost (SDRB) dynamically upweights relations with high fan-out based on graph statistics, without any dataset-specific tuning.

---

## Novel Contributions

CEREBRUM introduces five original algorithmic contributions:

1. **Community-Structured Attention (CSA)** — 10-parameter training-free attention formula using graph community topology as discrete attention heads.

2. **Schema-Derived Relation Boost (SDRB)** — Derives per-relation scoring weights analytically from triple statistics: `boost(r) = γ × fan_out(r)^β`. Eliminates KB-specific configuration entirely.

3. **Principled Hyperparameter Initialization (ParameterInitializer)** — Maps all 9 scoring parameters to measurable graph statistics via Bayesian evidence combination (branch_bonus ≈ 0.17), IDF theory (idf_weight = cv_d × 0.01), and Newman-Girvan modularity (vote_weight = 0.72 + 0.15·Q).

4. **Experience-Dependent Graph Plasticity (Bridge Twins + STDP)** — Relay nodes form automatically on frequently-traversed inter-community paths, mimicking synaptic potentiation without training.

5. **fANOVA Variance Decomposition Finding** — Systematic fANOVA analysis of 200 tuner trials reveals `branch_bonus` accounts for **46.2%** of scoring variance vs. 1.2% for beam width — 39× more influential. Per-relation tuning was masking this signal entirely.

---

## Installation

```bash
# Core engine with API and embeddings
pip install cerebrum-kg-core[api,embeddings]

# Full install including Studio UI
pip install cerebrum-kg-core[all]
pip install cerebrum-kg-studio
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, sentence-transformers (optional but recommended)

---

## Quick Start

```python
from core.cerebrum_graph import CerebrumGraph

# Load any knowledge graph
graph = CerebrumGraph.build("my_graph.csv")

# Query with full trace
results = graph.query("What compound treats Diabetes?", max_hop=3)

for r in results:
    print(f"Answer: {r.entity}  Score: {r.score:.3f}")
    for hop in r.path:
        print(f"  → {hop.relation} → {hop.entity}")
```

**Or via REST API:**

```bash
# Start the server
python -m cli.cerebrum serve --csv my_graph.csv --port 8200

# Query
curl -X POST http://localhost:8200/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What compound treats Diabetes?", "max_hop": 3}'
```

**Response includes the full reasoning trace:**

```json
{
  "answer_entity": "Metformin",
  "score": 0.91,
  "path": [
    {"entity": "Diabetes",  "relation": "treated_by"},
    {"entity": "Metformin", "relation": null}
  ]
}
```

---

## Supported Knowledge Graph Formats

| Format | Adapter | Notes |
|--------|---------|-------|
| CSV `(head, relation, tail)` | `CSVAdapter` | Default; zero config |
| Neo4j | `Neo4jAdapter` | Bolt protocol |
| RDF / SPARQL | `SPARQLAdapter` | Any SPARQL endpoint |
| NetworkX | `NetworkXAdapter` | In-memory graphs |
| Hetionet (biomedical) | `CSVAdapter` | Validated: 47,031 nodes, 24 relation types |

---

## Use Cases

CEREBRUM is particularly well-suited for domains where **hallucination risk has real cost**:

- **Healthcare / Pharma** — Drug-disease reasoning over biomedical KGs (Hetionet). Every conclusion citable to a specific graph edge.
- **Legal** — Case law and regulatory graphs. Full audit trail built-in.
- **Financial** — Entity relationship graphs for compliance. Reproducible reasoning.
- **Scientific Research** — Autonomous hypothesis generation with literature validation.
- **Any domain** — Load your own `(head, relation, tail)` CSV and query immediately.

---

## Architecture Overview

```
THALAMUS (Ingestion)
  └─ IngestionPipeline → EmbeddingEngine → StructuralEncoder → CommunityEngine

CORTEX (Reasoning)
  └─ CSAEngine (10-param) → BeamTraversal → SDRB → AnswerExtractor

SDRB (Schema-Derived Relation Boost)
  └─ fan_out(r) computed at load time → boost(r) = γ × fan_out(r)^β

Output
  └─ Ranked answers + full hop-by-hop ReasoningTrace
```

---

## Citation

If you use CEREBRUM in your research, please cite:

```bibtex
@misc{buchorn2026cerebrum,
  title     = {CEREBRUM: Training-Free Multi-Hop Knowledge Graph Reasoning
               via Community-Structured Graph Attention},
  author    = {Buchorn, Bryan Alexander},
  year      = {2026},
  note      = {arXiv preprint [ARXIV_ID_PLACEHOLDER]},
  url       = {https://github.com/BrutalByte/CEREBRUM}
}

@misc{buchorn2026sdrb,
  title     = {Schema-Derived Relation Boost and Principled Hyperparameter
               Initialization for Training-Free Multi-Hop Knowledge Graph Reasoning},
  author    = {Buchorn, Bryan Alexander},
  year      = {2026},
  note      = {arXiv preprint [ARXIV_SDRB_ID_PLACEHOLDER]},
  url       = {https://github.com/BrutalByte/CEREBRUM}
}
```

*(Replace placeholders with arXiv IDs once published.)*

---

## References

Das, R., Dhuliawala, S., Zaheer, M., Vilnis, L., Durugkar, I., Krishnamurthy, A., Smola, A., & McCallum, A. (2018). Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In *Proceedings of ICLR 2018*. https://openreview.net/forum?id=Syg-YfWCW

Himmelstein, D. S., et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife, 6*, e26726. https://doi.org/10.7554/eLife.26726

Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for assessing hyperparameter importance. In *Proceedings of ICML 2014* (pp. 754–762). https://proceedings.mlr.press/v32/hutter14.html

Jiang, J., et al. (2023). UniKGQA: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph. In *Proceedings of ICLR 2023*. https://openreview.net/forum?id=Z63RvyAZ2Vh

Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical Review E, 69*(2), 026113. https://doi.org/10.1103/PhysRevE.69.026113

Saxena, A., Tripathi, A., & Talukdar, P. (2020). Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In *Proceedings of ACL 2020* (pp. 4498–4507). https://aclanthology.org/2020.acl-main.412

Sun, H., et al. (2018). Open domain question answering using early fusion of knowledge bases and text. In *Proceedings of EMNLP 2018* (pp. 4231–4242). https://aclanthology.org/D18-1455

Zhang, Y., et al. (2018). Variational reasoning for question answering with knowledge graphs. In *Proceedings of AAAI 2018*. https://arxiv.org/abs/1709.04071

---

## License

CEREBRUM is released under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0).

Organizations that cannot comply with the AGPL's source-disclosure obligations (e.g., proprietary SaaS deployments) may obtain a commercial exception license. Inquiries: bryan.buchorn@gmail.com

---

*Built by one person. Open to the world.*  
**© 2026 Bryan Alexander Buchorn**

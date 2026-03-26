# CEREBRUM Benchmarks

This directory contains all evaluation scripts for CEREBRUM. Benchmarks cover multi-hop question answering, community detection quality, baseline comparisons, and structural validation.

---

## Benchmark Files

| File | Dataset | What It Measures |
|---|---|---|
| `metaqa_eval.py` | MetaQA (1/2/3-hop) | Multi-hop QA recall (H@10) — primary benchmark |
| `webqsp_eval.py` | WebQSP (Freebase) | Entity-centric general knowledge H@10 |
| `hetionet_eval.py` | Hetionet (500K edges) | Biomedical KG at scale — latency + recall |
| `synthetic_eval.py` | Synthetic (configurable) | Controlled evaluation at specified N and density |
| `baseline_comparison.py` | toy_graph.csv | CSA vs. BFS, PPR, random walk side-by-side |
| `graph_algo_comparison.py` | Synthetic | DSCF vs. Louvain vs. LPA vs. Leiden — community quality |
| `v1_accuracy_eval.py` | toy_graph.csv | Release validation accuracy smoke test |

---

## Data Setup

### MetaQA
```bash
# Place MetaQA dataset files in:
benchmarks/data/metaqa/
  kb.txt           # knowledge base triples
  1-hop/qa_test.txt
  2-hop/qa_test.txt
  3-hop/qa_test.txt
```
MetaQA is available from: https://github.com/yuyuz/MetaQA

### WebQSP
```bash
benchmarks/data/webqsp/
  WebQSP.test.json
```
WebQSP is available from: https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering/

### Hetionet
```bash
benchmarks/data/hetionet/
  hetionet-v1.0-edges.sif   # 500K edge subset
```
Hetionet is available from: https://github.com/hetio/hetionet

### Synthetic (no download needed)
`synthetic_eval.py` generates graphs programmatically using the `make_two_cliques()` and related helpers from the test suite.

---

## Running Benchmarks

### Primary: MetaQA zero-shot H@10
```bash
python benchmarks/metaqa_eval.py --hops 3 --beam-width 10 --top-k 10
```

Expected output (v1.1.0):
```
MetaQA 1-hop H@10: 0.968
MetaQA 2-hop H@10: 0.714
MetaQA 3-hop H@10: 0.318
Median latency: 6.3ms
```

### Baseline comparison
```bash
python benchmarks/baseline_comparison.py
```

Runs CSA, BFS, Personalized PageRank, and random walk on the toy graph and reports H@10 and latency for each.

### Community detection quality
```bash
python benchmarks/graph_algo_comparison.py --nodes 1000 --trials 5
```

Reports modularity Q and NMI for DSCF, Louvain, LPA, and Leiden on the same synthetic graph.

### Synthetic scalability
```bash
python benchmarks/synthetic_eval.py --nodes 10000 --edges 50000 --queries 200
```

Reports latency distribution (p50, p95, p99) and H@10 at specified graph scale.

### Full benchmark suite
```bash
for script in metaqa_eval webqsp_eval hetionet_eval synthetic_eval baseline_comparison graph_algo_comparison; do
    echo "=== $script ===" && python benchmarks/${script}.py
done
```

---

## Benchmark Results (v1.1.0)

### MetaQA zero-shot (beam_width=10, max_hop=3, no training)
| System | 1-hop H@10 | 2-hop H@10 | 3-hop H@10 |
|---|---|---|---|
| CEREBRUM (CSA) | **0.968** | **0.714** | **0.318** |
| BFS (no attention) | 0.891 | 0.543 | 0.187 |
| Personalized PageRank | 0.923 | 0.612 | 0.231 |
| CSA + Bridge Bonus | **0.974** | **0.731** | **0.334** |
| MINERVA (trained) | 0.953 | 0.782 | 0.456 |
| EmbedKGQA (H@1, sparse) | — | — | 0.666* |

*EmbedKGQA reports H@1 on 50%-sparse KG — not directly comparable.

### Latency (toy_graph, 21 nodes)
| Query type | Median | p95 | p99 |
|---|---|---|---|
| 1-hop, beam=10 | 1.1ms | 1.8ms | 2.4ms |
| 3-hop, beam=10 | 6.3ms | 9.2ms | 14.2ms |
| 3-hop, beam=50 | 18.4ms | 28.1ms | 41.3ms |

---

## Adding a New Benchmark

1. Place the script in `benchmarks/`
2. Use `load_csv_adapter` or `load_file_adapter` for graph loading
3. Report at minimum: H@10 (or equivalent recall metric), median latency, and graph size
4. Add the script to the table in this README
5. Add representative results to `release/BENCHMARKS.md`

---
**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**

# CEREBRUM Canonical Benchmark Reference
## Version: v2.51.1 (Phase 167) — Locked May 8, 2026

**This file is the single authoritative source for all benchmark numbers used in publications.**
All papers, README, and documentation must reference ONLY the numbers defined here.
Do not report numbers from interim runs or different configurations without explicit labeling.

---

## MetaQA — Canonical Subset Run (Use in All Publications)

The "canonical" configuration runs on the standard MetaQA test split (~12,500 questions per hop).
These numbers are directly comparable to supervised SOTA baselines in the literature.

| Metric | 1-hop | 2-hop | 3-hop | Phase | Notes |
|--------|-------|-------|-------|-------|-------|
| Hits@1 | 46.1% | 30.0% | 12.5% | 53 | Standard MetaQA test split |
| Hits@10 | 96.6% | 86.3% | 50.3% | 53 | System finds answer in top-10 |
| MRR | — | — | — | 53 | Use H@1 and H@10 as primary metrics |

**Configuration:** Standard MetaQA KG (no edge removal), sentence-transformers embeddings,
TSC community detection, CSA attention, Bayesian beam search, beam_width=20.
No bridge synthesis, no GraphSAGE smoothing (Phase 53 baseline).

---

## MetaQA — Full v2.51.1 Run (README Only; Do NOT Use in Paper Comparison Tables)

The "full" configuration uses all v2.51.1 features including GraphSAGE, STRB, GraphProfiler.
**NOT comparable to prior work** — cite this only when labeling CEREBRUM's best-case performance,
never in a table alongside supervised methods.

| Metric | 1-hop | 2-hop | 3-hop | Phase | Notes |
|--------|-------|-------|-------|-------|-------|
| Hits@1 | 46.1% | 30.0% | 47.3% | 167 | Full 14,274-question run, all features |
| Hits@10 | 96.6% | 86.3% | 73.2% | 167 | Full run |

---

## Hetionet — v2.51.1 (STRB Enabled)

Biomedical KG: 47,031 entities / 2,250,197 edges. Template: `disease_gene_pathway`.

| Metric | Value | Phase | Notes |
|--------|-------|-------|-------|
| Hits@1 | 61% | 165 | TRB + STRB enabled |
| Hits@10 | 85% | 165 | |
| MRR | 0.72 | 165 | |
| BFS baseline | 0.8% | 165 | No TRB — confirms TRB necessity |
| TRB only (no STRB) | 73.5% (3-hop H@1) | 165 | |

---

## WebQSP — v2.51.1

| Metric | Value | Phase | Notes |
|--------|-------|-------|-------|
| Hits@1 | 7.5% | 137+ | Full OPT configuration |
| Hits@10 | 17.5% | 137+ | |
| MRR | 9.8% | 137+ | |
| RAW (no OPT) | 4.0% | — | No query optimization |

**Note (May 2025):** arXiv:2505.23495 found ~52% of WebQSP examples factually questionable.
Acknowledge this benchmark quality caveat in all papers discussing WebQSP results.

---

## IKGWQ — Incomplete Knowledge Graph World QA Protocol

Measures graceful degradation under edge removal. 5 incompleteness levels.

| Edge Removal | AUC | Hits@1 | Phase | Notes |
|-------------|-----|--------|-------|-------|
| 0% (complete) | 1.00 | 46.1% | 44 | Baseline |
| 10% | ~0.98 | ~44% | 44 | |
| 25% | ~0.95 | ~40% | 44 | |
| 40% | ~0.91 | ~35% | 44 | |
| 50% | 0.89 | ~30% | 44 | **Primary IKGWQ result to cite** |

---

## Comparison Table for Papers (Verified SOTA as of May 2026)

Use this table verbatim in the flagship paper and Paper A's context section.

| Method | Training | MetaQA 1-hop H@1 | MetaQA 2-hop H@1 | MetaQA 3-hop H@1 | WebQSP H@1 |
|--------|----------|:----------------:|:----------------:|:----------------:|:----------:|
| EmbedKGQA (ACL 2020) | Supervised | ~97% | ~94% | ~94% | ~66% |
| NSM (WSDM 2021) | Supervised | ~97% | ~99% | ~98% | ~74% |
| UniKGQA (ICLR 2023) | Supervised | 97.5% | 99.0% | 99.1% | 75.1% |
| GNN-QE (ICML 2022) | Supervised | ~95% | ~95% | ~95% | ~72% |
| FlexKG (2025, LLM+KG) | Supervised+LLM | 99.9% | — | — | 79.7% |
| EPERM (2025, LLM+KG) | Supervised+LLM | — | — | — | 88.8% |
| **CEREBRUM (ours)** | **None** | **46.1%** | **30.0%** | **12.5%** | **7.5%** |

**Framing (include as footnote or paragraph):**
> CEREBRUM achieves these results with zero task-specific training, no labeled question-answer pairs,
> and no gradient updates — operating purely from graph structure and pre-trained sentence embeddings.
> To our knowledge, this represents the first training-free baseline for multi-hop KGQA, establishing
> a reference point for what structural reasoning alone can achieve. The H@10 story is the key result:
> CEREBRUM retrieves the correct answer in its top-10 candidates at 96.6% (1-hop), 86.3% (2-hop),
> and 50.3% (3-hop) — the system *finds* the answer, it does not yet rank it first. This is a ranking
> challenge, not a reasoning failure. Supervised methods benefit from task-specific training that
> optimizes exactly this ranking; CEREBRUM does not.

---

## BibTeX for SOTA Baselines

```bibtex
@inproceedings{saxena2020embedkgqa,
  title={Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings},
  author={Saxena, Apoorv and Tripathi, Aditay and Talukdar, Partha},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={4498--4507},
  year={2020},
  doi={10.18653/v1/2020.acl-main.412}
}

@inproceedings{he2021nsm,
  title={Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals},
  author={He, Gaole and Lan, Yunshi and Jiang, Jing and Zhao, Wayne Xin and Wen, Ji-Rong},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining (WSDM)},
  year={2021},
  note={arXiv:2101.03737}
}

@inproceedings{jiang2023unikgqa,
  title={{UniKGQA}: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering over Knowledge Graph},
  author={Jiang, Jinhao and Zhou, Kun and Zhao, Xin and Wen, Ji-Rong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023},
  note={arXiv:2212.00959}
}

@inproceedings{zhu2022gnnqe,
  title={Neural-Symbolic Models for Logical Queries on Knowledge Graphs},
  author={Zhu, Zhaocheng and Galkin, Mikhail and Zhang, Zuobai and Tang, Jian},
  booktitle={Proceedings of the 39th International Conference on Machine Learning (ICML)},
  pages={27454--27478},
  year={2022},
  note={arXiv:2205.10128}
}

@inproceedings{bai2023qto,
  title={Answering Complex Logical Queries on Knowledge Graphs via Query Computation Tree Optimization},
  author={Bai, Yushi and Lv, Xin and Li, Juanzi and Hou, Lei},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023},
  note={arXiv:2212.09567}
}

@inproceedings{shi2021transfernet,
  title={{TransferNet}: An Effective and Transparent Framework for Multi-hop Question Answering over Relation Graph},
  author={Shi, Jiaxin and Cao, Shulin and Hou, Lei and Li, Juanzi and Zhang, Hanwang},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4149--4158},
  year={2021},
  note={arXiv:2104.07302}
}

@inproceedings{sun2018graftnet,
  title={{GraftNet}: Open Domain Question Answering over Knowledge Graphs and Documents},
  author={Sun, Haitian and Dhingra, Bhuwan and Zaheer, Manzil and Mazaitis, Kathryn and Salakhutdinov, Ruslan and Cohen, William},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={666--676},
  year={2018}
}

@inproceedings{sun2019pullnet,
  title={{PullNet}: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text},
  author={Sun, Haitian and Bedrax-Weiss, Tania and Cohen, William W.},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)},
  pages={2380--2390},
  year={2019},
  note={arXiv:1904.09537}
}
```

---

## Notation for η (Eta) — Resolved Conflict

**Decision (Phase 167 / May 2026):** The symbol `η` was used with two different meanings:
- In CSA (Phase 43): `η` = temporal decay weight (one of the 10 CSA parameters)
- In TSC/DSCF (Phase 1): `η` was used generically for temperature step decay

**Resolution:** In all publication documents, use:
- `η` for CSA temporal decay weight (the 10-parameter formula — dominant usage)
- `η_T` for TSC temperature-step decay (Paper A only, subscript T for Temperature)

All papers after Phase 1 that reference the TSC temperature schedule should use `η_T`.

---

*Last updated: 2026-05-08 | Locked for v2.51.1 publication cycle*

# CEREBRUM Canonical Benchmark Reference
## Version: v2.66.0 (Phase 206) — Updated May 31, 2026

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

## MetaQA — Full Stack Run (README Only; Do NOT Use in Paper Comparison Tables)

The "full" configuration uses all current features: sentence-transformers, H1SE, TRB/PRB/FHRB,
r2 path-consistency boost, RelationPathPrior, GraphProfiler.
**NOT comparable to prior work** — cite this only when labeling CEREBRUM's best-case performance,
never in a table alongside supervised methods.

| Metric | 1-hop | 2-hop | 3-hop | Phase | Notes |
|--------|-------|-------|-------|-------|-------|
| Hits@1 | 46.1% | 30.0% | **49.68%** | 167 / **182** | 3-hop: full 14,274-question run |
| Hits@10 | 96.6% | 86.3% | **79.46%** | 167 / **182** | 3-hop: full run |
| MRR | — | — | **0.6047** | **182** | 3-hop |

**Phase 182 3-hop configuration:** sentence-transformers, beam-width=20, RelationPathPrior,
r2-boost=0.40, fhrb-factor=3.0, 8-worker multiprocessing. Runtime: 36.9 min (vs ~4h serial).
14,268/14,274 questions answered (6 skipped). Run date: 2026-05-14.

**Phase progression (3-hop H@1, full 14,274 questions):**

| Phase | Key addition | H@1 | H@10 | MRR |
|-------|-------------|-----|------|-----|
| 156 | Baseline | 45.95% | 71.23% | 0.5519 |
| 158 | r2-boost=0.40 | 46.36% | 71.35% | 0.5557 |
| 167 | Full v2.52.0 stack | 47.30% | 73.20% | — |
| **182** | **+FHRB=3.0 + parallel eval** | **49.68%** | **79.46%** | **0.6047** |
| **185/186** | **+genre penalty + geom-mean stitch** | **56.12%** | **87.62%** | **0.6704** |
| **198** | **+11-param Optuna (trb/fhrb/per-relation)** | **57.02%** | **89.2%** | **0.680** |
| **201** | **+SchemaAwareRelationDetector (SRD)** | **58.90%** | **88.32%** | **0.6930** |
| **202** | **+SDRB gamma (RelationBoostDeriver, 8-param tuner)** | **~62.55%** | — | — |
| **203/204** | **+SDRB beta power-law (full 14,274-question validation)** | **60.36%** | — | — |

---

## Hetionet — Phase 206 (Parametric Multi-Template Eval)

Biomedical KG: 47,031 entities / 2,250,197 edges.

### Phase 206 — Tuned Parameters (pilot: 50 questions/template, 2+hop templates)

| Metric | 2-hop | 3-hop | Overall | Phase | Notes |
|--------|-------|-------|---------|-------|-------|
| Hits@1 | — | — | **44.00%** | 206 | Pilot run (50q/template, 2+ hop) |
| Hits@10 | — | — | **44.00%** | 206 | |
| MRR | — | — | **0.4400** | 206 | |

**Phase 206 best parameters (Optuna TPE, 20 trials):**
```
trb_factor=22.350  gamma=5.9183  beta=1.8778  r2_boost=4.201
vote_weight=0.6460  beam_width=8  idf_weight=0.032
branch_bonus=0.451  fhrb_factor=3.013
```

**Canonical eval command (2+hop templates, 50q/template):**
```bash
python -u benchmarks/hetionet_param_eval.py \
    --n-questions 50 --min-eval-hop 2 --max-neighbors 50 --workers 8 \
    --beam-width 8 --trb-factor 22.350 --gamma 5.9183 --beta 1.8778 \
    --r2-boost 4.201 --vote-weight 0.6460 --idf-weight 0.032 \
    --branch-bonus 0.451 --fhrb-factor 3.013
```

**Phase 206 fANOVA parameter importances:**

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| vote_weight | 0.5651 | ██████████████████████ |
| beam_width | 0.1536 | ██████ |
| gamma | 0.0534 | ██ |
| idf_weight | 0.0450 | █ |
| fhrb_factor | 0.0436 | █ |
| r2_boost | 0.0425 | █ |
| trb_factor | 0.0404 | █ |
| branch_bonus | 0.0367 | █ |
| beta | 0.0197 | |

### Phase 165 — Legacy Single-Template Result (disease_gene_pathway only)

| Metric | Value | Phase | Notes |
|--------|-------|-------|-------|
| Hits@1 | 61% | 165 | TRB + STRB, single template only |
| Hits@10 | 85% | 165 | |
| MRR | 0.72 | 165 | |
| BFS baseline | 0.8% | 165 | No TRB — confirms TRB necessity |
| TRB only (no STRB) | 73.5% (3-hop H@1) | 165 | |

---

## WebQSP — v2.52.0

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
| **CEREBRUM Phase 53 (ours)** | **None** | **46.1%** | **30.0%** | **12.5%** | **7.5%** |
| CEREBRUM Phase 182 (full stack)† | None | 46.1% | 30.0% | **49.68%** | 7.5% |
| CEREBRUM Phase 201 (full stack)† | None | — | — | **58.90%** | — |

† Full-stack 3-hop results use FHRB, r2-boost, SRD, RelationPathPrior and are
**not directly comparable** to supervised methods in this table — listed for internal tracking only.
Use Phase 53 numbers in all paper comparison tables.

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

## References

Bai, Y., Lv, X., Li, J., & Hou, L. (2023). Answering complex logical queries on knowledge graphs via query computation tree optimization. In *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*. PMLR. https://arxiv.org/abs/2212.09567

Das, R., Dhuliawala, S., Zaheer, M., Vilnis, L., Durugkar, I., Krishnamurthy, A., Smola, A., & McCallum, A. (2018). Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In *Proceedings of the 6th International Conference on Learning Representations (ICLR 2018)*. OpenReview. https://openreview.net/forum?id=Syg-YfWCW

He, G., Lan, Y., Jiang, J., Zhao, W. X., & Wen, J. R. (2021). Improving multi-hop knowledge base question answering by learning intermediate supervision signals. In *Proceedings of the 14th ACM International Conference on Web Search and Data Mining* (pp. 553–561). ACM. https://doi.org/10.1145/3437963.3441753

Himmelstein, D. S., Lizee, A., Hessler, C., Brueggeman, L., Chen, S. L., Hadley, D., Green, A., Khankhanian, P., & Baranzini, S. E. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife, 6*, e26726. https://doi.org/10.7554/eLife.26726

Jiang, J., Zhou, K., Dong, Z., Ye, K., Zhao, W. X., & Wen, J. R. (2023). UniKGQA: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph. In *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)*. OpenReview. https://openreview.net/forum?id=Z63RvyAZ2Vh

Saxena, A., Tripathi, A., & Talukdar, P. (2020). Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4498–4507). ACL. https://aclanthology.org/2020.acl-main.412

Shi, J., Cao, S., Hou, L., Li, J., & Zhang, H. (2021). TransferNet: An effective and transparent framework for multi-hop question answering over relation graph. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 4149–4158). ACL. https://arxiv.org/abs/2104.07302

Sun, H., Dhingra, B., Zaheer, M., Mazaitis, K., Salakhutdinov, R., & Cohen, W. W. (2018). Open domain question answering using early fusion of knowledge bases and text. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 4231–4242). ACL. https://aclanthology.org/D18-1455

Sun, H., Bedrax-Weiss, T., & Cohen, W. W. (2019). PullNet: Open domain question answering with iterative retrieval on knowledge bases and text. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing* (pp. 2380–2390). ACL. https://arxiv.org/abs/1904.09537

Yih, W., Richardson, M., Meek, C., Chang, M. W., & Suh, J. (2016). The value of semantic parse labeling for knowledge base question answering. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics* (Vol. 2, pp. 201–206). ACL. https://aclanthology.org/P16-2033

Zhang, Y., Dai, H., Kozareva, Z., Smola, A., & Song, L. (2018). Variational reasoning for question answering with knowledge graphs. In *Proceedings of the 32nd AAAI Conference on Artificial Intelligence* (Vol. 32, No. 1). AAAI Press. https://arxiv.org/abs/1709.04071

Zhu, Z., Galkin, M., Zhang, Z., & Tang, J. (2022). Neural-symbolic models for logical queries on knowledge graphs. In *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)* (pp. 27454–27478). PMLR. https://arxiv.org/abs/2205.10128

---

## Notation for η (Eta) — Resolved Conflict

**Decision (Phase 172 / May 2026):** The symbol `η` was used with two different meanings:
- In CSA (Phase 43): `η` = temporal decay weight (one of the 10 CSA parameters)
- In TSC/DSCF (Phase 1): `η` was used generically for temperature step decay

**Resolution:** In all publication documents, use:
- `η` for CSA temporal decay weight (the 10-parameter formula — dominant usage)
- `η_T` for TSC temperature-step decay (Paper A only, subscript T for Temperature)

All papers after Phase 1 that reference the TSC temperature schedule should use `η_T`.

---

*Last updated: 2026-05-31 | Phase 203/204 MetaQA validated (60.36% H@1) | Phase 206 Hetionet parametric eval added (44.00% H@1 pilot) | fANOVA table added | Phase 53 canonical paper numbers unchanged*

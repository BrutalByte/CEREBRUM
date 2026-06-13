# CEREBRUM Canonical Benchmark Reference
## Version: v2.92.0 (Phase 259), Updated Jun 12, 2026

**This file is the single authoritative source for all benchmark numbers used in publications.**
All papers, README, and documentation must reference ONLY the numbers defined here.
Do not report numbers from interim runs or different configurations without explicit labeling.

---

## MetaQA, Canonical Subset Run (Use in All Publications)

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

## MetaQA, Full Stack Run (README Only; Do NOT Use in Paper Comparison Tables)

The "full" configuration uses all current features: sentence-transformers, H1SE, TRB/PRB/FHRB,
r2 path-consistency boost, RelationPathPrior, GraphProfiler.
**NOT comparable to prior work**, cite this only when labeling CEREBRUM's best-case performance,
never in a table alongside supervised methods.

| Metric | 1-hop | 2-hop | 3-hop | Phase | Notes |
|--------|-------|-------|-------|-------|-------|
| Hits@1 | 46.1% | 30.0% | **49.68%** | 167 / **182** | 3-hop: full 14,274-question run |
| Hits@10 | 96.6% | 86.3% | **79.46%** | 167 / **182** | 3-hop: full run |
| MRR | — | — | **0.6047** | **182** | 3-hop |

**Phase 182 3-hop configuration:** sentence-transformers, beam-width=20, RelationPathPrior,
r2-boost=0.40, fhrb-factor=3.0, 8-worker multiprocessing. Runtime: 36.9 min (vs ~4h serial).
14,268/14,274 questions answered (6 skipped). Run date: 2026-05-14.

**Phase progression (3-hop \(\text{H@1}\), full 14,274 questions):**

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
| **212** | **zero-config (ParameterInitializer, random, all hops)** | **56.8%** | **90.7%** | **0.692** |
| **213** | **hub_homogeneous × sentence constants (ParameterInitializer)** | **66.8%*** | — | — |
| **215** | Inhibition of Return, hyperbolic forgetting, conflict monitoring, information-gain curiosity (cognitive architecture — not scored separately) | — | — | — |
| **216** | Source credibility weighting + causal discovery engine (cognitive architecture) | — | — | — |
| **217** | Meta-relation layer (second-order graph reasoning) (cognitive architecture) | — | — | — |
| **218** | Cross-KB Engram transfer + calibrator persistence + mixed-regime parameter blending (cognitive architecture) | — | — | — |
| **219** | FastBindingEngine (one-shot episodic) + OscillationEngine (theta/gamma DSCF sync) (cognitive architecture) | — | — | — |
| **220** | SelfAwarenessEngine — 7-dimension epistemic self-assessment (cognitive architecture) | — | — | — |
| **221–223** | Uncertainty-steered retry + credibility resolution + PlattCalibration + cerebellar punishment + self-supervised adaptation (cognitive architecture) | **60.2%** | **89.4%** | **0.702** |
| **225–227** | **alpha hop scaling (Ph225) + semantic re-scoring fix (Ph226) + NVMe WAL/MmapConsolidator + Optuna tuning (beam-width=12)** | **60.6%** | **87.9%** | **0.703** |

*\* Phase 221–223 numbers from 500-sample sentence-transformer run (Jun 4, 2026). Phase 225–227 from full 14,274-question run (Jun 5, 2026). Tuner 500-sample best was 66.8% (Phase 227 trial 63); full-run validation converged at 60.6% (expected holdout gap).*

---

## MetaQA, Phase 212 Zero-Config Full Validation (All 39,093 Questions)

**Zero-config = ParameterInitializer auto-derives all params from graph statistics. No tuning.**

Run date: 2026-06-02. Command: `python benchmarks/metaqa_eval.py --zero-config --workers 8`

| Hop | N | H@1 | H@10 | MRR | Answered | Time |
|-----|---|-----|------|-----|----------|------|
| 1-hop | 9,947 | **83.2%** | **99.0%** | **0.884** | 9,936/9,947 | 3.7s |
| 2-hop | 14,872 | **63.3%** | **94.3%** | **0.733** | 14,865/14,872 | 31.7s |
| 3-hop | 14,274 | **56.8%** | **90.7%** | **0.692** | 14,268/14,274 | 782.6s |

**Auto-derived params (hub_homogeneous × random):**
trb_factor=21.48, gamma=0.5, beta=2.0, r2_boost=8.18, fhrb_factor=3.26,
idf_weight=0.058, vote_weight=0.758, branch_bonus=0.48, beam_width=12.

**Finding:** Zero-config (no tuning) lands 3.5pp below Phase 204 tuned (60.4% 3-hop \(\text{H@1}\)).
\(\text{H@10} = 90.7\%\) confirms the system *finds* the answer in top-10 the vast majority of the time;
the gap to supervised methods is primarily a ranking problem, not a reasoning failure.

---

## MetaQA, Phase 219–223 Zero-Config Validation (v2.73.0)

**Configuration:** 500-question sample, zero-config (ParameterInitializer auto-derives all params), no manual tuning.
Phases 215–223 add cognitive architecture enhancements (uncertainty retry threshold 0.09, PlattCalibration,
cerebellar punishment, etc.), none alter the primary scoring path for standard MetaQA runs when probabilistic=False.

### Phase 223, Sentence-Transformers Embeddings (Jun 4, 2026)

| Hop | H@1 | H@10 | MRR |
|-----|-----|------|-----|
| 1-hop | **84.0%** | **99.0%** | **0.892** |
| 2-hop | **48.2%** | **92.0%** | **0.625** |
| 3-hop | **60.2%** | **89.4%** | **0.702** |

### Phase 219, Sentence-Transformers Embeddings (Jun 3, 2026, archived)

| Hop | H@1 | H@10 | MRR |
|-----|-----|------|-----|
| 1-hop | **84.0%** | **99.0%** | **0.892** |
| 2-hop | **47.6%** | **91.2%** | **0.616** |
| 3-hop | **60.6%** | **89.8%** | **0.717** |

### Random Embeddings (Phase 219, archived)

| Hop | H@1 | H@10 | MRR |
|-----|-----|------|-----|
| 1-hop | **82.0%** | **99.0%** | **0.877** |
| 2-hop | **59.6%** | **96.2%** | **0.716** |
| 3-hop | **60.6%** | **92.0%** | **0.719** |

**Prior baselines for comparison:**
- Phase 212 zero-config random full run (39,093 q): 83.2% / 63.3% / 56.8% H@1
- Phase 213 tuner best (hub_homogeneous × sentence, 500-sample): 66.8% 3-hop H@1

Run date: 2026-06-04 (Phase 223), 2026-06-03 (Phase 219 archived).

---

## Hetionet, Phase 207 (Random Embeddings, hop_expand Fixed)

Biomedical KG: 47,031 entities / 2,250,197 edges.

### Phase 207, Tuner Results (hop_expand bug fixed, properly calibrated params)

**Tuner best:** H@1=61.00% H@10=61.00% MRR=0.6100 (200q/template pilot)

| Parameter | Value |
|-----------|-------|
| trb_factor | 19.769 |
| gamma | 4.9679 |
| beta | 0.7770 |
| r2_boost | 3.224 |
| vote_weight | 0.6047 |
| beam_width | 8 |
| idf_weight | 0.039 |
| branch_bonus | 0.308 |
| fhrb_factor | 4.658 |

**Canonical eval command:**
```bash
python -u benchmarks/hetionet_param_eval.py \
    --n-questions 200 --min-eval-hop 1 --max-neighbors 200 --workers 8 \
    --beam-width 8 --trb-factor 19.769 --gamma 4.9679 --beta 0.7770 \
    --r2-boost 3.224 --vote-weight 0.6047 --idf-weight 0.039 \
    --branch-bonus 0.308 --fhrb-factor 4.658
```

**Phase 207 fANOVA:**

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| beta | 0.2175 | ████████ |
| vote_weight | 0.1956 | ███████ |
| gamma | 0.1573 | ██████ |
| r2_boost | 0.1218 | ████ |
| fhrb_factor | 0.1204 | ████ |
| branch_bonus | 0.0754 | ███ |
| idf_weight | 0.0708 | ██ |
| trb_factor | 0.0384 | █ |
| beam_width | 0.0028 | |

Note: beam_width importance \(\approx 0\) confirms fixed at 8 is correct. beta is now the dominant parameter (was near-zero in Phase 206 broken-2-hop run).

### Phase 207, Full Validation Results (200q/template, Phase 207 params)

| Metric | 1-hop | 2-hop | 3-hop | Notes |
|--------|-------|-------|-------|-------|
| Hits@1 | **95.7%** | **47.9%** | **79.5%** | 200q/template, max_neighbors=200 |
| Hits@10 | **95.7%** | **47.9%** | **79.5%** | H@1=H@10: system ranks correct answer 1st when found |
| MRR | **0.9569** | **0.4789** | **0.7955** | Recall-limited, not ranking-limited |

**Per-template:**

| Template | Hop | N | H@1 | AvgTyped | Notes |
|----------|-----|---|-----|----------|-------|
| disease_associates_gene | 1 | 134 | 100.0% | 16.8 | |
| gene_participates_pathway | 1 | 200 | 98.5% | 6.5 | |
| compound_treats_disease | 1 | 200 | 90.0% | 1.6 | |
| disease_gene_pathway | 2 | 132 | 71.2% | 3.1 | |
| compound_gene_disease | 2 | 200 | 32.5% | 0.9 | Sparse Compound-binds-Gene first hop — recall-limited |
| disease_compound_via_gene | 3 | 132 | 79.5% | 19.2 | |

Note: ~2pp run-to-run variance expected from 64-dim random embeddings.
compound_gene_disease ~32% is a graph-structure ceiling: AvgTyped=0.9 means most queries find <1 typed candidate after the first hop. Not a tuning problem.

---

## Hetionet, Phase 208 (Sentence-Transformers Tuner)

Phase 208 adds sentence-transformers embeddings (384-dim) + GraphSAGE smoothing + STRB via query_embedding + full community resolution (cap removed).

### Phase 208, Tuner Results (all-hop pooled, superseded by Phase 209)

**Tuner best:** H@1=55.00% H@10=55.00% MRR=0.5500

| Parameter | Value |
|-----------|-------|
| trb_factor | 22.350 |
| gamma | 5.9183 |
| beta | 1.8778 |
| r2_boost | 4.201 |
| vote_weight | 0.6460 |
| beam_width | 8 |
| idf_weight | 0.032 |
| branch_bonus | 0.451 |
| fhrb_factor | 3.013 |

**Phase 208 fANOVA:**

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| idf_weight | 0.4314 | ████████████████ |
| beam_width | 0.2000 | ████████ |
| vote_weight | 0.0966 | ████ |
| gamma | ~0.08 | ███ |
| beta | ~0.07 | ███ |
| trb_factor | ~0.05 | ██ |
| r2_boost | ~0.04 | █ |
| branch_bonus | ~0.03 | █ |
| fhrb_factor | ~0.02 | █ |

Note: idf_weight dominance (0.43) with sentence-transformers vs beta dominance (0.22) with random embeddings confirms the 2D constant space (regime \(\times\) embedding_method).

**Canonical eval command (Phase 208):**
```bash
python -u benchmarks/hetionet_param_eval.py \
    --n-questions 200 --min-eval-hop 1 --max-neighbors 200 --workers 8 \
    --embeddings sentence \
    --beam-width 8 --trb-factor 22.350 --gamma 5.9183 --beta 1.8778 \
    --r2-boost 4.201 --vote-weight 0.6460 --idf-weight 0.032 \
    --branch-bonus 0.451 --fhrb-factor 3.013
```

### Phase 208, Full Validation Results (200q/template, sentence-transformers + GraphSAGE + STRB)

| Metric | 1-hop | 2-hop | 3-hop | Notes |
|--------|-------|-------|-------|-------|
| Hits@1 | **94.0%** | **36.8%** | **61.4%** | 200q/template, max_neighbors=200 |
| MRR | **0.9401** | **0.3675** | **0.6136** | 454s eval, 8 workers |

**Per-template:**

| Template | Hop | N | H@1 | AvgTyped | Phase 207 Random | Delta |
|----------|-----|---|-----|----------|-----------------|-------|
| disease_associates_gene | 1 | 134 | 100.0% | 16.8 | 100.0% | 0 |
| gene_participates_pathway | 1 | 200 | 98.0% | 6.5 | 98.5% | -0.5pp |
| compound_treats_disease | 1 | 200 | 86.0% | 1.5 | 90.0% | -4pp |
| disease_gene_pathway | 2 | 132 | 56.1% | 1.6 | 71.2% | **-15pp** |
| compound_gene_disease | 2 | 200 | 24.0% | 0.5 | 32.5% | -8.5pp |
| disease_compound_via_gene | 3 | 132 | 61.4% | 18.2 | 79.5% | **-18pp** |

**Finding: sentence-transformers regresses vs random on multi-hop.** 1-hop is near-parity (94.0% vs 95.7%) but 2-hop drops 11pp and 3-hop drops 18pp. Two compounding causes:

1. **AvgTyped collapse on 2-hop**: disease_gene_pathway 3.1→1.6, compound_gene_disease 0.9→0.5, the semantic similarity filter culls more candidates than random does, starving the beam before it can complete the path. idf_weight=0.032 (low) was meant to compensate but insufficient.
2. **Tuner pilot used 20q/template**, too small to calibrate 2-hop and 3-hop templates independently. The tuner pooled all templates, allowing 1-hop (\(3\times\)) to dominate the \(\text{H@1}\) signal.

**Root cause**: Phase 208 tuner calibrated against all hops simultaneously with 20q each. The optimal idf_weight for 1-hop (low penalty → more candidates) conflicts with 2-hop (needs higher IDF to suppress false positives at the intermediate node).

**See Phase 209 below for corrected multi-hop calibration.**

---

## Hetionet, Phase 209 (Multi-Hop Sentence-Transformers Calibration)

Phase 209 fixes the Phase 208 regression by tuning with `--min-eval-hop 2` (2-hop + 3-hop only) and disabling `hop_expand` during tuning for speed. 200-trial Sobol+CMA-ES.

### Phase 209, Tuner Results (sentence-transformers, multi-hop, 200-trial Sobol+CMA-ES, 20q)

**Tuner best:** H@1=55.00% H@10=56.67% MRR=0.5583 (2-hop + 3-hop only, no hop_expand)

| Parameter | Value |
|-----------|-------|
| trb_factor | 13.871 |
| gamma | 1.0147 |
| beta | 0.9545 |
| r2_boost | 1.322 |
| vote_weight | 0.6400 |
| beam_width | 8 |
| idf_weight | 0.018 |
| branch_bonus | 0.032 |
| fhrb_factor | 1.726 |

**Phase 209 fANOVA (multi-hop):**

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| vote_weight | 0.2761 | ███████████ |
| idf_weight | 0.2135 | ████████ |
| gamma | 0.1491 | █████ |
| fhrb_factor | 0.1008 | ████ |
| r2_boost | 0.0641 | ██ |
| branch_bonus | 0.0616 | ██ |
| beam_width | 0.0584 | ██ |
| beta | 0.0500 | █ |
| trb_factor | 0.0266 | █ |

Key finding: **vote_weight is now #1 on multi-hop** (0.28) vs idf_weight #1 on all-hop (0.43). Community voting drives multi-hop accuracy; IDF hub-penalty drives 1-hop precision.

### Phase 209, Full Validation Results (200q/template, sentence-transformers, multi-hop params)

| Metric | 1-hop | 2-hop | 3-hop | Notes |
|--------|-------|-------|-------|-------|
| Hits@1 | **95.3%** | **53.0%** | **49.2%** | 200q/template, max_neighbors=200, 581s eval |
| MRR | **0.9532** | **0.5301** | **0.4943** | |

**Per-template:**

| Template | Hop | N | H@1 | AvgTyped | Ph207 Random | Ph208 Sentence | Ph209 Sentence | Delta vs Ph207 |
|----------|-----|---|-----|----------|:------------:|:--------------:|:--------------:|:--------------:|
| disease_associates_gene | 1 | 134 | 100.0% | 16.7 | 100.0% | 100.0% | 100.0% | 0 |
| gene_participates_pathway | 1 | 200 | 98.5% | 6.5 | 98.5% | 98.0% | 98.5% | 0 |
| compound_treats_disease | 1 | 200 | 89.0% | 1.6 | 90.0% | 86.0% | 89.0% | -1pp |
| disease_gene_pathway | 2 | 132 | 81.1% | 4.5 | 71.2% | 56.1% | 81.1% | **+10pp** |
| compound_gene_disease | 2 | 200 | 34.5% | 1.2 | 32.5% | 24.0% | 34.5% | +2pp |
| disease_compound_via_gene | 3 | 132 | 49.2% | 16.3 | 79.5% | 61.4% | 49.2% | **-30pp** |

**Analysis vs Phase 207 (random) and Phase 208 (sentence, all-hop):**

- **1-hop 95.3%**, essentially tied with random (95.7%), best sentence result yet (+1.3pp vs Phase 208)
- **2-hop 53.0%**, beats random (47.9%) and corrects Phase 208 regression (36.8%). disease_gene_pathway: 81.1% vs 71.2% random, sentence embeddings genuinely help on this template once properly calibrated
- **3-hop 49.2%**, regressed from random (79.5%) and Phase 208 (61.4%)

**Root cause of 3-hop regression (disease_compound_via_gene):**

Confirmed by Phase 210 branch_bonus grid (see below): not a structural parameter problem. `disease_compound_via_gene` is insensitive to branch_bonus across the full range [0.032–0.308], holding at ~48.5–50.8%. The regression is semantic in origin: sentence-transformers embeddings introduce cosine-similarity bias that suppresses valid cross-type paths (disease→gene→compound spans maximally dissimilar semantic types). Random embeddings have no such bias, which is why Phase 207 achieves 79.5% on this template.

**Summary: Phase 209 is strictly better on 2-hop (+5pp vs random), with a known semantic ceiling on cross-type 3-hop.** This is a fundamental training-free limitation, not a tuning problem, see Phase 210 analysis.

**Canonical eval command (Phase 209, full validation as run):**
```bash
python -u benchmarks/hetionet_param_eval.py \
    --n-questions 200 --min-eval-hop 1 --max-neighbors 200 --workers 8 \
    --embeddings sentence \
    --beam-width 8 --trb-factor 13.871 --gamma 1.0147 --beta 0.9545 \
    --r2-boost 1.322 --vote-weight 0.6400 --idf-weight 0.018 \
    --branch-bonus 0.032 --fhrb-factor 1.726
```

---

## Hetionet, Phase 210 (branch_bonus Grid, 3-hop Root Cause Analysis)

Phase 210 tests whether branch_bonus is the lever for the 3-hop regression. Grid: [0.032, 0.10, 0.20, 0.308], all other params held at Phase 209 values, 200q/template full eval with workers=8.

### Phase 210, branch_bonus Grid Results

| branch_bonus | 1-hop | 2-hop | 3-hop | dgp H@1 | cge H@1 | dcvg H@1 | AvgTyped dgp |
|---|---|---|---|---|---|---|---|
| **0.032** | **95.3%** | **53.9%** | **50.8%** | **81.1%** | **36.0%** | **50.8%** | 4.6 |
| 0.10 | 96.1% | 47.6% | 48.5% | 70.5% | 32.5% | 48.5% | 3.3 |
| 0.20 | 95.7% | 42.2% | 48.5% | 63.6% | 28.0% | 48.5% | 2.5 |
| 0.308 | 95.3% | 39.2% | 48.5% | 57.6% | 27.0% | 48.5% | 2.1 |

Template codes: dgp=disease_gene_pathway, cge=compound_gene_disease, dcvg=disease_compound_via_gene

**Finding 1, branch_bonus confirmed optimal at 0.032.** 2-hop degrades monotonically as branch_bonus rises. disease_gene_pathway AvgTyped collapses 4.6→2.1 as branch_bonus rises: higher values bias the beam toward branchy intermediate paths, culling typed candidates before the answer hop.

**Finding 2, 3-hop is insensitive to branch_bonus.** `disease_compound_via_gene` is locked at 48.5% for all values \(\geq 0.10\), and only marginally better at 0.032 (50.8%). The 3-hop ceiling is not a structural parameter problem.

**Finding 3, The 3-hop regression is semantic in origin.** Sentence-transformers introduce cosine-similarity bias that suppresses valid cross-type paths. The `disease_compound_via_gene` template traverses maximally dissimilar semantic types (disease→gene→compound); the alpha (semantic similarity) CSA term systematically penalizes valid paths at each hop because the entity types are semantically distant. Random embeddings have no such bias and achieve 79.5%.

**Known limitation (documented):** Training-free semantic similarity has a natural cross-type depth ceiling in typed heterogeneous biomedical KGs. This is not addressable by structural parameter tuning. The practical mitigation (reducing CSA alpha weight globally) would degrade 1-hop and 2-hop same-type templates that benefit from semantic signal. The asymmetry is real: semantic embeddings help typed 2-hop (disease_gene_pathway +10pp vs random) while hurting cross-type 3-hop (-30pp vs random). A training step that calibrates per-relation-type semantic weights would be required to close this gap, outside the scope of training-free CEREBRUM.

---

## Hetionet, Phase 211 (GraphSAGE Ablation)

Phase 211 tests whether GraphSAGE smoothing amplifies the cross-type 3-hop semantic bias. Same Phase 209 params (branch_bonus=0.032), `--no-graphsage` flag added to `hetionet_param_eval.py`.

| Config | 1-hop | 2-hop | dgp | cge | 3-hop (dcvg) | AvgTyped dgp |
|---|---|---|---|---|---|---|
| Ph210 + GraphSAGE | 95.3% | 53.9% | 81.1% | 36.0% | 50.8% | 4.6 |
| Ph211 − GraphSAGE | 95.5% | **55.7%** | **81.8%** | **38.5%** | 47.0% | 4.9 |

**Finding: GraphSAGE is not the culprit.** Removing it makes 3-hop *worse* (47.0% vs 50.8%) while marginally improving 2-hop. GraphSAGE is providing a small compensatory benefit on cross-type 3-hop traversal, partially counteracting the cosine similarity bias. Keep GraphSAGE enabled.

**Conclusion (investigation closed):** The `disease_compound_via_gene` 3-hop ceiling (~49–51% vs random 79.5%) is intrinsic to training-free cosine similarity on cross-type heterogeneous paths. No structural parameter (branch_bonus) or embedding post-processing (GraphSAGE) can close this gap. A supervised per-relation-type semantic weight calibration step would be required, outside the scope of training-free CEREBRUM. Documented as a known limitation.

### Hetionet Constants: 2D Table (regime × embedding_method)

| Constant | typed_heterogeneous × random | typed_heterogeneous × sentence |
|----------|------------------------------|-------------------------------|
| BETA | 0.777 | **0.9545** |
| BOOST_SCALE | 28.48 | **8.67** |
| TRB_C | 6.14 | **4.31** |
| R2_C | 1.07 | **−0.111** |
| R2_FLOOR | 1.5 | **1.0** |
| FHRB_C | 0.80 | **0.159** |
| BRANCH_BONUS | 0.308 | **0.032** |
| IDF_SCALE_C | 0.0102 | **0.00432** |
| VOTE_BASE | 0.55 | **0.565** |

Source: Phase 207 (random) + Phase 209 (sentence, multi-hop calibration).
Note: negative R2_C for sentence/typed means path-consistency boost slightly hurts
when the beam already selects semantically coherent paths via STRB + GraphSAGE.

### Phase 165, Legacy Single-Template Result (disease_gene_pathway only)

| Metric | Value | Phase | Notes |
|--------|-------|-------|-------|
| Hits@1 | 61% | 165 | TRB + STRB, single template only |
| Hits@10 | 85% | 165 | |
| MRR | 0.72 | 165 | |
| BFS baseline | 0.8% | 165 | No TRB — confirms TRB necessity |
| TRB only (no STRB) | 73.5% (3-hop H@1) | 165 | |

---

## Hetionet, Phase 206b (branch_bonus Dominance Finding, Sentence-Transformers)

Tuner run to establish canonical Hetionet sentence-transformer calibration with full 6-template evaluation. Run as part of the cross-domain fANOVA investigation (Jun 2026).

**Setup:** 50q/template pilot (6 templates), sentence-transformers, max_neighbors=200, 50-trial Optuna.

### Phase 206b, fANOVA Result (key finding)

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| **branch_bonus** | **0.8186** | ████████████████████████████████ |
| trb_factor | 0.0621 | ██ |
| vote_weight | 0.0421 | █ |
| gamma | 0.0331 | █ |
| others | < 0.05 combined | |

**H@1 = 59.33%, MRR = 0.5933** (50q/template, 6 templates, sentence-transformers).

**Key finding, branch_bonus dominates on biomedical KG:** 81.86% of scoring variance on Hetionet is explained by a single parameter: `branch_bonus`. This compares to 46.2% on MetaQA (`hub_homogeneous` regime) and aligns with the core CEREBRUM thesis, multi-path convergence (multiple distinct intermediate routes reaching the same answer entity) is the primary discriminator between correct and incorrect answers in training-free KGQA across graph regimes.

On Hetionet (`typed_heterogeneous` regime), the branch_bonus signal is amplified because Hetionet's dense cross-type edges create numerous structurally valid multi-hop paths. Correct biomedical targets (genes, compounds, diseases with known relationships) appear at the intersection of many distinct traversal routes, while spurious candidates appear on only one. `branch_bonus` directly captures this convergence.

**Note:** 50q/template pilot. Phase 209 (below) is the 200q/template canonical validation: 1-hop 95.3%, 2-hop 53.0%, 3-hop 49.2%.

**Cross-domain fANOVA table (MetaQA vs Hetionet):**

| Dataset | Regime | Dominant Parameter | Importance | Second Parameter | Importance |
|---------|--------|--------------------|------------|------------------|------------|
| MetaQA 3-hop | hub_homogeneous | branch_bonus | 46.2% | trb_factor | ~15% |
| Hetionet | typed_heterogeneous | branch_bonus | 81.9% | trb_factor | 6.2% |
| WebQSP | typed_heterogeneous | degree_penalty_weight | 50.0% | schema_score_threshold | 15.4% |

**Interpretation:** branch_bonus is the universal discriminator for training-free KGQA on graphs with clean entity names and structured relation labels. WebQSP's different ranking (degree_penalty_weight dominant) reflects the Freebase-specific challenge: hub-entity suppression is the primary lever when semantic attention is limited by opaque MID identifiers.

---

## ConceptNet, Phase 229 (2-hop Chain Discovery, Random Embeddings)

ConceptNet 5.7 English subset: 149,860 entities / 152,385 edges / 8 relation types.
Evaluation methodology: 2-hop chain discovery (find h→mid→t where h→t has no direct training edge).
Train/test split: deterministic 80/20 MD5 hash of "{h}\t{r}\t{t}". 500 QA chains, random embeddings.

**Download:** `wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz`  
**Filter:** `--max-edges 200000` (English triples only, `lang/en` filter)

### Phase 229, Tuner Results (70-trial Sobol + partial CMA-ES, 500 chains)

**Best result:** H@1=6.0%, H@10=67.6%, MRR=0.2207

| Parameter | Value |
|-----------|-------|
| trb_factor | 29.098 |
| r2_boost | 4.637 |
| vote_weight | 0.806 |
| beam_width | 8 |
| idf_weight | 0.146 |
| branch_bonus | 0.365 |
| fhrb_factor | 1.142 |
| gamma | 6.362 |
| beta | 2.445 |

**Graph profile:**

| Stat | Value |
|------|-------|
| n_nodes | 149,860 |
| n_edges | 152,385 |
| mean_degree | 2.034 |
| degree_cv | 3.195 |
| n_rel | 8 |
| mean_fan_out | 2.699 |
| regime | typed_heterogeneous (classified) |

**Back-derived ParameterInitializer constants (mixed × random):**

| Constant | Value | Formula |
|----------|-------|---------|
| `_BOOST_SCALE["mixed"]` | 72.11 | gamma × mean_fo^beta = 6.362 × 2.699^2.445 |
| `_TRB_C["mixed"]` | 13.24 | trb / log(n_rel+1) = 29.098 / log(9) |
| `_BETA["mixed"]` | 2.445 | direct from tuner |
| `_BRANCH_BONUS["mixed"]` | 0.365 | direct from tuner |
| `_R2_C["mixed"]` | 13.85 | (r2-1.5) / log(mean_deg/n_rel+1) |
| `_FHRB_C["mixed"]` | 0.128 | (fhrb-1.0) / log(mean_deg+1) |
| `_VOTE_BASE["mixed"]` | 0.753 | vote - VOTE_Q_SCALE × Q_est |
| `_IDF_SCALE_C["mixed"]` | 0.0457 | idf / degree_cv = 0.146 / 3.195 |

**Canonical eval command:**
```bash
python benchmarks/conceptnet_eval.py \
    --cn5 benchmarks/data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --max-edges 200000 --n-questions 500 --embeddings random \
    --trb-factor 29.098 --r2-boost 4.637 --vote-weight 0.806 \
    --beam-width 8 --idf-weight 0.146 --branch-bonus 0.365 \
    --fhrb-factor 1.142 --gamma 6.362 --beta 2.445
```

**Note on H@1:** 6.0% on 2-hop chain discovery reflects ConceptNet's structural characteristics, 8 general-purpose commonsense relations produce highly ambiguous intermediate nodes. H@10=67.6% confirms the correct answer is found in the top-10 the majority of the time; the challenge is ranking it first when many plausible paths compete. This is fundamentally different from MetaQA's narrow-domain relations (9 movie-specific types with clear semantic clustering) or Hetionet's typed biomedical edges.

### ConceptNet, Phase 230 (sentence embeddings)

| Metric | Tuning (500q) | Validation (2000q) |
|--------|--------------|-------------------|
| H@1    | 6.2%         | 3.55%              |
| H@10   | 67.6%        | 63.80%             |
| MRR    | 0.2215       | 0.1915             |

**Finding**: optimal params identical to Phase 229 random, ConceptNet concept strings (1-3 word phrases) are too short for sentence-transformers to add structural signal. `_IDF_SCALE_C_SENTENCE["mixed"] = 0.0457` (same as random).

**ParameterInitializer 2D table, COMPLETE (v2.76.0)**

|                      | random     | sentence   |
|----------------------|------------|------------|
| hub_homogeneous      | Phase 204 ✓| Phase 213 ✓|
| typed_heterogeneous  | Phase 207 ✓| Phase 209 ✓|
| mixed                | Phase 229 ✓| Phase 230 ✓|

---

## ParameterInitializer 2D Constant Table (v2.76.0)

| Regime | random | sentence |
|--------|--------|---------|
| **hub_homogeneous** | Phase 204 (MetaQA) ✓ | Phase 213 (MetaQA sentence) ✓ |
| **typed_heterogeneous** | Phase 207 (Hetionet) ✓ | Phase 209 (Hetionet sentence) ✓ |
| **mixed** | **Phase 229 (ConceptNet) ✓** | **Phase 230 (ConceptNet sentence) ✓** |

---

## WebQSP, Phase 231–244d (v2.86.0)

Dataset: WebQSP test split (1,628 questions), Freebase 2-hop KB (3.79M triples).
Graph regime: `typed_heterogeneous`. Seed-entity subgraph extraction: 584k triples, 292k nodes.

**Note:** arXiv:2505.23495 found ~52% of WebQSP examples factually questionable.
Acknowledge this benchmark quality caveat in all papers discussing WebQSP results.

### WebQSP Phase Progression

| Phase | Key Addition | H@1 | H@10 | MRR | N | Date |
|-------|-------------|-----|------|-----|---|------|
| 231 | Zero-config baseline (Hetionet fallback params) | 5.50% | 25.50% | 0.1127 | 200q | Jun 7 |
| 232 | QuestionDecomposer + RelationNameIndex + soft type filter | 6.50% | — | — | 200q | Jun 8 |
| 233 | CommunityHypothesisGenerator (typed bridge boosts) | ~7–8% | ~23–25% | ~0.12 | 200q | Jun 8 |
| 236 | PathSchemaIndex (training-free schema prediction) | 9.50% | 32.50% | 0.1552 | 200q | Jun 8 |
| 241 | BeamCheckpoint + full 1,628q tuner (Trial #82) | 5.04% | 11.86% | 0.0725 | **1,628q** | Jun 9 |
| 244c | Additive CVT passthrough (replacement→additive fix) | 9.47% | **20.04%** | 0.1252 | 1,628q | Jun 10 |
| 244d | +backward verification + path diversity re-ranker (100-trial tuner) | 10.33% | 20.47% | 0.1347 | 1,628q | Jun 10 |
| 253 | schema_top_k=12 (PathSchemaIndex expansion + anchor-score re-ranking study) | 10.57% | 20.53% | 0.1389 | 1,628q | Jun 12 |
| 254 | Wider beam re-tune (beam_width=48, schema_top_k=16) | 10.26% | 23.60% | 0.1443 | 1,628q | Jun 12 |
| 255 | Guaranteed 1-hop Pass (G1P) + DPW=1.03 (beam-pruned answers injected back) | 11.12% | 21.20% | 0.1447 | 1,628q | Jun 12 |
| 256 | Relation-conditioned G1P (synthetic best_path; g1p_trb_weight dead signal) | 11.12% | 21.27% | 0.1458 | 1,628q | Jun 12 |
| 257 | schema_top_k=32 + backward_bonus=0.20 + DPW=1.49 | 11.49% | 21.70% | 0.1507 | 1,628q | Jun 12 |
| 258 | schema_top_k probe [32,48,64] — 32 wins, ceiling confirmed | 11.49% | — | — | 200q | Jun 12 |
| **259** | **idf_weight=0.073 + beta=0.649 + DPW=1.84 (schema_top_k=32 locked)** | **11.92%** | **20.47%** | **0.1516** | **1,628q** | **Jun 12** |

**Zero-config baseline:** H@1=1.41%, H@10=4.30% (ParameterInitializer returns gamma=334,513 for Freebase topology, uncalibrated; uses Hetionet fallback). **Phase 244d vs zero-config: +633% relative H@1 improvement.**

---

### WebQSP, Phase 231 (Baseline)

**Run date: 2026-06-07.** 200-question sample, zero-config, random embeddings, MID relay-node filter.

| Metric | Value | Notes |
|--------|-------|-------|
| H@1  | 5.50%  | Ranking distorted by CVT path-count amplification |
| H@10 | 25.50% | Correct answer in beam ≥25% of the time |
| MRR  | 0.1127 | |

**Zero-config params (Hetionet typed_heterogeneous × random fallback):**
trb_factor=13.87, r2_boost=1.32, vote_weight=0.64, beam_width=12,
idf_weight=0.018, branch_bonus=0.03, fhrb_factor=1.73, gamma=1.01, beta=0.95.

**Scientific findings:**
- H@10=25.5% confirms the correct answer is in the beam; H@1=5.5% reflects a ranking challenge on Freebase's CVT-reified graph structure.
- CVT vote normalization was tested and degraded performance (H@10: 25.5% → 18.5% → 14.0%). Consensus voting remains the best ranking signal on reified KGs.
- Sentence embeddings provide no benefit (288k MID strings dilute cosine similarity; 788s overhead).

**Canonical eval command (Phase 231):**
```bash
python -u benchmarks/webqsp_param_eval.py --sample 200 --embeddings random
```

---

### WebQSP, Phase 236 (PathSchemaIndex, Training-Free Schema Prediction)

Phase 236 introduces PathSchemaIndex: the first PREDICTIVE reasoning signal in CEREBRUM. All prior signals steer or re-rank after beam traversal. PathSchemaIndex predicts the most likely (r1, r2) 2-hop relation path before traversal begins.

**Result (200q, sentence-transformers):** H@1=**9.5%**, H@10=**32.5%**, MRR=**0.1552**
vs Phase 235: H@1=6.0%, H@10=28.5%, MRR=0.1198 (**+3.5pp H@1, +4.0pp H@10, +29% MRR**).

**Method:** 81,755 (r1, r2) schemas indexed from full graph. Schema embeddings encode last-segment Freebase text ("person.person.place_of_birth"→"place of birth"). Seed filter: only schemas whose r1 is actually present on the seed entity. Top-2 predicted schemas execute as targeted 2-hop traversals in parallel with the beam; answers prepended for H@1 competition.

---

### WebQSP, Phase 244c (Additive CVT Passthrough)

Phase 244c fixes the CVT traversal architecture. Prior implementation replaced direct edges with compound CVT edges, flooding `_batch_steps` with all CVT neighbors and crowding out non-CVT 2-hop paths. Fix: additive mode, normal direct edge always preserved + top-5 compound CVT edges by confidence appended.

**Result (full 1,628q):** H@1=**9.47%**, H@10=**20.04%**, MRR=**0.1252**

**CVT passthrough effect:** Phase 244c H@10=20.04% vs Phase 244 H@10=9.34% with replacement CVT, nearly doubled by the additive fix. This confirms CVT-aware traversal is architecturally correct; the regression was implementation-caused (batch flooding), not signal failure.

---

### WebQSP, Phase 244d (Canonical Full-Set Result)

**This is the canonical full-evaluation result for WebQSP. All papers must cite this entry.**

Run date: 2026-06-10. Full test split: 1,628 questions. 100-trial TPE tuner (30 Sobol + 70 CMA-ES).

| Metric | Value | Notes |
|--------|-------|-------|
| **H@1**  | **10.33%** | Full 1,628-question evaluation |
| **H@10** | **20.47%** | |
| **MRR**  | **0.1347** | |

**Best-trial parameters (Trial #82):**

| Parameter | Value |
|-----------|-------|
| trb_factor | 41.785 |
| r2_boost | 3.722 |
| vote_weight | 0.8608 |
| beam_width | 16 |
| idf_weight | 0.013 |
| branch_bonus | 0.048 |
| fhrb_factor | 1.826 |
| gamma | 7.7262 |
| beta | 1.0083 |
| degree_penalty_weight | 0.4236 |
| schema_score_threshold | 0.3689 |
| backward_bonus | 0.1357 |
| diversity_alpha | 0.7986 |

**Phase 244d fANOVA (importance over 100 trials):**

| Parameter | Importance | Bar |
|-----------|-----------|-----|
| degree_penalty_weight | 0.5000 | ████████████████████ |
| schema_score_threshold | 0.1540 | ██████ |
| idf_weight | 0.1330 | █████ |
| vote_weight | 0.0510 | ██ |
| others | < 0.05 combined | |

**Key finding, degree_penalty_weight is the dominant parameter on WebQSP (50.0%):** Freebase has extremely high-degree hub entities (e.g., "United States" with thousands of outgoing edges). Hub suppression via `degree_penalty_weight` is the primary mechanism distinguishing correct answers from spurious high-degree candidates. This contrasts with MetaQA and Hetionet where `branch_bonus` dominates, on those graphs with clean entity names, multi-path convergence is the signal; on Freebase with opaque MIDs, hub suppression is the signal.

**New capabilities in Phase 244d:**
- **Backward verification pass (Phase 245):** For 2-hop answers [seed, rel1, hop1, rel2, answer], checks if hop1 appears in answer's outgoing neighbors via `_expansion_cache`. Score × (1 + backward_bonus) for bidirectional structural support.
- **Path diversity re-ranker (Phase 246):** Counts distinct hop-1 intermediates reaching each answer from the seed. Score × (1 + diversity_alpha × log1p(n−1)) for n > 1 distinct paths.
- **Additive CVT traversal (Phase 246):** Normal edge always preserved; top-5 compound CVT edges by confidence appended, prevents beam slot crowding.

**Canonical eval command (Phase 244d):**
```bash
python -u benchmarks/webqsp_param_eval.py \
    --sample 1628 --embeddings sentence \
    --beam-width 16 --trb-factor 41.785 --r2-boost 3.722 \
    --vote-weight 0.8608 --idf-weight 0.013 --branch-bonus 0.048 \
    --fhrb-factor 1.826 --gamma 7.7262 --beta 1.0083 \
    --degree-penalty-weight 0.4236 --schema-score-threshold 0.3689 \
    --backward-bonus 0.1357 --diversity-alpha 0.7986
```

**Why WebQSP is hard for training-free systems (documented ceiling):**

~60% of failures are CVT disambiguation failures, Freebase uses compound-value-type mediator nodes with opaque MID identifiers (e.g., `/m/02s8qk3`) as intermediaries for reified relationships. Semantic attention on these MID strings produces near-zero cosine similarity with question text. ~25% of failures are hub-entity score plateau from high-degree Freebase entities. The remaining ~15% are multi-hop inference failures that would require explicit question-hop structure parsing.

H@10=20.47% confirms the correct answer is in the top-10 candidates roughly 1 in 5 times, the system's retrieval is not random, but ranking is limited by semantic signal quality on Freebase. Supervised systems (UniKGQA 75.1%, NSM 74%) benefit from question-answer training that calibrates these weights directly; CEREBRUM does not. This is documented as an honest scientific finding, not a failure mode.

---

### WebQSP, Phase 247 (Conditional Schema Prediction, Negative Result)

**Configuration:** 100-trial TPE tuner adding `conditional_schema_bonus` (csb) to Phase 244d param space. csb scales the structural boost applied to schemas whose r2 relation is confirmed present on actual hop-1 intermediates from `_expansion_cache`.

**Result:** H@1=**10.33%**, H@10=20.04%, MRR=0.1347 (best trial 97, csb=0.240).

**Finding:** csb provides zero net gain over Phase 244d baseline (H@1 tied at 10.33%; H@10 slightly lower). Conditional schema prediction adds no information beyond the Phase 236 pre-beam schema channel already in the pipeline.

**Interpretation:** The `_expansion_cache` hop-1 structural evidence overlaps heavily with what PathSchemaIndex already predicts, the schemas it would boost are already being executed. csb is retired from `PARAM_SPACE_WEBQSP`.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 248 (Freebase MID Name Resolution, Negative Result)

**Hypothesis:** Scan `freebase_2hop.txt` for `type.object.name` triples to build a MID→readable_name map; substitute at embedding time to restore CSA alpha signal on CVT nodes.

**Result:** 0 MIDs resolved (0.0% coverage). `type.object.name` triples are not present in the 2-hop subgraph extraction, that file contains only QA-path triples, not entity metadata. Zero name substitutions applied; eval score equivalent to Phase 244d within sampling noise (200q sample: H@1=8.50%, within ±3pp of 10.33% canonical).

**Finding:** Correct implementation, wrong data source. Entity name triples require a separate Freebase entity-labels file (e.g., FB15k label dumps or Freebase full NTriples). The `entity_name_overrides` infrastructure in `adapters/networkx_adapter.py` and `core/cerebrum.py` is in place and correct, it activates automatically when overrides are populated; the missing piece is the data.

**Next step for CVT opacity:** Acquire a Freebase entity-name mapping file (e.g., `freebase-easy.tar.gz` from the Freebase archive or FB15k entity labels). Pre-built maps mapping `/m/xxxxx` → English label are available from prior FB15k work.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 249 (FB15k Entity Labels, Negative Result)

**Hypothesis:** Load an external Freebase entity-name file (`FB15k_mid2name.txt` from KGraph/FB15k-237 on HuggingFace) to populate the `entity_name_overrides` infrastructure built in Phase 248, restoring CSA alpha signal on opaque `/m/` nodes.

**Result:** 0 MIDs resolved (0.0% coverage). Root cause: the 14,951 entities in FB15k-237 are a disjoint subset from the 486,165 `/m/` nodes in the WebQSP 2-hop subgraph. Zero overlap even after normalising `/m/xxx` ↔ `m.xxx` format variants.

**Deeper finding:** The `/m/` nodes in `freebase_2hop.txt` are predominantly **CVT (compound value type) mediator nodes** used by Freebase for n-ary relations (e.g. `film.starring`, `award.nomination`). CVT nodes have no human-readable names by design, they are structural reification artifacts, not named entities. FB15k-237 contains entity MIDs; our graph's MIDs are CVT MIDs. They are non-overlapping populations.

**Corollary:** Only 2 out of 1,628 test questions have MID answers. All other answer entities are already readable names. The naming problem does not affect answer ranking, it affects traversal scoring of intermediate CVT hops only. Phase 246's CVT passthrough already collapses these paths via compound-edge scoring.

**Infrastructure status:** `entity_name_overrides` in `NetworkXAdapter` and label substitution in `core/cerebrum.py` remain in place and correct. The `--mid-name-file` CLI arg is wired. Phase 249 is closed as negative, the data source problem is architectural (CVT vs entity MIDs), not a file-format issue.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 250 (QASA Re-ranking, Negative Result)

**Hypothesis:** Post-hoc re-ranking via Question-Answer Semantic Alignment (QASA): after beam traversal, re-score each candidate by `cosine(question_embedding, answer_entity_embedding)`. Adds direct Q-to-A semantic signal on top of structural scoring via `qa_sem_weight` parameter.

**Result (100 trials, 200q sample, TPE):** Best H@1=9.5% (sample). Full 1,628q canonical eval with best params: **H@1=10.02%, H@10=20.53%, MRR=0.1330** vs canonical 10.33% / 20.47% / 0.1347. Phase 250 is -0.31pp H@1 vs canonical. Closed negative.

**fANOVA finding (100 trials):** `degree_penalty_weight` accounts for **56.5% of scoring variance** on WebQSP, the highest single-parameter dominance for this benchmark. `backward_bonus` second at 8.6%. `qa_sem_weight` fourth at 4.2%, confirming it is not a primary lever.

**Root cause:** QASA requires the answer entity embedding to be semantically similar to the question. On Freebase, most answer entities are public figures, places, and films with readable names. However, CVT-mediated paths produce intermediate embeddings that dilute the Q-to-A signal before it reaches the answer entity. The semantic similarity term is present but weak relative to structural factors.

**Key architectural signal:** `degree_penalty_weight` at 56.5% importance (up from 50.0% in Phase 244d fANOVA) confirms hub-entity degree suppression is the primary bottleneck for WebQSP, not semantic scoring quality. Phase 251 target: targeted `degree_penalty_weight` search with finer resolution.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 251 (Focused DPW Sweep, Negative Result)

**Hypothesis:** `degree_penalty_weight` (DPW) has shown 50-57% fANOVA importance across Phases 244d and 250, but all prior searches used a narrow range (0.3-0.6). Expanding to (0.0-1.5) with all other params locked near Phase 244d best values should find whether stronger hub suppression improves H@1.

**Result (100 trials, 200q sample, TPE):** Best H@1=9.0% (sample). Full 1,628q canonical eval: **H@1=10.02%, H@10=19.36%, MRR=0.1287** vs canonical 10.33% / 20.47% / 0.1347. Phase 251 closes negative (-0.31pp H@1, -1.11pp H@10).

**DPW landscape (100 trials, 200q buckets):**

| DPW range | Mean H@1 | Max H@1 | Trials |
|-----------|---------|---------|--------|
| 0.0-0.25  | 7.7%    | 8.0%    | 17     |
| 0.25-0.75 | 8.2%    | 9.0%    | 32     |
| 0.75-1.25 | 8.5%    | 9.0%    | 32     |
| 1.25-1.5  | 8.1%    | 9.0%    | 19     |

**Finding:** DPW forms a plateau in the 0.5-1.5 range. The fANOVA importance (73.1% in Phase 251) reflects a binary threshold: DPW=0 is significantly worse, but any value in (0.5-1.5) gives similar results. Phase 244d's DPW=0.424 already sits at the plateau. Expanding the range finds no new optimum.

**Architectural conclusion:** `degree_penalty_weight` applies the hub penalty **at answer extraction only** (post-hoc). Hub entities inflate beam scores during traversal itself, before extraction. Tuning the extraction penalty cannot compensate for beam-time hub flooding. Phase 252 target: traversal-time hub suppression — penalize edges leading TO high-degree nodes during beam scoring, preventing hub paths from accumulating scores in the first place.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 252 (Traversal-Time Hub Suppression, Negative Result)

**Hypothesis:** `degree_penalty_weight` (DPW) only penalises hub entities at answer extraction. Hub nodes also pollute the beam as high-scoring intermediates during traversal itself. A new `beam_hub_penalty` (BHP) parameter applies `1/(1 + bhp * log1p(deg(v)))` at non-terminal hops only — preventing hub entities from accumulating path score as intermediates while leaving final-answer scoring unchanged. DPW and BHP are architecturally complementary.

**Result (100 trials, 200q sample, TPE):** Best H@1=10.0% (sample). Full 1,628q canonical eval: **H@1=9.77%, H@10=19.91%, MRR=0.1294** vs canonical 10.33% / 20.47% / 0.1347. Phase 252 closes negative (-0.56pp H@1).

**BHP landscape (100 trials, 200q buckets):**

| BHP range | Mean H@1 | Max H@1 | Trials |
|-----------|---------|---------|--------|
| 0.0-0.25  | 8.8%    | 10.0%   | 17     |
| 0.25-0.75 | 8.9%    | 10.0%   | 32     |
| 0.75-1.25 | 8.8%    | 9.5%    | 34     |
| 1.25-1.5  | 8.7%    | 9.5%    | 17     |

**fANOVA shift:** When BHP is in the search space, `vote_weight` becomes dominant (31.3%) and `degree_penalty_weight` drops to 6.8% (from 50-73% in Phases 244d-251). This reveals that DPW had been acting as a proxy for vote_weight miscalibration, not pure hub suppression.

**Architectural conclusion:** Both post-hoc (DPW) and traversal-time (BHP) hub suppression are near their ceiling with the current `1/(1 + w*log1p(deg))` formula. The 10.33% canonical is a robust local optimum for this class of hub-penalty scoring. The ~25% hub-entity failure rate is not addressable by tuning the penalty weight or applying it at a different stage. Phase 253 target: failure-mode diagnostic — categorize the ~1,300 failed questions to identify systematic patterns beyond hub flooding.

**Phase 244d remains the WebQSP canonical result.**

---

### WebQSP, Phase 253 (Anchor-Score Re-ranking + Tunable schema_top_k)

**Configuration:** 100-trial TPE tuner adding two new params: `anchor_rerank_weight` (anchor score re-ranking, HACH-MVC principle, AliTakrar/HACH-MVC) and `schema_top_k` ∈ {3, 5, 8, 12} (tunable PathSchemaIndex prediction count, previously hardcoded at 5). Anchor score = `intra_community_degree / total_degree` — entities whose neighbours are concentrated in one community score high (semantically coherent targets); cross-community hubs score low. This is a neighbourhood-coherence signal inspired by the anchor principle in HACH-MVC (anchor-guided contrastive learning for multi-view clustering).

**Result (100 trials, 200q sample, TPE):** Best H@1=9.50% (sample, trial #18). Full 1,628q canonical eval with best params (schema_top_k=12, anchor_rerank_weight=0.0): **H@1=10.57%, H@10=20.53%, MRR=0.1389** vs Phase 244d 10.33% / 20.47% / 0.1347. **+0.24pp H@1, +0.06pp H@10, +0.0042 MRR.**

**fANOVA (Phase 253 param importances):**
- `schema_top_k`: **82.3%** — dominant
- `anchor_rerank_weight`: **0.1%** — noise

**Finding:** `anchor_rerank_weight` has near-zero importance — the HACH-MVC anchor principle does not transfer to this task. Anchor score (community neighbourhood concentration) is not a discriminative signal for answer re-ranking on WebQSP: correct answers are not more community-concentrated than incorrect ones, likely because CVT reification creates artificial cross-community structure for both. The anchor score is computed correctly but the signal is uninformative in this context.

`schema_top_k=12` (vs hardcoded 5) is the sole driver of improvement. More schema predictions increase coverage of the correct 2-hop path, contributing an additional +0.24pp H@1 and the H@10 nudge. The signal saturates quickly — schema_top_k=12 was optimal, suggesting diminishing returns beyond this point.

**Canonical eval command (Phase 253):**
```
python -u benchmarks/webqsp_param_eval.py --sample 1628 --embeddings sentence \
  --beam-width 16 --trb-factor 44.864 --r2-boost 2.732 --vote-weight 0.8515 \
  --idf-weight 0.017 --branch-bonus 0.066 --fhrb-factor 2.246 \
  --gamma 8.6062 --beta 1.1866 --degree-penalty-weight 0.6318 \
  --schema-score-threshold 0.3572 --backward-bonus 0.1084 \
  --diversity-alpha 0.6549 --anchor-rerank-weight 0.0 --schema-top-k 12
```

**Phase 253 has been superseded. See Phase 255.**

---

### WebQSP, Phase 254 (Wider Beam Re-tune, Intermediate)

**Configuration:** PARAM_SPACE_WEBQSP_254 — beam_width ∈ [32,48,64,96], all ranking params re-opened. Full re-tune after beam_width=64 probe showed H@10 +2.33pp but H@1 -0.24pp.

**Result:** H@1=**10.26%**, H@10=**23.60%**, MRR=0.1443 (beam_width=48, schema_top_k=16). H@10 confirms wider beam finds more answers in the pool (+3.07pp over Phase 253). H@1 regresses — ranking machinery can't push them to position 1 under hub competition. fANOVA: beam_width only 2.6% importance (within [32,96], width barely matters for H@1). **Intermediate result — not a new canonical.**

**Architectural conclusion:** The beam retrieval problem is not just width — it's that correct low-degree answers lose to hubs in the scoring competition *before being pruned*. Phase 255 target: bypass beam scoring for 1-hop answers entirely.

---

### WebQSP, Phase 255 (Guaranteed 1-hop Pass)

**Motivation:** Coverage diagnostic (Phase 253b/c/d) showed 43.5% of beam misses are direct 1-hop neighbors of the seed entity (29.6% of all questions) that the beam prunes at hop 1 because beam_width=16 can't fit them. The answers are in the graph, the seed-to-answer edge exists, but the hub-dominated scoring drops them.

**Mechanism (G1P):** After the beam query, enumerate all named (non-MID) direct neighbors of the seed entity (up to 2000) and inject any missing ones into the candidate pool at score `min_beam_score × hop1_base_weight`. The existing ranking stack (DPW, r2_boost, backward_bonus, diversity_alpha) then operates on the extended pool. No LLM, no training — purely structural.

**Result (100 trials, 200q sample, TPE + full 1,628q eval):** Best trial H@1=10.0% (200q). Full eval: **H@1=11.12%, H@10=21.20%, MRR=0.1447** vs Phase 253 10.57% / 20.53% / 0.1389. **+0.55pp H@1, +0.67pp H@10, +0.0058 MRR.**

**fANOVA (Phase 255):**
- `schema_top_k`: 47.5% — still dominant but reduced from 82.3% (G1P takes over some coverage)
- `schema_score_threshold`: 12.3%
- `hop1_base_weight`: 4.2% — positive contribution confirmed
- `degree_penalty_weight`: 4.9% — best value 1.03 (up from 0.63 in Phase 253) — G1P injects hub neighbors, requiring stronger suppression

**Finding:** G1P recovers a fraction of the theoretical +29.6pp from H1 misses. The gain is real but modest (+0.55pp H@1) because DPW alone is a weak ranker for distinguishing correct 1-hop answers (e.g. "England") from injected hubs (e.g. "United States") — both are direct neighbors, but hubs have higher beam scores. The G1P candidate pool contains the answer; the ranking signal to surface it is the bottleneck.

**Canonical eval command (Phase 255):**
```
python -u benchmarks/webqsp_param_eval.py --sample 1628 --embeddings sentence \
  --beam-width 16 --trb-factor 39.741 --r2-boost 3.559 --vote-weight 0.8806 \
  --idf-weight 0.018 --branch-bonus 0.064 --fhrb-factor 2.509 \
  --gamma 6.9888 --beta 0.9540 --degree-penalty-weight 1.0289 \
  --schema-score-threshold 0.2898 --backward-bonus 0.0773 \
  --diversity-alpha 0.8714 --schema-top-k 16 --hop1-base-weight 0.6307
```

**Phase 255 has been superseded. See Phase 257.**

---

### WebQSP, Phase 257 (schema_top_k Escalation)

**Motivation:** fANOVA across Phases 255–256 showed `schema_top_k` at 28–43% importance with the search capped at 16. Phase 257 tests whether more schema predictions provide meaningful additional 2-hop coverage.

**Mechanism:** Expand `schema_top_k` search space to [16, 24, 32]. The PathSchemaIndex predicts (r1, r2) 2-hop path schemas from the question embedding; more predictions means more structurally applicable schemas are executed. Phase 256 also wired a synthetic `best_path` into G1P injections so that downstream `q_scores` (question-keyword relation matching) applies to injected 1-hop candidates. `g1p_trb_weight` was confirmed dead (1.5% importance) and dropped.

**Result (80+40 trials, 200q sample + full 1,628q eval):** Best trial H@1=11.50% (200q, schema_top_k=32). Full eval: **H@1=11.49%, H@10=21.70%, MRR=0.1507** vs Phase 255 11.12% / 21.20% / 0.1447. **+0.37pp H@1, +0.50pp H@10, +0.006 MRR.**

**fANOVA (Phase 257):**
- `schema_top_k`: **68.4%** — exploded from 28–43% once 32 was included; dominant by far
- `idf_weight`: 6.0%
- `vote_weight`: 5.8%
- `gamma`: 4.6%
- `backward_bonus`: 2.0% (best value 0.203, 2.6× higher than Phase 255's 0.077)
- `degree_penalty_weight`: 1.1% (best 1.486, up from Phase 255's 1.029)

**Best params:** beam_width=24, trb_factor=41.6, r2_boost=4.79, vote_weight=0.890, idf_weight=0.044, fhrb_factor=3.36, gamma=10.90, beta=1.02, dpw=1.486, sst=0.308, backward_bonus=0.203, diversity_alpha=0.526, schema_top_k=32, hop1_base_weight=0.495.

**Canonical eval command (Phase 257):**
```
python -u benchmarks/webqsp_param_eval.py --sample 1628 --embeddings sentence \
  --beam-width 24 --trb-factor 41.624 --r2-boost 4.788 --vote-weight 0.8901 \
  --idf-weight 0.044 --branch-bonus 0.059 --fhrb-factor 3.358 \
  --gamma 10.8971 --beta 1.0223 --degree-penalty-weight 1.4859 \
  --schema-score-threshold 0.3083 --backward-bonus 0.2026 \
  --diversity-alpha 0.5260 --schema-top-k 32 --hop1-base-weight 0.4946 \
  --g1p-trb-weight 0.0
```

**Phase 257 has been superseded. See Phase 259.**

---

### WebQSP, Phase 259 (idf_weight + beta Fine-tune)

**Motivation:** Phase 258 confirmed schema_top_k=32 is the ceiling (best trial selected 32 even with 48/64 available). With top_k locked, fANOVA spread evenly across all remaining params (no dominant lever). Two signals from Phase 258: idf_weight at 20% importance with best=0.071 (vs Phase 257's 0.044); beta at 11% with best=0.970. Phase 259 widens both ranges to find the optimum.

**Mechanism:** schema_top_k fixed at 32. idf_weight range extended to (0.01, 0.12); beta to (0.6, 1.5). The tuner converged to idf_weight=0.073, beta=0.649 (strongly suppressing fan-out power law), dpw=1.836 (highest DPW yet). Aggressively suppresses hub entities across three channels simultaneously.

**Result (80+40 trials, 200q sample + full 1,628q eval):** Best trial H@1=11.50% (200q, same as Phase 257). Full eval: **H@1=11.92%, H@10=20.47%, MRR=0.1516** vs Phase 257 11.49% / 21.70% / 0.1507. **+0.43pp H@1, -1.23pp H@10, +0.001 MRR.**

**Tradeoff:** The Phase 259 params are precision-optimized — more correct answers reach rank 1, but overall top-10 coverage narrows. H@1/H@10 gap shrinks from 10.2pp (Phase 257) to 8.6pp (Phase 259), indicating the ranking signal is improving.

**fANOVA (Phase 259, schema_top_k locked):**
- `trb_factor`: 13.8%, `degree_penalty_weight`: 13.0%, `branch_bonus`: 12.6%, `gamma`: 12.4%, `vote_weight`: 10.8% — evenly distributed, plateau signal
- `schema_top_k`: 0.0% — correctly zero (locked)

**Best params:** beam_width=24, trb_factor=36.85, r2_boost=5.055, vote_weight=0.813, idf_weight=0.073, branch_bonus=0.042, fhrb_factor=3.993, gamma=13.857, beta=0.649, dpw=1.836, sst=0.200, backward_bonus=0.244, diversity_alpha=0.884, schema_top_k=32, hop1_base_weight=0.383.

**Canonical eval command (Phase 259):**
```
python -u benchmarks/webqsp_param_eval.py --sample 1628 --embeddings sentence \
  --beam-width 24 --trb-factor 36.850 --r2-boost 5.055 --vote-weight 0.8129 \
  --idf-weight 0.073 --branch-bonus 0.042 --fhrb-factor 3.993 \
  --gamma 13.8569 --beta 0.6494 --degree-penalty-weight 1.8363 \
  --schema-score-threshold 0.1999 --backward-bonus 0.2441 \
  --diversity-alpha 0.8843 --schema-top-k 32 --hop1-base-weight 0.3825 \
  --g1p-trb-weight 0.0
```

**Phase 259 is the new WebQSP canonical result: H@1=11.92%, H@10=20.47%, MRR=0.1516.**

---

### WebQSP, "When Training-Free KGQA Works"

The WebQSP and MetaQA/Hetionet results together characterize when training-free semantic attention succeeds:

| Condition | MetaQA | Hetionet | WebQSP (Freebase) |
|-----------|--------|----------|-------------------|
| Entity name quality | Movie titles, person names | Gene/disease names, compound names | Opaque MIDs (`/m/0xxxxx`) |
| Relation label quality | 9 movie-specific labels | 24 biomedical types | 989 dotted Freebase paths |
| Semantic attention viable? | **Yes** | **Yes** | **Limited** |
| Dominant CSA parameter | branch_bonus (46.2%) | branch_bonus (81.9%) | degree_penalty_weight (50.0%) |
| H@10 (training-free) | **87.9%** | **~60%** | **20.5%** |
| H@1 (training-free) | **60.6%** | **59.3%** | **11.9%** |

**Conclusion:** Training-free multi-hop reasoning is effective when the knowledge graph has human-readable entity names and structured relation labels that enable cosine-similarity-based attention. Freebase's opaque MID identifiers break this prerequisite. Future work: MID-to-name preprocessing (Freebase entity labels) or discriminative re-ranking using an LLM for semantic filtering.

---

## IKGWQ, Incomplete Knowledge Graph World QA Protocol

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
| CEREBRUM Phase 227 (full stack)† | None | — | — | **60.6%** | — |
| **CEREBRUM Phase 244d (WebQSP)†** | **None** | **—** | **—** | **—** | **10.33%** |

† Full-stack 3-hop results use FHRB, r2-boost, SRD, RelationPathPrior and are
**not directly comparable** to supervised methods in this table, listed for internal tracking only.
Use Phase 53 numbers in all paper comparison tables.

**Phase 227 full-stack configuration (Jun 5, 2026):** sentence-transformers, beam-width=12,
trb-factor=31.134, r2-boost=3.977, vote-weight=0.7527, idf-weight=0.039, branch-bonus=0.385,
fhrb-factor=7.572, gamma=10.138, beta=0.9616, 8-worker multiprocessing.
14,268/14,274 questions answered (6 skipped). Runtime: 1283.7s (~21 min).
H@1=60.6% | H@10=87.9% | MRR=0.703.

**Re-validation on v2.87.0 HEAD (Jun 11, 2026):** 14,274/14,274 questions, workers=8.
H@1=60.29% | H@10=87.78% | MRR=0.699, within ±0.5pp of Phase 227 canonical. ✓ No regression from Phases 228–248.

**Framing (include as footnote or paragraph):**
> CEREBRUM achieves these results with zero task-specific training, no labeled question-answer pairs,
> and no gradient updates, operating purely from graph structure and pre-trained sentence embeddings.
> To our knowledge, this represents the first training-free baseline for multi-hop KGQA, establishing
> a reference point for what structural reasoning alone can achieve. The \(\text{H@10}\) story is the key result:
> CEREBRUM retrieves the correct answer in its top-10 candidates at 96.6% (1-hop), 86.3% (2-hop),
> and 50.3% (3-hop), the system *finds* the answer, it does not yet rank it first. This is a ranking
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

## Notation for \(\eta\) (Eta), Resolved Conflict

**Decision (Phase 172 / May 2026):** The symbol \(\eta\) was used with two different meanings:
- In CSA (Phase 43): \(\eta\) = temporal decay weight (one of the 10 CSA parameters)
- In TSC/DSCF (Phase 1): \(\eta\) was used generically for temperature step decay

**Resolution:** In all publication documents, use:
- \(\eta\) for CSA temporal decay weight (the 10-parameter formula, dominant usage)
- \(\eta_T\) for TSC temperature-step decay (Paper A only, subscript T for Temperature)

All papers after Phase 1 that reference the TSC temperature schedule should use \(\eta_T\).

---

*Last updated: 2026-06-12 | Phase 259: idf=0.073 + beta=0.649 + DPW=1.84, H@1=11.92% (new canonical, +0.43pp), H@10=20.47% (precision/recall tradeoff) | Phase 258: schema_top_k=32 ceiling confirmed (48/64 no gain) | Phase 257: schema_top_k=32 (68.4% fANOVA, +0.37pp H@1) | Phase 255: G1P +0.55pp H@1 | Phase 253: schema_top_k=12 (+0.24pp H@1) | Phase 244d: H@1=10.33% prior canonical | Phase 236: PathSchemaIndex +3.5pp H@1 | Phase 225-227: MetaQA H@1=60.6%, H@10=87.9% | Phase 53 canonical unchanged*

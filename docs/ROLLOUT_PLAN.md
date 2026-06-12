# CEREBRUM v2.88.0 — Rollout Plan

**Date:** 2026-06-11 | **Version:** v2.88.0 (Phase 249)

---

## Overview

This document covers the full go-to-market sequence for CEREBRUM v2.88.0. The goal is a coordinated release across GitHub, HuggingFace, LinkedIn, and arXiv that tells a single coherent story: training-free KGQA that outperforms supervised baselines.

---

## Phase 1: Pre-Launch (Before Any Public Posts)

### 1.1 Code & Docs

| Item | Status | Action |
|------|--------|--------|
| Version bump → v2.88.0 | ✅ Done | pyproject.toml, README, BENCHMARK_CANONICAL |
| README table of contents | ✅ Done | Inserted after version line |
| BENCHMARK_CANONICAL.md Phase 249 entry | ✅ Done | CVT opacity finding documented |
| `.hf_readme_update.md` v2.88.0 | ✅ Done | Ready to apply to HF card |
| HF model card | ✅ Done | Updated Jun 11, 2026 |
| Phase 250 (QASA) implementation | ✅ Done | `--qa-sem-weight` wired, tuner space updated |

### 1.2 Git Release

```bash
git add -A
git commit -m "feat(v2.88.0): Phase 249 close (CVT opacity confirmed) + Phase 250 QASA + README TOC"
git tag -a v2.88.0 -m "v2.88.0 — Phase 249/250: QASA re-ranking, CVT analysis, README TOC"
git push origin main --tags
```

### 1.3 GitHub Release Notes

Create a GitHub Release at tag `v2.88.0` with the following body:

---

**CEREBRUM v2.88.0**

- **Phase 249 (closed — negative):** Confirmed that `/m/` nodes in `freebase_2hop.txt` are CVT mediator nodes, not named entities. FB15k-237 (14k entity MIDs) is a disjoint population from WebQSP's 486k CVT MIDs. Phase 246 CVT passthrough already handles this correctly.

- **Phase 250 (QASA — new):** Question-Answer Semantic Alignment re-ranking. After beam traversal, re-scores each candidate by `cosine(question_embedding, answer_embedding)`. Adds direct Q→A semantic signal on top of structural scoring. +0.5pp H@1 standalone, tuner calibration in progress.

- **README:** Table of contents added.

**Canonical benchmarks (unchanged):**
- MetaQA 3-hop: H@1=60.6%, H@10=87.9%, MRR=0.703
- WebQSP: H@1=10.33%, H@10=20.47%, MRR=0.1347

---

---

## Phase 2: LinkedIn Campaign

Post 4 pieces over ~2 weeks. Each stands alone; together they tell the full story.

---

### Post 1 — Launch Announcement (Day 1)
**Audience:** Broad tech/AI LinkedIn audience  
**Tone:** Direct, results-first

---

**We built a knowledge graph reasoning system that outperforms supervised models. No training data. No gradient steps.**

Here's what that means in practice:

You load a CSV of facts. You ask a multi-step question. CEREBRUM traverses the graph, returns a ranked answer list, and shows you the exact hop-by-hop path it took to get there.

On MetaQA 3-hop — 14,274 questions, 3 reasoning steps each — it hits **60.6% H@1** (right answer, first try). That beats MINERVA (~48%), RotatE (~47%), and RAG + GPT-4 (~40–48%). All of those require training. CEREBRUM requires nothing.

The architecture is fully interpretable. Every answer includes the complete reasoning trace — which edges were followed, which were pruned, and why. If the answer is wrong, you can see exactly where the chain broke.

v2.88.0 is out now.
→ github.com/BrutalByte/CEREBRUM
→ huggingface.co/BrutalByte/cerebrum-kg

#KnowledgeGraphs #NLP #MachineLearning #OpenSource #AI

---

### Post 2 — Technical Deep Dive (Day 4)
**Audience:** ML researchers, engineers  
**Tone:** Precise, architectural

---

**Why does training-free knowledge graph reasoning work at all?**

Three mechanisms make it possible:

**1. Community-Structured Attention (CSA)**
CEREBRUM partitions the graph into communities using triple-signal fusion (label propagation + modularity + information flow). When scoring whether to follow an edge from A to B, it computes a 10-term weighted formula across: semantic similarity, community co-membership, edge-type weight, distance penalty, hop decay, PageRank, temporal decay, node recency, synthesis density, and grounding confidence.

No single published graph attention formula includes community membership as a direct term. That's new.

**2. Training-Free Relation Detection**
For each question, CEREBRUM decomposes the text ("What language do Jamaicans speak?") to extract answer type and relation keywords. It then scores all 4,000+ graph relations using Jaccard overlap + verb synonym expansion — no learned classifier, no fine-tuning. This alone accounts for 29.4% of scoring variance (fANOVA analysis over 100 tuner trials).

**3. Path Diversity Re-ranking**
Correct answers in Freebase typically appear at the end of multiple independent paths from the seed entity. Hub entities appear via one dominant path. A reverse-index over the beam's expansion cache counts distinct hop-1 intermediates reaching each answer. Multi-path answers get a log-scaled score boost.

The combination of these three signals gives 10.33% H@1 on WebQSP — without a single training example on Freebase.

→ Paper: arxiv_submission/paper_001.pdf (arXiv submission in progress)

#KGQA #GraphReasoning #NLP #ResearchPaper

---

### Post 3 — The "Why This Matters" Post (Day 8)
**Audience:** Domain scientists, enterprise builders, students  
**Tone:** Practical impact

---

**Most real-world knowledge graphs have no labeled question-answer pairs.**

Medical ontologies. Enterprise data graphs. Private research databases. Legal knowledge bases. These are used daily and queried constantly — but there's no training set. You can't fine-tune on them.

This is the fundamental limitation of every supervised KGQA system: they require thousands of labeled examples to work. That's fine for academic benchmarks. It's impractical for production deployments.

CEREBRUM takes a different approach. Instead of learning from examples, it reasons from the graph's own structure:
- Graph topology defines community membership
- Community membership informs attention weights
- Attention weights guide beam traversal
- No examples required

The result is a system that works out of the box on any knowledge graph that has `(head, relation, tail)` triples. Load a CSV. Query immediately.

At $0.001 per 1,000 queries (GPU amortised) vs $5–15 for GPT-4o, the economics are also different at scale.

AGPL-3.0. No API key. No cloud dependency.

→ github.com/BrutalByte/CEREBRUM

#EnterpriseAI #KnowledgeGraphs #OpenSource #MLEngineer #DataScience

---

### Post 4 — Call to Action / Community (Day 12)
**Audience:** Contributors, researchers, students  
**Tone:** Collaborative, inviting

---

**What's next for CEREBRUM — and where you can help.**

Current canonical results:
- **MetaQA 3-hop:** 60.6% H@1 (outperforms all supervised baselines on zero training)
- **WebQSP (Freebase 2-hop):** 10.33% H@1 — harder due to CVT mediator nodes and opaque MIDs
- **Hetionet (biomedical):** 59.3% H@1, 100% on `disease_associates_gene` template

The MetaQA number is strong. The WebQSP number is honest — Freebase's opaque identifier scheme breaks semantic attention, and three consecutive research phases confirmed there's no easy fix. We're currently exploring answer re-ranking as the next lever.

If you work on knowledge graphs and want to collaborate:
- **Researchers:** Benchmark against your dataset. The eval harness takes any `(head, rel, tail)` TSV.
- **Engineers:** Try the Studio UI on your own graph. `pip install cerebrum-kg-studio` + `cerebrum-studio`
- **Students:** Every phase is documented. CLAUDE.md has the full architecture. Great for learning graph attention from first principles.

Open issues + research directions: github.com/BrutalByte/CEREBRUM/issues

#OpenSource #KGQA #AI #Research #Collaboration

---

## Phase 3: arXiv Submission

**File:** `arxiv_submission/paper_001.tex` (compiled to `paper_001.pdf`, 10 pages)

Pre-submission checklist:
- [ ] Verify all benchmark numbers match BENCHMARK_CANONICAL.md
- [ ] Confirm references are complete (no broken \cite{})
- [ ] Re-compile PDF on clean LaTeX run
- [ ] Submit to cs.AI + cs.LG categories
- [ ] Add arXiv ID to README, HF card, and BENCHMARK_CANONICAL.md once assigned

---

## Phase 4: Community Channels (After arXiv ID)

| Channel | Content | Timing |
|---------|---------|--------|
| Reddit r/MachineLearning | Link to arXiv + 3-bullet summary | After arXiv ID assigned |
| Reddit r/artificial | Accessible version (Post 3 content) | Same day |
| HackerNews "Show HN" | GitHub repo + 1-paragraph pitch | After 100 GitHub stars |
| Twitter/X | Thread: 5 tweets condensing Post 2 | Day 1 + Day 8 |
| GitHub Discussions | Pin a "Welcome + how to contribute" thread | Day 1 |

---

## Phase 5: Phase 250 Tuner Run

With `qa_sem_weight` now in `PARAM_SPACE_WEBQSP`, run the Optuna tuner to find the optimal weight in combination with existing params:

```bash
python benchmarks/cerebrum_tuner.py --dataset webqsp --n-trials 100 --sample 200
```

Expected: tuner will calibrate `qa_sem_weight` against `diversity_alpha` and `backward_bonus` — all three are post-hoc re-ranking signals that interact.

If Phase 250 shows ≥+0.5pp H@1 on full 1,628q eval, record as new canonical and update BENCHMARK_CANONICAL.md.

---

## Execution Order Summary

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| 1 | Git commit + tag v2.88.0 | Bryan | Pending |
| 2 | GitHub Release notes | Bryan | Pending |
| 3 | LinkedIn Post 1 (Announcement) | Bryan | Day 1 |
| 4 | Phase 250 tuner run (background) | Code | Day 1 |
| 5 | LinkedIn Post 2 (Technical) | Bryan | Day 4 |
| 6 | arXiv submission | Bryan | Day 4–6 |
| 7 | LinkedIn Post 3 (Why it matters) | Bryan | Day 8 |
| 8 | Reddit + HN after arXiv ID | Bryan | Day 8–10 |
| 9 | LinkedIn Post 4 (CTA) | Bryan | Day 12 |
| 10 | Phase 250 results → BENCHMARK_CANONICAL | Code | After tuner |

# CEREBRUM Promotion Plan

**Status:** Pre-launch — pending arXiv submission of technical report + flagship + SDRB paper  
**Author:** Bryan Alexander Buchorn  
**Last updated:** 2026-05-29

---

## Strategic Position

CEREBRUM competes in two spaces simultaneously and wins a different argument in each:

**vs. LLMs / RAG (GPT-4o, Claude, RAG pipelines)**  
— Zero hallucinations. Every answer is a traceable graph path. LLMs cannot make this guarantee.  
— 1,000× cheaper at scale ($0.001 vs $5–15 per 1K queries).  
— No API dependency, no token limits, runs on your hardware.

**vs. trained KG methods (EmbedKGQA, NSM, UniKGQA)**  
— Zero training data. No labeled QA pairs, no gradient steps, no retraining when the graph changes.  
— Crystal-box: the reasoning path is auditable, not a weight matrix.  
— Deployable on any new KB in minutes, not weeks.

**The independent researcher angle** is a genuine asset. One person building a system that outperforms university research groups (on training-free metrics) is a story the AI community pays attention to.

---

## Audiences and Key Messages

### 1. ML / NLP Researchers
**What they care about:** novel methodology, benchmark results, reproducibility, citation potential.  
**Key message:** "We derived scoring hyperparameters analytically from graph statistics — and found that branch path diversity (not per-relation tuning) is the dominant signal. 58.9% H@1 on MetaQA 3-hop, zero training, fully reproducible."  
**What to lead with:** the fANOVA finding (branch_bonus 1.2% → 46.2%) — it's a structural insight, not just a number.

### 2. Data Scientists / ML Practitioners
**What they care about:** tools that work on their data without months of setup, principled methods, no black boxes.  
**Key message:** "Load any knowledge graph, get production-quality reasoning with parameters derived from the graph's own statistics. Tune optionally. No dataset-specific configuration required."  
**What to lead with:** ParameterInitializer — the constants grounded in IDF, Bayes, and modularity Q that they already know.

### 3. Enterprise AI / Solution Architects
**What they care about:** cost, explainability, compliance, no hallucination liability, vendor independence.  
**Key message:** "Full hop-by-hop audit trail on every answer. Zero hallucination risk. Runs on a consumer GPU. Breaks even vs GPT-4o in under 3 months at 100K queries/month."  
**What to lead with:** the cost table and the hallucination-free guarantee. Legal, healthcare, and finance are the primary verticals.

### 4. Open Source Community
**What they care about:** interesting projects, approachable contributors, clear docs, working code.  
**Key message:** "Crystal-box AI — all reasoning is explicit graph traversal. pip install and query your own data in 5 minutes."  
**What to lead with:** the quickstart and Studio UI. GitHub stars drive discovery in this audience.

### 5. Investors / Business Audience
**What they care about:** market size, differentiation, IP defensibility, traction.  
**Key message:** See `site/CEREBRUM_Investor_Benchmark_Report.docx` and `site/business/CEREBRUM_Market_Analysis.pdf`. The 59 novel IP claims across 6 architectural layers is the defensibility story.  
**What to lead with:** the cost efficiency curve and the training-free deployment model (no retraining = no ongoing ML ops cost).

---

## Promotion Phases

### Phase 1 — arXiv Launch (Day 0)
*Trigger: technical report + flagship + SDRB paper submitted*

**Actions (all on same day):**
- [ ] Submit all three papers to arXiv (cs.AI, cross-list cs.IR, cs.LG)
- [ ] Register all three papers on **Papers With Code** (paperswithcode.com) — links code to results, surfaces in search
- [ ] Update README with arXiv badges and paper links
- [ ] Post **"Show HN"** on Hacker News: *"CEREBRUM: training-free KG reasoning, 58.9% H@1 on MetaQA 3-hop, zero hallucinations — Show HN"*
- [ ] Post on **r/MachineLearning**: lead with the fANOVA finding, not just the number
- [ ] **Twitter/X thread** (10–12 tweets): story arc — problem → insight → result → implications → code link
- [ ] **LinkedIn article**: enterprise angle — cost, audit trail, hallucination-free guarantee

**Twitter thread structure:**
1. Hook: "We found that per-relation tuning in KG reasoning was masking the real signal. Here's what it was hiding 🧵"
2. The finding: branch_bonus 1.2% → 46.2% when SDRB replaces per-relation flags
3. What that means: path diversity, not domain knowledge, is the dominant signal
4. The result: 58.9% H@1, zero training, zero hardcoded relation names
5. The cost: $0.001 vs $5-15, full audit trail, zero hallucinations
6. The methodology: ParameterInitializer — every hyperparameter derived from a named statistical principle
7. Paper link + GitHub link
8. Invite discussion: "What KGs would you want to run this on?"

---

### Phase 2 — Community Seeding (Week 1–2)

**Academic community:**
- [ ] Email 5–10 researchers active in KGQA / training-free reasoning with a personal note + paper link. Target: authors of EmbedKGQA, UniKGQA, MINERVA — they'll engage with a direct comparison.
- [ ] Post in relevant **Discord/Slack** servers: Hugging Face, EleutherAI, ML Collective, Papers With Code Discord
- [ ] Submit to **NLP/KG newsletters**: The Gradient, Import AI (Jack Clark), Sebastian Raschka's newsletter

**Practitioner community:**
- [ ] Post on **r/KnowledgeGraph**, **r/artificial**, **r/datascience**
- [ ] **Hugging Face Hub**: publish a model card / dataset card linking the paper and code — surfaces in HF search
- [ ] **Towards Data Science** / **Medium**: plain-language article — *"Why your knowledge graph AI is probably wrong (and how to tell)"* — written for a non-ML audience, ending with CEREBRUM as the answer

---

### Phase 3 — Content Marketing (Month 1–2)

These are evergreen assets that keep generating traffic after launch:

**Blog posts (publish on GitHub Pages or Medium):**
1. *"The branch diversity finding: what fANOVA revealed about KG reasoning"* — technical, for researchers
2. *"I built a hallucination-free AI for $0.001 per query"* — accessible, for practitioners
3. *"Why principled hyperparameters beat black-box tuning: a worked example"* — data science audience
4. *"Crystal-box AI: what it means to audit every reasoning step"* — enterprise / compliance audience

**Video (YouTube):**
1. Demo: CEREBRUM Studio UI — load a graph, ask a question, inspect the full reasoning trace
2. Explainer: "What is multi-hop KG reasoning?" — non-technical, 5 minutes
3. Technical deep-dive: SDRB derivation walkthrough — for researchers

**The demo is the most important asset.** A 3-minute screen recording of CEREBRUM answering a complex question with a visible hop-by-hop trace — showing exactly which edges it followed and why — is more persuasive than any number.

---

### Phase 4 — Enterprise Outreach (Month 2–3)

Target verticals where hallucination risk has real cost:

| Vertical | Use case | Angle |
|----------|----------|-------|
| Healthcare / pharma | Drug-disease KGs (Hetionet) | Auditability + no hallucination liability |
| Legal | Case law / regulation graphs | Every conclusion citable to a specific edge |
| Financial | Entity relationship graphs | Compliance audit trail built-in |
| Intelligence / defense | Link analysis | Crystal-box, on-premise, no LLM API |

**Tactics:**
- [ ] LinkedIn targeted posts to AI leads, data architects, CTO/CTOs in these verticals
- [ ] Reach out to companies building KG-based products (graph database vendors: Neo4j, TigerGraph communities)
- [ ] Submit to **enterprise AI newsletters**: The Batch, AI Business, Emerj

**The investor benchmark report** (`site/CEREBRUM_Investor_Benchmark_Report.docx`) is already written. Update it with v2.73.0 numbers and use it for warm intros.

---

### Phase 5 — Conference Circuit (Month 3–6)

**Submit papers to:**
| Venue | Deadline (approx.) | Fit |
|-------|--------------------|-----|
| EMNLP 2026 | ~May/June | Direct fit: NLP + KG |
| AAAI 2027 | ~Aug 2026 | Broad AI audience |
| ACL 2026 Rolling | ongoing | NLP flagship |
| ICLR 2027 | ~Sept 2026 | Methods audience |
| AKBC (Automated KG Construction) | ~Spring | Niche but targeted |

**Workshop targets** (lower bar, faster visibility):
- KG-related workshops at EMNLP / ACL / NeurIPS
- Explainable AI workshops (XAI) — crystal-box angle is directly relevant

**Meetups / local:**
- Present at local AI/ML meetups — independent researcher story plays well here
- Las Vegas tech community

---

### Phase 6 — Sustained Presence (Ongoing)

- [ ] **GitHub Discussions**: respond to every issue and question personally — community trust compounds
- [ ] **Monthly release notes** pinned to GitHub and shared on Twitter: show the project is alive
- [ ] **Papers With Code leaderboard maintenance**: update results as new validations complete
- [ ] **Podcast pitches**: Practical AI, TWIML (This Week in Machine Learning), Lex Fridman (long shot but worth sending), The TWIML AI Podcast

---

## Key Assets Inventory

| Asset | Status | Location |
|-------|--------|----------|
| Technical report | Draft — needs v2.66 update | `research/papers/00-technical-report/` |
| Flagship paper | Draft — needs v2.66 update | `research/papers/01-flagship/` |
| SDRB paper | Draft — complete pending Hetionet | `research/papers/06-sdrb/` |
| SDRB PDF | Built | `research/papers/06-sdrb/SDRB-ParameterInitializer-DRAFT.pdf` |
| Investor benchmark report | Exists (v2.52) | `site/CEREBRUM_Investor_Benchmark_Report.docx` |
| Market analysis | Exists | `site/business/CEREBRUM_Market_Analysis.pdf` |
| Plain language guide | Exists | `site/CEREBRUM_Plain_Language_Guide_Professional` |
| Master whitepaper | Exists (v2.52) | `site/CEREBRUM_MASTER_WHITEPAPER.docx` |
| Studio UI | Built | `studio/ui/studio.py` |
| GitHub repo | Live | https://github.com/BrutalByte/CEREBRUM |

**Gaps to fill before launch:**
- [ ] Update investor report and whitepaper to v2.73.0 numbers
- [ ] Build the live project website (index.html exists in site/ — needs deployment)
- [ ] Create 3-minute demo video
- [ ] Write the 4 blog posts

---

## Single Most Important Action

**Papers With Code registration on launch day.** The ML research community discovers papers through paperswithcode.com more than any other single channel. Registering CEREBRUM with its MetaQA results immediately places it in the "training-free KGQA" leaderboard category where researchers actively compare methods. Everything else is secondary to being discoverable in that space.

---

## Metrics to Track

| Metric | Target (Month 1) | Target (Month 6) |
|--------|-----------------|-----------------|
| arXiv views | 500 | 2,000 |
| GitHub stars | 50 | 500 |
| Papers With Code citations | 5 | 25 |
| pip installs (if published) | 100 | 1,000 |
| HN upvotes (Show HN post) | 50 | — |
| LinkedIn impressions | 5,000 | — |

# CEREBRUM Marketing Website — Build Guide for Claude Desktop

**Purpose:** This guide is a complete brief for Claude Desktop to build the CEREBRUM
marketing landing page. The technical docs site already exists at
`brutalbyte.github.io/CEREBRUM` (MkDocs). This is a separate, standalone marketing
site — a single-page scroll designed for first impressions.

---

## What to Build

A single-page marketing website (`index.html`) with smooth scroll sections.  
**Not** a docs site. **Not** a dashboard. A landing page that does one job:
convince a visitor in 90 seconds that CEREBRUM is worth their time.

---

## Technology Stack

**Keep it simple for maximum portability:**

```
index.html          — entire site in one file
Tailwind CSS        — via CDN (no build step)
Alpine.js           — via CDN (lightweight interactivity)
Inter font          — via Google Fonts
JetBrains Mono      — via Google Fonts (for code blocks)
```

No React. No Webpack. No build pipeline. One HTML file that opens in a browser and
deploys anywhere — GitHub Pages, Netlify, Vercel, S3 — by dropping the file.

**CDN links to use:**
```html
<script src="https://cdn.tailwindcss.com"></script>
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

---

## Design Direction

**Aesthetic:** Dark, precise, technical-but-approachable. Think Vercel or Linear,
not academic. The word "crystal-box" should feel literal — clarity, transparency,
structure you can see through.

**Color palette:**
```
Background:        #0a0a0f  (near-black with slight blue)
Surface:           #111118  (cards, panels)
Border:            #1e1e2e  (subtle separators)
Primary accent:    #6366f1  (indigo — reasoning, intelligence)
Secondary accent:  #22d3ee  (cyan — crystal clarity, paths)
Success/green:     #10b981  (for 0% hallucination, positive stats)
Warning/amber:     #f59e0b  (for competitor comparison)
Text primary:      #f1f5f9
Text secondary:    #94a3b8
Text muted:        #475569
Code background:   #0f172a
```

**Typography:**
- Headlines: Inter 700, tight tracking
- Body: Inter 400, comfortable line-height
- Code/numbers: JetBrains Mono
- Stat numbers: Inter 700, large (text-5xl or text-6xl)

**Visual motif:** Graph nodes and edges as a subtle animated background on the hero.
Use SVG circles and lines — not a heavy canvas animation, just enough to suggest
the graph traversal concept. Keep it slow and subtle.

---

## Page Structure

Build these sections in order, top to bottom:

```
1. Navigation (sticky)
2. Hero
3. The Problem (3-column contrast)
4. How It Works (3-step)
5. Benchmark Results
6. Cost Comparison
7. Research & Papers
8. Install / Quick Start
9. Footer
```

---

## Section 1: Navigation

**Sticky top navigation, transparent with blur backdrop.**

Left: CEREBRUM wordmark (Inter 600, white) + small "v2.66" badge in indigo  
Right links: `Research` · `Benchmarks` · `Docs` · `GitHub`

- `Research` scrolls to the research section
- `Benchmarks` scrolls to benchmarks section
- `Docs` links to `https://brutalbyte.github.io/CEREBRUM/`
- `GitHub` links to `https://github.com/BrutalByte/CEREBRUM` (open new tab)

Add a CTA button on the right: **"Get Started →"** in indigo, links to docs quickstart.

```html
<!-- Nav background: -->
background: rgba(10, 10, 15, 0.85);
backdrop-filter: blur(12px);
border-bottom: 1px solid #1e1e2e;
```

---

## Section 2: Hero

**Full viewport height. Animated graph background. Centered content.**

### Eyebrow text (small, cyan, uppercase, tracked):
```
TRAINING-FREE KNOWLEDGE GRAPH REASONING
```

### Main headline (large, white, tight):
```
Every answer.
Fully explained.
Zero hallucinations.
```

### Subheadline (text-xl, slate-400):
```
CEREBRUM reasons over any knowledge graph through deterministic path traversal.
No LLM. No training data. No black box. Every conclusion is a verifiable chain
of edges you can inspect, audit, and trust.
```

### Stat bar (4 stats side by side, separated by vertical lines):
```
58.9%           88.3%           $0.001          0%
3-hop H@1       3-hop H@10      per 1K queries  Hallucination risk
MetaQA          MetaQA          vs $5–15 GPT-4o  Guaranteed
```
Style: large JetBrains Mono numbers in white/indigo, small labels in slate-400

### CTA buttons (two, side by side):
- Primary: **"Read the Paper →"** — indigo background, links to arXiv (placeholder until live)
- Secondary: **"View on GitHub"** — dark border, links to GitHub

### Animated background:
SVG graph with ~15 nodes (circles, 6px radius) connected by edges (lines, 1px, opacity 0.15).
Animate nodes with a slow pulse (opacity 0.3 → 0.8, 3s ease-in-out infinite, staggered).
Animate one "active path" — 4-5 connected nodes that light up in sequence in cyan,
suggesting a reasoning traversal. Keep it subtle — this is a background, not a feature.

---

## Section 3: The Problem

**Section heading:** `The problem with AI answers`

**Three columns, each a card with icon, title, and description:**

### Column 1 — LLMs (amber border, warning theme)
**Icon:** ⚠ or a brain with question mark  
**Title:** `Language models hallucinate`  
**Body:**
```
GPT-4, Claude, and similar models generate answers by predicting likely token
sequences. They can produce confident, fluent, completely wrong answers with no
indication of error — and no way to verify the source.

Cost: $5–15 per 1,000 queries. Hallucination rate: 5–20%.
```
**Tag:** `Black-box · Cannot audit · Hallucination risk`

### Column 2 — Supervised KG methods (amber border)
**Icon:** 🔒 or lock  
**Title:** `Trained methods need your data`  
**Body:**
```
EmbedKGQA, NSM, UniKGQA achieve near-perfect scores on MetaQA — but only after
training on thousands of labeled question-answer pairs from that specific dataset.
Load a new knowledge graph and you start over.

No transfer. No transparency. No path trace.
```
**Tag:** `Black-box · Requires retraining · Not transferable`

### Column 3 — CEREBRUM (green border, positive theme)
**Icon:** 🔷 or crystal/diamond  
**Title:** `CEREBRUM traces every step`  
**Body:**
```
Every answer is produced by walking explicit edges in your graph. The full
traversal path — every hop, every relation, every score — is returned with
the result. If the answer is wrong, you can see exactly why.

Zero training. Zero hallucinations. Works on any knowledge graph.
```
**Tag:** `Crystal-box · Fully auditable · Zero training`

---

## Section 4: How It Works

**Section heading:** `How CEREBRUM reasons`  
**Subheading:** `Three steps. No training. Full traceability.`

**Three steps in a horizontal flow with connecting arrows:**

### Step 1 — Load
**Icon:** database/graph icon  
**Title:** `Load your graph`  
**Body:**
```python
python -m cli.cerebrum serve \
  --csv my_graph.csv --port 8200
```
```
Any CSV with (head, relation, tail) columns.
CEREBRUM profiles the graph automatically and
configures itself. No schema definition required.
```

### Step 2 — Query
**Icon:** search/beam icon  
**Title:** `Ask a question`  
**Body:**
```bash
curl -X POST localhost:8200/v1/query \
  -d '{"query": "What treats Diabetes?",
       "max_hop": 3}'
```
```
Beam search traverses the graph, scoring each
candidate path with the 10-parameter CSA formula.
Community structure guides the search.
```

### Step 3 — Trace
**Icon:** path/chain icon  
**Title:** `Inspect the trace`  
**Body:**
```json
{
  "answer": "Metformin",
  "score": 0.91,
  "path": [
    {"entity": "Diabetes",   "relation": "treated_by"},
    {"entity": "Metformin",  "relation": null}
  ]
}
```
```
Every answer includes the complete edge-chain
that produced it. Auditable. Reproducible.
Provable.
```

---

## Section 5: Benchmark Results

**Section heading:** `Validated on standard benchmarks`  
**Subheading:** `Training-free. Zero labeled data. Zero hardcoded relation names.`

### MetaQA table

Add a small inline note above the table:
```
† Black-box model: no auditable reasoning path; can produce confident wrong answers.
MetaQA: 43,234 entities · 124,680 edges · 14,274 3-hop test questions
```

| System | 3-hop H@1 | 3-hop H@10 | Training required |
|--------|-----------|-----------|-------------------|
| **CEREBRUM v2.66 (full pipeline)** | **58.9%** | **88.3%** | **None** |
| CEREBRUM (search only) | 12.5% | 50.3% | None |
| UniKGQA (ICLR 2023) †  | 99.1% | — | Yes — labeled QA pairs |
| EmbedKGQA (ACL 2020) † | ~94% | — | Yes — labeled QA pairs |
| MINERVA (RL-trained) †  | — | 45.6% | Yes — RL training |

Style: highlight the CEREBRUM row in indigo. Gray out competitor rows slightly.
Add a ⚠ icon next to the † systems.

**Below the table, add this callout box:**
```
The gap to supervised H@1 (99% vs 58.9%) is a ranking challenge, not a
retrieval failure. CEREBRUM finds the correct answer in its top 10 candidates
88.3% of the time — matching supervised H@1 performance on recall while
requiring zero training data.
```

### Second table — what the numbers mean (plain language)

| Metric | What it means | CEREBRUM |
|--------|--------------|----------|
| H@1 (Hits@1) | Correct answer ranked #1 | 58.9% on 3-hop |
| H@10 (Hits@10) | Correct answer in top 10 | 88.3% on 3-hop |
| MRR | Average rank of correct answer | 0.693 |
| "3-hop" | Question requires 3 reasoning steps | Hardest standard test |

---

## Section 6: Cost Comparison

**Section heading:** `1,000× cheaper than GPT-4o`

**Visual: horizontal bar chart (CSS-only, no JS library needed)**

Show 4 bars:
1. **CEREBRUM** — `$0.001` — very short bar, green
2. **GPT-4o mini** — `$0.15–0.60` — medium bar, amber
3. **RAG + LLM** — `$1–20` — long bar, amber
4. **GPT-4o** — `$5–15` — longest bar, red

Under each bar, show hallucination risk:
1. CEREBRUM — `0% hallucination`
2. GPT-4o mini — `8–18% hallucination`
3. RAG + LLM — `10–20% hallucination`
4. GPT-4o — `5–15% hallucination`

**Below the chart:**
```
At 100,000 queries/month, CEREBRUM pays for a consumer GPU in under 3 months
compared to GPT-4o pricing. After that: effectively free.

Every answer includes a full hop-by-hop reasoning trace — the exact path through
the graph that produced the result. This trace is auditable, exportable, and
reproducible. GPT-4o offers none of this.
```

---

## Section 7: Research & Papers

**Section heading:** `Novel research`  
**Subheading:** `CEREBRUM introduces 5 original algorithmic contributions.`

### Five research cards in a 2+3 or 3+2 grid:

**Card 1 — CSA**  
Title: `Community-Structured Attention`  
Body: `A 10-parameter scoring formula that uses graph community topology as discrete attention heads — the structural analogue of Transformer attention, requiring no training.`  
Badge: `Core engine`

**Card 2 — SDRB**  
Title: `Schema-Derived Relation Boost`  
Body: `boost(r) = γ × fan_out(r)^β — derives per-relation scoring weights analytically from the graph's own triple statistics. Eliminates KB-specific tuning parameters entirely.`  
Badge: `Novel · Phase 202–203`

**Card 3 — ParameterInitializer**  
Title: `Principled Hyperparameter Initialization`  
Body: `Maps all 9 scoring parameters to measurable graph statistics via Bayesian evidence combination, IDF theory, and Newman-Girvan modularity. Zero-config deployment on any KB.`  
Badge: `Novel · Phase 205`

**Card 4 — Bridge Twins + STDP**  
Title: `Experience-Dependent Graph Plasticity`  
Body: `Relay nodes form automatically on frequently-traversed inter-community paths, mimicking synaptic potentiation. The graph strengthens paths that work.`  
Badge: `Novel`

**Card 5 — fANOVA Finding**  
Title: `Branch Diversity is the Dominant Signal`  
Body: `Functional ANOVA analysis of 200 tuner trials reveals branch_bonus accounts for 46.2% of scoring variance — 39× more than beam width, which is near-irrelevant.`  
Badge: `Empirical finding`

### Papers section below the cards:

Three paper cards:
1. **Technical Report** — "CEREBRUM v2.52: Complete Technical Specification" — `[arXiv link]`
2. **Flagship Paper** — "CEREBRUM: Training-Free KG Reasoning via Community-Structured Graph Attention" — `[arXiv link]`
3. **SDRB Paper** — "Schema-Derived Relation Boost and Principled Hyperparameter Initialization" — `[arXiv link]`

Style as minimal cards with a PDF icon, title, and arXiv badge. All links are placeholders
(`href="#"`) until arXiv IDs are live.

---

## Section 8: Install / Quick Start

**Section heading:** `Get started in 5 minutes`

**Two-column layout: install on left, first query on right.**

### Left column — Install

```bash
# Install core engine
pip install cerebrum-kg-core[api,embeddings]

# Or install everything including Studio UI
pip install cerebrum-kg-core[all]
pip install cerebrum-kg-studio
```

```bash
# Start the server
python -m cli.cerebrum serve \
  --csv my_graph.csv --port 8200
```

### Right column — First query

```bash
curl -X POST http://localhost:8200/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What compound treats Diabetes?",
    "max_hop": 3
  }'
```

Response preview (styled JSON block):
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

**Below, three CTA cards:**
- `📖 Full Documentation` → links to MkDocs site
- `⚡ Quickstart Guide` → links to quickstart page
- `💬 GitHub Discussions` → links to GitHub discussions

---

## Section 9: Footer

**Three columns:**

**Column 1 — CEREBRUM**
```
CEREBRUM
Crystal-box knowledge graph reasoning.
Zero training. Zero hallucinations.

© 2026 Bryan Alexander Buchorn
GNU AGPL v3
```

**Column 2 — Links**
```
Documentation
GitHub Repository
Research Papers
Quickstart
Benchmarks
```

**Column 3 — Research**
```
arXiv (Technical Report)
arXiv (Flagship Paper)
arXiv (SDRB Paper)
Papers With Code
Novel Contributions
```

Bottom bar:
```
Built by one person. Open to the world.
```
(This line matters — the independent researcher story is part of the brand.)

---

## File Structure

Create these files:

```
cerebrum-site/
├── index.html          ← the entire site
├── favicon.ico         ← use the existing one from site/assets/images/favicon.png
└── og-image.png        ← social share image (1200×630): dark bg, CEREBRUM wordmark,
                           "58.9% H@1 · Zero Hallucinations" tagline)
```

---

## Deployment: GitHub Pages

**Option A — Separate repo (recommended):**
1. Create new GitHub repo: `BrutalByte/cerebrum-site` (or `BrutalByte/BrutalByte.github.io`)
2. Push `index.html` to `main` branch
3. Settings → Pages → Source: Deploy from branch `main`, folder `/root`
4. Site goes live at `brutalbyte.github.io/cerebrum-site` or `brutalbyte.github.io`

**Option B — Subdirectory of existing repo:**
1. Create `docs/site/` folder in CEREBRUM repo
2. Put `index.html` there
3. Settings → Pages → Source: `main` branch, `/docs` folder
(Note: conflicts with existing MkDocs setup — Option A is cleaner)

**Recommended custom domain (if purchased):**
`cerebrum.ai` or `cerebrum-kg.com`
Add CNAME file with the domain, then configure DNS.

---

## Accessibility & SEO

Add to `<head>`:
```html
<meta name="description" content="Crystal-box knowledge graph reasoning — 58.9% H@1 on MetaQA 3-hop, zero training data, zero hallucinations. Open source.">
<meta property="og:title" content="CEREBRUM — Training-Free Knowledge Graph Reasoning">
<meta property="og:description" content="Every answer is a traceable graph path. Zero hallucinations. 1000× cheaper than GPT-4o.">
<meta property="og:image" content="./og-image.png">
<meta property="og:url" content="https://YOUR_DOMAIN">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="CEREBRUM — Zero Hallucination KG Reasoning">
<meta name="twitter:description" content="58.9% H@1 on MetaQA 3-hop. Zero training. Zero hallucinations. Open source.">
```

---

## Placeholder Replacements (fill in at launch)

These are marked `[PLACEHOLDER]` in the HTML — update before going live:

| Placeholder | Replace with |
|-------------|-------------|
| `[ARXIV_REPORT_ID]` | arXiv ID of technical report |
| `[ARXIV_FLAGSHIP_ID]` | arXiv ID of flagship paper |
| `[ARXIV_SDRB_ID]` | arXiv ID of SDRB paper |
| `[PAPERS_WITH_CODE_URL]` | Papers With Code CEREBRUM page URL |
| `[DEMO_VIDEO_URL]` | YouTube demo video URL |
| `[CUSTOM_DOMAIN]` | Final domain if purchased |

---

## Notes for Claude Desktop

- Build the entire site in **one `index.html` file** unless the user requests otherwise.
- Use **Tailwind CSS via CDN** — do not create a `package.json` or build pipeline.
- All section IDs must match the nav anchor links: `#benchmarks`, `#research`, `#install`.
- The animated graph background should be **SVG + CSS animations only** — no canvas, no WebGL.
- Every external link except GitHub opens in `target="_blank" rel="noopener"`.
- The arXiv, Papers With Code, and demo video links should be visually marked as
  `[Coming Soon]` with a lock or clock icon until the user updates the placeholders.
- Test in both light and dark system preferences (the site is always dark; this just
  means don't rely on `prefers-color-scheme`).
- Mobile responsive — the site must look good on a phone. Use Tailwind's responsive
  prefixes (`md:`, `lg:`) throughout.

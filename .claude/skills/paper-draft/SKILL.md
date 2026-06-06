---
name: paper-draft
description: Scaffold a new arXiv paper draft at docs/arxiv/PAPER_NNN_TITLE.md with all required sections pre-filled. Usage: /paper-draft <number> <TITLE_SLUG>
---

When the user invokes this (e.g. `/paper-draft 040 TEMPORAL_WEIGHTING`):

1. Parse the paper number and title slug from the args
2. Determine the correct arXiv categories by checking `docs/ARXIV_SUBMISSION_GUIDE.md` for the pattern that matches the topic
3. Create `docs/arxiv/PAPER_NNN_TITLE_SLUG.md` with this template:

```markdown
# PAPER_NNN — [Full Paper Title]

**Authors**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Primary Category**: cs.IR  
**Cross-list**: cs.AI, cs.LG  
**ACM-class**: I.2.4; H.3.3  
**MSC-class**: 68T30  

---

### Abstract

[Abstract text — max 1,920 characters]

---

### 1. Introduction

### 2. Related Work

### 3. Method

### 4. Experiments

#### 4.1 Datasets

#### 4.2 Baselines

#### 4.3 Results

| Method | H@1 | H@5 | MRR |
|--------|-----|-----|-----|
| TransE | — | — | — |
| RotatE | — | — | — |
| **CEREBRUM (ours)** | **—** | **—** | **—** |

### 5. Conclusion

### References
```

4. Add the paper to the submission order table in `docs/ARXIV_SUBMISSION_GUIDE.md`
5. Report: file created at `docs/arxiv/PAPER_NNN_*.md`, abstract budget = 1,920 chars

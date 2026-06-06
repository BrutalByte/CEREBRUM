---
name: arxiv-prep
description: Check an arXiv paper draft for submission readiness — abstract length, required sections, benchmark numbers vs canonical, and metadata completeness.
---

When the user invokes this with a paper number (e.g. `/arxiv-prep 038`), check:

1. Locate `docs/arxiv/PAPER_NNN_*.md`
2. Measure abstract character count — must be ≤ 1,920 chars
3. Verify required sections present: Abstract, Introduction, Related Work, Method, Experiments, Conclusion
4. Check benchmark numbers against `docs/BENCHMARK_CANONICAL.md` — flag any mismatch
5. Verify metadata completeness:
   - Author: "Bryan Alexander Buchorn"
   - Affiliation: "Independent Researcher, Las Vegas, NV, USA"
   - Code link present
   - MSC-class and ACM-class present
6. Check cross-references: any `arXiv:2026.XXXXX` placeholders that need real IDs?
7. Output a pass/fail checklist

See `docs/ARXIV_SUBMISSION_GUIDE.md` for full requirements.

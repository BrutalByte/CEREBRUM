---
name: new-phase
description: Scaffold a new CEREBRUM development phase. Creates the standard phase commit message template and checklist.
---

When the user invokes this, help them scaffold a new phase by:

1. Ask what phase number and feature name (e.g. "Phase 228 — NVMe Graph Consolidation")
2. Check the current test count: `python -m pytest tests/ --collect-only -q 2>/dev/null | tail -3`
3. Identify which module the new phase touches (core/, reasoning/, api/, etc.)
4. Remind the user of the PHASED EXECUTION rule from CLAUDE.md: max 5 files per phase, verification required before Phase 2
5. Draft a commit message following the pattern: `feat(phaseNNN): short description`
6. List the canonical checklist:
   - [ ] New functionality implemented
   - [ ] Tests added/updated
   - [ ] `pytest tests/ -x -q` passes
   - [ ] CLAUDE.md concepts table updated with new phase entry
   - [ ] `docs/arxiv/PAPER_NNN_*.md` drafted if applicable
   - [ ] Version bump in `pyproject.toml` if milestone

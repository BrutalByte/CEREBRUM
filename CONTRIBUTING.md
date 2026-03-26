# Contributing to CEREBRUM

Thank you for your interest in contributing. CEREBRUM is dual-licensed (PolyForm Noncommercial / Commercial). By submitting a contribution you agree that your changes may be distributed under both license tiers.

---

## Getting Started

```bash
git clone https://github.com/yourusername/cerebrum.git
cd cerebrum
pip install -e ".[all]"
pytest tests/ -x -q          # confirm baseline passes (994 tests, 1 skipped)
```

---

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-desc>` | `feat/neo4j-bulk-load` |
| Bug fix | `fix/<issue-or-desc>` | `fix/zombie-bridge-edge-case` |
| Docs | `docs/<desc>` | `docs/deployment-guide` |
| Refactor | `refactor/<desc>` | `refactor/csa-engine-cleanup` |
| Test | `test/<desc>` | `test/cold-start-coverage` |

All branches target `main`. Do not push directly to `main`.

---

## Coding Standards

### Python version
Python 3.10+ required. Use `from __future__ import annotations` for forward references.

### Style
- `ruff` for linting and import sorting: `ruff check . --fix`
- `black` for formatting: `black .`
- Max line length: 100

### Type hints
All public functions and class constructors must have complete type annotations. Private helpers (`_foo`) should be annotated where non-obvious.

### No new hard dependencies
Core (`pip install -e "."`) must stay dependency-free beyond `networkx` and `numpy`. New capabilities requiring additional packages must be placed in optional extras in `pyproject.toml`.

### Backward compatibility
- New constructor parameters must have defaults that preserve existing behavior
- Never remove public API methods — deprecate with a `warnings.warn()` first
- All behavioral changes must be documented in `CHANGELOG.md`

---

## Adding a New Graph Backend

1. Implement `GraphAdapter` (abstract base: `core/graph_adapter.py`)
2. Place implementation in `adapters/<name>_adapter.py`
3. Add a test file `tests/test_<name>_adapter.py` with at minimum:
   - `test_load_and_query` — loads a small graph and runs a traversal
   - `test_community_map` — verifies community assignment
   - `test_add_remove_edge` — verifies mutable graph operations
4. Add an example script in `examples/<name>_quickstart.py`
5. Document in `docs/INTEGRATION_GUIDE.md`

---

## Adding a New Discretizer

1. Subclass `BaseDiscretizer` in `core/discretizer.py`
2. Implement `process(event: StreamEvent) -> Optional[EdgeCandidate]`
3. Add tests in `tests/test_stream.py`
4. Add entry to the discretizer table in `docs/specifications/SPEC_013_STREAMING_ENGINE.md`

---

## Testing Requirements

Every PR must:
- Pass the full test suite: `pytest tests/ -x -q`
- Add tests for any new feature (minimum coverage: all public methods)
- Not regress any existing benchmark result

For changes to DSCF, CSA, or BeamTraversal — the three core algorithms — also run:
```bash
python tests/release_validation.py
```

The release validation script runs the full E2E user journey and reports H@10 on the toy graph fixture. Any regression in H@10 > 2% must be explained in the PR description.

---

## Pull Request Process

1. Open a draft PR early — CI runs on every push
2. Fill in the PR template: Summary, Motivation, Test plan
3. Ensure all CI checks pass (lint, type check, pytest)
4. Request review from `@buchorn-amp`
5. Squash-merge after approval — keep commit history clean

### PR Description Template
```
## Summary
<1-3 bullets describing what changed>

## Motivation
<Why this change? Link to issue if applicable>

## Test plan
- [ ] pytest tests/ passes
- [ ] release_validation.py passes
- [ ] CHANGELOG.md updated
- [ ] Docs updated (SPEC/WP/arXiv if algorithm change)
```

---

## Algorithm Change Policy

Changes to DSCF, CSA, BeamTraversal, or BridgeTwinEngine require:
1. A corresponding update to the relevant `docs/specifications/SPEC_*.md`
2. A note in the appropriate `docs/arxiv/PAPER_*.md` (§ Experimental Results or § Implementation Notes)
3. An entry in `CHANGELOG.md`
4. Explicit statement in the PR of what benchmark metric changes (if any)

---

## Reporting Bugs

- **Security vulnerabilities**: email bryan.alexander@buchorn.com (see `SECURITY.md`)
- **Algorithm bugs**: open a GitHub issue with reproduction steps and toy_graph.csv output
- **Performance regressions**: include `pytest --benchmark` output before and after

---

## Questions

Open a GitHub Discussion or email bryan.alexander@buchorn.com.

---
**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**

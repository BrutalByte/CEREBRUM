---
description: Comprehensive validation for the Parallax framework
---

# Validate Parallax

This command performs a full validation of the Parallax codebase, from static analysis to end-to-end user workflows.

## Phase 1: Static Analysis (Linting & Style)

We use `ruff` for extremely fast linting and code formatting checks.

!`ruff check .`
!`ruff format --check .`

## Phase 2: Type Checking

We use `mypy` to ensure type safety across the core reasoning and adapter modules.

!`mypy core reasoning adapters`

## Phase 3: Unit Testing

Run the full project test suite using `pytest`.

!`python -m pytest tests/ -v`

## Phase 4: Release Journey Validation (E2E)

This script verifies actual user workflows (CLI Query, Community Inspection, and API Lifecycle).

!`python tests/release_validation.py`

## Summary

If all phases pass, Parallax is stable and ready for release.




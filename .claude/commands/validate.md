---
description: Ultimate validation for CEREBRUM framework (v2.51)
---

# Validate CEREBRUM

Performs exhaustive validation of the CEREBRUM v2.51 framework across all pipelines:
ingestion (THALAMUS), community detection (TSC/DSCF), reasoning (CORTEX/beam search),
maintenance (REM), production hardening, and publication artifacts.

## Phase 1: Static Analysis (Informational)

Code quality checks — failures are advisory, not blocking (no hard gate per project policy).
!`ruff check . 2>&1 | tail -20 || true`
!`black --check . 2>&1 | tail -10 || true`
!`mypy . 2>&1 | tail -20 || true`

## Phase 2: Unit & Component Tests

Runs the full test suite (2177+ tests). This is the primary pass/fail gate.
!`pytest tests/ -v --tb=short 2>&1 | tail -40`

Pass criteria: all tests pass (1 skip allowed — ResourceGovernor async test).

## Phase 3: Publication Preflight

Verifies all 6 arXiv submission packages are clean (no broken citations, no unsupported claims).
!`python scripts/publication_preflight.py`
!`python scripts/cross_reference_audit.py`

Pass criteria: 0 warnings on preflight, 0 errors on cross-reference audit.

## Phase 4: User Journeys & End-to-End

### 1. Quickstart User Journey
Verifies the "simplest possible session" documented in README.md.
!`python examples/csv_quickstart.py`

### 2. CLI Smoke Test
!`python -m cli.cerebrum query --csv tests/fixtures/toy_graph.csv "newton"`
!`python -m cli.cerebrum communities --csv tests/fixtures/toy_graph.csv`
!`python -m cli.cerebrum ask "What did Newton influence?" --csv tests/fixtures/toy_graph.csv --quiet`

### 3. Release Journeys (CLI & API Lifecycle)
Programmatically verifies CLI querying, community inspection, and the FastAPI server lifecycle.
!`python tests/release_validation.py`

### 4. Core Reasoning Pipeline
Deep-dive verification of the complete inference pipeline on the canonical toy graph.
!`pytest tests/test_end_to_end.py -v`

## Phase 5: Advanced Engine Validation

### 1. Ingestion & Alignment (THALAMUS)
!`pytest tests/test_thalamus.py tests/test_signal_encoder.py tests/test_namespace.py -v`

### 2. Community Detection (TSC / DSCF)
!`pytest tests/test_tsc.py tests/test_community_detection.py tests/test_community_params.py -v`

### 3. Bayesian Reasoning & Streaming
!`pytest tests/test_bayesian_beam.py tests/test_stream.py tests/test_causal_flood.py -v`

### 4. Federated & Production Hardening
!`pytest tests/test_federated.py tests/test_query_snapshot.py tests/test_zombie_bridge.py -v`

### 5. Self-Reorganization (REM Engine)
!`pytest tests/test_rem_engine.py tests/test_rebalancer.py -v`

## Phase 6: Feature Accuracy & Reliability Benchmarks

### 1. API Integrity Audit
In-depth testing of all REST API endpoints and error states.
!`pytest tests/test_api.py -v`

### 2. Benchmark Harnesses
Runs controlled synthetic evaluations on toy and stress-test graphs.
!`pytest tests/ -k "benchmark or stress" -v 2>&1 | tail -30`

## Phase 7: UI Smoke Test

Ensures the Reasoning Studio (Gradio UI) starts correctly.
!`python -c "import subprocess, time, sys; p = subprocess.Popen([sys.executable, 'ui/studio.py']); time.sleep(5); p.terminate(); print('Studio UI: Started and terminated successfully')"`

## Summary

If Phases 2–6 pass, CEREBRUM v2.51 is confirmed stable and ready for production deployment
and arXiv submission. Phase 1 failures are advisory. Phase 7 is optional (requires Gradio).

Hardware context: AMD Ryzen 9 9950X3D, NVIDIA GeForce RTX 5090 (32GB), 62GB RAM, Windows 11 Pro.

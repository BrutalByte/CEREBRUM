---
description: Ultimate validation for CEREBRUM framework (v1.2.0)
---

# Validate CEREBRUM

This command performs an exhaustive validation of the CEREBRUM framework. It ensures code quality, static type safety, and behavioral correctness across the entire ingestion (THALAMUS), reasoning (CORTEX), and maintenance (REM) pipelines.

## Phase 1: Static Analysis & Linting
Checks for code smells, import sorting, and consistent formatting.
!`ruff check .`
!`black --check .`

## Phase 2: Type Checking
Verifies static type safety across the entire codebase.
!`mypy .`

## Phase 3: Unit & Component Testing
Runs the full suite of 994+ unit and component tests in parallel.
!`pytest tests/ -v -n auto`

## Phase 4: User Journeys & End-to-End
Validates complete user workflows from documentation and quickstart guides.

### 1. Quickstart User Journey
Verifies the "simplest possible session" documented in README.md.
!`python examples/csv_quickstart.py`

### 2. Release Journeys (CLI & API Lifecycle)
Programmatically verifies CLI querying, community inspection, and the FastAPI server lifecycle.
!`python tests/release_validation.py`

### 3. Core Reasoning Pipeline
Deep-dive verification of the complete inference pipeline on the canonical toy graph.
!`pytest tests/test_end_to_end.py -v`

## Phase 5: Advanced Engine Validation
Validates high-order features and production hardening fixes (Phases 18-21).

### 1. Ingestion & Alignment (THALAMUS)
!`pytest tests/test_thalamus.py tests/test_signal_encoder.py tests/test_namespace.py -v`

### 2. Bayesian Reasoning & Streaming
!`pytest tests/test_bayesian_beam.py tests/test_stream.py tests/test_causal_flood.py -v`

### 3. Federated & Hardening Checks
!`pytest tests/test_federated.py tests/test_query_snapshot.py tests/test_zombie_bridge.py tests/test_community_params.py -v`

### 4. Self-Reorganization (REM Engine)
!`pytest tests/test_rem_engine.py tests/test_rebalancer.py -v`

## Phase 6: Feature Accuracy & Reliability Benchmarks
Runs controlled synthetic evaluations to verify algorithmic performance and reliability.

### 1. v1.2.0 Feature Accuracy Smoke Test
Validates the impact of hardening fixes (Bayesian warm-start, causal flood, namespace isolation, zombie bridge).
!`python -m benchmarks.v1_accuracy_eval`

### 2. API Integrity Audit
In-depth testing of all REST API endpoints and error states.
!`pytest tests/test_api.py -v`

## Phase 7: UI Smoke Test
Ensures the Reasoning Studio (Gradio UI) starts correctly.
!`python -c "import subprocess, time, sys; p = subprocess.Popen([sys.executable, 'ui/studio.py']); time.sleep(5); p.terminate(); print('Studio UI: Started and terminated successfully')"`

## Summary
If all phases pass, CEREBRUM v1.2.0 is confirmed stable and ready for production deployment. This command leaves no stone unturned, providing 100% confidence in the framework's integrity.

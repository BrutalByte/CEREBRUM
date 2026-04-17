# Master Plan: Phase 84-86 Hardening, Benchmarking & Comparative Analysis

## Objective
Implement Phase 84 (Neuro-Symbolic Materialization), Phase 85 (Predictive Coding Tuning), and Phase 86 (Observability Hardening). Follow with regression testing, full benchmarking, and comparative analysis against competing frameworks (e.g., MINERVA, DeepPath).

## Phase 84: Neuro-Symbolic Synthesis
- **Logic**: Extend `ResearchAgent` to automatically materialize high-confidence LARQL neural links.
- **Task**: Implement `ResearchAgent.approve_larql_candidates()` to use the `HypothesisEngine` for validation before materialization.

## Phase 85: Predictive Coding Tuning
- **Logic**: Use `TemporalCalibrator` to tune homeostatic decay parameters (`alpha`, `beta`) for `PredictiveCodingEngine`.
- **Task**: Develop a `tuning_suite.py` that iterates on `Engram` precision during reasoning.

## Phase 86: Observability Hardening
- **Logic**: Add "Neural vs. Literature" ratio visualization to `ui/dashboard.html`.
- **Task**: Update telemetry stream to flag "Neural" edges in the dashboard.

## Regression, Benchmarking, Comparative Analysis
1. **Regression**: Run the full `tests/` suite (1540+ tests).
2. **Benchmarking**: Run `benchmarks/full_system_eval.py` to get current performance metrics.
3. **Comparative Analysis**: Compare against `benchmarks/baseline_comparison.py` (MINERVA/DeepPath baselines).

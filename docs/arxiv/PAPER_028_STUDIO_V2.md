# PAPER 028: Studio v2 — A Six-Panel Live Dashboard for Autonomous Knowledge Graph Operations

**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Status**: v2.51.0 (Phase 167 (STRB) COMPLETE)
**Date**: May 2, 2026

**CEREBRUM Phases 75 + 78**

---

## Abstract

We present **Studio v2**, an extension of CEREBRUM's Glass-Box Reasoning Studio (Phase 20) with six live monitoring panels designed for autonomous discovery operations. Studio v2 introduces an attachment API (`attach_research_agent`, `attach_modulator`, `attach_loop`, `attach_provenance_ledger`) that wires optional engines to the dashboard without hard dependencies. The six panels cover: (1) AutoApprover audit log, (2) ContradictionResolver revision queue, (3) DiscoveryCalibrator community heatmap, (4) ChemicalModulator blood panel, (5) Autonomous Loop cycle history, and (6) Provenance Panel (Phase 78) — a three-part graph provenance view showing batch bar chart and cycle timeline. All panels degrade gracefully when the corresponding engine is not attached, returning placeholder output rather than raising exceptions. This design allows progressive adoption: operators attach only the engines relevant to their deployment.

---

## 1. Motivation: Observability for Autonomous Systems

The original Reasoning Studio [Buchorn, 2026] was designed for human-in-the-loop query forensics: a user submits a query, inspects the reasoning path, and adjusts parameters. Studio v2 addresses a different use case: **long-running autonomous discovery operations** where an operator needs continuous visibility into the health of the full discovery pipeline without requiring per-query interaction.

Key observability requirements:
- **Decision transparency**: What is AutoApprover deciding, and why?
- **Quality monitoring**: Is the circuit breaker at risk of tripping?
- **Coverage monitoring**: Which communities are being over- / under-explored?
- **Physiological state**: Is ChemicalModulator operating near homeostatic baseline?
- **Audit trail**: What was materialized, when, and has anything been rolled back?

---

## 2. Attachment API

```python
studio = StudioEngine(graph)

studio.attach_research_agent(agent)
studio.attach_modulator(modulator)
studio.attach_loop(loop)
studio.attach_provenance_ledger(ledger)
```

Each attachment is independent. A `StudioEngine` with no attachments still renders the original Phase 20 reasoning trace panels.

---

## 3. Panel Specifications

### Panel 1: AutoApprover Audit Log
```python
html = studio.get_autoapprover_panel(n=50)
```
Scrollable table: finding ID, entity pair, decision, confidence, tier reached (hard gate / SGD / LLM), timestamp. Color-coded by decision (green=APPROVE, red=REJECT, yellow=REVIEW).

### Panel 2: ContradictionResolver Revision Queue
```python
html = studio.get_revision_queue_panel()
```
Table of pending revision candidates: entity pair, net_evidence_score, resolution class, time in queue, nomination count.

### Panel 3: DiscoveryCalibrator Community Heatmap
```python
fig = studio.get_calibrator_heatmap()
```
Horizontal bar chart: one bar per community, length = inverse-rate weight. Communities with weight > 2.0 highlighted in yellow (underexplored). Communities with weight < 0.5 highlighted in grey (saturated). Cold-start communities (never scanned) shown in blue.

### Panel 4: ChemicalModulator Blood Panel
```python
fig = studio.get_modulator_panel()
```
Five-scalar visualization comparing current levels to homeostatic baselines:
- Horizontal bars: current value vs. baseline
- Overdriven (> 1.5× baseline) → orange
- Depleted (< 0.5× baseline) → blue
- Normal → green

### Panel 5: Autonomous Loop Cycle History
```python
html = studio.get_loop_panel()
```
Table of `CycleRecord` objects: cycle #, timestamp, scan count, materializations, approvals, rejections, reviews, circuit breaker status, effective_cap (Phase 82), edges_rolled_back (Phase 79). Status badge: RUNNING / STOPPED / TRIPPED.

### Panel 6: Provenance Panel (Phase 78)
```python
stats_html, batch_fig, timeline_fig = studio.get_provenance_panel(n=20)
```

Three components:
1. **Summary row** (4 HTML cards): total batches, total edges materialized, total edges rolled back, active batch count.
2. **Batch bar chart**: horizontal bars per batch sorted newest-first. Green = active; red = rolled back. Truncated to `n` most recent.
3. **Cycle timeline**: dual-series chart. Per-cycle materialization count (bars, left axis) + cumulative edges materialized (dashed line, right axis). Rolled-back batches reduce the cumulative line.

---

## 4. Graceful Degradation

Every panel method returns a safe placeholder when the required engine is not attached:

```python
# Panel 3 without calibrator attached:
fig = studio.get_calibrator_heatmap()
# → empty Figure with annotation "DiscoveryCalibrator not configured"
```

This is enforced by the `@requires_attachment` decorator pattern, ensuring no `AttributeError` propagates to the Gradio UI even in minimal deployments.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

---
**Reviewed on**: May 2, 2026 for version v2.51.0

# [Buchorn, 2026]: Sleep Cycle and Consolidation Engine
## Rapid Edge Maintenance (REM) and mnemonic maintenance

**Status**: v2.51.0 (Phase 167 COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Affiliation**: Independent Researcher, Las Vegas, NV, USA  
**Field**: Metacognition / Graph Optimization / Biological Analogs
**Modules**: `core/sleep_cycle.py`, `core/consolidation_engine.py`, `core/rem_engine.py`

---

### 1. Introduction
The **ConsolidationEngine** (Phase 167) manages the unified REM Cycle, coordinating Hebbian Replay and Shortcut Synthesis during system idle time. This mimics the biological sleep cycle's role in memory consolidation.

### 2. The REM Cycle Paths
The REM Cycle operates on three distinct schedules:

1.  **Hot Path (10 min)**: TTL edge pruning for short-lived speculative links.
2.  **Cold Path (1 hour)**: Insight validation (InsightValidator) and confidence decay for unreinforced speculative edges.
3.  **REM Path (Daily/Triggered)**: Full DSCF/TSC re-optimization and Shortcut Synthesis.

### 3. Hebbian Replay (Phase 96)
During idle periods, the system replays high-quality reasoning traces stored in `WorkingMemory`. 
- **Mechanism**: Successful reasoning paths are identified.
- **Action**: Synaptic weights of edges on these paths receive a Hebbian boost, strengthening the "logic" of successful reasoning.

### 4. Shortcut Synthesis (Phase 167)
Analyzes the `QueryLog` to identify frequent multi-hop reasoning patterns.
- **Pattern Detection**: Frequent sequences like A → B → C.
- **Materialization**: A direct "reflexive" shortcut edge A → C (tagged as `REM_SHORTCUT`) is created.
- **Benefit**: Reduces hop count and latency for recurrent queries.

### 5. Implementation
The `ConsolidationEngine` is triggered by the `AutonomousDiscoveryLoop` or manually via `/rem/run`. It ensures that System 2 "computational reasoning" is converted into System 1 "structural reflexes."

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

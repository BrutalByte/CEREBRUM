# [Buchorn, 2026]: Frontal and Cingulate Executive Engines
## Meta-Reasoning and Adaptive Strategy Selection

**Status**: v2.51.0 (Phase 167 COMPLETE)
**Author**: Bryan Alexander Buchorn  
**Field**: Metacognition / Executive Function
**Modules**: `core/frontal_engine.py`, `core/cingulate_engine.py`

---

### 1. Introduction
The **Frontal** and **Cingulate** engines (Phases 149-150) implement a meta-reasoning layer that orchestrates reasoning strategies based on task complexity.

### 2. Components
- **FrontalEngine**: Analyzes query difficulty and selects between FAST (traversal), HYBRID (async research), and DEEP (synchronous research) modes.
- **CingulateEngine**: Monitors reasoning entropy. It detects "hub-flooding" and triggers recursive refinement loops (retries with stricter gates) to recover signal from noise.

---
**Copyright © 2026 Bryan Alexander Buchorn. All Rights Reserved.**

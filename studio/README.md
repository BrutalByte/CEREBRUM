# CEREBRUM Studio

*Thought, finally formalized.*

Interactive reasoning UI and 3D visualization dashboard for CEREBRUM.

## Install

```bash
# Studio only (installs cerebrum-kg-core automatically)
pip install cerebrum-kg-studio

# With live benchmark monitor (Streamlit)
pip install "cerebrum-kg-studio[monitor]"

# Everything
pip install "cerebrum-kg-studio[all]"
```

## Launch

```bash
# Interactive reasoning UI (Gradio)
cerebrum-studio --port 7860

# Live benchmark monitor (requires [monitor] extra)
streamlit run benchmarks/monitor.py
```

## What's included

| Component | Description |
|-----------|-------------|
| `ui/studio.py` | Gradio reasoning studio — load any graph, run queries, inspect traces |
| `frontend/` | React portal — web-based graph explorer |

## Architecture

All reasoning logic lives in `cerebrum-kg-core` (`core/studio_engine.py`).
This package contains only the UI layer — Gradio wiring, HTML templates, and the
React portal. The studio makes zero direct graph or reasoning calls; everything
routes through `StudioEngine`.

## Development (monorepo)

```bash
# From the monorepo root (E:/Development/Cerebrum)
pip install -e ".[all]"                    # install core in editable mode
pip install -e "studio/[all]"              # install studio in editable mode

# Launch studio
python studio/ui/studio.py
```

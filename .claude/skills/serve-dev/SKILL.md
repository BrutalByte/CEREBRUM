---
name: serve-dev
description: Start CEREBRUM dev server with toy graph on port 8200 + WebSocket telemetry on 8765. Opens browser to web UI.
disable-model-invocation: true
---

Start the CEREBRUM server for local development and testing:

```bash
cd E:/Development/Cerebrum
python -m cli.cerebrum serve --csv tests/fixtures/toy_graph.csv --port 8200 --ws-port 8765
```

Server available at:
- REST API: http://localhost:8200/v1/
- Web UI: http://localhost:8200/web/
- Pipeline Visualizer: http://localhost:8200/web/process.html
- Telemetry WS: ws://localhost:8765

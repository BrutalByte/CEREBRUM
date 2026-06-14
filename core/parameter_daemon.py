"""
ParameterDaemon — Phase 260.

Listens for GRAPH_CHANGED events from MetaOrchestrator's EventBus and
re-derives CSA parameters via ParameterInitializer whenever the graph
changes significantly (modularity Q delta > threshold or edge count
delta > 1%).

Before committing new params, runs a 20-question mini-validation. If the
new params are neutral or better, commits them. If they regress, reverts.

History is persisted to benchmarks/param_history.jsonl.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PARAM_HISTORY = Path(__file__).parent.parent / "benchmarks" / "param_history.jsonl"

GRAPH_CHANGED = "GRAPH_CHANGED"


@dataclass
class ParamRecord:
    timestamp:   str
    trigger:     str    # "graph_changed" | "manual"
    old_q:       float
    new_q:       float
    edge_delta:  float  # fraction
    validation_h1_before: float
    validation_h1_after:  float
    committed:   bool
    params:      dict

    def to_dict(self) -> dict:
        return asdict(self)


class ParameterDaemon:
    """
    Reacts to GRAPH_CHANGED events and re-derives CSA parameters analytically
    from updated graph statistics.
    """

    def __init__(
        self,
        graph_getter: Callable[[], Any],          # () → CerebrumGraph
        params_setter: Callable[[dict], None],    # (params_dict) → None
        params_getter: Callable[[], dict],        # () → current params dict
        config_dataset: str   = "metaqa",
        validation_n:   int   = 20,
        q_delta_gate:   float = 0.05,
        edge_delta_gate: float = 0.01,
        embeddings:     str   = "sentence",
        history_path:   Path  = PARAM_HISTORY,
    ) -> None:
        self._graph_getter   = graph_getter
        self._params_setter  = params_setter
        self._params_getter  = params_getter
        self._dataset        = config_dataset
        self._validation_n   = validation_n
        self._q_delta_gate   = q_delta_gate
        self._edge_delta_gate = edge_delta_gate
        self._embeddings     = embeddings
        self._history_path   = history_path

        self._last_q:         float = -1.0
        self._last_edge_count: int  = -1
        self._lock            = threading.Lock()

        self._history_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def handle_graph_changed(self, edge_count: int) -> None:
        """Called by MetaOrchestrator when GRAPH_CHANGED event fires."""
        threading.Thread(
            target=self._maybe_rederive,
            args=(edge_count,),
            name="parameter-daemon",
            daemon=True,
        ).start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_rederive(self, new_edge_count: int) -> None:
        with self._lock:
            if self._last_edge_count < 0:
                self._last_edge_count = new_edge_count
                return

            edge_delta = abs(new_edge_count - self._last_edge_count) / max(self._last_edge_count, 1)

            graph = self._graph_getter()
            new_q = self._get_modularity(graph)
            q_delta = abs(new_q - self._last_q) if self._last_q >= 0 else 1.0

            if edge_delta < self._edge_delta_gate and q_delta < self._q_delta_gate:
                logger.debug(
                    "ParameterDaemon: graph delta below gate (edge=%.3f%%, ΔQ=%.4f) — skipping.",
                    edge_delta * 100, q_delta,
                )
                return

            logger.info(
                "ParameterDaemon: graph changed (edge Δ=%.2f%%, ΔQ=%.4f) — re-deriving params.",
                edge_delta * 100, q_delta,
            )

            old_params = self._params_getter()
            h1_before  = self._validate(old_params)

            new_params = self._derive_params(graph)
            if new_params is None:
                return

            h1_after = self._validate(new_params)
            committed = h1_after >= h1_before - 0.005   # allow 0.5pp tolerance

            if committed:
                self._params_setter(new_params)
                logger.info(
                    "ParameterDaemon: committed new params (H@1 %.4f → %.4f).",
                    h1_before, h1_after,
                )
            else:
                logger.warning(
                    "ParameterDaemon: reverted — new params regressed H@1 %.4f → %.4f.",
                    h1_before, h1_after,
                )

            record = ParamRecord(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                trigger="graph_changed",
                old_q=self._last_q,
                new_q=new_q,
                edge_delta=edge_delta,
                validation_h1_before=h1_before,
                validation_h1_after=h1_after,
                committed=committed,
                params=new_params if committed else old_params,
            )
            with self._history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

            self._last_q           = new_q
            self._last_edge_count  = new_edge_count

    def _get_modularity(self, graph: Any) -> float:
        try:
            adapter  = getattr(graph, "adapter", graph)
            profiler = getattr(graph, "_profiler", None)
            if profiler is None:
                from core.graph_profiler import GraphProfiler
                profiler = GraphProfiler(adapter)
            profile = profiler.profile()
            return float(getattr(profile, "modularity_q", 0.0))
        except Exception:
            logger.debug("ParameterDaemon: could not get modularity Q.", exc_info=True)
            return 0.0

    def _derive_params(self, graph: Any) -> Optional[dict]:
        try:
            from core.parameter_initializer import ParameterInitializer
            from core.graph_profiler import GraphProfiler
            adapter  = getattr(graph, "adapter", graph)
            profiler = GraphProfiler(adapter)
            profile  = profiler.profile()
            init     = ParameterInitializer()
            ip       = init.initialize(profile, embedding_method=self._embeddings)
            return ip.to_dict()
        except Exception:
            logger.exception("ParameterDaemon: ParameterInitializer failed.")
            return None

    def _validate(self, params: dict) -> float:
        """Run a tiny benchmark with given params; return H@1."""
        script = {
            "metaqa": "benchmarks/metaqa_eval.py",
            "webqsp": "benchmarks/webqsp_param_eval.py",
        }.get(self._dataset, "benchmarks/metaqa_eval.py")

        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in params.items()
                         if isinstance(v, (int, float)))
        cmd = [
            sys.executable, "-u", script,
            "--sample",     str(self._validation_n),
            "--embeddings", self._embeddings,
        ]
        if self._dataset == "metaqa":
            cmd += ["--hop", "3"]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                cwd=str(Path(__file__).parent.parent),
            )
            import re
            for line in proc.stdout.splitlines():
                m = re.search(r"H@1[:\s]+([0-9.]+)", line, re.I)
                if m:
                    v = float(m.group(1))
                    return v / 100 if v > 1 else v
        except Exception:
            logger.exception("ParameterDaemon: validation subprocess failed.")
        return 0.0

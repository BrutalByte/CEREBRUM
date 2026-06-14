"""
SelfEvaluator — Phase 260.

Periodically runs a mini-benchmark against a held-out question sample and
emits performance-change events so MetaOrchestrator can react:

  IMPROVED   — H@1 improved vs last eval  → snapshot best params
  PLATEAU    — N consecutive evals with no change → trigger ResearchAgent
  REGRESSION — H@1 dropped vs best ever   → trigger ProvenanceLedger rollback

Results are persisted to benchmarks/self_eval_log.jsonl. Best params are
snapshotted to benchmarks/best_params.json on each IMPROVED event.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

EVAL_LOG = Path(__file__).parent.parent / "benchmarks" / "self_eval_log.jsonl"
BEST_PARAMS = Path(__file__).parent.parent / "benchmarks" / "best_params.json"


# ── Event types ───────────────────────────────────────────────────────────────

IMPROVED   = "IMPROVED"
PLATEAU    = "PLATEAU"
REGRESSION = "REGRESSION"


@dataclass
class EvalResult:
    timestamp: str
    h1:        float
    h10:       float
    mrr:       float
    n_questions: int
    dataset:   str
    duration_s: float
    params:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalEvent:
    kind:   str          # IMPROVED | PLATEAU | REGRESSION
    result: EvalResult
    delta:  float        # H@1 change vs previous


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SelfEvaluatorConfig:
    dataset:          str   = "metaqa"   # "metaqa" or "webqsp"
    n_questions:      int   = 100        # questions per mini-run
    interval_seconds: float = 600.0      # 10 minutes between evals
    plateau_window:   int   = 3          # consecutive stable evals = PLATEAU
    regression_threshold: float = 0.005  # H@1 drop that triggers REGRESSION
    embeddings:       str   = "sentence"
    eval_log:         Path  = EVAL_LOG
    best_params_path: Path  = BEST_PARAMS


# ── SelfEvaluator ─────────────────────────────────────────────────────────────

class SelfEvaluator:
    """
    Background daemon that runs mini-benchmarks and emits IMPROVED /
    PLATEAU / REGRESSION events to registered listeners.
    """

    def __init__(
        self,
        config: Optional[SelfEvaluatorConfig] = None,
        params_getter: Optional[Callable[[], dict]] = None,
    ) -> None:
        self._config        = config or SelfEvaluatorConfig()
        self._params_getter = params_getter   # callable → current CSA param dict

        self._listeners: list[Callable[[EvalEvent], None]] = []
        self._history:   list[EvalResult] = []
        self._best_h1:   float = 0.0
        self._plateau_count: int = 0

        self._running     = False
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock        = threading.Lock()

        self._config.eval_log.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_listener(self, fn: Callable[[EvalEvent], None]) -> None:
        self._listeners.append(fn)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name="self-evaluator", daemon=True
            )
            self._thread.start()
        logger.info("SelfEvaluator: started (dataset=%s, n=%d, interval=%.0fs).",
                    self._config.dataset, self._config.n_questions, self._config.interval_seconds)

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("SelfEvaluator: stopped.")

    def trigger(self) -> Optional[EvalResult]:
        """Run one evaluation synchronously (used by REST endpoint or tests)."""
        return self._run_eval()

    def history(self, n: int = 20) -> list[EvalResult]:
        with self._lock:
            return list(self._history[-n:])

    def status(self) -> dict:
        with self._lock:
            last = self._history[-1] if self._history else None
            return {
                "running":       self._running,
                "dataset":       self._config.dataset,
                "n_questions":   self._config.n_questions,
                "interval_s":    self._config.interval_seconds,
                "evals_run":     len(self._history),
                "best_h1":       self._best_h1,
                "plateau_count": self._plateau_count,
                "last_eval":     last.to_dict() if last else None,
            }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_eval()
            except Exception:
                logger.exception("SelfEvaluator: eval failed.")
            self._stop_event.wait(self._config.interval_seconds)

    def _run_eval(self) -> Optional[EvalResult]:
        cfg = self._config
        t0  = time.time()

        # Build subprocess command
        script = {
            "metaqa": "benchmarks/metaqa_eval.py",
            "webqsp": "benchmarks/webqsp_param_eval.py",
        }.get(cfg.dataset, "benchmarks/metaqa_eval.py")

        cmd = [
            sys.executable, "-u", script,
            "--sample",     str(cfg.n_questions),
            "--embeddings", cfg.embeddings,
            "--json-out",   "-",   # print JSON summary to stdout
        ]
        if cfg.dataset == "metaqa":
            cmd += ["--hop", "3"]

        logger.info("SelfEvaluator: running %d-question %s eval...", cfg.n_questions, cfg.dataset)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600,
                cwd=str(Path(__file__).parent.parent),
            )
        except subprocess.TimeoutExpired:
            logger.error("SelfEvaluator: eval timed out.")
            return None
        except Exception as exc:
            logger.error("SelfEvaluator: eval subprocess error: %s", exc)
            return None

        # Parse JSON summary from stdout
        result = self._parse_output(proc.stdout, cfg.dataset, cfg.n_questions, time.time() - t0)
        if result is None:
            logger.warning("SelfEvaluator: could not parse eval output.\nstderr: %s", proc.stderr[-500:])
            return None

        if self._params_getter:
            try:
                result.params = self._params_getter()
            except Exception:
                pass

        self._record(result)
        return result

    def _parse_output(self, stdout: str, dataset: str, n: int, duration: float) -> Optional[EvalResult]:
        """Extract H@1, H@10, MRR from eval script stdout.

        Eval scripts print lines like:
          H@1: 0.6029  H@10: 0.8778  MRR: 0.6990
        or a JSON blob when --json-out - is supported.
        """
        # Try JSON first
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and "h1" in line:
                try:
                    d = json.loads(line)
                    return EvalResult(
                        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        h1=float(d.get("h1", d.get("hits_at_1", 0))),
                        h10=float(d.get("h10", d.get("hits_at_10", 0))),
                        mrr=float(d.get("mrr", 0)),
                        n_questions=n,
                        dataset=dataset,
                        duration_s=round(duration, 1),
                    )
                except Exception:
                    pass

        # Fallback: parse printed summary line
        import re
        h1 = h10 = mrr = None
        for line in stdout.splitlines():
            m = re.search(r"H@1[:\s]+([0-9.]+)", line, re.I)
            if m:
                h1 = float(m.group(1))
                h1 = h1 / 100 if h1 > 1 else h1   # handle percentage vs fraction
            m = re.search(r"H@10[:\s]+([0-9.]+)", line, re.I)
            if m:
                h10 = float(m.group(1))
                h10 = h10 / 100 if h10 > 1 else h10
            m = re.search(r"MRR[:\s]+([0-9.]+)", line, re.I)
            if m:
                mrr = float(m.group(1))
                mrr = mrr / 100 if mrr > 1 else mrr
        if h1 is not None:
            return EvalResult(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                h1=h1, h10=h10 or 0.0, mrr=mrr or 0.0,
                n_questions=n, dataset=dataset, duration_s=round(duration, 1),
            )
        return None

    def _record(self, result: EvalResult) -> None:
        with self._lock:
            prev_h1 = self._history[-1].h1 if self._history else 0.0
            self._history.append(result)

        # Persist
        with self._config.eval_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        delta = result.h1 - prev_h1

        if result.h1 > self._best_h1 + 1e-6:
            self._best_h1 = result.h1
            self._plateau_count = 0
            self._snapshot_params(result)
            self._emit(EvalEvent(IMPROVED, result, delta))
            logger.info("SelfEvaluator: IMPROVED — H@1=%.4f (+%.4f)", result.h1, delta)

        elif abs(delta) < self._config.regression_threshold:
            self._plateau_count += 1
            if self._plateau_count >= self._config.plateau_window:
                self._emit(EvalEvent(PLATEAU, result, delta))
                logger.info("SelfEvaluator: PLATEAU — H@1=%.4f (%d stable evals)", result.h1, self._plateau_count)

        elif delta < -self._config.regression_threshold:
            self._plateau_count = 0
            self._emit(EvalEvent(REGRESSION, result, delta))
            logger.warning("SelfEvaluator: REGRESSION — H@1=%.4f (%.4f vs prev)", result.h1, delta)

    def _snapshot_params(self, result: EvalResult) -> None:
        if result.params:
            try:
                self._config.best_params_path.parent.mkdir(parents=True, exist_ok=True)
                self._config.best_params_path.write_text(
                    json.dumps({"h1": result.h1, "timestamp": result.timestamp, "params": result.params}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                logger.exception("SelfEvaluator: failed to snapshot params.")

    def _emit(self, event: EvalEvent) -> None:
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:
                logger.exception("SelfEvaluator: listener error.")

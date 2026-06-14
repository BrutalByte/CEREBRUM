"""
SensoryThalamus — Phase 300.

Sensory relay hub between the vision pipeline and CEREBRUM's knowledge graphs.
Receives PerceptualEvent objects, routes extracted CandidateTriples through
the appropriate vetting path, and materializes them into the perception_kb
domain of FederatedGraphRegistry.

Biological analogy: the thalamus relays and gates all sensory signals before
they reach cortical areas. Nothing enters the knowledge graphs unvetted.

Two ingestion paths:
  Fast path (tier-2): YOLO detections + InsightFace identities — curated model
    outputs with known confidence. Written directly via adapter.add_edge() +
    IngestionPipeline normalization. No HTTP calls, <1ms per triple.

  Slow path (tier-3): Florence-2 free-text captions — require corroboration.
    Routed through KnowledgeHarvester._vet() before materialization. May be
    rejected; rejection logged to benchmarks/rejected_knowledge.jsonl.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.perceptual_grounder import CandidateTriple as _CT, PerceptualEvent
from core.thalamus import IngestionPipeline

logger = logging.getLogger(__name__)


# ── Stats ──────────────────────────────────────────────────────────────────────

@dataclass
class ThalamusStats:
    events_received: int = 0
    events_dispatched: int = 0
    triples_produced: int = 0
    tier2_materialized: int = 0
    tier3_passed: int = 0
    tier3_rejected: int = 0
    perception_events_published: int = 0
    last_event_ts: float = 0.0


# ── SensoryThalamus ────────────────────────────────────────────────────────────

class SensoryThalamus:
    """
    Thread-safe sensory relay. Call start() once, then feed events via ingest().

    Parameters
    ----------
    adapter
        GraphAdapter for the perception_kb domain (NetworkXAdapter backed by
        a DiGraph is recommended — add_edge must accept relation, confidence,
        provenance keyword args).
    grounder
        PerceptualGrounder that converts PerceptualEvent → List[CandidateTriple].
    knowledge_harvester
        Optional KnowledgeHarvester for tier-3 slow-path vetting. If None,
        tier-3 triples are silently dropped (logged at DEBUG level).
    event_bus
        Optional MetaOrchestrator EventBus. If provided, publishes
        "PERCEPTION_EVENT" after each dispatched frame.
    ingestion_pipeline
        Optional IngestionPipeline for entity/relation normalization on the
        fast path. Defaults to a safe no-op pipeline (strip + uppercase).
    frame_skip
        Process 1-in-N events. When the camera runs at 30fps and CEREBRUM
        only needs a graph update every second, set frame_skip=30.
    max_queue
        Backpressure limit. Events beyond this are dropped with a warning.
    """

    def __init__(
        self,
        adapter: Any,
        grounder: Any,
        knowledge_harvester: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        ingestion_pipeline: Optional[IngestionPipeline] = None,
        frame_skip: int = 5,
        max_queue: int = 100,
    ) -> None:
        self._adapter    = adapter
        self._grounder   = grounder
        self._harvester  = knowledge_harvester
        self._bus        = event_bus
        self._pipeline   = ingestion_pipeline or IngestionPipeline()
        self._frame_skip = max(1, frame_skip)
        self._max_q      = max_queue

        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread: Optional[threading.Thread] = None
        self._stop_evt   = threading.Event()
        self._stats      = ThalamusStats()
        self._event_counter = 0  # for frame-skip tracking

        # Recent materializations ring buffer (last 200 triples)
        self._recent: List[Dict] = []
        self._recent_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="sensory-thalamus",
            daemon=True,
        )
        self._thread.start()
        logger.info("SensoryThalamus: started.")

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._q.put_nowait(None)  # unblock the dispatch loop
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SensoryThalamus: stopped. Stats: %s", self._stats)

    def ingest(self, event: PerceptualEvent) -> None:
        """Enqueue a PerceptualEvent for dispatch. Non-blocking."""
        self._stats.events_received += 1
        try:
            self._q.put_nowait(event)
        except queue.Full:
            logger.warning(
                "SensoryThalamus: queue full (%d), dropping frame %d.",
                self._max_q, event.frame_id,
            )

    def stats(self) -> dict:
        s = self._stats
        return {
            "events_received":           s.events_received,
            "events_dispatched":         s.events_dispatched,
            "triples_produced":          s.triples_produced,
            "tier2_materialized":        s.tier2_materialized,
            "tier3_passed":              s.tier3_passed,
            "tier3_rejected":            s.tier3_rejected,
            "perception_events_published": s.perception_events_published,
            "last_event_ts":             s.last_event_ts,
            "queue_depth":               self._q.qsize(),
        }

    def recent_triples(self, n: int = 50) -> List[Dict]:
        with self._recent_lock:
            return list(self._recent[-n:])

    # ── Dispatch loop ──────────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                event = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            if event is None:  # stop sentinel
                break

            self._event_counter += 1
            if self._event_counter % self._frame_skip != 0:
                continue  # skip frame

            try:
                self._dispatch(event)
            except Exception:
                logger.exception(
                    "SensoryThalamus: error dispatching frame %d.", event.frame_id
                )
            finally:
                self._q.task_done()

    def _dispatch(self, event: PerceptualEvent) -> None:
        self._stats.events_dispatched += 1
        self._stats.last_event_ts = event.ts

        triples = self._grounder.ground(event)
        self._stats.triples_produced += len(triples)

        for triple in triples:
            if triple.source_tier <= 2:
                self._fast_path(triple)
            else:
                self._slow_path(triple)

        # Publish to EventBus after each frame
        if self._bus is not None:
            try:
                self._bus.publish("PERCEPTION_EVENT", {
                    "frame_id":      event.frame_id,
                    "camera_id":     event.camera_id,
                    "triples_count": len(triples),
                    "ts":            event.ts,
                })
                self._stats.perception_events_published += 1
            except Exception:
                logger.debug("SensoryThalamus: EventBus publish failed.", exc_info=True)

    # ── Fast path (tier ≤ 2) ──────────────────────────────────────────────────

    def _fast_path(self, triple: _CT) -> None:
        try:
            edge = self._pipeline.process(
                source   = triple.source,
                target   = triple.target,
                relation = triple.relation,
                metadata = {
                    "confidence": triple.confidence,
                    "provenance": triple.source_url,
                    "weight":     triple.confidence,
                },
            )
            self._adapter.add_edge(
                edge.source,
                edge.target,
                relation   = edge.relation,
                confidence = edge.confidence,
                provenance = edge.provenance,
            )
            self._stats.tier2_materialized += 1
            self._record_recent(triple, "fast_path")
        except Exception:
            logger.debug("SensoryThalamus: fast_path error for %s.", triple.triple_id, exc_info=True)

    # ── Slow path (tier 3 — Florence-2 captions) ─────────────────────────────

    def _slow_path(self, triple: _CT) -> None:
        if self._harvester is None:
            logger.debug(
                "SensoryThalamus: no harvester configured; dropping tier-3 triple %s.",
                triple.triple_id,
            )
            self._stats.tier3_rejected += 1
            return

        try:
            passed, reason = self._harvester._vet(triple)
            if passed:
                self._harvester._materialize(triple)
                self._stats.tier3_passed += 1
                self._record_recent(triple, "slow_path")
            else:
                self._stats.tier3_rejected += 1
                logger.debug(
                    "SensoryThalamus: tier-3 triple %s rejected: %s",
                    triple.triple_id, reason,
                )
        except Exception:
            logger.debug("SensoryThalamus: slow_path error.", exc_info=True)
            self._stats.tier3_rejected += 1

    # ── Ring buffer ────────────────────────────────────────────────────────────

    def _record_recent(self, triple: _CT, path: str) -> None:
        record = {
            "triple_id": triple.triple_id,
            "source":    triple.source,
            "relation":  triple.relation,
            "target":    triple.target,
            "tier":      triple.source_tier,
            "confidence":triple.confidence,
            "path":      path,
            "ts":        time.time(),
        }
        with self._recent_lock:
            self._recent.append(record)
            if len(self._recent) > 200:
                self._recent = self._recent[-200:]

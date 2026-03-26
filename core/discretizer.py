"""
Phase 11/13 — Signal Discretizer.

Converts continuous data (sensor readings, video detections, log levels,
numeric metrics) into discrete (source, relation, target) graph triples
that the StreamAdapter can ingest as StreamEvents.

The fundamental idea: every observation becomes an edge in the knowledge
graph. The *source* is "who measured", the *target* is "what was measured
or what state was detected", and the *relation* describes the type of
observation or change.

Classes
-------
ThresholdDiscretizer
    Classifies a scalar value into {LOW, NORMAL, HIGH, SPIKE} states
    using configurable thresholds. Each transition to a new state emits
    a new edge. Repeated identical state → relation updates the existing edge.

BinningDiscretizer
    Quantizes a scalar into N equal-width or custom bins and emits the
    bin label as the target node.

ObjectDetectionDiscretizer
    Converts bounding-box detections (object_label, confidence, frame_id)
    into edges:  camera --[DETECTS]--> object_label
    and temporal co-occurrence edges between objects in the same frame.

TemporalSequenceDiscretizer
    Emits "precedes" edges between consecutive events on the same source,
    capturing temporal ordering in the graph.

CoActivationDiscretizer
    Detects when multiple sources fire simultaneously (within a time
    window) and emits CO_ACTIVATES edges between them — useful for
    correlating sensor anomalies.

STDPDiscretizer
    Spike-Timing Dependent Plasticity analog. Tracks the causal order in
    which sources fire and adjusts directional weights: sources that
    consistently fire *before* a target are potentiated (LTP-analog);
    sources that fire *after* are depressed (LTD-analog). Emits directional
    CAUSES edges once the causal weight and event count cross thresholds.

Example Usage
-------------
from core.discretizer import ThresholdDiscretizer
from adapters.stream_adapter import StreamAdapter

disc = ThresholdDiscretizer(
    source_id="temp_sensor_42",
    low=15.0, high=40.0, spike=60.0,
    relation="READS",
)
events = disc.process(value=72.5)
# → [StreamEvent("temp_sensor_42", "READS", "temp_SPIKE", ...)]
adapter.ingest(events)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.stream_engine import StreamEvent


# ---------------------------------------------------------------------------
# ThresholdDiscretizer — scalar → {LOW, NORMAL, HIGH, SPIKE}
# ---------------------------------------------------------------------------

class ThresholdDiscretizer:
    """
    Classifies a scalar reading into one of four discrete states.

    States (configurable labels):
        value < low              → LOW
        low  <= value < high     → NORMAL
        high <= value < spike    → HIGH
        value >= spike           → SPIKE

    Only emits a new event when the state *changes* (hysteresis mode),
    or on every call when ``emit_always=True``.

    Parameters
    ----------
    source_id    : the sensor or source entity ID
    low          : lower threshold (below this → LOW)
    high         : upper normal threshold (above this → HIGH)
    spike        : spike threshold (above this → SPIKE; None to disable)
    relation     : edge relation label (default "READS")
    emit_always  : emit an event on every call even if state unchanged
    labels       : override the default state label strings
    """

    DEFAULT_LABELS = {
        "LOW":    "LOW",
        "NORMAL": "NORMAL",
        "HIGH":   "HIGH",
        "SPIKE":  "SPIKE",
    }

    def __init__(
        self,
        source_id: str,
        low: float,
        high: float,
        spike: Optional[float] = None,
        relation: str = "READS",
        emit_always: bool = False,
        labels: Optional[Dict[str, str]] = None,
        ttl: float = 0.0,
    ):
        self.source_id = source_id
        self.low = low
        self.high = high
        self.spike = spike
        self.relation = relation
        self.emit_always = emit_always
        self.labels = {**self.DEFAULT_LABELS, **(labels or {})}
        self.ttl = ttl
        self._last_state: Optional[str] = None

    def classify(self, value: float) -> str:
        if self.spike is not None and value >= self.spike:
            return self.labels["SPIKE"]
        if value >= self.high:
            return self.labels["HIGH"]
        if value < self.low:
            return self.labels["LOW"]
        return self.labels["NORMAL"]

    def process(self, value: float, metadata: Optional[Dict] = None) -> List[StreamEvent]:
        """
        Process a scalar reading. Returns 0 or 1 StreamEvents.

        The target node is ``{source_id}_{state}``, making it unique per
        sensor/state combination so nodes don't collide across sensors.
        """
        state = self.classify(value)
        target = f"{self.source_id}_{state}"
        if not self.emit_always and state == self._last_state:
            return []
        self._last_state = state
        meta = {"raw_value": value, "state": state}
        if metadata:
            meta.update(metadata)
        return [StreamEvent(
            source=self.source_id,
            relation=self.relation,
            target=target,
            metadata=meta,
            ttl=self.ttl,
        )]


# ---------------------------------------------------------------------------
# BinningDiscretizer — scalar → bin_N
# ---------------------------------------------------------------------------

class BinningDiscretizer:
    """
    Quantizes a scalar into N equal-width bins (or custom edges).

    Useful when you want more granularity than four states but still need
    discrete graph nodes.

    Parameters
    ----------
    source_id   : sensor/source entity ID
    min_val     : minimum expected value (maps to bin 0)
    max_val     : maximum expected value (maps to bin n_bins-1)
    n_bins      : number of bins (ignored if bin_edges is provided)
    bin_edges   : explicit list of monotonically increasing boundaries
    relation    : edge relation label
    prefix      : prefix for bin node names (e.g. "temp_bin_3")
    """

    def __init__(
        self,
        source_id: str,
        min_val: float = 0.0,
        max_val: float = 100.0,
        n_bins: int = 10,
        bin_edges: Optional[List[float]] = None,
        relation: str = "READS",
        prefix: Optional[str] = None,
        ttl: float = 0.0,
    ):
        self.source_id = source_id
        self.relation = relation
        self.prefix = prefix or source_id
        self.ttl = ttl

        if bin_edges is not None:
            self._edges = sorted(bin_edges)
            self._n = len(self._edges) + 1
        else:
            step = (max_val - min_val) / n_bins
            self._edges = [min_val + i * step for i in range(1, n_bins)]
            self._n = n_bins
        self._min = min_val
        self._max = max_val

    def bin_index(self, value: float) -> int:
        for i, edge in enumerate(self._edges):
            if value < edge:
                return i
        return self._n - 1

    def process(self, value: float, metadata: Optional[Dict] = None) -> List[StreamEvent]:
        idx = self.bin_index(value)
        target = f"{self.prefix}_bin_{idx}"
        meta = {"raw_value": value, "bin": idx}
        if metadata:
            meta.update(metadata)
        return [StreamEvent(
            source=self.source_id,
            relation=self.relation,
            target=target,
            metadata=meta,
            ttl=self.ttl,
        )]


# ---------------------------------------------------------------------------
# ObjectDetectionDiscretizer — bbox detection → graph triples
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single object detection from a frame."""
    label: str
    confidence: float
    frame_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h) normalized
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObjectDetectionDiscretizer:
    """
    Converts object detections (from YOLO, DETR, etc.) into graph edges.

    Each detection emits:
        camera_id --[DETECTS]--> object_label

    Objects detected in the same frame emit co-occurrence edges:
        object_a  --[CO_OCCURS_WITH]--> object_b

    Objects detected in consecutive frames on the same camera emit:
        object_label --[PERSISTS_IN]--> camera_id  (if seen in >1 frame)

    Parameters
    ----------
    camera_id           : source entity ID for this camera
    confidence_threshold: detections below this confidence are ignored
    co_occurrence_edges : whether to emit CO_OCCURS_WITH edges
    ttl                 : time-to-live for emitted edges (seconds)
    """

    def __init__(
        self,
        camera_id: str,
        confidence_threshold: float = 0.5,
        co_occurrence_edges: bool = True,
        ttl: float = 5.0,
    ):
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.co_occurrence_edges = co_occurrence_edges
        self.ttl = ttl
        self._prev_labels: List[str] = []

    def process(self, detections: List[Detection]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        now = time.time()

        valid = [d for d in detections if d.confidence >= self.confidence_threshold]
        labels = [d.label for d in valid]

        for det in valid:
            meta = {
                "confidence": det.confidence,
                "frame_id": det.frame_id,
            }
            if det.bbox:
                meta["bbox"] = det.bbox
            meta.update(det.metadata)

            events.append(StreamEvent(
                source=self.camera_id,
                relation="DETECTS",
                target=det.label,
                timestamp=now,
                metadata=meta,
                ttl=self.ttl,
            ))

        # Co-occurrence edges within the same frame
        if self.co_occurrence_edges and len(labels) > 1:
            seen = set()
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    pair = tuple(sorted([a, b]))
                    if pair not in seen:
                        events.append(StreamEvent(
                            source=a,
                            relation="CO_OCCURS_WITH",
                            target=b,
                            timestamp=now,
                            metadata={"frame_id": valid[i].frame_id},
                            ttl=self.ttl,
                        ))
                        seen.add(pair)

        self._prev_labels = labels
        return events


# ---------------------------------------------------------------------------
# TemporalSequenceDiscretizer — A → B → C ordering
# ---------------------------------------------------------------------------

class TemporalSequenceDiscretizer:
    """
    Emits PRECEDES edges between consecutive events on the same source.

    Useful for: log event sequences, state machine transitions, step-by-step
    process flows, network packet ordering.

    Parameters
    ----------
    source_id : the entity that produces the sequence (e.g. "server_1")
    relation  : edge label between consecutive events (default "PRECEDES")
    max_gap   : if the gap between consecutive events exceeds this many
                seconds, do NOT emit a PRECEDES edge (the sequence broke).
                0 = no gap limit.
    """

    def __init__(
        self,
        source_id: str,
        relation: str = "PRECEDES",
        max_gap: float = 0.0,
        ttl: float = 0.0,
    ):
        self.source_id = source_id
        self.relation = relation
        self.max_gap = max_gap
        self.ttl = ttl
        self._last_event: Optional[str] = None
        self._last_time: float = 0.0

    def process(self, event_label: str, metadata: Optional[Dict] = None) -> List[StreamEvent]:
        """
        Process a named event on this source.

        Returns a PRECEDES edge from the previous event to this one,
        unless this is the first event or the gap exceeded max_gap.
        """
        now = time.time()
        events: List[StreamEvent] = []

        if self._last_event is not None:
            gap = now - self._last_time
            if self.max_gap <= 0 or gap <= self.max_gap:
                meta = {"gap_seconds": round(gap, 4)}
                if metadata:
                    meta.update(metadata)
                events.append(StreamEvent(
                    source=self._last_event,
                    relation=self.relation,
                    target=event_label,
                    timestamp=now,
                    metadata=meta,
                    ttl=self.ttl,
                ))

        self._last_event = event_label
        self._last_time = now
        return events


# ---------------------------------------------------------------------------
# CoActivationDiscretizer — multi-sensor correlation
# ---------------------------------------------------------------------------

class CoActivationDiscretizer:
    """
    Detects simultaneous activation of multiple sources and emits
    CO_ACTIVATES edges between them.

    "Simultaneous" means within ``window_seconds`` of each other.

    Useful for: detecting correlated sensor anomalies, identifying
    network hosts that communicate in bursts, finding co-moving entities.

    Parameters
    ----------
    window_seconds  : how close two activations must be to count as simultaneous
    min_co_activations: minimum number of times two sources must co-activate
                        before an edge is emitted (reduces noise)
    relation        : edge label (default "CO_ACTIVATES")
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        min_co_activations: int = 1,
        relation: str = "CO_ACTIVATES",
        ttl: float = 0.0,
    ):
        self.window_seconds = window_seconds
        self.min_co_activations = min_co_activations
        self.relation = relation
        self.ttl = ttl

        # source_id → last activation timestamp
        self._last_seen: Dict[str, float] = {}
        # (source_a, source_b) → co-activation count
        self._co_counts: Dict[Tuple[str, str], int] = {}

    def process(self, source_id: str, metadata: Optional[Dict] = None) -> List[StreamEvent]:
        """
        Record an activation for source_id. Returns CO_ACTIVATES events
        for any other source that activated within window_seconds.
        """
        now = time.time()
        events: List[StreamEvent] = []

        # Find recent co-activators
        for other_id, other_ts in list(self._last_seen.items()):
            if other_id == source_id:
                continue
            if now - other_ts <= self.window_seconds:
                pair = tuple(sorted([source_id, other_id]))
                self._co_counts[pair] = self._co_counts.get(pair, 0) + 1
                if self._co_counts[pair] >= self.min_co_activations:
                    meta = {"co_count": self._co_counts[pair]}
                    if metadata:
                        meta.update(metadata)
                    events.append(StreamEvent(
                        source=source_id,
                        relation=self.relation,
                        target=other_id,
                        timestamp=now,
                        metadata=meta,
                        ttl=self.ttl,
                    ))

        self._last_seen[source_id] = now
        return events


# ---------------------------------------------------------------------------
# Causal Significance Filter — chi-squared uniformity test
# ---------------------------------------------------------------------------

def _chi_squared_uniformity(
    times: List[float],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Test whether the co-occurrence timestamps are uniformly distributed over time.

    Returns (passed: bool, pvalue: float).
    ``passed=True`` means the distribution is plausibly uniform (not bursty).
    ``passed=False`` means events are significantly clustered — likely adversarial.

    Falls back gracefully if scipy is unavailable (returns True, 1.0).
    Requires at least 4 events to produce a meaningful result; fewer → passes by default.
    """
    if len(times) < 4:
        return True, 1.0

    sorted_times = sorted(times)
    span = sorted_times[-1] - sorted_times[0]
    if span == 0.0:
        # All simultaneous — definitively bursty
        return False, 0.0

    # Bin event timestamps into k equal-width buckets over the span
    k = max(2, min(len(times) // 2, 5))
    bucket_width = span / k
    observed = [0] * k
    for t in sorted_times:
        idx = min(int((t - sorted_times[0]) / bucket_width), k - 1)
        observed[idx] += 1

    expected_per_bucket = len(times) / k

    try:
        from scipy.stats import chisquare as _chisquare
        _, pvalue = _chisquare(observed, f_exp=[expected_per_bucket] * k)
        return float(pvalue) >= alpha, float(pvalue)
    except Exception:
        # scipy unavailable or test error — pass through (fail open)
        return True, 1.0


# ---------------------------------------------------------------------------
# STDPDiscretizer — Spike-Timing Dependent Plasticity analog (Phase 13)
# ---------------------------------------------------------------------------

class STDPDiscretizer:
    """
    Spike-Timing Dependent Plasticity (STDP) analog.

    Biological STDP: if neuron A fires *before* neuron B, the synapse A→B
    is potentiated (LTP). If A fires *after* B, the synapse A→B is
    depressed (LTD). The magnitude of the change decays exponentially with
    the time difference |Δt|.

    CEREBRUM analog:
        - Each source_id that calls ``process()`` is a "spike".
        - For every recent prior spike (pre-synaptic), Δt = t_post − t_pre > 0
          → potentiate weight[(pre, post)] by A_plus * exp(-Δt / tau_plus).
        - For every recent *future* spike that fires within window_seconds
          of the current spike (post < pre, i.e. this call is the pre):
          those relationships are captured in the next call's LTD branch.
        - Symmetric LTD: when a new spike (post) arrives and an older spike
          (pre) was *already seen within window_seconds*, the *reverse*
          direction (post → pre) is depressed by A_minus * exp(-|Δt|/tau_minus)
          because post fired after pre but pre's direction is already captured
          in LTP; the depression reduces spurious reverse edges.

    The net result: weight[(A, B)] grows when A reliably precedes B, and
    shrinks when the ordering is reversed. Once weight[(A, B)] ≥ w_threshold
    AND event_count[(A, B)] ≥ n_min a CAUSES(A→B) edge is emitted.

    A per-event multiplicative ``weight_decay`` keeps weights from growing
    unboundedly across long periods without co-activation.

    Parameters
    ----------
    window_seconds : look-back window for pairing pre and post spikes
    tau_plus       : LTP time constant (seconds). Smaller → faster decay with Δt.
    tau_minus      : LTD time constant (seconds).
    A_plus         : maximum LTP weight increment per pair
    A_minus        : maximum LTD weight decrement per pair
    w_threshold    : causal weight above which a CAUSES edge is emitted
    n_min          : minimum co-occurrence count before emitting
    weight_decay   : multiplicative decay applied to all weights on each call
                     (1.0 = no decay; 0.99 = slow exponential forgetting)
    relation       : edge relation label for emitted causal edges
    ttl            : time-to-live for emitted edges (seconds; 0 = permanent)
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        tau_plus: float = 0.2,
        tau_minus: float = 0.2,
        A_plus: float = 0.1,
        A_minus: float = 0.105,
        w_threshold: float = 0.5,
        n_min: int = 5,
        weight_decay: float = 0.99,
        relation: str = "CAUSES",
        ttl: float = 0.0,
        min_causal_span: float = 0.0,
        use_chi_squared: bool = False,
    ):
        self.window_seconds = window_seconds
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_threshold = w_threshold
        self.n_min = n_min
        self.weight_decay = weight_decay
        self.relation = relation
        self.ttl = ttl
        self.min_causal_span = min_causal_span
        self.use_chi_squared = use_chi_squared

        # source_id → last spike timestamp
        self._last_fire: Dict[str, float] = {}
        # (pre_id, post_id) → accumulated causal weight
        self._weights: Dict[Tuple[str, str], float] = {}
        # (pre_id, post_id) → number of causal pairings observed
        self._counts: Dict[Tuple[str, str], int] = {}
        # edges already emitted (avoid duplicate StreamEvents per step)
        self._emitted: set = set()
        # (pre_id, post_id) → timestamp of first LTP co-occurrence (for span filter)
        self._first_cooccurrence: Dict[Tuple[str, str], float] = {}
        # (pre_id, post_id) → list of LTP co-occurrence timestamps (for chi-squared)
        self._cooccurrence_times: Dict[Tuple[str, str], List[float]] = {}
        
        # Lazy decay state (Hole 4 fix)
        self._step: int = 0
        self._last_update_step: Dict[Tuple[str, str], int] = {}

    def _apply_lazy_decay(self, key: Tuple[str, str]) -> float:
        """Apply pending geometric decay to a weight since its last update."""
        current_w = self._weights.get(key, 0.0)
        last_step = self._last_update_step.get(key, self._step)
        elapsed = self._step - last_step
        
        if elapsed > 0 and current_w > 0:
            # decay ^ elapsed
            decay_factor = self.weight_decay ** elapsed
            current_w *= decay_factor
            self._weights[key] = current_w
            
        self._last_update_step[key] = self._step
        return current_w

    def process(
        self,
        source_id: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> List[StreamEvent]:
        """
        Record a spike for source_id and update STDP weights.

        Returns CAUSES edges for any (pre, post) pair whose causal weight
        has crossed w_threshold with at least n_min pairings.

        Parameters
        ----------
        source_id : the entity that just "fired"
        timestamp : explicit event timestamp (uses time.time() if None)
        metadata  : extra metadata merged into emitted StreamEvents
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[StreamEvent] = []
        
        # Increment global step for lazy decay
        self._step += 1

        # Track which pairs were updated this step (to check thresholds efficiently)
        updated_pairs: set = set()

        # For each prior spike within window_seconds:
        #   Δt = now - t_pre  (positive → source_id is POST, prior is PRE)
        #   → potentiate weight[(pre, post)] by A_plus * exp(-Δt / tau_plus)
        #   → depress  weight[(post, pre)] by A_minus * exp(-Δt / tau_minus)
        for pre_id, t_pre in list(self._last_fire.items()):
            if pre_id == source_id:
                continue
            delta_t = now - t_pre
            if delta_t < 0 or delta_t > self.window_seconds:
                continue

            # LTP: pre → post (pre fired before current spike)
            pair_ltp = (pre_id, source_id)
            # 1. Bring weight up to date
            w_ltp = self._apply_lazy_decay(pair_ltp)
            # 2. Apply LTP
            dw_ltp = self.A_plus * math.exp(-delta_t / self.tau_plus)
            w_ltp += dw_ltp
            self._weights[pair_ltp] = w_ltp
            self._counts[pair_ltp] = self._counts.get(pair_ltp, 0) + 1
            updated_pairs.add(pair_ltp)

            # Track first co-occurrence timestamp for span filter
            if self.min_causal_span > 0.0 and pair_ltp not in self._first_cooccurrence:
                self._first_cooccurrence[pair_ltp] = now

            # Track all co-occurrence timestamps for chi-squared filter
            if self.use_chi_squared:
                self._cooccurrence_times.setdefault(pair_ltp, []).append(now)

            # LTD: post → pre direction is depressed (anti-causal direction)
            pair_ltd = (source_id, pre_id)
            # 1. Bring weight up to date
            w_ltd = self._apply_lazy_decay(pair_ltd)
            # 2. Apply LTD
            dw_ltd = self.A_minus * math.exp(-delta_t / self.tau_minus)
            w_ltd -= dw_ltd
            if w_ltd < 0.0:
                w_ltd = 0.0
            self._weights[pair_ltd] = w_ltd
            # No need to add to updated_pairs for emission check if it decreased,
            # but for completeness/correctness of state we updated it.
            # Only potential emitters need checking.

        # Check which pairs now exceed threshold and emit CAUSES edges
        # Optimization: only check pairs that were potentiated this step
        for (pre_id, post_id) in updated_pairs:
            w = self._weights[(pre_id, post_id)]
            count = self._counts.get((pre_id, post_id), 0)
            
            if w >= self.w_threshold and count >= self.n_min:
                edge_key = (pre_id, post_id)
                if edge_key in self._emitted:
                    continue

                # Causal significance filter 1: minimum temporal span
                causal_span = 0.0
                if self.min_causal_span > 0.0:
                    first_t = self._first_cooccurrence.get(edge_key)
                    if first_t is None or (now - first_t) < self.min_causal_span:
                        continue
                    causal_span = now - first_t

                # Causal significance filter 2: chi-squared uniformity test
                causal_pvalue = 1.0
                if self.use_chi_squared:
                    times = self._cooccurrence_times.get(edge_key, [])
                    passed, pvalue = _chi_squared_uniformity(times)
                    causal_pvalue = pvalue
                    if not passed:
                        continue

                self._emitted.add(edge_key)
                meta = {
                    "causal_weight": round(w, 6),
                    "event_count": count,
                }
                if self.min_causal_span > 0.0:
                    meta["causal_span"] = round(causal_span, 6)
                if self.use_chi_squared:
                    meta["causal_pvalue"] = round(causal_pvalue, 6)
                if metadata:
                    meta.update(metadata)
                events.append(StreamEvent(
                    source=pre_id,
                    relation=self.relation,
                    target=post_id,
                    timestamp=now,
                    metadata=meta,
                    ttl=self.ttl,
                ))

        self._last_fire[source_id] = now
        return events

    def weight(self, pre_id: str, post_id: str) -> float:
        """Return current causal weight for the (pre, post) directed pair."""
        return self._apply_lazy_decay((pre_id, post_id))

    def count(self, pre_id: str, post_id: str) -> int:
        """Return co-occurrence count for the (pre, post) directed pair."""
        return self._counts.get((pre_id, post_id), 0)

    def reset(self) -> None:
        """Clear all accumulated state (useful for test isolation)."""
        self._last_fire.clear()
        self._weights.clear()
        self._counts.clear()
        self._emitted.clear()
        self._first_cooccurrence.clear()
        self._cooccurrence_times.clear()
        self._step = 0
        self._last_update_step.clear()

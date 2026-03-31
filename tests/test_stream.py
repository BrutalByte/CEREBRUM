"""
Phase 11 — Streaming tests.

Tests: StreamEvent, SlidingWindowBuffer, IncrementalCommunityUpdater,
       ThresholdDiscretizer, BinningDiscretizer, ObjectDetectionDiscretizer,
       TemporalSequenceDiscretizer, CoActivationDiscretizer,
       StreamAdapter (ingest, eviction, thread safety, community update).
"""
import time
import threading

from core.stream_engine import StreamEvent, SlidingWindowBuffer, IncrementalCommunityUpdater, StreamStats
from core.discretizer import (
    ThresholdDiscretizer, BinningDiscretizer,
    ObjectDetectionDiscretizer, Detection,
    TemporalSequenceDiscretizer, CoActivationDiscretizer,
)
from adapters.stream_adapter import StreamAdapter, PythonCallbackSource


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

class TestStreamEvent:
    def test_defaults(self):
        ev = StreamEvent(source="a", relation="R", target="b")
        assert ev.source == "a"
        assert ev.relation == "R"
        assert ev.target == "b"
        assert ev.timestamp > 0
        assert ev.ttl == 0.0

    def test_edge_key(self):
        ev = StreamEvent(source="a", relation="R", target="b")
        assert ev.edge_key() == ("a", "R", "b")

    def test_custom_timestamp(self):
        ev = StreamEvent(source="x", relation="Y", target="z", timestamp=1000.0)
        assert ev.timestamp == 1000.0

    def test_metadata_stored(self):
        ev = StreamEvent(source="s", relation="R", target="t", metadata={"value": 42.0})
        assert ev.metadata["value"] == 42.0


# ---------------------------------------------------------------------------
# SlidingWindowBuffer
# ---------------------------------------------------------------------------

class TestSlidingWindowBuffer:
    def test_push_and_len(self):
        buf = SlidingWindowBuffer(time_window_seconds=60, max_edges=100)
        ev = StreamEvent(source="a", relation="R", target="b")
        buf.push(ev)
        assert len(buf) == 1

    def test_live_edges_after_push(self):
        buf = SlidingWindowBuffer(time_window_seconds=60, max_edges=100)
        ev = StreamEvent(source="a", relation="R", target="b")
        buf.push(ev)
        assert ("a", "R", "b") in buf.live_edges()

    def test_time_eviction(self):
        buf = SlidingWindowBuffer(time_window_seconds=0.05, max_edges=100)
        ev = StreamEvent(source="a", relation="R", target="b",
                         timestamp=time.time() - 1.0)  # already stale
        buf._queue.append(ev)
        buf._edge_refs[ev.edge_key()] = buf._edge_refs.get(ev.edge_key(), 0) + 1
        evicted = buf._evict_stale(time.time())
        assert len(evicted) == 1
        assert ("a", "R", "b") not in buf.live_edges()

    def test_max_edges_cap(self):
        buf = SlidingWindowBuffer(time_window_seconds=3600, max_edges=3)
        for i in range(5):
            buf.push(StreamEvent(source=f"s{i}", relation="R", target=f"t{i}"))
        assert len(buf) == 3

    def test_permanent_ttl_not_evicted(self):
        buf = SlidingWindowBuffer(time_window_seconds=0.01, max_edges=100)
        ev = StreamEvent(source="a", relation="R", target="b",
                         timestamp=time.time() - 10.0, ttl=-1)
        buf._queue.append(ev)
        buf._edge_refs[ev.edge_key()] = 1
        evicted = buf._evict_stale(time.time())
        assert len(evicted) == 0


# ---------------------------------------------------------------------------
# IncrementalCommunityUpdater
# ---------------------------------------------------------------------------

class TestIncrementalCommunityUpdater:
    def _small_graph(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("x", "y")])
        return G

    def test_mark_and_should_update(self):
        upd = IncrementalCommunityUpdater(min_events_before_update=3)
        assert not upd.should_update()
        upd.mark_affected(["a", "b"])
        upd.mark_affected(["c"])
        assert not upd.should_update()
        upd.mark_affected(["d"])
        assert upd.should_update()

    def test_run_returns_community_map(self):
        G = self._small_graph()
        upd = IncrementalCommunityUpdater(min_events_before_update=1)
        community_map = {n: 0 for n in G.nodes()}
        upd.mark_affected(["a", "b"])
        result = upd.run(G, community_map)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(G.nodes())

    def test_run_full_returns_full_map(self):
        G = self._small_graph()
        upd = IncrementalCommunityUpdater()
        result = upd.run_full(G)
        assert isinstance(result, dict)
        assert all(n in result for n in G.nodes())

    def test_empty_pending_no_crash(self):
        G = self._small_graph()
        upd = IncrementalCommunityUpdater(min_events_before_update=1)
        community_map = {}
        upd.mark_affected([])
        upd._events_since_update = 1
        upd._pending_nodes = set()
        result = upd.run(G, community_map)
        assert result == {}


# ---------------------------------------------------------------------------
# ThresholdDiscretizer
# ---------------------------------------------------------------------------

class TestThresholdDiscretizer:
    def make(self):
        return ThresholdDiscretizer("sensor_1", low=10.0, high=30.0, spike=50.0)

    def test_low_state(self):
        d = self.make()
        events = d.process(5.0)
        assert len(events) == 1
        assert events[0].target == "sensor_1_LOW"

    def test_normal_state(self):
        d = self.make()
        events = d.process(20.0)
        assert events[0].target == "sensor_1_NORMAL"

    def test_high_state(self):
        d = self.make()
        events = d.process(40.0)
        assert events[0].target == "sensor_1_HIGH"

    def test_spike_state(self):
        d = self.make()
        events = d.process(60.0)
        assert events[0].target == "sensor_1_SPIKE"

    def test_no_event_same_state(self):
        d = self.make()
        d.process(20.0)   # NORMAL
        events = d.process(22.0)  # still NORMAL — no emit
        assert events == []

    def test_emit_always(self):
        d = ThresholdDiscretizer("s", low=10, high=30, emit_always=True)
        d.process(20.0)
        events = d.process(22.0)  # same state but emit_always
        assert len(events) == 1

    def test_metadata_included(self):
        d = self.make()
        events = d.process(5.0)
        assert "raw_value" in events[0].metadata
        assert events[0].metadata["raw_value"] == 5.0

    def test_source_id(self):
        d = self.make()
        events = d.process(5.0)
        assert events[0].source == "sensor_1"

    def test_relation(self):
        d = ThresholdDiscretizer("s", low=10, high=30, relation="MEASURES")
        events = d.process(5.0)
        assert events[0].relation == "MEASURES"


# ---------------------------------------------------------------------------
# BinningDiscretizer
# ---------------------------------------------------------------------------

class TestBinningDiscretizer:
    def test_basic_binning(self):
        d = BinningDiscretizer("cpu", min_val=0.0, max_val=100.0, n_bins=5)
        events = d.process(55.0)
        assert len(events) == 1
        assert events[0].target.startswith("cpu_bin_")

    def test_bin_boundaries(self):
        d = BinningDiscretizer("x", min_val=0, max_val=10, n_bins=2)
        ev_low = d.process(3.0)
        ev_high = d.process(7.0)
        assert ev_low[0].target != ev_high[0].target

    def test_custom_bin_edges(self):
        d = BinningDiscretizer("y", bin_edges=[10.0, 20.0, 30.0])
        ev = d.process(15.0)
        assert "_bin_1" in ev[0].target

    def test_below_min(self):
        d = BinningDiscretizer("z", min_val=0, max_val=100, n_bins=5)
        ev = d.process(-5.0)
        assert "_bin_0" in ev[0].target

    def test_above_max(self):
        d = BinningDiscretizer("z", min_val=0, max_val=100, n_bins=5)
        ev = d.process(150.0)
        assert "_bin_4" in ev[0].target


# ---------------------------------------------------------------------------
# ObjectDetectionDiscretizer
# ---------------------------------------------------------------------------

class TestObjectDetectionDiscretizer:
    def test_single_detection(self):
        d = ObjectDetectionDiscretizer("cam_1", confidence_threshold=0.5)
        dets = [Detection(label="person", confidence=0.9, frame_id="f1")]
        events = d.process(dets)
        # At least one DETECTS edge
        detects = [e for e in events if e.relation == "DETECTS"]
        assert len(detects) == 1
        assert detects[0].source == "cam_1"
        assert detects[0].target == "person"

    def test_below_threshold_filtered(self):
        d = ObjectDetectionDiscretizer("cam", confidence_threshold=0.8)
        dets = [Detection(label="car", confidence=0.4, frame_id="f1")]
        events = d.process(dets)
        assert events == []

    def test_co_occurrence_edges(self):
        d = ObjectDetectionDiscretizer("cam", co_occurrence_edges=True)
        dets = [
            Detection(label="person", confidence=0.9, frame_id="f1"),
            Detection(label="car",    confidence=0.9, frame_id="f1"),
        ]
        events = d.process(dets)
        co = [e for e in events if e.relation == "CO_OCCURS_WITH"]
        assert len(co) == 1

    def test_no_co_occurrence_when_disabled(self):
        d = ObjectDetectionDiscretizer("cam", co_occurrence_edges=False)
        dets = [
            Detection(label="a", confidence=0.9, frame_id="f"),
            Detection(label="b", confidence=0.9, frame_id="f"),
        ]
        events = d.process(dets)
        co = [e for e in events if e.relation == "CO_OCCURS_WITH"]
        assert co == []


# ---------------------------------------------------------------------------
# TemporalSequenceDiscretizer
# ---------------------------------------------------------------------------

class TestTemporalSequenceDiscretizer:
    def test_first_event_no_edge(self):
        d = TemporalSequenceDiscretizer("server")
        events = d.process("login")
        assert events == []

    def test_second_event_creates_edge(self):
        d = TemporalSequenceDiscretizer("server")
        d.process("login")
        events = d.process("query")
        assert len(events) == 1
        assert events[0].source == "login"
        assert events[0].target == "query"
        assert events[0].relation == "PRECEDES"

    def test_max_gap_blocks_edge(self):
        d = TemporalSequenceDiscretizer("s", max_gap=0.01)
        d.process("a")
        time.sleep(0.05)
        events = d.process("b")
        assert events == []

    def test_gap_metadata(self):
        d = TemporalSequenceDiscretizer("s")
        d.process("a")
        events = d.process("b")
        assert "gap_seconds" in events[0].metadata


# ---------------------------------------------------------------------------
# CoActivationDiscretizer
# ---------------------------------------------------------------------------

class TestCoActivationDiscretizer:
    def test_no_co_activation_alone(self):
        d = CoActivationDiscretizer(window_seconds=1.0)
        events = d.process("sensor_a")
        assert events == []

    def test_co_activation_detected(self):
        d = CoActivationDiscretizer(window_seconds=1.0)
        d.process("sensor_a")
        events = d.process("sensor_b")
        co = [e for e in events if e.relation == "CO_ACTIVATES"]
        assert len(co) == 1

    def test_min_co_activations_filter(self):
        # Each process() call from either side increments the shared co-count.
        # With min_co_activations=4 the sequence is:
        #   process(a) → 0 (no others seen yet)
        #   process(b) → count=1, no emit
        #   process(a) → count=2, no emit
        #   process(b) → count=3, no emit
        #   process(a) → count=4, emit (a sees b)
        d = CoActivationDiscretizer(window_seconds=1.0, min_co_activations=4)
        d.process("a")
        events1 = d.process("b")  # count=1
        assert events1 == []
        events2 = d.process("a")  # count=2
        assert events2 == []
        events3 = d.process("b")  # count=3
        assert events3 == []
        events4 = d.process("a")  # count=4 → emit
        assert len([e for e in events4 if e.relation == "CO_ACTIVATES"]) == 1

    def test_outside_window_no_edge(self):
        d = CoActivationDiscretizer(window_seconds=0.01)
        d.process("a")
        time.sleep(0.05)
        events = d.process("b")
        co = [e for e in events if e.relation == "CO_ACTIVATES"]
        assert co == []


# ---------------------------------------------------------------------------
# StreamAdapter
# ---------------------------------------------------------------------------

class TestStreamAdapter:
    def make(self):
        return StreamAdapter(time_window_seconds=60, max_edges=1000, directed=True)

    def test_ingest_adds_edge(self):
        adapter = self.make()
        ev = StreamEvent(source="a", relation="R", target="b")
        adapter.ingest(ev)
        G = adapter.to_networkx()
        assert G.has_edge("a", "b")

    def test_ingest_adds_nodes(self):
        adapter = self.make()
        adapter.ingest(StreamEvent(source="x", relation="T", target="y"))
        G = adapter.to_networkx()
        assert "x" in G.nodes()
        assert "y" in G.nodes()

    def test_ingest_batch(self):
        adapter = self.make()
        events = [
            StreamEvent(source=f"s{i}", relation="R", target=f"t{i}")
            for i in range(5)
        ]
        adapter.ingest_batch(events)
        assert adapter.node_count() == 10

    def test_stale_edge_evicted(self):
        adapter = StreamAdapter(time_window_seconds=0.01, max_edges=1000)
        ev = StreamEvent(source="a", relation="R", target="b",
                         timestamp=time.time() - 10.0)
        # Directly push to buffer as stale
        adapter._buffer._queue.append(ev)
        adapter._buffer._edge_refs[ev.edge_key()] = 1
        adapter._G.add_edge("a", "b", relation="R")
        evicted = adapter._buffer._evict_stale(time.time())
        assert len(evicted) == 1

    def test_find_entities_after_ingest(self):
        adapter = self.make()
        adapter.ingest(StreamEvent(source="newton", relation="INFLUENCED", target="einstein"))
        results = adapter.find_entities("newton")
        assert any(e.id == "newton" for e in results)

    def test_thread_safety(self):
        """Multiple threads ingesting simultaneously should not corrupt the graph."""
        adapter = self.make()
        errors = []

        def ingest_loop(thread_id):
            try:
                for i in range(20):
                    adapter.ingest(StreamEvent(
                        source=f"t{thread_id}_s{i}",
                        relation="R",
                        target=f"t{thread_id}_t{i}",
                    ))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=ingest_loop, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        assert adapter.node_count() > 0

    def test_mutation_listener_fired(self):
        adapter = self.make()
        received = []
        adapter.add_mutation_listener(lambda action, ev: received.append((action, ev.source)))
        adapter.ingest(StreamEvent(source="alpha", relation="R", target="beta"))
        assert any(r[0] == "add" and r[1] == "alpha" for r in received)

    def test_live_stats(self):
        adapter = self.make()
        adapter.ingest(StreamEvent(source="a", relation="R", target="b"))
        stats = adapter.live_stats()
        assert stats["nodes"] >= 2
        assert stats["edges"] >= 1
        assert "events_per_second" in stats

    def test_python_callback_source(self):
        """PythonCallbackSource delivers events to the adapter."""
        adapter = self.make()
        counter = {"n": 0}

        def cb():
            if counter["n"] < 3:
                ev = StreamEvent(source="src", relation="R", target=f"tgt_{counter['n']}")
                counter["n"] += 1
                return ev
            return None

        source = PythonCallbackSource(cb, poll_interval=0.01)
        adapter.add_source(source)
        adapter.start()
        time.sleep(0.2)
        adapter.stop()

        assert adapter.node_count() > 0

    def test_force_community_update(self):
        adapter = self.make()
        for i in range(6):
            adapter.ingest(StreamEvent(source=f"a{i}", relation="R", target=f"b{i}"))
        adapter.force_community_update()
        assert isinstance(adapter.community_map, dict)

    def test_stream_stats_track_ingestion(self):
        adapter = self.make()
        for i in range(5):
            adapter.ingest(StreamEvent(source="x", relation="R", target=f"y{i}"))
        assert adapter.stats.total_ingested == 5


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------

class TestStreamStats:
    def test_events_per_second(self):
        stats = StreamStats(window_seconds=5.0)
        for _ in range(10):
            stats.record_event()
        # Should be > 0 and within a plausible range
        eps = stats.events_per_second
        assert eps > 0

    def test_totals(self):
        stats = StreamStats()
        stats.record_event()
        stats.record_event()
        stats.record_eviction(3)
        stats.record_community_update()
        d = stats.to_dict()
        assert d["total_ingested"] == 2
        assert d["total_evicted"] == 3
        assert d["total_community_updates"] == 1

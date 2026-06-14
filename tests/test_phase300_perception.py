"""
Phase 300 — Perceptual Grounding Tests.

All tests run fully offline: no real camera, no GPU, no model downloads.
Tests validate the PerceptualGrounder triple-extraction logic, the
SensoryThalamus routing paths, and end-to-end offline pipeline.
"""
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from adapters.networkx_adapter import NetworkXAdapter
from core.perceptual_grounder import (
    BoundingBox,
    FaceIdentity,
    ObjectDetection,
    PerceptualEvent,
    PerceptualGrounder,
    SceneCaption,
)
from core.sensory_thalamus import SensoryThalamus
from core.vision_pipeline import VisionPipeline, list_cameras


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def perception_adapter():
    return NetworkXAdapter(nx.DiGraph())


@pytest.fixture
def grounder():
    return PerceptualGrounder(camera_id="test_cam", scene_label="lab")


@pytest.fixture
def thalamus(perception_adapter, grounder):
    st = SensoryThalamus(
        adapter=perception_adapter,
        grounder=grounder,
        knowledge_harvester=None,
        event_bus=None,
        frame_skip=1,   # process every frame in tests
    )
    st.start()
    yield st
    st.stop()


def _bbox(x1=10, y1=10, x2=100, y2=100):
    return BoundingBox(x1, y1, x2, y2)


def _event_with_detections(*labels) -> PerceptualEvent:
    dets = [
        ObjectDetection(label=lbl, confidence=0.9, bbox=_bbox(i*110, 10, i*110+100, 100), track_id=i)
        for i, lbl in enumerate(labels)
    ]
    return PerceptualEvent(
        frame_id=1, camera_id="test_cam", frame_shape=(720, 1280), detections=dets
    )


# ── PerceptualGrounder: object detections ─────────────────────────────────────

class TestPerceptualGrounder:

    def test_detections_produce_triples(self, grounder):
        event = _event_with_detections("person", "laptop")
        triples = grounder.ground(event)
        assert len(triples) > 0

    def test_detection_tier2(self, grounder):
        event = _event_with_detections("chair")
        triples = grounder.ground(event)
        tier2 = [t for t in triples if t.source_tier == 2]
        assert len(tier2) > 0

    def test_detection_is_in_scene_relation(self, grounder):
        event = _event_with_detections("keyboard")
        triples = grounder.ground(event)
        relations = {t.relation for t in triples}
        assert "IS_IN_SCENE" in relations

    def test_detection_is_a_relation(self, grounder):
        event = _event_with_detections("monitor")
        triples = grounder.ground(event)
        relations = {t.relation for t in triples}
        assert "IS_A" in relations

    def test_identity_produces_triples(self, grounder):
        identity = FaceIdentity(
            name="Bryan", confidence=0.95, bbox=_bbox(),
            emotion="focused", age=32, gender="M"
        )
        event = PerceptualEvent(frame_id=2, camera_id="test_cam",
                                frame_shape=(720, 1280), identities=[identity])
        triples = grounder.ground(event)
        relations = {t.relation for t in triples}
        assert "IS_PRESENT_IN" in relations
        assert "EXPRESSES" in relations

    def test_identity_low_confidence_filtered(self, grounder):
        identity = FaceIdentity(name="Unknown", confidence=0.1, bbox=_bbox())
        event = PerceptualEvent(frame_id=3, camera_id="test_cam",
                                frame_shape=(720, 1280), identities=[identity])
        triples = grounder.ground(event)
        assert len(triples) == 0

    def test_spatial_relation_two_objects(self, grounder):
        det1 = ObjectDetection("person", 0.9, BoundingBox(10, 10, 100, 200), track_id=0)
        det2 = ObjectDetection("laptop", 0.85, BoundingBox(110, 10, 220, 200), track_id=1)
        event = PerceptualEvent(frame_id=4, camera_id="test_cam",
                                frame_shape=(720, 1280), detections=[det1, det2])
        triples = grounder.ground(event)
        spatial = [t for t in triples if t.relation in
                   {"IS_NEAR", "IS_ABOVE", "IS_BELOW", "IS_IN_SAME_SCENE_AS"}]
        assert len(spatial) > 0

    def test_caption_tier3(self, grounder):
        scene = SceneCaption(
            caption="A person sitting at a desk with a laptop",
            dense_tags=["person", "desk", "laptop", "chair"],
        )
        event = PerceptualEvent(frame_id=5, camera_id="test_cam",
                                frame_shape=(720, 1280), scene=scene)
        triples = grounder.ground(event)
        tier3 = [t for t in triples if t.source_tier == 3]
        assert len(tier3) > 0

    def test_confidence_clamped(self, grounder):
        det = ObjectDetection("person", 1.5, _bbox(), track_id=0)
        event = PerceptualEvent(frame_id=6, camera_id="test_cam",
                                frame_shape=(720, 1280), detections=[det])
        triples = grounder.ground(event)
        for t in triples:
            assert 0.0 <= t.confidence <= 1.0

    def test_empty_event_no_triples(self, grounder):
        event = PerceptualEvent(frame_id=7, camera_id="test_cam", frame_shape=(720, 1280))
        triples = grounder.ground(event)
        assert triples == []


# ── SensoryThalamus ───────────────────────────────────────────────────────────

class TestSensoryThalamus:

    def test_tier2_fast_path_writes_to_adapter(self, thalamus, perception_adapter):
        event = _event_with_detections("monitor")
        thalamus.ingest(event)
        time.sleep(0.1)
        G = perception_adapter.to_networkx()
        assert G.number_of_nodes() > 0

    def test_stats_update_after_ingest(self, thalamus):
        event = _event_with_detections("keyboard")
        thalamus.ingest(event)
        time.sleep(0.1)
        stats = thalamus.stats()
        assert stats["events_received"] >= 1
        assert stats["tier2_materialized"] >= 0

    def test_tier3_without_harvester_drops_silently(self, perception_adapter, grounder):
        st = SensoryThalamus(
            adapter=perception_adapter,
            grounder=grounder,
            knowledge_harvester=None,
            frame_skip=1,
        )
        st.start()
        scene = SceneCaption(caption="test scene", dense_tags=["object"])
        event = PerceptualEvent(frame_id=10, camera_id="test_cam",
                                frame_shape=(720, 1280), scene=scene)
        st.ingest(event)
        time.sleep(0.1)
        stats = st.stats()
        assert stats["tier3_rejected"] >= 0  # graceful drop, no crash
        st.stop()

    def test_tier3_slow_path_calls_harvester_vet(self, perception_adapter, grounder):
        mock_harvester = MagicMock()
        mock_harvester._vet.return_value = (True, "")
        mock_harvester._materialize.return_value = None

        st = SensoryThalamus(
            adapter=perception_adapter,
            grounder=grounder,
            knowledge_harvester=mock_harvester,
            frame_skip=1,
        )
        st.start()
        scene = SceneCaption(caption="test scene", dense_tags=["desk", "monitor"])
        event = PerceptualEvent(frame_id=11, camera_id="test_cam",
                                frame_shape=(720, 1280), scene=scene)
        st.ingest(event)
        time.sleep(0.2)
        assert mock_harvester._vet.called
        st.stop()

    def test_event_bus_receives_perception_event(self, perception_adapter, grounder):
        received = []
        mock_bus = MagicMock()
        mock_bus.publish.side_effect = lambda kind, payload: received.append(kind)

        st = SensoryThalamus(
            adapter=perception_adapter,
            grounder=grounder,
            event_bus=mock_bus,
            frame_skip=1,
        )
        st.start()
        event = _event_with_detections("chair")
        st.ingest(event)
        time.sleep(0.15)
        assert "PERCEPTION_EVENT" in received
        st.stop()

    def test_frame_skip_reduces_dispatch(self, perception_adapter, grounder):
        st = SensoryThalamus(
            adapter=perception_adapter,
            grounder=grounder,
            frame_skip=10,  # process only 1-in-10
        )
        st.start()
        for i in range(10):
            event = PerceptualEvent(frame_id=i, camera_id="test_cam", frame_shape=(720, 1280))
            st.ingest(event)
        time.sleep(0.2)
        stats = st.stats()
        assert stats["events_dispatched"] <= 2   # at most 2 dispatched from 10 enqueued
        st.stop()

    def test_recent_triples_ring_buffer(self, thalamus):
        for i in range(5):
            event = _event_with_detections("laptop")
            event.frame_id = i
            thalamus.ingest(event)
        time.sleep(0.3)
        recent = thalamus.recent_triples(n=50)
        assert isinstance(recent, list)

    def test_stop_is_idempotent(self, thalamus):
        thalamus.stop()
        thalamus.stop()  # second stop must not raise


# ── VisionPipeline (stub/offline mode) ───────────────────────────────────────

class TestVisionPipelineOffline:

    def test_snapshot_raises_without_camera(self):
        mock_thalamus = MagicMock()
        mock_grounder = MagicMock()
        vp = VisionPipeline(thalamus=mock_thalamus, grounder=mock_grounder, camera_id=99)
        with pytest.raises(RuntimeError):
            vp.snapshot()

    def test_status_initial_state(self):
        mock_thalamus = MagicMock()
        mock_grounder = MagicMock()
        vp = VisionPipeline(thalamus=mock_thalamus, grounder=mock_grounder)
        s = vp.status()
        assert s["running"] is False
        assert s["frames_captured"] == 0
        assert "models_loaded" in s

    def test_process_frame_no_models(self):
        """_process_frame with no models loaded returns an empty PerceptualEvent."""
        mock_thalamus = MagicMock()
        mock_grounder = MagicMock()
        vp = VisionPipeline(thalamus=mock_thalamus, grounder=mock_grounder)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = vp._process_frame(frame, frame_id=42)
        assert event.frame_id == 42
        assert event.detections == []
        assert event.identities == []
        assert event.scene is None

    def test_list_cameras_no_crash(self):
        # May return [] on CI — just must not raise
        result = list_cameras(max_index=3)
        assert isinstance(result, list)

    def test_stop_before_start_no_crash(self):
        mock_thalamus = MagicMock()
        mock_grounder = MagicMock()
        vp = VisionPipeline(thalamus=mock_thalamus, grounder=mock_grounder)
        vp.stop()  # must not raise

    def test_start_without_opencv_exits_cleanly(self):
        """If opencv is unavailable, start() logs and returns without crashing."""
        mock_thalamus = MagicMock()
        mock_grounder = MagicMock()
        vp = VisionPipeline(thalamus=mock_thalamus, grounder=mock_grounder)
        with patch("core.vision_pipeline._try_import_cv2", return_value=None):
            vp.start()
            time.sleep(0.2)
            assert vp.status()["running"] is False
        vp.stop()


# ── FederatedGraphRegistry: perception_kb domain ─────────────────────────────

class TestFederatedRegistryPerceptionKb:

    def test_perception_kb_registered(self):
        from core.federated_registry import FederatedGraphRegistry
        registry = FederatedGraphRegistry()
        perception_adapter = NetworkXAdapter(nx.DiGraph())
        registry.graphs["perception_kb"] = perception_adapter
        assert "perception_kb" in registry.graphs

    def test_perception_kb_queriable_alongside_engineering_kb(self):
        from core.federated_registry import FederatedGraphRegistry
        registry = FederatedGraphRegistry()

        eng_G = nx.DiGraph()
        eng_G.add_edge("newton", "einstein", relation="INFLUENCED", weight=0.9)
        registry.graphs["engineering_kb"] = NetworkXAdapter(eng_G)

        perc_G = nx.DiGraph()
        perc_G.add_edge("bryan", "laptop", relation="IS_IN_SAME_SCENE_AS", weight=0.85)
        registry.graphs["perception_kb"] = NetworkXAdapter(perc_G)

        eng_adapter  = registry.get_adapter("engineering_kb")
        perc_adapter = registry.get_adapter("perception_kb")

        eng_nodes  = list(eng_adapter.to_networkx().nodes())
        perc_nodes = list(perc_adapter.to_networkx().nodes())

        assert "newton" in eng_nodes
        assert "bryan"  in perc_nodes
        assert len(set(eng_nodes) & set(perc_nodes)) == 0  # no cross-contamination


# ── Full offline end-to-end pipeline ─────────────────────────────────────────

class TestFullPipelineOffline:

    def test_event_to_graph_edge(self):
        """
        A PerceptualEvent with an object detection flows through
        PerceptualGrounder -> SensoryThalamus -> adapter.add_edge()
        and the resulting edge appears in the NetworkX graph.
        """
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        grounder = PerceptualGrounder(camera_id="test_cam", scene_label="office")
        st = SensoryThalamus(adapter=adapter, grounder=grounder, frame_skip=1)
        st.start()

        det = ObjectDetection("person", 0.92, BoundingBox(10, 10, 200, 400), track_id=1)
        event = PerceptualEvent(
            frame_id=1, camera_id="test_cam",
            frame_shape=(1080, 1920), detections=[det],
        )
        st.ingest(event)
        time.sleep(0.25)
        st.stop()

        stats = st.stats()
        print(f"\n[Phase300] Stats after one detection event:")
        print(f"  events_received:    {stats['events_received']}")
        print(f"  events_dispatched:  {stats['events_dispatched']}")
        print(f"  triples_produced:   {stats['triples_produced']}")
        print(f"  tier2_materialized: {stats['tier2_materialized']}")
        print(f"  Graph nodes: {list(G.nodes())}")
        print(f"  Graph edges: {G.number_of_edges()}")

        assert stats["tier2_materialized"] >= 1
        assert G.number_of_nodes() >= 2
        assert G.number_of_edges() >= 1

    def test_multiple_detections_build_scene_graph(self):
        """Multiple detections in one frame build a small scene graph."""
        G = nx.DiGraph()
        adapter = NetworkXAdapter(G)
        grounder = PerceptualGrounder(camera_id="test_cam", scene_label="office",
                                      emit_spatial_relations=True)
        st = SensoryThalamus(adapter=adapter, grounder=grounder, frame_skip=1)
        st.start()

        dets = [
            ObjectDetection("person",   0.95, BoundingBox(10, 10, 150, 400), track_id=0),
            ObjectDetection("monitor",  0.90, BoundingBox(200, 50, 500, 350), track_id=1),
            ObjectDetection("keyboard", 0.88, BoundingBox(200, 380, 500, 430), track_id=2),
        ]
        event = PerceptualEvent(
            frame_id=1, camera_id="test_cam",
            frame_shape=(1080, 1920), detections=dets,
        )
        st.ingest(event)
        time.sleep(0.25)
        st.stop()

        print(f"\n[Phase300] Scene graph nodes: {sorted(G.nodes())}")
        print(f"[Phase300] Scene graph edges: {G.number_of_edges()}")
        for u, v, d in G.edges(data=True):
            print(f"  {u} --{d.get('relation', '?')}--> {v}")

        assert G.number_of_nodes() >= 4  # person, monitor, keyboard, office + scene label
        assert G.number_of_edges() >= 3

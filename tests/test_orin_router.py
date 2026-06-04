"""
Tests for api/orin_router.py

All SSH/Orin calls are mocked — no live device required.
Uses FastAPI TestClient with the router mounted at /orin.
"""
from __future__ import annotations

import asyncio
import io
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import networkx as nx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from adapters.networkx_adapter import NetworkXAdapter
from core.thalamus import ProcessedEdge


# ---------------------------------------------------------------------------
# App fixture — fresh router state per module
# ---------------------------------------------------------------------------

def _make_app() -> tuple[FastAPI, Any]:
    """Return (app, client) with orin_router freshly imported."""
    import importlib
    import api.orin_router as mod

    # Reset module-level state between test groups
    mod._node = None
    mod._llm_fn = None
    mod._nexus_graph_getter = None
    mod._local_graph = NetworkXAdapter(nx.DiGraph())
    mod._watch_active.clear()

    app = FastAPI()
    app.include_router(mod.orin_router, prefix="/orin")
    client = TestClient(app, raise_server_exceptions=False)
    return app, client, mod


@pytest.fixture()
def router_ctx():
    """Yield (client, module) with clean state."""
    _, client, mod = _make_app()
    yield client, mod
    # Cleanup
    mod._node = None
    mod._llm_fn = None
    mod._nexus_graph_getter = None
    mod._watch_active.clear()


def _make_edge(source="A", relation="rel", target="B", provenance="orin:test", confidence=0.9):
    return ProcessedEdge(source=source, target=target, relation=relation,
                         confidence=confidence, provenance=provenance)


def _make_mock_node(connected=True):
    node = MagicMock()
    node.connected = connected
    node._host = "agxorin.local"
    node._whisper_model = "base"
    node.connect.return_value = None
    node.disconnect.return_value = None
    node.exec.return_value = ("OK", "", 0)
    node.open_tunnel = MagicMock()
    return node


def _make_mock_pa(edges=None):
    pa = MagicMock()
    if edges is None:
        edges = [_make_edge()]
    pa.ingest_audio.return_value = edges
    pa.ingest_image.return_value = edges
    pa.ingest_document.return_value = edges
    pa.ingest_text.return_value = edges
    return pa


# ===========================================================================
# A. Graph getter injection
# ===========================================================================

class TestGraphGetter:
    def test_register_graph_getter_uses_nexus_graph(self, router_ctx):
        client, mod = router_ctx
        mock_adapter = NetworkXAdapter(nx.DiGraph())
        mock_adapter._G.add_node("X")
        mod.register_graph_getter(lambda: mock_adapter)
        assert mod._get_graph() is mock_adapter

    def test_get_graph_fallback_when_getter_returns_none(self, router_ctx):
        client, mod = router_ctx
        mod.register_graph_getter(lambda: None)
        assert mod._get_graph() is mod._local_graph

    def test_get_graph_fallback_when_no_getter(self, router_ctx):
        client, mod = router_ctx
        mod._nexus_graph_getter = None
        assert mod._get_graph() is mod._local_graph


# ===========================================================================
# B. Status endpoint
# ===========================================================================

class TestStatus:
    def test_status_disconnected(self, router_ctx):
        client, mod = router_ctx
        resp = client.get("/orin/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is False

    def test_status_returns_graph_counts(self, router_ctx):
        client, mod = router_ctx
        g = NetworkXAdapter(nx.DiGraph())
        g._G.add_edge("A", "B")
        mod.register_graph_getter(lambda: g)
        resp = client.get("/orin/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_nodes"] == 2
        assert data["graph_edges"] == 1

    def test_status_connected_returns_system_info(self, router_ctx):
        client, mod = router_ctx
        node = _make_mock_node()
        node.exec.return_value = ("OK\n", "", 0)
        mod._node = node
        mod._llm_fn = None
        resp = client.get("/orin/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["host"] == "agxorin.local"


# ===========================================================================
# C. Connect / Disconnect
# ===========================================================================

class TestConnectDisconnect:
    @patch("api.orin_router.OrinPerceptionNode")
    @patch("api.orin_router._start_persistent_tunnel")
    def test_connect_calls_orin_node(self, mock_tunnel, MockNode, router_ctx):
        client, mod = router_ctx
        mock_node = _make_mock_node()
        MockNode.return_value = mock_node
        resp = client.post("/orin/connect", json={
            "host": "agxorin.local", "user": "nvidia", "port": 22,
            "whisper_model": "base", "ollama_model": "llava:7b",
        })
        assert resp.status_code == 200
        MockNode.assert_called_once()
        mock_node.connect.assert_called_once()

    @patch("api.orin_router.OrinPerceptionNode")
    @patch("api.orin_router._start_persistent_tunnel")
    def test_connect_starts_tunnel(self, mock_tunnel, MockNode, router_ctx):
        client, mod = router_ctx
        MockNode.return_value = _make_mock_node()
        client.post("/orin/connect", json={"host": "agxorin.local"})
        mock_tunnel.assert_called_once()

    @patch("api.orin_router.OrinPerceptionNode")
    def test_connect_ssh_failure_returns_500(self, MockNode, router_ctx):
        client, mod = router_ctx
        mock_node = _make_mock_node()
        mock_node.connect.side_effect = RuntimeError("Connection refused")
        MockNode.return_value = mock_node
        resp = client.post("/orin/connect", json={"host": "agxorin.local"})
        assert resp.status_code == 500
        assert "SSH connection failed" in resp.json()["detail"]

    def test_disconnect_clears_node(self, router_ctx):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mod._llm_fn = MagicMock()
        resp = client.post("/orin/disconnect")
        assert resp.status_code == 200
        assert mod._node is None
        assert mod._llm_fn is None


# ===========================================================================
# D. Perceive endpoints
# ===========================================================================

class TestPerceiveEndpoints:
    def test_perceive_text_requires_connection(self, router_ctx):
        client, mod = router_ctx
        resp = client.post("/orin/perceive/text", json={"text": "hello"})
        assert resp.status_code == 503

    @patch("api.orin_router._get_perception_adapter")
    def test_perceive_text_returns_result(self, mock_gpa, router_ctx):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mock_gpa.return_value = _make_mock_pa()
        resp = client.post("/orin/perceive/text", json={"text": "AGX >> made_by >> NVIDIA"})
        assert resp.status_code == 200
        data = resp.json()
        assert "triples" in data
        assert data["triples"] == [["A", "rel", "B"]]

    @patch("api.orin_router._get_perception_adapter")
    def test_perceive_audio_uploads_tempfile(self, mock_gpa, router_ctx, tmp_path):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mock_gpa.return_value = _make_mock_pa()
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 36)
        with open(audio, "rb") as f:
            resp = client.post("/orin/perceive/audio",
                               files={"file": ("test.wav", f, "audio/wav")})
        assert resp.status_code == 200

    @patch("api.orin_router._get_perception_adapter")
    def test_perceive_image_uploads_tempfile(self, mock_gpa, router_ctx, tmp_path):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mock_gpa.return_value = _make_mock_pa()
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        with open(img, "rb") as f:
            resp = client.post("/orin/perceive/image",
                               files={"file": ("frame.jpg", f, "image/jpeg")})
        assert resp.status_code == 200

    @patch("api.orin_router._get_perception_adapter")
    def test_perceive_document_uploads_tempfile(self, mock_gpa, router_ctx, tmp_path):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mock_gpa.return_value = _make_mock_pa()
        doc = tmp_path / "doc.txt"
        doc.write_text("Node A relates to Node B.")
        with open(doc, "rb") as f:
            resp = client.post("/orin/perceive/document",
                               files={"file": ("doc.txt", f, "text/plain")})
        assert resp.status_code == 200

    @patch("api.orin_router._get_perception_adapter")
    def test_perceive_result_edges_added_counted(self, mock_gpa, router_ctx):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        # Pre-populate graph with 1 edge, then mock pa adds 1 more
        g = NetworkXAdapter(nx.DiGraph())
        g._G.add_edge("pre", "existing", relation="existed", confidence=1.0)
        mod._local_graph = g

        pa = MagicMock()
        def _side_effect(text, context=""):
            g._G.add_edge("A", "B", relation="rel", confidence=0.9, provenance="orin:test")
            return [_make_edge()]
        pa.ingest_text.side_effect = _side_effect
        mock_gpa.return_value = pa

        resp = client.post("/orin/perceive/text", json={"text": "A rel B"})
        assert resp.status_code == 200
        assert resp.json()["edges_added"] == 1


# ===========================================================================
# E. Graph endpoints
# ===========================================================================

class TestGraphEndpoints:
    def test_get_graph_returns_nodes_edges(self, router_ctx):
        client, mod = router_ctx
        g = NetworkXAdapter(nx.DiGraph())
        g._G.add_edge("Alice", "Bob", relation="KNOWS", confidence=0.8, provenance="test")
        mod._local_graph = g
        resp = client.get("/orin/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["relation"] == "KNOWS"

    def test_clear_graph_local_resets_local_graph(self, router_ctx):
        client, mod = router_ctx
        g = NetworkXAdapter(nx.DiGraph())
        g._G.add_edge("X", "Y")
        mod._local_graph = g
        mod._nexus_graph_getter = None
        resp = client.delete("/orin/graph")
        assert resp.status_code == 200
        assert mod._local_graph._G.number_of_edges() == 0

    def test_clear_graph_nexus_removes_orin_edges_only(self, router_ctx):
        client, mod = router_ctx
        g = NetworkXAdapter(nx.DiGraph())
        g._G.add_edge("A", "B", relation="REL", confidence=1.0, provenance="orin:perceive")
        g._G.add_edge("C", "D", relation="REL", confidence=1.0, provenance="metaqa:train")
        mod.register_graph_getter(lambda: g)
        resp = client.delete("/orin/graph")
        assert resp.status_code == 200
        # Orin edge removed, MetaQA edge kept
        assert not g._G.has_edge("A", "B")
        assert g._G.has_edge("C", "D")


# ===========================================================================
# F. Watch endpoints
# ===========================================================================

class TestWatchEndpoints:
    def test_watch_start_requires_connection(self, router_ctx):
        client, mod = router_ctx
        resp = client.post("/orin/watch/start", json={"remote_dir": "/home/user/frames"})
        assert resp.status_code == 503

    @patch("api.orin_router.threading.Thread")
    def test_watch_start_spawns_thread(self, MockThread, router_ctx):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mock_thread = MagicMock()
        MockThread.return_value = mock_thread
        resp = client.post("/orin/watch/start", json={"remote_dir": "/home/user/frames"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "watching"
        mock_thread.start.assert_called_once()

    @patch("api.orin_router.threading.Thread")
    def test_watch_already_watching_returns_status(self, MockThread, router_ctx):
        client, mod = router_ctx
        mod._node = _make_mock_node()
        mod._watch_active.set()
        resp = client.post("/orin/watch/start", json={"remote_dir": "/home/user/frames"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_watching"
        MockThread.assert_not_called()

    def test_watch_stop_clears_active_flag(self, router_ctx):
        client, mod = router_ctx
        mod._watch_active.set()
        resp = client.post("/orin/watch/stop")
        assert resp.status_code == 200
        assert not mod._watch_active.is_set()


# ===========================================================================
# G. SSE stream — test the async generator directly to avoid blocking
# ===========================================================================

class TestSSEStream:
    @pytest.mark.asyncio
    async def test_watch_stream_yields_connected_event(self, router_ctx):
        from api.orin_router import watch_stream
        resp = await watch_stream()
        gen = resp.body_iterator
        first_chunk = await gen.__anext__()
        assert '"type":"connected"' in first_chunk
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_watch_stream_media_type(self, router_ctx):
        from api.orin_router import watch_stream
        resp = await watch_stream()
        assert resp.media_type == "text/event-stream"
        await resp.body_iterator.aclose()

    @pytest.mark.asyncio
    async def test_watch_stream_delivers_queued_event(self, router_ctx):
        client, mod = router_ctx
        mod._event_queue.put_nowait({"type": "text", "triples": [["A", "rel", "B"]]})
        from api.orin_router import watch_stream
        resp = await watch_stream()
        gen = resp.body_iterator
        first = await gen.__anext__()
        assert '"type":"connected"' in first
        second = await gen.__anext__()
        event = json.loads(second.replace("data:", "").strip())
        assert event["type"] == "text"
        assert event["triples"] == [["A", "rel", "B"]]
        await gen.aclose()

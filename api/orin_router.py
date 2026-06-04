"""
Orin Interface Router — FastAPI endpoints for AGX Orin perception management.

Mount into the main CEREBRUM server via:
    from api.orin_router import orin_router
    app.include_router(orin_router, prefix="/orin", tags=["orin"])

Endpoints
---------
GET  /orin/status                  — connection state + system info
POST /orin/connect                 — open SSH connection
POST /orin/disconnect              — close SSH connection
POST /orin/perceive/audio          — upload audio file, transcribe on Orin
POST /orin/perceive/image          — upload image, run vision on Orin
POST /orin/perceive/document       — upload document, extract triples
POST /orin/perceive/text           — extract triples from plain text
GET  /orin/graph                   — current perception graph (nodes + edges)
DELETE /orin/graph                 — clear the perception graph
POST /orin/watch/start             — start watching a remote directory
POST /orin/watch/stop              — stop watching
GET  /orin/watch/stream            — SSE stream of live perception events
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from adapters.networkx_adapter import NetworkXAdapter
from adapters.orin_node import OrinPerceptionNode
from adapters.perception_adapter import PerceptionAdapter, _extract_triples_llm

logger = logging.getLogger("cerebrum.orin")

orin_router = APIRouter()

# ---------------------------------------------------------------------------
# Shared state (one Orin connection per server process)
# ---------------------------------------------------------------------------

_node: Optional[OrinPerceptionNode] = None
_node_lock = threading.Lock()
_graph = NetworkXAdapter(nx.DiGraph())
_event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
_watch_thread: Optional[threading.Thread] = None
_watch_active = threading.Event()
_llm_fn: Any = None   # set to OllamaAdapter when connected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_node() -> OrinPerceptionNode:
    if _node is None or not _node.connected:
        raise HTTPException(status_code=503, detail="Not connected to Orin.")
    return _node


def _get_perception_adapter() -> PerceptionAdapter:
    n = _require_node()
    return n.create_perception_adapter(_graph, confidence_threshold=0.6, llm_fn=_llm_fn)


def _edges_snapshot() -> List[Dict]:
    edges = []
    for u, v, data in _graph._G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relation": data.get("relation", "RELATED_TO"),
            "confidence": data.get("confidence", 1.0),
            "provenance": data.get("provenance", ""),
        })
    return edges


def _nodes_snapshot() -> List[Dict]:
    return [
        {"id": n, "label": _graph._G.nodes[n].get("label", n)}
        for n in _graph._G.nodes
    ]


def _put_event(event: Dict) -> None:
    """Put an event on the queue; drop oldest if full."""
    try:
        _event_queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            _event_queue.get_nowait()
            _event_queue.put_nowait(event)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ConnectRequest(BaseModel):
    host: str
    user: str = "nvidia"
    port: int = 22
    password: Optional[str] = None
    key_path: Optional[str] = None
    whisper_model: str = "base"
    ollama_model: str = "llama3.1:8b"
    ollama_port: int = 11434
    ollama_local_port: int = 21434


class PerceiveTextRequest(BaseModel):
    text: str
    context: str = ""


class WatchStartRequest(BaseModel):
    remote_dir: str
    modality: str = "auto"
    context: str = ""
    poll_interval: float = 2.0


class OrinStatus(BaseModel):
    connected: bool
    host: Optional[str] = None
    whisper_model: Optional[str] = None
    ollama_model: Optional[str] = None
    system_info: Dict[str, str] = {}
    graph_nodes: int = 0
    graph_edges: int = 0
    watch_active: bool = False


class PerceiveResult(BaseModel):
    transcript: str = ""
    confidence: float = 0.0
    provenance: str = ""
    triples: List[List[str]] = []
    edges_added: int = 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@orin_router.get("/status", response_model=OrinStatus)
async def get_status():
    global _node, _graph
    if _node is None or not _node.connected:
        return OrinStatus(
            connected=False,
            graph_nodes=_graph._G.number_of_nodes(),
            graph_edges=_graph._G.number_of_edges(),
        )
    info: Dict[str, str] = {}
    try:
        for cmd, key in [
            ("cat /etc/nv_tegra_release 2>/dev/null | head -1 || uname -r", "jetpack"),
            ("free -h | awk '/Mem:/{print $2\" total, \"$3\" used\"}'", "ram"),
            ("df -h / | awk 'NR==2{print $4\" free of \"$2}'", "disk"),
            ("python3 -c \"import faster_whisper; print('OK')\" 2>/dev/null || echo 'not installed'", "whisper"),
            ("ollama list 2>/dev/null | awk 'NR>1{print $1}' | tr '\n' ',' | sed 's/,$//'", "ollama_models"),
        ]:
            out, _, _ = _node.exec(cmd, timeout=5.0)
            info[key] = out.strip()[:100]
    except Exception as exc:
        info["error"] = str(exc)

    return OrinStatus(
        connected=True,
        host=_node._host,
        whisper_model=_node._whisper_model,
        ollama_model=_llm_fn._model if _llm_fn else None,
        system_info=info,
        graph_nodes=_graph._G.number_of_nodes(),
        graph_edges=_graph._G.number_of_edges(),
        watch_active=_watch_active.is_set(),
    )


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------

@orin_router.post("/connect")
async def connect(req: ConnectRequest):
    global _node, _llm_fn

    with _node_lock:
        if _node and _node.connected:
            _node.disconnect()

        node = OrinPerceptionNode(
            host=req.host,
            user=req.user,
            port=req.port,
            password=req.password,
            key_path=req.key_path,
            whisper_model=req.whisper_model,
        )
        try:
            node.connect()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SSH connection failed: {exc}")

        _node = node

    # Wire OllamaAdapter through SSH tunnel on the ollama port
    try:
        from llm_bridge.adapters import OllamaAdapter
        # We keep the tunnel open for the duration of the server by using the
        # node's open_channel mechanism rather than the context manager.
        _llm_fn = OllamaAdapter(
            model=req.ollama_model,
            url=f"http://localhost:{req.ollama_local_port}",
        )
        # Start a persistent tunnel thread
        _start_persistent_tunnel(req.ollama_port, req.ollama_local_port)
    except Exception as exc:
        logger.warning("Could not wire Ollama: %s", exc)
        _llm_fn = None

    logger.info("Connected to Orin at %s", req.host)
    return {"status": "connected", "host": req.host}


@orin_router.post("/disconnect")
async def disconnect():
    global _node, _llm_fn
    _watch_active.clear()
    with _node_lock:
        if _node:
            try:
                _node.disconnect()
            except Exception:
                pass
            _node = None
    _llm_fn = None
    return {"status": "disconnected"}


# Persistent SSH tunnel for the Ollama port
_tunnel_thread: Optional[threading.Thread] = None
_tunnel_stop = threading.Event()


def _start_persistent_tunnel(remote_port: int, local_port: int) -> None:
    global _tunnel_thread, _tunnel_stop
    _tunnel_stop.set()   # stop any existing tunnel
    time.sleep(0.1)
    _tunnel_stop.clear()

    def _run() -> None:
        while not _tunnel_stop.is_set():
            try:
                if _node and _node.connected:
                    with _node.open_tunnel(remote_port, local_port):
                        while not _tunnel_stop.is_set() and _node and _node.connected:
                            time.sleep(1.0)
            except Exception as exc:
                logger.debug("Tunnel error (will retry): %s", exc)
                time.sleep(3.0)

    _tunnel_thread = threading.Thread(target=_run, daemon=True, name="orin-tunnel")
    _tunnel_thread.start()


# ---------------------------------------------------------------------------
# Perception endpoints
# ---------------------------------------------------------------------------

@orin_router.post("/perceive/audio", response_model=PerceiveResult)
async def perceive_audio(context: str = "", file: UploadFile = File(...)):
    pa = _get_perception_adapter()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "clip.wav").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    before = _graph._G.number_of_edges()
    try:
        edges = pa.ingest_audio(tmp_path, context=context)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = _make_result(edges, before)
    _put_event({"type": "audio", "filename": file.filename, **result.model_dump()})
    return result


@orin_router.post("/perceive/image", response_model=PerceiveResult)
async def perceive_image(context: str = "", file: UploadFile = File(...)):
    pa = _get_perception_adapter()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "frame.jpg").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    before = _graph._G.number_of_edges()
    try:
        edges = pa.ingest_image(tmp_path, context=context)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = _make_result(edges, before)
    _put_event({"type": "image", "filename": file.filename, **result.model_dump()})
    return result


@orin_router.post("/perceive/document", response_model=PerceiveResult)
async def perceive_document(context: str = "", file: UploadFile = File(...)):
    pa = _get_perception_adapter()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "doc.txt").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    before = _graph._G.number_of_edges()
    try:
        edges = pa.ingest_document(tmp_path, context=context)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    result = _make_result(edges, before)
    _put_event({"type": "document", "filename": file.filename, **result.model_dump()})
    return result


@orin_router.post("/perceive/text", response_model=PerceiveResult)
async def perceive_text(req: PerceiveTextRequest):
    pa = _get_perception_adapter()
    before = _graph._G.number_of_edges()
    edges = pa.ingest_text(req.text, context=req.context)
    result = _make_result(edges, before)
    _put_event({"type": "text", "filename": "(direct)", **result.model_dump()})
    return result


def _make_result(edges: list, edges_before: int) -> PerceiveResult:
    triples = [[e.source, e.relation, e.target] for e in edges]
    transcript = edges[0].provenance if edges else ""
    return PerceiveResult(
        confidence=edges[0].confidence if edges else 0.0,
        provenance=edges[0].provenance if edges else "",
        triples=triples,
        edges_added=_graph._G.number_of_edges() - edges_before,
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@orin_router.get("/graph")
async def get_graph():
    return {
        "nodes": _nodes_snapshot(),
        "edges": _edges_snapshot(),
    }


@orin_router.delete("/graph")
async def clear_graph():
    global _graph
    _graph = NetworkXAdapter(nx.DiGraph())
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Sensor watch
# ---------------------------------------------------------------------------

@orin_router.post("/watch/start")
async def watch_start(req: WatchStartRequest):
    global _watch_thread
    node = _require_node()

    if _watch_active.is_set():
        return {"status": "already_watching"}

    _watch_active.set()

    def _watch_worker() -> None:
        pa = node.create_perception_adapter(_graph, confidence_threshold=0.6, llm_fn=_llm_fn)
        for result in node.watch_directory(req.remote_dir, req.modality, req.poll_interval):
            if not _watch_active.is_set():
                break
            before = _graph._G.number_of_edges()
            edges = pa._process_result(result, req.context)
            ev = {
                "type": "watch",
                "provenance": result.provenance,
                "transcript": result.raw_text[:300],
                "triples": [[e.source, e.relation, e.target] for e in edges],
                "edges_added": _graph._G.number_of_edges() - before,
            }
            _put_event(ev)

    _watch_thread = threading.Thread(target=_watch_worker, daemon=True, name="orin-watch")
    _watch_thread.start()
    return {"status": "watching", "remote_dir": req.remote_dir}


@orin_router.post("/watch/stop")
async def watch_stop():
    _watch_active.clear()
    return {"status": "stopped"}


@orin_router.get("/watch/stream")
async def watch_stream():
    """Server-Sent Events stream of live perception results."""

    async def _generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(_event_queue.get(), timeout=15.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield "data: {\"type\":\"heartbeat\"}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

"""
Phase 11 — StreamAdapter: Live, Mutable Knowledge Graph Adapter.

Extends NetworkXAdapter with:
  - Thread-safe live graph mutation (add/remove nodes and edges)
  - Sliding window edge eviction (time- and count-bounded)
  - Incremental DSCF community updates on affected subgraphs
  - Pluggable StreamSource backends (file tail, WebSocket, HTTP poll,
    Python callback, MQTT)
  - Background ingestion loop (runs in a daemon thread)

Quick Start
-----------
from adapters.stream_adapter import StreamAdapter, PythonCallbackSource
from core.discretizer import ThresholdDiscretizer

disc = ThresholdDiscretizer("cpu_load", low=20, high=70, spike=90)
adapter = StreamAdapter(time_window_seconds=30, max_edges=5000)

# Push events via callback
source = PythonCallbackSource(lambda: disc.process(get_cpu_load()))
adapter.add_source(source)
adapter.start()

# Query the live graph normally
adapter.find_entities("cpu_load")
"""
from __future__ import annotations

import json
import queue
import threading
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.graph_adapter import Entity, Edge
from core.stream_engine import (
    StreamEvent,
    SlidingWindowBuffer,
    IncrementalCommunityUpdater,
    StreamStats,
)

logger = logging.getLogger("parallax.stream")


# ---------------------------------------------------------------------------
# StreamSource ABC — all source backends implement this
# ---------------------------------------------------------------------------

class StreamSource(ABC):
    """
    Abstract base class for stream sources.

    Implementations must yield ``StreamEvent`` objects indefinitely
    (or until ``stop()`` is called) from ``read()``.
    """

    @abstractmethod
    def read(self) -> Iterator[StreamEvent]:
        """
        Yield StreamEvents. Must be a generator.
        Block until the next event is available.
        The generator exits when ``stop()`` has been called.
        """
        ...

    def stop(self) -> None:
        """Signal the source to stop producing events."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# PythonCallbackSource — wrap any callable/generator
# ---------------------------------------------------------------------------

class PythonCallbackSource(StreamSource):
    """
    Wrap a Python callable or generator factory as a stream source.

    The callable is invoked repeatedly and must return either:
      - A single StreamEvent
      - A list of StreamEvents
      - None / [] (no event this tick)

    Parameters
    ----------
    callback        : callable() → StreamEvent | List[StreamEvent] | None
    poll_interval   : seconds to wait between calls (0 = as fast as possible)
    """

    def __init__(
        self,
        callback: Callable[[], Any],
        poll_interval: float = 0.1,
    ):
        self._callback = callback
        self._poll_interval = poll_interval
        self._running = True

    def read(self) -> Iterator[StreamEvent]:
        while self._running:
            try:
                result = self._callback()
                if result is None:
                    pass
                elif isinstance(result, StreamEvent):
                    yield result
                elif isinstance(result, list):
                    yield from result
            except Exception as e:
                logger.warning("PythonCallbackSource error: %s", e)
            if self._poll_interval > 0:
                time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# FileTailSource — tail a file line-by-line (CSV or JSON Lines)
# ---------------------------------------------------------------------------

class FileTailSource(StreamSource):
    """
    Tail a file and emit one StreamEvent per line.

    Supports two line formats:
      CSV:  ``source,relation,target[,metadata_json]``
      JSON: ``{"source": "...", "relation": "...", "target": "..."}``

    The file may be written to by another process while Parallax reads it
    (the classic tail -f pattern). New lines are yielded as they arrive.

    Parameters
    ----------
    path            : path to the file to tail
    format          : "csv" or "json" (auto-detected from extension if "auto")
    poll_interval   : seconds between read attempts when no new data
    source_col      : CSV column name for source entity
    target_col      : CSV column name for target entity
    relation_col    : CSV column name for relation type
    encoding        : file encoding
    """

    def __init__(
        self,
        path: str,
        format: str = "auto",
        poll_interval: float = 0.05,
        source_col: str = "source",
        target_col: str = "target",
        relation_col: str = "relation",
        encoding: str = "utf-8",
        ttl: float = 0.0,
    ):
        self._path = Path(path)
        self._format = format
        self._poll = poll_interval
        self._src = source_col
        self._tgt = target_col
        self._rel = relation_col
        self._enc = encoding
        self._ttl = ttl
        self._running = True

        if format == "auto":
            ext = self._path.suffix.lower()
            self._format = "json" if ext in (".json", ".jsonl") else "csv"

    def read(self) -> Iterator[StreamEvent]:
        with open(self._path, "r", encoding=self._enc) as f:
            # Seek to end to tail only new lines
            f.seek(0, 2)
            while self._running:
                line = f.readline()
                if not line:
                    time.sleep(self._poll)
                    continue
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ev = self._parse_line(line)
                if ev:
                    yield ev

    def _parse_line(self, line: str) -> Optional[StreamEvent]:
        try:
            if self._format == "json":
                d = json.loads(line)
                return StreamEvent(
                    source=str(d.get(self._src, "")),
                    relation=str(d.get(self._rel, "RELATED_TO")),
                    target=str(d.get(self._tgt, "")),
                    metadata={k: v for k, v in d.items()
                               if k not in (self._src, self._rel, self._tgt)},
                    ttl=self._ttl,
                )
            else:
                parts = line.split(",", 3)
                if len(parts) < 2:
                    return None
                src = parts[0].strip()
                tgt = parts[1].strip() if len(parts) > 1 else ""
                rel = parts[2].strip() if len(parts) > 2 else "RELATED_TO"
                meta: Dict[str, Any] = {}
                if len(parts) > 3:
                    try:
                        meta = json.loads(parts[3])
                    except json.JSONDecodeError:
                        pass
                if not src or not tgt:
                    return None
                return StreamEvent(source=src, relation=rel, target=tgt,
                                   metadata=meta, ttl=self._ttl)
        except Exception as e:
            logger.debug("FileTailSource parse error: %s | line: %r", e, line)
            return None

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# HTTPPollingSource — poll a REST endpoint
# ---------------------------------------------------------------------------

class HTTPPollingSource(StreamSource):
    """
    Poll an HTTP endpoint at a fixed interval and parse the response
    as StreamEvents.

    The response must be one of:
      - A JSON object with ``source``, ``relation``, ``target`` fields
      - A JSON array of such objects
      - A JSON Lines string (one object per line)

    Parameters
    ----------
    url           : endpoint to poll
    poll_interval : seconds between requests
    headers       : optional request headers (e.g. auth tokens)
    source_col    : JSON field name for source
    target_col    : JSON field name for target
    relation_col  : JSON field name for relation
    timeout       : request timeout in seconds
    """

    def __init__(
        self,
        url: str,
        poll_interval: float = 1.0,
        headers: Optional[Dict[str, str]] = None,
        source_col: str = "source",
        target_col: str = "target",
        relation_col: str = "relation",
        timeout: float = 5.0,
        ttl: float = 0.0,
    ):
        self._url = url
        self._poll = poll_interval
        self._headers = headers or {}
        self._src = source_col
        self._tgt = target_col
        self._rel = relation_col
        self._timeout = timeout
        self._ttl = ttl
        self._running = True

    def read(self) -> Iterator[StreamEvent]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for HTTPPollingSource: pip install httpx")

        with httpx.Client(headers=self._headers, timeout=self._timeout) as client:
            while self._running:
                try:
                    resp = client.get(self._url)
                    resp.raise_for_status()
                    yield from self._parse_response(resp.text)
                except Exception as e:
                    logger.warning("HTTPPollingSource fetch error: %s", e)
                time.sleep(self._poll)

    def _parse_response(self, text: str) -> List[StreamEvent]:
        events = []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try JSON Lines
            data = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if isinstance(data, dict):
            data = [data]
        for item in data:
            if not isinstance(item, dict):
                continue
            src = str(item.get(self._src, "")).strip()
            tgt = str(item.get(self._tgt, "")).strip()
            rel = str(item.get(self._rel, "RELATED_TO")).strip() or "RELATED_TO"
            if src and tgt:
                meta = {k: v for k, v in item.items()
                        if k not in (self._src, self._tgt, self._rel)}
                events.append(StreamEvent(source=src, relation=rel, target=tgt,
                                          metadata=meta, ttl=self._ttl))
        return events

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# WebSocketSource — subscribe to a WebSocket feed
# ---------------------------------------------------------------------------

class WebSocketSource(StreamSource):
    """
    Subscribe to a WebSocket endpoint and emit StreamEvents from messages.

    Messages must be JSON objects or JSON arrays of objects with
    ``source``, ``relation``, ``target`` fields.

    Parameters
    ----------
    url          : WebSocket URL (ws:// or wss://)
    headers      : optional connection headers
    source_col   : JSON field for source entity
    target_col   : JSON field for target entity
    relation_col : JSON field for relation type
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        source_col: str = "source",
        target_col: str = "target",
        relation_col: str = "relation",
        ttl: float = 0.0,
    ):
        self._url = url
        self._headers = headers or {}
        self._src = source_col
        self._tgt = target_col
        self._rel = relation_col
        self._ttl = ttl
        self._running = True
        self._queue: queue.Queue = queue.Queue()

    def read(self) -> Iterator[StreamEvent]:
        """Runs the WebSocket client in a background thread and yields from a queue."""
        thread = threading.Thread(target=self._ws_thread, daemon=True)
        thread.start()
        while self._running or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.1)
                yield event
            except queue.Empty:
                continue

    def _ws_thread(self) -> None:
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client is required for WebSocketSource: pip install websocket-client"
            )

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    src = str(item.get(self._src, "")).strip()
                    tgt = str(item.get(self._tgt, "")).strip()
                    rel = str(item.get(self._rel, "RELATED_TO")).strip() or "RELATED_TO"
                    if src and tgt:
                        meta = {k: v for k, v in item.items()
                                if k not in (self._src, self._tgt, self._rel)}
                        self._queue.put(StreamEvent(source=src, relation=rel, target=tgt,
                                                    metadata=meta, ttl=self._ttl))
            except Exception as e:
                logger.warning("WebSocketSource message parse error: %s", e)

        def on_error(ws, error):
            logger.error("WebSocketSource error: %s", error)

        def on_close(ws, *args):
            self._running = False

        ws = websocket.WebSocketApp(
            self._url,
            header=self._headers,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# MQTTSource — subscribe to MQTT topics
# ---------------------------------------------------------------------------

class MQTTSource(StreamSource):
    """
    Subscribe to one or more MQTT topics and emit StreamEvents from messages.

    Requires ``paho-mqtt``: pip install paho-mqtt

    Messages must be JSON objects with ``source``, ``relation``, ``target``
    fields, or plain strings interpreted as ``target`` with the topic as
    ``source`` and "PUBLISHES" as relation.

    Parameters
    ----------
    broker       : MQTT broker host
    port         : MQTT broker port (default 1883)
    topics       : list of topics to subscribe to
    client_id    : MQTT client identifier
    username     : optional MQTT username
    password     : optional MQTT password
    """

    def __init__(
        self,
        broker: str,
        topics: List[str],
        port: int = 1883,
        client_id: str = "parallax_stream",
        username: Optional[str] = None,
        password: Optional[str] = None,
        source_col: str = "source",
        target_col: str = "target",
        relation_col: str = "relation",
        ttl: float = 0.0,
    ):
        self._broker = broker
        self._port = port
        self._topics = topics
        self._client_id = client_id
        self._username = username
        self._password = password
        self._src = source_col
        self._tgt = target_col
        self._rel = relation_col
        self._ttl = ttl
        self._running = True
        self._queue: queue.Queue = queue.Queue()

    def read(self) -> Iterator[StreamEvent]:
        thread = threading.Thread(target=self._mqtt_thread, daemon=True)
        thread.start()
        while self._running or not self._queue.empty():
            try:
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def _mqtt_thread(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("paho-mqtt is required for MQTTSource: pip install paho-mqtt")

        def on_message(client, userdata, msg):
            payload = msg.payload.decode("utf-8", errors="replace")
            try:
                data = json.loads(payload)
                src = str(data.get(self._src, msg.topic)).strip()
                tgt = str(data.get(self._tgt, payload)).strip()
                rel = str(data.get(self._rel, "PUBLISHES")).strip() or "PUBLISHES"
                meta = {k: v for k, v in data.items()
                        if k not in (self._src, self._tgt, self._rel)}
            except json.JSONDecodeError:
                src = msg.topic
                rel = "PUBLISHES"
                tgt = payload.strip()
                meta = {}
            if src and tgt:
                self._queue.put(StreamEvent(source=src, relation=rel, target=tgt,
                                            metadata=meta, ttl=self._ttl))

        client = mqtt.Client(client_id=self._client_id)
        if self._username:
            client.username_pw_set(self._username, self._password)
        client.on_message = on_message
        client.connect(self._broker, self._port)
        for topic in self._topics:
            client.subscribe(topic)
        client.loop_forever()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# StreamAdapter — the live, mutable graph adapter
# ---------------------------------------------------------------------------

class StreamAdapter(NetworkXAdapter):
    """
    A thread-safe, live-mutable graph adapter backed by a sliding-window
    event buffer and incremental community detection.

    Extends NetworkXAdapter — the full Parallax reasoning stack (CSAEngine,
    BeamTraversal, etc.) works with StreamAdapter unchanged.

    Parameters
    ----------
    time_window_seconds     : evict edges older than this many seconds (0 = no eviction)
    max_edges               : hard cap on total live edges (0 = no cap)
    neighborhood_radius     : ego-network radius for incremental DSCF re-runs
    min_events_before_update: batch N events before triggering community update
    directed                : use DiGraph (True) or Graph (False)
    on_event_callbacks      : list of callables invoked on every ingested event
    """

    def __init__(
        self,
        time_window_seconds: float = 60.0,
        max_edges: int = 10_000,
        neighborhood_radius: int = 2,
        min_events_before_update: int = 10,
        directed: bool = True,
        on_event_callbacks: Optional[List[Callable[[StreamEvent], None]]] = None,
    ):
        G = nx.DiGraph() if directed else nx.Graph()
        super().__init__(G)

        self._lock = threading.RLock()
        self._buffer = SlidingWindowBuffer(
            time_window_seconds=time_window_seconds,
            max_edges=max_edges,
        )
        self._updater = IncrementalCommunityUpdater(
            neighborhood_radius=neighborhood_radius,
            min_events_before_update=min_events_before_update,
        )
        self.stats = StreamStats()
        self._sources: List[StreamSource] = []
        self._source_threads: List[threading.Thread] = []
        self._running = False
        self._on_event: List[Callable[[StreamEvent], None]] = on_event_callbacks or []
        self._mutation_listeners: List[Callable[[str, StreamEvent], None]] = []

        # Public: CSA engine reference, set by Studio/API after initial setup
        self.csa_engine = None
        self.community_map: Dict[str, int] = {}
        self.embeddings: Dict[str, Any] = {}

    # -- Graph mutation methods ----------------------------------------------

    def ingest(self, event: StreamEvent) -> None:
        """
        Ingest a single StreamEvent into the live graph.

        Thread-safe. Adds the edge, evicts stale edges, schedules an
        incremental community update if the update threshold is met.
        """
        with self._lock:
            # Add edge to graph
            self._G.add_edge(
                event.source,
                event.target,
                relation=event.relation,
                timestamp=event.timestamp,
                **event.metadata,
            )
            # Invalidate n-gram index for new nodes
            for node in (event.source, event.target):
                if node not in (self._ngram_index if hasattr(self, "_ngram_index") else {}):
                    if hasattr(self, "_ngram_index"):
                        self._invalidate_ngram(node)

            # Track in sliding window
            evicted = self._buffer.push(event)
            if evicted:
                self._evict_from_graph(evicted)
                self.stats.record_eviction(len(evicted))

            # Mark affected nodes for incremental community update
            self._updater.mark_affected([event.source, event.target])
            self.stats.record_event()

            # Trigger community update if threshold met
            if self._updater.should_update() and self.community_map is not None:
                self.community_map = self._updater.run(self._G, self.community_map)
                self.stats.record_community_update()

        # Fire callbacks outside the lock
        for cb in self._on_event:
            try:
                cb(event)
            except Exception as e:
                logger.warning("on_event callback error: %s", e)

        for listener in self._mutation_listeners:
            try:
                listener("add", event)
            except Exception as e:
                logger.warning("mutation listener error: %s", e)

    def ingest_batch(self, events: List[StreamEvent]) -> None:
        """Ingest a batch of events efficiently under a single lock acquisition."""
        with self._lock:
            for ev in events:
                self._G.add_edge(
                    ev.source, ev.target,
                    relation=ev.relation,
                    timestamp=ev.timestamp,
                    **ev.metadata,
                )
                evicted = self._buffer.push(ev)
                if evicted:
                    self._evict_from_graph(evicted)
                    self.stats.record_eviction(len(evicted))
                self._updater.mark_affected([ev.source, ev.target])
                self.stats.record_event()

            if self._updater.should_update() and self.community_map is not None:
                self.community_map = self._updater.run(self._G, self.community_map)
                self.stats.record_community_update()

        for ev in events:
            for cb in self._on_event:
                try:
                    cb(ev)
                except Exception:
                    pass

    def force_community_update(self, resolution: float = 1.0) -> None:
        """Force a full DSCF re-run over the entire live graph."""
        with self._lock:
            self.community_map = self._updater.run_full(self._G, resolution=resolution)
            self.stats.record_community_update()

    def add_mutation_listener(
        self, callback: Callable[[str, StreamEvent], None]
    ) -> None:
        """
        Register a callback invoked on every graph mutation.
        callback(action, event) where action is "add" or "remove".
        Used by the API SSE endpoint.
        """
        self._mutation_listeners.append(callback)

    def remove_mutation_listener(
        self, callback: Callable[[str, StreamEvent], None]
    ) -> None:
        self._mutation_listeners = [l for l in self._mutation_listeners if l is not callback]

    # -- Thread-safe GraphAdapter overrides ---------------------------------

    def get_neighbors(self, entity_id, edge_types=None, max_neighbors=50):
        with self._lock:
            return super().get_neighbors(entity_id, edge_types, max_neighbors)

    def find_entities(self, query, top_k=10):
        with self._lock:
            return super().find_entities(query, top_k)

    def to_networkx(self):
        with self._lock:
            return self._G

    # -- Source management --------------------------------------------------

    def add_source(self, source: StreamSource) -> None:
        """Register a StreamSource. Sources are started when ``start()`` is called."""
        self._sources.append(source)

    def start(self) -> None:
        """
        Start all registered sources in background daemon threads.
        Returns immediately; ingestion runs concurrently.
        """
        if self._running:
            return
        self._running = True
        for source in self._sources:
            t = threading.Thread(
                target=self._ingestion_loop,
                args=(source,),
                daemon=True,
                name=f"parallax-stream-{source.__class__.__name__}",
            )
            t.start()
            self._source_threads.append(t)
            logger.info("Started stream source: %s", source)

    def stop(self) -> None:
        """Signal all sources to stop. Ingestion threads exit gracefully."""
        self._running = False
        for source in self._sources:
            source.stop()
        logger.info("Stream stopped. Stats: %s", self.stats.to_dict())

    # -- Internal -----------------------------------------------------------

    def _ingestion_loop(self, source: StreamSource) -> None:
        try:
            for event in source.read():
                if not self._running:
                    break
                self.ingest(event)
        except Exception as e:
            logger.error("Ingestion loop error (%s): %s", source, e, exc_info=True)

    def _evict_from_graph(self, evicted: List[StreamEvent]) -> None:
        """Remove graph edges whose window entries were evicted."""
        for ev in evicted:
            stale = self._buffer.stale_edges()
            if ev.edge_key() in stale:
                if self._G.has_edge(ev.source, ev.target):
                    self._G.remove_edge(ev.source, ev.target)
                    # Remove isolated nodes
                    for node in (ev.source, ev.target):
                        if node in self._G and self._G.degree(node) == 0:
                            self._G.remove_node(node)
                            self.community_map.pop(node, None)
                    self._updater.mark_affected([ev.source, ev.target])
                    for listener in self._mutation_listeners:
                        try:
                            listener("remove", ev)
                        except Exception:
                            pass

    def _invalidate_ngram(self, node: str) -> None:
        """Add a new node to the n-gram index without full rebuild."""
        if not hasattr(self, "_ngram_index"):
            return
        e = self.get_entity(node)
        text = (e.label + " " + node).lower() if e else node.lower()
        ngrams = set(self._get_ngrams(text))
        self._node_ngram_len[node] = len(ngrams)
        for ng in ngrams:
            self._ngram_index.setdefault(ng, []).append(node)

    # -- Status helpers -----------------------------------------------------

    def live_stats(self) -> Dict[str, Any]:
        """Return a snapshot of current graph and stream statistics."""
        with self._lock:
            n_nodes = self._G.number_of_nodes()
            n_edges = self._G.number_of_edges()
            n_communities = len(set(self.community_map.values())) if self.community_map else 0
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "communities": n_communities,
            "buffer_size": len(self._buffer),
            "sources": len(self._sources),
            "running": self._running,
            **self.stats.to_dict(),
        }

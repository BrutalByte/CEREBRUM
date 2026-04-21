import asyncio
import json
import logging
import websockets
from typing import List, Optional, Callable, Awaitable, Any
from core.telemetry import NeuralEvent, NeuralCommand

log = logging.getLogger("cerebrum.telemetry")

class TelemetryBridge:
    """
    WebSocket bridge that forwards CEREBRUM neural events 
    to a connected Unreal Engine visualization client, 
    and handles incoming NeuralCommands (Phase 90).
    """
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.subscribers: List[Callable[[NeuralEvent], Any]] = []
        self.command_handler: Optional[Callable[[NeuralCommand], Awaitable[None]]] = None
        self._queues: List[asyncio.Queue] = []  # FastAPI WebSocket subscribers

    def subscribe_queue(self) -> asyncio.Queue:
        """Create and register an async queue fed by every broadcast() call."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._queues.append(q)
        return q

    def unsubscribe_queue(self, q: asyncio.Queue) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass


    def add_subscriber(self, callback: Callable[[NeuralEvent], Any]):
        """Register an in-process callback to receive all neural events."""
        self.subscribers.append(callback)

    async def _handle_connection(self, websocket):
        self.clients.add(websocket)
        log.info(f"Telemetry client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                if self.command_handler:
                    try:
                        data = json.loads(message)
                        cmd = NeuralCommand(**data)
                        # We schedule the handler as a task to avoid blocking the socket loop
                        asyncio.create_task(self.command_handler(cmd))
                    except Exception as e:
                        log.error(f"Error parsing incoming NeuralCommand: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            log.info(f"Telemetry client disconnected: {websocket.remote_address}")

    def broadcast(self, event: NeuralEvent):
        """Notifies all subscribers and WebSocket clients of a neural event."""
        # 1. Notify in-process subscribers (Phase 92)
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                log.error(f"Error in telemetry subscriber: {e}")

        # 2. Notify WebSocket clients + FastAPI queue subscribers
        try:
            message = event.model_dump_json()
        except AttributeError:
            message = event.json()

        # Push to FastAPI queue subscribers (non-blocking)
        for q in list(self._queues):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass

        # Push to legacy websockets clients (UE5 / external)
        live_clients = set(self.clients)
        if live_clients:
            asyncio.ensure_future(
                asyncio.gather(
                    *[client.send(message) for client in live_clients],
                    return_exceptions=True,
                )
            )

    async def start_server(self):
        log.info(f"Starting Telemetry Server on {self.host}:{self.port}")
        async with websockets.serve(self._handle_connection, self.host, self.port):
            await asyncio.Future()  # run forever

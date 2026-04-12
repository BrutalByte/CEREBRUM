import asyncio
import json
import websockets
from core.cerebrum import CerebrumGraph
from core.telemetry import NeuralEvent

class TelemetryBridge:
    """
    WebSocket bridge that forwards CEREBRUM neural events 
    to a connected Unreal Engine visualization client.
    """
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()

    async def _handle_connection(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                pass # Client-to-server messages not implemented yet
        finally:
            self.clients.remove(websocket)

    def broadcast(self, event: NeuralEvent):
        """Serializes event to JSON and sends to all UE clients."""
        if not self.clients:
            return
        
        message = event.json()
        asyncio.create_task(
            asyncio.gather(*[client.send(message) for client in self.clients])
        )

    async def start_server(self):
        async with websockets.serve(self._handle_connection, self.host, self.port):
            await asyncio.Future()  # run forever

# Usage example:
# bridge = TelemetryBridge()
# graph.subscribe(bridge.broadcast)
# asyncio.create_task(bridge.start_server())

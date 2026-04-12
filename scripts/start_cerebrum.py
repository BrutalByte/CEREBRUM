import asyncio
import argparse
from api.server import create_app
from api.telemetry_bridge import TelemetryBridge
import uvicorn
import threading

def run_fastapi(port):
    # For standalone orchestration, we load a dummy graph or expect 
    # dynamic loading. In production, we'd load via /reload or init_app
    app = create_app() 
    uvicorn.run(app, host="0.0.0.0", port=port)

async def run_telemetry(port):
    bridge = TelemetryBridge(port=port)
    print(f"Telemetry Bridge started on port {port}")
    await bridge.start_server()

def main():
    parser = argparse.ArgumentParser(description="CEREBRUM Launch Orchestrator")
    parser.add_argument("--api-port", type=int, default=8200)
    parser.add_argument("--ws-port", type=int, default=8765)
    args = parser.parse_args()

    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, args=(args.api_port,), daemon=True)
    api_thread.start()

    # Start Telemetry Bridge in the main thread (async loop)
    try:
        asyncio.run(run_telemetry(args.ws_port))
    except KeyboardInterrupt:
        print("Shutting down CEREBRUM Orchestrator...")

if __name__ == "__main__":
    main()

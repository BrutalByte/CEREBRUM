"""
Scene Initializer for CEREBRUM UE5 Visualizer (Phase 92).

Uses the ue-llm-toolkit to prepare the environment for neural visualization.
"""
import os
import requests
import time

def init_scene(toolkit_url="http://localhost:3000/api/v1"):
    print(f"Connecting to UE LLM Toolkit at {toolkit_url}...")
    
    def run(tool, args):
        try:
            resp = requests.post(f"{toolkit_url}/toolkit/execute", json={"tool": tool, "args": args}, timeout=5)
            return resp.json()
        except Exception as e:
            print(f"Error: {e}")
            return None

    # 1. Setup Global Environment
    print("Initializing environment...")
    run("spawn_actor", {
        "actor_class": "PostProcessVolume",
        "name": "PostProcessVolume_Global",
        "properties": {
            "bUnbound": True,
            "Settings": {
                "BloomIntensity": 1.0,
                "BloomThreshold": 0.5,
                "VignetteIntensity": 0.4
            }
        }
    })

    # 2. Spawn the Main Brain Orchestrator
    print("Spawning CerebrumBrain orchestrator...")
    run("spawn_actor", {
        "actor_class": "ACerebrumBrain",
        "name": "CerebrumBrain_Main",
        "location": {"x": 0, "y": 0, "z": 0}
    })

    # 3. Configure Lighting for "Neural Void" Aesthetic
    print("Setting lighting...")
    run("set_property", {
        "actor": "DirectionalLight",
        "property": "Intensity",
        "value": 0.0
    })
    run("spawn_actor", {
        "actor_class": "SkyLight",
        "name": "SkyLight_Neural",
        "properties": {
            "Intensity": 0.2,
            "bRealTimeCapture": True
        }
    })

    print("Interface setup complete. CEREBRUM is ready to animate.")

if __name__ == "__main__":
    url = os.getenv("CEREBRUM_UE_TOOLKIT_URL", "http://localhost:3000/api/v1")
    init_scene(url)

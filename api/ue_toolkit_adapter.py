"""
UE LLM Toolkit Adapter for CEREBRUM (Phase 92).

Translates CEREBRUM NeuralEvents into REST commands for the 
ue-llm-toolkit (https://github.com/ColtonWilley/ue-llm-toolkit).
"""
import logging
import requests
import json
from typing import Dict, Any, Optional
from core.telemetry import NeuralEvent, NeuralEventType

log = logging.getLogger("cerebrum.ue_toolkit")

class UEToolkitClient:
    """Client for communicating with the ue-llm-toolkit REST API (Port 3000)."""
    def __init__(self, base_url: str = "http://localhost:3000/api/v1"):
        self.base_url = base_url

    def send_command(self, tool: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sends a JSON command to the UE5 editor."""
        try:
            resp = requests.post(
                f"{self.base_url}/toolkit/execute",
                json={"tool": tool, "args": args},
                timeout=2
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning(f"UE Toolkit Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            log.error(f"Failed to connect to UE Toolkit at {self.base_url}: {e}")
        return None

class NeuralToToolkitTranslator:
    """Maps CEREBRUM events to UE5 editor/runtime actions."""
    def __init__(self, client: UEToolkitClient):
        self.client = client
        self._node_actor_map: Dict[str, str] = {} # node_id -> UE5 Actor Name


    def initialize_scene(self):
        """Performs the initial setup of the UE5 environment."""
        log.info("Initializing UE5 Neural Interface...")
        
        # 1. Global Effects
        self.client.send_command("spawn_actor", {
            "actor_class": "PostProcessVolume",
            "name": "PostProcessVolume_Global",
            "properties": {"bUnbound": True, "Settings": {"BloomIntensity": 1.5}}
        })

        # 2. Main Brain Orchestrator
        self.client.send_command("spawn_actor", {
            "actor_class": "ACerebrumBrain",
            "name": "CerebrumBrain_Main",
            "location": {"x": 0, "y": 0, "z": 0}
        })

        # 3. Aesthetics
        self.client.send_command("set_property", {
            "actor": "DirectionalLight",
            "property": "Intensity",
            "value": 0.0
        })
        log.info("UE5 Neural Interface setup complete.")

    def handle_event(self, event: NeuralEvent):
        """Translates a single NeuralEvent into one or more UE commands."""
        etype = event.event_type
        payload = event.payload

        if etype == NeuralEventType.NEUROGENESIS:
            self._spawn_neuron(payload)
        elif etype == NeuralEventType.SYNAPTIC_PULSE:
            self._animate_pulse(payload)
        elif etype == NeuralEventType.METABOLIC_FLUX:
            self._update_world_metabolism(payload)
        elif etype == NeuralEventType.SYNAPTOGENESIS:
            self._create_synapse(payload)

    def _spawn_neuron(self, payload: Dict[str, Any]):
        node_id = payload.get("node_id")
        label = payload.get("label", node_id)
        
        # Args for spawn_actor (toolkit convention)
        args = {
            "actor_class": "BP_Neuron",
            "name": f"Neuron_{node_id}",
            "location": {"x": 0, "y": 0, "z": 0}, # To be refined by layout engine
            "properties": {
                "NodeID": node_id,
                "NodeLabel": label
            }
        }
        res = self.client.send_command("spawn_actor", args)
        if res and "actor_name" in res:
            self._node_actor_map[node_id] = res["actor_name"]

    def _animate_pulse(self, payload: Dict[str, Any]):
        source = payload.get("source_node")
        target = payload.get("target_node")
        weight = payload.get("weight", 1.0)

        # 1. Glow the source
        if source in self._node_actor_map:
            self.client.send_command("set_property", {
                "actor": self._node_actor_map[source],
                "property": "EmissiveIntensity",
                "value": 10.0 * weight
            })
            # 2. Trigger Blueprint event for Niagara FX
            self.client.send_command("blueprint_modify", {
                "actor": self._node_actor_map[source],
                "action": "call_function",
                "function": "OnPulse",
                "args": {"TargetID": target}
            })

    def _update_world_metabolism(self, payload: Dict[str, Any]):
        state = payload.get("state", {})
        arousal = state.get("arousal", 1.0)
        
        # Map arousal to global Bloom intensity
        self.client.send_command("set_property", {
            "actor": "PostProcessVolume_Global",
            "property": "Settings.BloomIntensity",
            "value": arousal * 2.0
        })

    def _create_synapse(self, payload: Dict[str, Any]):
        # Implementation for drawing a 3D line/mesh between two neurons
        pass

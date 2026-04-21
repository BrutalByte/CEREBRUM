"""
ChemicalModulator (Phase 68).

Simulates metabolic modulation of the reasoning engine using 
dynamic scalars (Reinforcement, Arousal, etc.) that decay to baseline 
(Homeostasis) and drive reasoning state adaptation.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("cerebrum.modulator")

class ChemicalModulator:
    """
    Manages global reasoning state based on metabolic scalars.
    
    Functional Mappings:
      - Reinforcement: Reward signal / Confidence Gain (Meta: Dopamine)
      - Arousal: Focus / Search breadth (Meta: Norepinephrine)
      - Novelty: Sensory Input vs Prior (Meta: Acetylcholine)
      - Cohesion: Structural Trust / Community binding (Meta: Oxytocin)
      - Persistence: Memory Persistence / Engram steer (Meta: Vasopressin)
    """
    def __init__(self, baseline: float = 1.0, decay_rate: float = 0.048):
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.state = {
            "reinforcement": baseline,
            "arousal":       baseline,
            "novelty":       baseline,
            "cohesion":      baseline,
            "persistence":   baseline
        }
        
    def step(self):
        """Naturally decay scalars back to baseline (Homeostasis)."""
        for k in self.state:
            # Homeostatic pull: current drifts toward baseline
            self.state[k] += (self.baseline - self.state[k]) * self.decay_rate
            
    def update_reinforcement(self, reward_signal: float):
        """Update based on feedback RPE (Reward Prediction Error)."""
        self.state["reinforcement"] = 0.9 * self.state["reinforcement"] + 0.1 * reward_signal
        logger.debug(f"Reinforcement (Reward) adjusted to {self.state['reinforcement']:.3f}")

    def update_arousal(self, dissonance: float):
        """Update based on cognitive dissonance (L2/L3 verification failure)."""
        # Higher dissonance increases arousal (widens beam to find alternative paths)
        self.state["arousal"] = 1.0 + (min(dissonance, 2.0)) 
        logger.debug(f"Arousal adjusted to {self.state['arousal']:.3f}")

    def update_novelty(self, novelty_score: float):
        """Update based on novelty/uncertainty of the query."""
        # High novelty increases focus on semantic data (alpha) over structure (beta)
        self.state["novelty"] = 1.0 + (min(novelty_score, 1.5))
        logger.debug(f"Novelty (Focus) adjusted to {self.state['novelty']:.3f}")

    def update_cohesion(self, stability: float):
        """Link Cohesion to community cohesion/stability metrics."""
        self.state["cohesion"] = max(0.5, min(2.0, stability))
        logger.debug(f"Cohesion (Binding) adjusted to {self.state['cohesion']:.3f}")

    def update_persistence(self, validation_success: float):
        """Update based on successful L3 Gold Standard validation."""
        # Increases persistence of relation patterns in Engram
        self.state["persistence"] = 1.0 + (min(validation_success, 1.0))
        logger.debug(f"Persistence adjusted to {self.state['persistence']:.3f}")

    def modulate_params(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust CSAEngine parameters based on current metabolic state.
        
        Logic:
        - alpha (semantic): Boosted by Reinforcement and Novelty.
        - beta (community): Boosted by Cohesion, suppressed by Novelty.
        - gamma (edge weight): Boosted by Reinforcement.
        """
        p = base_params.copy()
        
        # Novelty: Shift focus from structure (beta) to input (alpha)
        nov = self.state["novelty"]
        p["alpha"] *= nov
        p["beta"]  /= max(0.1, nov)
        
        # Cohesion: Boost trust in community structure
        p["beta"] *= self.state["cohesion"]
        
        # Reinforcement: Boost confidence gain on successful signals
        re = self.state["reinforcement"]
        p["alpha"] *= re
        p["gamma"] *= re
        
        return p

    def modulate_traversal(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply active modulation to traversal configuration (Beam, Engram)."""
        # Arousal scales beam width
        ar = self.state["arousal"]
        if ar > 1.0:
            config["beam_width"] = int(config.get("beam_width", 10) * ar)
        
        # Persistence scales Engram steer strength
        pe = self.state["persistence"]
        if pe > 1.0 and "engram_strength" in config:
            config["engram_strength"] *= pe
            
        return config

    def modulate_evolution(self) -> Dict[str, float]:
        """
        Phase 104: Metaplasticity.
        Adjusts the AutonomousResearcher behavior based on metabolic state.
        
        - High Arousal (Norepinephrine): Increases mutation probability (exploration).
        - High Reinforcement (Dopamine): Decreases commit threshold (exploitation).
        """
        return {
            "mutation_rate": 0.1 * self.state["arousal"],
            "commit_threshold_multiplier": 1.0 / max(0.1, self.state["reinforcement"]),
            "sample_size": int(2 * self.state["arousal"])
        }

    def to_state(self) -> Dict[str, float]:
        return self.state.copy()



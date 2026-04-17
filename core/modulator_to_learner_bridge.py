"""
Modulator-to-Learner Bridge (Phase 88).

Couples metabolic scalars (Arousal) from the ChemicalModulator 
to the SGD learning_rate of the MetaParameterLearner.
"""
import logging
from core.chemical_modulator import ChemicalModulator
from core.parameter_learner import MetaParameterLearner

log = logging.getLogger("cerebrum.metabolic_bridge")

class ModulatorToLearnerBridge:
    """
    Synchronizes metabolic state with online parameter optimization.
    
    Logic:
    - High Arousal (dissonance) -> High learning_rate (fast adaptation).
    - Low Arousal (stability) -> Low learning_rate (fine-tuning).
    """
    def __init__(self, modulator: ChemicalModulator, learner: MetaParameterLearner):
        self.modulator = modulator
        self.learner = learner

    def sync(self):
        """
        Map current metabolic state to learner hyper-parameters.
        
        Formula:
          learning_rate_scale = modulator.arousal
        """
        arousal = self.modulator.state.get("arousal", 1.0)
        
        # Scale the learning rate based on arousal
        # arousal = 1.0 (baseline) -> scale = 1.0
        # arousal = 3.0 (high dissonance) -> scale = 3.0 (fast learning)
        self.learner.learning_rate_scale = max(0.1, arousal)
        
        log.debug(f"Metabolic Bridge: learning_rate_scale set to {self.learner.learning_rate_scale:.3f} (arousal={arousal:.3f})")

def step_metabolic_system(modulator: ChemicalModulator, learner: MetaParameterLearner):
    """Convenience function to step homeostasis and sync hyperparameters."""
    modulator.step()
    bridge = ModulatorToLearnerBridge(modulator, learner)
    bridge.sync()

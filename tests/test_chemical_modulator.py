import pytest
from core.chemical_modulator import ChemicalModulator

def test_reinforcement_adaptation():
    modulator = ChemicalModulator()
    initial_re = modulator.state["reinforcement"]
    
    # Simulate high reward
    modulator.update_reinforcement(2.0)
    assert modulator.state["reinforcement"] > initial_re
    
    # Simulate low reward
    modulator.update_reinforcement(0.0)
    assert modulator.state["reinforcement"] < 2.0

def test_arousal_modulation():
    modulator = ChemicalModulator()
    config = {"beam_width": 10}
    
    # Normal state
    new_config = modulator.modulate_traversal(config.copy())
    assert new_config["beam_width"] == 10
    
    # High dissonance state
    modulator.update_arousal(1.0)
    new_config = modulator.modulate_traversal(config.copy())
    # Arousal = 1.0 + min(1.0, 2.0) = 2.0
    # Beam width = 10 * 2.0 = 20
    assert new_config["beam_width"] == 20

def test_novelty_focus():
    modulator = ChemicalModulator()
    params = {"alpha": 0.4, "beta": 0.4, "gamma": 0.1}
    
    # Increase novelty
    modulator.update_novelty(1.0)
    mod_params = modulator.modulate_params(params)
    
    # Novelty > 1.0 boosts alpha, suppresses beta
    assert mod_params["alpha"] > params["alpha"]
    assert mod_params["beta"] < params["beta"]

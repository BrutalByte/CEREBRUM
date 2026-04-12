import pytest
from core.chemical_modulator import ChemicalModulator

def test_homeostatic_decay():
    modulator = ChemicalModulator(baseline=1.0, decay_rate=0.5)
    # Set to a high value
    modulator.state["reinforcement"] = 2.0
    
    # Run a step: 1.0 + (1.0 - 2.0) * 0.5 = 1.5
    modulator.step()
    assert modulator.state["reinforcement"] == 1.5
    
    # Run another step: 1.5 + (1.0 - 1.5) * 0.5 = 1.25
    modulator.step()
    assert modulator.state["reinforcement"] == 1.25

def test_cohesion_update():
    modulator = ChemicalModulator()
    modulator.update_cohesion(1.8)
    assert modulator.state["cohesion"] == 1.8

    # Test clamping
    modulator.update_cohesion(5.0)
    assert modulator.state["cohesion"] == 2.0

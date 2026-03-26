
import pytest
import time
from core.discretizer import STDPDiscretizer

def test_stdp_lazy_decay_mathematical_consistency():
    """
    Verify that lazy decay produces the same weights as 
    synchronous decay would.
    """
    # 1. Setup with high decay
    # Weight = w * (0.5 ^ steps)
    discretizer = STDPDiscretizer(weight_decay=0.5)
    
    # Pair (A, B)
    discretizer.process("A", timestamp=100.0)
    # This spike at 101.0 will potentiate (A, B)
    # Δw = A_plus * exp(-1/20) ≈ 0.1 * 0.9512 = 0.09512
    discretizer.process("B", timestamp=101.0)
    
    w1 = discretizer.weight("A", "B")
    assert w1 > 0
    
    # 2. Skip 2 steps (two unrelated spikes)
    # Each step should apply decay 0.5
    # Total decay: 0.5 * 0.5 = 0.25
    discretizer.process("C", timestamp=102.0)
    discretizer.process("D", timestamp=103.0)
    
    expected_w = w1 * 0.5 * 0.5
    actual_w = discretizer.weight("A", "B")
    
    assert actual_w == pytest.approx(expected_w)
    
def test_stdp_optimized_threshold_check():
    """
    Verify that only potentiated pairs are emitted as CAUSES.
    """
    # Threshold 0.5
    discretizer = STDPDiscretizer(w_threshold=0.5, n_min=1)
    
    # Spike A at 100
    discretizer.process("A", timestamp=100.0)
    
    # Spike B at 101. LTP (A, B).
    # Weight will be ~0.095 (below 0.5 threshold)
    events = discretizer.process("B", timestamp=101.0)
    assert len(events) == 0
    
    # Manually spike many times to cross threshold
    # A_plus is 0.1, tau_plus is 0.2. At dt=1.0, dw = 0.1 * exp(-5) = 0.00067
    # Wait, the default tau_plus is 0.2. At dt=1.0, dw is very small.
    # In my test, dt = 1.0 (101 - 100).
    # So we need many iterations if w_threshold is 0.5.
    for i in range(50):
        discretizer.process("A", timestamp=200.0 + i*10)
        events = discretizer.process("B", timestamp=200.0 + i*10 + 0.1) # small dt=0.1
        if events:
            break
            
    assert len(events) == 1
    assert events[0].source == "A"
    assert events[0].target == "B"
    assert events[0].relation == "CAUSES"

def test_stdp_reset_clears_lazy_state():
    discretizer = STDPDiscretizer()
    discretizer.process("A")
    discretizer.process("B")
    assert discretizer._step > 0
    assert len(discretizer._last_update_step) > 0
    
    discretizer.reset()
    assert discretizer._step == 0
    assert len(discretizer._last_update_step) == 0

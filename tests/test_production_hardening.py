import pytest
from unittest.mock import MagicMock, patch
from core.resource_governor import ResourceGovernor

def test_production_constraints():
    # Setup governor with high limits to ensure it passes on current machine
    governor = ResourceGovernor(memory_threshold_pct=99.0)
    
    # Test CPU/RAM constraint check
    assert governor.check_constraints(cpu_limit_pct=100.0) is True

    # Test forced failure by setting a very low CPU limit
    # (assuming the machine has some background CPU usage)
    assert governor.check_constraints(cpu_limit_pct=-1.0) is False

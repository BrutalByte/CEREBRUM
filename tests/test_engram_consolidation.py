import pytest
from reasoning.engram_traversal import Engram
from reasoning.engram_consolidation import EngramConsolidator

def test_engram_consolidation():
    # Setup mock Engram
    engram = Engram()
    # Mock the counts attribute
    engram._counts = {
        ("DISCOVERED", "PART_OF"): 10,  # Should be promoted
        ("ENEMY_OF",): 2                # Should not be promoted
    }
    
    consolidator = EngramConsolidator(engram, min_success_threshold=5)
    
    promoted = consolidator.consolidate()
    
    assert promoted == 1
    patterns = consolidator.get_canonical_patterns()
    assert ("DISCOVERED", "PART_OF") in patterns
    assert ("ENEMY_OF",) not in patterns

def test_no_double_promotion():
    engram = Engram()
    engram._counts = {("DISCOVERED",): 10}
    consolidator = EngramConsolidator(engram, min_success_threshold=5)
    
    consolidator.consolidate()
    promoted = consolidator.consolidate()
    
    assert promoted == 0

import pytest
from core.cerebellar_engine import CerebellarEngine, DissonanceEvent
from reasoning.answer_extractor import Answer
from reasoning.traversal import TraversalPath

class MockResearchAgent:
    def __init__(self):
        self.pushed = []
    def push_candidate(self, candidate):
        self.pushed.append(candidate)

def test_dissonance_detection():
    # 1. Setup mock components
    agent = MockResearchAgent()
    ce = CerebellarEngine(research_agent=agent, dissonance_threshold=0.3, min_path_score=0.5)

    # 2. Create answers
    # Case A: Normal (High path score, High consensus)
    ans_normal = Answer(
        entity_id="target_1",
        score=0.8,
        path_score=0.8,
        consensus_score=0.7,
        best_path=TraversalPath(nodes=["seed", "rel1", "target_1"])
    )

    # Case B: Dissonant (High path score, Low consensus)
    ans_dissonant = Answer(
        entity_id="target_2",
        score=0.6,
        path_score=0.8,
        consensus_score=0.2,
        best_path=TraversalPath(nodes=["seed", "rel2", "target_2"])
    )

    # Case C: Low Confidence (Low path score, Low consensus) - should NOT trigger
    ans_low = Answer(
        entity_id="target_3",
        score=0.2,
        path_score=0.3,
        consensus_score=0.1,
        best_path=TraversalPath(nodes=["seed", "rel3", "target_3"])
    )

    # 3. Process results
    events = ce.process_results("seed_1", [ans_normal, ans_dissonant, ans_low])

    # 4. Assertions
    assert len(events) == 1
    assert events[0].target_id == "target_2"
    assert events[0].dissonance == pytest.approx(0.6)
    assert events[0].best_path_relations == ["rel2"]

    # Check if research agent was notified
    assert len(agent.pushed) == 1
    assert agent.pushed[0].source_id == "seed_1"
    assert agent.pushed[0].target_id == "target_2"
    assert agent.pushed[0].seeded_by == "cerebellar_error"

def test_engine_initialization_defaults():
    ce = CerebellarEngine()
    assert ce.dissonance_threshold == 0.35
    assert ce.min_path_score == 0.50
    assert ce.research_agent is None
    assert ce.meta_learner is None

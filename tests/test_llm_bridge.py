"""
Phase 3 tests for llm_bridge/context_formatter.py.

Tests the two public functions:
  - to_prompt()      — formats answers as a natural-language LLM prompt string
  - to_structured()  — formats answers as a structured dict for JSON/API use

Both functions are pure (no I/O, no state). Tests use synthetic Answer/
TraversalPath objects rather than running the full pipeline.
"""
import random
from pathlib import Path

import pytest

from reasoning.traversal import TraversalPath
from reasoning.answer_extractor import Answer
from llm_bridge.context_formatter import to_prompt, to_structured


# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

def make_answer(
    entity_id: str,
    score: float,
    nodes: list,
    community_sequence: list,
    rank: int = 1,
) -> Answer:
    """Build a minimal Answer object without running the pipeline."""
    path = TraversalPath(
        nodes=nodes,
        score=score,
        attention_weights=[0.7] * ((len(nodes) - 1) // 2),
        community_sequence=community_sequence,
    )
    return Answer(
        entity_id=entity_id,
        score=score,
        best_path=path,
        score_breakdown={"attention": 0.49, "community": 0.9},
        community_trace=community_sequence,
    )


ANSWER_A = make_answer(
    entity_id="einstein",
    score=0.85,
    nodes=["newton", "INFLUENCED", "einstein"],
    community_sequence=[0, 0],
)

ANSWER_B = make_answer(
    entity_id="faraday",
    score=0.72,
    nodes=["newton", "INFLUENCED", "faraday"],
    community_sequence=[0, 0],
)

ANSWER_MULTI = make_answer(
    entity_id="bohr",
    score=0.61,
    nodes=["newton", "INFLUENCED", "einstein", "COLLABORATED", "bohr"],
    community_sequence=[0, 0, 1],
)


# ---------------------------------------------------------------------------
# to_prompt
# ---------------------------------------------------------------------------

class TestToPrompt:

    def test_returns_string(self):
        result = to_prompt([ANSWER_A], query="Who did Newton influence?")
        assert isinstance(result, str)

    def test_contains_query(self):
        query  = "Who did Newton influence?"
        result = to_prompt([ANSWER_A], query=query)
        assert query in result

    def test_contains_answer_entity(self):
        result = to_prompt([ANSWER_A], query="test")
        assert "einstein" in result

    def test_contains_score(self):
        result = to_prompt([ANSWER_A], query="test")
        assert "0.8500" in result

    def test_contains_path_nodes(self):
        """Both entity and relation labels from the path must appear."""
        result = to_prompt([ANSWER_A], query="test")
        assert "newton" in result
        assert "INFLUENCED" in result

    def test_contains_community_ids(self):
        """Community IDs must be shown for each entity node."""
        result = to_prompt([ANSWER_A], query="test")
        assert "COMMUNITY" in result

    def test_contains_score_breakdown(self):
        result = to_prompt([ANSWER_A], query="test")
        assert "attention" in result

    def test_default_instruction_present(self):
        result = to_prompt([ANSWER_A], query="test")
        assert "Summarize" in result

    def test_custom_instruction_overrides_default(self):
        custom = "Explain this in German."
        result = to_prompt([ANSWER_A], query="test", instruction=custom)
        assert custom in result
        assert "Summarize" not in result

    def test_max_paths_limits_output(self):
        """With max_paths=1, only the first answer must appear."""
        result = to_prompt([ANSWER_A, ANSWER_B], query="test", max_paths=1)
        assert "einstein" in result
        assert "faraday" not in result

    def test_multiple_paths_numbered(self):
        """Multiple answers must produce Path 1 and Path 2 labels."""
        result = to_prompt([ANSWER_A, ANSWER_B], query="test")
        assert "Path 1" in result
        assert "Path 2" in result

    def test_multi_hop_path_includes_all_nodes(self):
        """A 2-hop path must include all three entities in the output."""
        result = to_prompt([ANSWER_MULTI], query="test")
        assert "newton" in result
        assert "einstein" in result
        assert "bohr" in result

    def test_empty_answers_returns_valid_string(self):
        result = to_prompt([], query="empty test")
        assert isinstance(result, str)
        assert "empty test" in result


# ---------------------------------------------------------------------------
# to_structured
# ---------------------------------------------------------------------------

class TestToStructured:

    def test_returns_dict(self):
        result = to_structured([ANSWER_A], query="test")
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = to_structured([ANSWER_A], query="test")
        assert "query" in result
        assert "paths" in result

    def test_query_echoed(self):
        result = to_structured([ANSWER_A], query="Who influenced Einstein?")
        assert result["query"] == "Who influenced Einstein?"

    def test_paths_is_list(self):
        result = to_structured([ANSWER_A, ANSWER_B], query="test")
        assert isinstance(result["paths"], list)
        assert len(result["paths"]) == 2

    def test_path_has_rank(self):
        result = to_structured([ANSWER_A], query="test")
        assert result["paths"][0]["rank"] == 1

    def test_ranks_are_sequential(self):
        result = to_structured([ANSWER_A, ANSWER_B, ANSWER_MULTI], query="test")
        ranks  = [p["rank"] for p in result["paths"]]
        assert ranks == [1, 2, 3]

    def test_path_has_answer_entity(self):
        result = to_structured([ANSWER_A], query="test")
        assert result["paths"][0]["answer_entity"] == "einstein"

    def test_path_has_score(self):
        result = to_structured([ANSWER_A], query="test")
        assert result["paths"][0]["score"] == 0.85

    def test_path_has_score_breakdown(self):
        result = to_structured([ANSWER_A], query="test")
        breakdown = result["paths"][0]["score_breakdown"]
        assert "attention" in breakdown
        assert "community" in breakdown

    def test_path_nodes_structure(self):
        """Each path node must have 'type' and 'label' fields."""
        result = to_structured([ANSWER_A], query="test")
        for node in result["paths"][0]["path"]:
            assert "type" in node
            assert "label" in node

    def test_path_nodes_alternate_types(self):
        """Even-index nodes are entities, odd-index nodes are relations."""
        result = to_structured([ANSWER_A], query="test")
        nodes  = result["paths"][0]["path"]
        for i, node in enumerate(nodes):
            if i % 2 == 0:
                assert node["type"] == "entity"
            else:
                assert node["type"] == "relation"

    def test_entity_nodes_have_id_and_community(self):
        """Entity nodes must carry 'id' and 'community' fields."""
        result = to_structured([ANSWER_A], query="test")
        for node in result["paths"][0]["path"]:
            if node["type"] == "entity":
                assert "id" in node
                assert "community" in node

    def test_multi_hop_path_node_count(self):
        """2-hop path: newton-INFLUENCED-einstein-COLLABORATED-bohr = 5 nodes."""
        result = to_structured([ANSWER_MULTI], query="test")
        nodes  = result["paths"][0]["path"]
        assert len(nodes) == 5

    def test_empty_answers_returns_empty_paths(self):
        result = to_structured([], query="test")
        assert result["paths"] == []




"""
Tests for llm_bridge — context_formatter and adapters.

Covers:
  - to_prompt()      — formats answers as a natural-language LLM prompt string
  - to_structured()  — formats answers as a structured dict for JSON/API use
  - generate()       — end-to-end: format + call llm_fn + return GenerationResult
  - Adapter wrappers — AnthropicAdapter, OpenAIAdapter, OllamaAdapter,
                       HuggingFaceAdapter (all tested with mocks, no real API calls)

All tests use synthetic Answer/TraversalPath objects.
No external API calls are made.
"""
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reasoning.traversal import TraversalPath
from reasoning.answer_extractor import Answer
from llm_bridge.context_formatter import to_prompt, to_structured, generate, GenerationResult


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


# ---------------------------------------------------------------------------
# generate() — core function
# ---------------------------------------------------------------------------

class TestGenerate:

    def test_returns_generation_result(self):
        result = generate([ANSWER_A], query="test", llm_fn=lambda p: "response")
        assert isinstance(result, GenerationResult)

    def test_response_is_llm_fn_output(self):
        result = generate([ANSWER_A], query="test", llm_fn=lambda p: "Hello world")
        assert result.response == "Hello world"

    def test_llm_fn_receives_prompt(self):
        captured = {}
        def _llm(prompt):
            captured["prompt"] = prompt
            return "ok"
        generate([ANSWER_A], query="Who influenced Einstein?", llm_fn=_llm)
        assert "Who influenced Einstein?" in captured["prompt"]

    def test_prompt_stored_in_result(self):
        """Prompt must be in the result for auditability."""
        result = generate([ANSWER_A], query="test query", llm_fn=lambda p: "resp")
        assert "test query" in result.prompt

    def test_query_stored_in_result(self):
        result = generate([ANSWER_A], query="original query", llm_fn=lambda p: "resp")
        assert result.query == "original query"

    def test_paths_used_count(self):
        result = generate([ANSWER_A, ANSWER_B], query="test", llm_fn=lambda p: "resp")
        assert result.paths_used == 2

    def test_paths_used_capped_by_max_paths(self):
        result = generate(
            [ANSWER_A, ANSWER_B, ANSWER_MULTI], query="test",
            llm_fn=lambda p: "resp", max_paths=2
        )
        assert result.paths_used == 2

    def test_source_entities_populated(self):
        result = generate([ANSWER_A, ANSWER_B], query="test", llm_fn=lambda p: "resp")
        assert "einstein" in result.source_entities
        assert "faraday" in result.source_entities

    def test_source_entities_respects_max_paths(self):
        result = generate(
            [ANSWER_A, ANSWER_B], query="test",
            llm_fn=lambda p: "resp", max_paths=1
        )
        assert result.source_entities == ["einstein"]
        assert "faraday" not in result.source_entities

    def test_duration_seconds_non_negative(self):
        result = generate([ANSWER_A], query="test", llm_fn=lambda p: "resp")
        assert result.duration_seconds >= 0.0

    def test_empty_answers_handled(self):
        result = generate([], query="empty test", llm_fn=lambda p: "nothing found")
        assert result.paths_used == 0
        assert result.source_entities == []
        assert result.response == "nothing found"

    def test_custom_instruction_in_prompt(self):
        captured = {}
        def _llm(prompt):
            captured["prompt"] = prompt
            return "ok"
        generate([ANSWER_A], query="test",
                 llm_fn=_llm, instruction="Answer in French.")
        assert "Answer in French." in captured["prompt"]

    def test_plain_lambda_works_as_llm_fn(self):
        """Any callable(str)->str is a valid llm_fn — no adapter required."""
        fn = lambda prompt: f"Processed: {len(prompt)} chars"
        result = generate([ANSWER_A], query="test", llm_fn=fn)
        assert result.response.startswith("Processed:")


# ---------------------------------------------------------------------------
# Adapter wrappers (all mocked — no real API calls)
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:

    def test_calls_client_messages_create(self):
        from llm_bridge.adapters import AnthropicAdapter

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Anthropic response")]
        )
        adapter = AnthropicAdapter(client=mock_client)
        result  = adapter("test prompt")

        assert result == "Anthropic response"
        mock_client.messages.create.assert_called_once()

    def test_passes_model_and_max_tokens(self):
        from llm_bridge.adapters import AnthropicAdapter

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        adapter = AnthropicAdapter(model="claude-opus-4-6", max_tokens=256, client=mock_client)
        adapter("prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert call_kwargs["max_tokens"] == 256

    def test_system_prompt_included_when_set(self):
        from llm_bridge.adapters import AnthropicAdapter

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )
        adapter = AnthropicAdapter(client=mock_client, system="You are a scientist.")
        adapter("prompt")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "You are a scientist."

    def test_repr(self):
        from llm_bridge.adapters import AnthropicAdapter
        mock_client = MagicMock()
        adapter = AnthropicAdapter(model="claude-haiku-4-5-20251001", client=mock_client)
        assert "claude-haiku-4-5-20251001" in repr(adapter)


class TestOpenAIAdapter:

    def test_calls_chat_completions_create(self):
        from llm_bridge.adapters import OpenAIAdapter

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="OpenAI response"))]
        )
        adapter = OpenAIAdapter(client=mock_client)
        result  = adapter("test prompt")

        assert result == "OpenAI response"
        mock_client.chat.completions.create.assert_called_once()

    def test_passes_model_and_max_tokens(self):
        from llm_bridge.adapters import OpenAIAdapter

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        adapter = OpenAIAdapter(model="gpt-4o", max_tokens=128, client=mock_client)
        adapter("prompt")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["max_tokens"] == 128

    def test_system_prompt_in_messages(self):
        from llm_bridge.adapters import OpenAIAdapter

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        adapter = OpenAIAdapter(client=mock_client, system="Be concise.")
        adapter("user prompt")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[1]["role"] == "user"

    def test_repr(self):
        from llm_bridge.adapters import OpenAIAdapter
        mock_client = MagicMock()
        adapter = OpenAIAdapter(model="gpt-4o-mini", client=mock_client)
        assert "gpt-4o-mini" in repr(adapter)


class TestOllamaAdapter:

    def test_posts_to_generate_endpoint(self):
        from llm_bridge.adapters import OllamaAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            adapter = OllamaAdapter(model="llama3.2", url="http://localhost:11434")
            result  = adapter("test prompt")

        assert result == "Ollama response"
        call_args = mock_post.call_args
        assert "api/generate" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "llama3.2"
        assert call_args[1]["json"]["prompt"] == "test prompt"
        assert call_args[1]["json"]["stream"] is False

    def test_repr(self):
        from llm_bridge.adapters import OllamaAdapter
        adapter = OllamaAdapter(model="mistral")
        assert "mistral" in repr(adapter)


class TestHuggingFaceAdapter:

    def test_calls_pipeline_with_prompt(self):
        from llm_bridge.adapters import HuggingFaceAdapter

        mock_pipeline = MagicMock(
            return_value=[{"generated_text": "HF response"}]
        )
        adapter = HuggingFaceAdapter(mock_pipeline)
        result  = adapter("test prompt")

        assert result == "HF response"
        mock_pipeline.assert_called_once()

    def test_strips_input_prompt_from_generated_text(self):
        """text-generation pipelines often echo the input — strip it."""
        from llm_bridge.adapters import HuggingFaceAdapter

        prompt = "Tell me about Newton."
        mock_pipeline = MagicMock(
            return_value=[{"generated_text": prompt + " He was a physicist."}]
        )
        adapter = HuggingFaceAdapter(mock_pipeline)
        result  = adapter(prompt)

        assert result == "He was a physicist."

    def test_handles_summary_text_key(self):
        from llm_bridge.adapters import HuggingFaceAdapter

        mock_pipeline = MagicMock(
            return_value=[{"summary_text": "A brief summary."}]
        )
        adapter = HuggingFaceAdapter(mock_pipeline)
        assert adapter("long text...") == "A brief summary."

    def test_repr(self):
        from llm_bridge.adapters import HuggingFaceAdapter
        mock_pipeline = MagicMock()
        adapter = HuggingFaceAdapter(mock_pipeline)
        assert "HuggingFaceAdapter" in repr(adapter)




"""
LLM bridge: format CEREBRUM traversal output as structured LLM context
and invoke any LLM for natural language generation.

CEREBRUM performs all reasoning. The LLM's role is purely natural language
generation from the grounded, verified paths (Section 10.2).

Usage:
    from llm_bridge.context_formatter import generate
    from llm_bridge.adapters import AnthropicAdapter

    answers = extract(paths, top_k=3)
    result  = generate(answers, query="What did Einstein work on?",
                       llm_fn=AnthropicAdapter())
    print(result.response)          # natural language answer
    print(result.prompt)            # the prompt that was sent (auditable)

Any callable (str) -> str works as llm_fn â€” the adapter wrappers in
llm_bridge/adapters.py are convenience helpers, not requirements.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# GenerationResult â€” output of generate()
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    The output of a generate() call.

    Carries the LLM response alongside the full context that produced it,
    so the caller can always audit what was sent and which paths were used.

    Fields
    ------
    response        : Natural language text returned by the LLM.
    prompt          : The exact prompt string sent to the LLM.
    query           : The original query passed to generate().
    paths_used      : Number of CEREBRUM answer paths included in the prompt.
    source_entities : Entity IDs of the answers that were included.
    duration_seconds: Wall-clock time for the LLM call.
    """
    response:         str
    prompt:           str
    query:            str
    paths_used:       int
    source_entities:  List[str]
    duration_seconds: float = 0.0
    timestamp:        float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# generate() â€” the main entry point
# ---------------------------------------------------------------------------

DEFAULT_INSTRUCTION = (
    "Using only the knowledge graph paths above, answer the query in clear, "
    "concise natural language. Cite the path steps where relevant. "
    "Do not add information that is not present in the paths."
)


def generate(
    answers,                              # List[Answer] from answer_extractor.extract()
    query: str,
    llm_fn: Callable[[str], str],
    max_paths: int = 5,
    instruction: str = DEFAULT_INSTRUCTION,
    adapter=None,                         # Optional GraphAdapter for entity label lookup
) -> GenerationResult:
    """
    Format CEREBRUM answers into a grounded prompt and call an LLM for
    natural language generation.

    Parameters
    ----------
    answers     : List[Answer] from ``reasoning.answer_extractor.extract()``.
    query       : The original query string (included verbatim in the prompt).
    llm_fn      : Any callable with signature ``(prompt: str) -> str``.
                  Use the adapter wrappers in ``llm_bridge.adapters`` for
                  Anthropic, OpenAI, Ollama, or HuggingFace. A plain lambda
                  or function works equally well.
    max_paths   : Maximum number of answer paths to include in the prompt.
                  Fewer paths = shorter prompt = cheaper/faster LLM call.
    instruction : Task instruction appended at the end of the prompt.
                  Override to change tone, language, or output format.
    adapter     : Optional GraphAdapter used to resolve entity IDs to human-
                  readable labels in the prompt. Pass None to use raw IDs.

    Returns
    -------
    GenerationResult with the LLM response, the prompt, and metadata.

    Examples
    --------
    Plain callable (no dependencies):
        result = generate(answers, "Who influenced Einstein?",
                          llm_fn=lambda p: "Isaac Newton influenced Einstein.")

    Anthropic:
        from llm_bridge.adapters import AnthropicAdapter
        result = generate(answers, query, AnthropicAdapter())

    OpenAI:
        from llm_bridge.adapters import OpenAIAdapter
        result = generate(answers, query, OpenAIAdapter("gpt-4o-mini"))

    Ollama:
        from llm_bridge.adapters import OllamaAdapter
        result = generate(answers, query, OllamaAdapter("llama3.2"))
    """
    prompt = to_prompt(
        answers,
        query=query,
        adapter=adapter,
        max_paths=max_paths,
        instruction=instruction,
    )

    t0 = time.time()
    response = llm_fn(prompt)
    duration = time.time() - t0

    used = answers[:max_paths]
    return GenerationResult(
        response=response,
        prompt=prompt,
        query=query,
        paths_used=len(used),
        source_entities=[a.entity_id for a in used],
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# to_prompt
# ---------------------------------------------------------------------------

def to_prompt(
    answers,                # List[Answer] from answer_extractor.extract()
    query: str,
    adapter=None,           # Optional GraphAdapter â€” used to look up entity labels
    max_paths: int = 5,
    instruction: str = "Summarize what this tells us about the query in natural language.",
) -> str:
    """
    Format top-K answers as a structured LLM prompt.

    Produces the format described in Section 10.2:

        You are reasoning about: [query]

        The knowledge graph traversal found these paths:

        Path 1 (score: 0.94):
          Entity_A [COMMUNITY: 0]
          -> [RELATION] ->
          Entity_B [COMMUNITY: 0]
          -> [RELATION] ->
          Entity_C [COMMUNITY: 1]

        [instruction]

    Parameters
    ----------
    answers    : List[Answer] objects
    query      : the original query string
    adapter    : optional adapter for resolving entity labels
    max_paths  : maximum paths to include
    instruction: task instruction for the LLM

    Returns
    -------
    Formatted prompt string
    """
    lines = [
        f"You are reasoning about: {query}",
        "",
        "The knowledge graph traversal found these paths:",
        "",
    ]

    for i, answer in enumerate(answers[:max_paths], 1):
        path  = answer.best_path
        score = answer.score
        lines.append(f"Path {i} (score: {score:.4f}):")

        nodes = path.nodes
        cseq  = path.community_sequence

        entity_idx = 0
        for j, node in enumerate(nodes):
            if j % 2 == 0:
                # Entity node
                cid   = cseq[entity_idx] if entity_idx < len(cseq) else -1
                label = _resolve_label(node, adapter)
                lines.append(f"  {label} [COMMUNITY: {cid}]")
                entity_idx += 1
            else:
                # Relation label
                lines.append(f"  -> [{node}] ->")

        lines.append(f"  Score breakdown: {answer.score_breakdown}")
        lines.append("")

    lines += [instruction, ""]
    return "\n".join(lines)


def to_structured(
    answers,
    query: str,
    adapter=None,
) -> Dict[str, Any]:
    """
    Format answers as a structured dict (for JSON API responses or tool use).

    Returns:
    {
        "query": str,
        "paths": [
            {
                "rank": int,
                "answer_entity": str,
                "score": float,
                "score_breakdown": dict,
                "path": [
                    {"type": "entity", "id": str, "label": str, "community": int},
                    {"type": "relation", "label": str},
                    ...
                ],
            }
        ]
    }
    """
    result_paths = []

    for rank, answer in enumerate(answers, 1):
        path  = answer.best_path
        nodes = path.nodes
        cseq  = path.community_sequence

        path_nodes = []
        entity_idx = 0
        for j, node in enumerate(nodes):
            if j % 2 == 0:
                cid = cseq[entity_idx] if entity_idx < len(cseq) else -1
                path_nodes.append({
                    "type":      "entity",
                    "id":        node,
                    "label":     _resolve_label(node, adapter),
                    "community": cid,
                })
                entity_idx += 1
            else:
                path_nodes.append({"type": "relation", "label": node})

        result_paths.append({
            "rank":            rank,
            "answer_entity":   answer.entity_id,
            "score":           answer.score,
            "score_breakdown": answer.score_breakdown,
            "path":            path_nodes,
        })

    return {"query": query, "paths": result_paths}


def _resolve_label(entity_id: str, adapter=None) -> str:
    """Look up entity label if adapter is available."""
    if adapter is None:
        return entity_id
    try:
        entity = adapter.get_entity(entity_id)
        return entity.label if entity else entity_id
    except Exception:
        return entity_id




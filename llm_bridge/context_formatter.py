"""
LLM bridge: format Parallax traversal output as structured LLM context.

Parallax performs all reasoning. The LLM's role is purely natural language
generation from the grounded, verified paths (Section 10.2).

Usage:
    answers = extract(paths, top_k=3)
    prompt  = to_prompt(answers, query="What did Einstein work on?")
    # Pass prompt to any LLM (OpenAI, Claude, local model)
"""
from typing import List, Dict, Any, Optional


def to_prompt(
    answers,                # List[Answer] from answer_extractor.extract()
    query: str,
    adapter=None,           # Optional GraphAdapter — used to look up entity labels
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

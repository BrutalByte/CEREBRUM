# CEREBRUM LLM Bridge

The LLM Bridge is an optional module that formats CEREBRUM reasoning results as grounded context for Large Language Models, and routes generation requests through any supported LLM backend.

The bridge is a one-way enhancer: CEREBRUM provides the verified graph paths; the LLM provides natural language generation. The LLM never touches the reasoning engine. This separation preserves the zero-hallucination guarantee for the factual claims — only the prose generation is probabilistic.

---

## Installation

```bash
pip install -e ".[llm]"
# or for specific adapters:
pip install anthropic          # Anthropic (Claude)
pip install openai             # OpenAI (GPT)
# Ollama: no pip install needed, just have Ollama running locally
# HuggingFace: pip install transformers torch
```

---

## Quick Start

```python
from reasoning.answer_extractor import extract
from llm_bridge import generate, AnthropicAdapter

# Assume `answers` is the output of extract(traversal.traverse([...]))

result = generate(
    answers=answers,
    query="What did Marie Curie discover?",
    adapter=AnthropicAdapter(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    ),
)

print(result.text)               # LLM-generated natural language response
print(result.grounded_paths)     # The CEREBRUM paths provided as context
print(result.source_entities)    # Entities in the context
```

---

## API Reference

### `generate(answers, query, adapter, max_tokens=512, temperature=0.3) -> GenerationResult`

The primary function. Formats graph paths as a structured prompt and routes to the configured adapter.

**Parameters:**
- `answers: List[AnswerResult]` — output from `extract()`
- `query: str` — the original question (included in the prompt)
- `adapter` — one of the four adapter types (see below)
- `max_tokens: int` — maximum tokens for LLM response
- `temperature: float` — generation temperature (default: 0.3 for factual grounding)

**Returns `GenerationResult`:**
```python
@dataclass
class GenerationResult:
    text: str                         # LLM response text
    grounded_paths: List[str]         # Path strings provided as context
    source_entities: List[str]        # Entity IDs in context
    model: str                        # Model used
    input_tokens: int
    output_tokens: int
```

### `to_prompt(answers, query) -> str`

Formats answers as a plain-text prompt without calling any LLM. Useful for manual inspection or custom LLM integrations.

```python
from llm_bridge import to_prompt

prompt = to_prompt(answers, query="What did Marie Curie discover?")
print(prompt)
# Output:
# Context (from Knowledge Graph):
# - Marie Curie → discovered → Polonium (score: 0.891)
# - Marie Curie → discovered → Radium (score: 0.847)
#   path: Marie Curie → affiliated_with → Paris_University → research_on → Radium
#
# Question: What did Marie Curie discover?
# Answer based only on the context above:
```

### `to_structured(answers, query) -> dict`

Returns the context as a structured dictionary for programmatic use.

---

## Adapters

### AnthropicAdapter (Claude)

```python
from llm_bridge import AnthropicAdapter

adapter = AnthropicAdapter(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-sonnet-4-6",      # or "claude-opus-4-6", "claude-haiku-4-5-20251001"
    max_retries=3,
    timeout=30.0,
)
```

### OpenAIAdapter (GPT)

```python
from llm_bridge import OpenAIAdapter

adapter = OpenAIAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",                 # or "gpt-4o-mini", "gpt-3.5-turbo"
    organization=os.environ.get("OPENAI_ORG"),  # optional
)
```

### OllamaAdapter (Local models)

```python
from llm_bridge import OllamaAdapter

# Requires Ollama running locally: https://ollama.ai
adapter = OllamaAdapter(
    model="llama3.2",               # any model pulled with `ollama pull`
    base_url="http://localhost:11434",
)
```

### HuggingFaceAdapter (Local transformers)

```python
from llm_bridge import HuggingFaceAdapter

adapter = HuggingFaceAdapter(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device="cuda",                  # or "cpu", "mps"
    load_in_4bit=True,              # quantization for large models
)
```

---

## Prompt Architecture

The bridge constructs a three-section prompt:

```
[SYSTEM]
You are a factual assistant. Answer questions using only the provided Knowledge Graph context.
Do not add information not present in the context. If the context is insufficient, say so.

[CONTEXT]
Knowledge Graph paths (verified, hallucination-free):
1. {entity} — score: {score}
   Path: {hop1} →[{rel1}]→ {hop2} →[{rel2}]→ {entity}
...

[QUESTION]
{query}
```

The system prompt explicitly instructs the LLM to stay grounded in the provided paths. The LLM's role is prose generation only — CEREBRUM has already done the reasoning.

---

## Design Philosophy

**The LLM bridge is optional.** CEREBRUM's core value — zero-hallucination graph reasoning — is fully realized without it. The bridge exists for deployments that need natural language output.

**CEREBRUM reasons; the LLM narrates.** The reasoning paths are computed deterministically from graph topology. The LLM converts verified paths into readable sentences. Any hallucination from the LLM is clearly separated from the grounded graph evidence.

**Low temperature by default.** The `temperature=0.3` default minimizes LLM creativity in favor of factual fidelity to the provided context.

---
**Copyright © 2026 Bryan Alexander Buchorn (AMP). All Rights Reserved.**

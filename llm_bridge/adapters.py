"""
LLM-agnostic adapter wrappers for llm_bridge.

The core contract is simple: any callable with the signature
    (prompt: str) -> str
works as an LLM function. These adapters wrap specific SDK patterns
so callers don't need to know the exact API of each provider.

All adapters:
  - Implement __call__(self, prompt: str) -> str
  - Raise ImportError with an install hint if the underlying library
    is not installed (never a silent failure)
  - Accept their underlying client/pipeline at construction time
    (makes mocking trivial in tests — just pass a mock object)

Usage
-----
    from llm_bridge.adapters import AnthropicAdapter, OpenAIAdapter
    from llm_bridge.context_formatter import generate

    # Anthropic
    llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")
    result = generate(answers, query="Who influenced Einstein?", llm_fn=llm)

    # OpenAI
    llm = OpenAIAdapter(model="gpt-4o-mini")
    result = generate(answers, query="Who influenced Einstein?", llm_fn=llm)

    # Ollama (local)
    llm = OllamaAdapter(model="llama3.2")
    result = generate(answers, query="Who influenced Einstein?", llm_fn=llm)

    # HuggingFace
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model="gpt2")
    llm  = HuggingFaceAdapter(pipe)
    result = generate(answers, query="Who influenced Einstein?", llm_fn=llm)

    # Any callable (plain Python — useful for testing or custom integrations)
    result = generate(answers, query="test", llm_fn=lambda prompt: "Natural language answer.")
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """
    Wraps the Anthropic Python SDK (``pip install anthropic``).

    Parameters
    ----------
    model       : Claude model ID. Default: claude-haiku-4-5-20251001 (fast, cheap).
    max_tokens  : Maximum tokens in the response.
    api_key     : Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    client      : Optional pre-built ``anthropic.Anthropic`` client (useful for
                  testing — pass a mock instead of a real client).
    system      : Optional system prompt prepended to every request.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    **({"api_key": api_key} if api_key else {})
                )
            except ImportError:
                raise ImportError(
                    "anthropic is required for AnthropicAdapter: pip install anthropic"
                )
        self._model      = model
        self._max_tokens = max_tokens
        self._system     = system

    def __call__(self, prompt: str) -> str:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._system:
            kwargs["system"] = self._system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def __repr__(self) -> str:
        return f"AnthropicAdapter(model={self._model!r}, max_tokens={self._max_tokens})"


# ---------------------------------------------------------------------------
# OpenAIAdapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """
    Wraps the OpenAI Python SDK (``pip install openai``).

    Also works with any OpenAI-compatible endpoint (Azure, Together AI,
    Mistral, etc.) by passing a custom ``base_url``.

    Parameters
    ----------
    model       : Model ID. Default: gpt-4o-mini.
    max_tokens  : Maximum tokens in the response.
    api_key     : OpenAI API key. If None, reads from OPENAI_API_KEY env var.
    base_url    : Optional custom endpoint URL for OpenAI-compatible APIs.
    client      : Optional pre-built ``openai.OpenAI`` client (for mocking).
    system      : Optional system prompt.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            try:
                import openai
                init_kwargs: dict = {}
                if api_key:
                    init_kwargs["api_key"] = api_key
                if base_url:
                    init_kwargs["base_url"] = base_url
                self._client = openai.OpenAI(**init_kwargs)
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIAdapter: pip install openai"
                )
        self._model      = model
        self._max_tokens = max_tokens
        self._system     = system

    def __call__(self, prompt: str) -> str:
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content

    def __repr__(self) -> str:
        return f"OpenAIAdapter(model={self._model!r}, max_tokens={self._max_tokens})"


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------

class OllamaAdapter:
    """
    Wraps the Ollama local model server (``https://ollama.ai``).

    Requires ``pip install httpx`` and a running Ollama server.
    The server is typically started with ``ollama serve`` and listens
    on http://localhost:11434 by default.

    Parameters
    ----------
    model   : Ollama model name (e.g. "llama3.2", "mistral", "phi3").
    url     : Ollama server base URL.
    timeout : HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self._model   = model
        self._url     = url.rstrip("/") + "/api/generate"
        self._timeout = timeout

    def __call__(self, prompt: str) -> str:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for OllamaAdapter: pip install httpx"
            )
        response = httpx.post(
            self._url,
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def __repr__(self) -> str:
        return f"OllamaAdapter(model={self._model!r}, url={self._url!r})"


# ---------------------------------------------------------------------------
# HuggingFaceAdapter
# ---------------------------------------------------------------------------

class HuggingFaceAdapter:
    """
    Wraps a HuggingFace ``transformers`` pipeline.

    The pipeline must already be constructed and passed in — this adapter
    does not import or load models itself.

    Parameters
    ----------
    pipeline    : A ``transformers.pipeline`` object (text-generation,
                  text2text-generation, summarization, etc.)
    max_new_tokens : Passed to the pipeline call if the pipeline supports it.

    Example:
        from transformers import pipeline
        pipe = pipeline("text-generation", model="gpt2", max_new_tokens=200)
        adapter = HuggingFaceAdapter(pipe)
    """

    def __init__(self, pipeline: Any, max_new_tokens: int = 256) -> None:
        self._pipeline       = pipeline
        self._max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        try:
            result = self._pipeline(prompt, max_new_tokens=self._max_new_tokens)
        except TypeError:
            # Some pipelines don't accept max_new_tokens
            result = self._pipeline(prompt)

        # Normalise varied pipeline output formats
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            # text-generation returns {"generated_text": "..."}
            # summarization returns {"summary_text": "..."}
            for key in ("generated_text", "summary_text", "translation_text"):
                if key in result:
                    text = result[key]
                    # text-generation often prepends the input prompt
                    if isinstance(text, str) and text.startswith(prompt):
                        text = text[len(prompt):].strip()
                    return text
        return str(result)

    def __repr__(self) -> str:
        return f"HuggingFaceAdapter(pipeline={self._pipeline!r})"

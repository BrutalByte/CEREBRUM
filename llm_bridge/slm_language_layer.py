"""
SLMLanguageLayer — Phase 280.

Thin wrapper around the existing OllamaAdapter (or any llm_bridge adapter)
that confines the SLM to exactly two jobs:

  1. ground(nl_question) → (seed_entity, path_schema)
     NL question → structured CEREBRUM query via forced JSON output prompt.

  2. surface(graph_answer, original_question) → str
     CEREBRUM beam result → natural language response.
     No knowledge injection — the SLM only formats, never invents facts.

The SLM handles syntax and pragmatics.
CEREBRUM handles all knowledge retrieval and reasoning.
This keeps the hallucination surface area minimal: the SLM can only
affect how queries are framed, not what the answers are.

Recommended model: Phi-3.5-mini-instruct (3.8B) via Ollama
  - Fully offline, ~4GB VRAM, fast on RTX 5090
  - Language-only: excellent at NL→JSON grounding, minimal hallucination risk

Usage
-----
    from llm_bridge.slm_language_layer import SLMLanguageLayer

    slm = SLMLanguageLayer()  # auto-detects Ollama at localhost:11434
    seed, schema = slm.ground("Who directed Inception?")
    # → ("inception", ["directed_by"])

    answer_text = slm.surface(
        graph_answer={"answers": ["Christopher Nolan"], "confidence": 0.95},
        original_question="Who directed Inception?",
    )
    # → "Christopher Nolan directed Inception."
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "phi3.5"  # Phi-3.5-mini-instruct — best balance for NL↔structured
_FALLBACK_MODEL = "llama3.2"

_GROUND_PROMPT = """\
You are a query parser. Convert the user's natural language question into a structured CEREBRUM knowledge graph query. Output ONLY valid JSON with this exact schema:

{{"seed_entity": "<main entity as lowercase_underscored_id>", "path_schema": ["<relation_1>", "<relation_2>"]}}

Rules:
- seed_entity: the main entity the question is about, lowercased with underscores (e.g. "christopher_nolan")
- path_schema: 1-3 most likely relation types that connect to the answer (e.g. ["directed_by", "produced_by"])
- Do NOT include any explanation. Output ONLY the JSON object.

Question: {question}
"""

_SURFACE_PROMPT = """\
You are a concise answer generator. Use ONLY the provided graph answers — do not add any information beyond what is given. Write 1-2 sentences.

Question: {question}
Graph answers: {answers}
Confidence: {confidence:.2f}

Answer:"""


class SLMLanguageLayer:
    """
    Two-method language interface backed by a local SLM via Ollama.

    Falls back to a no-op stub if Ollama is unavailable (CEREBRUM continues
    to work with structured queries; only natural language endpoints degrade).
    """

    def __init__(
        self,
        model:      str   = _DEFAULT_MODEL,
        ollama_url: str   = "http://localhost:11434",
        timeout:    float = 30.0,
        llm_fn:     Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model       : Ollama model name. Default: phi3.5 (Phi-3.5-mini-instruct).
        ollama_url  : Ollama server base URL.
        timeout     : Request timeout in seconds.
        llm_fn      : Override with any (prompt: str) -> str callable (e.g. AnthropicAdapter).
                      When provided, model/ollama_url/timeout are ignored.
        """
        self._available = False

        if llm_fn is not None:
            self._llm = llm_fn
            self._available = True
            return

        try:
            from llm_bridge.adapters import OllamaAdapter
            self._llm = OllamaAdapter(model=model, url=ollama_url, timeout=timeout)
            # Probe Ollama availability
            self._probe_ollama(ollama_url)
        except Exception as exc:
            logger.warning("SLMLanguageLayer: Ollama unavailable (%s). Running in stub mode.", exc)
            self._llm = self._stub_llm

    # ── Public API ────────────────────────────────────────────────────────────

    def ground(self, nl_question: str) -> Tuple[str, List[str]]:
        """
        Convert a natural language question to a structured CEREBRUM query.

        Returns
        -------
        (seed_entity, path_schema)
            seed_entity  : primary entity ID (lowercase_underscored)
            path_schema  : ordered list of relation types to traverse

        Falls back to naive tokenization if the SLM output cannot be parsed.
        """
        if not self._available:
            return self._naive_ground(nl_question)

        prompt = _GROUND_PROMPT.format(question=nl_question)
        try:
            raw = self._llm(prompt)
            return self._parse_ground(raw, nl_question)
        except Exception as exc:
            logger.warning("SLMLanguageLayer.ground: SLM call failed (%s). Using naive fallback.", exc)
            return self._naive_ground(nl_question)

    def surface(
        self,
        graph_answer: Any,
        original_question: str,
    ) -> str:
        """
        Convert a CEREBRUM beam result to a natural language response.

        Parameters
        ----------
        graph_answer : dict with keys "answers" (list[str]) and "confidence" (float),
                       OR a list of answer strings, OR a plain string.
        original_question : the original NL question.

        Returns
        -------
        A 1-2 sentence natural language answer. If SLM is unavailable,
        returns a plain formatted string from the graph answers directly.
        """
        answers, confidence = self._extract_answers(graph_answer)

        if not self._available or not answers:
            return self._naive_surface(answers, original_question)

        prompt = _SURFACE_PROMPT.format(
            question   = original_question,
            answers    = ", ".join(answers[:5]),
            confidence = confidence,
        )
        try:
            return self._llm(prompt).strip()
        except Exception as exc:
            logger.warning("SLMLanguageLayer.surface: SLM call failed (%s). Using naive fallback.", exc)
            return self._naive_surface(answers, original_question)

    @property
    def available(self) -> bool:
        """True if an SLM backend is reachable."""
        return self._available

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_ground(self, raw: str, question: str) -> Tuple[str, List[str]]:
        """Extract (seed_entity, path_schema) from SLM JSON output."""
        # Strip markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Find first JSON object
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                seed   = str(obj.get("seed_entity", "")).strip()
                schema = [str(r).strip() for r in obj.get("path_schema", [])]
                if seed and schema:
                    return _slugify(seed), schema
            except json.JSONDecodeError:
                pass
        logger.debug("SLMLanguageLayer: could not parse grounding output: %r", raw[:200])
        return self._naive_ground(question)

    def _naive_ground(self, question: str) -> Tuple[str, List[str]]:
        """Fallback: extract longest token as seed, empty schema."""
        tokens = re.findall(r"[A-Za-z]{3,}", question)
        stopwords = {"who", "what", "where", "when", "why", "how", "did", "does",
                     "was", "were", "the", "and", "for", "are", "has", "have"}
        significant = [t for t in tokens if t.lower() not in stopwords]
        seed = _slugify(significant[0]) if significant else "unknown"
        return seed, []

    def _extract_answers(self, graph_answer: Any) -> Tuple[List[str], float]:
        """Normalize graph_answer to (answers_list, confidence)."""
        if isinstance(graph_answer, dict):
            answers    = graph_answer.get("answers", [])
            confidence = float(graph_answer.get("confidence", 0.0))
        elif isinstance(graph_answer, list):
            answers    = [str(a) for a in graph_answer]
            confidence = 0.9
        elif isinstance(graph_answer, str):
            answers    = [graph_answer]
            confidence = 0.9
        else:
            answers    = [str(graph_answer)] if graph_answer else []
            confidence = 0.0
        return answers, confidence

    def _naive_surface(self, answers: List[str], question: str) -> str:
        """Fallback: plain English from answer list."""
        if not answers:
            return "No answer found in the knowledge graph."
        if len(answers) == 1:
            return f"The answer is: {answers[0]}."
        return f"The answers are: {', '.join(answers[:5])}."

    def _probe_ollama(self, base_url: str) -> None:
        """Quick connectivity check — sets self._available."""
        try:
            import httpx
            resp = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=3.0)
            resp.raise_for_status()
            self._available = True
            logger.info("SLMLanguageLayer: Ollama available at %s.", base_url)
        except Exception as exc:
            logger.info("SLMLanguageLayer: Ollama probe failed (%s). Stub mode active.", exc)

    @staticmethod
    def _stub_llm(prompt: str) -> str:
        return ""


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:120]

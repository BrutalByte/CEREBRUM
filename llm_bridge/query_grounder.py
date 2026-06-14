"""
QueryGrounder — Phase 280.

Routes NL query grounding requests:
  1. If local Ollama is available → SLMLanguageLayer (offline, private)
  2. Else if cloud adapter configured → AnthropicAdapter / OpenAIAdapter
  3. Else → naive tokenization fallback (no external calls)

Also implements CerebrumVerifier: when CEREBRUM returns knowledge_gap=True,
the grounder proposes a candidate triple via the SLM and routes it through
KnowledgeHarvester's vetting pipeline before materializing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryGrounder:
    """
    Unified entry point for NL→structured grounding and
    structured→NL surfacing.

    Usage
    -----
        grounder = QueryGrounder(adapter=adapter, knowledge_harvester=harvester)
        seed, schema = grounder.ground("Who wrote Hamlet?")
        answer_text  = grounder.surface({"answers": ["Shakespeare"], "confidence": 0.9},
                                        "Who wrote Hamlet?")
    """

    def __init__(
        self,
        ollama_model:        str   = "phi3.5",
        ollama_url:          str   = "http://localhost:11434",
        cloud_llm_fn:        Optional[Callable[[str], str]] = None,
        adapter:             Optional[Any] = None,       # GraphAdapter
        knowledge_harvester: Optional[Any] = None,       # KnowledgeHarvester (Phase 270)
    ) -> None:
        from llm_bridge.slm_language_layer import SLMLanguageLayer

        # Primary: local SLM
        self._slm = SLMLanguageLayer(model=ollama_model, ollama_url=ollama_url)

        # Fallback: cloud LLM if Ollama unavailable
        if not self._slm.available and cloud_llm_fn is not None:
            self._slm = SLMLanguageLayer(llm_fn=cloud_llm_fn)

        self._adapter   = adapter
        self._harvester = knowledge_harvester

    # ── Public API ────────────────────────────────────────────────────────────

    def ground(self, nl_question: str) -> Tuple[str, List[str]]:
        """NL question → (seed_entity, path_schema)."""
        return self._slm.ground(nl_question)

    def surface(self, graph_answer: Any, original_question: str) -> str:
        """CEREBRUM answer → natural language string."""
        return self._slm.surface(graph_answer, original_question)

    def verify_and_fill_gap(
        self,
        nl_question:   str,
        missing_entity: str,
    ) -> bool:
        """
        Called when CEREBRUM returns knowledge_gap=True.

        Asks the SLM to propose a (subject, relation, object) triple that
        would fill the gap, then routes it through KnowledgeHarvester's
        5-stage vetting pipeline.

        Returns True if the gap was filled (a triple passed vetting and was
        materialized), False otherwise.
        """
        if self._harvester is None:
            return False

        prompt = (
            f"A knowledge graph could not answer: '{nl_question}'.\n"
            f"The missing entity is '{missing_entity}'.\n"
            f"Propose ONE factual triple to fill this gap as JSON: "
            f"{{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}}\n"
            f"Use only well-known facts. Output ONLY the JSON."
        )
        try:
            raw = self._slm._llm(prompt)
        except Exception:
            return False

        import json, re
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        m   = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            return False

        try:
            obj = json.loads(m.group(0))
            subject  = str(obj.get("subject",  "")).strip()
            relation = str(obj.get("relation", "")).strip().upper()
            object_  = str(obj.get("object",   "")).strip()
        except Exception:
            return False

        if not (subject and relation and object_):
            return False

        from core.knowledge_harvester import CandidateTriple
        import uuid
        triple = CandidateTriple(
            triple_id   = str(uuid.uuid4()),
            source      = _slugify(subject),
            relation    = relation,
            target      = _slugify(object_),
            source_name = subject,
            target_name = object_,
            source_url  = "slm_inference",
            source_tier = 3,          # SLM-proposed: treat as tier-3, needs corroboration
            confidence  = 0.60,
        )

        # SLM-proposed triples are tier-3 — the corroboration check will
        # reject them unless ≥3 reputable sources back the same claim.
        # We run vetting directly; the tier-3 gate will block if uncorroborated.
        ok, reason = self._harvester._vet(triple)
        if ok:
            self._harvester._materialize(triple)
            logger.info(
                "QueryGrounder: gap filled via SLM triple (%s, %s, %s).",
                triple.source, triple.relation, triple.target,
            )
            return True
        else:
            logger.debug("QueryGrounder: SLM-proposed triple rejected: %s", reason)
            self._harvester._log_rejected(triple, f"slm_proposed: {reason}")
            return False

    @property
    def slm_available(self) -> bool:
        return self._slm.available


def _slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:120]

"""
CEREBRUM LangChain Retriever Adapter.

Drop-in BaseRetriever that backs any LangChain chain with CEREBRUM's
crystal-box knowledge-graph traversal instead of vector similarity search.

    from llm_bridge.langchain_adapter import CerebrumRetriever

    retriever = CerebrumRetriever.from_csv("kb.csv", top_k=5)

    # Use in any LangChain chain
    from langchain.chains import RetrievalQA
    from langchain_anthropic import ChatAnthropic

    qa = RetrievalQA.from_chain_type(
        llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
        retriever=retriever,
    )
    print(qa.invoke({"query": "Who directed Inception?"}))

How it works
------------
CEREBRUM performs the multi-hop KB traversal; each answer entity + its
reasoning path becomes a LangChain Document.  The LLM only handles
natural-language generation over verified facts — hallucination-free by
design (crystal-box).
"""
from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False


def _require_langchain():
    if not _LC_AVAILABLE:
        raise ImportError(
            "langchain-core is required: pip install langchain-core"
        )


class CerebrumRetriever:
    """
    LangChain-compatible retriever backed by CEREBRUM's crystal-box traversal.

    Instantiate via factory methods, not directly.
    """

    def __init__(self, cerebrum_instance, *, top_k: int = 5):
        _require_langchain()
        self._c = cerebrum_instance
        self._top_k = top_k

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str, *, top_k: int = 5, **kwargs) -> "CerebrumRetriever":
        """Build a retriever from a CSV edge-list."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_csv(path, **kwargs), top_k=top_k)

    @classmethod
    def from_kb(cls, path: str, *, top_k: int = 5, sep: str = "|", **kwargs) -> "CerebrumRetriever":
        """Build a retriever from a pipe-separated KB file."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_kb(path, sep=sep, **kwargs), top_k=top_k)

    @classmethod
    def from_triples(cls, triples: list, *, top_k: int = 5, **kwargs) -> "CerebrumRetriever":
        """Build a retriever from (subject, relation, object) tuples."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_triples(triples, **kwargs), top_k=top_k)

    # ── LangChain BaseRetriever interface ────────────────────────────────────

    def get_relevant_documents(self, query: str) -> List[Any]:
        """Synchronous retrieval — compatible with LangChain v0.1 chains."""
        _require_langchain()
        return self._retrieve(query)

    async def aget_relevant_documents(self, query: str) -> List[Any]:
        """Async retrieval — runs sync in executor to avoid blocking event loop."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_relevant_documents, query)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Any]:
        """LangChain v0.2+ internal protocol."""
        return self._retrieve(query)

    def _retrieve(self, query: str) -> List[Any]:
        _require_langchain()
        result = self._c.ask(query, top_k=self._top_k)
        docs = []
        for candidate in result.top_k:
            # Build a trace string for page_content
            trace_parts = []
            for step in result.trace_path:
                trace_parts.append(f"{step.entity} --[{step.relation}]-->")
            trace_str = " ".join(trace_parts) + f" {candidate['entity']}" if trace_parts else candidate["entity"]

            docs.append(Document(
                page_content=trace_str,
                metadata={
                    "entity": candidate["entity"],
                    "confidence": candidate["confidence"],
                    "source": "CEREBRUM-crystal-box",
                    "query": query,
                    "elapsed_ms": result.elapsed_ms,
                },
            ))
        return docs

    # ── Make it act as a LangChain BaseRetriever via duck typing ────────────

    def __class_getitem__(cls, item):
        return cls

    def as_retriever(self) -> "CerebrumRetriever":
        """Return self — already a retriever."""
        return self

    # ── Direct query access ──────────────────────────────────────────────────

    def query(self, question: str, top_k: Optional[int] = None):
        """Direct CEREBRUM query returning a typed Result (bypasses LangChain)."""
        return self._c.ask(question, top_k=top_k or self._top_k)

    @property
    def stats(self) -> dict:
        """Return KB statistics."""
        return self._c.stats


# ---------------------------------------------------------------------------
# Try to register as a proper LangChain BaseRetriever subclass if available
# ---------------------------------------------------------------------------

if _LC_AVAILABLE:
    class CerebrumRetriever(CerebrumRetriever, BaseRetriever):  # type: ignore[no-redef]
        """
        LangChain BaseRetriever subclass (auto-registered when langchain-core is installed).
        Adds full compatibility with LCEL, RunnableSequence, and LangChain v0.2 chains.
        """

        class Config:
            arbitrary_types_allowed = True

        def __init__(self, cerebrum_instance, *, top_k: int = 5):
            # Pydantic-based BaseRetriever needs special init
            object.__setattr__(self, "_c", cerebrum_instance)
            object.__setattr__(self, "_top_k", top_k)

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
            return self._retrieve(query)

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: Any,
        ) -> List[Document]:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._retrieve, query)

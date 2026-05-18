"""
CEREBRUM LlamaIndex Retriever Adapter.

Drop-in BaseRetriever that backs any LlamaIndex query engine with
CEREBRUM's crystal-box knowledge-graph traversal.

    from llm_bridge.llamaindex_adapter import CerebrumLlamaRetriever

    retriever = CerebrumLlamaRetriever.from_csv("kb.csv", top_k=5)

    # Use directly
    nodes = retriever.retrieve("Who directed Inception?")
    for n in nodes:
        print(n.text, n.metadata)

    # Or plug into a LlamaIndex query engine
    from llama_index.core.query_engine import RetrieverQueryEngine
    engine = RetrieverQueryEngine.from_args(retriever)
    response = engine.query("Who directed Inception?")

How it works
------------
CEREBRUM performs the multi-hop KB traversal; each answer entity + its
reasoning path is wrapped as a LlamaIndex NodeWithScore / TextNode.
"""
from __future__ import annotations

from typing import Any, List, Optional

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    _LI_AVAILABLE = True
except ImportError:
    _LI_AVAILABLE = False


def _require_llamaindex():
    if not _LI_AVAILABLE:
        raise ImportError(
            "llama-index-core is required: pip install llama-index-core"
        )


class CerebrumLlamaRetriever:
    """
    LlamaIndex-compatible retriever backed by CEREBRUM's crystal-box traversal.

    Instantiate via factory methods, not directly.
    """

    def __init__(self, cerebrum_instance, *, top_k: int = 5):
        self._c = cerebrum_instance
        self._top_k = top_k

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str, *, top_k: int = 5, **kwargs) -> "CerebrumLlamaRetriever":
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_csv(path, **kwargs), top_k=top_k)

    @classmethod
    def from_kb(cls, path: str, *, top_k: int = 5, sep: str = "|", **kwargs) -> "CerebrumLlamaRetriever":
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_kb(path, sep=sep, **kwargs), top_k=top_k)

    @classmethod
    def from_triples(cls, triples: list, *, top_k: int = 5, **kwargs) -> "CerebrumLlamaRetriever":
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from sdk.python.cerebrum_sdk import Cerebrum
        return cls(Cerebrum.from_triples(triples, **kwargs), top_k=top_k)

    # ── Core retrieval ───────────────────────────────────────────────────────

    def retrieve(self, query_or_bundle) -> List[Any]:
        """Retrieve relevant nodes for a query string or QueryBundle."""
        _require_llamaindex()
        query_str = (
            query_or_bundle.query_str
            if hasattr(query_or_bundle, "query_str")
            else str(query_or_bundle)
        )
        return self._retrieve(query_str)

    async def aretrieve(self, query_or_bundle) -> List[Any]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query_or_bundle)

    def _retrieve(self, query: str) -> List[Any]:
        _require_llamaindex()
        result = self._c.ask(query, top_k=self._top_k)
        nodes = []
        for candidate in result.top_k:
            trace_parts = []
            for step in result.trace_path:
                trace_parts.append(f"{step.entity} --[{step.relation}]-->")
            trace_str = " ".join(trace_parts) + f" {candidate['entity']}" if trace_parts else candidate["entity"]

            text_node = TextNode(
                text=trace_str,
                metadata={
                    "entity": candidate["entity"],
                    "confidence": candidate["confidence"],
                    "source": "CEREBRUM-crystal-box",
                    "query": query,
                    "elapsed_ms": result.elapsed_ms,
                },
            )
            nodes.append(NodeWithScore(node=text_node, score=candidate["confidence"]))
        return nodes

    def as_query_engine(self, llm=None):
        """Convenience: wrap in a RetrieverQueryEngine if llama-index is available."""
        _require_llamaindex()
        from llama_index.core.query_engine import RetrieverQueryEngine
        return RetrieverQueryEngine.from_args(self, llm=llm)

    @property
    def stats(self) -> dict:
        return self._c.stats


# ---------------------------------------------------------------------------
# Try to register as a proper LlamaIndex BaseRetriever subclass
# ---------------------------------------------------------------------------

if _LI_AVAILABLE:
    class CerebrumLlamaRetriever(CerebrumLlamaRetriever, BaseRetriever):  # type: ignore[no-redef]
        """
        LlamaIndex BaseRetriever subclass (auto-registered when llama-index-core is installed).
        """

        def _retrieve(self, query_bundle: Any) -> List[Any]:
            query_str = (
                query_bundle.query_str
                if hasattr(query_bundle, "query_str")
                else str(query_bundle)
            )
            result = self._c.ask(query_str, top_k=self._top_k)
            nodes = []
            for candidate in result.top_k:
                trace_parts = []
                for step in result.trace_path:
                    trace_parts.append(f"{step.entity} --[{step.relation}]-->")
                trace_str = (
                    " ".join(trace_parts) + f" {candidate['entity']}"
                    if trace_parts else candidate["entity"]
                )
                text_node = TextNode(
                    text=trace_str,
                    metadata={
                        "entity": candidate["entity"],
                        "confidence": candidate["confidence"],
                        "source": "CEREBRUM-crystal-box",
                        "query": query_str,
                        "elapsed_ms": result.elapsed_ms,
                    },
                )
                nodes.append(NodeWithScore(node=text_node, score=candidate["confidence"]))
            return nodes

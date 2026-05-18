# LangChain Integration

`CerebrumRetriever` is a drop-in `BaseRetriever` that replaces vector similarity search with crystal-box knowledge-graph traversal.

## Install

```bash
pip install "cerebrum-kg[all]" langchain-core langchain-anthropic
```

## Basic usage

```python
from llm_bridge.langchain_adapter import CerebrumRetriever

retriever = CerebrumRetriever.from_csv("kb.csv", top_k=5)

# Retrieve Documents for a query
docs = retriever.get_relevant_documents("Who directed Inception?")
for doc in docs:
    print(doc.page_content)   # Full trace: Inception --[directed_by]--> Christopher_Nolan
    print(doc.metadata)       # {'entity': 'Christopher_Nolan', 'confidence': 0.923, ...}
```

## In a RetrievalQA chain

```python
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

retriever = CerebrumRetriever.from_csv("kb.csv")

qa = RetrievalQA.from_chain_type(
    llm=ChatAnthropic(model="claude-haiku-4-5-20251001"),
    retriever=retriever,
)
response = qa.invoke({"query": "Who directed Inception?"})
print(response["result"])
```

CEREBRUM handles the multi-hop KB traversal. The LLM only generates natural-language prose over the verified facts — hallucination-free by design.

## In an LCEL chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

retriever = CerebrumRetriever.from_csv("kb.csv")

prompt = ChatPromptTemplate.from_template("""
Answer based only on the following knowledge graph paths:
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatAnthropic(model="claude-haiku-4-5-20251001")
    | StrOutputParser()
)

print(chain.invoke("Who directed Inception?"))
```

## Factory methods

```python
# From CSV
retriever = CerebrumRetriever.from_csv("kb.csv", top_k=5)

# From pipe-separated KB file
retriever = CerebrumRetriever.from_kb("metaqa_kb.txt", top_k=10)

# From triples
retriever = CerebrumRetriever.from_triples([
    ("A", "knows", "B"),
    ("B", "works_with", "C"),
])
```

## Document format

Each retrieved `Document` has:

- `page_content`: full hop-by-hop trace string, e.g. `Inception --[directed_by]--> Christopher_Nolan`
- `metadata.entity`: the answer entity
- `metadata.confidence`: CEREBRUM confidence score (0–1)
- `metadata.source`: `"CEREBRUM-crystal-box"`
- `metadata.query`: original query string
- `metadata.elapsed_ms`: traversal time

## Direct query (bypass LangChain)

```python
result = retriever.query("Who directed Inception?")
print(result.answer, result.trace_path)
```

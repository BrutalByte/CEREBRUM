"""
KB Construction Quickstart: extract triples from text and query the result.

Demonstrates CEREBRUM's end-to-end pipeline:
  raw text → (head, relation, tail) triples → queryable knowledge graph

No labeled data, no training. The extracted graph is immediately queryable
with full hop-by-hop reasoning traces.

Usage:
    python examples/text_to_graph.py

Optional: install spaCy for enhanced NLP extraction
    pip install spacy && python -m spacy download en_core_web_sm

Optional: install sentence-transformers for semantic embeddings
    pip install sentence-transformers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extraction_engine import ExtractionEngine, ExtractionConfig
from adapters.networkx_adapter import NetworkXAdapter

# ---------------------------------------------------------------------------
# 1. Seed the graph with a few known facts (optional — can start empty)
# ---------------------------------------------------------------------------
seed_triples = [
    ("aspirin",     "TREATS",    "headache"),
    ("aspirin",     "TREATS",    "fever"),
    ("ibuprofen",   "TREATS",    "inflammation"),
    ("ibuprofen",   "INHIBITS",  "COX-2"),
    ("COX-2",       "CAUSES",    "inflammation"),
    ("metformin",   "TREATS",    "diabetes"),
    ("insulin",     "REGULATES", "blood_glucose"),
    ("diabetes",    "INVOLVES",  "blood_glucose"),
]

adapter = NetworkXAdapter.from_triples(seed_triples)
print(f"Seed graph: {adapter.node_count()} nodes, {adapter.edge_count()} edges")

# ---------------------------------------------------------------------------
# 2. Extract triples from new text passages
# ---------------------------------------------------------------------------
passages = [
    "Aspirin also reduces the risk of heart attack by inhibiting platelet aggregation.",
    "Metformin activates AMPK which in turn inhibits gluconeogenesis in the liver.",
    "Ibuprofen treats pain and fever, similar to aspirin but with fewer gastric side effects.",
    "Insulin resistance causes type 2 diabetes and is associated with obesity.",
]

engine = ExtractionEngine(
    adapter,
    config=ExtractionConfig(backend="local", min_confidence=0.45),
)

total_new = 0
for passage in passages:
    report = engine.ingest_text(passage)
    new = len(report.new_triples)
    total_new += new
    if new:
        for t in report.new_triples:
            print(f"  + ({t.source}, {t.relation}, {t.target})  conf={t.confidence:.2f}")

print(f"\nExtracted {total_new} new triples from {len(passages)} passages")
print(f"Graph now: {adapter.node_count()} nodes, {adapter.edge_count()} edges\n")

# ---------------------------------------------------------------------------
# 3. Build queryable CerebrumGraph and run multi-hop queries
# ---------------------------------------------------------------------------
try:
    from core.cerebrum import CerebrumGraph
    graph = CerebrumGraph.from_adapter(adapter, embeddings="sentence")
except Exception:
    from core.cerebrum import CerebrumGraph
    graph = CerebrumGraph.from_adapter(adapter, embeddings="random")

graph.build(cache_dir=None, min_community_size=0)
print("Graph built. Running queries...\n")

queries = [
    ("aspirin",   "What does aspirin treat or inhibit?"),
    ("metformin", "What pathway does metformin affect?"),
    ("diabetes",  "What is diabetes associated with?"),
]

for seed, question in queries:
    print(f"Q: {question}")
    results = graph.query(seeds=[seed], top_k=5, max_hop=2)
    for r in results[:3]:
        path_str = " → ".join(r.best_path.nodes) if r.best_path else seed
        print(f"  {r.entity_id:25s}  score={r.score:.3f}  path: {path_str}")
    print()

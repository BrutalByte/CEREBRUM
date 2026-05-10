import sys
import time
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.cerebrum import CerebrumGraph
from reasoning.trace import ReasoningTrace
from reasoning.answer_extractor import extract
import numpy as np

# Load KB and build graph
KB_FILE = Path("E:/Development/Cerebrum/benchmarks/data/metaqa/kb.txt")
print(f"Loading KB from {KB_FILE}...")

t0 = time.time()
graph = CerebrumGraph.from_kb(
    KB_FILE,
    sep="|",
    directed=False,
    embeddings="random",
    beam_width=10,
    max_hop=3
)
print(f"Graph loaded ({time.time()-t0:.1f}s)")

# Phase 3: Build communities
graph.build(community_engine="dscf", n_trials=1)

seed = "The Green Mile"
target_rel = "directed_by"
trb = {target_rel: 25.0}

print(f"\nQuerying: {seed} (3-hop, TRB={trb}, branch_bonus=0.25)")
# Simulate graph.query but capture paths for manual extraction analysis
mh = 3
bw = 10
eff_widths = {h: bw * h for h in range(1, mh + 1)}

from reasoning.expanded_traversal import HopExpandedTraversal
traversal = HopExpandedTraversal(
    adapter=graph.adapter,
    csa_engine=graph._csa,
    beam_width=bw,
    max_hop=mh,
    beam_widths=eff_widths,
    terminal_relation_boost=trb,
    hop_expand=True
)

paths = traversal.traverse([seed])

# Extract with different weights to see what happens
for dpw in [0.0, 0.05, 0.1, 0.2, 0.5]:
    print(f"\n--- Extraction Analysis (degree_penalty_weight={dpw}) ---")
    # Manually pass degree penalty logic since adapter.get_degree is missing
    # or just fix answer_extractor first.
    
    answers = extract(
        paths,
        top_k=5,
        min_hop=3,
        branch_bonus_weight=0.25,
        degree_penalty_weight=dpw,
        adapter=graph.adapter, # answer_extractor will fail if it calls .get_degree()
        vote_weight=0.45
    )
    for i, ans in enumerate(answers):
        deg = graph.adapter._G.degree(ans.entity_id)
        print(f"{i+1}. {ans.entity_id} | Score: {ans.score:.4f} | Branches: {ans.branch_count} | Deg: {deg} | PathScore: {ans.path_score:.4f}")
        # Print relation sequence of best path
        p = ans.best_path
        rels = [p.nodes[j] for j in range(1, len(p.nodes), 2)]
        print(f"   Best Path Rels: {' -> '.join(rels)}")

sys.stdout.flush()

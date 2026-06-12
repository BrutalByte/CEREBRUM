"""
Phase 253d: WebQSP 1-hop Reachability Check
For A2 beam-miss cases (gold in graph, not in beam top-90), checks whether
the gold answer is a direct 1-hop neighbor of the seed entity.

Splits the 1114 beam-miss cases into:
  H1: gold is a direct 1-hop neighbor (exhaustive 1-hop enum would find it)
  H2: gold requires 2 hops (need beam or schema to reach)
  HX: gold is in graph but not reachable in 1 or 2 hops from seed
"""
import sys, os, math, collections
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "benchmarks")
sys.path.insert(0, ".")

from webqsp_param_eval import build_webqsp_state, _SKIP_RELATIONS
from core.question_decomposer import QuestionDecomposer

PARAMS = {
    "trb_factor": 44.864, "r2_boost": 2.732, "vote_weight": 0.8515,
    "beam_width": 16, "idf_weight": 0.017, "branch_bonus": 0.066,
    "fhrb_factor": 2.246, "gamma": 8.6062, "beta": 1.1866,
    "degree_penalty_weight": 0.6318, "schema_score_threshold": 0.3572,
    "backward_bonus": 0.1084, "diversity_alpha": 0.6549,
    "cvt_passthrough": True, "max_loops": 1, "schema_top_k": 12,
}

N_QUESTIONS = 1628
FETCH_K = 30

def main():
    print("[HopCheck] Building WebQSP state...", flush=True)
    state = build_webqsp_state(n_questions=N_QUESTIONS, embeddings="sentence", seed=42, use_cache=True)
    graph      = state["graph"]
    qa_pairs   = state["qa_pairs"]
    deriver    = state["deriver"]
    rel_index  = state.get("rel_index")
    schema_idx = state.get("schema_idx")
    nx_g       = graph.adapter.to_networkx()
    graph_nodes = set(nx_g.nodes())
    adapter    = graph.adapter

    gamma = float(PARAMS["gamma"]); beta = float(PARAMS["beta"])
    trb_factor  = float(PARAMS["trb_factor"]); fhrb_factor = float(PARAMS["fhrb_factor"])
    trb_map  = deriver.boost_map(trb_factor, beta) if deriver.is_built else {}
    fhrb_map = deriver.boost_map(fhrb_factor, beta) if deriver.is_built else {}
    schema_score_thresh = float(PARAMS["schema_score_threshold"])
    schema_top_k = int(PARAMS.get("schema_top_k", 5))

    qd = QuestionDecomposer()
    emb_engine = getattr(graph, "_embedding_engine", None)

    _platt = getattr(graph, "_platt", None)
    if _platt:
        _platt._samples = []; _platt._fitted = False
    graph._recent_answer_cache = {}
    graph._feedback_buf = []

    h1_miss  = 0   # gold is direct 1-hop neighbor but beam missed it
    h2_miss  = 0   # gold requires exactly 2 hops, beam missed
    hx_miss  = 0   # gold in graph but not reachable in <=2 hops from seed
    h1_hit   = 0   # gold is 1-hop, beam already found it
    total_beam_miss = 0

    # For characterizing H1 misses
    h1_miss_examples = []
    h2_miss_examples = []

    print(f"[HopCheck] Processing {len(qa_pairs)} questions...", flush=True)

    for qi, (seed_ent, gold_answers, question) in enumerate(qa_pairs):
        if qi % 200 == 0:
            print(f"  {qi}/{len(qa_pairs)}", flush=True)

        if not seed_ent or seed_ent not in graph_nodes:
            continue
        n_present = sum(1 for a in gold_answers if a in graph_nodes)
        if n_present == 0:
            continue
        gold_set = set(a for a in gold_answers if a in graph_nodes)

        # Run beam to check if answer is found
        decomp = qd.decompose(question)
        q_scores = rel_index.score_relations(decomp.relation_keywords, min_score=0.05) if rel_index and decomp.relation_keywords else {}
        qemb = (emb_engine.encode_one(question) if emb_engine and hasattr(emb_engine, "encode_one") else None)

        schema_extra = []
        if schema_idx is not None and qemb is not None:
            seed_rels = adapter.get_all_relation_types(seed_ent)
            preds = schema_idx.predict_schemas_for_seed(qemb, seed_rels, top_k=schema_top_k)
            if preds:
                hits = schema_idx.execute_schemas(seed_ent, preds, adapter, skip_rels=_SKIP_RELATIONS)
                schema_extra = [(eid, sc) for eid, sc in hits]

        q_fhrb = {r: (1.0 + fhrb_map.get(r, 0.0)) for r in adapter.get_all_relation_types(seed_ent)} if fhrb_map else {}

        try:
            answers_obj = graph.query(
                seeds=[seed_ent], top_k=FETCH_K * 3, min_hop=1, max_hop=2,
                beam_width=int(PARAMS["beam_width"]),
                terminal_relation_boost=trb_map or None,
                vote_weight=float(PARAMS["vote_weight"]),
                branch_bonus_weight=float(PARAMS["branch_bonus"]),
                initial_relation_boost=q_fhrb,
                query_embedding=qemb, max_loops=1,
                degree_penalty_weight=float(PARAMS["degree_penalty_weight"]),
                cvt_passthrough=True,
            )
        except Exception:
            continue

        beam_ids = {a.entity_id for a in answers_obj if not str(a.entity_id).startswith("/m/")}
        gold_in_beam = gold_set & beam_ids

        if gold_in_beam:
            # Beam found at least one gold — not a miss case
            # Check if any gold is 1-hop for characterization
            hop1_neighbors = {e.target_id for e in adapter.get_neighbors(seed_ent, max_neighbors=5000)}
            if gold_set & hop1_neighbors:
                h1_hit += 1
            continue

        # Beam missed all gold answers in graph — now characterize hop distance
        total_beam_miss += 1

        # Get all 1-hop neighbors (may include CVT nodes)
        hop1_neighbors = {e.target_id for e in adapter.get_neighbors(seed_ent, max_neighbors=5000)}

        # Check direct 1-hop
        gold_at_hop1 = gold_set & hop1_neighbors
        if gold_at_hop1:
            h1_miss += 1
            if len(h1_miss_examples) < 10:
                h1_miss_examples.append((question, seed_ent, list(gold_at_hop1)[:2]))
            continue

        # Check 2-hop via CVT: seed -> CVT -> gold
        # Walk hop1 nodes to find hop2 neighbors
        found_at_2 = False
        for h1 in list(hop1_neighbors)[:200]:  # cap for speed
            hop2_neighbors = {e.target_id for e in adapter.get_neighbors(h1, max_neighbors=200)}
            if gold_set & hop2_neighbors:
                found_at_2 = True
                break

        if found_at_2:
            h2_miss += 1
            if len(h2_miss_examples) < 10:
                h2_miss_examples.append((question, seed_ent, list(gold_set)[:2]))
        else:
            hx_miss += 1

    print(f"\n{'='*65}")
    print(f"WebQSP Hop Reachability  ({total_beam_miss} beam-miss A2 cases)")
    print(f"{'='*65}")
    print(f"")
    print(f"H1: gold is direct 1-hop neighbor, beam missed:  {h1_miss:5d}  ({h1_miss/max(total_beam_miss,1)*100:.1f}%)")
    print(f"H2: gold reachable at 2 hops, beam missed:       {h2_miss:5d}  ({h2_miss/max(total_beam_miss,1)*100:.1f}%)")
    print(f"HX: gold in graph, not found in <=2 hop check:   {hx_miss:5d}  ({hx_miss/max(total_beam_miss,1)*100:.1f}%)")
    print(f"")

    total_q = N_QUESTIONS
    print(f"H1 miss as fraction of all questions: {h1_miss/total_q*100:.1f}%")
    print(f"H2 miss as fraction of all questions: {h2_miss/total_q*100:.1f}%")
    print(f"")
    print(f"Exhaustive 1-hop enum + LLM would recover at most: {h1_miss/total_q*100:.1f}pp H@1")

    print(f"\nSample H1 misses (gold is 1-hop, beam missed):")
    for q, s, g in h1_miss_examples[:5]:
        print(f"  Q: {q[:75]}")
        print(f"     seed={s}  gold={g}")

    print(f"\nSample H2 misses (gold at 2 hops, beam missed):")
    for q, s, g in h2_miss_examples[:5]:
        print(f"  Q: {q[:75]}")
        print(f"     seed={s}  gold={g}")


if __name__ == "__main__":
    main()

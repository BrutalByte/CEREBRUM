"""
Phase 253c: WebQSP Beam Reach Diagnostic
For A2 failures (gold in graph, beam missed top-10), checks what rank
the gold answer actually appears at in the extended beam output (top-30).

Key question: are answers in top-30 (re-rankable) or truly absent from beam?
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
FETCH_K = 30   # beam fetches this many candidates

def main():
    print("[BeamReach] Building WebQSP state...", flush=True)
    state = build_webqsp_state(n_questions=N_QUESTIONS, embeddings="sentence", seed=42, use_cache=True)
    graph     = state["graph"]
    qa_pairs  = state["qa_pairs"]
    deriver   = state["deriver"]
    rel_index = state.get("rel_index")
    schema_idx = state.get("schema_idx")
    nx_g      = graph.adapter.to_networkx()
    graph_nodes = set(nx_g.nodes())
    adapter   = graph.adapter

    gamma = float(PARAMS["gamma"]); beta = float(PARAMS["beta"])
    trb_factor  = float(PARAMS["trb_factor"]); fhrb_factor = float(PARAMS["fhrb_factor"])
    trb_map  = deriver.boost_map(trb_factor, beta) if deriver.is_built else {}
    fhrb_map = deriver.boost_map(fhrb_factor, beta) if deriver.is_built else {}
    schema_score_thresh = float(PARAMS["schema_score_threshold"])
    schema_top_k = int(PARAMS.get("schema_top_k", 5))

    qd = QuestionDecomposer()
    emb_engine = getattr(graph, "_embedding_engine", None)

    # Reset cross-trial state
    _platt = getattr(graph, "_platt", None)
    if _platt:
        _platt._samples = []; _platt._fitted = False
    graph._recent_answer_cache = {}
    graph._feedback_buf = []

    # rank buckets: where does the BEST gold answer appear?
    rank_buckets = collections.Counter()  # rank -> count
    not_in_beam  = 0   # gold present in graph but not in top-30 beam output at all
    hit_at_1     = 0
    total_a2     = 0   # questions where all gold answers are in graph

    print(f"[BeamReach] Running beam on all {len(qa_pairs)} questions...", flush=True)

    for qi, (seed_ent, gold_answers, question) in enumerate(qa_pairs):
        if qi % 200 == 0:
            print(f"  {qi}/{len(qa_pairs)}", flush=True)

        if not seed_ent or seed_ent not in graph_nodes:
            continue
        n_present = sum(1 for a in gold_answers if a in graph_nodes)
        if n_present == 0:
            continue   # A1: skip, not an A2 case

        total_a2 += 1
        gold_set = set(gold_answers)

        # Build query
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

        # Post-processing (same as canonical)
        if q_scores:
            for ans in answers_obj:
                bp = getattr(ans, "best_path", None)
                nodes = getattr(bp, "nodes", ()) if bp else ()
                if len(nodes) >= 2:
                    tr = nodes[-2]
                    if tr in q_scores:
                        ans.score *= (1.0 + q_scores[tr] * 2.0)
                    elif len(nodes) >= 2 and nodes[1] in q_scores:
                        ans.score *= (1.0 + q_scores[nodes[1]] * 2.0)
            answers_obj.sort(key=lambda a: a.score, reverse=True)

        # r2 boost
        boost_map = deriver.boost_map(gamma, beta) if deriver.is_built else {}
        r2_boost = float(PARAMS["r2_boost"])
        if r2_boost > 0.0 and boost_map:
            for ans in answers_obj:
                bp = getattr(ans, "best_path", None)
                if bp and len(getattr(bp, "nodes", ())) >= 2:
                    pen_rel = bp.nodes[-2] if hasattr(bp, "nodes") else None
                    if pen_rel and pen_rel in boost_map:
                        ans.score *= (1.0 + r2_boost * boost_map[pen_rel])
            answers_obj.sort(key=lambda a: a.score, reverse=True)

        # IDF
        answer_freq = state.get("answer_freq", {})
        idf_w = float(PARAMS["idf_weight"])
        if idf_w > 0.0:
            for ans in answers_obj:
                freq = answer_freq.get(ans.entity_id, 1)
                ans.score *= 1.0 / (1.0 + idf_w * math.log1p(freq))
            answers_obj.sort(key=lambda a: a.score, reverse=True)

        # Schema merge
        if schema_extra:
            beam_set = {a.entity_id for a in answers_obj[:10]}
            high_conf = [eid for eid, sc in schema_extra if sc >= schema_score_thresh and eid not in beam_set and not eid.startswith("/m/")]
            prepend = high_conf[:2]
            prepend_set = set(prepend)
            top10 = prepend + [a.entity_id for a in answers_obj[:10] if a.entity_id not in prepend_set]
        else:
            top10 = [a.entity_id for a in answers_obj[:10]]

        # Extended list up to FETCH_K * 3 for reach check
        named = [a for a in answers_obj if not str(a.entity_id).startswith("/m/")]
        full_list = [a.entity_id for a in named]

        # Find best rank of any gold answer
        best_rank = None
        for rank, eid in enumerate(full_list):
            if eid in gold_set:
                best_rank = rank + 1
                break

        if best_rank is None:
            not_in_beam += 1
            rank_buckets["not_in_beam"] += 1
        elif best_rank == 1:
            hit_at_1 += 1
            rank_buckets[1] += 1
        elif best_rank <= 10:
            rank_buckets[f"2-10"] += 1
        elif best_rank <= 30:
            rank_buckets[f"11-30"] += 1
        elif best_rank <= 100:
            rank_buckets[f"31-100"] += 1
        else:
            rank_buckets[f"101+"] += 1

    print(f"\n{'='*65}")
    print(f"WebQSP Beam Reach Diagnostic  ({total_a2} A2 questions)")
    print(f"(A2 = gold answer present in graph, beam miss)")
    print(f"{'='*65}")
    print(f"")
    print(f"Rank of best gold answer in extended beam output:")
    ordered = [1, "2-10", "11-30", "31-100", "101+", "not_in_beam"]
    for k in ordered:
        n = rank_buckets.get(k, 0)
        print(f"  Rank {str(k):12s}: {n:5d}  ({n/total_a2*100:.1f}%)")
    print(f"")
    print(f"In top-10  (currently captured): {rank_buckets.get(1,0) + rank_buckets.get('2-10',0):5d}  ({(rank_buckets.get(1,0)+rank_buckets.get('2-10',0))/total_a2*100:.1f}%)")
    print(f"In top-30  (LLM re-rank pool):   {rank_buckets.get(1,0)+rank_buckets.get('2-10',0)+rank_buckets.get('11-30',0):5d}  ({(rank_buckets.get(1,0)+rank_buckets.get('2-10',0)+rank_buckets.get('11-30',0))/total_a2*100:.1f}%)")
    print(f"In top-100 (wide beam pool):     {sum(rank_buckets.get(k,0) for k in [1,'2-10','11-30','31-100']):5d}  ({sum(rank_buckets.get(k,0) for k in [1,'2-10','11-30','31-100'])/total_a2*100:.1f}%)")
    print(f"Not in beam at all:              {not_in_beam:5d}  ({not_in_beam/total_a2*100:.1f}%)")
    print(f"")
    print(f"LLM re-ranking potential (answers in top-30 of A2 cases): {(rank_buckets.get(1,0)+rank_buckets.get('2-10',0)+rank_buckets.get('11-30',0))/len(qa_pairs)*100:.1f}pp additional H@1 theoretically achievable")


if __name__ == "__main__":
    main()

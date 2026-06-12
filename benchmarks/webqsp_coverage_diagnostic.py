"""
Phase 253b: WebQSP Coverage Diagnostic
For each question, checks whether gold answers are present in the 2-hop subgraph.
Splits A-failures (answer not in top-10) into:
  A1: answer absent from graph entirely   — coverage failure (subgraph extraction ceiling)
  A2: answer in graph but beam missed it  — traversal failure (tunable)

No beam search required — pure graph membership check on all 1628 questions.
"""
import sys, os, collections
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "benchmarks")
sys.path.insert(0, ".")

from webqsp_param_eval import build_webqsp_state

N_QUESTIONS = 1628

def main():
    print("[Coverage] Building WebQSP state...", flush=True)
    state = build_webqsp_state(n_questions=N_QUESTIONS, embeddings="sentence", seed=42, use_cache=True)
    graph    = state["graph"]
    qa_pairs = state["qa_pairs"]
    nx_g     = graph.adapter.to_networkx()
    graph_nodes = set(nx_g.nodes())
    print(f"[Coverage] {len(qa_pairs)} questions, {len(graph_nodes):,} graph nodes", flush=True)

    # Per-question coverage check
    total            = len(qa_pairs)
    no_seed          = 0   # D: seed entity not in graph
    all_answers_absent = 0  # A1: no gold answer in graph at all
    some_answers_present = 0  # A1-partial: some gold answers in graph, some not
    all_answers_present  = 0  # A2: all gold answers in graph (beam missed them)
    hit_questions    = 0   # questions where seed+gold both found (beam could in theory hit)

    hop_distances: list[int] = []  # shortest path lengths for A2 cases (answers present, beam missed)
    absent_counts: list[int] = []  # how many gold answers absent per question
    present_counts: list[int] = []

    a1_questions = []  # sample of A1 (absent) questions for inspection
    a2_questions = []  # sample of A2 (present but missed) questions for inspection

    for seed_ent, gold_answers, question in qa_pairs:
        if not seed_ent or seed_ent not in graph_nodes:
            no_seed += 1
            continue

        n_gold   = len(gold_answers)
        n_present = sum(1 for a in gold_answers if a in graph_nodes)
        n_absent  = n_gold - n_present

        present_counts.append(n_present)
        absent_counts.append(n_absent)

        if n_present == 0:
            all_answers_absent += 1
            if len(a1_questions) < 20:
                a1_questions.append((question, gold_answers))
        elif n_present == n_gold:
            all_answers_present += 1
            if len(a2_questions) < 20:
                a2_questions.append((question, gold_answers, seed_ent))
        else:
            some_answers_present += 1

        if n_present > 0:
            hit_questions += 1

    # Summary
    answerable = total - no_seed
    print(f"\n{'='*65}")
    print(f"WebQSP Coverage Diagnostic  ({total} questions)")
    print(f"{'='*65}")
    print(f"Seed not in graph (D):          {no_seed:5d}  ({no_seed/total*100:.1f}%)")
    print(f"")
    print(f"Of {answerable} questions with valid seed:")
    print(f"  A1 All gold answers ABSENT:   {all_answers_absent:5d}  ({all_answers_absent/total*100:.1f}%)")
    print(f"  A1 Some gold answers absent:  {some_answers_present:5d}  ({some_answers_present/total*100:.1f}%)")
    print(f"  A2 All gold answers PRESENT:  {all_answers_present:5d}  ({all_answers_present/total*100:.1f}%)")
    print(f"")

    # Upper bound analysis
    # H@1 ceiling if beam were perfect: questions where seed present AND >=1 gold answer present
    ceiling = hit_questions
    print(f"Theoretical H@1 ceiling (beam perfect, >=1 gold in graph): {ceiling/total*100:.1f}%")
    print(f"Current H@1: 10.57%  Gap to ceiling: {ceiling/total*100 - 10.57:.1f}pp")
    print(f"")

    # Coverage breakdown across all questions
    import statistics
    all_pc = present_counts + [0]*no_seed
    frac_with_any = sum(1 for x in all_pc if x > 0) / total
    print(f"Questions with >=1 gold answer in graph:  {sum(1 for x in all_pc if x>0):5d}  ({frac_with_any*100:.1f}%)")
    print(f"Questions with ALL gold answers in graph: {all_answers_present:5d}  ({all_answers_present/total*100:.1f}%)")
    print(f"")

    # Gold answer set sizes
    all_set_sizes = [len(g) for _, g, _ in qa_pairs]
    print(f"Gold answer set size: mean={statistics.mean(all_set_sizes):.2f}  median={statistics.median(all_set_sizes):.0f}  max={max(all_set_sizes)}")

    # Multi-answer breakdown: for questions with >1 gold answer, how many are in graph?
    multi = [(p, a) for p, a in zip(present_counts, absent_counts) if (p+a) > 1]
    if multi:
        frac_multi_any = sum(1 for p, _ in multi if p > 0) / len(multi)
        print(f"Multi-answer questions ({len(multi)}): {frac_multi_any*100:.1f}% have >=1 answer in graph")

    print(f"\n{'='*65}")
    print(f"VERDICT:")
    a1_total = all_answers_absent + some_answers_present
    print(f"  Coverage failures (A1, no/partial answer in graph): {a1_total:5d}  ({a1_total/total*100:.1f}%)")
    print(f"  Traversal failures (A2, answer in graph, beam miss): {all_answers_present:5d}  ({all_answers_present/total*100:.1f}%)")
    print(f"")
    print(f"  -> {'Coverage' if a1_total > all_answers_present else 'Traversal'} is the dominant failure mode.")

    # Sample A1 questions (absent)
    print(f"\nSample A1 questions (gold answers absent from graph):")
    for q, g in a1_questions[:5]:
        print(f"  Q: {q[:80]}")
        print(f"     gold: {g[:3]}")

    # Sample A2 questions (present but missed)
    print(f"\nSample A2 questions (gold answers in graph, beam missed):")
    for q, g, s in a2_questions[:5]:
        print(f"  Q: {q[:80]}")
        print(f"     seed: {s}  gold: {g[:3]}")


if __name__ == "__main__":
    main()

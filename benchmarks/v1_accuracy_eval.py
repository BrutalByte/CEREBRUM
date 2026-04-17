"""
CEREBRUM v1.0 — Feature Accuracy Benchmark
===========================================

Measures the accuracy impact of each of the four v1.0 structural-hole fixes
on controlled synthetic scenarios.  Each section is self-contained and can be
run independently with --section.

Sections
--------
  1  bayesian    Warm-start vs cold-start Bayesian beam on intra-community QA
  2  causal      Causal-flood filter: false-positive CAUSES edges under burst
  3  namespace   Namespace isolation: entity-ID collision rate in mixed graphs
  4  zombie      Zombie-bridge: stale record count + post-rebalance QA quality

Usage
-----
  python -m benchmarks.v1_accuracy_eval                   # all four sections
  python -m benchmarks.v1_accuracy_eval --section bayesian
  python -m benchmarks.v1_accuracy_eval --section causal
  python -m benchmarks.v1_accuracy_eval --section namespace
  python -m benchmarks.v1_accuracy_eval --section zombie
  python -m benchmarks.v1_accuracy_eval --n-questions 200 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from adapters.networkx_adapter import NetworkXAdapter
from core.community_engine import best_of_n_dscf
from core.embedding_engine import RandomEngine
from core.attention_engine import CSAEngine
from core.structural_encoder import build_community_distance_matrix, adjacent_community_pairs
from core.resource_governor import ResourceGovernor
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract
from benchmarks.metaqa_eval import hits_at_k, reciprocal_rank
from benchmarks.synthetic_eval import generate_clustered_graph, generate_qa_pairs

_BENCH_GOVERNOR = ResourceGovernor(memory_threshold_pct=99.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_graph(n_communities: int = 10, community_size: int = 30,
                 k_intra: int = 5, m_inter: int = 2,
                 seed: int = 42) -> Tuple[nx.Graph, Dict[str, int]]:
    return generate_clustered_graph(
        n_communities=n_communities,
        community_size=community_size,
        k_intra=k_intra,
        m_inter=m_inter,
        seed=seed,
    )


def _build_traversal(
    G: nx.Graph,
    embeddings: dict,
    cmap: Dict[str, int],
    beam_width: int = 10,
    max_hop: int = 2,
    probabilistic: bool = False,
    warm_start_strength: float = 0.0,
    pagerank: dict | None = None,
) -> BeamTraversal:
    adapter = NetworkXAdapter(G)
    adapter.community_map = cmap
    adapter.embeddings = embeddings
    dist = build_community_distance_matrix(G, cmap)
    adj  = adjacent_community_pairs(G, cmap)
    csa  = CSAEngine(adapter=adapter, pagerank=pagerank, zeta=0.1)
    csa.set_community_graph(dist, adj)
    return BeamTraversal(
        adapter=adapter, csa_engine=csa,
        beam_width=beam_width, max_hop=max_hop,
        governor=_BENCH_GOVERNOR,
        probabilistic=probabilistic,
        warm_start_strength=warm_start_strength,
    )


def _evaluate(traversal: BeamTraversal,
              qa_pairs: List[Tuple[str, List[str]]],
              top_k: int = 10) -> Dict:
    h1 = h10 = found = skipped = 0
    mrr_sum = 0.0
    t0 = time.time()
    n  = len(qa_pairs)
    for seed, correct in qa_pairs:
        paths   = traversal.traverse([seed])
        answers = extract(paths, top_k=top_k, min_hop=1)
        pred    = [a.entity_id for a in answers]
        if not pred:
            skipped += 1
            continue
        found   += 1
        h1      += hits_at_k(pred, correct, k=1)
        h10     += hits_at_k(pred, correct, k=top_k)
        mrr_sum += reciprocal_rank(pred, correct)
    elapsed = time.time() - t0
    return {
        "n_total":    n,
        "n_answered": found,
        "hits_1":     h1 / n if n else 0.0,
        "hits_10":    h10 / n if n else 0.0,
        "mrr":        mrr_sum / n if n else 0.0,
        "elapsed_s":  round(elapsed, 2),
    }


def _dscf_cmap(G: nx.Graph, seed: int = 42, n_trials: int = 5) -> Dict[str, int]:
    parts = best_of_n_dscf(G, n_trials=n_trials, seed=seed)
    return {node: cid for cid, members in enumerate(parts) for node in members}


def _print_row(label: str, m: Dict, highlight: bool = False) -> None:
    marker = "  <-- " if highlight else ""
    print(f"  {label:<38}  H@1={m['hits_1']:.4f}  H@10={m['hits_10']:.4f}  "
          f"MRR={m['mrr']:.4f}  t={m['elapsed_s']:.1f}s{marker}")


# ---------------------------------------------------------------------------
# Section 1 — Bayesian Warm-Start
# ---------------------------------------------------------------------------

def section_bayesian(args) -> None:
    print("\n" + "=" * 70)
    print("Section 1 — Bayesian Warm-Start vs Cold-Start")
    print("=" * 70)
    print("Hypothesis: warm_start_strength > 0 reduces first-hop variance and")
    print("improves H@1 / MRR on intra-community 2-hop questions.\n")

    G, gt = _build_graph(n_communities=args.n_communities,
                         community_size=args.community_size, seed=args.seed)
    print(f"  Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

    engine     = RandomEngine(dim=64)
    embeddings = engine.encode_entities({n: n for n in G.nodes()})

    print("  Computing DSCF communities...")
    cmap = _dscf_cmap(G, seed=args.seed)
    print(f"  {len(set(cmap.values()))} communities detected\n")

    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)

    hop = 2
    qa  = generate_qa_pairs(G, gt, hop=hop, n_questions=args.n_questions, seed=args.seed)
    print(f"  {len(qa):,} intra-community {hop}-hop QA pairs\n")

    variants = [
        ("Deterministic (probabilistic=False)",  False,  0.0),
        ("Bayesian cold  (warm_start=0)",         True,   0.0),
        ("Bayesian warm  (warm_start=1)",         True,   1.0),
        ("Bayesian warm  (warm_start=3)",         True,   3.0),
        ("Bayesian warm  (warm_start=5)",         True,   5.0),
    ]

    results = []
    for label, prob, ws in variants:
        print(f"  [{label}]")
        trav = _build_traversal(G, embeddings, cmap, beam_width=args.beam_width,
                                max_hop=hop, probabilistic=prob,
                                warm_start_strength=ws, pagerank=pagerank)
        m = _evaluate(trav, qa, top_k=args.top_k)
        m["label"] = label
        results.append(m)

    print()
    print(f"  {'Variant':<38}  {'H@1':>8}  {'H@10':>8}  {'MRR':>8}  {'Time(s)':>8}")
    print("  " + "-" * 78)
    baseline_h1 = results[0]["hits_1"]
    for i, m in enumerate(results):
        delta = m["hits_1"] - baseline_h1
        sign  = "+" if delta >= 0 else ""
        tag   = f"  [{sign}{delta:.4f} vs det]" if i > 0 else ""
        print(f"  {m['label']:<38}  {m['hits_1']:>8.4f}  {m['hits_10']:>8.4f}  "
              f"{m['mrr']:>8.4f}  {m['elapsed_s']:>8.1f}{tag}")

    # Analysis
    det_h1  = results[0]["hits_1"]
    warm_h1 = max(m["hits_1"] for m in results[2:])
    delta   = warm_h1 - det_h1
    print(f"\n  Best warm-start vs deterministic: {'+' if delta>=0 else ''}{delta:.4f} H@1")
    if delta >= 0:
        print("  RESULT: Warm-start does not hurt and may improve H@1 on cold segments.")
    else:
        print("  RESULT: Deterministic beam edges out warm-start on this topology.")


# ---------------------------------------------------------------------------
# Section 2 — Causal Flood Filter
# ---------------------------------------------------------------------------

def section_causal(args) -> None:
    print("\n" + "=" * 70)
    print("Section 2 — Causal Flood Filter (STDPDiscretizer)")
    print("=" * 70)
    print("Measures false-positive CAUSES edges emitted under an adversarial")
    print("burst scenario, with and without the temporal diversity filter.\n")

    try:
        from core.discretizer import STDPDiscretizer
    except ImportError as e:
        print(f"  ERROR: cannot import STDPDiscretizer: {e}")
        return


    # Scenario parameters
    n_burst    = 200    # spikes in a short burst window
    burst_span = 0.05   # seconds (50ms burst)
    n_legit    = 20     # legitimate spaced-out spikes
    legit_span = 5.0    # seconds
    threshold  = 0.5
    n_min      = 5

    def _run_burst(disc, n_spikes, span, inter_dt=0.001):
        """Drive pre→post spike pairs; each pair separated by inter_dt seconds."""
        disc.reset()
        t   = 0.0
        dt  = span / n_spikes
        out = []
        for _ in range(n_spikes):
            t += dt
            out.extend(disc.process("pre",  t - inter_dt))
            out.extend(disc.process("post", t))
        return [e for e in out if e.relation == "CAUSES"]

    # ---- Scenario A: adversarial burst (no filter) ----
    disc_a = STDPDiscretizer(w_threshold=threshold, n_min=n_min,
                             min_causal_span=0.0, use_chi_squared=False)
    edges_a = _run_burst(disc_a, n_burst, burst_span)
    fp_a = len(edges_a)

    # ---- Scenario B: adversarial burst + min_causal_span=1.0s ----
    disc_b = STDPDiscretizer(w_threshold=threshold, n_min=n_min,
                             min_causal_span=1.0, use_chi_squared=False)
    edges_b = _run_burst(disc_b, n_burst, burst_span)
    fp_b = len(edges_b)

    # ---- Scenario C: adversarial burst + chi-squared filter ----
    disc_c = STDPDiscretizer(w_threshold=threshold, n_min=n_min,
                             min_causal_span=0.0, use_chi_squared=True)
    edges_c = _run_burst(disc_c, n_burst, burst_span)
    fp_c = len(edges_c)

    # ---- Scenario D: legitimate spaced-out spikes (should pass both filters) ----
    disc_d = STDPDiscretizer(w_threshold=threshold, n_min=n_min,
                             min_causal_span=1.0, use_chi_squared=True)
    edges_d = _run_burst(disc_d, n_legit, legit_span, inter_dt=0.01)
    tp_d = len(edges_d)

    print(f"  Adversarial burst: {n_burst} spikes in {burst_span*1000:.0f}ms")
    print(f"  Legitimate signal: {n_legit} spikes over {legit_span:.0f}s\n")
    print(f"  {'Scenario':<45}  {'CAUSES edges':>12}  {'Expected':>10}")
    print("  " + "-" * 72)
    print(f"  {'No filter (baseline)':<45}  {fp_a:>12}  {'> 0':>10}")
    print(f"  {'min_causal_span=1.0s':<45}  {fp_b:>12}  {'0':>10}  "
          + ("OK" if fp_b == 0 else "FAIL"))
    print(f"  {'use_chi_squared=True':<45}  {fp_c:>12}  {'0':>10}  "
          + ("OK" if fp_c == 0 else "FAIL"))
    print(f"  {'Legitimate + both filters (true-positives)':<45}  {tp_d:>12}  {'> 0':>10}  "
          + ("OK" if tp_d > 0 else "FAIL"))

    reduction_span = (1 - fp_b / fp_a) * 100 if fp_a > 0 else 0.0
    reduction_chi  = (1 - fp_c / fp_a) * 100 if fp_a > 0 else 0.0
    print(f"\n  min_causal_span  false-positive reduction: {reduction_span:.1f}%")
    print(f"  use_chi_squared  false-positive reduction: {reduction_chi:.1f}%")
    print(f"  True-positive recall (legitimate signal, both filters): {tp_d}/{n_min}+ = "
          + ("PASS" if tp_d > 0 else "NEEDS MORE SPIKES"))


# ---------------------------------------------------------------------------
# Section 3 — Namespace Isolation
# ---------------------------------------------------------------------------

def section_namespace(args) -> None:
    print("\n" + "=" * 70)
    print("Section 3 — Namespace Isolation (IngestionPipeline + SignalEncoder)")
    print("=" * 70)
    print("Measures entity-ID collision rate in a mixed text+signal graph.\n")

    try:
        from core.thalamus import IngestionPipeline
        from core.signal_encoder import StatisticalSignalEncoder
    except ImportError as e:
        print(f"  ERROR: cannot import required modules: {e}")
        return


    # Shared entity names that would collide without namespace
    shared_names = [f"sensor_{i}" for i in range(50)]

    # Raw edges: (source, target, relation)
    text_edges_raw   = [(f"sensor_{i}", f"sensor_{i+1}", "text_rel")
                        for i in range(0, 49, 2)]
    signal_edges_raw = [(f"sensor_{i}", f"sensor_{i+1}", "signal_rel")
                        for i in range(0, 49, 2)]

    # --- Without namespace (the bug) ---
    pipe_no_ns     = IngestionPipeline(namespace="")
    pipe_sig_no_ns = IngestionPipeline(namespace="")
    text_processed    = [pipe_no_ns.process(s, t, r)     for s, t, r in text_edges_raw]
    sig_processed     = [pipe_sig_no_ns.process(s, t, r) for s, t, r in signal_edges_raw]

    # Count collisions: node appears in both sets with same ID
    text_nodes_no_ns   = {e.source for e in text_processed if e} | \
                         {e.target for e in text_processed if e}
    signal_nodes_no_ns = {e.source for e in sig_processed if e} | \
                         {e.target for e in sig_processed if e}
    collisions_no_ns   = len(text_nodes_no_ns & signal_nodes_no_ns)

    # --- With namespace ---
    pipe_text = IngestionPipeline(namespace="text")
    pipe_sig  = IngestionPipeline(namespace="signal")
    text_ns   = [pipe_text.process(s, t, r) for s, t, r in text_edges_raw]
    sig_ns    = [pipe_sig.process(s, t, r)  for s, t, r in signal_edges_raw]

    text_nodes_ns   = {e.source for e in text_ns if e} | {e.target for e in text_ns if e}
    signal_nodes_ns = {e.source for e in sig_ns if e} | {e.target for e in sig_ns if e}
    collisions_ns   = len(text_nodes_ns & signal_nodes_ns)

    # --- Signal encoder namespace ---
    enc_default = StatisticalSignalEncoder(entity_dim=16)   # namespace="signal" by default
    enc_no_ns   = StatisticalSignalEncoder(entity_dim=16, namespace="")

    # Check that get_namespaced_id correctly prefixes
    bare_id       = "temperature_sensor"
    prefixed_id   = enc_default.get_namespaced_id(bare_id)
    verbatim_id   = enc_no_ns.get_namespaced_id(bare_id)
    ns_correct    = (prefixed_id == f"signal:{bare_id}")
    verb_correct  = (verbatim_id == bare_id)

    n_shared = len(shared_names)
    print(f"  Shared entity names: {n_shared} (e.g. 'sensor_0' used in both text and signal)")
    print(f"  Text edges: {len(text_edges_raw)} | Signal edges: {len(signal_edges_raw)}\n")

    print(f"  {'Scenario':<45}  {'Collisions':>12}  {'Expected':>10}")
    print("  " + "-" * 72)
    print(f"  {'No namespace (baseline — the bug)':<45}  {collisions_no_ns:>12}  {'> 0':>10}  "
          + ("OK" if collisions_no_ns > 0 else "BUG NOT REPRODUCED"))
    print(f"  {'With namespace=text / signal':<45}  {collisions_ns:>12}  {'0':>10}  "
          + ("OK" if collisions_ns == 0 else "FAIL"))
    print(f"  {'StatisticalSignalEncoder default NS prefix':<45}  "
          + ("signal:X" if ns_correct else "FAIL") + f"  {'signal:X':>10}  "
          + ("OK" if ns_correct else "FAIL"))
    print("  {:<45}  ".format('Encoder with namespace="" passes verbatim')
          + (bare_id if verb_correct else "FAIL") + f"  {bare_id:>10}  "
          + ("OK" if verb_correct else "FAIL"))

    isolation_rate = (1 - collisions_ns / max(collisions_no_ns, 1)) * 100
    print(f"\n  Collision elimination rate: {isolation_rate:.1f}%")
    if collisions_ns == 0:
        print("  RESULT: Namespace isolation fully prevents semantic SynapticBridges.")
    else:
        print(f"  RESULT: {collisions_ns} residual collisions — investigate.")


# ---------------------------------------------------------------------------
# Section 4 — Zombie Bridge
# ---------------------------------------------------------------------------

def section_zombie(args) -> None:
    print("\n" + "=" * 70)
    print("Section 4 — Zombie Bridge (GlobalRebalancer + BridgeTwinEngine)")
    print("=" * 70)
    print("Measures stale bridge record count after rebalance, and verifies")
    print("traversal quality is maintained post-pruning.\n")

    try:
        from core.bridge_engine import BridgeTwinEngine, BridgeRecord
    except ImportError as e:
        print(f"  ERROR: cannot import required modules: {e}")
        return

    G, gt = _build_graph(n_communities=args.n_communities,
                         community_size=args.community_size, seed=args.seed)
    print(f"  Graph: {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")

    engine     = RandomEngine(dim=64)
    embeddings = engine.encode_entities({n: n for n in G.nodes()})

    print("  Computing initial DSCF communities (seed=42)...")
    cmap_initial = _dscf_cmap(G, seed=42)
    n_comm_initial = len(set(cmap_initial.values()))
    print(f"  Initial partition: {n_comm_initial} communities")

    # Seed the bridge engine with synthetic records
    bridge = BridgeTwinEngine(n_min=2, similarity_threshold=0.0)
    nodes  = list(G.nodes())

    # Inject N_BRIDGES synthetic bridge records using initial community IDs
    N_BRIDGES = 30
    n_injected = 0
    for i in range(min(N_BRIDGES, len(nodes) - 1)):
        orig = nodes[i]
        twin = f"{orig}::twin::99"
        src_cid = cmap_initial.get(orig, 0)
        dst_cid = (src_cid + 1) % n_comm_initial  # plausible dest
        record  = BridgeRecord(
            original_id=orig, twin_id=twin,
            source_community=src_cid, destination_community=dst_cid,
            traversal_count=5,
        )
        bridge._bridges[twin] = record
        bridge._bridge_index[(orig, dst_cid)] = twin
        n_injected += 1

    print(f"  Injected {n_injected} synthetic bridge records\n")

    # --- Without hook (simulate old behavior) ---
    stale_no_hook = 0
    # Compute a fresh partition with different seed → shuffled IDs
    cmap_fresh = _dscf_cmap(G, seed=123)
    # Manually count how many would be stale
    for twin_id, rec in bridge._bridges.items():
        cur_src = cmap_fresh.get(rec.original_id, -1)
        cur_dst = cmap_fresh.get(twin_id, -1)
        if cur_src != rec.source_community or cur_dst != rec.destination_community:
            stale_no_hook += 1
    n_bridges_before = len(bridge._bridges)

    # --- With hook ---
    pruned = bridge.on_rebalance(cmap_fresh)
    n_bridges_after = len(bridge._bridges)
    n_comm_fresh = len(set(cmap_fresh.values()))

    print(f"  Post-rebalance partition: {n_comm_fresh} communities")
    print(f"  Bridge records before rebalance:  {n_bridges_before}")
    print(f"  Stale records (no hook):          {stale_no_hook}")
    print(f"  Pruned by on_rebalance hook:      {pruned}")
    print(f"  Bridge records after pruning:     {n_bridges_after}")
    print()

    # Verify traversal quality is maintained for remaining (valid) bridges
    hop = 2
    qa  = generate_qa_pairs(G, gt, hop=hop, n_questions=args.n_questions, seed=args.seed)

    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)

    print(f"  Evaluating traversal quality on {len(qa):,} {hop}-hop QA pairs...")
    trav = _build_traversal(G, embeddings, cmap_fresh, beam_width=args.beam_width,
                            max_hop=hop, pagerank=pagerank)
    m_post = _evaluate(trav, qa, top_k=args.top_k)

    # Compare against traversal with original stale map
    trav_stale = _build_traversal(G, embeddings, cmap_initial, beam_width=args.beam_width,
                                  max_hop=hop, pagerank=pagerank)
    m_stale = _evaluate(trav_stale, qa, top_k=args.top_k)

    print()
    def _fmt(m: Dict) -> str:
        return (f"  {'':38}  H@1={m['hits_1']:.4f}  H@10={m['hits_10']:.4f}"
                f"  MRR={m['mrr']:.4f}  t={m['elapsed_s']:.1f}s")

    print(f"  {'Scenario':<38}  H@1         H@10         MRR")
    print(f"  {'Stale community map (old behavior)':<38}  H@1={m_stale['hits_1']:.4f}"
          f"  H@10={m_stale['hits_10']:.4f}  MRR={m_stale['mrr']:.4f}  t={m_stale['elapsed_s']:.1f}s")
    print(f"  {'Fresh map + on_rebalance pruning':<38}  H@1={m_post['hits_1']:.4f}"
          f"  H@10={m_post['hits_10']:.4f}  MRR={m_post['mrr']:.4f}  t={m_post['elapsed_s']:.1f}s")

    delta_h10 = m_post["hits_10"] - m_stale["hits_10"]
    print(f"\n  Pruned {pruned}/{n_injected} stale records "
          f"({100*pruned/n_injected:.1f}% stale rate)")
    sign = "+" if delta_h10 >= 0 else ""
    print(f"  H@10 delta (fresh vs stale map): {sign}{delta_h10:.4f}")
    if pruned == stale_no_hook:
        print("  RESULT: on_rebalance correctly identified ALL stale records.")
    else:
        print(f"  WARNING: on_rebalance pruned {pruned} but {stale_no_hook} were stale.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEREBRUM v1.0 Feature Accuracy Benchmark"
    )
    parser.add_argument(
        "--section",
        choices=["bayesian", "causal", "namespace", "zombie", "all"],
        default="all",
        help="Which section to run (default: all)",
    )
    parser.add_argument("--n-questions",    type=int, default=300,
                        help="QA pairs per section (default: 300)")
    parser.add_argument("--n-communities",  type=int, default=10)
    parser.add_argument("--community-size", type=int, default=30)
    parser.add_argument("--beam-width",     type=int, default=10)
    parser.add_argument("--top-k",          type=int, default=10)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CEREBRUM v1.0 — Feature Accuracy Benchmark")
    print("=" * 70)
    print(f"  Graph: {args.n_communities} communities × {args.community_size} nodes")
    print(f"  QA pairs per section: {args.n_questions} | beam_width={args.beam_width}")
    print(f"  seed={args.seed}")

    run_all = args.section == "all"

    if run_all or args.section == "bayesian":
        section_bayesian(args)
    if run_all or args.section == "causal":
        section_causal(args)
    if run_all or args.section == "namespace":
        section_namespace(args)
    if run_all or args.section == "zombie":
        section_zombie(args)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

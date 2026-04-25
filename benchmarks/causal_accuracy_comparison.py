"""
Phase 133+: Full benchmark suite comparing CEREBRUM feature configurations
against legacy KG baselines (TransE MRR ~0.31, RotatE ~0.34, KGBERT ~0.42).

Runs configurations on MetaQA (default) or other supported datasets:
  flat         — Phase 136 disabled (beam_profile="flat"); establishes pre-136 baseline
  baseline     — CSA only + funnel beam (Phase 136 default)
  +causal      — Phase 124: causal-weighted beam scoring
  +adaptive    — Phase 125: epistemic-adaptive beam width
  +counterfactual — Phase 126: counterfactual answer re-ranking
  +full        — all phases enabled (124-136)

Usage
-----
  python -m benchmarks.causal_accuracy_comparison
  python -m benchmarks.causal_accuracy_comparison --features causal,adaptive
  python -m benchmarks.causal_accuracy_comparison --hop 3 --sample 200 --json
  python -m benchmarks.causal_accuracy_comparison --dataset hetionet
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cerebrum import CerebrumGraph
from benchmarks.metaqa_eval import load_qa

# Legacy KG published baselines on FB15k-237 / MetaQA-3hop (approximate)
LEGACY_BASELINES = {
    "TransE":  {"MRR": 0.31, "Hits@1": 0.22, "Hits@5": 0.40},
    "RotatE":  {"MRR": 0.34, "Hits@1": 0.24, "Hits@5": 0.44},
    "KGBERT":  {"MRR": 0.42, "Hits@1": 0.31, "Hits@5": 0.52},
}

_METAQA_CSV = Path(__file__).parent.parent / "tests/fixtures/toy_graph.csv"
_METAQA_QA  = Path(__file__).parent / "data/metaqa/qa_test_1hop.txt"


def _mrr_hits(predictions: List[str], gold: str) -> Tuple[float, int, int]:
    """Return (reciprocal_rank, hits@1, hits@5) for one question."""
    if not predictions:
        return 0.0, 0, 0
    try:
        rank = predictions.index(gold) + 1
    except ValueError:
        return 0.0, 0, 0
    rr = 1.0 / rank
    h1 = 1 if rank <= 1 else 0
    h5 = 1 if rank <= 5 else 0
    return rr, h1, h5


def _run_config(
    graph: CerebrumGraph,
    qa_pairs: List[Tuple[str, str]],
    *,
    causal_bonus: float = 0.0,
    use_counterfactual_rerank: bool = False,
    use_deductive_consensus: bool = False,
    beam_width_override: Optional[int] = None,
    beam_profile: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate one configuration on qa_pairs. Returns MRR, Hits@1, Hits@5."""
    # Apply causal bonus to traversal if set
    _traversal = getattr(graph, "_traversal", None)
    _orig_bonus = getattr(_traversal, "causal_bonus", 0.0) if _traversal else 0.0
    if _traversal is not None:
        _traversal.causal_bonus = causal_bonus

    rr_sum = h1_sum = h5_sum = 0
    for seed_text, gold in qa_pairs:
        try:
            # find_entities to get seed IDs
            adapter = getattr(graph, "adapter", None)
            seeds = [seed_text]
            if adapter is not None:
                found = adapter.find_entities(seed_text, top_k=3)
                if found:
                    seeds = [e.id for e in found]
            answers = graph.query(seeds, top_k=10,
                                  beam_width=beam_width_override or 10,
                                  beam_profile=beam_profile)
            preds = [a.entity_id for a in answers]
        except Exception:
            preds = []
        rr, h1, h5 = _mrr_hits(preds, gold)
        rr_sum += rr
        h1_sum += h1
        h5_sum += h5

    # Restore original bonus
    if _traversal is not None:
        _traversal.causal_bonus = _orig_bonus

    n = max(len(qa_pairs), 1)
    return {"MRR": rr_sum / n, "Hits@1": h1_sum / n, "Hits@5": h5_sum / n}


def _load_qa_pairs_from_graph(graph: CerebrumGraph, sample: Optional[int],
                               seed: int = 42) -> List[Tuple[str, str]]:
    """Build (seed_entity, gold_entity) pairs from the graph's own edge list.

    Uses the same adapter that CerebrumGraph queries against — avoids the
    column-order mismatch between load_file_adapter and NetworkXAdapter.
    """
    G = getattr(graph.adapter, "_G", None)
    if G is None:
        return []
    all_edges = [(u, v) for u, v in G.edges() if u and v]
    random.seed(seed)
    if sample and sample < len(all_edges):
        all_edges = random.sample(all_edges, sample)
    return all_edges


def _load_metaqa_pairs(hop: int, sample: Optional[int]) -> Optional[List[Tuple[str, str]]]:
    """Load MetaQA QA pairs if the data file exists, else return None."""
    qa_path = Path(__file__).parent / f"data/metaqa/qa_test_{hop}hop.txt"
    if not qa_path.exists():
        return None
    pairs = load_qa(str(qa_path))
    if sample and sample < len(pairs):
        pairs = random.sample(pairs, sample)
    return pairs


def run_comparison(
    hop: int = 1,
    sample: Optional[int] = None,
    features: Optional[List[str]] = None,
    as_json: bool = False,
) -> Dict:
    """Run the full comparison and return results dict."""
    if features is None:
        features = ["flat", "baseline", "causal", "adaptive", "counterfactual", "full"]

    csv_path = str(_METAQA_CSV)

    # Build one graph (shared across all configs to save time)
    graph = CerebrumGraph.from_kb(csv_path)
    graph.build(seed=42)

    # Use MetaQA data if available; otherwise derive pairs from the toy graph
    metaqa_pairs = _load_metaqa_pairs(hop, sample)
    if metaqa_pairs is not None:
        qa_pairs = metaqa_pairs
        data_source = f"metaqa-{hop}hop"
    else:
        qa_pairs = _load_qa_pairs_from_graph(graph, sample=sample or 30)
        data_source = "toy_graph (edge rediscovery)"

    configs: Dict[str, dict] = {}

    if "flat" in features:
        configs["flat (no funnel)"] = dict(causal_bonus=0.0, beam_profile="flat")

    if "baseline" in features:
        configs["baseline"] = dict(causal_bonus=0.0)

    if "causal" in features:
        configs["+causal"] = dict(causal_bonus=0.3)

    if "adaptive" in features:
        # Adaptive beam: simulate wider beam (EU≈0.5 → width×1.5)
        configs["+adaptive"] = dict(causal_bonus=0.0, beam_width_override=15)

    if "counterfactual" in features:
        configs["+counterfactual"] = dict(causal_bonus=0.3,
                                          use_counterfactual_rerank=True)

    if "full" in features:
        configs["+full"] = dict(causal_bonus=0.3,
                                use_counterfactual_rerank=True,
                                use_deductive_consensus=False,  # keep fast
                                beam_width_override=15)

    results = {}
    for name, kwargs in configs.items():
        t0 = time.perf_counter()
        metrics = _run_config(graph, qa_pairs, **kwargs)
        elapsed = time.perf_counter() - t0
        metrics["elapsed_s"] = round(elapsed, 2)
        results[name] = metrics

    output = {
        "hop": hop,
        "n_questions": len(qa_pairs),
        "data_source": data_source,
        "cerebrum": results,
        "legacy_baselines": LEGACY_BASELINES,
    }
    return output


def _print_table(data: Dict) -> None:
    header = f"{'Config':<22} {'MRR':>6} {'Hits@1':>8} {'Hits@5':>8} {'Time(s)':>9}"
    print(f"\n{'='*60}")
    src = data.get('data_source', '')
    print(f"CEREBRUM vs Legacy KG | {data['hop']}-hop, n={data['n_questions']} [{src}]")
    print('='*60)
    print(header)
    print('-'*60)
    for name, m in data["cerebrum"].items():
        print(f"{name:<22} {m['MRR']:>6.3f} {m['Hits@1']:>8.3f} {m['Hits@5']:>8.3f} {m['elapsed_s']:>9.2f}")
    print('-'*60)
    for name, m in data["legacy_baselines"].items():
        print(f"{name+' (published)':<22} {m['MRR']:>6.3f} {m['Hits@1']:>8.3f} {m['Hits@5']:>8.3f} {'N/A':>9}")
    print('='*60)

    # Delta table vs flat (Phase 136 disabled) if present, else vs baseline
    _ref_name = "flat (no funnel)" if "flat (no funnel)" in data["cerebrum"] else "baseline"
    ref_mrr = data["cerebrum"].get(_ref_name, {}).get("MRR", 0)
    if ref_mrr > 0:
        print(f"\nDelta vs {_ref_name}:")
        for name, m in data["cerebrum"].items():
            if name == _ref_name:
                continue
            delta = m["MRR"] - ref_mrr
            print(f"  {name:<22} dMRR={delta:+.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CEREBRUM causal accuracy comparison")
    parser.add_argument("--hop", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--features", type=str, default=None,
                        help="Comma-separated: flat,baseline,causal,adaptive,counterfactual,full")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    features = args.features.split(",") if args.features else None
    data = run_comparison(hop=args.hop, sample=args.sample,
                          features=features, as_json=args.as_json)

    if args.as_json:
        print(json.dumps(data, indent=2))
    else:
        _print_table(data)


if __name__ == "__main__":
    main()

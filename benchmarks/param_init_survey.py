"""
Phase 205: Multi-KB ParameterInitializer Survey.

Runs ParameterInitializer on every available KB and reports:
  - GraphProfiler regime classification
  - Raw graph statistics (degree_cv, hub_score, mean_rel_coverage, n_rels, etc.)
  - Fan-out statistics (max, mean, harmonic)
  - ParameterInitializer predictions for each param
  - Side-by-side diff vs. any known validated values

Usage:
    python -u benchmarks/param_init_survey.py

Output: benchmarks/data/param_init_survey.csv + console table
"""
from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.graph_profiler import GraphProfiler
from core.parameter_initializer import ParameterInitializer
from core.relation_boost_deriver import RelationBoostDeriver

DATA_DIR  = Path(__file__).parent / "data"
OUT_FILE  = DATA_DIR / "param_init_survey.csv"

# ---------------------------------------------------------------------------
# Known validated params (from tuner runs) — used to compute prediction error
# ---------------------------------------------------------------------------

VALIDATED = {
    "metaqa_3hop": {
        "trb_factor": 21.486, "gamma": 8.7319, "beta": 2.0846,
        "r2_boost": 8.185, "fhrb_factor": 3.260, "idf_weight": 0.058,
        "vote_weight": 0.764, "branch_bonus": 0.482, "beam_width": 12,
        "h1": 0.6036, "source": "Phase 204 full 14,274-question validation",
    },
}


# ---------------------------------------------------------------------------
# KB loaders
# ---------------------------------------------------------------------------

def _load_metaqa():
    kb_file = DATA_DIR / "metaqa" / "kb.txt"
    if not kb_file.exists():
        return None, None, "kb.txt not found"
    import networkx as nx
    G = nx.Graph()
    with open(kb_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                h, r, t = parts
                G.add_edge(h, t, relation=r)
    deriver = RelationBoostDeriver()
    deriver.build_from_file(str(kb_file))
    return G, deriver, None


def _load_hetionet():
    import pickle
    pkl = DATA_DIR / "hetionet" / "hetionet_graph.pkl"
    if not pkl.exists():
        return None, None, "hetionet_graph.pkl not found"
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    G = obj[0] if isinstance(obj, tuple) else obj
    deriver = RelationBoostDeriver()
    deriver.build_from_triples(
        (u, d["relation"], v) for u, v, d in G.edges(data=True)
    )
    return G, deriver, None


def _load_synthetic(n_communities=20, community_size=50, p_in=0.15, p_out=0.005,
                    n_relations=6):
    """Generate a planted-partition synthetic KB with typed relations."""
    import networkx as nx, random
    random.seed(42)
    n = n_communities * community_size
    G = nx.Graph()
    G.add_nodes_from(range(n))
    relations_in  = [f"rel_intra_{i}" for i in range(n_relations // 2)]
    relations_out = [f"rel_inter_{i}" for i in range(n_relations - n_relations // 2)]

    for c in range(n_communities):
        nodes = list(range(c * community_size, (c + 1) * community_size))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < p_in:
                    r = random.choice(relations_in)
                    G.add_edge(nodes[i], nodes[j], relation=r)
    # inter-community (sparse)
    all_nodes = list(range(n))
    for _ in range(int(p_out * n * (n - 1) / 2)):
        u, v = random.sample(all_nodes, 2)
        if not G.has_edge(u, v):
            r = random.choice(relations_out)
            G.add_edge(u, v, relation=r)

    deriver = RelationBoostDeriver()
    deriver.build_from_triples(
        (str(u), d["relation"], str(v)) for u, v, d in G.edges(data=True)
    )
    return G, deriver, None


def _load_synthetic_hub(n_hubs=5, n_spokes=1000, edges_per_spoke=3, n_relations=4):
    """Generate a star/hub-and-spoke synthetic KB."""
    import networkx as nx, random
    random.seed(7)
    G = nx.Graph()
    relations = [f"rel_{i}" for i in range(n_relations)]
    hubs = [f"hub_{i}" for i in range(n_hubs)]
    spokes = [f"spoke_{i}" for i in range(n_spokes)]
    for s in spokes:
        targets = random.sample(hubs, min(edges_per_spoke, n_hubs))
        for h in targets:
            r = random.choice(relations)
            G.add_edge(s, h, relation=r)
    # Hub-to-hub links
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            G.add_edge(hubs[i], hubs[j], relation=random.choice(relations))
    deriver = RelationBoostDeriver()
    deriver.build_from_triples(
        (u, d["relation"], v) for u, v, d in G.edges(data=True)
    )
    return G, deriver, None


# ---------------------------------------------------------------------------
# Survey runner
# ---------------------------------------------------------------------------

@dataclass
class KBEntry:
    name:             str
    n_nodes:          int
    n_edges:          int
    n_relations:      int
    mean_degree:      float
    degree_cv:        float
    hub_score:        float
    mean_rel_cov:     float
    max_fan_out:      float
    harmonic_fan_out: float
    regime_raw:       str
    regime_eff:       str
    # predictions
    p_trb_factor:   float
    p_gamma:        float
    p_beta:         float
    p_r2_boost:     float
    p_fhrb_factor:  float
    p_idf_weight:   float
    p_vote_weight:  float
    p_branch_bonus: float
    # validated (if known)
    v_trb_factor:   Optional[float] = None
    v_gamma:        Optional[float] = None
    v_beta:         Optional[float] = None
    v_r2_boost:     Optional[float] = None
    v_fhrb_factor:  Optional[float] = None
    v_idf_weight:   Optional[float] = None
    v_vote_weight:  Optional[float] = None
    v_branch_bonus: Optional[float] = None
    v_h1:           Optional[float] = None
    v_source:       str = ""
    load_time_s:    float = 0.0
    error:          str = ""


def _survey_kb(name: str, loader_fn, validated_key: Optional[str] = None,
               modularity_Q: float = 0.5) -> KBEntry:
    t0 = time.time()
    try:
        G, deriver, err = loader_fn()
        if err:
            return KBEntry(name=name, n_nodes=0, n_edges=0, n_relations=0,
                           mean_degree=0, degree_cv=0, hub_score=0,
                           mean_rel_cov=0, max_fan_out=0, harmonic_fan_out=0,
                           regime_raw="", regime_eff="",
                           p_trb_factor=0, p_gamma=0, p_beta=0, p_r2_boost=0,
                           p_fhrb_factor=0, p_idf_weight=0, p_vote_weight=0,
                           p_branch_bonus=0, error=err,
                           load_time_s=time.time() - t0)

        class _Adapter:
            def to_networkx(_): return G
            def is_directed(_): return False

        profile = GraphProfiler.profile(_Adapter(), {})
        params  = ParameterInitializer.compute(profile, deriver, modularity_Q=modularity_Q)

        max_fo, mean_fo, harmonic_fo, n_rel = deriver.fan_out_stats()
        mean_deg = 2 * profile.n_edges / max(profile.n_nodes, 1)

        val = VALIDATED.get(validated_key, {}) if validated_key else {}

        return KBEntry(
            name             = name,
            n_nodes          = profile.n_nodes,
            n_edges          = profile.n_edges,
            n_relations      = n_rel,
            mean_degree      = round(mean_deg, 3),
            degree_cv        = round(profile.degree_cv, 3),
            hub_score        = round(profile.hub_score, 3),
            mean_rel_cov     = round(profile.mean_rel_coverage, 3),
            max_fan_out      = round(max_fo, 3),
            harmonic_fan_out = round(harmonic_fo, 3),
            regime_raw       = profile.regime,
            regime_eff       = params.effective_regime,
            p_trb_factor     = params.trb_factor,
            p_gamma          = params.gamma,
            p_beta           = params.beta,
            p_r2_boost       = params.r2_boost,
            p_fhrb_factor    = params.fhrb_factor,
            p_idf_weight     = params.idf_weight,
            p_vote_weight    = params.vote_weight,
            p_branch_bonus   = params.branch_bonus,
            v_trb_factor     = val.get("trb_factor"),
            v_gamma          = val.get("gamma"),
            v_beta           = val.get("beta"),
            v_r2_boost       = val.get("r2_boost"),
            v_fhrb_factor    = val.get("fhrb_factor"),
            v_idf_weight     = val.get("idf_weight"),
            v_vote_weight    = val.get("vote_weight"),
            v_branch_bonus   = val.get("branch_bonus"),
            v_h1             = val.get("h1"),
            v_source         = val.get("source", ""),
            load_time_s      = round(time.time() - t0, 1),
        )
    except Exception as exc:
        return KBEntry(name=name, n_nodes=0, n_edges=0, n_relations=0,
                       mean_degree=0, degree_cv=0, hub_score=0,
                       mean_rel_cov=0, max_fan_out=0, harmonic_fan_out=0,
                       regime_raw="", regime_eff="",
                       p_trb_factor=0, p_gamma=0, p_beta=0, p_r2_boost=0,
                       p_fhrb_factor=0, p_idf_weight=0, p_vote_weight=0,
                       p_branch_bonus=0, error=str(exc),
                       load_time_s=round(time.time() - t0, 1))


def _pct_err(pred, val):
    if pred is None or val is None or val == 0:
        return ""
    return f"{(pred - val) / abs(val) * 100:+.1f}%"


def _print_table(entries: list[KBEntry]):
    W_NAME = 22
    PARAMS = ["trb_factor", "gamma", "beta", "r2_boost", "fhrb_factor",
              "idf_weight", "vote_weight", "branch_bonus"]

    print("\n" + "=" * 120)
    print("  ParameterInitializer Multi-KB Survey")
    print("=" * 120)

    for e in entries:
        if e.error:
            print(f"\n  [{e.name}]  ERROR: {e.error}")
            continue
        print(f"\n  -- {e.name} {'-'*(60-len(e.name))}")
        print(f"     Nodes={e.n_nodes:,}  Edges={e.n_edges:,}  Rels={e.n_relations}  "
              f"MeanDeg={e.mean_degree:.2f}  CV={e.degree_cv:.2f}  "
              f"HubScore={e.hub_score:.3f}  MeanRelCov={e.mean_rel_cov:.3f}")
        print(f"     Regime: raw={e.regime_raw}  effective={e.regime_eff}  "
              f"MaxFanOut={e.max_fan_out:.2f}  HarmonicFO={e.harmonic_fan_out:.2f}  "
              f"Loaded in {e.load_time_s}s")
        print()
        print(f"     {'Param':<14}  {'Predicted':>10}  {'Validated':>10}  {'Error':>8}")
        print(f"     {'-'*14}  {'-'*10}  {'-'*10}  {'-'*8}")
        for p in PARAMS:
            pred = getattr(e, f"p_{p}")
            val  = getattr(e, f"v_{p}", None)
            err  = _pct_err(pred, val)
            val_s = f"{val:.4f}" if val is not None else "—"
            print(f"     {p:<14}  {pred:>10.4f}  {val_s:>10}  {err:>8}")
        if e.v_h1:
            print(f"\n     Validated H@1: {e.v_h1*100:.2f}%  ({e.v_source})")


def main():
    print("ParameterInitializer Multi-KB Survey — loading KBs...\n")

    kbs = [
        ("metaqa_3hop",     _load_metaqa,                   "metaqa_3hop", 0.293),
        ("hetionet",        _load_hetionet,                  None,          0.50),
        ("synthetic_typed", lambda: _load_synthetic(
            n_communities=20, community_size=50, p_in=0.15, p_out=0.005, n_relations=6),
                                                             None,          0.70),
        ("synthetic_hub",   lambda: _load_synthetic_hub(
            n_hubs=5, n_spokes=2000, edges_per_spoke=4, n_relations=4),
                                                             None,          0.10),
    ]

    entries = []
    for name, loader, val_key, Q in kbs:
        print(f"  Loading {name}...", end=" ", flush=True)
        entry = _survey_kb(name, loader, val_key, Q)
        entries.append(entry)
        if entry.error:
            print(f"ERROR: {entry.error}")
        else:
            print(f"ok ({entry.load_time_s}s)")

    _print_table(entries)

    # Save CSV
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f.name for f in KBEntry.__dataclass_fields__.values()])
        for e in entries:
            w.writerow([getattr(e, f) for f in KBEntry.__dataclass_fields__])
    print(f"\n  Saved -> {OUT_FILE}\n")


if __name__ == "__main__":
    main()

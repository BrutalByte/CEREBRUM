"""
CEREBRUM Auto-Tune Benchmark
============================

Profiles the current machine and determines the optimal CEREBRUM runtime
configuration for peak accuracy and throughput on this specific hardware.

What this measures
------------------
  Phase 1  — Hardware inventory: CPU cores, RAM headroom, GPU VRAM, backends
  Phase 2  — DSCF GPU crossover: at what graph size does GPU beat CPU on THIS box?
  Phase 3  — Embedding device selection: CPU vs GPU for SentenceEngine
  Phase 4  — Beam-width sweep: accuracy vs latency on a small representative graph
  Phase 5  — Coarsen-target sweep: community granularity vs traversal quality
  Phase 6  — Loop depth sweep: does max_loops > 1 pay off on this hardware?
  Phase 7  — ResourceGovernor threshold: safe RAM headroom given current load

Output
------
  * Console report with per-phase observations
  * JSON config profile written to benchmarks/autotune_profile.json
    (suitable for --params-file or programmatic CerebrumGraph construction)

Usage
-----
  python -m benchmarks.autotune_benchmark

  # Skip slow phases (quick mode — phases 1-3 only, DSCF limited to 3 sizes):
  python -m benchmarks.autotune_benchmark --quick

  # Control toy graph size for sweep phases:
  python -m benchmarks.autotune_benchmark --sweep-nodes 3000

  # Save profile to a custom path:
  python -m benchmarks.autotune_benchmark --output myprofile.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import psutil

from core.hardware import (
    HAS_CUDA,
    HAS_MPS,
    HAS_HPU,
    HAS_XLA,
    device_info,
    get_gpu_vram_mb,
)
from core.community_engine import best_of_n_dscf
from core.dscf_gpu import GPUDSCFEngine, GPUDSCFConfig
from core.structural_encoder import (
    coarsen_communities,
    build_community_distance_matrix,
    adjacent_community_pairs,
    build_community_graph,
)
from adapters.networkx_adapter import NetworkXAdapter
from core.attention_engine import CSAEngine
from core.embedding_engine import RandomEngine
from core.resource_governor import ResourceGovernor
from reasoning.traversal import BeamTraversal
from reasoning.answer_extractor import extract

try:
    from core.embedding_engine import SentenceEngine
    _HAS_SENTENCE = True
except Exception:
    _HAS_SENTENCE = False

# ---------------------------------------------------------------------------
# Utility: synthetic graph + QA generation
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int, seed: int = 42) -> nx.Graph:
    """Barabasi-Albert scale-free graph — realistic community structure."""
    m = max(2, min(5, n_nodes // 200))
    G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    mapping = {i: f"e{i}" for i in G.nodes()}
    return nx.relabel_nodes(G, mapping)


def _make_qa(G: nx.Graph, n_pairs: int, min_hop: int, max_hop: int,
             seed: int = 42) -> List[Tuple[str, List[str]]]:
    """Sample seed→answer pairs by BFS at distance in [min_hop, max_hop]."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    pairs: List[Tuple[str, List[str]]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        src = rng.choice(nodes)
        try:
            dist = nx.single_source_shortest_path_length(G, src, cutoff=max_hop)
        except Exception:
            continue
        candidates = [n for n, d in dist.items() if min_hop <= d <= max_hop]
        if not candidates:
            continue
        tgt = rng.choice(candidates)
        pairs.append((src, [tgt]))
    return pairs


def _build_adapter(G: nx.Graph, coarsen_target: int = 200,
                   dscf_gpu: bool = False) -> Tuple[NetworkXAdapter, CSAEngine]:
    """Build adapter + CSA engine for traversal experiments."""
    if dscf_gpu and HAS_CUDA:
        engine = GPUDSCFEngine(GPUDSCFConfig(device="auto", max_iter=50))
        parts = engine.detect(G)
    else:
        parts = best_of_n_dscf(G, n_trials=1, seed=42)

    cmap_raw = {node: cid for cid, members in enumerate(parts) for node in members}
    cmap = coarsen_communities(G, cmap_raw, target_max=coarsen_target)

    emb_engine = RandomEngine(dim=64)
    labels = {n: n for n in G.nodes()}
    embeddings = emb_engine.encode_entities(labels)

    adapter = NetworkXAdapter(G)
    adapter.community_map = cmap
    adapter.embeddings = embeddings

    distances = build_community_distance_matrix(G, cmap)
    adj = adjacent_community_pairs(G, cmap)
    cg = build_community_graph(G, cmap)

    csa = CSAEngine(adapter=adapter)
    csa.set_community_graph(distances, adj, community_graph=cg)

    return adapter, csa


def _eval_traversal(
    adapter: NetworkXAdapter,
    csa: CSAEngine,
    qa_pairs: List[Tuple[str, List[str]]],
    beam_width: int,
    max_hop: int,
    n_loops: int = 1,
) -> Tuple[float, float]:
    """Run traversal over qa_pairs. Returns (H@10, ms_per_query)."""
    gov = ResourceGovernor(memory_threshold_pct=100.0)
    traversal = BeamTraversal(
        adapter=adapter,
        csa_engine=csa,
        beam_width=beam_width,
        max_hop=max_hop,
        governor=gov,
    )

    if n_loops > 1:
        from reasoning.looped_traversal import LoopedBeamTraversal
        runner: Any = LoopedBeamTraversal(traversal, max_loops=n_loops)
    else:
        runner = traversal

    hits = 0
    t0 = time.time()
    for seed, correct in qa_pairs:
        if n_loops > 1:
            paths, _ = runner.traverse([seed])
        else:
            paths = traversal.traverse([seed])
        answers = extract(paths, top_k=10, min_hop=1)
        pred = {a.entity_id for a in answers}
        if any(c in pred for c in correct):
            hits += 1
    elapsed = time.time() - t0
    h10 = hits / len(qa_pairs) if qa_pairs else 0.0
    ms_q = elapsed * 1000 / len(qa_pairs) if qa_pairs else 0.0
    return h10, ms_q


# ---------------------------------------------------------------------------
# Phase 1 — Hardware inventory
# ---------------------------------------------------------------------------

def phase1_hardware() -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print("  PHASE 1 — Hardware Inventory")
    print("=" * 68)

    cpu_count = os.cpu_count() or 1
    mem = psutil.virtual_memory()
    ram_total_gb = mem.total / 1024 ** 3
    ram_free_gb  = mem.available / 1024 ** 3
    ram_pct      = mem.percent

    print(f"  CPU cores         : {cpu_count}")
    print(f"  System RAM        : {ram_free_gb:.1f} GB free / {ram_total_gb:.1f} GB total ({ram_pct:.1f}% used)")

    info = device_info()
    gpu_available = info["gpu_available"]
    vram_free_mb = vram_total_mb = 0

    if HAS_CUDA:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        vram_free_mb, vram_total_mb = get_gpu_vram_mb(0)
        print(f"  GPU               : {gpu_name}")
        print(f"  VRAM              : {vram_free_mb:,} MB free / {vram_total_mb:,} MB total")
    elif HAS_MPS:
        print("  GPU               : Apple MPS (unified memory)")
    else:
        print("  GPU               : not available")

    print(f"  Active backend    : {info['backend']}")
    print(f"  SentenceEngine    : {'available' if _HAS_SENTENCE else 'not installed'}")

    return {
        "cpu_count": cpu_count,
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_free_gb": round(ram_free_gb, 2),
        "ram_pct": round(ram_pct, 1),
        "gpu_available": gpu_available,
        "vram_free_mb": vram_free_mb,
        "vram_total_mb": vram_total_mb,
        "has_sentence_engine": _HAS_SENTENCE,
        "backend": info["backend"],
    }


# ---------------------------------------------------------------------------
# Phase 2 — DSCF GPU crossover
# ---------------------------------------------------------------------------

DSCF_PROBE_SIZES       = [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
DSCF_PROBE_SIZES_QUICK = [1_000, 5_000, 25_000]


def phase2_dscf_crossover(quick: bool = False) -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print("  PHASE 2 — DSCF GPU Crossover Point")
    print("  (find the graph size where GPU DSCF beats CPU DSCF on THIS machine)")
    print("=" * 68)

    if not HAS_CUDA:
        print("  [Skipped — no CUDA GPU available; use CPU DSCF always]")
        return {"crossover_nodes": None, "recommendation": "cpu", "rows": []}

    probe_sizes = DSCF_PROBE_SIZES_QUICK if quick else DSCF_PROBE_SIZES
    if quick:
        print("  (quick mode: 3 probe sizes only)")

    print(f"  {'Nodes':>8}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>9}  Winner")
    print("  " + "-" * 56)

    # Create one engine instance before the loop to avoid CUDA init overhead
    # contaminating per-size measurements.
    gpu_engine = GPUDSCFEngine(GPUDSCFConfig(device="auto", max_iter=50))

    rows = []
    crossover: Optional[int] = None

    for n in probe_sizes:
        G = _make_graph(n)

        t0 = time.time()
        best_of_n_dscf(G, n_trials=1, seed=42)
        cpu_s = time.time() - t0

        # detect_with_stats() never raises — it catches OOM internally and
        # falls back to CPU, recording stats.device_used = "cpu".
        # Check the returned stats to know whether GPU was actually used.
        t0 = time.time()
        _parts, stats = gpu_engine.detect_with_stats(G)
        gpu_s = time.time() - t0

        actually_gpu = not stats.device_used.startswith("cpu")

        if not actually_gpu:
            # Engine fell back to CPU — do not record as a valid GPU time.
            print(f"  {n:>8,}  {cpu_s:>10.3f}  {'FALLBACK':>10}  {'n/a':>9}  CPU")
            rows.append({"n": n, "cpu_s": round(cpu_s, 4), "gpu_s": None,
                         "speedup": None, "winner": "CPU",
                         "note": f"GPU fallback to CPU ({stats.device_used})"})
        else:
            speedup = cpu_s / gpu_s if gpu_s > 0 else float("inf")
            winner = "GPU" if gpu_s < cpu_s else "CPU"
            if winner == "GPU" and crossover is None:
                crossover = n
            print(f"  {n:>8,}  {cpu_s:>10.3f}  {gpu_s:>10.3f}  {speedup:>8.2f}x  {winner}  [{stats.device_used}]")
            rows.append({"n": n, "cpu_s": round(cpu_s, 4), "gpu_s": round(gpu_s, 4),
                         "speedup": round(speedup, 3), "winner": winner})

    rec = "gpu" if crossover is not None else "cpu"
    if crossover:
        print(f"\n  GPU crossover: {crossover:,} nodes — use GPU DSCF for graphs larger than this.")
    else:
        print("\n  CPU DSCF faster at all tested sizes — recommend CPU DSCF.")

    return {"crossover_nodes": crossover, "recommendation": rec, "rows": rows}


# ---------------------------------------------------------------------------
# Phase 3 — Embedding device selection
# ---------------------------------------------------------------------------

EMBED_PROBE_SIZES = [500, 2_000, 10_000]
_WARMUP_LABELS = {f"w{i}": f"warmup entity {i}" for i in range(32)}


def phase3_embedding_device() -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print("  PHASE 3 — Embedding Device Selection")
    print("=" * 68)

    if not _HAS_SENTENCE:
        print("  [Skipped — SentenceEngine not installed; RandomEngine will be used]")
        # "random" is not a valid device string — return a safe flag instead
        return {"recommendation": "cpu", "use_random_embeddings": True, "speedup": None}

    if not (HAS_CUDA or HAS_MPS):
        print("  [No GPU — SentenceEngine will use CPU]")
        return {"recommendation": "cpu", "use_random_embeddings": False, "speedup": None}

    # Resolve GPU device string (SentenceEngine needs explicit string, not "auto")
    if HAS_CUDA:
        import torch as _torch
        _gpu_dev = f"cuda:{_torch.cuda.current_device()}"
    elif HAS_MPS:
        _gpu_dev = "mps"
    else:
        _gpu_dev = "cpu"

    # Create engines ONCE before the loop — model load time must not pollute
    # per-batch encoding measurements.
    try:
        cpu_engine = SentenceEngine(device="cpu")
        gpu_engine = SentenceEngine(device=_gpu_dev)
    except Exception as exc:
        print(f"  [Skipped — could not load SentenceEngine: {exc}]")
        return {"recommendation": "cpu", "use_random_embeddings": False, "speedup": None}

    # Warm up both engines to trigger JIT compilation before timing.
    cpu_engine.encode_entities(_WARMUP_LABELS)
    try:
        gpu_engine.encode_entities(_WARMUP_LABELS)
    except Exception:
        print("  [GPU warm-up failed — GPU engine unavailable; using CPU]")
        return {"recommendation": "cpu", "use_random_embeddings": False, "speedup": None}

    print(f"  {'Entities':>10}  {'CPU (s)':>10}  {'GPU (s)':>10}  {'Speedup':>9}")
    print("  " + "-" * 48)

    speedups = []
    for n in EMBED_PROBE_SIZES:
        labels = {f"e{i}": f"entity label {i}" for i in range(n)}

        t0 = time.time()
        cpu_engine.encode_entities(labels)
        cpu_s = time.time() - t0

        try:
            t0 = time.time()
            gpu_engine.encode_entities(labels)
            gpu_s = time.time() - t0
        except Exception as exc:
            print(f"  {n:>10,}  {cpu_s:>10.3f}  {'OOM/err':>10}  {'n/a':>9}  ({exc!s:.60})")
            continue

        speedup = cpu_s / gpu_s if gpu_s > 0 else float("inf")
        speedups.append(speedup)
        print(f"  {n:>10,}  {cpu_s:>10.3f}  {gpu_s:>10.3f}  {speedup:>8.2f}x")

    mean_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    rec = "gpu" if mean_speedup > 1.1 else "cpu"
    print(f"\n  Mean speedup: {mean_speedup:.2f}x — recommend embedding device: {rec.upper()}")

    return {"recommendation": rec, "use_random_embeddings": False,
            "speedup": round(mean_speedup, 3)}


# ---------------------------------------------------------------------------
# Phase 4 — Beam-width sweep
# ---------------------------------------------------------------------------

BEAM_WIDTHS = [2, 4, 8, 16, 32]
_DEGENERATE_H10_THRESHOLD = 0.02  # below this, synthetic graph is too sparse to calibrate


def phase4_beam_width(n_nodes: int) -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print(f"  PHASE 4 — Beam-Width Sweep ({n_nodes:,}-node graph)")
    print("  (find the knee: first beam_width that captures >=80% of peak H@10)")
    print("=" * 68)

    G = _make_graph(n_nodes)
    qa = _make_qa(G, n_pairs=200, min_hop=1, max_hop=3)
    if not qa:
        print("  [Skipped — could not generate QA pairs]")
        return {"optimal_beam_width": 8, "rows": [], "degenerate": True}

    adapter, csa = _build_adapter(G, coarsen_target=150)

    print(f"  {len(qa)} questions, max_hop=3")
    print(f"  {'beam_width':>12}  {'H@10':>8}  {'ms/Q':>8}  {'H@10 gain':>10}")
    print("  " + "-" * 52)

    rows = []
    peak_h10 = 0.0
    prev_h10 = 0.0

    for bw in BEAM_WIDTHS:
        h10, ms_q = _eval_traversal(adapter, csa, qa, beam_width=bw, max_hop=3)
        peak_h10 = max(peak_h10, h10)
        h10_gain = h10 - prev_h10
        prev_h10 = h10
        print(f"  {bw:>12}  {h10:>8.4f}  {ms_q:>8.2f}  {h10_gain:>+10.4f}")
        rows.append({"beam_width": bw, "h10": round(h10, 4),
                     "ms_q": round(ms_q, 3), "h10_gain": round(h10_gain, 4)})

    # Check for degenerate result (synthetic graph too sparse to differentiate)
    if peak_h10 < _DEGENERATE_H10_THRESHOLD:
        best_bw = 8
        print(f"\n  WARNING: Peak H@10={peak_h10:.4f} is below {_DEGENERATE_H10_THRESHOLD} —")
        print("  synthetic graph too sparse to calibrate beam width.")
        print(f"  Defaulting to beam_width=8 (production safe default).")
        print("  Re-run with --sweep-nodes 5000+ or use real graph data for better calibration.")
        return {"optimal_beam_width": best_bw, "rows": rows, "degenerate": True}

    # Find the "knee" — first beam_width that captures >=80% of peak H@10
    threshold_80 = 0.80 * peak_h10
    best_bw = BEAM_WIDTHS[-1]  # fallback to widest if no knee found
    for row in rows:
        if row["h10"] >= threshold_80:
            best_bw = row["beam_width"]
            break

    print(f"\n  Peak H@10: {peak_h10:.4f}")
    print(f"  Optimal beam_width: {best_bw} (first to reach >=80% of peak H@10)")
    print("  Note: sweep uses a synthetic graph. On denser KGs (MetaQA, WebQSP),")
    print("        higher beam widths further improve multi-hop recall.")
    return {"optimal_beam_width": best_bw, "rows": rows, "degenerate": False}


# ---------------------------------------------------------------------------
# Phase 5 — Coarsen-target sweep
# ---------------------------------------------------------------------------

COARSEN_TARGETS = [50, 100, 200, 400, 800]


def phase5_coarsen_target(n_nodes: int, beam_width: int) -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print(f"  PHASE 5 — Coarsen-Target Sweep ({n_nodes:,}-node graph)")
    print("  (find community granularity that maximises traversal quality)")
    print("=" * 68)

    G = _make_graph(n_nodes)
    qa = _make_qa(G, n_pairs=200, min_hop=1, max_hop=3)
    if not qa:
        print("  [Skipped — could not generate QA pairs]")
        return {"optimal_coarsen_target": 200, "rows": []}

    # Tiebreaker target: ~150 nodes per community (scales with graph size)
    tiebreak_target = max(50, n_nodes // 150)

    print(f"  beam_width={beam_width}, {len(qa)} questions, max_hop=3")
    print(f"  (tiebreaker: target closest to {tiebreak_target} = {n_nodes:,} / 150)")
    print(f"  {'coarsen_target':>16}  {'communities':>12}  {'H@10':>8}  {'ms/Q':>8}")
    print("  " + "-" * 56)

    rows = []
    best_target = tiebreak_target
    best_h10 = -1.0

    for ct in COARSEN_TARGETS:
        adapter, csa = _build_adapter(G, coarsen_target=ct)
        n_comms = len(set(adapter.community_map.values()))
        h10, ms_q = _eval_traversal(adapter, csa, qa, beam_width=beam_width, max_hop=3)
        marker = ""
        if h10 > best_h10:
            best_h10 = h10
            best_target = ct
            marker = " <-- best"
        elif h10 == best_h10:
            # Tie: prefer the target closest to tiebreak_target
            if abs(ct - tiebreak_target) < abs(best_target - tiebreak_target):
                best_target = ct
                marker = " <-- tie, closer to ideal ratio"
        print(f"  {ct:>16}  {n_comms:>12,}  {h10:>8.4f}  {ms_q:>8.2f}{marker}")
        rows.append({"coarsen_target": ct, "n_communities": n_comms,
                     "h10": round(h10, 4), "ms_q": round(ms_q, 3)})

    print(f"\n  Optimal coarsen_target: {best_target}")
    return {"optimal_coarsen_target": best_target, "rows": rows}


# ---------------------------------------------------------------------------
# Phase 6 — Loop depth sweep
# ---------------------------------------------------------------------------

LOOP_DEPTHS = [1, 2, 3, 4]
_LOOP_H10_GAIN_THRESHOLD = 0.005  # minimum meaningful gain to justify latency cost


def phase6_loop_depth(n_nodes: int, optimal_bw: int, optimal_ct: int) -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print(f"  PHASE 6 — Loop Depth Sweep ({n_nodes:,}-node graph)")
    print("  (does max_loops > 1 pay off on this hardware?)")
    print("=" * 68)

    G = _make_graph(n_nodes)
    qa = _make_qa(G, n_pairs=100, min_hop=2, max_hop=3)
    if not qa:
        print("  [Skipped — could not generate QA pairs]")
        return {"optimal_max_loops": 1, "rows": []}

    adapter, csa = _build_adapter(G, coarsen_target=optimal_ct)

    print(f"  {len(qa)} questions, beam_width={optimal_bw}, max_hop=3")
    print(f"  NOTE: Loop seed expansion requires finding answer entities (H@10 > 0).")
    print(f"        On sparse synthetic graphs, loop benefit may not be observable here.")
    print(f"        For real KGs with attached PredictiveCodingEngine, loop benefit is greater.")
    print(f"  {'max_loops':>10}  {'H@10':>8}  {'ms/Q':>8}  {'H@10 gain':>10}  {'Latency cost':>14}")
    print("  " + "-" * 62)

    rows = []
    best_loops = 1
    baseline_ms = None
    prev_h10 = None

    for loops in LOOP_DEPTHS:
        h10, ms_q = _eval_traversal(adapter, csa, qa, beam_width=optimal_bw,
                                     max_hop=3, n_loops=loops)
        if baseline_ms is None:
            baseline_ms = ms_q if ms_q > 0 else 1.0
        latency_cost = ms_q / baseline_ms if baseline_ms else 1.0
        h10_gain = (h10 - prev_h10) if prev_h10 is not None else 0.0
        prev_h10 = h10

        # Only credit loop N if it produced a meaningful H@10 gain. Without
        # gain, the extra latency cost makes the loop worse than single-pass.
        if h10_gain >= _LOOP_H10_GAIN_THRESHOLD:
            best_loops = loops

        marker = " <-- optimal" if loops == best_loops else ""
        print(f"  {loops:>10}  {h10:>8.4f}  {ms_q:>8.2f}  "
              f"{h10_gain:>+10.4f}  {latency_cost:>13.2f}x{marker}")
        rows.append({"max_loops": loops, "h10": round(h10, 4),
                     "ms_q": round(ms_q, 3), "h10_gain": round(h10_gain, 4),
                     "latency_cost": round(latency_cost, 3)})

    print(f"\n  Optimal max_loops: {best_loops}")
    if best_loops == 1:
        print("  (No loop produced H@10 gain > 0.005 — single-pass is optimal here)")
    else:
        print(f"  (max_loops={best_loops} produced meaningful accuracy gain)")

    return {"optimal_max_loops": best_loops, "rows": rows}


# ---------------------------------------------------------------------------
# Phase 7 — ResourceGovernor threshold
# ---------------------------------------------------------------------------

def phase7_resource_governor() -> Dict[str, Any]:
    print("\n" + "=" * 68)
    print("  PHASE 7 — ResourceGovernor Threshold")
    print("  (safe memory_threshold_pct for this machine's current load)")
    print("=" * 68)

    mem = psutil.virtual_memory()
    current_pct = mem.percent
    free_gb = mem.available / 1024 ** 3

    # Target: current_pct + 5% headroom, bounded by [95%, 97%].
    # Floor is 95% (matches ResourceGovernor's own production default) so
    # autotune never recommends a MORE conservative threshold than the default.
    headroom = 5.0
    candidate = current_pct + headroom

    if candidate >= 98.0:
        candidate = 97.0
        note = "RAM is heavily loaded — threshold capped at 97%"
    elif candidate < 95.0:
        candidate = 95.0  # floor at 95% — matches ResourceGovernor default
        note = f"Headroom ({current_pct:.1f}% + {headroom}%) < 95% — floored at production default"
    else:
        note = f"Current RAM usage {current_pct:.1f}% + {headroom}% headroom"

    threshold = round(candidate, 1)

    print(f"  Current RAM usage   : {current_pct:.1f}%  ({free_gb:.1f} GB free)")
    print(f"  Recommended threshold: {threshold:.1f}%")
    print(f"  Rationale           : {note}")

    return {
        "current_ram_pct": round(current_pct, 1),
        "recommended_threshold_pct": threshold,
        "rationale": note,
    }


# ---------------------------------------------------------------------------
# Profile assembly and output
# ---------------------------------------------------------------------------

_SECTION = "=" * 68


def assemble_profile(
    hw: Dict,
    dscf: Dict,
    embed: Dict,
    beam: Dict,
    coarsen: Dict,
    loops: Dict,
    gov: Dict,
    graph_size_hint: int,
    sweep_n: int,
) -> Dict[str, Any]:
    """Combine phase results into a single recommended configuration."""

    crossover = dscf.get("crossover_nodes")
    use_gpu_dscf = (
        hw["gpu_available"]
        and crossover is not None
        and graph_size_hint >= crossover
    )

    profile = {
        "_version": "1.0",
        "_benchmark_date": __import__("datetime").date.today().isoformat(),
        "_graph_size_hint": graph_size_hint,
        "_calibrated_on_n_nodes": sweep_n,
        "hardware": {
            "cpu_count": hw["cpu_count"],
            "ram_total_gb": hw["ram_total_gb"],
            "vram_free_mb": hw["vram_free_mb"],
            "backend": hw["backend"],
        },
        "dscf": {
            "use_gpu": use_gpu_dscf,
            "gpu_crossover_nodes": crossover,
            "recommendation": dscf["recommendation"],
        },
        "embedding": {
            "device": embed["recommendation"],
            "use_random_embeddings": embed.get("use_random_embeddings", False),
            "gpu_speedup": embed.get("speedup"),
        },
        "traversal": {
            "beam_width": beam["optimal_beam_width"],
            "coarsen_target": coarsen["optimal_coarsen_target"],
            "max_loops": loops["optimal_max_loops"],
            "max_hop": 3,
            "beam_width_note": (
                "defaulted — synthetic graph too sparse to calibrate"
                if beam.get("degenerate") else
                "calibrated on synthetic graph; tune against real KG data"
            ),
            "max_loops_note": (
                "loop benefit requires PredictiveCodingEngine attached "
                "and a semantically structured graph"
            ),
        },
        "resource_governor": {
            "memory_threshold_pct": gov["recommended_threshold_pct"],
        },
    }
    return profile


def print_profile_summary(profile: Dict) -> None:
    t = profile["traversal"]
    e = profile["embedding"]
    d = profile["dscf"]
    g = profile["resource_governor"]

    print("\n" + _SECTION)
    print("  RECOMMENDED CONFIGURATION PROFILE")
    print(_SECTION)
    if d["gpu_crossover_nodes"]:
        print(f"  DSCF device         : {'GPU' if d['use_gpu'] else 'CPU'}  "
              f"(GPU crossover at {d['gpu_crossover_nodes']:,} nodes)")
    else:
        print("  DSCF device         : CPU")
    embed_line = f"  Embedding device    : {e['device'].upper()}"
    if e.get("use_random_embeddings"):
        embed_line += "  (RandomEngine — sentence-transformers not installed)"
    elif e.get("gpu_speedup"):
        embed_line += f"  ({e['gpu_speedup']:.2f}x speedup)"
    print(embed_line)
    print(f"  beam_width          : {t['beam_width']}  ({t['beam_width_note']})")
    print(f"  coarsen_target      : {t['coarsen_target']}")
    print(f"  max_loops           : {t['max_loops']}"
          + ("  (single-pass optimal)" if t["max_loops"] == 1 else
             f"  (looping improves accuracy on this hardware)"))
    print(f"  RAM threshold       : {g['memory_threshold_pct']:.1f}%")
    print()
    print("  To use this profile programmatically:")
    print("    import json")
    print("    profile = json.load(open('benchmarks/autotune_profile.json'))")
    print("    t = profile['traversal']")
    print("    graph.build(coarsen_target=t['coarsen_target'])")
    print("    graph.query(seeds, beam_width=t['beam_width'], max_loops=t['max_loops'])")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEREBRUM Auto-Tune: find optimal settings for this machine"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only Phases 1-3; limit DSCF probe to 3 sizes (faster, ~2 min)"
    )
    parser.add_argument(
        "--sweep-nodes", type=int, default=2_000,
        metavar="N",
        help="Synthetic graph size for sweep phases 4-6 (default: 2000)"
    )
    parser.add_argument(
        "--graph-size-hint", type=int, default=None,
        metavar="N",
        help="Expected real graph size (for DSCF device recommendation). "
             "Defaults to --sweep-nodes value."
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/autotune_profile.json",
        help="Path to write the JSON profile (default: benchmarks/autotune_profile.json)"
    )
    args = parser.parse_args()

    sweep_n = args.sweep_nodes
    size_hint = args.graph_size_hint or sweep_n

    print()
    print(_SECTION)
    print("  CEREBRUM Auto-Tune Benchmark")
    print("  Finding optimal settings for peak performance on this machine")
    print(_SECTION)

    # Phase 1 — always run
    hw = phase1_hardware()

    # Phase 2 — always run; limited probe sizes in quick mode
    dscf = phase2_dscf_crossover(quick=args.quick)

    # Phase 3 — always run
    embed = phase3_embedding_device()

    if args.quick:
        beam    = {"optimal_beam_width": 8,   "rows": [], "degenerate": True}
        coarsen = {"optimal_coarsen_target": 200, "rows": []}
        loops   = {"optimal_max_loops": 1,    "rows": []}
        gov     = phase7_resource_governor()
        print("\n  [Phases 4-6 skipped — quick mode. Using safe defaults: "
              "beam_width=8, coarsen_target=200, max_loops=1]")
    else:
        beam    = phase4_beam_width(sweep_n)
        coarsen = phase5_coarsen_target(sweep_n, beam["optimal_beam_width"])
        loops   = phase6_loop_depth(sweep_n, beam["optimal_beam_width"],
                                    coarsen["optimal_coarsen_target"])
        gov     = phase7_resource_governor()

    profile = assemble_profile(hw, dscf, embed, beam, coarsen, loops, gov,
                                size_hint, sweep_n)
    print_profile_summary(profile)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"  Profile written to: {out_path}")
    print()


if __name__ == "__main__":
    main()

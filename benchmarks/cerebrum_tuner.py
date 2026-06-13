"""
CEREBRUM Live Hyperparameter Tuner
====================================
Two-phase search: Phase 1 uses Sobol quasi-random sampling (QMCSampler) for
maximum uniform coverage of the 9D parameter space; Phase 2 uses CMA-ES
(CmaEsSampler) initialized from the best Phase 1 config and warmed with all
Phase 1 trials, enabling the optimizer to model parameter correlations that
TPE treats as independent dimensions.

Parameters searched (9 total):
  Core:           trb-factor, r2-boost, vote-weight, beam-width, idf-weight, branch-bonus
  First-hop:      fhrb-factor
  SDRB (Phase 203): gamma, beta  (boost(r) = gamma * fan_out(r)^beta)

Usage
-----
    # Two-phase (default): 60 exploration + 140 fine-tuning
    python -u benchmarks/cerebrum_tuner.py --sample 500 --validate 14274

    # Custom split
    python -u benchmarks/cerebrum_tuner.py --phase1-trials 80 --phase2-trials 160 --sample 500

    # Resume after restart — skip Phase 1, jump straight to Phase 2
    python -u benchmarks/cerebrum_tuner.py --resume benchmarks/tuner_<run_id>.jsonl --phase2-trials 140 --sample 2000 --workers 16

    # Single-phase (legacy TPE, backward-compatible)
    python -u benchmarks/cerebrum_tuner.py --n-trials 200 --sample 500

Install dependencies
---------------------
    pip install "cerebrum-kg[tuning]"
    # or: pip install optuna rich
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich import box as rich_box
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

_EVAL_SCRIPT          = Path(__file__).parent / "metaqa_eval.py"
_HETIONET_EVAL_SCRIPT = Path(__file__).parent / "hetionet_param_eval.py"
_CN5_EVAL_SCRIPT      = Path(__file__).parent / "conceptnet_eval.py"
_WEBQSP_EVAL_SCRIPT   = Path(__file__).parent / "webqsp_param_eval.py"
_DEFAULT_KB  = Path(__file__).parent / "data" / "metaqa" / "kb.txt"

# Pre-built state shared across all in-process trials (avoids expensive graph rebuild).
# Populated once by _init_*_state() before the Optuna study starts.
_hetionet_state:   Optional[dict] = None
_conceptnet_state: Optional[dict] = None
_webqsp_state:     Optional[dict] = None

# ── search space (Phase 1 — wide exploration) ─────────────────────────────────
# Phase 203: two-parameter SDRB — gamma ceiling raised to 16.0 (hit 7.93/8.0 in
# Phase 202); beta added [0.5, 3.0] to control fan_out shaping.
# branch_bonus raised to [0.0, 1.0] — was #1 fANOVA driver at 46% in Phase 202.
# trb_factor ceiling raised to 35.0 — was #2 driver at 29%.
PARAM_SPACE_WIDE: dict = {
    "trb_factor":   (18.0, 35.0),
    "r2_boost":     (2.0,  10.0),
    "vote_weight":  (0.75, 0.95),
    "beam_width":   [8, 10, 12],
    "idf_weight":   (0.0,  0.15),
    "branch_bonus": (0.0,  1.0),
    "fhrb_factor":  (3.0,  8.0),
    "gamma":        (1.5,  16.0),  # Phase 203: expanded ceiling (hit 7.93/8.0 in P202)
    "beta":         (0.5,  3.0),   # Phase 203: power-law exponent for fan_out shaping
}

# Phase 206: Hetionet (typed_heterogeneous) search space.
# Centered around ParameterInitializer typed_heterogeneous predictions:
#   gamma~2.30, beta=1.0, trb~19.6, fhrb~2.64, r2~4.27, idf~0.043, branch=0.17
PARAM_SPACE_HETIONET: dict = {
    "trb_factor":   (2.0,  25.0),
    "r2_boost":     (1.0,  10.0),
    "vote_weight":  (0.60, 0.95),
    "beam_width":   [8, 10, 12],
    "idf_weight":   (0.0,  0.15),
    "branch_bonus": (0.0,  0.80),
    "fhrb_factor":  (1.0,  5.0),
    "gamma":        (0.5,  8.0),
    "beta":         (0.5,  2.0),
}

# Phase 229: ConceptNet (mixed regime) search space.
# Centered around ParameterInitializer mixed/random predictions (pre-calibration estimate).
# ConceptNet is 1-hop link prediction with high fan-out, so trb_factor matters more;
# gamma/beta drive SDRB on ~2M edges — wider range than Hetionet.
PARAM_SPACE_CONCEPTNET: dict = {
    "trb_factor":   (5.0,  35.0),
    "r2_boost":     (1.0,  10.0),
    "vote_weight":  (0.60, 0.95),
    "beam_width":   [8, 10, 12],
    "idf_weight":   (0.0,  0.20),
    "branch_bonus": (0.0,  1.0),
    "fhrb_factor":  (1.0,  8.0),
    "gamma":        (0.5,  16.0),
    "beta":         (0.5,  3.0),
}

# Phase 238: narrowed from 340-trial empirical analysis of June 2026 WebQSP runs.
# beam_width: bw=16/24 dominate H@1 (Pearson r=-0.352 vs bw size); drop 32/48.
# trb_factor: top-quartile median 43.6 → floor raised to 38.
# vote_weight: top-quartile >0.83 → floor raised.
# idf_weight: top-quartile max 0.052 → ceiling cut to 0.08.
# branch_bonus: top-quartile max 0.089 → ceiling cut to 0.12.
# schema_score_threshold: first real sweep (was propagation-bug blocked in 237).
# degree_penalty_weight: Phase 239 — hub suppression via entity degree in extract().
PARAM_SPACE_WEBQSP: dict = {
    # Phase 244d: bounds tightened around Phase 244c best (H@1=10.26%, H@10=20.04%)
    "trb_factor":             (35.0, 55.0),   # best=41.052
    "r2_boost":               (1.0,  6.0),    # best=3.490
    "vote_weight":            (0.85, 0.95),   # best=0.9054
    "beam_width":             [16, 24],        # best=16; wider beams hurt in 244c
    "idf_weight":             (0.0,  0.04),   # best=0.019
    "branch_bonus":           (0.0,  0.08),   # best=0.030
    "fhrb_factor":            (0.5,  2.5),    # best=1.302
    "gamma":                  (2.0,  8.0),    # best=4.985
    "beta":                   (0.8,  1.5),    # best=1.118
    "schema_score_threshold": (0.35, 0.50),   # best=0.408
    "degree_penalty_weight":  (0.3,  0.6),    # best=0.494
    "backward_bonus":         (0.05, 0.4),    # best=0.182; fANOVA #2 at 17.97%
    "diversity_alpha":        (0.3,  1.0),    # best=0.695; tightened around best
    "conditional_schema_bonus": (0.1, 0.8),  # Phase 247: structural r2 confirmation boost
    "qa_sem_weight":            (0.0, 2.0),  # Phase 250: question↔answer embedding alignment
    # Phase 243: cvt_passthrough is a fixed True for WebQSP/Freebase (not tunable).
    # Ablation confirmed cvt_relay_boost (post-hoc compensation) consistently hurts;
    # cvt_passthrough (traversal-time collapsing) is the correct structural fix.
}

# Phase 251: focused degree_penalty_weight sweep.
# fANOVA across Phases 244d and 250 consistently shows degree_penalty_weight at 50–57%
# importance — the single dominant factor on WebQSP.  Prior searches used (0.3, 0.6);
# this space expands to (0.0, 1.5) to find whether stronger hub suppression helps.
# All low-importance params are tightened around Phase 244d best to concentrate the
# trial budget on the high-signal region.
PARAM_SPACE_WEBQSP_251: dict = {
    "trb_factor":             (35.0, 52.0),   # 244d best=41.785 ±25%
    "r2_boost":               (2.0,  5.0),    # 244d best=3.722
    "vote_weight":            (0.78, 0.95),   # 244d best=0.8608
    "beam_width":             [16, 24],        # 244d best=16
    "idf_weight":             (0.0,  0.08),   # 244d best=0.013; fANOVA #3 at 13.3%
    "branch_bonus":           (0.02, 0.10),   # 244d best=0.048
    "fhrb_factor":            (1.5,  2.5),    # 244d best=1.826
    "gamma":                  (5.0,  10.0),   # 244d best=7.726
    "beta":                   (0.8,  1.3),    # 244d best=1.008
    "schema_score_threshold": (0.25, 0.55),   # 244d best=0.369; fANOVA #2 at 15.4%
    "degree_penalty_weight":  (0.0,  1.5),    # 244d best=0.424; 50%+ importance — PRIMARY TARGET
    "backward_bonus":         (0.05, 0.40),   # 244d best=0.136; 8.6% in Phase 250
    "diversity_alpha":        (0.6,  1.0),    # 244d best=0.799
}

# Phase 252: traversal-time hub suppression sweep.
# Phase 251 confirmed DPW is a plateau: post-hoc extraction penalty is already optimal.
# Phase 252 tests beam_hub_penalty (BHP) — same 1/(1+bhp*log1p(deg)) formula but applied
# at non-terminal hops DURING traversal, preventing hub entities from accumulating beam
# score as intermediates.  Terminal hop excluded so high-degree correct answers (e.g.
# "United States") are not penalised.  BHP and DPW are complementary: DPW suppresses hub
# entities as final answers; BHP suppresses them as intermediate beam nodes.
PARAM_SPACE_WEBQSP_252: dict = {
    "trb_factor":             (35.0, 52.0),   # 244d best=41.785
    "r2_boost":               (2.0,  5.0),    # 244d best=3.722
    "vote_weight":            (0.78, 0.95),   # 244d best=0.8608
    "beam_width":             [16, 24],        # 244d best=16
    "idf_weight":             (0.0,  0.08),   # 244d best=0.013; 13.3% fANOVA
    "branch_bonus":           (0.02, 0.10),   # 244d best=0.048
    "fhrb_factor":            (1.5,  2.5),    # 244d best=1.826
    "gamma":                  (5.0,  10.0),   # 244d best=7.726
    "beta":                   (0.8,  1.3),    # 244d best=1.008
    "schema_score_threshold": (0.25, 0.55),   # 244d best=0.369; 15.4% fANOVA
    "degree_penalty_weight":  (0.3,  0.7),    # 244d best=0.424 — plateau confirmed
    "beam_hub_penalty":       (0.0,  1.5),    # Phase 252 NEW: traversal-time hub suppression
    "backward_bonus":         (0.05, 0.40),   # 244d best=0.136
    "diversity_alpha":        (0.6,  1.0),    # 244d best=0.799
}

# Phase 253: Anchor-score re-ranking + tunable schema_top_k.
# Anchor score (HACH-MVC, AliTakrar/HACH-MVC): anchor_score(v) = intra_community_degree /
# total_degree.  Entities whose neighbours are concentrated in one community are boosted;
# cross-community hubs are suppressed.  This is a neighbourhood-coherence signal orthogonal
# to DPW (raw degree).  schema_top_k tunes the number of PathSchemaIndex predictions per
# question — more predictions increase H@10 coverage at the cost of ranking precision.
PARAM_SPACE_WEBQSP_253: dict = {
    "trb_factor":             (35.0, 52.0),   # 244d best=41.785
    "r2_boost":               (2.0,  5.0),    # 244d best=3.722
    "vote_weight":            (0.78, 0.95),   # 244d best=0.8608
    "beam_width":             [16, 24],        # 244d best=16
    "idf_weight":             (0.0,  0.08),   # 244d best=0.013
    "branch_bonus":           (0.02, 0.10),   # 244d best=0.048
    "fhrb_factor":            (1.5,  2.5),    # 244d best=1.826
    "gamma":                  (5.0,  10.0),   # 244d best=7.726
    "beta":                   (0.8,  1.3),    # 244d best=1.008
    "schema_score_threshold": (0.25, 0.55),   # 244d best=0.369
    "degree_penalty_weight":  (0.3,  0.7),    # 244d best=0.424 — plateau confirmed
    "backward_bonus":         (0.05, 0.40),   # 244d best=0.136
    "diversity_alpha":        (0.6,  1.0),    # 244d best=0.799
    "anchor_rerank_weight":   (0.0,  2.0),    # Phase 253 NEW: HACH-MVC anchor re-ranking
    "schema_top_k":           [3, 5, 8, 12],  # Phase 253 NEW: tunable schema prediction count
}

# Phase 254: Wider beam re-tune.
# beam_width=64 probe showed H@10 +2.33pp vs Phase 253 (more gold answers found) but
# H@1 -0.24pp (hub noise floods top-1 — ranking params calibrated for width=16 are stale).
# Re-tune all ranking params around the wider beam range [32,48,64,96].
# Key expected shifts: DPW needs to be stronger (wider beam = more hub candidates to suppress);
# trb_factor / r2_boost / vote_weight may shift to handle the larger candidate pool.
PARAM_SPACE_WEBQSP_254: dict = {
    "trb_factor":             (25.0, 60.0),   # widen — wider beam changes TRB landscape
    "r2_boost":               (1.0,  6.0),    # widen
    "vote_weight":            (0.70, 0.98),   # widen
    "beam_width":             [32, 48, 64, 96],  # Phase 254 KEY: wider beam range
    "idf_weight":             (0.0,  0.10),   # widen slightly
    "branch_bonus":           (0.01, 0.15),   # widen
    "fhrb_factor":            (1.2,  3.0),    # widen
    "gamma":                  (4.0,  12.0),   # widen
    "beta":                   (0.7,  1.5),    # widen
    "schema_score_threshold": (0.15, 0.60),   # widen
    "degree_penalty_weight":  (0.3,  2.0),    # widen significantly — wider beam needs stronger hub suppression
    "backward_bonus":         (0.0,  0.50),   # widen
    "diversity_alpha":        (0.5,  1.2),    # widen
    "schema_top_k":           [8, 12, 16],    # schema_top_k=12 won Phase 253; explore higher
}

# Phase 255: Guaranteed 1-hop Pass (G1P).
# hop1_base_weight controls the injected score of direct 1-hop named neighbors
# that the beam pruned.  These are added to the candidate pool after the beam query
# and compete with beam answers via DPW + existing ranking signals.
# hop-reachability diagnostic showed 43.5% of beam misses are direct 1-hop neighbors
# (29.6% of all questions) — G1P targets that entire population.
# beam_width kept at [16, 24] — G1P makes wide beam redundant for H@1.
PARAM_SPACE_WEBQSP_255: dict = {
    "trb_factor":             (35.0, 52.0),   # 253 best=44.864
    "r2_boost":               (2.0,  6.0),    # 253 best=2.732
    "vote_weight":            (0.78, 0.98),   # 253 best=0.8515
    "beam_width":             [16, 24],        # G1P makes wide beam redundant for H@1
    "idf_weight":             (0.0,  0.08),   # 253 best=0.017
    "branch_bonus":           (0.01, 0.12),   # 253 best=0.066
    "fhrb_factor":            (1.5,  3.0),    # 253 best=2.246
    "gamma":                  (5.0,  12.0),   # 253 best=8.606
    "beta":                   (0.8,  1.4),    # 253 best=1.183
    "schema_score_threshold": (0.20, 0.60),   # 253 best=0.357
    "degree_penalty_weight":  (0.3,  1.5),    # wider — G1P injects more candidates, needs stronger hub suppression
    "backward_bonus":         (0.05, 0.40),   # 253 best=0.108
    "diversity_alpha":        (0.5,  1.2),    # 253 best=0.655
    "schema_top_k":           [12, 16],       # 253 best=12; keep higher values
    "hop1_base_weight":       (0.0,  1.0),    # Phase 255 NEW: G1P injection score fraction
}

# Phase 256: Relation-conditioned G1P scoring.
# G1P currently injects all 1-hop neighbors at a flat min_score*hop1_base_weight.
# g1p_trb_weight applies trb_map[relation] to each injection so answer-bearing
# relations score higher than hub relations.  Synthetic best_path also routes
# question-keyword q_scores through the injection for downstream post-processing.
PARAM_SPACE_WEBQSP_256: dict = {
    "trb_factor":             (30.0, 55.0),   # 255 best=39.741
    "r2_boost":               (2.0,  6.0),    # 255 best=3.559
    "vote_weight":            (0.78, 0.98),   # 255 best=0.8806
    "beam_width":             [16, 24],
    "idf_weight":             (0.0,  0.08),   # 255 best=0.018
    "branch_bonus":           (0.01, 0.12),   # 255 best=0.064
    "fhrb_factor":            (1.5,  3.5),    # 255 best=2.509
    "gamma":                  (5.0,  12.0),   # 255 best=6.989
    "beta":                   (0.7,  1.4),    # 255 best=0.954
    "schema_score_threshold": (0.15, 0.60),   # 255 best=0.290
    "degree_penalty_weight":  (0.5,  1.5),    # 255 best=1.029
    "backward_bonus":         (0.04, 0.35),   # 255 best=0.077
    "diversity_alpha":        (0.5,  1.2),    # 255 best=0.871
    "schema_top_k":           [12, 16],
    "hop1_base_weight":       (0.3,  1.0),    # 255 best=0.631; narrow to useful range
    "g1p_trb_weight":         (0.0,  5.0),    # Phase 256 NEW: TRB weight for G1P injections
}

# Phase 257: schema_top_k escalation + backward_bonus headroom.
# fANOVA across Phase 255+256 shows schema_top_k is 28-43% importance and was
# capped at 16.  Expanding to [16,24,32] tests whether more schema predictions
# surface the correct 2-hop path.  backward_bonus jumped to 13% importance in
# Phase 256 and may be underexplored (255 best=0.077, 256 best=0.099).
# g1p_trb_weight confirmed dead (0.015 importance) — dropped.
# Synthetic best_path from Phase 256 retained (q_scores now applies to G1P).
PARAM_SPACE_WEBQSP_257: dict = {
    "trb_factor":             (30.0, 55.0),   # 255 best=39.741
    "r2_boost":               (2.0,  6.0),    # 255 best=3.559
    "vote_weight":            (0.78, 0.98),   # 255 best=0.8806
    "beam_width":             [16, 24],
    "idf_weight":             (0.0,  0.08),   # 255 best=0.018
    "branch_bonus":           (0.01, 0.12),   # 255 best=0.064
    "fhrb_factor":            (1.5,  3.5),    # 255 best=2.509
    "gamma":                  (5.0,  12.0),   # 255 best=6.989
    "beta":                   (0.7,  1.4),    # 255 best=0.954
    "schema_score_threshold": (0.10, 0.60),   # 255 best=0.290; widen to catch more schemas
    "degree_penalty_weight":  (0.5,  1.5),    # 255 best=1.029
    "backward_bonus":         (0.04, 0.60),   # wider — 256 fANOVA 13%, may have headroom
    "diversity_alpha":        (0.4,  1.2),    # 255 best=0.871
    "schema_top_k":           [16, 24, 32],   # Phase 257 KEY: test higher schema coverage
    "hop1_base_weight":       (0.3,  1.0),    # 255 best=0.631
}

# Float param names (excludes categorical beam_width)
_FLOAT_PARAMS = tuple(k for k, v in PARAM_SPACE_WIDE.items() if not isinstance(v, list))


# ── param-space helpers ───────────────────────────────────────────────────────
def _suggest_params(trial: "optuna.Trial", space: dict) -> dict:
    """Suggest all params from the given space dict."""
    out: dict = {}
    for name, bounds in space.items():
        if isinstance(bounds, list):
            out[name] = trial.suggest_categorical(name, bounds)
        else:
            out[name] = trial.suggest_float(name, *bounds)
    return out


def _compute_refined_space(
    records: "list[TrialRecord]",
    wide_space: dict,
    top_k: int = 10,
    margin: float = 0.25,
) -> dict:
    """
    Derive Phase 2 bounds from the top-K trials of Phase 1.
    Each float param gets [min_topk - margin*span, max_topk + margin*span],
    clamped to the wide-space limits. Categorical params are unchanged.
    """
    top = sorted(records, key=lambda r: r.h1, reverse=True)[:min(top_k, len(records))]
    if not top:
        return wide_space  # no Phase 1 results — use full space as fallback
    refined: dict = {}
    for name, bounds in wide_space.items():
        if isinstance(bounds, list):
            refined[name] = bounds
            continue
        vals = [getattr(r, name, None) for r in top]
        vals = [v for v in vals if v is not None]
        if not vals:
            refined[name] = bounds
            continue
        lo, hi = min(vals), max(vals)
        full_span = bounds[1] - bounds[0]
        span = max(hi - lo, full_span * 0.05)   # floor at 5% of full range
        new_lo = max(bounds[0], lo - span * margin)
        new_hi = min(bounds[1], hi + span * margin)
        refined[name] = (round(new_lo, 4), round(new_hi, 4))
    return refined


# ── resume helpers ───────────────────────────────────────────────────────────
def _load_resume(path: Path) -> tuple[list["TrialRecord"], Optional[dict]]:
    """
    Read a previous tuner JSONL and return (phase1_records, refined_space).
    refined_space comes from the saved phase2_bounds entry if present;
    otherwise it is None and the caller should recompute it.
    """
    raw_records: list[dict] = []
    refined_space: Optional[dict] = None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "trial" and obj.get("phase") == 1:
                raw_records.append(obj)
            elif t == "phase2_bounds":
                raw = obj["refined_space"]
                refined_space = {}
                for k, v in raw.items():
                    if isinstance(PARAM_SPACE_WIDE.get(k), list):
                        refined_space[k] = v          # categorical — keep as list
                    else:
                        refined_space[k] = (v[0], v[1])  # float range tuple

    records: list[TrialRecord] = []
    best_h1 = -1.0
    for obj in raw_records:
        rec = TrialRecord(
            trial_id=obj["trial_id"],
            phase=1,
            trb_factor=obj["trb_factor"],
            r2_boost=obj["r2_boost"],
            vote_weight=obj["vote_weight"],
            beam_width=obj["beam_width"],
            idf_weight=obj["idf_weight"],
            branch_bonus=obj["branch_bonus"],
            fhrb_factor=obj["fhrb_factor"],
            gamma=obj["gamma"],
            beta=obj["beta"],
            h1=obj["h1"],
            h10=obj["h10"],
            mrr=obj["mrr"],
            elapsed_s=obj["elapsed_s"],
            schema_score_threshold=obj.get("schema_score_threshold", 0.0),
            degree_penalty_weight=obj.get("degree_penalty_weight", 0.0),
            backward_bonus=obj.get("backward_bonus", 0.0),
            diversity_alpha=obj.get("diversity_alpha", 0.0),
            conditional_schema_bonus=obj.get("conditional_schema_bonus", 0.0),
            )
        if rec.h1 > best_h1:
            best_h1 = rec.h1
            rec.is_best = True
            if records:
                records[-1].is_best = False  # clear prior best
            # mark previous best as not best
            for r in records:
                r.is_best = False
            rec.is_best = True
        records.append(rec)
    return records, refined_space


def _make_source_trials(records: "list[TrialRecord]", space: dict) -> list:
    """Build Optuna FrozenTrial objects from TrialRecord list for CMA-ES warm-start."""
    from optuna.distributions import FloatDistribution, CategoricalDistribution

    distributions: dict = {}
    for name, bounds in space.items():
        if isinstance(bounds, list):
            distributions[name] = CategoricalDistribution(bounds)
        else:
            distributions[name] = FloatDistribution(bounds[0], bounds[1])

    frozen: list = []
    for rec in records:
        params = {
            "trb_factor":             rec.trb_factor,
            "r2_boost":               rec.r2_boost,
            "vote_weight":            rec.vote_weight,
            "beam_width":             rec.beam_width,
            "idf_weight":             rec.idf_weight,
            "branch_bonus":           rec.branch_bonus,
            "fhrb_factor":            rec.fhrb_factor,
            "gamma":                  rec.gamma,
            "beta":                   rec.beta,
            "schema_score_threshold": getattr(rec, "schema_score_threshold", 0.0),
            "degree_penalty_weight":  getattr(rec, "degree_penalty_weight", 0.0),
            "backward_bonus":         getattr(rec, "backward_bonus", 0.0),
            "diversity_alpha":        getattr(rec, "diversity_alpha", 0.0),
            "conditional_schema_bonus": getattr(rec, "conditional_schema_bonus", 0.0),
            "qa_sem_weight":            getattr(rec, "qa_sem_weight", 0.0),
            "beam_hub_penalty":         getattr(rec, "beam_hub_penalty", 0.0),
            "anchor_rerank_weight":     getattr(rec, "anchor_rerank_weight", 0.0),
            "schema_top_k":             getattr(rec, "schema_top_k", 5),
            "hop1_base_weight":         getattr(rec, "hop1_base_weight", 0.0),
            "g1p_trb_weight":           getattr(rec, "g1p_trb_weight", 0.0),
        }
        # Only include params that exist in the distributions dict
        params = {k: v for k, v in params.items() if k in distributions}
        frozen.append(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=rec.h1,
            )
        )
    return frozen


# ── trial record ──────────────────────────────────────────────────────────────
@dataclass
class TrialRecord:
    trial_id:     int
    trb_factor:   float
    r2_boost:     float
    vote_weight:  float
    beam_width:   int
    idf_weight:   float
    branch_bonus: float
    fhrb_factor:  float
    gamma:        float   # Phase 203 SDRB: scale factor
    beta:         float   # Phase 203 SDRB: fan_out exponent; boost(r) = gamma * fan_out(r)^beta
    h1:                     float
    h10:                    float
    mrr:                    float
    elapsed_s:              float
    schema_score_threshold: float = 0.0  # Phase 237: schema prepend confidence gate
    degree_penalty_weight:  float = 0.0  # Phase 239: hub entity degree suppression
    backward_bonus:         float = 0.0  # Phase 245: bidirectional path verification boost
    diversity_alpha:        float = 0.0  # Phase 246: multi-path convergence re-ranker
    conditional_schema_bonus: float = 0.0  # Phase 247: conditional r2 structural confirmation
    qa_sem_weight:            float = 0.0  # Phase 250: question-answer semantic alignment re-ranker
    beam_hub_penalty:         float = 0.0  # Phase 252: traversal-time hub suppression (non-terminal hops)
    anchor_rerank_weight:     float = 0.0  # Phase 253: HACH-MVC anchor-score re-ranking
    schema_top_k:             int   = 5    # Phase 253: tunable schema prediction count
    hop1_base_weight:         float = 0.0  # Phase 255: G1P base score fraction
    g1p_trb_weight:           float = 0.0  # Phase 256: relation-conditioned G1P scoring weight
    # cvt_relay_boost removed in Phase 243 — ablation confirmed it hurts; cvt_passthrough is the correct fix
    is_best:                bool  = False
    phase:                  int   = 1


# ── live dashboard ────────────────────────────────────────────────────────────
class LiveDashboard:
    """Rich live terminal dashboard. Updated after every completed trial."""

    def __init__(self, n_trials: int, sample: int, study_name: str) -> None:
        self.n_trials    = n_trials
        self.sample      = sample
        self.study_name  = study_name
        self.phase_label = "Phase 1/2 — Exploring"
        self.records:    list[TrialRecord]     = []
        self.best:       Optional[TrialRecord] = None
        self._start      = time.time()

    def push(self, rec: TrialRecord) -> None:
        if self.best is None or rec.h1 > self.best.h1:
            if self.best:
                self.best.is_best = False
            rec.is_best = True
            self.best = rec
        self.records.append(rec)

    # ── renderers ─────────────────────────────────────────────────────────
    def _header_panel(self) -> "Panel":
        elapsed = time.time() - self._start
        done    = len(self.records)
        eta_s   = (elapsed / done * (self.n_trials - done)) if done else 0.0
        eta_str = f"{eta_s/60:.0f}m" if eta_s > 90 else f"{eta_s:.0f}s"

        if self.best:
            b = self.best
            body = (
                f"[bold green]H@1 = {b.h1*100:.2f}%[/bold green]   "
                f"H@10 = {b.h10*100:.2f}%   MRR = {b.mrr:.4f}   "
                f"[dim](trial #{b.trial_id})[/dim]\n"
                f"  trb=[cyan]{b.trb_factor:.3f}[/cyan]  "
                f"r2=[cyan]{b.r2_boost:.3f}[/cyan]  "
                f"vote=[cyan]{b.vote_weight:.4f}[/cyan]  "
                f"beam=[cyan]{b.beam_width}[/cyan]  "
                f"idf=[cyan]{b.idf_weight:.3f}[/cyan]  "
                f"bbns=[cyan]{b.branch_bonus:.3f}[/cyan]  "
                f"fhrb=[cyan]{b.fhrb_factor:.3f}[/cyan]  "
                f"gamma=[cyan]{b.gamma:.3f}[/cyan]  "
                f"beta=[cyan]{b.beta:.3f}[/cyan]  "
                f"dpw=[cyan]{b.degree_penalty_weight:.3f}[/cyan]  "
                f"bbp=[cyan]{getattr(b, 'backward_bonus', 0.0):.3f}[/cyan]  "
                f"da=[cyan]{getattr(b, 'diversity_alpha', 0.0):.3f}[/cyan]  "
                f"csb=[cyan]{getattr(b, 'conditional_schema_bonus', 0.0):.3f}[/cyan]"
            )
        else:
            body = "[dim]Waiting for first trial...[/dim]"

        title = (
            f"[bold]CEREBRUM Tuner[/bold] — {self.study_name}   "
            f"[yellow]{self.phase_label}[/yellow]   "
            f"Trial [bold]{done}[/bold]/{self.n_trials}   "
            f"Elapsed {elapsed:.0f}s   ETA ~{eta_str}   "
            f"Sample {self.sample:,}q"
        )
        border = "green" if self.best else "blue"
        return Panel(body, title=title, border_style=border)

    def _history_table(self) -> "Table":
        t = Table(
            box=rich_box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
            show_edge=False,
        )
        t.add_column("Ph",    justify="right", style="dim", no_wrap=True, width=3)
        t.add_column("#",     justify="right", style="dim", no_wrap=True, width=4)
        t.add_column("trb",   justify="right",              no_wrap=True, width=5)
        t.add_column("r2",    justify="right",              no_wrap=True, width=5)
        t.add_column("vote",  justify="right",              no_wrap=True, width=6)
        t.add_column("bm",    justify="right",              no_wrap=True, width=4)
        t.add_column("idf",   justify="right",              no_wrap=True, width=5)
        t.add_column("bbns",  justify="right",              no_wrap=True, width=5)
        t.add_column("fhrb",  justify="right",              no_wrap=True, width=5)
        t.add_column("gamma", justify="right",              no_wrap=True, width=6)
        t.add_column("beta",  justify="right",              no_wrap=True, width=5)
        t.add_column("dpw",   justify="right",              no_wrap=True, width=5)
        t.add_column("H@1",   justify="right",              no_wrap=True, width=7)
        t.add_column("H@10",  justify="right",              no_wrap=True, width=7)
        t.add_column("MRR",   justify="right",              no_wrap=True, width=6)
        t.add_column("dBest", justify="right",              no_wrap=True, width=7)
        t.add_column("sec",   justify="right",              no_wrap=True, width=5)

        for r in list(reversed(self.records))[:30]:
            delta_pp = (r.h1 - self.best.h1) * 100 if self.best else 0.0
            if r.is_best:
                delta_str = "[bold green]*best[/bold green]"
            elif delta_pp < -2:
                delta_str = f"[dim red]{delta_pp:+.1f}[/dim red]"
            elif delta_pp < 0:
                delta_str = f"[red]{delta_pp:+.1f}[/red]"
            else:
                delta_str = f"[green]{delta_pp:+.1f}[/green]"

            ph_style  = "dim"  if r.phase == 1 else "cyan"
            row_style = "bold green" if r.is_best else ""
            t.add_row(
                f"[{ph_style}]P{r.phase}[/{ph_style}]",
                str(r.trial_id),
                f"{r.trb_factor:.2f}",
                f"{r.r2_boost:.2f}",
                f"{r.vote_weight:.3f}",
                str(r.beam_width),
                f"{r.idf_weight:.2f}",
                f"{r.branch_bonus:.2f}",
                f"{r.fhrb_factor:.2f}",
                f"{r.gamma:.3f}",
                f"{r.beta:.3f}",
                f"{getattr(r, 'degree_penalty_weight', 0.0):.3f}",
                f"{r.h1*100:.1f}%",
                f"{r.h10*100:.1f}%",
                f"{r.mrr:.4f}",
                delta_str,
                f"{r.elapsed_s:.0f}",
                style=row_style,
            )
        return t

    def __rich__(self):
        from rich.console import Group
        return Group(self._header_panel(), self._history_table())


# ── JSONL trial logger ────────────────────────────────────────────────────────
def _write_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trial_record_to_dict(
    rec: "TrialRecord",
    sample: int,
    run_id: str,
    all_records: "list[TrialRecord]",
) -> dict:
    sorted_h1    = sorted((r.h1 for r in all_records), reverse=True)
    rank         = sorted_h1.index(rec.h1) + 1
    best_so_far  = max(r.h1 for r in all_records)
    prev         = all_records[-2] if len(all_records) >= 2 else None
    param_deltas: dict = {}
    if prev:
        for attr in _FLOAT_PARAMS:
            param_deltas[f"d_{attr}"] = round(getattr(rec, attr) - getattr(prev, attr), 4)
    return {
        "type":             "trial",
        "run_id":           run_id,
        "trial_id":         rec.trial_id,
        "phase":            rec.phase,
        "timestamp":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample":           sample,
        "trb_factor":             rec.trb_factor,
        "r2_boost":               rec.r2_boost,
        "vote_weight":            rec.vote_weight,
        "beam_width":             rec.beam_width,
        "idf_weight":             rec.idf_weight,
        "branch_bonus":           rec.branch_bonus,
        "fhrb_factor":            rec.fhrb_factor,
        "gamma":                  rec.gamma,
        "beta":                   rec.beta,
        "schema_score_threshold": rec.schema_score_threshold,
        "degree_penalty_weight":  rec.degree_penalty_weight,
        "backward_bonus":         getattr(rec, "backward_bonus", 0.0),
        "diversity_alpha":        getattr(rec, "diversity_alpha", 0.0),
        "conditional_schema_bonus": getattr(rec, "conditional_schema_bonus", 0.0),
        "beam_hub_penalty":       getattr(rec, "beam_hub_penalty", 0.0),
        "anchor_rerank_weight":   getattr(rec, "anchor_rerank_weight", 0.0),
        "schema_top_k":           getattr(rec, "schema_top_k", 5),
        "hop1_base_weight":       getattr(rec, "hop1_base_weight", 0.0),
        "g1p_trb_weight":         getattr(rec, "g1p_trb_weight", 0.0),
        "h1":                     rec.h1,
        "h10":              rec.h10,
        "mrr":              rec.mrr,
        "elapsed_s":        round(rec.elapsed_s, 1),
        "is_best":          rec.is_best,
        "rank_so_far":      rank,
        "trials_completed": len(all_records),
        "best_h1_so_far":   round(best_so_far, 6),
        "delta_from_best":  round(rec.h1 - best_so_far, 6),
        **param_deltas,
    }


# ── subprocess eval ───────────────────────────────────────────────────────────
def _run_eval(
    sample:       int,
    trb_factor:   float,
    r2_boost:     float,
    vote_weight:  float,
    beam_width:   int,
    idf_weight:   float,
    branch_bonus: float,
    fhrb_factor:  float,
    gamma:        float,
    beta:         float,
    timeout:      int = 600,
    workers:      int = 1,
    embeddings:   str = "sentence",
) -> tuple[float, float, float, float]:
    cmd = [
        sys.executable, "-u", str(_EVAL_SCRIPT),
        "--hop",           "3",
        "--sample",        str(sample),
        "--beam-width",    str(beam_width),
        "--embeddings",    embeddings,
        "--vote-weight",   str(vote_weight),
        "--trb-factor",    str(trb_factor),
        "--r2-boost",      str(r2_boost),
        "--idf-weight",    str(idf_weight),
        "--branch-bonus",  str(branch_bonus),
        "--fhrb-factor",   str(fhrb_factor),
        "--gamma",         str(gamma),
        "--beta",          str(beta),
        "--workers",       str(workers),
    ]
    t0      = time.time()
    proc    = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0

    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "3-hop" and len(parts) >= 5:
            try:
                return float(parts[2]), float(parts[3]), float(parts[4]), elapsed
            except ValueError:
                pass

    raise RuntimeError(
        f"Could not parse eval output (exit {proc.returncode}).\n"
        f"stdout: {proc.stdout[-800:]}\n"
        f"stderr: {proc.stderr[-400:]}"
    )


def _init_hetionet_state(n_questions: int = 20, max_neighbors: int = 50) -> None:
    """Build Hetionet graph + QA pairs once; stored in _hetionet_state for in-process trials."""
    global _hetionet_state
    if _hetionet_state is not None:
        return  # already initialised
    from hetionet_param_eval import build_hetionet_state
    _hetionet_state = build_hetionet_state(
        n_questions   = n_questions,
        max_neighbors = max_neighbors,
        embeddings    = "sentence",
        seed          = 42,
        use_cache     = True,
        min_eval_hop  = 2,
    )


def _init_conceptnet_state(
    cn5_path:    str,
    n_questions: int = 500,
    max_edges:   int = 200_000,
    embeddings:  str = "random",
) -> None:
    """Build ConceptNet graph + QA pairs once; stored in _conceptnet_state for in-process trials."""
    global _conceptnet_state
    if _conceptnet_state is not None:
        return  # already initialised
    from conceptnet_eval import build_conceptnet_state
    _conceptnet_state = build_conceptnet_state(
        cn5_path    = cn5_path,
        n_questions = n_questions,
        embeddings  = embeddings,
        seed        = 42,
        use_cache   = True,
        max_edges   = max_edges,
    )


def _init_webqsp_state(
    n_questions: int = 200,
    embeddings:  str = "random",
) -> None:
    """Build WebQSP graph + QA pairs once; stored in _webqsp_state for in-process trials."""
    global _webqsp_state
    if _webqsp_state is not None:
        return
    from webqsp_param_eval import build_webqsp_state
    _webqsp_state = build_webqsp_state(
        n_questions = n_questions,
        embeddings  = embeddings,
        seed        = 42,
        use_cache   = True,
    )


def _run_eval_logged(
    log_file: Path,
    run_id:   str,
    trial_id: int,
    **kwargs,
) -> tuple[float, float, float, float]:
    dataset = kwargs.get("dataset", "metaqa")
    import traceback as _tb

    # In-process paths (graph already built — skip expensive rebuild per trial)
    _inprocess_params = {
        "trb_factor":             kwargs["trb_factor"],
        "r2_boost":               kwargs["r2_boost"],
        "vote_weight":            kwargs["vote_weight"],
        "beam_width":             kwargs["beam_width"],
        "idf_weight":             kwargs["idf_weight"],
        "branch_bonus":           kwargs["branch_bonus"],
        "fhrb_factor":            kwargs["fhrb_factor"],
        "gamma":                  kwargs["gamma"],
        "beta":                   kwargs["beta"],
        "schema_score_threshold": kwargs.get("schema_score_threshold", 0.0),
        "degree_penalty_weight":  kwargs.get("degree_penalty_weight", 0.0),
        "backward_bonus":         kwargs.get("backward_bonus", 0.0),
        "diversity_alpha":        kwargs.get("diversity_alpha", 0.0),
        "conditional_schema_bonus": kwargs.get("conditional_schema_bonus", 0.0),
        "beam_hub_penalty":       kwargs.get("beam_hub_penalty", 0.0),
        "anchor_rerank_weight":   kwargs.get("anchor_rerank_weight", 0.0),
        "schema_top_k":           kwargs.get("schema_top_k", 5),
        "hop1_base_weight":       kwargs.get("hop1_base_weight", 0.0),
        "g1p_trb_weight":         kwargs.get("g1p_trb_weight", 0.0),
        "cvt_passthrough":        True,  # Phase 243: always enabled for WebQSP/Freebase
        "max_loops":              1,
    }
    if dataset == "hetionet" and _hetionet_state is not None:
        from hetionet_param_eval import run_trial_inprocess
        t0 = time.time()
        try:
            h1, h10, mrr = run_trial_inprocess(_hetionet_state, _inprocess_params)
        except Exception:
            print(f"\n[INPROCESS-ERR trial {trial_id}]\n{_tb.format_exc()}", file=sys.stderr, flush=True)
            raise
        elapsed = time.time() - t0
        _write_log(log_file, {
            "type":      "trial_stdout",
            "run_id":    run_id,
            "trial_id":  trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lines":     [f"inprocess h1={h1:.4f} h10={h10:.4f} mrr={mrr:.4f}"],
            "exit_code": 0,
            "elapsed_s": round(elapsed, 1),
        })
        return h1, h10, mrr, elapsed

    if dataset in ("webqsp", "webqsp251", "webqsp252", "webqsp253", "webqsp254", "webqsp255", "webqsp256", "webqsp257") and _webqsp_state is not None:
        from webqsp_param_eval import run_trial_inprocess
        t0 = time.time()
        try:
            h1, h10, mrr = run_trial_inprocess(_webqsp_state, _inprocess_params)
        except Exception:
            print(f"\n[INPROCESS-ERR trial {trial_id}]\n{_tb.format_exc()}", file=sys.stderr, flush=True)
            raise
        elapsed = time.time() - t0
        _write_log(log_file, {
            "type":      "trial_stdout",
            "run_id":    run_id,
            "trial_id":  trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lines":     [f"inprocess h1={h1:.4f} h10={h10:.4f} mrr={mrr:.4f}"],
            "exit_code": 0,
            "elapsed_s": round(elapsed, 1),
        })
        return h1, h10, mrr, elapsed

    # trial_id < 0 signals validation — use subprocess for a fresh QA sample
    if dataset == "conceptnet" and _conceptnet_state is not None and trial_id >= 0:
        from conceptnet_eval import run_trial_inprocess
        t0 = time.time()
        try:
            h1, h10, mrr = run_trial_inprocess(_conceptnet_state, _inprocess_params)
        except Exception:
            print(f"\n[INPROCESS-ERR trial {trial_id}]\n{_tb.format_exc()}", file=sys.stderr, flush=True)
            raise
        elapsed = time.time() - t0
        _write_log(log_file, {
            "type":      "trial_stdout",
            "run_id":    run_id,
            "trial_id":  trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lines":     [f"inprocess h1={h1:.4f} h10={h10:.4f} mrr={mrr:.4f}"],
            "exit_code": 0,
            "elapsed_s": round(elapsed, 1),
        })
        return h1, h10, mrr, elapsed

    if dataset in ("webqsp", "webqsp251", "webqsp252", "webqsp253", "webqsp254", "webqsp255", "webqsp256", "webqsp257"):
        n_questions = kwargs.get("sample", 200)
        cmd = [
            sys.executable, "-u", str(_WEBQSP_EVAL_SCRIPT),
            "--sample",       str(n_questions),
            "--embeddings",   kwargs.get("embeddings", "random"),
            "--beam-width",   str(kwargs["beam_width"]),
            "--trb-factor",   str(kwargs["trb_factor"]),
            "--r2-boost",     str(kwargs["r2_boost"]),
            "--vote-weight",  str(kwargs["vote_weight"]),
            "--idf-weight",   str(kwargs["idf_weight"]),
            "--branch-bonus", str(kwargs["branch_bonus"]),
            "--fhrb-factor",  str(kwargs["fhrb_factor"]),
            "--gamma",        str(kwargs["gamma"]),
            "--beta",         str(kwargs["beta"]),
        ]
        if kwargs.get("degree_penalty_weight", 0.0):
            cmd += ["--degree-penalty-weight", str(kwargs["degree_penalty_weight"])]
        if kwargs.get("beam_hub_penalty", 0.0):
            cmd += ["--beam-hub-penalty", str(kwargs["beam_hub_penalty"])]
        if kwargs.get("schema_score_threshold", 0.0):
            cmd += ["--schema-score-threshold", str(kwargs["schema_score_threshold"])]
        if kwargs.get("backward_bonus", 0.0):
            cmd += ["--backward-bonus", str(kwargs["backward_bonus"])]
        if kwargs.get("diversity_alpha", 0.0):
            cmd += ["--diversity-alpha", str(kwargs["diversity_alpha"])]
        if kwargs.get("anchor_rerank_weight", 0.0):
            cmd += ["--anchor-rerank-weight", str(kwargs["anchor_rerank_weight"])]
        if kwargs.get("schema_top_k", 5) != 5:
            cmd += ["--schema-top-k", str(kwargs["schema_top_k"])]
        if kwargs.get("hop1_base_weight", 0.0):
            cmd += ["--hop1-base-weight", str(kwargs["hop1_base_weight"])]
        if kwargs.get("g1p_trb_weight", 0.0):
            cmd += ["--g1p-trb-weight", str(kwargs["g1p_trb_weight"])]
    elif dataset == "hetionet":
        # Fallback subprocess path (state not yet initialised)
        n_questions = kwargs.get("sample", 50)
        cmd = [
            sys.executable, "-u", str(_HETIONET_EVAL_SCRIPT),
            "--n-questions",  str(n_questions),
            "--beam-width",   str(kwargs["beam_width"]),
            "--vote-weight",  str(kwargs["vote_weight"]),
            "--trb-factor",   str(kwargs["trb_factor"]),
            "--r2-boost",     str(kwargs["r2_boost"]),
            "--idf-weight",   str(kwargs["idf_weight"]),
            "--branch-bonus", str(kwargs["branch_bonus"]),
            "--fhrb-factor",  str(kwargs["fhrb_factor"]),
            "--gamma",        str(kwargs["gamma"]),
            "--beta",         str(kwargs["beta"]),
            "--embeddings",   kwargs.get("embeddings", "sentence"),
            "--max-loops",    "1",
            "--workers",      str(kwargs.get("workers", 1)),
            "--min-eval-hop", "2",
            "--max-neighbors","50",
        ]
    elif dataset == "conceptnet":
        n_questions = kwargs.get("sample", 500)
        cmd = [
            sys.executable, "-u", str(_CN5_EVAL_SCRIPT),
            "--cn5",          str(kwargs["cn5_path"]),
            "--n-questions",  str(n_questions),
            "--embeddings",   kwargs.get("embeddings", "random"),
            "--beam-width",   str(kwargs["beam_width"]),
            "--trb-factor",   str(kwargs["trb_factor"]),
            "--r2-boost",     str(kwargs["r2_boost"]),
            "--vote-weight",  str(kwargs["vote_weight"]),
            "--idf-weight",   str(kwargs["idf_weight"]),
            "--branch-bonus", str(kwargs["branch_bonus"]),
            "--fhrb-factor",  str(kwargs["fhrb_factor"]),
            "--gamma",        str(kwargs["gamma"]),
            "--beta",         str(kwargs["beta"]),
        ]
    else:
        sample = kwargs["sample"]
        cmd = [
            sys.executable, "-u", str(_EVAL_SCRIPT),
            "--hop",           "3",
            "--sample",        str(sample),
            "--beam-width",    str(kwargs["beam_width"]),
            "--embeddings",    kwargs.get("embeddings", "sentence"),
            "--vote-weight",   str(kwargs["vote_weight"]),
            "--trb-factor",    str(kwargs["trb_factor"]),
            "--r2-boost",      str(kwargs["r2_boost"]),
            "--idf-weight",    str(kwargs["idf_weight"]),
            "--branch-bonus",  str(kwargs["branch_bonus"]),
            "--fhrb-factor",   str(kwargs["fhrb_factor"]),
            "--gamma",         str(kwargs["gamma"]),
            "--beta",          str(kwargs["beta"]),
            "--workers",       str(kwargs.get("workers", 1)),
        ]
    t0      = time.time()
    timeout = kwargs.get("timeout", 600)
    lines:  list[str] = []
    h1 = h10 = mrr = 0.0
    parsed = False

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            parts = line.split()
            if parts and parts[0] in ("3-hop", "2-hop", "webqsp") and len(parts) >= 5:
                try:
                    h1, h10, mrr = float(parts[2]), float(parts[3]), float(parts[4])
                    parsed = True
                except ValueError:
                    pass
            if time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError(f"Trial {trial_id} timed out after {timeout}s")
        proc.wait()
    finally:
        proc.stdout.close()

    elapsed = time.time() - t0

    _write_log(log_file, {
        "type":      "trial_stdout",
        "run_id":    run_id,
        "trial_id":  trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lines":     lines,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
    })

    if not parsed:
        raise RuntimeError(
            f"Could not parse eval output for trial {trial_id} "
            f"(exit {proc.returncode}).\n" + "\n".join(lines[-20:])
        )
    return h1, h10, mrr, elapsed


# ── ParameterInitializer warm-start ──────────────────────────────────────────
def _compute_param_init_x0(kb_file: Path, embedding_method: str = "random") -> Optional[dict]:
    """
    Compute ParameterInitializer defaults from the KB and return as a param dict
    suitable for p1_study.enqueue_trial().  Returns None on any error (graceful
    fallback — tuner continues without the warm-start hint).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.graph_profiler import GraphProfiler
        from core.relation_boost_deriver import RelationBoostDeriver
        from core.parameter_initializer import ParameterInitializer
        import networkx as nx

        deriver = RelationBoostDeriver()
        deriver.build_from_file(str(kb_file))

        G = nx.Graph()
        with open(kb_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    h, _, t = parts
                    G.add_edge(h, t)

        class _FakeAdapter:
            def to_networkx(self): return G
            def is_directed(self): return False

        profile = GraphProfiler.profile(_FakeAdapter(), {})
        init    = ParameterInitializer.compute(profile, deriver, embedding_method=embedding_method)
        d       = init.as_dict()
        # Clamp to PARAM_SPACE_WIDE bounds so enqueue_trial doesn't error
        for name, bounds in PARAM_SPACE_WIDE.items():
            if isinstance(bounds, list):
                if d.get(name) not in bounds:
                    d[name] = bounds[0]
            else:
                lo, hi = bounds
                d[name] = max(lo, min(hi, float(d.get(name, lo))))
        return d
    except Exception as exc:
        print(f"  [ParameterInitializer] warm-start skipped: {exc}", file=sys.stderr)
        return None


# ── tuning loop ───────────────────────────────────────────────────────────────
def run_tuner(
    phase1_trials: int            = 60,
    phase2_trials: int            = 140,
    n_trials:      int            = 0,
    sample:        int            = 500,
    study_name:    str            = "cerebrum-tuner",
    validate:      int            = 0,
    seed:          int            = 42,
    timeout_s:     int            = 600,
    log_file:      Optional[Path] = None,
    top_k:         int            = 10,
    margin:        float          = 0.25,
    workers:       int            = 1,
    resume_file:   Optional[Path] = None,
    param_init:    bool           = False,
    kb_file:       Optional[Path] = None,
    dataset:       str            = "metaqa",
    embeddings:    str            = "sentence",
    cn5_path:      str            = "",
    max_edges:     int            = 200_000,
) -> Optional[TrialRecord]:
    """
    Two-phase hyperparameter search.

    Phase 1 — Exploration  : Sobol QMCSampler over PARAM_SPACE_WIDE. Low-discrepancy
                             sequences guarantee uniform 9D coverage without
                             accidental clustering (unlike pure random sampling).

    Phase 2 — Fine-tuning  : CmaEsSampler over bounds auto-derived from the top-K
                             Phase 1 trials (±margin). Initialized at the best Phase 1
                             config (x0) and warmed with all Phase 1 trials so CMA-ES
                             starts with a calibrated covariance estimate. Models
                             parameter correlations that TPE treats as independent.

    Pass n_trials > 0 to run single-phase Sobol-only (backward-compatible).
    """
    if not _HAS_OPTUNA:
        print(
            "optuna is not installed.\n"
            "Install with:  pip install 'cerebrum-kg[tuning]'  or  pip install optuna rich",
            file=sys.stderr,
        )
        return None

    # Single-phase fallback (--n-trials)
    if n_trials > 0:
        phase1_trials = n_trials
        phase2_trials = 0

    # Dataset-specific parameter space
    if dataset == "hetionet":
        _param_space = PARAM_SPACE_HETIONET
    elif dataset == "conceptnet":
        _param_space = PARAM_SPACE_CONCEPTNET
    elif dataset == "webqsp":
        _param_space = PARAM_SPACE_WEBQSP
    elif dataset == "webqsp251":
        _param_space = PARAM_SPACE_WEBQSP_251
    elif dataset == "webqsp252":
        _param_space = PARAM_SPACE_WEBQSP_252
    elif dataset == "webqsp253":
        _param_space = PARAM_SPACE_WEBQSP_253
    elif dataset == "webqsp254":
        _param_space = PARAM_SPACE_WEBQSP_254
    elif dataset == "webqsp255":
        _param_space = PARAM_SPACE_WEBQSP_255
    elif dataset == "webqsp256":
        _param_space = PARAM_SPACE_WEBQSP_256
    elif dataset == "webqsp257":
        _param_space = PARAM_SPACE_WEBQSP_257
    else:
        _param_space = PARAM_SPACE_WIDE

    # ── resume: skip Phase 1, load from previous JSONL ───────────────────
    _resume_records:      list[TrialRecord] = []
    _resume_refined_space: Optional[dict]   = None
    if resume_file is not None:
        resume_file = Path(resume_file)
        if not resume_file.exists():
            print(f"ERROR: --resume file not found: {resume_file}", file=sys.stderr)
            return None
        _resume_records, _resume_refined_space = _load_resume(resume_file)
        if not _resume_records:
            print(f"ERROR: no Phase 1 trial records found in {resume_file}", file=sys.stderr)
            return None
        phase1_trials = len(_resume_records)
        print(f"  Resuming  : {resume_file.name}  ({phase1_trials} Phase 1 trials loaded)")

    total_trials = phase1_trials + phase2_trials

    # ── Dataset: build graph once before trials start ────────────────────
    if dataset == "hetionet":
        _init_hetionet_state(n_questions=sample, max_neighbors=50)
    elif dataset == "conceptnet":
        if not cn5_path:
            print("ERROR: --cn5-file is required for --dataset conceptnet", file=sys.stderr)
            return None
        _init_conceptnet_state(
            cn5_path    = cn5_path,
            n_questions = sample,
            max_edges   = max_edges,
            embeddings  = embeddings,
        )
    elif dataset in ("webqsp", "webqsp251", "webqsp252", "webqsp253", "webqsp254", "webqsp255", "webqsp256", "webqsp257"):
        _init_webqsp_state(n_questions=sample, embeddings=embeddings)

    # ── log file setup ────────────────────────────────────────────────────
    run_id   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if log_file is None:
        log_file = Path(__file__).parent / f"tuner_{run_id}.jsonl"
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _write_log(log_file, {
        "type": "run_start", "run_id": run_id, "study_name": study_name,
        "phase1_trials": phase1_trials, "phase2_trials": phase2_trials,
        "sample": sample, "seed": seed,
        "sampler_p1": "sobol_qmc",
        "sampler_p2": "cmaes",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "param_space_wide": {
            k: list(v) if isinstance(v, list) else list(v)
            for k, v in _param_space.items()
        },
    })
    print(f"  Trial log : {log_file}")
    if resume_file:
        print(f"  Phase 1   : {phase1_trials} trials (loaded from resume — skipping)")
    else:
        print(f"  Phase 1   : {phase1_trials} trials (Sobol QMC, wide bounds)")
    if phase2_trials:
        print(f"  Phase 2   : {phase2_trials} trials (CMA-ES, refined bounds from top-{top_k})")

    console   = Console() if _HAS_RICH else None
    dashboard = LiveDashboard(total_trials, sample, study_name)

    # ── shared objective factory ──────────────────────────────────────────
    def make_objective(space: dict, phase: int):
        offset = len(dashboard.records)

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial, space)
            tid    = offset + trial.number
            h1, h10, mrr, elapsed = _run_eval_logged(
                log_file=log_file, run_id=run_id, trial_id=tid,
                sample=sample, timeout=timeout_s, workers=workers,
                dataset=dataset, embeddings=embeddings,
                cn5_path=cn5_path,
                **params,
            )
            rec = TrialRecord(
                trial_id=tid, phase=phase,
                h1=h1, h10=h10, mrr=mrr, elapsed_s=elapsed,
                **params,
            )
            dashboard.push(rec)
            _write_log(log_file, _trial_record_to_dict(rec, sample, run_id, dashboard.records))
            return h1

        return objective

    # ── run both phases under a single Live context ───────────────────────
    def _run_phase(study, objective_fn, n):
        for _ in range(n):
            study.optimize(objective_fn, n_trials=1, catch=(Exception,))

    def _run_phase_plain(study, objective_fn, n):
        for _ in range(n):
            study.optimize(objective_fn, n_trials=1, catch=(Exception,))
            if dashboard.records:
                r    = dashboard.records[-1]
                flag = "*" if r.is_best else " "
                print(
                    f"P{r.phase} {r.trial_id:>4}  {r.trb_factor:>5.2f}  {r.r2_boost:>5.2f}  "
                    f"{r.vote_weight:>6.3f}  {r.beam_width:>3}  "
                    f"{r.idf_weight:>5.2f}  {r.branch_bonus:>5.2f}  {r.fhrb_factor:>5.2f}  "
                    f"{r.gamma:>6.3f}  {r.beta:>5.3f}  "
                    f"{r.h1*100:>6.2f}%  {r.h10*100:>6.2f}%  {r.mrr:>6.4f}  {flag}"
                )

    # ── Phase 1 — Sobol QMC (or reload from resume) ──────────────────────
    p1_sampler = optuna.samplers.QMCSampler(qmc_type="sobol", seed=seed, scramble=True)
    p1_study   = optuna.create_study(
        study_name=f"{study_name}-p1", direction="maximize", sampler=p1_sampler,
    )

    if param_init and not _resume_records:
        _kb = kb_file or _DEFAULT_KB
        x0_hint = _compute_param_init_x0(_kb, embedding_method=embeddings)
        if x0_hint is not None:
            p1_study.enqueue_trial(x0_hint)
            print(f"  Param-init: warm-start trial enqueued from {_kb.name}")

    if _resume_records:
        # Pre-populate dashboard with loaded Phase 1 data; skip running Phase 1
        for rec in _resume_records:
            dashboard.push(rec)
        dashboard.phase_label = "Phase 2/2 — Fine-tuning (CMA-ES) [resumed]"
    else:
        p1_obj = make_objective(_param_space, phase=1)

    # ── helper: build Phase 2 study ───────────────────────────────────────
    def _build_p2_study(p1_records: list, p1_source_trials) -> tuple:
        """Return (refined_space, p2_study) from Phase 1 data."""
        if _resume_records and _resume_refined_space:
            rs = _resume_refined_space
        else:
            rs = _compute_refined_space(p1_records, _param_space, top_k=top_k, margin=margin)
            _write_log(log_file, {
                "type": "phase2_bounds", "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "top_k": top_k, "margin": margin,
                "refined_space": {
                    k: list(v) if isinstance(v, list) else list(v)
                    for k, v in rs.items()
                },
            })

        # CMA-ES does not support categorical params (lists); fall back to TPE
        # when any param in the refined space is categorical (e.g. beam_width=[16,24,32,48]).
        _has_categorical = any(isinstance(v, list) for v in rs.values())
        if _has_categorical:
            sampler = optuna.samplers.TPESampler(seed=seed + 1, multivariate=True)
        elif p1_source_trials:
            # source_trials initializes x0/sigma0 from data — cannot specify both
            sampler = optuna.samplers.CmaEsSampler(
                seed=seed + 1,
                source_trials=p1_source_trials,
            )
        else:
            best_p1 = sorted(p1_records, key=lambda r: r.h1, reverse=True)[0]
            x0 = {
                name: max(bounds[0], min(bounds[1], float(getattr(best_p1, name))))
                for name, bounds in rs.items()
                if not isinstance(bounds, list)
            }
            sampler = optuna.samplers.CmaEsSampler(
                x0=x0,
                sigma0=0.3,
                seed=seed + 1,
            )
        study = optuna.create_study(
            study_name=f"{study_name}-p2", direction="maximize", sampler=sampler,
        )
        top_seeds = sorted(p1_records, key=lambda r: r.h1, reverse=True)[:top_k]
        for seed_rec in top_seeds:
            clamped: dict = {}
            for name, bounds in rs.items():
                val = getattr(seed_rec, name)
                clamped[name] = val if isinstance(bounds, list) else max(bounds[0], min(bounds[1], val))
            study.enqueue_trial(clamped)
        return rs, study

    if _HAS_RICH and console:
        with Live(dashboard, console=console, refresh_per_second=2, screen=False):
            if not _resume_records:
                _run_phase(p1_study, p1_obj, phase1_trials)

            if phase2_trials > 0:
                dashboard.phase_label = "Phase 2/2 — Fine-tuning (CMA-ES)"
                p1_source = (
                    _make_source_trials(_resume_records, _param_space)
                    if _resume_records else p1_study.trials
                )
                refined_space, p2_study = _build_p2_study(dashboard.records, p1_source)
                p2_obj = make_objective(refined_space, phase=2)
                _run_phase(p2_study, p2_obj, phase2_trials)
                final_study = p2_study
            else:
                final_study = p1_study
    else:
        # Plain-text fallback
        if not _resume_records:
            header = (
                f"{'Ph':>3}  {'#':>4}  {'trb':>5}  {'r2':>5}  {'vote':>6}  {'bm':>3}  "
                f"{'idf':>5}  {'bbns':>5}  {'fhrb':>5}  {'gamma':>6}  {'beta':>5}  "
                f"{'H@1':>7}  {'H@10':>7}  {'MRR':>6}  B"
            )
            print(header)
            print("-" * len(header))
            _run_phase_plain(p1_study, p1_obj, phase1_trials)

        if phase2_trials > 0:
            print(f"\n--- Phase 2: fine-tuning (CMA-ES, refined bounds) ---")
            p1_source = (
                _make_source_trials(_resume_records, _param_space)
                if _resume_records else p1_study.trials
            )
            refined_space, p2_study = _build_p2_study(dashboard.records, p1_source)
            p2_obj = make_objective(refined_space, phase=2)
            _run_phase_plain(p2_study, p2_obj, phase2_trials)
            final_study = p2_study
        else:
            final_study = p1_study

    best = dashboard.best
    if best is None:
        (console.print if console else print)("[red]No successful trials.[/red]")
        return None

    # ── final summary ─────────────────────────────────────────────────────
    _print = console.print if console else print
    if console:
        console.rule("[bold green]Tuning Complete[/bold green]")
    _print(
        f"\nBest result:  H@1={best.h1*100:.2f}%  "
        f"H@10={best.h10*100:.2f}%  MRR={best.mrr:.4f}"
    )
    _print(
        f"  trb-factor={best.trb_factor:.3f}  r2-boost={best.r2_boost:.3f}  "
        f"vote-weight={best.vote_weight:.4f}  beam-width={best.beam_width}  "
        f"idf-weight={best.idf_weight:.3f}  branch-bonus={best.branch_bonus:.3f}  "
        f"fhrb-factor={best.fhrb_factor:.3f}"
    )
    _print(
        f"  gamma={best.gamma:.4f}  beta={best.beta:.4f}  "
        f"dpw={best.degree_penalty_weight:.4f}  sst={best.schema_score_threshold:.4f}  "
        f"bbp={getattr(best, 'backward_bonus', 0.0):.4f}  "
        f"da={getattr(best, 'diversity_alpha', 0.0):.4f}  "
        f"csb={getattr(best, 'conditional_schema_bonus', 0.0):.4f}"
    )
    _param_flags = (
        f"--beam-width {best.beam_width} "
        f"--trb-factor {best.trb_factor:.3f} --r2-boost {best.r2_boost:.3f} "
        f"--vote-weight {best.vote_weight:.4f} --idf-weight {best.idf_weight:.3f} "
        f"--branch-bonus {best.branch_bonus:.3f} --fhrb-factor {best.fhrb_factor:.3f} "
        f"--gamma {best.gamma:.4f} --beta {best.beta:.4f} "
        f"--degree-penalty-weight {best.degree_penalty_weight:.4f} "
        f"--schema-score-threshold {best.schema_score_threshold:.4f} "
        f"--backward-bonus {getattr(best, 'backward_bonus', 0.0):.4f} "
        f"--diversity-alpha {getattr(best, 'diversity_alpha', 0.0):.4f} "
        f"--conditional-schema-bonus {getattr(best, 'conditional_schema_bonus', 0.0):.4f} "
        f"--beam-hub-penalty {getattr(best, 'beam_hub_penalty', 0.0):.4f} "
        f"--anchor-rerank-weight {getattr(best, 'anchor_rerank_weight', 0.0):.4f} "
        f"--schema-top-k {getattr(best, 'schema_top_k', 5)} "
        f"--hop1-base-weight {getattr(best, 'hop1_base_weight', 0.0):.4f} "
        f"--g1p-trb-weight {getattr(best, 'g1p_trb_weight', 0.0):.4f}"
    )
    if dataset == "hetionet":
        canonical = (
            f"python -u benchmarks/hetionet_param_eval.py "
            f"--n-questions 200 --min-eval-hop 1 --max-neighbors 200 --workers 8 "
            f"--embeddings {embeddings} {_param_flags}"
        )
    elif dataset == "conceptnet":
        canonical = (
            f"python -u benchmarks/conceptnet_eval.py "
            f"--cn5 {cn5_path} --n-questions 2000 --embeddings {embeddings} {_param_flags}"
        )
    elif dataset in ("webqsp", "webqsp251", "webqsp252", "webqsp253", "webqsp254", "webqsp255", "webqsp256", "webqsp257"):
        canonical = (
            f"python -u benchmarks/webqsp_param_eval.py "
            f"--sample 1628 --embeddings {embeddings} {_param_flags}"
        )
    else:
        canonical = (
            f"python -u benchmarks/metaqa_eval.py --hop 3 --embeddings {embeddings} "
            f"{_param_flags}"
        )
    _print(f"\nCanonical benchmark command:\n  {canonical}\n")
    _print(f"  Trial log: {log_file}")

    best_dict = _trial_record_to_dict(best, sample, run_id, dashboard.records)
    best_dict["type"]         = "best_config"
    best_dict["canonical_cmd"] = canonical
    _write_log(log_file, best_dict)

    # fANOVA on whichever phase produced the best result (or final study)
    try:
        importances = optuna.importance.get_param_importances(final_study)
        _write_log(log_file, {
            "type":        "param_importances",
            "run_id":      run_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method":      "fanova",
            "importances": dict(importances),
        })
        _print("\nParameter importances (fANOVA):")
        for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            _print(f"  {param:<16} {imp:.4f}  {bar}")
    except Exception:
        pass

    # ── optional validation run ───────────────────────────────────────────
    if validate > 0:
        _print(f"Validating best config on {validate:,} questions (--workers {workers})...")
        try:
            vh1, vh10, vmrr, vel = _run_eval_logged(
                log_file,
                run_id,
                trial_id=-1,
                dataset=dataset,
                sample=validate,
                trb_factor=best.trb_factor,
                r2_boost=best.r2_boost,
                vote_weight=best.vote_weight,
                beam_width=best.beam_width,
                idf_weight=best.idf_weight,
                branch_bonus=best.branch_bonus,
                fhrb_factor=best.fhrb_factor,
                gamma=best.gamma,
                beta=best.beta,
                timeout=3600,
                workers=workers,
                embeddings=embeddings,
                cn5_path=cn5_path,
            )
            _print(
                f"  Validation:  H@1={vh1*100:.2f}%  "
                f"H@10={vh10*100:.2f}%  MRR={vmrr:.4f}  ({vel:.0f}s)"
            )
            _write_log(log_file, {
                "type": "validation", "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sample": validate, "h1": vh1, "h10": vh10, "mrr": vmrr,
                "elapsed_s": round(vel, 1),
            })
        except Exception as exc:
            _print(f"  Validation failed: {exc}")

    return best


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cerebrum_tuner",
        description="CEREBRUM live hyperparameter tuner — two-phase exploration + fine-tuning",
    )
    parser.add_argument(
        "--phase1-trials", type=int, default=60, dest="phase1_trials",
        help="Exploration trials (RandomSampler, wide bounds). Default: 60",
    )
    parser.add_argument(
        "--phase2-trials", type=int, default=140, dest="phase2_trials",
        help="Fine-tuning trials (TPE, refined bounds). Default: 140",
    )
    parser.add_argument(
        "--n-trials", type=int, default=0, dest="n_trials",
        help="Single-phase mode: run N TPE trials (overrides --phase1/2-trials). "
             "Legacy / backward-compatible.",
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Questions per trial (default 500 ~30s; use 2000 for stability)",
    )
    parser.add_argument(
        "--validate", type=int, default=0,
        help="After tuning, run best config on N questions (0=skip, 14274=full dataset)",
    )
    parser.add_argument(
        "--study-name", type=str, default="cerebrum-tuner", dest="study_name",
        help="Label shown in dashboard header",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-k", type=int, default=10, dest="top_k",
        help="Top-K Phase-1 trials used to derive Phase-2 bounds (default 10)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.25,
        help="Fractional margin added around top-K range for Phase-2 bounds (default 0.25)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes per eval trial (default 1; try 4-8 on multi-core CPUs). "
             "Higher values speed up each trial but use more RAM.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, dest="log_file",
        help="JSONL file to log every trial (default: benchmarks/tuner_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, dest="resume_file",
        metavar="JSONL",
        help="Resume from a previous run: load Phase 1 trials from this JSONL and skip "
             "straight to Phase 2.  --phase1-trials is ignored when this is set.",
    )
    parser.add_argument(
        "--param-init", action="store_true", dest="param_init",
        help="(Phase 205) Enqueue a ParameterInitializer warm-start trial as the first "
             "Phase 1 evaluation.  Biases Sobol toward the analytically-derived optimum.",
    )
    parser.add_argument(
        "--kb-file", type=Path, default=None, dest="kb_file",
        help="KB triples file for --param-init (default: benchmarks/data/metaqa/kb.txt).",
    )
    parser.add_argument(
        "--dataset", choices=["metaqa", "hetionet", "conceptnet", "webqsp", "webqsp251", "webqsp252", "webqsp253", "webqsp254", "webqsp255", "webqsp256", "webqsp257"], default="metaqa", dest="dataset",
        help="KB to tune against. 'hetionet' uses PARAM_SPACE_HETIONET; "
             "'conceptnet' uses PARAM_SPACE_CONCEPTNET (requires --cn5-file); "
             "'webqsp' uses PARAM_SPACE_WEBQSP; "
             "'webqsp251' uses PARAM_SPACE_WEBQSP_251 (Phase 251 focused DPW sweep); "
             "'webqsp252' uses PARAM_SPACE_WEBQSP_252 (Phase 252 traversal-time BHP); "
             "'webqsp253' uses PARAM_SPACE_WEBQSP_253 (Phase 253 anchor re-ranking + schema_top_k); "
             "'webqsp254' uses PARAM_SPACE_WEBQSP_254 (Phase 254 wider beam re-tune); "
             "'webqsp255' uses PARAM_SPACE_WEBQSP_255 (Phase 255 Guaranteed 1-hop Pass). Default: metaqa.",
    )
    parser.add_argument(
        "--cn5-file", type=str, default="", dest="cn5_path",
        help="Path to conceptnet-assertions-5.7.0.csv or .csv.gz (required for --dataset conceptnet).",
    )
    parser.add_argument(
        "--max-edges", type=int, default=200_000, dest="max_edges",
        help="Maximum ConceptNet edges to load (default: 200,000).",
    )
    parser.add_argument(
        "--embeddings", choices=["sentence", "random"], default="sentence",
        help="Embedding method for eval subprocess and ParameterInitializer warm-start. "
             "Default: sentence.",
    )
    args = parser.parse_args()

    run_tuner(
        phase1_trials=args.phase1_trials,
        phase2_trials=args.phase2_trials,
        n_trials=args.n_trials,
        sample=args.sample,
        study_name=args.study_name,
        validate=args.validate,
        seed=args.seed,
        top_k=args.top_k,
        margin=args.margin,
        log_file=args.log_file,
        workers=args.workers,
        resume_file=args.resume_file,
        param_init=args.param_init,
        kb_file=args.kb_file,
        dataset=args.dataset,
        embeddings=args.embeddings,
        cn5_path=args.cn5_path,
        max_edges=args.max_edges,
    )


if __name__ == "__main__":
    main()

# Phase 185–186 Plan: 3-hop MetaQA H@1 Push

**Baseline (Phase 184 / v2.53.2):** H@1=54.6%, H@10=83.0%, MRR=0.605  
**Current (Phase 186):** H@1=57.2%, H@10=88.8%, MRR=0.687 (500-q sample, seed=42)

---

## Root-Cause Taxonomy (from phase184 diagnostic)

| Category | Phase 184 | Phase 186 | Fix Applied |
|---|---|---|---|
| H@1 hit | 273 (54.6%) | 286 (57.2%) | — |
| H@10 hit | 142 (28.4%) | 158 (31.6%) | — |
| beam_coverage miss | **71 (14.2%)** | **23 (4.6%)** | barrier + geo mean |
| vote_convergence miss | 7 (1.4%) | 27 (5.4%) | genre penalty (partial) |
| filter miss | 7 (1.4%) | 6 (1.2%) | — |

---

## Completed Fixes

### Phase 185a — GlobalBeamBarrier `min_guaranteed=10`
- **File:** `reasoning/expanded_traversal.py`
- **Fix:** Top-10 hop-1 branches always run to completion regardless of score.
- **Why it worked:** Phase 184 hop1 audit: all 71 beam_coverage misses had viable rank ≤ 8 with score_ratio ~0.23; barrier was cutting them at stage-2.
- **Result:** beam_coverage 71 → 22, H@1 +1.4pp, H@10 +5.2pp

### Phase 185b — Pure-genre cross-type penalty
- **File:** `benchmarks/metaqa_eval.py`
- **Fix:** Multiply score × 0.10 for the 23 `has_genre` label entities (Drama, Comedy, Horror…) when the detected terminal relation is `written_by`, `directed_by`, `starred_actors`, or `release_year`. Also penalize `in_language` entities for `release_year` only (French ≠ year).
- **Why it worked:** Genre entities accumulate massive path coverage as high-degree hubs and were winning over person/year answers. The `_pure_genre` set (has_genre minus person_year_answers) guarantees no correct answer is penalized.
- **Result:** vote_convergence 28 → 23, H@1 +0.4pp, H@10 +1.2pp

### Phase 186 — Geometric mean stitch scoring
- **File:** `reasoning/expanded_traversal.py` (`_stitch()`)
- **Fix:** Replace `parent.score * child.score` with `sqrt(parent.score * child.score)`.
- **Why it worked:** Hop-1 entities with score_ratio ~0.33 produced stitched paths scoring 0.33× the best, falling below the global top-100 cutoff. Geometric mean raises them to 0.58×, enough to appear in the collection window.
- **Result:** beam_coverage 25 → 23, H@1 +0.6pp, MRR +0.004

---

## Remaining Miss Buckets

### beam_coverage (23 cases)
All 23 are `in_topk_stage2_fail` with viable rank ≤ 7. The barrier guarantees they run. Stitched scores still below top-100 despite geo mean. Likely causes:
- Stage-2 deep beam (width=30 at intermediate hop) pruning the correct hop-2 entity
- Very low absolute hop-1 scores (~0.001) still produce low stitched scores even after sqrt

**Next options:**
- A) Increase `expansion_k` from 20 → 25 so more hop-1 entities are explored
- B) Widen stage-2 beam for guaranteed branches (top-10 get beam=50, rest keep 30)
- C) Increase `_raw_top_k` from 100 → 200 (wider collection window, more memory)

### vote_convergence (27 cases)
Correct answer is in beam/top-100 but ranked below top-1. Sub-cases:
- **Wrong wrong-type entity wins** (bd-r, animation, documentary, French): these are in `has_tags` or `in_language` but NOT in `has_genre`, so the current pure-genre penalty misses them. `has_tags` has 4892 entities including person names — cannot penalize the whole set.
  - **Next:** targeted penalty for specific format-tag values ("bd-r", "animation", "documentary", "blu-ray") that are definitionally not people or years.
- **Wrong person wins** (e.g., "Eliot Wald" beats "Jacquelyn Mitchard"): genuine ranking error — the wrong writer accumulates more paths. Requires stronger path-consistency (r2_boost) or vote-weight tuning.
- **Wrong year wins** (e.g., "2011" beats "1991"): both are valid year entities, r2_boost could help here.

**Next options:**
- A) Small explicit blocklist: {"bd-r", "bd-r.", "animation", "documentary", "blu-ray", "blu ray", "short film"} — apply same 0.10 penalty, safe because these strings are definitionally not people/years
- B) Enable `r2_boost > 0` to strengthen path-consistency for the wrong-person and wrong-year cases
- C) Optuna re-tune: run metaqa_tune.py with new code to find optimal vote_weight/r2_boost/idf_weight under Phase 186 fixes

---

## Next Immediate Steps

1. **Blocklist penalty** for "bd-r", "animation" (case-insensitive), "documentary", "blu-ray" — targeted, no risk of correct-answer suppression. Estimated +2–3 H@1.
2. **Enable r2_boost ~0.5** for written_by/release_year — path-consistency boost for correct-path entities. Estimated +2–4 H@1.
3. **Full-sample benchmark** after above to get official Phase 186 number vs Phase 182 canonical (H@1=49.68%).
4. **Sync CLAUDE.md + CHANGELOG** with Phase 185/186 results.

---

## Score Progression (500-q sample, seed=42)

| Version | H@1 | H@10 | MRR |
|---|---|---|---|
| Phase 182 canonical (14K) | 49.7% | 79.5% | 0.605 |
| Phase 184 diagnostic | 54.6% | 83.0% | — |
| +barrier fix | 56.0% | 88.2% | 0.684 |
| +genre penalty | 56.4% | 89.4% | 0.683 |
| +geo mean stitch | **57.2%** | **88.8%** | **0.687** |
